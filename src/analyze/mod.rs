pub mod check;
mod report;
mod stats;

use std::{
    fs,
    io::{self, Write},
    ops::ControlFlow,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow, bail};
use bullet_lib::game::formats::bulletformat::ChessBoard;

use self::{
    report::{Combined, InputSummary, NetworkReport, ReliabilityBin, Report, Results, SCHEMA},
    stats::{BUCKETS, SCALE_CP, Stats, Tables},
};
use crate::{
    config::Config,
    control::{Control, Outcome, Stop},
    data::{
        Source,
        pipeline::{self, BLOCK, Block, Sampling, ScanOptions},
        select::Selection,
    },
    events::{Ending, Event, Level, Progress, Rows, ScanInfo, Task},
    model::{Architecture, Evaluator, Losses, sigmoid},
    telemetry::Hub,
    text,
};

pub const REPORTS: &str = "reports";
const PREVIEW: Duration = Duration::from_millis(500);
const NETWORK_BLOCKS: usize = 16;

pub struct Options
{
    pub inputs: Vec<Source>,
    pub sampling: Sampling,
    pub threads: usize,
    pub output: Option<PathBuf>,
    pub created: String,
    pub network: Option<NetworkOptions>,
    pub config: Option<(PathBuf, Config)>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkOptions
{
    pub checkpoint: PathBuf,
    pub architecture: Option<Architecture>,
    pub gpu: u32,
    pub batch_size: usize,
}

impl NetworkOptions
{
    pub const BATCH_SIZE: usize = 16_384;
}

pub fn run(options: &Options, control: &Control, hub: &Hub) -> Result<Ending>
{
    let began = Instant::now();
    if let Some(path) = &options.output {
        preflight(path)?;
    }

    let selections = (options.config.as_ref().map(|(_, config)| {
        let data = &config.training.data;
        let selection = Selection { filter: data.filter.clone(), labels: data.labels, ..Selection::default() };
        options.inputs.iter().map(|input| selection.clone().identified(input)).collect::<Result<Vec<_>, _>>()
    }))
    .transpose()?;
    let rules = options.config.as_ref().and_then(|(_, config)| config.training.data.filter.clone());
    let network = options.network.as_ref();
    let tables = Tables::new();

    for warning in &options.warnings {
        hub.message(Level::Warning, warning.as_str());
    }
    if let Some((path, _)) = &options.config {
        hub.message(Level::Info, format!("Filter and label policy from {}", path.display()));
    }

    let mut evaluator = None;
    if let Some(network) = network.filter(|_| control.stop() == Stop::Running) {
        hub.message(Level::Detail, format!("Loading {} onto GPU {}", network.checkpoint.display(), network.gpu));
        if !cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")) {
            bail!("Network analysis needs a GPU build");
        }
        evaluator = Some(Evaluator::open(&network.checkpoint, network.architecture, network.gpu, network.batch_size)?);
    }
    let tagging = evaluator.as_ref().map(|_| Selection { filter: rules.clone(), ..Selection::default() });
    let status = match network.filter(|_| evaluator.is_some()) {
        Some(network) => format!("Scanning and evaluating on GPU {}", network.gpu),
        None => "Scanning".to_owned(),
    };
    let mut evaluation =
        network.map(|network| Network { evaluator, batch_size: network.batch_size, ..Network::default() });

    let inputs: Vec<_> = options.inputs.iter().map(Source::input).collect();
    let mut summaries: Vec<InputSummary> = (inputs.iter())
        .map(|input| InputSummary {
            path: input.path.display().to_string(),
            bytes: input.total,
            complete: input.total == 0,
            ..InputSummary::default()
        })
        .collect();
    let (head, threads) = (options.sampling.head(), options.threads);
    let sampling = match head {
        Some(games) => format!("head:{games} games (the head of a file is not representative)"),
        None => "every game".to_owned(),
    };
    hub.emit(Event::Scan(ScanInfo { task: Task::Analyze, inputs, threads, sampling, head }));
    hub.message(Level::Detail, status);

    let started = Instant::now();
    let (mut total, mut finished, mut since, mut previewed) = (Stats::new(), 0, Instant::now(), None::<Instant>);
    let counts = |input: &InputSummary| (input.games, input.positions);
    let scanning =
        ScanOptions { sampling: options.sampling, threads, in_flight: network.map(|_| NETWORK_BLOCKS), block: BLOCK };
    let scanned = pipeline::scan(
        &options.inputs,
        &scanning,
        control,
        |scratch, block: &Block<'_>| -> Result<_> {
            let (mut stats, mut boards, mut black) = (Stats::new(), Vec::new(), Vec::new());
            for game in block.games {
                stats.observe(game, &tables);
                if let Some(selections) = &selections {
                    selections[game.source].select_each(game, Some(&mut stats.counters), scratch, |_, _| {})?;
                }
                if let Some(selection) = &tagging {
                    selection.select_each(game, None, scratch, |ply, board| {
                        boards.push(board);
                        black.push(!game.white_to_move(ply));
                    })?;
                }
            }
            Ok((stats, boards, black))
        },
        |meta, (stats, boards, black)| {
            if let Some(network) = &mut evaluation {
                network.boards.extend(boards);
                network.black.extend(black);
                network.evaluate(false)?;
                if network.evaluator.is_some() && previewed.is_none_or(|at| at.elapsed() >= PREVIEW) {
                    let evaluated = network.losses.positions;
                    let rate = text::rate(evaluated as f64 / started.elapsed().as_secs_f64().max(f64::EPSILON));
                    let rows = vec![("Evaluation".to_owned(), rate), ("Evaluated".to_owned(), text::count(evaluated))];
                    hub.emit(Event::Preview(rows));
                    previewed = Some(Instant::now());
                }
            }

            hub.finished_inputs(summaries[..meta.source].iter().map(counts), &mut finished, &mut since);
            let input = &mut summaries[meta.source];
            input.bytes_read = meta.end;
            input.games += stats.games;
            input.positions += stats.positions;
            input.complete = input.bytes_read >= input.bytes;
            let progress =
                Progress { input: meta.source, done: input.bytes_read, games: input.games, positions: input.positions };
            total.merge(stats);
            hub.emit(Event::Progress(progress));
            Ok(ControlFlow::Continue(()))
        },
    );
    let stopped = scanned?;

    let aborted = stopped && control.outcome() == Outcome::Aborted;
    if !aborted && let Some(network) = &mut evaluation {
        network.evaluate(true)?;
    }
    if !stopped {
        hub.finished_inputs(summaries.iter().map(counts), &mut finished, &mut since);
    }
    let elapsed = started.elapsed();

    let empty = summaries.iter().all(|input| input.games == 0);
    let halted = stopped || (empty && control.stop() != Stop::Running);
    if empty && !halted {
        bail!("The inputs hold no games; check that the inputs hold Monty data");
    }

    let outcome = if halted { control.outcome() } else { Outcome::Finished };
    if outcome == Outcome::Aborted || empty {
        return Ok(Ending::of(outcome));
    }

    if !halted {
        hub.message(Level::Detail, "Finalizing analysis");
    }
    let bytes_read = summaries.iter().map(|input| input.bytes_read).sum();
    let passes = !stopped && options.sampling == Sampling::All;
    let report = Report {
        schema: SCHEMA,
        bruce: env!("CARGO_PKG_VERSION"),
        created: options.created.clone(),
        complete: !stopped,
        scale_cp: SCALE_CP,
        sampling: options.sampling,
        inputs: summaries,
        combined: Combined {
            games: total.games(bytes_read),
            scores: total.scores(&tables),
            calibration: total.calibration(&tables),
            filter: options.config.as_ref().map(|(path, config)| total.filter(path, config, passes)),
            network: evaluation.and_then(|evaluation| evaluation.report(options)),
        },
    };

    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    let output = match &options.output {
        Some(path) => write(path, &json).map(|()| path.clone())?,
        None => filed(&options.created, &json)?,
    };

    let mut rows = summary(&report, elapsed);
    rows.push(("Elapsed".to_owned(), text::span(began.elapsed())));
    Ok(Ending { rows, files: vec![("Report".to_owned(), output.display().to_string())], ..Ending::of(outcome) })
}

#[derive(Default)]
struct Network
{
    evaluator: Option<Evaluator>,
    batch_size: usize,
    boards: Vec<ChessBoard>,
    black: Vec<bool>,
    losses: Losses,
    bins: [Bin; BUCKETS],
}

#[derive(Clone, Copy, Default)]
struct Bin
{
    positions: u64,
    prediction: f64,
    result: f64,
}

impl Network
{
    fn evaluate(&mut self, whole: bool) -> Result<()>
    {
        let Some(evaluator) = &mut self.evaluator else { return Ok(()) };
        let ready = if whole { self.boards.len() } else { self.boards.len() / self.batch_size * self.batch_size };
        if ready == 0 {
            return Ok(());
        }

        let mut outputs = Vec::new();
        evaluator.evaluate(&self.boards[..ready], &mut outputs)?;
        if outputs.len() != ready || !outputs.iter().all(|output| output.is_finite()) {
            bail!("The network returned an invalid score");
        }

        for ((board, black), output) in self.boards.drain(..ready).zip(self.black.drain(..ready)).zip(outputs) {
            self.losses.add(output, &board);
            let (probability, result) = (sigmoid(f64::from(output)), f64::from(board.result) / 2.0);
            let (prediction, result) = if black { (1.0 - probability, 1.0 - result) } else { (probability, result) };
            let bin = &mut self.bins[((prediction * BUCKETS as f64) as usize).min(BUCKETS - 1)];
            bin.positions += 1;
            bin.prediction += prediction;
            bin.result += result;
        }

        Ok(())
    }

    fn report(self, options: &Options) -> Option<NetworkReport>
    {
        let Self { evaluator, losses, bins, .. } = self;
        let network = options.network.as_ref()?;
        let mean = |sum: f64, positions: u64| (positions > 0).then(|| sum / positions as f64);
        let gaps: f64 = bins.iter().map(|bin| (bin.result - bin.prediction).abs()).sum();

        Some(NetworkReport {
            checkpoint: network.checkpoint.display().to_string(),
            architecture: evaluator.as_ref().map(Evaluator::architecture),
            evaluated: evaluator.is_some(),
            config: options.config.as_ref().map(|(path, _)| path.display().to_string()),
            filtered: options.config.as_ref().is_some_and(|(_, config)| config.training.data.filter.is_some()),
            batch_size: network.batch_size,
            positions: losses.positions,
            score: mean(losses.score, losses.positions),
            result: mean(losses.result, losses.positions),
            ece: mean(gaps, losses.positions),
            calibration: (0_u32..)
                .zip(bins)
                .map(|(index, bin)| ReliabilityBin {
                    from: f64::from(index) / BUCKETS as f64,
                    to: f64::from(index + 1) / BUCKETS as f64,
                    positions: bin.positions,
                    mean_prediction: mean(bin.prediction, bin.positions),
                    mean_result: mean(bin.result, bin.positions),
                })
                .collect(),
        })
    }
}

fn preflight(path: &Path) -> Result<()>
{
    if path.is_dir() || !directory(path).is_dir() {
        bail!(
            "Output {} is a directory or lies in a directory that does not exist; choose an output file in an \
             existing directory",
            path.display()
        );
    }

    if fs::symlink_metadata(path).is_ok() {
        bail!("Output {} already exists; choose another output path", path.display());
    }

    Ok(())
}

fn write(path: &Path, bytes: &[u8]) -> Result<()>
{
    let failed = |error: io::Error| anyhow!("Could not write {}: {error}", path.display());
    let mut file = tempfile::NamedTempFile::new_in(directory(path)).map_err(failed)?;
    file.write_all(bytes).and_then(|()| file.as_file().sync_all()).map_err(failed)?;

    file.persist_noclobber(path).map(drop).map_err(|error| match error.error.kind() {
        io::ErrorKind::AlreadyExists => anyhow!("Output {} already exists", path.display()),
        _ => failed(error.error),
    })
}

fn directory(path: &Path) -> &Path
{
    path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new("."))
}

fn filed(created: &str, report: &[u8]) -> Result<PathBuf>
{
    let stamp = created.replace(['-', ':', 'Z'], "").replace('T', "-");
    let mut path = Path::new(REPORTS).join(format!("analysis-{stamp}.json"));
    let mut copy = 1;
    while let Err(error) = write(&path, report) {
        if fs::symlink_metadata(&path).is_err() {
            return Err(error);
        }

        copy += 1;
        path = Path::new(REPORTS).join(format!("analysis-{stamp}-{copy}.json"));
    }

    Ok(path)
}

fn summary(report: &Report, elapsed: Duration) -> Rows
{
    let Combined { games, scores, calibration, filter, network } = &report.combined;
    let inputs = &report.inputs;
    let mut rows = Vec::new();
    let mut row = |label: &str, value: String| rows.push((label.to_owned(), value));

    if !report.complete {
        let read = inputs.iter().map(|input| input.bytes_read).sum();
        let whole = inputs.iter().filter(|input| input.complete).count();
        let total = inputs.iter().map(|input| input.bytes).sum();
        row("Read", format!("{} · {whole} of {} inputs whole", text::bytes_of(read, total), inputs.len()));
    }

    let mean = games.length.mean.map(|mean| format!(" · mean {mean:.1} plies")).unwrap_or_default();
    row("Games", format!("{}{mean}", text::count(games.games)));
    let positions: u64 = inputs.iter().map(|input| input.positions).sum();
    row("Positions", text::count(positions));
    let Results { white, draw, black } = games.results;
    let [white, draw, black] = [white, draw, black].map(|count| text::percent(count, white + draw + black));
    row("Results", format!("White {white} · draw {draw} · Black {black}"));
    let share = |count: u64| 100.0 * count as f64 / scores.positions.max(1) as f64;
    let forced = scores.forced.minimum + scores.forced.maximum;
    let large = scores.large_evals.at_least_4000;
    row("Scores", format!("{:.2}% at ±32767 · {:.2}% beyond ±4,000 cp", share(forced), share(large)));
    let parts = [
        calibration.brier.map(|brier| format!("Brier {brier:.4}")),
        calibration.expected_calibration_error.map(|error| format!("ECE {error:.4}")),
    ];
    row("Calibration", parts.into_iter().flatten().collect::<Vec<_>>().join(" · "));
    if let Some(filter) = filter {
        let passes = match filter.schedule.passes {
            Some(passes) => format!("{passes:.1} passes in {} superbatches", filter.schedule.superbatches),
            None => "passes need a complete scan".to_owned(),
        };
        row("Filter", format!("{} accepted · {passes}", text::percent(filter.accepted, filter.positions)));
    }
    if let Some(network) = network {
        let result = network.result.map_or_else(|| "not evaluated".to_owned(), |loss| format!("vs result {loss:.4}"));
        row("Network", format!("{} · {result}", text::file_name(Path::new(&network.checkpoint))));
    }
    let seconds = elapsed.as_secs_f64();
    if report.complete && seconds > 0.0 {
        let evaluated = network.as_ref().filter(|network| network.evaluated).map_or("", |_| " · 1 network");
        row("Throughput", format!("{} average{evaluated}", text::rate(positions as f64 / seconds)));
    }

    rows
}
