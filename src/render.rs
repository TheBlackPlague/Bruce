use std::{
    fmt::Write,
    iter,
    time::{Duration, Instant},
};

use anstyle::{AnsiColor, Style};
use bullet_trainer::run::TrainingSteps;

use crate::{
    control::{Outcome, Stop},
    events::{Ending, Rows, Stage, Task},
    telemetry::Live,
    text,
};

pub const WIDTH: usize = 78;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const GUTTER: usize = 4;
const PERCENT: usize = 7;
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const PLAIN: Style = Style::new();
const BOLD: Style = Style::new().bold();
const DIM: Style = Style::new().dimmed();
const CYAN: Style = AnsiColor::Cyan.on_default();
const GREEN: Style = AnsiColor::Green.on_default();
const YELLOW: Style = AnsiColor::Yellow.on_default();
const RED: Style = AnsiColor::Red.on_default();

type Row = Vec<(Style, String)>;

struct Frame
{
    status: String,
    bars: Vec<Bar>,
    fields: Vec<Field>,
}

struct Bar
{
    label: String,
    count: String,
    fraction: f64,
}

struct Field
{
    label: String,
    value: String,
    alert: bool,
}

pub fn width(columns: u16) -> usize { usize::from(columns).saturating_sub(2).min(WIDTH) }

pub fn block(live: &Live, stop: Stop, now: Instant, pulse: usize, width: usize, color: bool) -> Vec<String>
{
    let frame = frame(live, stop, now);
    let (style, hint) = match (stop, live.setup.task) {
        (Stop::Running, _) => (CYAN, "ctrl+c  stop safely"),
        (Stop::AtBoundary, Task::Train) => (YELLOW, "ctrl+c again  abort without saving"),
        (Stop::AtBoundary, Task::Analyze) => (YELLOW, "ctrl+c again  abort without a report"),
        (Stop::AtBoundary, Task::Check | Task::Benchmark) => (YELLOW, "ctrl+c again  abort without a summary"),
        (Stop::Committed, _) => (YELLOW, "ctrl+c twice  exit without waiting"),
        (Stop::Armed, _) => (YELLOW, "ctrl+c again  exit immediately"),
        (Stop::Now | Stop::Exit, _) => (RED, "ctrl+c again  exit immediately"),
    };

    let mut rows = header(live);
    rows.extend([Row::new(), vec![(style, format!("{} {}", SPINNER[pulse % SPINNER.len()], frame.status))]]);
    if frame.bars.is_empty() {
        rows.extend(frame.fields.iter().map(|field| cell(field, width)));
    } else {
        for bar in &frame.bars {
            rows.extend([cell(&field(&bar.label, bar.count.clone()), width), meter(bar.fraction, width)]);
        }
        rows.push(Row::new());
        rows.extend(grid(&frame.fields, width));
    }
    rows.extend([Row::new(), vec![(DIM, hint.to_owned())]]);
    paint(rows, width, color)
}

pub fn plain(live: &Live, stop: Stop, now: Instant) -> Option<String>
{
    let Frame { status, bars, fields } = frame(live, stop, now);
    let moving = bars.iter().any(|bar| bar.fraction > 0.0);
    let bars = bars.iter().map(|bar| format!("{} {} ({:.1}%)", bar.label, bar.count, bar.fraction * 100.0));
    let fields = fields.iter().map(|field| format!("{} {}", field.label, field.value));
    moving.then(|| iter::once(status).chain(bars).chain(fields).collect::<Vec<_>>().join("  "))
}

pub fn ended(
    live: &Live,
    ending: &Ending,
    failure: Option<&(String, Option<String>)>,
    warnings: &[String],
    now: Instant,
    width: usize,
    color: bool,
) -> Vec<String>
{
    let (task, outcome) = (live.setup.task, ending.outcome);
    let (word, style) = match (outcome, task, live.warnings) {
        (Outcome::Finished, Task::Check, 0) => ("passed".to_owned(), GREEN),
        (Outcome::Finished, Task::Check, 1) => ("passed with 1 warning".to_owned(), YELLOW),
        (Outcome::Finished, Task::Check, warnings) => (format!("passed with {warnings} warnings"), YELLOW),
        (Outcome::Finished, ..) => ("complete".to_owned(), GREEN),
        (Outcome::Stopped, ..) => ("stopped".to_owned(), YELLOW),
        (Outcome::Aborted, ..) => ("aborted".to_owned(), RED),
        (Outcome::Failed, ..) => ("failed".to_owned(), RED),
    };

    let mut rows = header(live);
    rows.extend([Row::new(), vec![(style.bold(), format!("{} {word}", task.name()))]]);
    if let Some((error, help)) = failure {
        rows.extend(wrap(&format!("Error: {error}"), RED.bold(), width));
        rows.extend(help.iter().flat_map(|help| wrap(&format!("Help: {help}"), PLAIN, width)));
    }

    let (trained, saved) = trained(live, outcome, now);
    let facts: Vec<Field> = fields(&ending.rows).chain(trained).collect();
    rows.extend((!ending.grid.is_empty() || !facts.is_empty()).then(Row::new));
    rows.extend(grid(&fields(&ending.grid).collect::<Vec<_>>(), width));
    rows.extend(facts.iter().map(|fact| cell(fact, width)));

    let log = live.setup.log.as_ref().map(|log| field("Log", log.display().to_string()));
    let see = if log.is_some() { ", see log" } else { "" };
    let warned = (live.warnings > 0).then(|| field("Warnings", format!("{}{see}", live.warnings)));
    let metrics = live.setup.metrics.as_ref().filter(|_| live.run.is_some());
    let metrics = metrics.map(|metrics| field("Metrics", metrics.display().to_string()));
    let files: Vec<Field> = fields(&ending.files).chain(warned).chain(log).chain(metrics).chain(saved).collect();
    rows.extend((!warnings.is_empty() || !files.is_empty()).then(Row::new));
    rows.extend(warnings.iter().flat_map(|warning| wrap(&format!("Warning: {warning}"), YELLOW.bold(), width)));
    rows.extend(files.into_iter().map(|file| vec![(DIM, file.label), (PLAIN, format!("  {}", file.value))]));
    paint(rows, width, color)
}

fn frame(live: &Live, stop: Stop, now: Instant) -> Frame
{
    let frame = match live.setup.task {
        Task::Train => training(live, now),
        _ => scanning(live, now),
    };
    let mut frame = frame.unwrap_or_else(|| {
        let status = live.status.clone().unwrap_or_else(|| "Preparing".to_owned());
        Frame { status, bars: Vec::new(), fields: vec![field("Elapsed", elapsed(live, now))] }
    });
    if stop != Stop::Running {
        frame.status = halting(live, stop);
    }
    frame
}

fn halting(live: &Live, stop: Stop) -> String
{
    let aborting = matches!(stop, Stop::Now | Stop::Exit);
    let superbatch = live.stop_committed.unwrap_or(live.superbatch);
    let measuring = live.scan.as_ref().and_then(|(info, _)| info.inputs.get(live.input).or(info.inputs.last()));
    match live.setup.task {
        Task::Train if aborting => "Aborting: no checkpoint will be saved".to_owned(),
        Task::Train => format!("Stopping after superbatch {superbatch}, then saving a checkpoint"),
        Task::Analyze if aborting => "Aborting: no report will be written".to_owned(),
        Task::Analyze => "Stopping: finishing the blocks in flight, then writing the report".to_owned(),
        Task::Check | Task::Benchmark if aborting => "Aborting".to_owned(),
        Task::Check => "Stopping after the blocks in flight".to_owned(),
        Task::Benchmark => match measuring {
            Some(input) => format!("Stopping · ending the {} measurement", text::file_name(&input.path)),
            None => "Stopping".to_owned(),
        },
    }
}

fn training(live: &Live, now: Instant) -> Option<Frame>
{
    let run = live.run.as_ref().filter(|_| live.ready)?;
    let TrainingSteps { batches_per_superbatch, end_superbatch: final_superbatch, .. } = run.steps;
    let phases = run.phases.len();
    let phase = run.phases.iter().position(|&(first, last)| (first..=last).contains(&live.superbatch));
    let status = match (live.stage, phase) {
        (Some(Stage::SavingCheckpoint), _) => "Saving checkpoint".to_owned(),
        (Some(Stage::Finalizing), _) => "Finalizing".to_owned(),
        (_, Some(phase)) if phases > 1 => format!("Training · phase {} of {phases}", phase + 1),
        _ => "Training".to_owned(),
    };

    let (superbatch, batch) = match live.stop_committed {
        Some(last) if live.superbatch > last => (last, batches_per_superbatch),
        _ => (live.superbatch, live.batch),
    };
    let (batch, per_superbatch) = (batch as u64, batches_per_superbatch as u64);
    let stalls = match live.stalls {
        0 => String::new(),
        1 => " · 1 stall".to_owned(),
        stalls => format!(" · {stalls} stalls"),
    };
    let idle = live.loader_idle(now);
    let starved = live.stalled_for(now).is_some() || idle.is_some_and(|idle| idle >= 0.05);
    let loader = idle.map_or_else(unknown, |idle| format!("idle {:.0}%", idle * 100.0)) + &stalls;
    let (superbatch_eta, run_eta) = live.eta(now);

    let overall = format!("Superbatch {superbatch}/{final_superbatch}");
    let trained = superbatch.saturating_sub(1) as u64 * per_superbatch + batch;
    let batches = format!("{}/{}", text::count(batch), text::count(per_superbatch));
    Some(Frame {
        status,
        bars: vec![
            bar("Overall training", overall, trained, final_superbatch as u64 * per_superbatch),
            bar("Batches", batches, batch, per_superbatch),
        ],
        fields: vec![
            field("Loss", live.loss().map_or_else(unknown, |loss| format!("{loss:.6}"))),
            field("Throughput", live.speed().map_or_else(unknown, text::rate)),
            field("Learning rate", format!("{:.3e}", live.learning_rate)),
            field("Elapsed", elapsed(live, now)),
            field("Superbatch ETA", superbatch_eta.map_or_else(unknown, text::span)),
            field("Run ETA", run_eta.map_or_else(unknown, text::span)),
            field("WDL", format!("{:.3}", live.wdl)),
            Field { alert: starved, ..field("Loader", loader) },
        ],
    })
}

fn scanning(live: &Live, now: Instant) -> Option<Frame>
{
    let (info, started) = live.scan.as_ref().filter(|(info, _)| !info.inputs.is_empty())?;
    let (inputs, current) = (info.inputs.len(), live.input.min(info.inputs.len() - 1));
    let amount = |index: usize| {
        let total = info.head.unwrap_or(info.inputs[index].total);
        let done = live.progress.get(index).map_or(0, |read| if info.head.is_some() { read.games } else { read.done });
        (if index < current { total } else { done.min(total) }, total)
    };
    let (own, whole) = amount(current);
    let (done, total) = (0..inputs).map(amount).fold((0, 0), |(done, total), (own, whole)| (done + own, total + whole));
    let since = now.saturating_duration_since(*started);
    let timed = done > 0 && since >= Duration::from_secs(1);
    let eta = |left: u64| timed.then(|| since.mul_f64(left as f64 / done as f64)).map_or_else(unknown, text::span);

    let (name, number) = (text::file_name(&info.inputs[current].path), current + 1);
    let games: u64 = live.progress.iter().map(|input| input.games).sum();
    let positions: u64 = live.progress.iter().map(|input| input.positions).sum();
    let detail = |default: &str| live.status.clone().unwrap_or_else(|| default.to_owned());
    let preview = live.preview.iter().map(|(label, value)| field(label, value.clone()));
    let speed = field("Throughput", live.scan_rate(now).map_or_else(unknown, text::rate));
    let clock = field("Elapsed", elapsed(live, now));
    let (status, bars, fields): (_, _, Vec<Field>) = match live.setup.task {
        Task::Check => {
            let count = |done, total| match info.head {
                Some(_) => format!("{}/{} games", text::count(done), text::count(total)),
                None => text::bytes_of(done, total),
            };
            let fields = preview.chain([clock, speed, field("ETA", eta(total - done))]).collect();
            let bars =
                vec![bar("All sources", count(done, total), done, total), bar(&name, count(own, whole), own, whole)];
            (format!("Scanning · source {number} of {inputs}"), bars, fields)
        },
        Task::Benchmark => {
            let throughput = live.scan_rate(now).map_or_else(unknown, |rate| format!("{:.2} Mpos/s", rate / 1e6));
            let measured = text::scaled(live.progress.get(current).map_or(0, |input| input.positions) as f64);
            let remaining = field("Remaining", text::span(Duration::from_millis(total - done)));
            let fields = vec![field("Throughput", throughput), clock, field("Positions", measured), remaining];
            let overall = bar("Overall benchmark", format!("Measurement {number}/{inputs}"), done, total);
            let seconds = bar("Seconds", format!("{}/{}", own / 1000, whole / 1000), own, whole);
            (format!("Measuring · {name}"), vec![overall, seconds], fields)
        },
        Task::Analyze => {
            let read = match info.head {
                Some(_) => format!("Games {}/{}", text::count(own), text::count(whole)),
                None => text::bytes_of(own, whole),
            };
            let counts = [field("Games", text::count(games)), speed, field("Positions", text::count(positions))];
            let etas = [clock, field("Input ETA", eta(whole - own)), field("Run ETA", eta(total - done))];
            let overall = bar("Overall analysis", format!("Input {number}/{inputs}"), done, total);
            let bars = vec![overall, bar(&name, read, own, whole)];
            (detail("Scanning"), bars, counts.into_iter().chain(preview).chain(etas).collect())
        },
        Task::Train => return None,
    };

    Some(Frame { status, bars, fields })
}

fn trained(live: &Live, outcome: Outcome, now: Instant) -> (Vec<Field>, Vec<Field>)
{
    let Some(run) = &live.run else { return Default::default() };
    let TrainingSteps { batches_per_superbatch, start_superbatch: first, end_superbatch: last, .. } = run.steps;
    let resumed = run.resumed.as_ref();
    let completed = live.last.as_ref().map_or(first.saturating_sub(1), |summary| summary.superbatch);
    let superbatches = match resumed {
        Some(_) if completed < first => format!("none of {last}"),
        Some(_) => format!("{first}–{completed} of {last}"),
        None => format!("{completed}/{last}"),
    };
    let label = if resumed.is_none() && outcome != Outcome::Finished { "Superbatch" } else { "Superbatches" };
    let batches = live.superbatch.saturating_sub(first) * batches_per_superbatch + live.batch;
    let stalls = match live.stalls {
        0 => "none".to_owned(),
        stalls => format!("{stalls} ({:.1}s)", live.stall_time.as_secs_f64()),
    };

    let mut rows = vec![field(label, superbatches), field("Batches", text::count(batches as u64))];
    let loss = live.last.as_ref().map(|summary| format!("{:.6} at wdl {:.3}", summary.loss, summary.wdl));
    rows.extend(loss.map(|loss| field("Final loss", loss)));
    rows.extend(live.session_speed(now).map(|speed| field("Throughput", format!("{} average", text::rate(speed)))));
    rows.extend([field("Loader stalls", stalls), field("Elapsed", elapsed(live, now))]);

    let latest = live.checkpoint.as_ref().or(resumed.map(|resumed| &resumed.checkpoint)).map(|path| path.display());
    let mut files: Vec<Field> = latest.iter().map(|path| field("Latest checkpoint", path.to_string())).collect();
    let resume = latest.filter(|_| outcome != Outcome::Finished);
    files.extend(resume.map(|path| field("Resume", format!("bruce train --resume {path}"))));
    if let Some(resumed) = resumed.filter(|resumed| resumed.run != run.name || !resumed.extended) {
        let label = if resumed.run == run.name { "Resumed from" } else { "Forked from" };
        files.push(field(label, resumed.checkpoint.display().to_string()));
    }
    (rows, files)
}

fn header(live: &Live) -> Vec<Row>
{
    let title = live.setup.task.name();
    let (subject, facts) = match &live.run {
        Some(run) => {
            let device = run.device.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ");
            let size = text::count(run.steps.batch_size as u64);
            let batches = text::count(run.steps.batches_per_superbatch as u64);
            let mut facts = format!("{} · {device} · {size} × {batches} per superbatch", run.architecture);
            if let Some(resumed) = &run.resumed {
                let how = match (resumed.run != run.name, resumed.extended) {
                    (true, _) => "forked",
                    (false, true) => "extended",
                    (false, false) => "resumed",
                };
                let _ = write!(facts, " · {how} after {}", resumed.completed);
            }
            (&run.name, facts)
        },
        None => (&live.setup.subject, live.setup.facts.clone()),
    };
    let identity = if subject.is_empty() { title.to_owned() } else { format!("{title} · {subject}") };

    let mut rows = vec![vec![(BOLD, format!("Bruce {VERSION}"))], vec![(PLAIN, identity)]];
    rows.extend((!facts.is_empty()).then(|| vec![(DIM, facts)]));
    rows
}

fn elapsed(live: &Live, now: Instant) -> String { text::span(now.saturating_duration_since(live.started)) }

fn unknown() -> String { "—".to_owned() }

fn field(label: &str, value: String) -> Field { Field { label: label.to_owned(), value, alert: false } }

fn fields(rows: &Rows) -> impl Iterator<Item = Field> { rows.iter().map(|(label, value)| field(label, value.clone())) }

fn bar(label: &str, count: String, done: u64, total: u64) -> Bar
{
    let fraction = if total == 0 { 0.0 } else { (done as f64 / total as f64).min(1.0) };
    Bar { label: label.to_owned(), count, fraction }
}

fn cell(field: &Field, width: usize) -> Row
{
    let used = field.label.chars().count() + field.value.chars().count();
    let value = if field.alert { YELLOW.bold() } else { BOLD };
    let gap = " ".repeat(width.saturating_sub(used).max(2));
    vec![(DIM, field.label.clone()), (PLAIN, gap), (value, field.value.clone())]
}

fn grid(fields: &[Field], width: usize) -> Vec<Row>
{
    let (column, gutter) = (width.saturating_sub(GUTTER) / 2, [(PLAIN, " ".repeat(GUTTER))]);
    let row = |pair: &[Field]| pair.iter().map(|field| cell(field, column)).collect::<Vec<_>>().join(&gutter[..]);
    fields.chunks(2).map(row).collect()
}

fn meter(fraction: f64, width: usize) -> Row
{
    let length = width.saturating_sub(PERCENT);
    let filled = (fraction * length as f64).round() as usize;
    let percent = format!("{:>PERCENT$}", format!("{:.1}%", fraction * 100.0));
    vec![(CYAN, "━".repeat(filled)), (DIM, "─".repeat(length - filled)), (DIM, percent)]
}

fn wrap(text: &str, style: Style, width: usize) -> Vec<Row>
{
    let mut lines: Vec<String> = Vec::new();
    for word in text.split(' ') {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => line.extend([" ", word]),
            Some(_) => lines.push(format!("  {word}")),
            None => lines.push(word.to_owned()),
        }
    }
    lines.into_iter().map(|line| vec![(style, line)]).collect()
}

fn paint(rows: Vec<Row>, width: usize, color: bool) -> Vec<String>
{
    rows.into_iter()
        .map(|row| {
            let (mut left, mut line) = (width, String::new());
            for (style, text) in row {
                let text: String = text.chars().take(left).collect();
                left -= text.chars().count();
                let style = if color { style } else { PLAIN };
                let _ = write!(line, "{style}{text}{style:#}");
            }
            line
        })
        .collect()
}
