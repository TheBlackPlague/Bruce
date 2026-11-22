use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    mem,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};

use anstream::adapter::strip_str;
use bullet_trainer::run::TrainingSteps;
use serde_json::json;
use time::{OffsetDateTime, UtcOffset};

use crate::{
    control::{Outcome, lock},
    events::{Batch, Event, Level, LoaderSample, Progress, Rows, RunInfo, ScanInfo, Stage, Superbatch, Task},
    text,
};

const SCHEMA: &str = "bruce.metrics/1";
const SPEED_TIME_CONSTANT: f64 = 2.0;
const ETA_SUPERBATCHES: usize = 3;
const SAMPLE_SPAN: Duration = Duration::from_secs(10);
const STALL_FLOOR: Duration = Duration::from_secs(1);
const STALL_INTERVALS: u32 = 20;
const STALL_WARNING_EVERY: Duration = Duration::from_secs(60);
const ZERO: LoaderSample = LoaderSample { blocked: Duration::ZERO, reading: Duration::ZERO, mapping: Duration::ZERO };

pub struct Setup
{
    pub task: Task,
    pub subject: String,
    pub facts: String,
    pub log: Option<PathBuf>,
    pub metrics: Option<PathBuf>,
}

pub struct Hub(Mutex<State>);

struct State
{
    live: Live,
    log: Sink,
    journal: Sink,
    warnings: Vec<String>,
}

struct Sink
{
    path: Option<PathBuf>,
    file: Option<File>,
}

impl Hub
{
    pub fn new(setup: Setup, log_every: usize) -> Self
    {
        let log = Sink { path: setup.log.clone(), file: None };
        let journal = Sink { path: setup.metrics.clone(), file: None };
        let live = Live::new(setup, log_every, Instant::now());
        Self(Mutex::new(State { live, log, journal, warnings: Vec::new() }))
    }

    pub fn emit(&self, event: Event)
    {
        let (now, at) = (Instant::now(), OffsetDateTime::now_utc());
        let mut state = lock(&self.0);
        let lines = state.live.apply(&event, now, at);
        state.write(&lines, at);
    }

    pub fn message(&self, level: Level, text: impl Into<String>) { self.emit(Event::Message(level, text.into())); }

    pub fn note(&self, level: Level, text: impl Into<String>)
    {
        let mut state = lock(&self.0);
        let line = state.live.message(level, &text.into(), false);
        state.write(&Lines { log: vec![line], journal: Vec::new() }, OffsetDateTime::now_utc());
    }

    pub fn finished_inputs(&self, inputs: impl Iterator<Item = (u64, u64)>, finished: &mut usize, since: &mut Instant)
    {
        for (input, (games, positions)) in inputs.enumerate().skip(*finished) {
            self.message(Level::Info, text::finished(input, games, positions, since.elapsed()));
            (*finished, *since) = (input + 1, Instant::now());
        }
    }

    pub fn view<T>(&self, read: impl FnOnce(&Live) -> T) -> T { read(&lock(&self.0).live) }

    pub fn finish(&self, outcome: Outcome) -> Vec<String>
    {
        let at = OffsetDateTime::now_utc();
        let mut state = lock(&self.0);
        let mut warnings = mem::take(&mut state.warnings);
        let mut lines = end(outcome, at);
        lines.log.splice(..0, warnings.iter().map(|warning| (Level::Warning, warning.clone())));
        state.write(&lines, at);
        warnings.append(&mut state.warnings);
        warnings
    }
}

impl State
{
    fn write(&mut self, lines: &Lines, at: OffsetDateTime)
    {
        if lines.log.is_empty() && lines.journal.is_empty() {
            return;
        }
        let at = at.to_offset(UtcOffset::UTC);
        let ((year, month, day), (hour, minute, second, milli)) = (at.to_calendar_date(), at.to_hms_milli());
        let stamp = format!("{year:04}-{:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{milli:03}Z", u8::from(month));
        let log = lines.log.iter().flat_map(|(level, content)| {
            let level = match level {
                Level::Error => "ERROR",
                Level::Warning => "WARN",
                Level::Info => "INFO",
                Level::Detail => "DETAIL",
            };
            let prefix = format!("{stamp} {level:<5}");
            content.trim_end_matches('\n').split('\n').map(move |part| format!("{prefix} {part}\n"))
        });
        let journal = lines.journal.iter().map(|line| format!("{line}\n"));

        for (sink, text, name) in
            [(&mut self.log, log.collect::<String>(), "Text"), (&mut self.journal, journal.collect(), "Metrics")]
        {
            let Some(path) = sink.path.as_deref().filter(|_| !text.is_empty()) else { continue };
            let file = sink.file.take().map_or_else(|| open(path), Ok);
            if let Err(error) = file.and_then(|file| sink.file.insert(file).write_all(text.as_bytes())) {
                self.warnings.push(sanitize(&format!("{name} logging disabled: {error}")));
                sink.path = None;
            }
        }
    }
}

fn open(path: &Path) -> io::Result<File>
{
    let context = |error: io::Error| io::Error::new(error.kind(), format!("cannot open {}: {error}", path.display()));

    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(context)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path).map_err(context)?;
    let mut last = *b"\n";
    if let Ok(mut tail) = File::open(path).and_then(|mut tail| tail.seek(SeekFrom::End(-1)).map(|_| tail)) {
        tail.read_exact(&mut last).map_err(context)?;
    }
    file.write_all(if last == *b"\n" { b"" } else { b"\n" }).map_err(context)?;
    Ok(file)
}

#[derive(Default)]
struct Lines
{
    log: Vec<(Level, String)>,
    journal: Vec<String>,
}

pub struct Live
{
    pub setup: Setup,
    pub started: Instant,
    pub run: Option<RunInfo>,
    pub ready: bool,
    pub stage: Option<Stage>,
    pub status: Option<String>,
    pub warnings: usize,
    pub last: Option<Superbatch>,
    pub checkpoint: Option<PathBuf>,
    pub stop_committed: Option<usize>,
    pub scan: Option<(ScanInfo, Instant)>,
    pub preview: Rows,
    pub progress: Vec<Progress>,
    pub input: usize,
    pub superbatch: usize,
    pub batch: usize,
    pub learning_rate: f32,
    pub stalls: u32,
    pub stall_time: Duration,
    log_every: usize,
    since: Option<Instant>,
    trained: u64,
    speed: Option<f64>,
    mark: Option<(Instant, bool)>,
    interval: (Duration, u32),
    loss: (f64, u32),
    window_loss: f64,
    window_batches: usize,
    window_start: Option<Instant>,
    durations: VecDeque<Duration>,
    loader: VecDeque<(Instant, LoaderSample)>,
    boundary: Instant,
    boundary_loader: LoaderSample,
    boundary_stalls: u32,
    warned: Option<Instant>,
    unwarned: u32,
    scanned: VecDeque<(Instant, u64)>,
}

impl Live
{
    pub fn new(setup: Setup, log_every: usize, now: Instant) -> Self
    {
        Self {
            setup,
            started: now,
            run: None,
            ready: false,
            stage: None,
            status: None,
            warnings: 0,
            last: None,
            checkpoint: None,
            stop_committed: None,
            scan: None,
            preview: Vec::new(),
            progress: Vec::new(),
            input: 0,
            superbatch: 0,
            batch: 0,
            learning_rate: 0.0,
            stalls: 0,
            stall_time: Duration::ZERO,
            log_every,
            since: None,
            trained: 0,
            speed: None,
            mark: None,
            interval: (Duration::ZERO, 0),
            loss: (0.0, 0),
            window_loss: 0.0,
            window_batches: 0,
            window_start: None,
            durations: VecDeque::new(),
            loader: VecDeque::new(),
            boundary: now,
            boundary_loader: ZERO,
            boundary_stalls: 0,
            warned: None,
            unwarned: 0,
            scanned: VecDeque::new(),
        }
    }

    fn apply(&mut self, event: &Event, now: Instant, at: OffsetDateTime) -> Lines
    {
        let t = stamp(at);
        let mut lines = Lines::default();
        let info = |text: String| (Level::Info, text);
        match event {
            Event::Started(run) => {
                let RunInfo { name, architecture, device, backend, labels, .. } = &**run;
                let TrainingSteps { batch_size, batches_per_superbatch, start_superbatch, end_superbatch } = run.steps;
                let chip = run.device_architecture.as_deref().unwrap_or("unknown");
                self.superbatch = start_superbatch;
                self.learning_rate = run.schedule.get(self.superbatch.saturating_sub(1)).map_or(0.0, |&(lr, _)| lr);
                self.run = Some((**run).clone());
                self.loader.clear();
                lines.log.push(info(format!(
                    "run started: {name}, {architecture}, {device} ({chip}, {backend}), batch {batch_size} x \
                     {batches_per_superbatch}, superbatches {start_superbatch}-{end_superbatch}, labels {labels}"
                )));
                lines.journal.push(session(run, t));
            },
            Event::Ready(setup) => {
                (self.ready, self.since, self.mark) = (true, Some(now), Some((now, false)));
                (self.window_start, self.boundary, self.boundary_loader) = (Some(now), now, self.latest());
                lines.log.push(info(format!("training ready: setup {:.1}s", setup.as_secs_f64())));
            },
            Event::Stage(stage) => {
                if *stage == Stage::Training && self.stage != Some(Stage::Training) {
                    self.mark = Some((now, false));
                }
                self.stage = Some(*stage);
                lines.log.push((Level::Detail, format!("stage: {}", stage.name())));
            },
            Event::Batch(sample) => self.batch(sample, now, t, &mut lines),
            Event::Loader(sample) => recent(&mut self.loader, now, sample.clone()),
            Event::Superbatch(summary) => self.close(summary, now, t, &mut lines),
            Event::Checkpoint(saved) => {
                self.checkpoint = Some(saved.path.clone());
                let (path, spent) = (saved.path.display(), saved.duration.as_millis());
                lines.log.push(info(format!("checkpoint {}: {path} ({spent} ms, {})", saved.superbatch, saved.reason)));
                lines.journal.push(json!({ "k": "checkpoint", "t": t, "sb": saved.superbatch }).to_string());
            },
            Event::StopCommitted(superbatch) => {
                self.stop_committed = Some(*superbatch);
                lines.log.push(info(format!("stop committed at superbatch {superbatch}")));
            },
            Event::Scan(scan @ ScanInfo { task, inputs, threads, sampling, .. }) => {
                let (task, inputs) = (task.name(), inputs.len());
                self.progress = (0..inputs).map(unread).collect();
                (self.input, self.scanned) = (0, VecDeque::from([(now, 0)]));
                self.scan = Some((scan.clone(), now));
                lines.log.push(info(format!("scan started: {task}, {inputs} inputs, {threads} threads, {sampling}")));
            },
            Event::Progress(progress) => {
                let known = self.progress.len();
                self.progress.extend((known..=progress.input).map(unread));
                (self.input, self.progress[progress.input]) = (progress.input, progress.clone());
                recent(&mut self.scanned, now, self.progress.iter().map(|input| input.positions).sum());
            },
            Event::Preview(rows) => self.preview = rows.clone(),
            Event::Message(level, text) => lines.log.push(self.message(*level, text, true)),
        }
        lines
    }

    pub fn loss(&self) -> Option<f32>
    {
        let (sum, count) = self.loss;
        if count > 0 { Some((sum / f64::from(count)) as f32) } else { self.last.as_ref().map(|last| last.loss) }
    }

    pub fn speed(&self) -> Option<f64> { self.speed }

    pub fn session_speed(&self, now: Instant) -> Option<f64>
    {
        let positions = self.trained as f64 * self.steps()?.batch_size as f64;
        rate(positions, now.saturating_duration_since(self.since?))
    }

    pub fn eta(&self, now: Instant) -> (Option<Duration>, Option<Duration>)
    {
        let Some(steps) = self.steps().filter(|steps| steps.batches_per_superbatch > 0) else { return (None, None) };
        let batches = steps.batches_per_superbatch as f64;
        let left = 1.0 - (self.batch as f64 / batches).min(1.0);
        let remaining = steps.end_superbatch.saturating_sub(self.superbatch) as f64;
        let seconds = |value: f64| Duration::try_from_secs_f64(value).ok();

        if !self.durations.is_empty() {
            let mean = self.durations.iter().map(Duration::as_secs_f64).sum::<f64>() / self.durations.len() as f64;
            return (seconds(mean * left), seconds(mean * (remaining + left)));
        }

        let Some(since) = self.since else { return (None, None) };
        let per_second = self.trained as f64 / now.saturating_duration_since(since).as_secs_f64();
        if !(per_second.is_finite() && per_second > 0.0) {
            return (None, None);
        }
        (seconds(batches * left / per_second), seconds(batches * (remaining + left) / per_second))
    }

    pub fn loader_idle(&self, now: Instant) -> Option<f32>
    {
        span(&self.loader, now).and_then(|(_, first, last)| idle(first, last))
    }

    pub fn stalled_for(&self, now: Instant) -> Option<Duration>
    {
        let (marked, _) = self.mark?;
        let gap = now.saturating_duration_since(marked);
        self.stalled(gap).then_some(gap)
    }

    pub fn scan_rate(&self, now: Instant) -> Option<f64>
    {
        let (elapsed, first, last) = span(&self.scanned, now)?;
        Some(last.saturating_sub(*first) as f64 / elapsed.as_secs_f64())
    }

    fn batch(&mut self, sample: &Batch, now: Instant, t: f64, lines: &mut Lines)
    {
        let Some(steps) = self.steps() else { return };
        if let Some((marked, after_batch)) = self.mark.replace((now, true)) {
            let gap = now.saturating_duration_since(marked);
            if !gap.is_zero() {
                let seconds = gap.as_secs_f64();
                let rate = steps.batch_size as f64 / seconds;
                let alpha = 1.0 - (-seconds / SPEED_TIME_CONSTANT).exp();
                self.speed = Some(self.speed.map_or(rate, |speed| speed + alpha * (rate - speed)));
            }
            if self.stalled(gap) {
                self.stalls += 1;
                self.stall_time += gap;
                self.boundary_stalls += 1;
                if self.warned.is_some_and(|warned| now.saturating_duration_since(warned) < STALL_WARNING_EVERY) {
                    self.unwarned += 1;
                } else {
                    self.warned = Some(now);
                    let more = match mem::take(&mut self.unwarned) {
                        0 => String::new(),
                        more => format!(" ({more} more since the last warning)"),
                    };
                    let (gap, superbatch, batch) = (gap.as_secs_f64(), sample.superbatch, sample.batch);
                    let warning = format!("loader stall {gap:.1}s at superbatch {superbatch} batch {batch}{more}");
                    lines.log.push(self.message(Level::Warning, &warning, true));
                }
            } else if after_batch {
                let (total, count) = self.interval;
                self.interval = (total + gap, count + 1);
            }
        }

        let (sum, count) = self.loss;
        (self.superbatch, self.batch) = (sample.superbatch, sample.batch);
        self.learning_rate = sample.learning_rate;
        self.trained += 1;
        self.loss = (sum + f64::from(sample.loss), count + 1);
        self.window_loss += f64::from(sample.loss);
        self.window_batches += 1;
        if self.window_batches < self.log_every && sample.batch != steps.batches_per_superbatch {
            return;
        }
        let (loss, batches) = (mem::take(&mut self.window_loss), mem::take(&mut self.window_batches));
        let start = self.window_start.replace(now);
        let positions = batches as f64 * steps.batch_size as f64;
        let speed = start.and_then(|start| rate(positions, now.saturating_duration_since(start))).unwrap_or(0.0);
        let line = json!({
            "k": "batch",
            "t": t,
            "sb": sample.superbatch,
            "b": sample.batch,
            "n": batches,
            "loss": short((loss / batches as f64) as f32),
            "lr": short(sample.learning_rate),
            "wdl": short(sample.wdl),
            "pps": whole(speed),
        });
        lines.journal.push(line.to_string());
    }

    fn close(&mut self, summary: &Superbatch, now: Instant, t: f64, lines: &mut Lines)
    {
        let Some(steps) = self.steps() else { return };
        let completed = summary.superbatch;
        let more = completed < steps.end_superbatch;
        (self.superbatch, self.batch) =
            if more { (completed + 1, 0) } else { (completed, steps.batches_per_superbatch) };
        let latest = self.latest();
        let share = idle(&self.boundary_loader, &latest);
        let (opened, stalls) = (mem::replace(&mut self.boundary, now), mem::take(&mut self.boundary_stalls));
        if self.durations.len() == ETA_SUPERBATCHES {
            self.durations.pop_front();
        }
        self.durations.push_back(now.saturating_duration_since(opened));
        (self.boundary_loader, self.loss, self.last) = (latest, (0.0, 0), Some(summary.clone()));

        let seconds = summary.duration.as_secs_f64();
        let positions = (steps.batch_size * steps.batches_per_superbatch) as f64;
        let speed = (!summary.duration.is_zero()).then(|| positions / seconds);
        let pace = speed.map(|speed| format!("{:.2} Mpos/s, ", speed / 1e6)).unwrap_or_default();
        let idling = share.map(|share| format!("idle {:.0}%, ", f64::from(share) * 100.0)).unwrap_or_default();
        let line = format!(
            "superbatch {completed}/{}: loss {:.6}, lr {:.3e}, wdl {:.3}, {pace}{seconds:.1}s, {idling}stalls {stalls}",
            steps.end_superbatch, summary.loss, summary.learning_rate, summary.wdl
        );
        lines.log.push((Level::Info, line));
        let line = json!({
            "k": "superbatch",
            "t": t,
            "sb": completed,
            "loss": short(summary.loss),
            "seconds": seconds,
            "speed": speed.and_then(whole),
            "idle": share.map(short),
            "stalls": stalls,
        });
        lines.journal.push(line.to_string());
    }

    fn message(&mut self, level: Level, text: &str, shown: bool) -> (Level, String)
    {
        let text = sanitize(text);
        match level {
            Level::Warning if shown => self.warnings += 1,
            Level::Detail if shown => self.status = Some(text.clone()),
            _ => {},
        }
        (level, text)
    }

    fn stalled(&self, gap: Duration) -> bool
    {
        let (total, count) = self.interval;
        self.stage == Some(Stage::Training) && count > 0 && gap > (total / count * STALL_INTERVALS).max(STALL_FLOOR)
    }

    fn steps(&self) -> Option<TrainingSteps> { self.run.as_ref().map(|run| run.steps) }

    fn latest(&self) -> LoaderSample { self.loader.back().map_or(ZERO, |(_, sample)| sample.clone()) }
}

fn session(info: &RunInfo, t: f64) -> String
{
    let steps = &info.steps;
    let parent = info.resumed.as_ref().filter(|resumed| resumed.run != info.name);
    let line = json!({
        "k": "session",
        "t": t,
        "schema": SCHEMA,
        "steps": {
            "batch_size": steps.batch_size,
            "batches_per_superbatch": steps.batches_per_superbatch,
            "first_superbatch": steps.start_superbatch,
            "final_superbatch": steps.end_superbatch,
        },
        "schedule": info.schedule.iter().map(|&(lr, wdl)| (short(lr), short(wdl))).collect::<Vec<_>>(),
        "phases": info.phases,
        "labels": info.labels,
        "config": info.configuration,
        "parent": parent.map(|resumed| json!({ "run": resumed.run, "superbatch": resumed.completed })),
    });
    line.to_string()
}

fn end(outcome: Outcome, at: OffsetDateTime) -> Lines
{
    Lines {
        log: vec![(Level::Info, format!("run ended: {}", outcome.name()))],
        journal: vec![json!({ "k": "end", "t": stamp(at) }).to_string()],
    }
}

fn sanitize(text: &str) -> String
{
    strip_str(text)
        .flat_map(str::chars)
        .filter_map(|ch| match ch {
            '\n' => Some(ch),
            '\t' => Some(' '),
            ch if ch.is_control() => None,
            ch => Some(ch),
        })
        .collect()
}

fn stamp(at: OffsetDateTime) -> f64 { (at.unix_timestamp_nanos() / 1_000_000) as f64 / 1000.0 }

fn short(value: f32) -> f64 { value.to_string().parse().unwrap_or(f64::NAN) }

fn whole(value: f64) -> Option<u64> { value.is_finite().then(|| value.round() as u64) }

fn unread(input: usize) -> Progress { Progress { input, done: 0, games: 0, positions: 0 } }

fn rate(amount: f64, elapsed: Duration) -> Option<f64>
{
    (amount > 0.0 && !elapsed.is_zero()).then(|| amount / elapsed.as_secs_f64())
}

fn idle(first: &LoaderSample, last: &LoaderSample) -> Option<f32>
{
    let blocked = last.blocked.saturating_sub(first.blocked);
    let total = blocked + last.reading.saturating_sub(first.reading) + last.mapping.saturating_sub(first.mapping);
    (!total.is_zero()).then(|| (blocked.as_secs_f64() / total.as_secs_f64()) as f32)
}

fn recent<T>(samples: &mut VecDeque<(Instant, T)>, now: Instant, value: T)
{
    samples.push_back((now, value));
    while samples.front().is_some_and(|(at, _)| now.saturating_duration_since(*at) > SAMPLE_SPAN) {
        samples.pop_front();
    }
}

fn span<T>(samples: &VecDeque<(Instant, T)>, now: Instant) -> Option<(Duration, &T, &T)>
{
    let (first_at, first) = samples.iter().find(|(at, _)| now.saturating_duration_since(*at) <= SAMPLE_SPAN)?;
    let (last_at, last) = samples.back()?;
    let elapsed = last_at.saturating_duration_since(*first_at);
    (!elapsed.is_zero()).then_some((elapsed, first, last))
}
