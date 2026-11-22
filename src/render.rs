use std::{
    f64::consts::TAU,
    fmt::Write,
    iter,
    path::Path,
    time::{Duration, Instant},
};

use anstyle::{RgbColor, Style};
use bullet_trainer::run::TrainingSteps;

use crate::{
    control::{Outcome, Stop},
    events::{Ending, Rows, Stage, Task},
    telemetry::Live,
    text,
};

pub const WIDTH: usize = 78;
pub const HEADER: usize = WIDTH - LOGO - 2;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const GUTTER: usize = 4;
const LABEL: usize = 14;
const COUNT: usize = 13;
const PERCENT: usize = 8;
const KEY: usize = 15;
const RESUME: &str = "bruce train --resume ";
const STARVED: f32 = 0.05;
const STILL: f64 = 17.0;
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const LOGO: usize = 14;
const SHAPE: [&[u8]; 6] = [
    b"..............####..",
    b"...........######...",
    b"........########....",
    b"......#########.....",
    b"....###########.....",
    b"..##############....",
];
const QUADRANTS: [char; 16] = [' ', '▘', '▝', '▀', '▖', '▌', '▞', '▛', '▗', '▚', '▐', '▜', '▄', '▙', '▟', '█'];
const EIGHTHS: [char; 6] = [' ', '▁', '▂', '▃', '▄', '▅'];
const GUSTS: [(usize, f64, f64, usize); 4] =
    [(0, 0.9, 3.0, 3), (1, 1.2, 11.0, 2), (0, 0.7, 20.0, 2), (1, 0.8, 17.0, 3)];

const FIN: RgbColor = RgbColor(0x8f, 0xa3, 0xb8);
const WATER: RgbColor = RgbColor(0x3a, 0x96, 0xdd);
const CREST: RgbColor = RgbColor(0x7c, 0xc0, 0xf0);
const DEEP: RgbColor = RgbColor(0x24, 0x60, 0x8f);
const WIND: RgbColor = RgbColor(0x4a, 0x4a, 0x4a);

const PLAIN: Style = Style::new();
const TEXT: Style = RgbColor(0xe6, 0xe6, 0xe6).on_default();
const STRONG: Style = RgbColor(0xff, 0xff, 0xff).on_default().bold();
const SLATE: Style = RgbColor(0x78, 0x8b, 0xa0).on_default();
const SEA: Style = WATER.on_default();
const ABYSS: Style = RgbColor(0x1b, 0x37, 0x53).on_default();
const GREEN: Style = RgbColor(0x78, 0xc4, 0x7e).on_default();
const AMBER: Style = RgbColor(0xff, 0xc1, 0x07).on_default();
const RED: Style = RgbColor(0xff, 0x6b, 0x80).on_default();

type Row = Vec<(Style, String)>;
type Cell = (char, Option<RgbColor>, Option<RgbColor>);

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
    let Frame { status, bars, fields } = frame(live, stop, now);
    let (key, hint) = match (stop, live.setup.task) {
        (Stop::Running, _) => ("ctrl+c", "stop safely"),
        (Stop::AtBoundary, Task::Train) => ("ctrl+c again", "abort without saving"),
        (Stop::AtBoundary, Task::Analyze) => ("ctrl+c again", "abort without a report"),
        (Stop::AtBoundary, Task::Check | Task::Benchmark) => ("ctrl+c again", "abort without a summary"),
        (Stop::Committed, _) => ("ctrl+c twice", "exit without waiting"),
        (Stop::Armed | Stop::Now | Stop::Exit, _) => ("ctrl+c again", "exit immediately"),
    };
    let (mood, cue) = match stop {
        Stop::Running => (SEA.bold(), TEXT.bold()),
        Stop::Now | Stop::Exit => (RED.bold(), RED.bold()),
        Stop::AtBoundary | Stop::Committed | Stop::Armed => (AMBER.bold(), AMBER.bold()),
    };

    let (word, detail) =
        status.split_once(" · ").map_or((status.as_str(), None), |(word, detail)| (word, Some(detail)));
    let mut line = vec![(mood, format!("{} {word}", SPINNER[pulse % SPINNER.len()]))];
    line.extend(detail.into_iter().flat_map(|detail| [(SLATE, " · ".to_owned()), (TEXT, detail.to_owned())]));
    line.push((SLATE, format!(" · {}", elapsed(live, now))));

    let tick = now.saturating_duration_since(live.started).as_millis() / 100;
    let hint = vec![(cue, key.to_owned()), (SLATE, format!("  {hint}"))];
    let sections = [vec![line], meters(&bars, width, color), grid(&fields, width), vec![hint]];
    paint(stack(header(live, color.then_some(tick as f64)), sections), width, color)
}

pub fn plain(live: &Live, stop: Stop, now: Instant) -> Option<String>
{
    let Frame { status, bars, fields } = frame(live, stop, now);
    let moving = bars.iter().any(|bar| bar.fraction > 0.0);
    let status = format!("{status} · {}", elapsed(live, now));
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
    let (glyph, word, style) = match (outcome, task) {
        (Outcome::Finished, Task::Check) => ("✓", "passed", GREEN),
        (Outcome::Finished, _) => ("✓", "complete", GREEN),
        (Outcome::Stopped, _) => ("■", "stopped", AMBER),
        (Outcome::Aborted, _) => ("×", "aborted", RED),
        (Outcome::Failed, _) => ("×", "failed", RED),
    };
    let title = vec![(style.bold(), format!("{glyph} {} {word}", task.name()))];

    let mut errors = Vec::new();
    if let Some((error, help)) = failure {
        errors.extend(error.lines().flat_map(|line| wrap(line, TEXT, width)));
        errors.extend(help.iter().flat_map(|help| wrap(help, SLATE, width)));
    }

    let room = width.saturating_sub(KEY);
    let (trained, saved) = trained(live, outcome, now, room);
    let see = if live.setup.log.is_some() { ", see log" } else { "" };
    let warned = (live.warnings > 0).then(|| fact("Warnings", vec![(AMBER, format!("{}{see}", live.warnings))]));
    let rows = ending
        .rows
        .iter()
        .map(|(label, value, qualifier)| fact(label, vec![(TEXT, value.clone()), (SLATE, qualifier.clone())]));
    let facts = rows.chain(trained).chain(warned).collect();
    let notes = warnings.iter().flat_map(|warning| wrap(&format!("Warning: {warning}"), AMBER, width)).collect();

    let log = live.setup.log.as_deref().map(|log| ("Log", log));
    let metrics = live.setup.metrics.as_deref().filter(|_| live.run.is_some()).map(|metrics| ("Metrics", metrics));
    let files = ending.files.iter().map(|(label, path)| (label.as_str(), Path::new(path))).chain(log).chain(metrics);
    let files = files.map(|(label, path)| fact(label, vec![(TEXT, text::path(path, room))])).chain(saved).collect();

    let grid = grid(&fields(&ending.grid).collect::<Vec<_>>(), width);
    let sections = [vec![title], errors, grid, facts, notes, files];
    paint(stack(header(live, color.then_some(STILL)), sections), width, color)
}

pub fn warning(text: &str) -> String { format!("{AMBER}Warning: {text}{AMBER:#}") }

fn frame(live: &Live, stop: Stop, now: Instant) -> Frame
{
    let frame = match live.setup.task {
        Task::Train => training(live, stop, now),
        _ => scanning(live, now),
    };
    let mut frame = frame.unwrap_or_else(|| {
        let status = live.status.clone().unwrap_or_else(|| "Preparing".to_owned());
        Frame { status, bars: Vec::new(), fields: Vec::new() }
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
        Task::Train if aborting => "Aborting · no checkpoint will be saved".to_owned(),
        Task::Train => format!("Stopping · after superbatch {superbatch}, then saving a checkpoint"),
        Task::Analyze if aborting => "Aborting · no report will be written".to_owned(),
        Task::Analyze => "Stopping · finishing the blocks in flight, then writing the report".to_owned(),
        Task::Check | Task::Benchmark if aborting => "Aborting".to_owned(),
        Task::Check => "Stopping · after the blocks in flight".to_owned(),
        Task::Benchmark => match measuring {
            Some(input) => format!("Stopping · ending the {} measurement", text::file_name(&input.path)),
            None => "Stopping".to_owned(),
        },
    }
}

fn training(live: &Live, stop: Stop, now: Instant) -> Option<Frame>
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
    let starved = live.stalled_for(now).is_some() || idle.is_some_and(|idle| idle < STARVED);
    let loader = match idle {
        _ if starved => "starved".to_owned(),
        Some(idle) => format!("idle {:.0}%", idle * 100.0),
        None => unknown(),
    };
    let (superbatch_eta, run_eta) = live.eta(now);
    let run_eta = run_eta.filter(|_| stop == Stop::Running);

    let trained = superbatch.saturating_sub(1) as u64 * per_superbatch + batch;
    let batches = format!("{}/{}", text::count(batch), text::count(per_superbatch));
    Some(Frame {
        status,
        bars: vec![
            bar(
                "Superbatches",
                format!("{superbatch}/{final_superbatch}"),
                trained,
                final_superbatch as u64 * per_superbatch,
            ),
            bar("Batches", batches, batch, per_superbatch),
        ],
        fields: vec![
            field("Loss", live.loss().map_or_else(unknown, |loss| format!("{loss:.6}"))),
            field("Throughput", live.speed().map_or_else(unknown, text::rate)),
            field("Learning rate", text::decimal(live.learning_rate)),
            Field { alert: starved, ..field("Loader", loader + &stalls) },
            field("Superbatch ETA", superbatch_eta.map_or_else(unknown, text::span)),
            field("Run ETA", run_eta.map_or_else(unknown, text::span)),
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
    let (status, bars, fields): (_, _, Vec<Field>) = match live.setup.task {
        Task::Check => {
            let count = |done, total| match info.head {
                Some(_) => format!("{}/{} games", text::count(done), text::count(total)),
                None => text::bytes_of(done, total),
            };
            let fields = preview.chain([speed, field("ETA", eta(total - done))]).collect();
            let bars =
                vec![bar("All sources", count(done, total), done, total), bar(&name, count(own, whole), own, whole)];
            (format!("Scanning · source {number} of {inputs}"), bars, fields)
        },
        Task::Benchmark => {
            let throughput = live.scan_rate(now).map_or_else(unknown, |rate| format!("{:.2} Mpos/s", rate / 1e6));
            let measured = text::scaled(live.progress.get(current).map_or(0, |input| input.positions) as f64);
            let remaining = field("Remaining", text::span(Duration::from_millis(total - done)));
            let fields = vec![field("Throughput", throughput), field("Positions", measured), remaining];
            let overall = bar("Measurements", format!("{number}/{inputs}"), done, total);
            let seconds = bar("Seconds", format!("{}/{}", own / 1000, whole / 1000), own, whole);
            (format!("Measuring · {name}"), vec![overall, seconds], fields)
        },
        Task::Analyze => {
            let read = match info.head {
                Some(_) => format!("Games {}/{}", text::count(own), text::count(whole)),
                None => text::bytes_of(own, whole),
            };
            let counts = [field("Games", text::count(games)), speed, field("Positions", text::count(positions))];
            let etas = [field("Input ETA", eta(whole - own)), field("Run ETA", eta(total - done))];
            let overall = bar("Inputs", format!("{number}/{inputs}"), done, total);
            let bars = vec![overall, bar(&name, read, own, whole)];
            (detail("Scanning"), bars, counts.into_iter().chain(preview).chain(etas).collect())
        },
        Task::Train => return None,
    };

    Some(Frame { status, bars, fields })
}

fn trained(live: &Live, outcome: Outcome, now: Instant, room: usize) -> (Vec<Row>, Vec<Row>)
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
    let batches = live.superbatch.saturating_sub(first) * batches_per_superbatch + live.batch;
    let stalls = match live.stalls {
        0 => "none".to_owned(),
        stalls => format!("{stalls} ({:.1}s)", live.stall_time.as_secs_f64()),
    };

    let mut rows = vec![("Superbatches", superbatches), ("Batches", text::count(batches as u64))];
    rows.extend(live.last.as_ref().map(|summary| ("Final loss", format!("{:.6}", summary.loss))));
    rows.extend(live.session_speed(now).map(|speed| ("Throughput", format!("{} average", text::rate(speed)))));
    rows.extend([("Loader stalls", stalls), ("Elapsed", elapsed(live, now))]);
    let rows = rows.into_iter().map(|(label, value)| fact(label, vec![(TEXT, value)])).collect();

    let latest = live.checkpoint.as_deref().or(resumed.map(|resumed| resumed.checkpoint.as_path()));
    let mut files: Vec<Row> =
        latest.iter().map(|path| fact("Checkpoint", vec![(TEXT, text::path(path, room))])).collect();
    let resume = latest.filter(|_| outcome != Outcome::Finished);
    let command = |path: &Path| format!("{RESUME}{}", text::path(path, room.saturating_sub(RESUME.len())));
    files.extend(resume.map(|path| fact("Resume", vec![(STRONG, command(path))])));
    if let Some(resumed) = resumed.filter(|resumed| resumed.run != run.name || !resumed.extended) {
        let label = if resumed.run == run.name { "Resumed from" } else { "Forked from" };
        files.push(fact(label, vec![(TEXT, text::path(&resumed.checkpoint, room))]));
    }
    (rows, files)
}

fn header(live: &Live, tick: Option<f64>) -> Vec<Row>
{
    let (subject, facts) = match &live.run {
        Some(run) => {
            let device = run.device.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ");
            let mut facts = format!("{} · {device}", run.architecture);
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

    let mut identity = vec![(TEXT, live.setup.task.name().to_owned())];
    if !subject.is_empty() {
        identity.extend([(SLATE, " · ".to_owned()), (STRONG, subject.clone())]);
    }
    let facts = if facts.is_empty() { Row::new() } else { vec![(SLATE, facts)] };
    let lines = [vec![(STRONG, "Bruce".to_owned()), (SLATE, format!(" {VERSION}"))], identity, facts];
    let Some(tick) = tick else { return lines.into_iter().filter(|line| !line.is_empty()).collect() };

    let beside = iter::once(Row::new()).chain(lines).chain(iter::once(Row::new()));
    let rows = logo(tick).into_iter().zip(beside).map(|(mut row, line)| {
        if !line.is_empty() {
            row.push((PLAIN, "  ".to_owned()));
            row.extend(line);
        }
        row
    });
    rows.collect()
}

fn logo(tick: f64) -> Vec<Row>
{
    let under = TAU * (7.0 - tick * 0.375) / 6.0;
    let sway = (4.0 + (0.9 * under.cos()).round_ties_even()) as usize;
    let bob = usize::from(under.sin() < -0.5);
    let fin = |x: usize, y: usize| {
        let line = y.checked_sub(bob).and_then(|y| SHAPE.get(y));
        line.zip(x.checked_sub(sway)).and_then(|(line, x)| line.get(x)) == Some(&b'#')
    };

    let mut cells: [[Cell; LOGO]; 5] = [[(' ', None, None); LOGO]; 5];
    for (r, row) in cells[1..4].iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let quadrant: usize = (0..4).filter(|&i| fin(2 * c + i % 2, 2 * r + i / 2)).map(|i| 1 << i).sum();
            if quadrant > 0 {
                *cell = (QUADRANTS[quadrant], Some(FIN), None);
            }
        }
    }

    let [.., base, floor] = &mut cells;
    for (c, (cell, ground)) in base.iter_mut().zip(floor).enumerate() {
        let x = c as f64 - tick * 0.375;
        let swell = (TAU * x / 6.0).sin();
        *ground = ('█', Some(if swell < -0.2 { DEEP } else { WATER }), None);
        let level = (1.6 + 2.6 * swell + 0.6 * (TAU * x / 2.9 + 1.3).sin()).round_ties_even().clamp(0.0, 5.0) as usize;
        if level == 0 {
            continue;
        }
        let crest = Some(if level >= 4 { CREST } else { WATER });
        *cell = match *cell {
            (_, None, _) => (EIGHTHS[level], crest, None),
            ('█', ..) => (EIGHTHS[level], crest, Some(FIN)),
            _ if level >= 3 => (EIGHTHS[level], crest, None),
            kept => kept,
        };
    }

    for (row, speed, offset, length) in GUSTS {
        let start = ((tick * speed + offset) % (LOGO + 8) as f64) as usize;
        for i in 0..length {
            let Some(c) = (start + i).checked_sub(4).filter(|&c| c < LOGO) else { continue };
            if cells[row][c].1.is_none() {
                cells[row][c] = (if i + 1 < length { '─' } else { '╴' }, Some(WIND), None);
            }
        }
    }

    let styled = |&(glyph, fg, bg): &Cell| {
        let style = fg.map_or(PLAIN, |fg| fg.on_default().bg_color(bg.map(Into::into)));
        (style, glyph.to_string())
    };
    cells.iter().map(|row| row.iter().map(styled).collect()).collect()
}

fn elapsed(live: &Live, now: Instant) -> String { text::span(now.saturating_duration_since(live.started)) }

fn unknown() -> String { "—".to_owned() }

fn field(label: &str, value: String) -> Field { Field { label: label.to_owned(), value, alert: false } }

fn fields(rows: &Rows) -> impl Iterator<Item = Field> { rows.iter().map(|(label, value)| field(label, value.clone())) }

fn fact(label: &str, value: Row) -> Row { iter::once((SLATE, format!("{label:<KEY$}"))).chain(value).collect() }

fn bar(label: &str, count: String, done: u64, total: u64) -> Bar
{
    let fraction = if total == 0 { 0.0 } else { (done as f64 / total as f64).min(1.0) };
    Bar { label: label.to_owned(), count, fraction }
}

fn meters(bars: &[Bar], width: usize, color: bool) -> Vec<Row>
{
    let track = if color { "━" } else { "─" };
    let label = bars.iter().map(|bar| bar.label.chars().count() + 2).fold(LABEL, usize::max);
    let count = bars.iter().map(|bar| bar.count.chars().count() + 2).fold(COUNT, usize::max);
    let length = width.saturating_sub(label + count + PERCENT);
    let meter = |bar: &Bar| {
        let filled = ((bar.fraction * length as f64).round() as usize).max(usize::from(bar.fraction > 0.0)).min(length);
        vec![
            (SLATE, format!("{:<label$}", bar.label)),
            (SEA, "━".repeat(filled)),
            (ABYSS, track.repeat(length - filled)),
            (TEXT, format!("{:>count$}", bar.count)),
            (SEA, format!("{:>PERCENT$}", format!("{:.1}%", bar.fraction * 100.0))),
        ]
    };
    bars.iter().map(meter).collect()
}

fn cell(field: &Field, width: usize) -> Row
{
    let used = field.label.chars().count() + field.value.chars().count();
    let value = if field.alert { AMBER } else { TEXT };
    let gap = " ".repeat(width.saturating_sub(used).max(2));
    vec![(SLATE, field.label.clone()), (PLAIN, gap), (value, field.value.clone())]
}

fn grid(fields: &[Field], width: usize) -> Vec<Row>
{
    let (column, gutter) = (width.saturating_sub(GUTTER) / 2, [(PLAIN, " ".repeat(GUTTER))]);
    let row = |pair: &[Field]| pair.iter().map(|field| cell(field, column)).collect::<Vec<_>>().join(&gutter[..]);
    fields.chunks(2).map(row).collect()
}

fn wrap(text: &str, style: Style, width: usize) -> Vec<Row>
{
    let room = width.saturating_sub(2).max(1);
    let pieces = |word: &str| word.chars().collect::<Vec<_>>().chunks(room).map(String::from_iter).collect::<Vec<_>>();
    let mut lines: Vec<String> = Vec::new();
    for word in text.split(' ').flat_map(pieces) {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => line.extend([" ", &word]),
            Some(_) => lines.push(format!("  {word}")),
            None => lines.push(word),
        }
    }
    lines.into_iter().map(|line| vec![(style, line)]).collect()
}

fn stack(head: Vec<Row>, sections: impl IntoIterator<Item = Vec<Row>>) -> Vec<Row>
{
    let sections = sections.into_iter().filter(|section| !section.is_empty());
    head.into_iter().chain(sections.flat_map(|section| iter::once(Row::new()).chain(section))).collect()
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
