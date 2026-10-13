use std::{
    io::{self, IsTerminal, Write},
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use console::{Style, Term, measure_text_width, strip_ansi_codes};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::logging::{Logger, Sample};
pub use crate::report::TrainingProgress;

const REDRAW_INTERVAL: Duration = Duration::from_millis(100);
const PLAIN_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum DisplayMode
{
    #[default]
    Auto,
    Live,
    Plain,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum Verbosity
{
    Quiet,
    #[default]
    Normal,
    Verbose,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ColorMode
{
    #[default]
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OutputOptions
{
    pub display: DisplayMode,
    pub verbosity: Verbosity,
    pub color: ColorMode,
}

#[derive(Clone, Debug)]
pub enum Progress
{
    Training(TrainingProgress),
    Work
    {
        completed: u64,
        total: Option<u64>,
        unit: &'static str,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome
{
    Success,
    Cancelled,
    Failed,
}

#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        formatter.write_str("The operation was cancelled")
    }
}

impl std::error::Error for Cancelled {}

impl Outcome
{
    fn label(self) -> &'static str
    {
        match self {
            Self::Success => "Complete",
            Self::Cancelled => "Cancelled",
            Self::Failed => "Failed",
        }
    }
}

#[derive(Clone, Copy)]
struct Capabilities
{
    terminal: bool,
    dumb: bool,
    ci: bool,
    no_color: bool,
}

impl Capabilities
{
    fn detect() -> Self
    {
        Self {
            terminal: io::stderr().is_terminal(),
            dumb: std::env::var("TERM").is_ok_and(|term| term == "dumb"),
            ci: std::env::var_os("CI").is_some(),
            no_color: std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()),
        }
    }
}

#[derive(Clone, Copy)]
struct Policy
{
    live: bool,
    color: bool,
    verbosity: Verbosity,
}

impl Policy
{
    fn resolve(options: OutputOptions, capabilities: Capabilities) -> Self
    {
        let live = options.display != DisplayMode::Plain
            && capabilities.terminal
            && !capabilities.dumb
            && !capabilities.ci
            && options.verbosity != Verbosity::Quiet;
        let color = options.display != DisplayMode::Plain
            && capabilities.terminal
            && !capabilities.dumb
            && !capabilities.ci
            && match options.color {
                ColorMode::Auto => !capabilities.no_color && !capabilities.ci,
                ColorMode::Always => true,
                ColorMode::Never => false,
            };
        Self { live, color, verbosity: options.verbosity }
    }
}

#[derive(Clone)]
pub struct Reporter
{
    owner: Arc<Owner>,
}

struct Owner
{
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
    finalization: Mutex<Option<std::result::Result<(), String>>>,
}

struct Shared
{
    state: Mutex<State>,
    wake: Condvar,
    closed: AtomicBool,
    logger: Mutex<Option<Logger>>,
    stop: Mutex<Option<Arc<AtomicBool>>>,
}

struct State
{
    renderer: Renderer,
    task: String,
    run: String,
    phase: String,
    phase_started: Instant,
    progress: Option<Progress>,
    last_training: Option<TrainingProgress>,
    updated: Instant,
    last_plain: Instant,
    stopping: bool,
    contexts: Vec<(String, String)>,
    output_error: Option<String>,
}

impl Reporter
{
    pub fn new(options: OutputOptions) -> Self
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                renderer: Renderer::new(Policy::resolve(options, Capabilities::detect())),
                task: String::new(),
                run: String::new(),
                phase: String::new(),
                phase_started: Instant::now(),
                progress: None,
                last_training: None,
                updated: Instant::now(),
                last_plain: Instant::now(),
                stopping: false,
                contexts: Vec::new(),
                output_error: None,
            }),
            wake: Condvar::new(),
            closed: AtomicBool::new(false),
            logger: Mutex::new(None),
            stop: Mutex::new(None),
        });

        let background = shared.clone();
        let worker = thread::Builder::new().name("bruce-display".into()).spawn(move || display_worker(&background));
        let reporter =
            Self { owner: Arc::new(Owner { shared, worker: Mutex::new(worker.ok()), finalization: Mutex::new(None) }) };

        if lock(&reporter.owner.worker).is_none() {
            lock(&reporter.owner.shared.state).renderer.disable_live();
            reporter.warning("Live updates are unavailable; progress will be printed when work reports an update.");
        }

        reporter
    }

    pub fn attach_log(&self, logger: Logger)
    {
        *lock(&self.owner.shared.logger) = Some(logger);

        let state = lock(&self.owner.shared.state);

        let mut context = format!("Task: {} {}", state.task, state.run);
        for (label, value) in &state.contexts {
            context.push_str(&format!("\n{label}: {value}"));
        }

        if !state.phase.is_empty() {
            context.push_str(&format!("\nPhase: {}", state.phase));
        }

        drop(state);
        self.record(&context);

        self.owner.shared.wake.notify_one();
    }

    pub fn start(&self, task: &str, run: &str)
    {
        let task = clean(task);
        let run = clean(run);

        let mut state = lock(&self.owner.shared.state);

        state.task.clone_from(&task);
        state.run.clone_from(&run);

        let brand = if state.renderer.policy.live { ">^> Bruce  " } else { "" };
        let suffix = if run.is_empty() { String::new() } else { format!("  {run}") };
        let header = format!("{brand}{task}{suffix}");

        state.permanent(&header, Level::Notice);
        if state.renderer.policy.verbosity != Verbosity::Quiet {
            state.permanent("Details: --verbose; display options: --help", Level::Notice);
        }

        drop(state);

        self.record(&format!("Task: {task}{suffix}"));
    }

    pub fn context(&self, label: &str, value: impl AsRef<str>)
    {
        let label = clean(label);
        let value = clean(value.as_ref());
        let message = format!("{label}: {value}");

        let mut state = lock(&self.owner.shared.state);
        if let Some(context) = state.contexts.iter_mut().find(|(key, _)| *key == label) {
            context.1.clone_from(&value);
        } else {
            state.contexts.push((label, value));
        }

        state.permanent(&message, Level::Notice);

        drop(state);

        self.record(&message);
    }

    pub fn phase(&self, phase: &str)
    {
        let mut phase = clean(phase);
        let mut state = lock(&self.owner.shared.state);

        if state.stopping {
            phase = format!("Stopping: {phase}");
        }

        if state.phase == phase {
            return;
        }

        state.phase.clone_from(&phase);
        state.phase_started = Instant::now();
        state.progress = None;
        state.last_plain = Instant::now();

        state.permanent(&format!("Phase: {phase}"), Level::Notice);

        drop(state);

        self.record(&format!("Phase: {phase}"));

        self.owner.shared.wake.notify_one();
    }

    pub fn progress(&self, progress: Progress)
    {
        let no_worker = lock(&self.owner.worker).is_none();
        let mut state = lock(&self.owner.shared.state);

        if self.owner.shared.closed.load(Ordering::Acquire) {
            return;
        }

        if let Progress::Training(training) = &progress {
            state.last_training = Some(*training);
        }

        state.progress = Some(progress);
        state.updated = Instant::now();

        if no_worker {
            state.redraw();
        }
    }

    pub fn sample(&self, sample: Sample)
    {
        if let Some(logger) = lock(&self.owner.shared.logger).as_ref() {
            logger.sample(sample);
        }
    }

    pub fn notice(&self, message: &str) { self.message(message, Level::Notice); }

    pub fn warning(&self, message: &str) { self.message(&format!("Warning: {message}"), Level::Warning); }

    pub fn error(&self, summary: &str, recovery: &str, details: &str)
    {
        let message = format!("Error: {summary}\nHelp: {recovery}");
        self.message(&message, Level::Error);
        if !details.is_empty() {
            self.message(&format!("Cause: {details}"), Level::Verbose);
        }
    }

    pub fn verbose(&self, message: &str) { self.message(message, Level::Verbose); }

    pub fn stopping(&self) { acknowledge_stop(&self.owner.shared); }

    pub fn set_stop_flag(&self, stop: Arc<AtomicBool>)
    {
        *lock(&self.owner.shared.stop) = Some(stop);
        self.owner.shared.wake.notify_one();
    }

    fn message(&self, message: &str, level: Level)
    {
        let message = clean(message);
        lock(&self.owner.shared.state).permanent(&message, level);
        self.record(&message);
    }

    fn record(&self, message: &str)
    {
        let logger = lock(&self.owner.shared.logger).clone();
        if let Some(logger) = logger {
            let _ = logger.record(&clean(message));
        }
    }

    pub fn finish(&self, requested: Outcome) -> Result<()>
    {
        let mut finalization = lock(&self.owner.finalization);

        if let Some(result) = finalization.as_ref() {
            return result.clone().map_err(|message| anyhow!(message));
        }

        if requested == Outcome::Cancelled {
            acknowledge_stop(&self.owner.shared);
        }

        self.owner.shared.closed.store(true, Ordering::Release);
        self.owner.shared.wake.notify_all();

        let mut failure = None;

        let worker = lock(&self.owner.worker).take();
        if let Some(worker) = worker
            && worker.join().is_err()
        {
            failure = Some("The display worker stopped unexpectedly".to_owned());
        }

        let logger = lock(&self.owner.shared.logger).clone();
        if let Some(logger) = logger.as_ref() {
            if lock(&self.owner.shared.state).stopping {
                let _ = logger.record("Stop requested; waited for the safe stopping boundary");
            }

            let snapshot = lock(&self.owner.shared.state).final_snapshot().join("\n");
            if !snapshot.is_empty() {
                let _ = logger.record(&snapshot);
            }

            let _ = logger.record(&format!("Command outcome: {}; draining reporting", requested.label()));
            if let Err(error) = logger.finish() {
                failure = Some(format!("Cannot finish saved diagnostics: {error:#}"));
            }

            show_diagnostics(&self.owner.shared, logger.take_diagnostics());
        }

        let mut state = lock(&self.owner.shared.state);
        if let Some(error) = state.output_error.take() {
            failure.get_or_insert(error);
        }

        state.renderer.clear();
        if let Some(message) = failure.as_ref() {
            state.permanent(
                &format!("Error: {message}\nHelp: Check the log destination and available disk space before retrying."),
                Level::Error,
            );
        }

        let outcome = if failure.is_some() { Outcome::Failed } else { requested };

        let snapshot = state.final_snapshot();
        for row in snapshot {
            state.permanent(&row, Level::Notice);
        }
        let task = state.task.clone();
        let message = format!("{}: {task}", outcome.label());
        let level = if outcome == Outcome::Success { Level::Notice } else { Level::Warning };

        state.permanent(&message, level);

        for (label, value) in state.contexts.clone() {
            if matches!(
                label.to_ascii_lowercase().as_str(),
                "log"
                    | "logs"
                    | "checkpoint"
                    | "checkpoints"
                    | "latest checkpoint"
                    | "output"
                    | "report"
                    | "tensorboard"
            ) {
                state.permanent(&format!("{label}: {value}"), Level::Notice);
            }
        }

        if let Some(error) = state.output_error.take() {
            failure.get_or_insert(error);
        }

        let result = failure.map_or(Ok(()), Err);

        *finalization = Some(result.clone());
        result.map_err(|message| anyhow!(message))
    }
}

impl Drop for Owner
{
    fn drop(&mut self)
    {
        self.shared.closed.store(true, Ordering::Release);
        self.shared.wake.notify_all();

        let worker = lock(&self.worker).take();

        if let Some(worker) = worker {
            let _ = worker.join();
        }

        lock(&self.shared.state).renderer.clear();
    }
}

fn display_worker(shared: &Shared)
{
    loop {
        if shared.closed.load(Ordering::Acquire) {
            break;
        }

        if lock(&shared.stop).as_ref().is_some_and(|flag| flag.load(Ordering::Acquire)) {
            acknowledge_stop(shared);
        }

        let logger = lock(&shared.logger).clone();
        if let Some(logger) = logger {
            show_diagnostics(shared, logger.take_diagnostics());
        }

        let mut state = lock(&shared.state);
        state.redraw();
        if shared.closed.load(Ordering::Acquire) {
            break;
        }

        let _ = shared.wake.wait_timeout(state, REDRAW_INTERVAL);
    }
}

fn acknowledge_stop(shared: &Shared)
{
    let mut state = lock(&shared.state);
    if state.stopping {
        return;
    }
    state.stopping = true;
    state.phase = "Stopping at the next safe boundary".into();
    state.phase_started = Instant::now();

    let message = "Stopping: stop requested; waiting for the current operation and safe checkpoint boundary.";
    state.permanent(message, Level::Warning);
    drop(state);

    shared.wake.notify_one();
}

fn show_diagnostics(shared: &Shared, diagnostics: Vec<String>)
{
    let mut state = lock(&shared.state);
    for diagnostic in diagnostics {
        state.permanent(&format!("Warning: {}", clean(&diagnostic)), Level::Warning);
    }
}

impl State
{
    fn permanent(&mut self, message: &str, level: Level)
    {
        if let Err(error) = self.renderer.permanent(message, level) {
            self.output_error.get_or_insert_with(|| format!("Cannot write status to stderr: {error}"));
        }
    }

    fn snapshot(&self, ticking: bool) -> Vec<String>
    {
        let width = if self.renderer.policy.live { self.renderer.width() } else { usize::MAX };

        let mut rows = Vec::new();

        if !self.phase.is_empty() {
            let elapsed = duration(self.phase_started.elapsed());
            let phase =
                if self.progress.is_none() { format!("{}  elapsed {elapsed}", self.phase) } else { self.phase.clone() };
            rows.extend(wrap(&phase, width));
        }

        if let Some(progress) = self.progress.as_ref() {
            let age = if ticking { self.updated.elapsed() } else { Duration::ZERO };
            rows.extend(progress_rows(progress, width, age));
        }

        rows
    }

    fn final_snapshot(&self) -> Vec<String>
    {
        if let Some(training) = &self.last_training {
            let mut rows = progress_rows(&Progress::Training(*training), usize::MAX, Duration::ZERO);
            rows.push(format!("Session batches completed: {}", training.session_batches));
            rows
        } else {
            self.snapshot(false)
        }
    }

    fn redraw(&mut self)
    {
        if self.renderer.policy.verbosity == Verbosity::Quiet || self.phase.is_empty() {
            return;
        }

        if self.renderer.policy.live {
            let rows = self.snapshot(true);
            self.renderer.draw(&rows);
        } else if self.last_plain.elapsed() >= PLAIN_INTERVAL {
            for row in self.snapshot(true) {
                self.permanent(&row, Level::Notice);
            }

            self.last_plain = Instant::now();
        }
    }
}

#[derive(Clone, Copy)]
enum Level
{
    Notice,
    Verbose,
    Warning,
    Error,
}

struct Renderer
{
    policy: Policy,
    multi: MultiProgress,
    rows: ProgressBar,
    #[cfg(test)]
    capture: Option<Arc<Mutex<Vec<u8>>>>,
}

impl Renderer
{
    fn new(policy: Policy) -> Self
    {
        let target = if policy.live { ProgressDrawTarget::stderr_with_hz(10) } else { ProgressDrawTarget::hidden() };
        let multi = MultiProgress::with_draw_target(target);

        let rows = multi.add(ProgressBar::new_spinner());
        rows.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .expect("Static progress template is valid")
                .tick_chars("|/-\\ "),
        );

        Self {
            policy,
            multi,
            rows,
            #[cfg(test)]
            capture: None,
        }
    }

    fn width(&self) -> usize { usize::from(Term::stderr().size().1).saturating_sub(2).max(2) }

    fn draw(&self, rows: &[String])
    {
        self.rows.set_message(rows.join("\n"));
        self.rows.tick();
    }

    fn permanent(&self, message: &str, level: Level) -> io::Result<()>
    {
        if self.policy.verbosity == Verbosity::Quiet && matches!(level, Level::Notice | Level::Verbose)
            || self.policy.verbosity != Verbosity::Verbose && matches!(level, Level::Verbose)
        {
            return Ok(());
        }

        let message = clean(message);
        let text = if self.policy.live { wrap(&message, self.width()).join("\n") } else { message };

        let style = match level {
            Level::Notice => Style::new(),
            Level::Verbose => Style::new(),
            Level::Warning => Style::new().yellow().bold(),
            Level::Error => Style::new().red().bold(),
        }
        .for_stderr()
        .force_styling(self.policy.color);

        let text = style.apply_to(text).to_string();

        if self.policy.live && !self.multi.is_hidden() {
            self.multi.println(text)
        } else {
            #[cfg(test)]
            if let Some(capture) = &self.capture {
                return writeln!(lock(capture), "{text}");
            }
            writeln!(io::stderr().lock(), "{text}")
        }
    }

    fn clear(&self) { self.rows.finish_and_clear(); }

    fn disable_live(&mut self)
    {
        self.rows.finish_and_clear();
        self.multi.set_draw_target(ProgressDrawTarget::hidden());
        self.policy.live = false;
    }
}

fn progress_rows(progress: &Progress, width: usize, age: Duration) -> Vec<String>
{
    match progress {
        Progress::Work { completed, total, unit } => {
            let count = match total {
                Some(total) if *total > 0 && completed <= total => {
                    format!("{completed}/{total} {unit} ({:.1}%)", *completed as f64 / *total as f64 * 100.0)
                },
                _ => format!("{completed} {unit}"),
            };

            wrap(&count, width)
        },

        Progress::Training(progress) => {
            let mut rows = pack(
                &[
                    format!("Superbatch {}/{}", progress.superbatch, progress.final_superbatch),
                    format!("Batches {}/{}", progress.batch, progress.batches_per_superbatch),
                    format!("Global step {}", progress.global_batch),
                ],
                width,
            );

            let mut metrics = Vec::new();
            if let Some(loss) = progress.loss {
                metrics.push(format!("Loss ({} batches) {loss:.6}", progress.loss_batches));
            }
            metrics.push(format!("LR {:.3e}", progress.learning_rate));

            rows.extend(pack(&metrics, width));

            let mut time = vec![format!("Elapsed {}", duration(progress.elapsed.saturating_add(age)))];
            if let Some(rate) = progress.positions_per_second.filter(|rate| rate.is_finite() && *rate > 0.0) {
                time.push(format!("{:.2} Mpos/s", rate / 1e6));
            }

            if let Some(eta) = progress.superbatch_eta {
                time.push(format!("SB ETA {}", duration(eta)));
            }

            if let Some(eta) = progress.run_eta {
                time.push(format!("Run ETA {}", duration(eta)));
            }

            rows.extend(pack(&time, width));
            rows
        },
    }
}

fn pack(fields: &[String], width: usize) -> Vec<String>
{
    let mut rows = Vec::new();
    let mut row = String::new();

    for field in fields {
        if !row.is_empty()
            && measure_text_width(&row).saturating_add(2).saturating_add(measure_text_width(field)) > width
        {
            rows.push(std::mem::take(&mut row));
        }

        if measure_text_width(field) > width {
            rows.extend(wrap(field, width));
        } else {
            if !row.is_empty() {
                row.push_str("  ");
            }
            row.push_str(field);
        }
    }

    if !row.is_empty() {
        rows.push(row);
    }

    rows
}

fn wrap(text: &str, width: usize) -> Vec<String>
{
    let width = width.max(2);

    let mut lines = Vec::new();
    for source in text.split('\n') {
        let mut line = String::new();

        for ch in source.chars() {
            let mut candidate = line.clone();

            candidate.push(ch);

            if measure_text_width(&candidate) > width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }

            line.push(ch);
        }

        lines.push(line);
    }

    lines
}

fn clean(text: &str) -> String
{
    strip_ansi_codes(text)
        .chars()
        .filter(|ch| !ch.is_control() || *ch == '\n' || *ch == '\t')
        .map(|ch| if ch == '\t' { ' ' } else { ch })
        .collect()
}

fn duration(value: Duration) -> String
{
    let seconds = value.as_secs();
    format!("{:02}:{:02}:{:02}", seconds / 3600, seconds / 60 % 60, seconds % 60)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

#[cfg(test)]
mod tests
{
    use super::*;

    fn terminal() -> Capabilities { Capabilities { terminal: true, dumb: false, ci: false, no_color: false } }

    #[test]
    fn display_and_color_are_independent_of_verbosity()
    {
        let mut capabilities = terminal();
        capabilities.no_color = true;

        let policy = Policy::resolve(OutputOptions::default(), capabilities);
        assert!(policy.live);
        assert!(!policy.color);

        let policy = Policy::resolve(OutputOptions { color: ColorMode::Always, ..Default::default() }, capabilities);
        assert!(policy.color);

        for verbosity in [Verbosity::Normal, Verbosity::Verbose] {
            let policy = Policy::resolve(
                OutputOptions { display: DisplayMode::Plain, verbosity, color: ColorMode::Always },
                terminal(),
            );

            assert!(!policy.live);
            assert!(!policy.color);
        }

        assert!(!Policy::resolve(OutputOptions { verbosity: Verbosity::Quiet, ..Default::default() }, terminal()).live);
    }

    #[test]
    fn redirect_ci_and_dumb_fall_back_even_for_requested_live()
    {
        for capabilities in [
            Capabilities { terminal: false, ..terminal() },
            Capabilities { ci: true, ..terminal() },
            Capabilities { dumb: true, ..terminal() },
        ] {
            let policy =
                Policy::resolve(OutputOptions { display: DisplayMode::Live, ..Default::default() }, capabilities);
            assert!(!policy.live);
            assert!(!policy.color);
        }
    }

    #[test]
    fn resize_preserves_every_essential_training_field()
    {
        let progress = Progress::Training(TrainingProgress {
            superbatch: 5,
            final_superbatch: 20,
            batch: 480,
            batches_per_superbatch: 1000,
            global_batch: 4480,
            session_batches: 480,
            loss: Some(0.03124),
            loss_batches: 100,
            learning_rate: 0.002,
            elapsed: Duration::from_secs(136),
            positions_per_second: Some(1_250_000.0),
            superbatch_eta: Some(Duration::from_secs(147)),
            run_eta: Some(Duration::from_secs(4395)),
        });

        for width in [120, 80, 40, 24, 12, 80] {
            let rows = progress_rows(&progress, width, Duration::ZERO);
            assert!(rows.iter().all(|row| measure_text_width(row) <= width), "{rows:?}");

            let joined = rows.join("");
            for essential in
                ["5/20", "480/1000", "4480", "0.031240", "2.000e-3", "00:02:16", "1.25 Mpos/s", "00:02:27", "01:13:15"]
            {
                assert!(joined.contains(essential), "{essential} missing at {width}: {joined}");
            }
        }
    }

    #[test]
    fn unicode_long_paths_and_ansi_are_preserved_as_safe_text()
    {
        let path = "/models/棋譜/cafe\u{301}/network-with-a-long-name/checkpoint";
        let lines = wrap(path, 13);

        assert_eq!(lines.concat(), path);
        assert!(lines.iter().all(|line| measure_text_width(line) <= 13));
        assert_eq!(clean("\x1b[31mError\x1b[0m\r\nfile\x07"), "Error\nfile");
    }

    #[test]
    fn unknown_or_exceeded_total_never_invents_percent()
    {
        for total in [None, Some(0), Some(1)] {
            let rows = progress_rows(&Progress::Work { completed: 2, total, unit: "positions" }, 80, Duration::ZERO);
            assert_eq!(rows, ["2 positions"]);
        }
    }

    fn captured_reporter(verbosity: Verbosity) -> (Reporter, Arc<Mutex<Vec<u8>>>)
    {
        let reporter = Reporter::new(OutputOptions { display: DisplayMode::Plain, verbosity, ..Default::default() });
        let capture = Arc::new(Mutex::new(Vec::new()));
        lock(&reporter.owner.shared.state).renderer.capture = Some(capture.clone());

        (reporter, capture)
    }

    fn captured(capture: &Mutex<Vec<u8>>) -> String { String::from_utf8(lock(capture).clone()).unwrap() }

    #[test]
    fn hidden_progress_does_not_hide_permanent_warnings()
    {
        let mut renderer = Renderer::new(Policy { live: true, color: false, verbosity: Verbosity::Normal });
        renderer.multi.set_draw_target(ProgressDrawTarget::hidden());

        let capture = Arc::new(Mutex::new(Vec::new()));
        renderer.capture = Some(capture.clone());
        renderer.permanent("Warning: check available disk space", Level::Warning).unwrap();

        assert_eq!(captured(&capture), "Warning: check available disk space\n");
    }

    #[test]
    fn quiet_preserves_warnings_errors_and_cancellation()
    {
        let (reporter, capture) = captured_reporter(Verbosity::Quiet);

        reporter.start("Train", "run");
        reporter.notice("routine");
        reporter.verbose("debug");
        reporter.warning("disk low");
        reporter.error("checkpoint failed", "Choose a writable destination", "underlying cause");
        reporter.stopping();
        reporter.finish(Outcome::Cancelled).unwrap();

        let output = captured(&capture);

        assert!(!output.contains("routine") && !output.contains("debug") && !output.contains("underlying cause"));
        assert!(output.contains("Warning: disk low"));
        assert!(output.contains("Error: checkpoint failed"));
        assert!(output.contains("Help: Choose a writable destination"));
        assert!(output.contains("Stopping:") && output.contains("Cancelled:"));
        assert!(!output.contains('\x1b'));
    }

    #[test]
    fn plain_progress_is_periodic_and_finalization_is_idempotent()
    {
        let (reporter, capture) = captured_reporter(Verbosity::Normal);

        reporter.start("Analyze", "dataset");
        reporter.phase("Reading positions");
        reporter.progress(Progress::Work { completed: 50, total: None, unit: "positions" });

        lock(&capture).clear();
        {
            let mut state = lock(&reporter.owner.shared.state);
            state.last_plain = Instant::now() - PLAIN_INTERVAL;

            state.redraw();
            state.redraw();
        }

        assert_eq!(captured(&capture).matches("50 positions").count(), 1);

        reporter.finish(Outcome::Success).unwrap();

        let output = captured(&capture);

        reporter.finish(Outcome::Success).unwrap();

        assert_eq!(captured(&capture), output);
        assert!(output.contains("Complete: Analyze"));
    }

    #[test]
    fn cancellation_is_acknowledged_without_an_observer_event()
    {
        let (reporter, capture) = captured_reporter(Verbosity::Normal);

        reporter.start("Train", "run");
        reporter.phase("Compiling network");

        let stop = Arc::new(AtomicBool::new(false));

        reporter.set_stop_flag(stop.clone());
        stop.store(true, Ordering::Release);

        let deadline = Instant::now() + Duration::from_secs(2);

        while !captured(&capture).contains("Stopping:") && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(captured(&capture).contains("Stopping:"));

        reporter.phase("Saving checkpoint");
        assert_eq!(lock(&reporter.owner.shared.state).phase, "Stopping: Saving checkpoint");

        reporter.finish(Outcome::Cancelled).unwrap();
        assert_eq!(captured(&capture).matches("stop requested;").count(), 1);
    }

    #[derive(Debug)]
    struct TestTerminal(Arc<Mutex<Vec<u8>>>);

    impl indicatif::TermLike for TestTerminal
    {
        fn width(&self) -> u16 { 80 }
        fn move_cursor_up(&self, _: usize) -> io::Result<()> { Ok(()) }
        fn move_cursor_down(&self, _: usize) -> io::Result<()> { Ok(()) }
        fn move_cursor_right(&self, _: usize) -> io::Result<()> { Ok(()) }
        fn move_cursor_left(&self, _: usize) -> io::Result<()> { Ok(()) }
        fn write_line(&self, text: &str) -> io::Result<()> { writeln!(lock(&self.0), "{text}") }
        fn write_str(&self, text: &str) -> io::Result<()> { write!(lock(&self.0), "{text}") }
        fn clear_line(&self) -> io::Result<()> { Ok(()) }
        fn flush(&self) -> io::Result<()> { Ok(()) }
    }

    #[test]
    fn warning_is_permanent_and_live_rows_continue()
    {
        let renderer = Renderer::new(Policy { live: true, color: false, verbosity: Verbosity::Normal });
        let capture = Arc::new(Mutex::new(Vec::new()));

        renderer.multi.set_draw_target(ProgressDrawTarget::term_like(Box::new(TestTerminal(capture.clone()))));
        renderer.draw(&["Training 1/10 batches".into()]);
        renderer.permanent("Warning: TensorBoard stopped; text logging continues", Level::Warning).unwrap();
        renderer.draw(&["Training 2/10 batches".into()]);

        let output = captured(&capture);
        assert_eq!(output.matches("Warning: TensorBoard stopped; text logging continues").count(), 1);
        assert!(output.contains("Training 1/10 batches") && output.contains("Training 2/10 batches"));

        renderer.clear();
    }

    struct FlushFailure;

    impl Write for FlushFailure
    {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("simulated disk flush failure")) }
    }

    #[test]
    fn failed_log_finalization_never_announces_success()
    {
        let (reporter, capture) = captured_reporter(Verbosity::Normal);

        reporter.start("Train", "run");
        reporter.attach_log(Logger::for_test(Box::new(FlushFailure)));
        assert!(reporter.finish(Outcome::Success).is_err());

        let output = captured(&capture);
        assert!(output.contains("Failed: Train"));
        assert!(output.contains("simulated disk flush failure"));
        assert!(!output.contains("Complete: Train"));
    }

    #[test]
    fn immediate_cancellation_acknowledges_before_final_outcome()
    {
        let (reporter, capture) = captured_reporter(Verbosity::Normal);

        reporter.start("Convert", "dataset");
        reporter.finish(Outcome::Cancelled).unwrap();

        let output = captured(&capture);
        assert!(output.find("Stopping:").unwrap() < output.find("Cancelled:").unwrap());
    }
}
