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
use console::{Style, Term};

use crate::logging::{Logger, Sample};
pub use crate::report::TrainingProgress;

mod output;
mod render;
mod terminal;
pub use output::{CommandOutput, OutputSection};

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

#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("The operation was cancelled") }
}

impl std::error::Error for Cancelled {}

#[derive(Clone)]
pub struct Reporter(Arc<Session>);

struct Session
{
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<()>>>,
    result: Mutex<Option<std::result::Result<(), String>>>,
}

struct Shared
{
    state: Mutex<State>,
    wake: Condvar,
    closed: AtomicBool,
}

struct State
{
    view: View,
    output: Output,
    messages: Vec<(String, Level)>,
    logger: Option<Logger>,
    stop: Option<Arc<AtomicBool>>,
    plain_at: Instant,
    write_error: Option<String>,
}

#[derive(Clone)]
struct View
{
    task: String,
    run: String,
    phase: String,
    started: Instant,
    phase_started: Instant,
    updated: Instant,
    progress: Option<Progress>,
    training: Option<TrainingProgress>,
    training_loss: Option<f64>,
    output: Option<CommandOutput>,
    failure: Option<(String, String)>,
    contexts: Vec<(String, String)>,
    stopping: bool,
    color: bool,
}

#[derive(Clone, Copy)]
enum Level
{
    Notice,
    Detail,
    Warning,
    Error,
}

struct Output
{
    live: bool,
    color: bool,
    verbosity: Verbosity,
}

impl Reporter
{
    pub fn new(options: OutputOptions) -> Self
    {
        let output = Output::new(options);

        let now = Instant::now();
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                view: View {
                    task: String::new(),
                    run: String::new(),
                    phase: String::new(),
                    started: now,
                    phase_started: now,
                    updated: now,
                    progress: None,
                    training: None,
                    training_loss: None,
                    output: None,
                    failure: None,
                    contexts: Vec::new(),
                    stopping: false,
                    color: output.color,
                },
                output,
                messages: Vec::new(),
                logger: None,
                stop: None,
                plain_at: now,
                write_error: None,
            }),
            wake: Condvar::new(),
            closed: AtomicBool::new(false),
        });

        let display = shared.clone();
        let worker = thread::Builder::new().name("bruce-display".into()).spawn(move || display.run()).ok();
        let available = worker.is_some();

        let reporter = Self(Arc::new(Session { shared, worker: Mutex::new(worker), result: Mutex::new(None) }));

        if !available {
            let mut state = reporter.state();
            state.output.live = false;
            drop(state);

            reporter.warning("Live updates are unavailable; progress will be printed when work reports an update.");
        }

        reporter
    }

    fn state(&self) -> MutexGuard<'_, State> { lock(&self.0.shared.state) }

    fn logger(&self) -> Option<Logger> { self.state().logger.clone() }

    fn record(&self, text: &str)
    {
        if let Some(logger) = self.logger() {
            let _ = logger.record(&clean(text));
        }
    }

    fn message(&self, text: &str, level: Level)
    {
        let text = clean(text);
        self.state().write(&text, level);
        self.record(&text);
    }

    pub fn attach_log(&self, logger: Logger)
    {
        let mut state = self.state();
        state.logger = Some(logger);

        let view = &state.view;

        let mut initial = format!("Task: {} {}", view.task, view.run);
        for (label, value) in &view.contexts {
            initial.push_str(&format!("\n{label}: {value}"));
        }

        if !view.phase.is_empty() {
            initial.push_str(&format!("\nPhase: {}", view.phase));
        }

        drop(state);
        self.record(&initial);
        self.0.shared.wake.notify_one();
    }

    pub fn start(&self, task: &str, run: &str)
    {
        let task = clean(task);
        let run = clean(run);

        let title = if run.is_empty() { task.clone() } else { format!("{task}  {run}") };

        let mut state = self.state();
        state.view.task = task;
        state.view.run = run;
        state.view.started = Instant::now();

        if !state.output.live {
            state.write(&title, Level::Notice);
        }

        drop(state);
        self.record(&format!("Task: {title}"));
    }

    pub fn context(&self, label: &str, value: impl AsRef<str>)
    {
        let label = clean(label);
        let value = clean(value.as_ref());

        let message = format!("{label}: {value}");

        let mut state = self.state();
        if let Some((_, previous)) = state.view.contexts.iter_mut().find(|(key, _)| key == &label) {
            *previous = value;
        } else {
            state.view.contexts.push((label, value));
        }

        if !state.output.live {
            state.write(&message, Level::Notice);
        }

        drop(state);
        self.record(&message);
    }

    pub fn phase(&self, phase: &str)
    {
        let mut phase = clean(phase);

        let mut state = self.state();
        if state.view.stopping {
            phase = format!("Stopping: {phase}");
        }

        if state.view.phase == phase {
            return;
        }

        state.view.phase = phase.clone();
        state.view.phase_started = Instant::now();
        state.view.progress = None;
        state.plain_at = Instant::now();

        if !state.output.live {
            state.write(&format!("Phase: {phase}"), Level::Notice);
        }

        drop(state);
        self.record(&format!("Phase: {phase}"));
        self.0.shared.wake.notify_one();
    }

    pub fn progress(&self, progress: Progress)
    {
        if self.0.shared.closed.load(Ordering::Acquire) {
            return;
        }

        let fallback = lock(&self.0.worker).is_none();

        let mut state = self.state();
        if let Progress::Training(training) = &progress {
            state.view.training = Some(*training);
        }

        state.view.progress = Some(progress);
        state.view.updated = Instant::now();

        if fallback {
            state.redraw();
        }
    }

    pub fn set_output(&self, output: CommandOutput) { self.state().view.output = Some(output); }

    pub fn set_training_loss(&self, loss: f64)
    {
        self.state().view.training_loss = Some(loss);
        self.record(&format!("Session mean loss: {loss:.6}"));
    }

    pub fn sample(&self, sample: Sample)
    {
        if let Some(logger) = self.logger() {
            logger.sample(sample);
        }
    }

    pub fn notice(&self, message: &str) { self.message(message, Level::Notice); }

    pub fn warning(&self, message: &str) { self.message(&format!("Warning: {message}"), Level::Warning); }

    pub fn verbose(&self, message: &str) { self.message(message, Level::Detail); }

    pub fn error(&self, summary: &str, recovery: &str, details: &str)
    {
        let summary = clean(summary);
        let recovery = clean(recovery);

        let message = format!("Error: {summary}\nHelp: {recovery}");

        let mut state = self.state();
        state.view.failure = Some((summary, recovery));

        if !state.output.live {
            state.write(&message, Level::Error);
        }

        drop(state);
        self.record(&message);

        if !details.is_empty() {
            self.verbose(&format!("Cause: {details}"));
        }
    }

    pub fn stopping(&self)
    {
        self.state().acknowledge_stop();
        self.0.shared.wake.notify_one();
    }

    pub fn set_stop_flag(&self, stop: Arc<AtomicBool>)
    {
        self.state().stop = Some(stop);
        self.0.shared.wake.notify_one();
    }

    pub fn finish(&self, requested: Outcome) -> Result<()>
    {
        let mut saved = lock(&self.0.result);
        if let Some(result) = saved.as_ref() {
            return result.clone().map_err(|error| anyhow!(error));
        }

        if requested == Outcome::Cancelled {
            self.stopping();
        }

        self.0.shared.closed.store(true, Ordering::Release);
        self.0.shared.wake.notify_all();

        let mut failure = None;
        if let Some(worker) = lock(&self.0.worker).take()
            && worker.join().is_err()
        {
            failure = Some("The display worker stopped unexpectedly".to_owned());
        }

        if let Some(logger) = self.logger() {
            let state = self.state();
            let snapshot = render::plain(&state.view, true).join("\n");
            let stopping = state.view.stopping;
            drop(state);

            if stopping {
                let _ = logger.record("Stop requested; waited for the safe stopping boundary");
            }

            if !snapshot.is_empty() {
                let _ = logger.record(&snapshot);
            }

            let _ = logger.record(&format!("Command outcome: {}; draining reporting", requested.label()));
            if let Err(error) = logger.finish() {
                failure = Some(format!("Cannot finish saved diagnostics: {error:#}"));
            }

            self.state().diagnostics(logger.take_diagnostics());
        }

        let mut state = self.state();

        if let Some(error) = state.write_error.take() {
            failure.get_or_insert(error);
        }

        state.flush_messages();

        let rich = state.output.live && io::stdout().is_terminal();

        if requested == Outcome::Success
            && failure.is_none()
            && !rich
            && let Some(output) = &state.view.output
        {
            let mut stdout = io::stdout().lock();
            if let Err(error) = writeln!(stdout, "{}", output.plain).and_then(|()| stdout.flush()) {
                failure = Some(format!("Cannot write the command result: {error}"));
            }
        }

        if let Some(message) = &failure {
            let recovery = "Check the output destination and available disk space before retrying.";
            state.view.failure = Some((message.clone(), recovery.into()));
            if !state.output.live {
                state.write(&format!("Error: {message}\nHelp: {recovery}"), Level::Error);
            }
        }

        let outcome = if failure.is_some() { Outcome::Failed } else { requested };

        if state.output.live {
            let rows = render::completed(&state.view, outcome, width(), rich);
            let mut stderr = io::stderr().lock();
            if let Err(error) = writeln!(stderr, "{}", rows.join("\n")).and_then(|()| stderr.flush()) {
                failure.get_or_insert_with(|| format!("Cannot write status to stderr: {error}"));
            }
        } else {
            state.finish_plain(outcome);
        }

        if let Some(error) = state.write_error.take() {
            failure.get_or_insert(error);
        }

        let result = failure.map_or(Ok(()), Err);
        *saved = Some(result.clone());
        result.map_err(|error| anyhow!(error))
    }
}

impl Drop for Session
{
    fn drop(&mut self)
    {
        self.shared.closed.store(true, Ordering::Release);
        self.shared.wake.notify_all();

        if let Some(worker) = lock(&self.worker).take() {
            let _ = worker.join();
        }

        lock(&self.shared.state).flush_messages();
    }
}

impl Shared
{
    fn run(&self)
    {
        let live = lock(&self.state).output.live;
        let mut terminal = if live {
            match terminal::Terminal::new() {
                Ok(terminal) => Some(terminal),
                Err(error) => {
                    let mut state = lock(&self.state);
                    state.output.live = false;
                    state.flush_messages();
                    state.write(&format!("Warning: Live display unavailable: {error}"), Level::Warning);
                    None
                },
            }
        } else {
            None
        };

        while !self.closed.load(Ordering::Acquire) {
            let frame_started = Instant::now();

            let logger = lock(&self.state).logger.clone();
            let diagnostics = logger.map(|logger| logger.take_diagnostics()).unwrap_or_default();

            let mut state = lock(&self.state);
            state.diagnostics(diagnostics);

            if state.stop.as_ref().is_some_and(|flag| flag.load(Ordering::Acquire)) {
                state.acknowledge_stop();
            }

            let view = state.view.clone();
            let messages = std::mem::take(&mut state.messages);

            state.redraw();
            drop(state);

            if let Some(display) = terminal.as_mut() {
                let result = self.display(display, &view, messages);
                if let Err(error) = result {
                    let cleanup = display.close();
                    terminal = None;

                    let mut state = lock(&self.state);
                    state.output.live = false;
                    state.write_error.get_or_insert_with(|| format!("Cannot update the live display: {error}"));

                    if let Err(error) = cleanup {
                        state.write(&format!("Warning: Cannot restore the terminal: {error}"), Level::Warning);
                    }

                    state.write("Warning: Live display stopped; continuing with plain output.", Level::Warning);
                }
            }

            let mut state = lock(&self.state);
            while !self.closed.load(Ordering::Acquire) {
                let remaining = Duration::from_millis(100).saturating_sub(frame_started.elapsed());
                if remaining.is_zero() {
                    break;
                }

                state = match self.wake.wait_timeout(state, remaining) {
                    Ok((state, _)) => state,
                    Err(error) => error.into_inner().0,
                };
            }
        }

        if let Some(mut terminal) = terminal {
            let mut state = lock(&self.state);
            let messages = std::mem::take(&mut state.messages);
            let color = state.output.color;
            drop(state);

            for (text, level) in messages {
                if let Err(error) = terminal.write(&text, level, color) {
                    lock(&self.state).write_error.get_or_insert_with(|| format!("Cannot write status: {error}"));
                }
            }

            if let Err(error) = terminal.close() {
                lock(&self.state).write_error.get_or_insert_with(|| format!("Cannot restore the terminal: {error}"));
            }
        }
    }

    fn display(&self, terminal: &mut terminal::Terminal, view: &View, messages: Vec<(String, Level)>)
    -> io::Result<()>
    {
        for (text, level) in messages {
            terminal.write(&text, level, view.color)?;
        }

        if lock(&self.state).stop.is_some() {
            terminal.enable_keyboard()?;
        }

        if terminal.poll_stop()? {
            let mut state = lock(&self.state);

            if let Some(stop) = &state.stop {
                stop.store(true, Ordering::Release);
                state.acknowledge_stop();
            }
        }

        if !view.phase.is_empty() {
            terminal.draw(view)?;
        }

        Ok(())
    }
}

impl State
{
    fn write(&mut self, text: &str, level: Level)
    {
        if !self.output.visible(level) {
            return;
        }

        if self.output.live {
            self.messages.push((text.to_owned(), level));
        } else if let Err(error) = self.output.write(text, level) {
            self.write_error.get_or_insert_with(|| format!("Cannot write status to stderr: {error}"));
        }
    }

    fn flush_messages(&mut self)
    {
        for (text, level) in std::mem::take(&mut self.messages) {
            if let Err(error) = self.output.write(&text, level) {
                self.write_error.get_or_insert_with(|| format!("Cannot write status to stderr: {error}"));
            }
        }
    }

    fn diagnostics(&mut self, messages: Vec<String>)
    {
        for message in messages {
            self.write(&format!("Warning: {}", clean(&message)), Level::Warning);
        }
    }

    fn acknowledge_stop(&mut self)
    {
        if self.view.stopping {
            return;
        }

        self.view.stopping = true;
        self.view.phase = "Stopping at the next safe boundary".into();
        self.view.phase_started = Instant::now();

        if !self.output.live {
            self.write(
                "Stopping: stop requested; waiting for the current operation and safe checkpoint boundary.",
                Level::Warning,
            );
        }
    }

    fn redraw(&mut self)
    {
        if self.output.verbosity == Verbosity::Quiet || self.view.phase.is_empty() {
            return;
        }

        if !self.output.live && self.plain_at.elapsed() >= Duration::from_secs(5) {
            for row in render::plain(&self.view, false) {
                self.write(&row, Level::Notice);
            }

            self.plain_at = Instant::now();
        }
    }

    fn finish_plain(&mut self, outcome: Outcome)
    {
        if self.view.output.is_none() {
            for row in render::plain(&self.view, true) {
                self.write(&row, Level::Notice);
            }
        }

        if let Some(loss) = self.view.training_loss {
            self.write(&format!("Run loss: {loss:.6}"), Level::Notice);
        }

        let level = if outcome == Outcome::Success { Level::Notice } else { Level::Warning };
        self.write(&format!("{}: {}", outcome.label(), self.view.task), level);

        for (label, value) in self.view.contexts.clone() {
            if matches!(
                label.as_str(),
                "Log" | "Logs" | "Checkpoint" | "Checkpoints" | "Latest checkpoint" | "Report" | "TensorBoard"
            ) || (label == "Output" && outcome == Outcome::Success)
            {
                self.write(&format!("{label}: {value}"), Level::Notice);
            }
        }
    }
}

impl Output
{
    fn new(options: OutputOptions) -> Self
    {
        let interactive = io::stderr().is_terminal()
            && io::stdout().is_terminal()
            && !std::env::var("TERM").is_ok_and(|term| term == "dumb");

        let live = interactive
            && options.verbosity != Verbosity::Quiet
            && match options.display {
                DisplayMode::Auto => std::env::var_os("CI").is_none(),
                DisplayMode::Live => true,
                DisplayMode::Plain => false,
            };

        let color = match options.color {
            ColorMode::Always => true,
            ColorMode::Never => false,
            ColorMode::Auto => interactive && !std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()),
        };

        Self { live, color, verbosity: options.verbosity }
    }

    fn visible(&self, level: Level) -> bool
    {
        match level {
            Level::Notice => self.verbosity != Verbosity::Quiet,
            Level::Detail => self.verbosity == Verbosity::Verbose,
            Level::Warning | Level::Error => true,
        }
    }

    fn write(&self, text: &str, level: Level) -> io::Result<()>
    {
        if !self.visible(level) {
            return Ok(());
        }

        let style = match level {
            Level::Warning => Style::new().yellow().bold(),
            Level::Error => Style::new().red().bold(),
            _ => Style::new(),
        };

        let text = style.for_stderr().force_styling(self.color).apply_to(clean(text)).to_string();
        writeln!(io::stderr().lock(), "{text}")
    }
}

fn width() -> usize { usize::from(Term::stderr().size().1).saturating_sub(2).clamp(2, 120) }

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|error| error.into_inner()) }

fn clean(text: &str) -> String
{
    console::strip_ansi_codes(text)
        .chars()
        .filter(|ch| !ch.is_control() || matches!(ch, '\n' | '\t'))
        .map(|ch| if ch == '\t' { ' ' } else { ch })
        .collect()
}
