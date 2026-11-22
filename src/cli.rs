use std::{
    env,
    io::{self, IsTerminal, Stderr, Write},
    iter, mem,
    path::{self, Path},
    process::{self, ExitCode, Stdio},
    sync::{
        Arc,
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use anstream::{AutoStream, ColorChoice};
use anyhow::Result;
use clap::Parser;

use crate::{
    analyze::{self, check, check::CheckMode},
    args::{Arguments, Command, TrainArguments},
    config::Config,
    control::{self, Control, Outcome, Stop},
    data::{Source, pipeline::Sampling},
    events::{Ending, Level, Task},
    render::{self, HEADER, WIDTH},
    screen::Screen,
    telemetry::{Hub, Live, Setup},
    text,
    train::{self, bench},
};

const TASK_STACK: usize = 16 << 20;
const TASK_POLL: Duration = Duration::from_millis(10);
const ABORT_DEADLINE: Duration = Duration::from_secs(5);
const FRAME: Duration = Duration::from_millis(100);
const STRAY_WAIT: Duration = Duration::from_secs(1);
const TRAINING_LINE_EVERY: Duration = Duration::from_secs(60);
const SCAN_LINE_EVERY: Duration = Duration::from_secs(30);
const FORCED: &str = "Exiting before the task stopped";
const UNEXPECTED: &str = "Operation failed unexpectedly";

type Work = Box<dyn FnOnce(&Control, &Hub) -> Result<Ending> + Send>;

pub fn main() -> ExitCode
{
    let Arguments { quiet, verbose, color, log_dir, no_log, command } = Arguments::parse();
    let task = match &command {
        Command::Train(train) if train.check.is_some() => Task::Check,
        Command::Train(train) if train.benchmark.is_some() => Task::Benchmark,
        Command::Train(_) => Task::Train,
        Command::Analyze(_) => Task::Analyze,
    };
    let viewer = !matches!(&command, Command::Train(train) if train.no_viewer) && io::stderr().is_terminal();
    let color = match color {
        clap::ColorChoice::Auto => ColorChoice::Auto,
        clap::ColorChoice::Always => ColorChoice::Always,
        clap::ColorChoice::Never => ColorChoice::Never,
    };
    let mut out = AutoStream::new(io::stderr(), color);
    let colored = out.current_choice() != ColorChoice::Never;

    let (config, subject, facts, work) = match prepare(command) {
        Ok(prepared) => prepared,
        Err(error) => return failed(&mut out, task, lines(&error), Some(advice(task)), colored),
    };
    let run = config
        .as_ref()
        .filter(|_| task == Task::Train)
        .map(|config| (config.run.name.clone(), config.training.metrics.clone().unwrap_or_else(|| "runs".into())));
    let named = run.as_ref().map_or_else(|| "bruce.log".to_owned(), |(name, _)| format!("{name}.log"));
    let log = (!no_log).then(|| log_dir.join(named));
    let metrics = run.as_ref().map(|(name, directory)| directory.join(name).join("metrics.jsonl"));

    control::hook();
    let control = Arc::new(Control::default());
    if let Err(error) = control::install(&control) {
        return failed(&mut out, task, format!("Could not install Ctrl+C handling: {error}"), None, colored);
    }

    let log_every = config.as_ref().map_or(1, |config| config.training.log_every);
    let hub = Hub::new(Setup { task, subject, facts, log, metrics }, log_every);
    let invocation: Vec<String> = env::args_os().skip(1).map(|argument| argument.to_string_lossy().into()).collect();
    let version = env!("CARGO_PKG_VERSION");
    hub.note(Level::Detail, format!("session start: bruce {version}, pid {}, {}", process::id(), invocation.join(" ")));
    if let Some(config) = &config {
        hub.note(Level::Detail, format!("config {}", config.checkpoint_value()));
    }
    if let Some((name, directory)) = run.filter(|_| viewer) {
        launch(&name, &path::absolute(&directory).unwrap_or(directory), &hub);
    }

    let dumb = env::var("TERM").is_ok_and(|term| term == "dumb");
    let live = !quiet && io::stderr().is_terminal() && !dumb && env::var_os("CI").is_none();
    let (sender, stray) = mpsc::channel();
    let unavailable = |error: &io::Error| hub.note(Level::Info, format!("live display unavailable: {error}"));
    let mut screen = if live { Screen::open(sender).inspect_err(unavailable).ok() } else { None };

    let (joined, crashed, now) = thread::scope(|scope| {
        let spawned = thread::Builder::new().name("bruce-task".into()).stack_size(TASK_STACK);
        let worker = match spawned.spawn_scoped(scope, || work(&control, &hub)) {
            Ok(worker) => worker,
            Err(error) => {
                leave(&hub, &mut screen, &stray);
                let error = anyhow::Error::new(error).context("Could not start the task thread");
                return (Ok(Err(error)), None, Instant::now());
            },
        };

        let finished = || worker.is_finished();
        let shown = control::contain(|| display(&hub, &control, &mut screen, &stray, colored, quiet, &finished));
        let now = Instant::now();
        leave(&hub, &mut screen, &stray);
        let crashed = match shown {
            Ok(true) => None,
            Ok(false) => forced(&hub, &mut out, 130),
            Err(message) => {
                control.abort();
                let since = Instant::now();
                while !finished() && since.elapsed() < ABORT_DEADLINE {
                    thread::sleep(TASK_POLL);
                }
                let failure = unexpected(&hub, message);
                if !finished() {
                    failed(&mut out, task, failure.0, Some(failure.1), colored);
                    forced(&hub, &mut out, 1);
                }
                Some(failure)
            },
        };
        (worker.join(), crashed, now)
    });

    let result = match joined {
        Ok(Ok(ending)) => Ok(ending),
        Ok(Err(error)) => {
            hub.message(Level::Error, format!("{error:#}"));
            Err((lines(&error), advice(task), String::new()))
        },
        Err(payload) => Err(unexpected(&hub, control::panic_message(payload.as_ref()))),
    };
    let (ending, failure, backtrace) = match crashed.map_or(result, Err) {
        Ok(ending) => (ending, None, String::new()),
        Err((error, help, backtrace)) => (Ending::of(Outcome::Failed), Some((error, Some(help.to_owned()))), backtrace),
    };
    if let Some((place, backtrace)) = control::panicked() {
        hub.message(Level::Detail, format!("Underlying panic: {place}\n{backtrace}"));
    }

    let warnings = hub.finish(ending.outcome);
    let verbose = verbose || env::var_os("RUST_BACKTRACE").is_some_and(|value| value != "0");
    let block = hub.view(|live| render::ended(live, &ending, failure.as_ref(), &warnings, now, columns(), colored));
    print(&mut out, &block);
    if verbose && !backtrace.is_empty() {
        print(&mut out, &[backtrace]);
    }

    match ending.outcome {
        Outcome::Finished => ExitCode::SUCCESS,
        Outcome::Stopped | Outcome::Aborted => ExitCode::from(130),
        Outcome::Failed => ExitCode::FAILURE,
    }
}

fn prepare(command: Command) -> Result<(Option<Config>, String, String, Work)>
{
    let TrainArguments { config: file, resume, check, benchmark, metrics_dir, log_every, .. } = match command {
        Command::Train(arguments) => arguments,
        Command::Analyze(analysis) => {
            let options = analysis.options()?;
            let network = options.network.as_ref();
            let subject = match (network, &options.config, &options.inputs[..]) {
                (Some(network), ..) => text::file_name(&network.checkpoint),
                (None, Some((path, _)), _) => text::path(path, HEADER - "Analysis · ".chars().count()),
                (None, None, [input]) => text::file_name(&input.path),
                (None, None, inputs) => counted(inputs, "input"),
            };
            let mut facts = vec![sized(&options.inputs, "input")];
            facts.push(match options.sampling {
                Sampling::All => "every game".to_owned(),
                Sampling::Head { games } => format!("first {} games each", text::count(games)),
            });
            match network {
                Some(network) => {
                    facts.extend(network.architecture.map(|architecture| architecture.name().to_owned()));
                    facts.push(format!("GPU {}", network.gpu));
                },
                None => facts.push(threads(options.threads)),
            }
            let facts = facts.join(" · ");
            return Ok((None, subject, facts, Box::new(move |control, hub| analyze::run(&options, control, hub))));
        },
    };

    let (stored, path) = (file.is_none(), file.unwrap_or_else(|| "config/training.toml".into()));
    let mut config = match resume.as_deref().filter(|_| stored) {
        Some(checkpoint) => train::resume::stored(checkpoint)?,
        None => Config::load(&path)?,
    };
    config.training.log_every = log_every.unwrap_or(config.training.log_every);
    config.training.metrics = metrics_dir.or(config.training.metrics);

    let (name, sources, owned) = (config.run.name.clone(), &config.training.data.sources, config.clone());
    let (subject, facts, work): (String, String, Work) = match (check, benchmark) {
        (Some(mode), _) => {
            let scope = match mode {
                CheckMode::Quick => "quick sample",
                CheckMode::Full => "full scan",
            };
            let facts = format!(" · {scope} · {}", sized(sources, "source"));
            let facts = format!("{}{facts}", text::path(&path, HEADER.saturating_sub(facts.chars().count())));
            (name, facts, Box::new(move |control, hub| check::run(&owned, &path, mode, control, hub)))
        },
        (None, Some(seconds)) => {
            let (arch, duration) = (config.run.arch, Duration::from_secs(seconds));
            let measured = format!("{} × {} · no GPU", bench::MEASUREMENTS.len(), text::span(duration));
            let sources = match &sources[..] {
                [source] => {
                    let size = text::bytes(source.input().total);
                    let taken = format!("{arch} ·  · {size} · {measured}").chars().count();
                    format!("{} · {size}", clipped(text::file_name(&source.path), taken))
                },
                sources => sized(sources, "source"),
            };
            let facts = format!("{arch} · {sources} · {measured}");
            (name, facts, Box::new(move |control, hub| bench::loader(&owned, duration, control, hub)))
        },
        (None, None) => {
            let arch = config.run.arch.to_string();
            (name, arch, Box::new(move |control, hub| train::run(owned, resume, control, hub)))
        },
    };
    Ok((Some(config), subject, facts, work))
}

fn display(
    hub: &Hub,
    control: &Control,
    screen: &mut Option<Screen>,
    stray: &Receiver<String>,
    colored: bool,
    quiet: bool,
    finished: &dyn Fn() -> bool,
) -> bool
{
    let (mut stop, mut aborting, mut printed, mut pulse, mut moved) = (Stop::Running, None, None, 0, (0, 0, 0, 0));
    loop {
        if !finished() {
            thread::sleep(FRAME);
        }
        let (done, now) = (finished(), Instant::now());
        stray.try_iter().for_each(|line| hub.note(Level::Info, line));

        let next = control.stop();
        if next == Stop::Now {
            aborting.get_or_insert(now);
        }
        let notice = match (mem::replace(&mut stop, next), next) {
            (Stop::Running, Stop::AtBoundary | Stop::Committed | Stop::Armed) => Some("stop requested"),
            (Stop::Running | Stop::AtBoundary, Stop::Now | Stop::Exit) | (Stop::Committed | Stop::Armed, Stop::Now) => {
                Some("abort requested")
            },
            _ => None,
        };
        if let Some(notice) = notice {
            hub.note(Level::Warning, notice);
        }
        if done || stop == Stop::Exit || aborting.is_some_and(|since| since.elapsed() >= ABORT_DEADLINE) {
            return done;
        }

        if let Some(open) = screen {
            let drawn = control::contain(|| {
                let size = open.size();
                let block = hub.view(|live| {
                    let done = live.progress.iter().map(|input| input.done).sum();
                    let positions = live.progress.iter().map(|input| input.positions).sum();
                    let seen = (live.superbatch, live.batch, done, positions);
                    if mem::replace(&mut moved, seen) != seen || !(live.ready || live.scan.is_some()) {
                        pulse += 1;
                    }
                    render::block(live, stop, now, pulse, render::width(size.0), colored)
                });
                open.draw(&block, size)
            });
            if let Err(error) = drawn.unwrap_or_else(|panic| Err(io::Error::other(panic))) {
                leave(hub, screen, stray);
                hub.message(Level::Warning, format!("live display stopped: {error}; continuing with plain output"));
            }
        } else if !quiet {
            let every = if hub.view(|live| live.run.is_some()) { TRAINING_LINE_EVERY } else { SCAN_LINE_EVERY };
            if printed.is_none_or(|printed| now.saturating_duration_since(printed) >= every)
                && let Some(line) = hub.view(|live| render::plain(live, stop, now))
            {
                printed = Some(now);
                let _ = writeln!(io::stderr(), "{line}");
            }
        }

        if control.closing() {
            leave(hub, screen, stray);
            control.flushed();
        }
    }
}

fn leave(hub: &Hub, screen: &mut Option<Screen>, stray: &Receiver<String>)
{
    let Some(mut screen) = screen.take() else { return };
    let closed = control::contain(|| screen.close()).unwrap_or_else(|panic| Err(io::Error::other(panic)));
    let deadline = Instant::now() + STRAY_WAIT;
    iter::from_fn(|| stray.recv_timeout(deadline.saturating_duration_since(Instant::now())).ok())
        .for_each(|line| hub.note(Level::Info, line));
    if let Err(error) = closed {
        let _ = writeln!(io::stderr(), "Warning: cannot restore the terminal: {error}");
    }
}

fn failed(out: &mut AutoStream<Stderr>, task: Task, error: String, help: Option<&str>, colored: bool) -> ExitCode
{
    let setup = Setup { task, subject: String::new(), facts: String::new(), log: None, metrics: None };
    let (now, ending, failure) = (Instant::now(), Ending::of(Outcome::Failed), (error, help.map(str::to_owned)));
    let live = Live::new(setup, 1, now);
    print(out, &render::ended(&live, &ending, Some(&failure), &[], now, columns(), colored));
    ExitCode::FAILURE
}

fn forced(hub: &Hub, out: &mut AutoStream<Stderr>, code: i32) -> !
{
    hub.message(Level::Warning, FORCED);
    print(out, &[render::warning(FORCED)]);
    process::exit(code)
}

fn unexpected(hub: &Hub, message: String) -> (String, &'static str, String)
{
    let (place, backtrace) = control::panicked().unwrap_or((message, String::new()));
    hub.message(Level::Error, format!("{UNEXPECTED}: {place}; inspect the log and report this issue"));
    hub.message(Level::Error, format!("Panic backtrace\n{backtrace}"));
    (format!("{UNEXPECTED}: {place}"), "Inspect the log and report this issue", backtrace)
}

fn print(out: &mut AutoStream<Stderr>, lines: &[String])
{
    let _ = writeln!(out, "{}", lines.join("\n")).and_then(|()| out.flush());
}

fn columns() -> usize
{
    let size = crossterm::terminal::size().ok().filter(|_| io::stderr().is_terminal());
    size.map_or(WIDTH, |(columns, _)| render::width(columns))
}

fn advice(task: Task) -> &'static str
{
    match task {
        Task::Train | Task::Check | Task::Benchmark => {
            "Check configuration and dataset paths with train --check; use --verbose and the log for details."
        },
        Task::Analyze => "Check input paths and checkpoint architecture; use --verbose and the log for details.",
    }
}

fn clipped(name: String, taken: usize) -> String
{
    match (name.chars().count() + taken).saturating_sub(HEADER) {
        0 => name,
        excess => format!("…{}", name.chars().skip(excess + 1).collect::<String>()),
    }
}

fn lines(error: &anyhow::Error) -> String { error.chain().map(ToString::to_string).collect::<Vec<_>>().join("\n") }

fn threads(count: usize) -> String { if count == 1 { "1 thread".to_owned() } else { format!("{count} threads") } }

fn counted(sources: &[Source], noun: &str) -> String
{
    let plural = if sources.len() == 1 { "" } else { "s" };
    format!("{} {noun}{plural}", sources.len())
}

fn sized(sources: &[Source], noun: &str) -> String
{
    let bytes = sources.iter().map(|source| source.input().total).sum();
    format!("{} · {}", counted(sources, noun), text::bytes(bytes))
}

fn launch(run: &str, root: &Path, hub: &Hub)
{
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("plot");
    let mut command = process::Command::new("uv");
    command.args(["run", "--project"]).arg(&project).args(["--frozen", "--quiet", "bruce-plot", "view", run]);
    command.arg("--root").arg(root).arg("--wait").env("PYTHONDONTWRITEBYTECODE", "1");
    command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::{io::AsRawHandle, process::CommandExt};

        use windows_sys::Win32::{
            Foundation::{FALSE, HANDLE_FLAG_INHERIT, SetHandleInformation},
            System::Threading::CREATE_NO_WINDOW,
        };

        command.creation_flags(CREATE_NO_WINDOW);
        for raw in [io::stdin().as_raw_handle(), io::stdout().as_raw_handle(), io::stderr().as_raw_handle()] {
            // SAFETY: the standard handle stays open for the call, which only clears its inherit flag.
            #[expect(unsafe_code)]
            let cleared = raw.is_null() || unsafe { SetHandleInformation(raw, HANDLE_FLAG_INHERIT, 0) } != FALSE;
            if !cleared {
                let error = io::Error::last_os_error();
                hub.message(Level::Detail, format!("viewer may keep this run's output open: {error}"));
                break;
            }
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        command.process_group(0);
    }

    if let Err(error) = command.spawn() {
        let open = format!("--project \"{}\" bruce-plot view {run} --root \"{}\"", project.display(), root.display());
        hub.message(Level::Warning, format!("viewer not started ({error}); open it with: uv run {open}"));
    }
}
