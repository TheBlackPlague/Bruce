use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, IsTerminal, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    events::{EVENT_PREFIX, Event},
    progress::{ProgressState, clean},
    tensorboard::TensorBoard,
};
use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

const REFRESH       : Duration = Duration::from_millis(100);
const PLAIN_INTERVAL: Duration = Duration::from_secs  ( 5 );

struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub fn run_child(
    mut command: Command,
    plain: bool,
    log_path: &Path,
    heading: &str,
    mut tensorboard: Option<TensorBoard>,
) -> Result<()> {
    if let Some(parent) = log_path.parent() { std::fs::create_dir_all(parent)?; }

    let log = Arc::new(Mutex::new(BufWriter::new(File::create(log_path)?)));
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal = interrupted.clone();

    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .context("installing interrupt handler")?;

    let mut child = Worker(
        command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin (Stdio:: null())
            .spawn()
            .context("starting Bruce worker")?,
    );

    let stdout = child.0.stdout.take().context("worker stdout")?;
    let stderr = child.0.stderr.take().context("worker stderr")?;

    let (sender, receiver) = mpsc::channel();

    let native_log = log.clone();

    let stdout_reader = thread::spawn(move || -> io::Result<()> {
        let mut flushed = Instant::now();

        for line in BufReader::new(stdout).lines() {
            let mut log = native_log.lock().unwrap();

            writeln!(log, "{}", clean(&line?))?;

            if flushed.elapsed() >= PLAIN_INTERVAL {
                log.flush()?;
                flushed = Instant::now();
            }
        }

        Ok(())
    });

    let event_log = log.clone();
    let stderr_reader = thread::spawn(move || -> io::Result<()> {
        let mut flushed = Instant::now();

        for line in BufReader::new(stderr).lines() {
            let line = line?;
            {
                let mut log = event_log.lock().unwrap();
                writeln!(log, "{}", clean(&line))?;

                if flushed.elapsed() >= PLAIN_INTERVAL {
                    log.flush()?;
                    flushed = Instant::now();
                }
            }

            let event = line
                .strip_prefix(EVENT_PREFIX)
                .and_then(|json| serde_json::from_str(json).ok())
                .unwrap_or_else(|| Event::Note { message: clean(&line) });

            if sender.send(event).is_err() {
                break;
            }
        }

        Ok(())
    });

    let plain = plain                                          ||
        !io::stdout().is_terminal()                            ||
        !io::stderr().is_terminal()                            ||
        std::env::var("TERM").is_ok_and(|term| term == "dumb") ||
        std::env::var_os("CI").is_some();

    let mut display = Display::new(plain);
    display.message(&format!("🦈 Bruce {}", env!("CARGO_PKG_VERSION")))?;

    for line in heading.lines() {
        display.message(line)?;
    }

    if let Some(logger) = &tensorboard {
        display.message(&format!("TensorBoard: {}", logger.directory.display()))?;
    }

    let result = monitor(
        &mut child.0,
        &receiver,
        &interrupted,
        &mut display,
        &mut tensorboard,
    );

    if result.is_err() {
        let _ = child.0.kill();
    }
    let _ = child.0.wait();

    let stdout_result = stdout_reader.join();
    let stderr_result = stderr_reader.join();

    for event in receiver.try_iter() {
        display.event(event, &mut tensorboard)?;
    }
    display.finish(result.is_ok())?;

    if let Some(logger) = &mut tensorboard {
        match logger.finish() {
            Ok (  0  ) => {},
            Ok (count) => display.message(
                &format!("Warning: TensorBoard omitted {count} metric samples because its writer \
                          could not keep up.")
            )?,
            Err(error) => display.message(
                &format!("Warning: TensorBoard logging failed: {error:#}")
            )?,
        }
    }

    log.lock().unwrap().flush()?;

    display.message(&format!("Worker log: {}", log_path.display()))?;

    result?;

    stdout_result.map_err(|_| anyhow::anyhow!("stdout reader panicked"))??;
    stderr_result.map_err(|_| anyhow::anyhow!("stderr reader panicked"))??;

    Ok(())
}

fn monitor(
    child: &mut Child,
    receiver: &Receiver<Event>,
    interrupted: &AtomicBool,
    display: &mut Display,
    tensorboard: &mut Option<TensorBoard>,
) -> Result<()> {
    loop {
        for event in receiver.try_iter().take(512) {
            display.event(event, tensorboard)?;
        }
        display.refresh(false)?;

        if  tensorboard.as_ref().is_some_and(TensorBoard::is_stopped) &&
            let Some(mut logger) = tensorboard.take() &&
            let Err(error) = logger.finish()
        {
            display.message(&format!("Warning: TensorBoard logging disabled: {error:#}"))?;
        }

        if interrupted.load(Ordering::Relaxed) {
            if tensorboard.is_none() && display.training_summary.is_none() {
                bail!("Interrupted");
            }

            bail!(
                "Interrupted. Work since the last completed checkpoint was not saved. {}",
                display.checkpoint
                    .as_ref()
                    .map_or_else(
                        || "No checkpoint was saved by this invocation.".into(),
                        |path| format!("Last completed checkpoint: {path}"
                    )
                )
            );
        }

        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!("Worker exited with {status}");
            }

            return Ok(());
        }

        thread::sleep(REFRESH);
    }
}

struct Display {
    bar: ProgressBar,
    plain: bool,
    state: ProgressState,
    last_plain: Instant,
    checkpoint: Option<String>,
    training_summary: Option<String>,
}

impl Display {
    fn new(plain: bool) -> Self {
        let bar = ProgressBar::with_draw_target(
            None,
            if plain {
                ProgressDrawTarget::hidden()
            } else {
                ProgressDrawTarget::stderr_with_hz(10)
            },
        );

        Self {
            bar,
            plain,
            state: ProgressState::default(),
            last_plain: Instant::now(),
            checkpoint: None,
            training_summary: None,
        }
    }

    fn message(&self, message: &str) -> io::Result<()> {
        let message = clean(message);

        if self.plain || self.bar.is_hidden() {
            writeln!(io::stderr().lock(), "{message}")
        } else {
            self.bar.println(message);

            Ok(())
        }
    }

    fn event(&mut self, event: Event, tensorboard: &mut Option<TensorBoard>) -> Result<()> {
        if let Some(logger) = tensorboard {
            logger.observe(&event);
        }

        if let Event::Checkpoint { path } = &event {
            self.checkpoint = Some(clean(&path.display().to_string()));
        }

        if let Some(message) = self.state.update(&event) {
            self.message(&message)?;
        }

        if matches!(event, Event::Metric { .. }) {
            self.training_summary = Some(self.state.summary());
        }

        Ok(())
    }

    fn refresh(&mut self, force: bool) -> Result<()> {
        if self.plain {
            if force || self.last_plain.elapsed() >= PLAIN_INTERVAL {
                self.message(&self.state.summary())?;
                self.last_plain = Instant::now();
            }
        } else {
            let template = if self.state.total.is_some() {
                "{prefix} {wide_bar:.cyan} {percent:>5.1}%\n{wide_msg}"
            } else {
                "{spinner:.cyan} {prefix}\n{wide_msg}"
            };

            self.bar.set_style(ProgressStyle::with_template(template)?.progress_chars("━╸─"));

            self.bar.set_prefix(if self.state.total.is_some() {
                self.state.count()
            } else {
                self.state.name.clone()
            });

            self.bar.set_length(self.state.total.unwrap_or(0));
            self.bar.set_position(self.state.completed.min(self.state.total.unwrap_or(u64::MAX)));
            self.bar.set_message(self.state.live_message());
            self.bar.tick();
        }

        Ok(())
    }

    fn finish(&mut self, success: bool) -> Result<()> {
        self.bar.finish_and_clear();

        self.message(self.training_summary.as_deref().unwrap_or(&self.state.summary()))?;

        self.message(if success {
            "Complete."
        } else {
            "Stopped before completion."
        })?;

        Ok(())
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        self.bar.finish_and_clear();
    }
}
