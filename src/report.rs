use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, IsTerminal, Write},
    path::Path,
    sync::mpsc::{self, SyncSender, TrySendError},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use bullet_trainer::run::{TrainingEvent, TrainingSteps};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use prost::Message;

use crate::config::Config;

pub struct Reporter
{
    steps: TrainingSteps,
    log_every: usize,
    setup: Duration,
    bar: ProgressBar,
    plain: bool,
    last_print: Instant,
    sender: Option<SyncSender<Report>>,
    writer: Option<JoinHandle<Result<()>>>,
    dropped: usize,
    loss_sum: f64,
    loss_batches: usize,
}

enum Report
{
    Message(String),
    Sample
    {
        step: i64,
        loss: f32,
        learning_rate: f32,
        speed: f32,
    },
}

impl Reporter
{
    pub fn new(config: &Config, steps: TrainingSteps, plain: bool) -> Result<Self>
    {
        fs::create_dir_all("logs").context("Cannot create log directory")?;
        let mut log = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(Path::new("logs").join(format!("{}.log", config.name)))?,
        );
        
        writeln!(log, "Training {}", config.name)?;
        log.flush()?;

        let tensorboard = if config.tensorboard.enabled {
            match TensorBoard::new(&config.tensorboard.path, &config.name) {
                Ok(writer) => Some(writer),
                Err(error) => {
                    eprintln!("TensorBoard is unavailable: {error}");
                    None
                },
            }
        } else {
            None
        };
        
        let (sender, receiver) = mpsc::sync_channel(256);
        
        let writer = thread::spawn(move || -> Result<()> {
            let mut tensorboard = tensorboard;
            let mut flushed = Instant::now();
            loop {
                match receiver.recv_timeout(Duration::from_secs(5).saturating_sub(flushed.elapsed())) {
                    Ok(Report::Message(message)) => writeln!(log, "{message}")?,
                    Ok(Report::Sample { step, loss, learning_rate, speed }) => {
                        writeln!(log, "Batch {step}: loss {loss:.6}, lr {learning_rate:.3e}, {speed:.3} Mpos/s")?;
                        tensorboard_action(&mut tensorboard, |writer| writer.sample(step, loss, learning_rate, speed));
                    },
                    Err(mpsc::RecvTimeoutError::Timeout) => {},
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                
                if flushed.elapsed() >= Duration::from_secs(5) {
                    log.flush()?;
                    tensorboard_action(&mut tensorboard, TensorBoard::flush);
                    flushed = Instant::now();
                }
            }
            log.flush()?;
            
            tensorboard_action(&mut tensorboard, TensorBoard::flush);
            
            Ok(())
        });

        let plain = plain
            || !io::stderr().is_terminal()
            || !io::stdout().is_terminal()
            || std::env::var_os("CI").is_some()
            || std::env::var("TERM").is_ok_and(|term| term == "dumb");
        
        let target = if plain { ProgressDrawTarget::hidden() } else { ProgressDrawTarget::stderr_with_hz(10) };
        let bar = ProgressBar::with_draw_target(Some(steps.batches_per_superbatch as u64), target);
        bar.set_style(ProgressStyle::with_template("{wide_msg}")?);
        
        Ok(Self {
            steps,
            log_every: config.training.log_every,
            setup: Duration::ZERO,
            bar,
            plain,
            last_print: Instant::now(),
            sender: Some(sender),
            writer: Some(writer),
            dropped: 0,
            loss_sum: 0.0,
            loss_batches: 0,
        })
    }

    pub fn message(&mut self, message: &str)
    {
        let message: String = message.chars().filter(|ch| !ch.is_control()).collect();
        if self.plain {
            eprintln!("{message}");
        } else {
            self.bar.println(&message);
        }
        
        self.send(Report::Message(message));
    }

    pub fn on_event(&mut self, event: &TrainingEvent)
    {
        match event {
            TrainingEvent::RunStarted { device_name, .. } => self.message(&format!("Training on {device_name}")),
            TrainingEvent::TrainingReady { setup_time, .. } => {
                self.setup = *setup_time;
                self.message("Training started");
            },
            TrainingEvent::BatchCompleted { step, loss, learning_rate, elapsed, .. } => {
                self.loss_sum += f64::from(*loss);
                self.loss_batches += 1;
                
                let batch = step.batch() + 1;
                if batch != 1 && batch != step.batches_per_superbatch() && !batch.is_multiple_of(self.log_every) {
                    return;
                }
                
                let loss = (self.loss_sum / self.loss_batches as f64) as f32;
                self.loss_sum = 0.0;
                self.loss_batches = 0;
                
                let global_batch = step.total_batches() + 1;
                let completed = global_batch - self.steps.start_batch();
                let seconds = elapsed.saturating_sub(self.setup).as_secs_f64();
                let batches_per_second = completed as f64 / seconds.max(f64::EPSILON);
                let speed = (batches_per_second * self.steps.batch_size as f64 / 1e6) as f32;
                let remaining = (step.final_superbatch() - step.superbatch()) * step.batches_per_superbatch()
                    + step.batches_per_superbatch()
                    - batch;
                let eta = duration(remaining as f64 / batches_per_second);
                let batch_eta = duration((step.batches_per_superbatch() - batch) as f64 / batches_per_second);
                
                let message = format!(
                    "Superbatch {}/{} · {batch}/{} · loss {loss:.6} · lr {learning_rate:.3e} · {speed:.2} Mpos/s · \
                     elapsed {} · ETA {batch_eta} · Total ETA {eta}",
                    step.superbatch(),
                    step.final_superbatch(),
                    step.batches_per_superbatch(),
                    duration(seconds),
                );
                
                self.bar.set_position(batch as u64);
                if self.plain {
                    if batch == 1 || self.last_print.elapsed() >= Duration::from_secs(5) {
                        eprintln!("{message}");
                        self.last_print = Instant::now();
                    }
                } else {
                    self.bar.set_message(message);
                }
                
                if let Ok(step) = i64::try_from(global_batch) {
                    self.send(Report::Sample { step, loss, learning_rate: *learning_rate, speed });
                }
            },
            TrainingEvent::SuperbatchCompleted { step, loss, .. } => {
                self.message(&format!("Superbatch {} complete · loss {loss:.6}", step.superbatch()));
            },
            TrainingEvent::RunCompleted { .. } => self.message("Training complete"),
            _ => {},
        }
    }

    fn send(&mut self, report: Report)
    {
        if let Some(sender) = &self.sender {
            match sender.try_send(report) {
                Ok(()) => {},
                Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => self.dropped += 1,
            }
        }
    }

    pub fn finish(&mut self) -> Result<()>
    {
        self.sender.take();
        self.bar.finish_and_clear();
        
        if let Some(writer) = self.writer.take() {
            match writer.join() {
                Ok(Ok(())) => {},
                Ok(Err(error)) => eprintln!("Reporting stopped: {error}"),
                Err(_) => eprintln!("Reporting stopped unexpectedly"),
            }
        }
        
        if self.dropped > 0 {
            eprintln!("Reporting skipped {} updates", self.dropped);
            self.dropped = 0;
        }
        
        Ok(())
    }
}

fn tensorboard_action(writer: &mut Option<TensorBoard>, action: impl FnOnce(&mut TensorBoard) -> Result<()>)
{
    if let Some(active) = writer
        && let Err(error) = action(active)
    {
        eprintln!("TensorBoard reporting stopped: {error}");
        *writer = None;
    }
}

impl Drop for Reporter
{
    fn drop(&mut self) { let _ = self.finish(); }
}

fn duration(seconds: f64) -> String
{
    if !seconds.is_finite() || seconds < 0.0 {
        return "--:--:--".into();
    }
    let seconds = seconds as u64;
    format!("{:02}:{:02}:{:02}", seconds / 3600, seconds / 60 % 60, seconds % 60)
}

pub struct TensorBoard
{
    writer: BufWriter<File>,
}

impl TensorBoard
{
    pub fn new(root: &Path, name: &str) -> Result<Self>
    {
        anyhow::ensure!(
            !name.is_empty() && name.bytes().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, b'-' | b'_')),
            "Invalid run name"
        );
        
        let directory = root.join(name);
        fs::create_dir_all(&directory)?;
        
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = directory.join(format!("events.out.tfevents.{timestamp}.{}.bruce", std::process::id()));
        
        let mut result = Self { writer: BufWriter::new(File::create_new(path)?) };
        result.write(Event { file_version: Some("brain.Event:2".into()), ..event(0) })?;
        result.flush()?;
        
        Ok(result)
    }

    pub fn sample(&mut self, step: i64, loss: f32, learning_rate: f32, speed: f32) -> Result<()>
    {
        let values =
            [("train/loss", loss), ("train/learning_rate", learning_rate), ("performance/mpos_per_second", speed)]
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(tag, simple_value)| Value { tag: tag.into(), simple_value })
                .collect();
        
        self.write(Event { summary: Some(Summary { value: values }), ..event(step) })
    }

    fn write(&mut self, event: Event) -> Result<()>
    {
        let data = event.encode_to_vec();
        let length = (data.len() as u64).to_le_bytes();
        
        self.writer.write_all(&length)?;
        self.writer.write_all(&checksum(&length))?;
        self.writer.write_all(&data)?;
        self.writer.write_all(&checksum(&data))?;
        
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> { self.writer.flush().context("Cannot flush TensorBoard events") }
}

fn checksum(bytes: &[u8]) -> [u8; 4] { crc32c::crc32c(bytes).rotate_right(15).wrapping_add(0xa282_ead8).to_le_bytes() }

fn event(step: i64) -> Event
{
    Event {
        wall_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64(),
        step,
        file_version: None,
        summary: None,
    }
}

#[derive(Clone, PartialEq, Message)]
struct Event
{
    #[prost(double, tag = "1")]
    wall_time: f64,
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(string, optional, tag = "3")]
    file_version: Option<String>,
    #[prost(message, optional, tag = "5")]
    summary: Option<Summary>,
}

#[derive(Clone, PartialEq, Message)]
struct Summary
{
    #[prost(message, repeated, tag = "1")]
    value: Vec<Value>,
}

#[derive(Clone, PartialEq, Message)]
struct Value
{
    #[prost(string, tag = "1")]
    tag: String,
    #[prost(float, tag = "2")]
    simple_value: f32,
}

#[cfg(test)]
#[path = "../tests/support/report_unit.rs"]
mod tests;
