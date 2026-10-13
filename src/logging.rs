use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use prost::Message;

pub use crate::report::MetricSample as Sample;

const QUEUE_CAPACITY: usize = 256;
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);
static SESSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct Logger
{
    owner: Arc<Owner>,
}

struct Owner
{
    shared: Arc<Shared>,
    worker: Mutex<Option<JoinHandle<std::result::Result<(), String>>>>,
}

struct Shared
{
    state: Mutex<State>,
    available: Condvar,
    space: Condvar,
}

#[derive(Default)]
struct State
{
    queue: VecDeque<Record>,
    closed: bool,
    failure: Option<String>,
    failure_reported: bool,
    auxiliary_warning: Option<String>,
    dropped_notice: u64,
    dropped_log: u64,
}

enum Record
{
    Message(String),
    Sample(Sample),
}

impl Logger
{
    #[cfg(test)]
    pub(crate) fn for_test(writer: Box<dyn Write + Send>) -> Self { Self::start(writer, None, None) }

    pub fn open(path: &Path, tensorboard: Option<(&Path, &str)>) -> Result<Self>
    {
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent).with_context(|| format!("Cannot create log directory {}", parent.display()))?;
        }

        let file =
            OpenOptions::new().create(true).append(true).open(path).with_context(|| {
                format!("Cannot open log {}; check the destination and permissions", path.display())
            })?;

        let (tensorboard, warning) = match tensorboard {
            Some((root, name)) => match TensorBoard::new(root, name) {
                Ok(writer) => (Some(writer), None),
                Err(error) => (None, Some(auxiliary_error(&error))),
            },
            None => (None, None),
        };

        Ok(Self::start(Box::new(BufWriter::new(file)), tensorboard, warning))
    }

    fn start(writer: Box<dyn Write + Send>, tensorboard: Option<TensorBoard>, warning: Option<String>) -> Self
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { auxiliary_warning: warning.clone(), ..State::default() }),
            available: Condvar::new(),
            space: Condvar::new(),
        });

        let target = shared.clone();

        let session =
            format!("{}-{}-{}", timestamp(), std::process::id(), SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed));

        let worker = thread::spawn(move || {
            let result =
                catch_unwind(AssertUnwindSafe(|| write_records(&target, writer, tensorboard, warning, &session)))
                    .unwrap_or_else(|_| Err(anyhow!("Log worker panicked; inspect the log destination and rerun")))
                    .map_err(|error| format!("{error:#}"));

            if let Err(error) = &result {
                target.fail(error.clone());
            }

            result
        });

        Self { owner: Arc::new(Owner { shared, worker: Mutex::new(Some(worker)) }) }
    }

    pub fn record(&self, message: &str) -> Result<()>
    {
        let shared = &self.owner.shared;

        let mut state = shared.state.lock().unwrap();
        while state.queue.len() >= QUEUE_CAPACITY && !state.closed {
            state = shared.space.wait(state).unwrap();
        }

        if let Some(error) = &state.failure {
            return Err(anyhow!(error.clone()));
        }

        anyhow::ensure!(!state.closed, "Log session is already closed");

        state.queue.push_back(Record::Message(clean_text(message)));

        shared.available.notify_one();
        Ok(())
    }

    pub fn sample(&self, sample: Sample)
    {
        let shared = &self.owner.shared;

        let mut state = shared.state.lock().unwrap();
        if state.closed {
            return;
        }

        if state.queue.len() >= QUEUE_CAPACITY {
            state.dropped_notice = state.dropped_notice.saturating_add(1);
            state.dropped_log = state.dropped_log.saturating_add(1);
        } else {
            state.queue.push_back(Record::Sample(sample));
            shared.available.notify_one();
        }
    }

    pub fn take_diagnostics(&self) -> Vec<String>
    {
        let mut state = self.owner.shared.state.lock().unwrap();

        let mut messages = Vec::with_capacity(3);
        if let Some(warning) = state.auxiliary_warning.take() {
            messages.push(warning);
        }

        if state.dropped_notice > 0 {
            messages.push(dropped_message(std::mem::take(&mut state.dropped_notice)));
        }

        if !state.failure_reported
            && let Some(error) = &state.failure
        {
            messages.push(format!("Text logging failed: {error}"));
            state.failure_reported = true;
        }

        messages
    }

    pub fn finish(&self) -> Result<()> { self.owner.finish() }
}

impl Shared
{
    fn fail(&self, error: String)
    {
        let mut state = self.state.lock().unwrap();
        state.failure.get_or_insert(error);
        state.closed = true;
        self.space.notify_all();
        self.available.notify_all();
    }
}

impl Owner
{
    fn finish(&self) -> Result<()>
    {
        let mut worker = self.worker.lock().unwrap();
        if let Some(worker) = worker.take() {
            {
                let mut state = self.shared.state.lock().unwrap();
                state.closed = true;
                self.shared.available.notify_one();
                self.shared.space.notify_all();
            }
            match worker.join() {
                Ok(Ok(())) => {},
                Ok(Err(error)) => self.shared.fail(error),
                Err(_) => {
                    self.shared.fail("Log worker stopped unexpectedly; inspect the log destination and rerun".into());
                },
            }
        }

        match &self.shared.state.lock().unwrap().failure {
            Some(error) => Err(anyhow!(error.clone())),
            None => Ok(()),
        }
    }
}

impl Drop for Owner
{
    fn drop(&mut self) { let _ = self.finish(); }
}

fn write_records(
    shared: &Shared,
    mut writer: Box<dyn Write + Send>,
    mut tensorboard: Option<TensorBoard>,
    warning: Option<String>,
    session: &str,
) -> Result<()>
{
    writeln!(writer, "\nSession {session} started (timestamps are Unix seconds UTC)")
        .context("Cannot begin the text log; check free space and permissions")?;

    if let Some(warning) = warning {
        write_message(&mut writer, &format!("Warning: {warning}"))?;
    }
    writer.flush().context("Cannot flush the text log; check free space and permissions")?;

    let mut flushed = Instant::now();

    loop {
        let (record, done) = {
            let mut state = shared.state.lock().unwrap();
            if state.queue.is_empty() && !state.closed {
                state =
                    shared.available.wait_timeout(state, FLUSH_INTERVAL.saturating_sub(flushed.elapsed())).unwrap().0;
            }

            let record = state.queue.pop_front();
            let done = state.closed && record.is_none();

            shared.space.notify_all();
            (record, done)
        };

        match record {
            Some(Record::Message(message)) => write_message(&mut writer, &message)?,
            Some(Record::Sample(sample)) => {
                writeln!(
                    writer,
                    "[{}] Batch {}: loss {:.6}, lr {:.3e}, {:.3} Mpos/s",
                    timestamp(),
                    sample.step,
                    sample.loss,
                    sample.learning_rate,
                    sample.speed
                )
                .context("Cannot write the text log; check free space and permissions")?;

                tensorboard_action(shared, &mut writer, &mut tensorboard, |writer| {
                    writer.sample(sample.step, sample.loss, sample.learning_rate, sample.speed)
                })?;
            },
            None => {},
        }

        if done || flushed.elapsed() >= FLUSH_INTERVAL {
            let dropped = std::mem::take(&mut shared.state.lock().unwrap().dropped_log);
            if dropped > 0 {
                write_message(&mut writer, &format!("Warning: {}", dropped_message(dropped)))?;
            }

            tensorboard_action(shared, &mut writer, &mut tensorboard, TensorBoard::flush)?;

            if done {
                write_message(&mut writer, &format!("Session {session} closed"))?;
            }

            writer.flush().context("Cannot flush the text log; check free space and permissions")?;

            flushed = Instant::now();
        }

        if done {
            return Ok(());
        }
    }
}

fn write_message(writer: &mut impl Write, message: &str) -> Result<()>
{
    for line in clean_text(message).lines() {
        writeln!(writer, "[{}] {line}", timestamp())
            .context("Cannot write the text log; check free space and permissions")?;
    }
    Ok(())
}

fn tensorboard_action(
    shared: &Shared,
    text: &mut impl Write,
    writer: &mut Option<TensorBoard>,
    action: impl FnOnce(&mut TensorBoard) -> Result<()>,
) -> Result<()>
{
    if let Some(active) = writer
        && let Err(error) = action(active)
    {
        let warning = auxiliary_error(&error);

        shared.state.lock().unwrap().auxiliary_warning = Some(warning.clone());
        *writer = None;

        write_message(text, &format!("Warning: {warning}"))?;
    }
    Ok(())
}

fn auxiliary_error(error: &anyhow::Error) -> String
{
    format!(
        "TensorBoard logging disabled: {error:#}. Training and text logging continue; check the TensorBoard \
         destination and free space"
    )
}

fn dropped_message(count: u64) -> String
{
    format!(
        "Logging skipped {count} metric samples because the bounded queue was full; lifecycle records are preserved"
    )
}

fn timestamp() -> String
{
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:09}", time.as_secs(), time.subsec_nanos())
}

fn clean_text(text: &str) -> String
{
    let mut result = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();

    while let Some(character) = characters.next() {
        match character {
            '\u{1b}' => match characters.next() {
                Some('[') => {
                    for character in characters.by_ref() {
                        if ('@'..='~').contains(&character) {
                            break;
                        }
                    }
                },

                Some(']') => {
                    while let Some(character) = characters.next() {
                        if character == '\u{7}' || (character == '\u{1b}' && characters.next_if_eq(&'\\').is_some()) {
                            break;
                        }
                    }
                },
                _ => {},
            },

            '\n' | '\t' => result.push(character),

            character if !character.is_control() => result.push(character),

            _ => {},
        }
    }

    result
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
#[path = "../tests/support/logging_unit.rs"]
mod tests;
