use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};

mod tensorboard;
pub use tensorboard::TensorBoard;

pub use crate::report::MetricSample as Sample;

const QUEUE_CAPACITY: usize = 256;
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);
static SESSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct Logger
{
    session: Arc<Session>,
}

struct Session
{
    sender: Mutex<Option<SyncSender<Record>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    diagnostics: Arc<Mutex<Diagnostics>>,
}

#[derive(Default)]
struct Diagnostics
{
    failure: Option<String>,
    failure_reported: bool,
    warnings: Vec<String>,
    dropped_for_display: u64,
    dropped_for_log: u64,
}

enum Record
{
    Message(String),
    Sample(Sample),
}

impl Logger
{
    pub fn open(path: &Path, tensorboard: Option<(&Path, &str)>) -> Result<Self>
    {
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent).with_context(|| format!("Cannot create log directory {}", parent.display()))?;
        }
        
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Cannot open log {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        let session =
            format!("{}-{}-{}", timestamp(), std::process::id(), SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed));
        writeln!(writer, "\nSession {session} started (timestamps are Unix seconds UTC)")?;

        let diagnostics = Arc::new(Mutex::new(Diagnostics::default()));
        let tensorboard = match tensorboard {
            Some((root, name)) => match TensorBoard::new(root, name) {
                Ok(board) => Some(board),
                Err(error) => {
                    let warning = tensorboard_warning(error);
                    write_message(&mut writer, &format!("Warning: {warning}"))?;
                    
                    diagnostics.lock().unwrap().warnings.push(warning);
                    
                    None
                },
            },
            None => None,
        };
        
        writer.flush().context("Cannot begin the text log; check free space and permissions")?;

        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let status = diagnostics.clone();
        let worker = thread::Builder::new()
            .name("bruce-log".into())
            .spawn(move || {
                let result =
                    catch_unwind(AssertUnwindSafe(|| write_records(receiver, writer, tensorboard, &status, &session)))
                        .unwrap_or_else(|_| Err(anyhow!("Log worker stopped unexpectedly")));
                if let Err(error) = result {
                    status.lock().unwrap_or_else(|error| error.into_inner()).failure = Some(format!("{error:#}"));
                }
            })
            .context("Cannot start diagnostic logging")?;

        Ok(Self {
            session: Arc::new(Session {
                sender: Mutex::new(Some(sender)),
                worker: Mutex::new(Some(worker)),
                diagnostics,
            }),
        })
    }

    pub fn record(&self, message: &str) -> Result<()>
    {
        let sender = self.session.sender.lock().unwrap().clone().context("Log session is already closed")?;
        sender.send(Record::Message(clean_text(message))).map_err(|_| self.session.failure())
    }

    pub fn sample(&self, sample: Sample)
    {
        let sender = self.session.sender.lock().unwrap().clone();
        if let Some(sender) = sender
            && let Err(TrySendError::Full(_)) = sender.try_send(Record::Sample(sample))
        {
            let mut diagnostics = self.session.diagnostics.lock().unwrap();
            
            diagnostics.dropped_for_display = diagnostics.dropped_for_display.saturating_add(1);
            diagnostics.dropped_for_log = diagnostics.dropped_for_log.saturating_add(1);
        }
    }

    pub fn take_diagnostics(&self) -> Vec<String>
    {
        let mut diagnostics = self.session.diagnostics.lock().unwrap_or_else(|error| error.into_inner());
        
        let mut messages = std::mem::take(&mut diagnostics.warnings);
        
        let dropped = std::mem::take(&mut diagnostics.dropped_for_display);
        if dropped > 0 {
            messages.push(dropped_message(dropped));
        }
        
        if !diagnostics.failure_reported
            && let Some(failure) = &diagnostics.failure
        {
            messages.push(format!("Text logging failed: {failure}"));
            diagnostics.failure_reported = true;
        }
        
        messages
    }

    pub fn finish(&self) -> Result<()> { self.session.finish() }
}

impl Session
{
    fn failure(&self) -> anyhow::Error
    {
        let status = self.diagnostics.lock().unwrap_or_else(|error| error.into_inner());
        anyhow!(status.failure.clone().unwrap_or_else(|| "Text log worker is unavailable".into()))
    }

    fn finish(&self) -> Result<()>
    {
        let mut worker = self.worker.lock().unwrap_or_else(|error| error.into_inner());
        
        self.sender.lock().unwrap_or_else(|error| error.into_inner()).take();
        
        if let Some(worker) = worker.take()
            && worker.join().is_err()
        {
            self.diagnostics.lock().unwrap_or_else(|error| error.into_inner()).failure =
                Some("Text log worker panicked".into());
        }
        
        let status = self.diagnostics.lock().unwrap_or_else(|error| error.into_inner());
        
        match &status.failure {
            Some(failure) => Err(anyhow!(failure.clone())),
            None => Ok(()),
        }
    }
}

impl Drop for Session
{
    fn drop(&mut self) { let _ = self.finish(); }
}

fn write_records(
    receiver: Receiver<Record>,
    mut writer: impl Write,
    mut tensorboard: Option<TensorBoard>,
    diagnostics: &Mutex<Diagnostics>,
    session: &str,
) -> Result<()>
{
    let mut flushed = Instant::now();
    loop {
        let received = receiver.recv_timeout(FLUSH_INTERVAL.saturating_sub(flushed.elapsed()));
        let closing = matches!(received, Err(RecvTimeoutError::Disconnected));
        
        match received {
            Ok(Record::Message(message)) => write_message(&mut writer, &message)?,
            Ok(Record::Sample(sample)) => {
                writeln!(
                    writer,
                    "[{}] Batch {}: loss {:.6}, lr {:.3e}, {:.3} Mpos/s",
                    timestamp(),
                    sample.step,
                    sample.loss,
                    sample.learning_rate,
                    sample.speed
                )?;
                
                tensorboard_action(&mut tensorboard, diagnostics, &mut writer, |board| {
                    board.sample(sample.step, sample.loss, sample.learning_rate, sample.speed)
                })?;
            },
            Err(_) => {},
        }

        if closing || flushed.elapsed() >= FLUSH_INTERVAL {
            let dropped = std::mem::take(&mut diagnostics.lock().unwrap().dropped_for_log);
            if dropped > 0 {
                write_message(&mut writer, &format!("Warning: {}", dropped_message(dropped)))?;
            }
            
            tensorboard_action(&mut tensorboard, diagnostics, &mut writer, TensorBoard::flush)?;
            
            if closing {
                write_message(&mut writer, &format!("Session {session} closed"))?;
            }
            
            writer.flush().context("Cannot flush the text log; check free space and permissions")?;
            flushed = Instant::now();
        }
        
        if closing {
            return Ok(());
        }
    }
}

fn tensorboard_action(
    board: &mut Option<TensorBoard>,
    diagnostics: &Mutex<Diagnostics>,
    writer: &mut impl Write,
    action: impl FnOnce(&mut TensorBoard) -> Result<()>,
) -> Result<()>
{
    if let Some(active) = board
        && let Err(error) = action(active)
    {
        let warning = tensorboard_warning(error);
        *board = None;
        diagnostics.lock().unwrap().warnings.push(warning.clone());
        write_message(writer, &format!("Warning: {warning}"))?;
    }
    
    Ok(())
}

fn tensorboard_warning(error: anyhow::Error) -> String
{
    format!(
        "TensorBoard logging disabled: {error:#}. Training and text logging continue; check the destination and free \
         space"
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

fn write_message(writer: &mut impl Write, text: &str) -> Result<()>
{
    for line in clean_text(text).lines() {
        writeln!(writer, "[{}] {line}", timestamp()).context("Cannot write the text log")?;
    }
    
    Ok(())
}

pub(crate) fn clean_text(text: &str) -> String
{
    let mut clean = String::with_capacity(text.len());
    
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\u{1b}' => match chars.next() {
                Some('[') => {
                    for ch in chars.by_ref() {
                        if ('@'..='~').contains(&ch) {
                            break;
                        }
                    }
                },
                Some(']') => {
                    while let Some(ch) = chars.next() {
                        if ch == '\u{7}' || (ch == '\u{1b}' && chars.next_if_eq(&'\\').is_some()) {
                            break;
                        }
                    }
                },
                _ => {},
            },
            '\n' | '\t' => clean.push(ch),
            ch if !ch.is_control() => clean.push(ch),
            _ => {},
        }
    }
    
    clean
}
