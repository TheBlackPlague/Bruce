use std::{
    io,
    sync::{Arc, Condvar, Mutex, mpsc},
    time::Duration,
};

use super::{Logger, QUEUE_CAPACITY, Sample};

#[derive(Clone, Default)]
struct MemoryWriter(Arc<Mutex<Vec<u8>>>);

impl io::Write for MemoryWriter
{
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize>
    {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

struct BlockedWriter
{
    memory: MemoryWriter,
    ready: Option<mpsc::Sender<()>>,
    released: Arc<(Mutex<bool>, Condvar)>,
}

impl io::Write for BlockedWriter
{
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize>
    {
        if let Some(ready) = self.ready.take() {
            ready.send(()).unwrap();
            let (lock, wake) = &*self.released;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
        }
        self.memory.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> { self.memory.flush() }
}

fn sample(step: i64) -> Sample { Sample { step, loss: 0.25, learning_rate: 0.001, speed: 1.5 } }

#[test]
fn saturated_samples_are_counted_while_lifecycle_records_apply_backpressure()
{
    let memory = MemoryWriter::default();
    let (ready, receiver) = mpsc::channel();
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    let logger = Logger::start(
        Box::new(BlockedWriter { memory: memory.clone(), ready: Some(ready), released: released.clone() }),
        None,
        None,
    );
    receiver.recv_timeout(Duration::from_secs(2)).unwrap();

    for step in 0..QUEUE_CAPACITY + 7 {
        logger.sample(sample(step as i64));
    }

    assert_eq!(logger.owner.shared.state.lock().unwrap().queue.len(), QUEUE_CAPACITY);

    let diagnostic = logger.take_diagnostics().join("\n");
    assert!(diagnostic.contains("7 metric samples"));
    assert!(logger.take_diagnostics().is_empty());

    let sender = logger.clone();
    let (completed, receiver) = mpsc::channel();
    let producer = std::thread::spawn(move || {
        sender.record("Checkpoint saved: checkpoints/final").unwrap();
        completed.send(()).unwrap();
    });
    assert!(receiver.recv_timeout(Duration::from_millis(25)).is_err());

    {
        let (lock, wake) = &*released;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }

    receiver.recv_timeout(Duration::from_secs(2)).unwrap();
    producer.join().unwrap();
    logger.finish().unwrap();

    let text = String::from_utf8(memory.0.lock().unwrap().clone()).unwrap();

    assert!(text.contains("Checkpoint saved: checkpoints/final"));
    assert!(text.contains("7 metric samples"));
    assert_eq!(text.matches("Batch ").count(), QUEUE_CAPACITY);
    assert!(text.contains("closed"));
}

struct FailingWriter
{
    fail_flush: bool,
}

impl io::Write for FailingWriter
{
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize>
    {
        if self.fail_flush { Ok(bytes.len()) } else { Err(io::Error::other("injected text write failure")) }
    }

    fn flush(&mut self) -> io::Result<()>
    {
        if self.fail_flush { Err(io::Error::other("injected text flush failure")) } else { Ok(()) }
    }
}

#[test]
fn text_write_and_flush_failures_are_prompt_and_retained()
{
    for fail_flush in [false, true] {
        let logger = Logger::start(Box::new(FailingWriter { fail_flush }), None, None);
        let start = std::time::Instant::now();

        let diagnostics = loop {
            let diagnostics = logger.take_diagnostics();
            if !diagnostics.is_empty() {
                break diagnostics.join("\n");
            }
            assert!(start.elapsed() < Duration::from_secs(2), "Worker failure was not published promptly");
            std::thread::yield_now();
        };

        assert!(diagnostics.contains("injected text"));
        assert!(logger.record("Cannot be persisted").is_err());
        assert!(logger.finish().is_err());
        assert!(logger.finish().is_err());
    }
}

struct PanickingWriter;

impl io::Write for PanickingWriter
{
    fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("injected worker panic") }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

#[test]
fn worker_panic_is_returned_as_a_failure()
{
    let logger = Logger::start(Box::new(PanickingWriter), None, None);

    assert!(logger.finish().unwrap_err().to_string().contains("worker panicked"));
    assert!(logger.take_diagnostics().join("\n").contains("worker panicked"));
}

struct FinalFlushFailure
{
    flushes: usize,
    memory: MemoryWriter,
}

impl io::Write for FinalFlushFailure
{
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.memory.write(bytes) }

    fn flush(&mut self) -> io::Result<()>
    {
        self.flushes += 1;
        if self.flushes == 1 { Ok(()) } else { Err(io::Error::other("injected final flush failure")) }
    }
}

#[test]
fn accepted_outcome_is_drained_before_final_flush_failure_is_returned()
{
    let memory = MemoryWriter::default();
    let logger = Logger::start(Box::new(FinalFlushFailure { flushes: 0, memory: memory.clone() }), None, None);

    logger.record("Outcome: failed; required final checkpoint failed").unwrap();

    assert!(logger.finish().unwrap_err().to_string().contains("final flush failure"));

    let text = String::from_utf8(memory.0.lock().unwrap().clone()).unwrap();

    assert!(text.contains("Outcome: failed; required final checkpoint failed"));
    assert!(logger.record("Already finished").is_err());
}

#[test]
fn sessions_append_with_identity_and_logs_never_contain_terminal_controls()
{
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("run.log");

    for _ in 0..2 {
        let logger = Logger::open(&path, None).unwrap();

        logger.record("\u{1b}[31mwarning\u{1b}[0m: path\u{1b}]8;;https://example.com\u{7}value\u{1b}]8;;\u{1b}\\\r\nunderlying cause").unwrap();
        logger.finish().unwrap();
    }

    let text = std::fs::read_to_string(path).unwrap();
    let starts: Vec<_> = text.lines().filter(|line| line.contains("Session ") && line.contains("started")).collect();

    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0], starts[1]);
    assert_eq!(text.matches("warning: pathvalue").count(), 2);
    assert_eq!(text.matches("underlying cause").count(), 2);
    assert!(!text.contains(['\u{1b}', '\r', '\u{7}']));
}

#[cfg(target_os = "linux")]
#[test]
fn failed_tensorboard_flush_is_nonfatal_and_saved_in_text_log()
{
    use super::{BufWriter, File, TensorBoard};

    let memory = MemoryWriter::default();
    let tensorboard = TensorBoard { writer: BufWriter::new(File::options().write(true).open("/dev/full").unwrap()) };
    let logger = Logger::start(Box::new(memory.clone()), Some(tensorboard), None);
    logger.sample(sample(42));
    logger.record("Required checkpoint saved").unwrap();
    logger.finish().unwrap();

    assert!(logger.take_diagnostics().join("\n").contains("TensorBoard logging disabled"));
    let text = String::from_utf8(memory.0.lock().unwrap().clone()).unwrap();
    assert!(text.contains("TensorBoard logging disabled"));
    assert!(text.contains("Required checkpoint saved"));
}
