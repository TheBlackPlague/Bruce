mod proto;

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::mpsc::{self, SyncSender, TrySendError},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{events::Event, progress::rate};
use anyhow::{Context, Result, ensure};
use prost::Message;

const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

pub struct TensorBoard {
    pub directory: PathBuf,
    sender: Option<SyncSender<proto::Event>>,
    worker: Option<JoinHandle<Result<()>>>,
    dropped: u64,
}

impl TensorBoard {
    pub fn start(root: &Path, name: &str) -> Result<Self> {
        ensure!(
            !name.is_empty() &&
             name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "invalid TensorBoard run name"
        );

        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = root.join(name);

        fs::create_dir_all(&directory).context("create TensorBoard run directory")?;

        let file = File::create_new(
            directory.join(format!("events.out.tfevents.{stamp}.{}.bruce", std::process::id()))
        )?;
        let mut writer = BufWriter::with_capacity(64 * 1024, file);

        write_record(
            &mut writer,
            &proto::Event {
                wall_time: wall_time(),
                step: 0,
                file_version: Some("brain.Event:2".into()),
                summary: None,
            },
        )?;
        writer.flush()?;

        let (sender, receiver) = mpsc::sync_channel::<proto::Event>(256);

        let worker = thread::spawn(move || {
            let mut flushed = Instant::now();
            loop {
                match receiver.recv_timeout(FLUSH_INTERVAL.saturating_sub(flushed.elapsed())) {
                    Ok(event) => write_record(&mut writer, &event)?,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }

                if flushed.elapsed() >= FLUSH_INTERVAL {
                    writer.flush()?;
                    flushed = Instant::now();
                }
            }

            writer.flush().context("flush TensorBoard events")
        });

        Ok(Self {
            directory,
            sender: Some(sender),
            worker: Some(worker),
            dropped: 0,
        })
    }

    pub fn observe(&mut self, event: &Event) {
        let Some(event) = metric_event(event) else {
            return;
        };

        if let Some(sender) = &self.sender {
            match sender.try_send(event) {
                Ok(()) => {}

                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                    self.dropped += 1
                }
            }
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.worker.as_ref().is_some_and(|worker| worker.is_finished())
    }

    pub fn finish(&mut self) -> Result<u64> {
        self.sender.take();

        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| anyhow::anyhow!("TensorBoard writer panicked"))??;
        }

        Ok(self.dropped)
    }
}

impl Drop for TensorBoard {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn wall_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

fn metric_event(event: &Event) -> Option<proto::Event> {
    let Event::Metric {
        superbatch,
        batch,
        batches_per_superbatch,
        final_superbatch: _,
        loss,
        learning_rate,
        positions,
        total_positions: _,
        elapsed_seconds,
    } = event
    else {
        return None;
    };

    let step = superbatch
        .checked_sub(1)?
        .checked_mul(*batches_per_superbatch)?
        .checked_add(*batch)?;

    let mut values = vec![
        ("train/loss", *loss),
        ("train/learning_rate", *learning_rate)
    ];

    if let Some(speed) = rate(*positions, *elapsed_seconds) {
        values.push(("performance/mpos_per_second", (speed / 1_000_000.0) as f32));
    }

    Some(proto::Event {
        wall_time: wall_time(),
        step: i64::try_from(step).ok()?,
        file_version: None,
        summary: Some(proto::Summary {
            value: values
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(tag, simple_value)| proto::Value {
                    tag: tag.into(),
                    simple_value,
                })
                .collect(),
        }),
    })
}

fn masked_crc(bytes: &[u8]) -> u32 {
    crc32c::crc32c(bytes).rotate_right(15).wrapping_add(0xA282EAD8)
}

fn write_record(writer: &mut impl Write, event: &proto::Event) -> Result<()> {
    let data = event.encode_to_vec();
    let length = (data.len() as u64).to_le_bytes();
    let mut record = Vec::with_capacity(data.len() + 16);

    record.extend(length);
    record.extend(masked_crc(&length).to_le_bytes());

    record.extend(&data);
    record.extend(masked_crc(&data).to_le_bytes());

    writer.write_all(&record).context("write TensorBoard record")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric() -> Event {
        Event::Metric {
            superbatch: 3,
            batch: 10,
            batches_per_superbatch: 100,
            final_superbatch: 5,
            loss: 0.1,
            learning_rate: 0.001,
            positions: 1000,
            total_positions: 21000,
            elapsed_seconds: 2.0,
        }
    }

    #[test]
    fn resumed_step_and_session_rate_are_not_confused() {
        let event = metric_event(&metric()).unwrap();
        assert_eq!(event.step, 210);

        let values = event.summary.unwrap().value;
        assert_eq!(
            values
                .iter()
                .find(|v| v.tag == "performance/mpos_per_second")
                .unwrap()
                .simple_value,
            0.0005
        );

        assert!(metric_event(&Event::Finished).is_none());
    }

    fn records(file: &Path) -> Vec<proto::Event> {
        let bytes = fs::read(file).unwrap();

        let mut offset = 0;
        let mut events = Vec::new();

        while offset < bytes.len() {
            let length = &bytes[offset..offset + 8];
            let size = u64::from_le_bytes(length.try_into().unwrap()) as usize;
            assert_eq!(
                masked_crc(length),
                u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap())
            );

            let data = &bytes[offset + 12..offset + 12 + size];
            assert_eq!(
                masked_crc(data),
                u32::from_le_bytes(
                    bytes[offset + 12 + size..offset + 16 + size].try_into().unwrap()
                )
            );

            events.push(proto::Event::decode(data).unwrap());
            offset += size + 16;
        }
        events
    }

    #[test]
    fn repeated_sessions_share_directory_and_preserve_readable_files() {
        let root = tempfile::tempdir().unwrap();
        let mut logger = TensorBoard::start(root.path(), "test-run").unwrap();
        let first = logger.directory.clone();
        logger.observe(&metric());

        assert_eq!(logger.finish().unwrap(), 0);
        assert_eq!(logger.finish().unwrap(), 0);

        assert_eq!(first, root.path().join("test-run"));

        let first_file = fs::read_dir(&first).unwrap().next().unwrap().unwrap().path();
        let events = records(&first_file);

        assert_eq!(events[0].file_version.as_deref(), Some("brain.Event:2"));
        assert_eq!(events[1].step, 210);

        let second;
        {
            let mut logger = TensorBoard::start(root.path(), "test-run").unwrap();
            second = logger.directory.clone();
            logger.observe(&metric());
        }

        assert_eq!(first, second);

        let files: Vec<_> = fs::read_dir(&second).unwrap().map(|f| f.unwrap().path()).collect();
        assert_eq!(files.len(), 2);

        for file in files {
            assert!(file.is_file());
            let events = records(&file);
            assert_eq!(events.len(), 2);
            assert_eq!(events[0].file_version.as_deref(), Some("brain.Event:2"));
            assert_eq!(events[1].step, 210);
        }
        assert!(TensorBoard::start(root.path(), "../escape").is_err());
    }

    #[test]
    fn saturated_and_failed_writers_never_block_submission() {
        let (sender, _receiver) = mpsc::sync_channel(1);

        let mut logger = TensorBoard {
            directory: PathBuf::new(),
            sender: Some(sender),
            worker: None,
            dropped: 0,
        };

        logger.observe(&metric());
        logger.observe(&metric());
        assert_eq!(logger.finish().unwrap(), 1);

        let (sender, receiver) = mpsc::sync_channel(1);

        drop(receiver);

        logger.sender = Some(sender);
        logger.observe(&metric());
        assert_eq!(logger.finish().unwrap(), 2);

        let mut buffer = [0u8; 1];
        let mut failing = std::io::Cursor::new(&mut buffer[..]);
        assert!(write_record(&mut failing, &metric_event(&metric()).unwrap()).is_err());
    }

    #[test]
    fn official_crc32c_check_vector() {
        assert_eq!(crc32c::crc32c(b"123456789"), 0xE3069283);
    }
}
