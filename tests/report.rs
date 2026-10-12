use std::fs;

use bruce::report::TensorBoard;
use prost::Message;

#[derive(Clone, PartialEq, Message)]
struct Event
{
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(string, optional, tag = "3")]
    version: Option<String>,
    #[prost(message, optional, tag = "5")]
    summary: Option<Summary>,
}

#[derive(Clone, PartialEq, Message)]
struct Summary
{
    #[prost(message, repeated, tag = "1")]
    values: Vec<Value>,
}

#[derive(Clone, PartialEq, Message)]
struct Value
{
    #[prost(string, tag = "1")]
    tag: String,
    #[prost(float, tag = "2")]
    value: f32,
}

fn checksum(bytes: &[u8]) -> u32 { crc32c::crc32c(bytes).rotate_right(15).wrapping_add(0xa282_ead8) }

#[test]
fn tensorboard_sessions_preserve_steps_tags_and_valid_records()
{
    let root = tempfile::tempdir().unwrap();

    for _ in 0..2 {
        let mut writer = TensorBoard::new(root.path(), "fixture").unwrap();
        writer.sample(12209, 0.125, 0.001, 1.25).unwrap();
        writer.sample(12210, f32::NAN, 0.002, 1.5).unwrap();
        writer.flush().unwrap();
    }

    let files = fs::read_dir(root.path().join("fixture")).unwrap().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(files.len(), 2);

    for file in files {
        let bytes = fs::read(file.path()).unwrap();
        let mut cursor = 0;
        let mut events = Vec::new();

        while cursor < bytes.len() {
            let length = &bytes[cursor..cursor + 8];
            assert_eq!(checksum(length), u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap()));

            let size = u64::from_le_bytes(length.try_into().unwrap()) as usize;
            let data = &bytes[cursor + 12..cursor + 12 + size];
            assert_eq!(
                checksum(data),
                u32::from_le_bytes(bytes[cursor + 12 + size..cursor + 16 + size].try_into().unwrap())
            );

            events.push(Event::decode(data).unwrap());
            cursor += size + 16;
        }

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].version.as_deref(), Some("brain.Event:2"));
        assert_eq!(events[1].step, 12209);

        let values = &events[1].summary.as_ref().unwrap().values;

        assert_eq!(
            values.iter().map(|value| value.tag.as_str()).collect::<Vec<_>>(),
            ["train/loss", "train/learning_rate", "performance/mpos_per_second"]
        );
        assert_eq!(values[0].value, 0.125);
        assert_eq!(events[2].summary.as_ref().unwrap().values.len(), 2);
    }
}

#[test]
fn tensorboard_rejects_unwritable_destination_and_invalid_name()
{
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("file");

    fs::write(&file, "occupied").unwrap();

    assert!(TensorBoard::new(&file, "fixture").is_err());
    assert!(TensorBoard::new(root.path(), "../escape").is_err());
}

#[test]
fn unavailable_tensorboard_does_not_prevent_training_reporting()
{
    use bruce::{config::Config, report::Reporter};
    use bullet_trainer::run::TrainingSteps;

    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("file");

    fs::write(&file, "occupied").unwrap();

    let mut config = Config::from_toml(include_str!("../config/training.toml.default")).unwrap();
    config.name = format!("report-test-{}", std::process::id());
    config.tensorboard.enabled = true;
    config.tensorboard.path = file;

    let steps = TrainingSteps { batch_size: 16, batches_per_superbatch: 6104, start_superbatch: 3, end_superbatch: 4 };

    let mut reporter = Reporter::new(&config, steps, true).unwrap();
    reporter.message("Checkpoint saved");
    reporter.finish().unwrap();

    let log = std::path::Path::new("logs").join(format!("{}.log", config.name));
    assert!(fs::read_to_string(&log).unwrap().contains("Checkpoint saved"));

    fs::remove_file(log).unwrap();
}
