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

fn decode_events(bytes: &[u8]) -> Vec<Event>
{
    const LENGTH_BYTES: usize = 8;
    const CHECKSUM_BYTES: usize = 4;
    const HEADER_BYTES: usize = LENGTH_BYTES + CHECKSUM_BYTES;

    let mut cursor = 0;
    let mut events = Vec::new();

    while cursor < bytes.len() {
        let length_end = cursor + LENGTH_BYTES;
        let payload_start = cursor + HEADER_BYTES;
        let length = &bytes[cursor..length_end];
        let length_checksum = u32::from_le_bytes(bytes[length_end..payload_start].try_into().unwrap());
        assert_eq!(checksum(length), length_checksum);

        let payload_length = u64::from_le_bytes(length.try_into().unwrap()) as usize;
        let payload_end = payload_start + payload_length;
        let record_end = payload_end + CHECKSUM_BYTES;
        let payload = &bytes[payload_start..payload_end];
        let payload_checksum = u32::from_le_bytes(bytes[payload_end..record_end].try_into().unwrap());
        assert_eq!(checksum(payload), payload_checksum);

        events.push(Event::decode(payload).unwrap());
        cursor = record_end;
    }

    events
}

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
        let events = decode_events(&bytes);

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
    use bruce::logging::Logger;

    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("file");

    fs::write(&file, "occupied").unwrap();

    let log = root.path().join("fixture.log");
    let logger = Logger::open(&log, Some((&file, "fixture"))).unwrap();
    assert!(logger.take_diagnostics().iter().any(|message| message.contains("TensorBoard")));
    logger.record("Checkpoint saved").unwrap();
    logger.finish().unwrap();
    assert!(fs::read_to_string(&log).unwrap().contains("Checkpoint saved"));
}
