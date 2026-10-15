use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use prost::Message;

pub struct TensorBoard
{
    writer: BufWriter<File>,
}

impl TensorBoard
{
    pub fn new(root: &Path, name: &str) -> Result<Self>
    {
        ensure!(
            !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "Invalid run name"
        );

        let directory = root.join(name);
        fs::create_dir_all(&directory)?;

        let time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = directory.join(format!("events.out.tfevents.{time}.{}.bruce", std::process::id()));

        let mut writer = Self { writer: BufWriter::new(File::create_new(path)?) };
        writer.write(Event { file_version: Some("brain.Event:2".into()), ..Event::at(0) })?;
        writer.flush()?;

        Ok(writer)
    }

    pub fn sample(&mut self, step: i64, loss: f32, learning_rate: f32, speed: f32) -> Result<()>
    {
        let value =
            [("train/loss", loss), ("train/learning_rate", learning_rate), ("performance/mpos_per_second", speed)]
                .into_iter()
                .filter(|(_, value)| value.is_finite())
                .map(|(tag, simple_value)| Value { tag: tag.into(), simple_value })
                .collect();
        self.write(Event { summary: Some(Summary { value }), ..Event::at(step) })
    }

    pub fn flush(&mut self) -> Result<()> { self.writer.flush().context("Cannot flush TensorBoard events") }

    fn write(&mut self, event: Event) -> Result<()>
    {
        let bytes = event.encode_to_vec();
        let length = (bytes.len() as u64).to_le_bytes();

        self.writer.write_all(&length)?;
        self.writer.write_all(&checksum(&length))?;
        self.writer.write_all(&bytes)?;
        self.writer.write_all(&checksum(&bytes))?;

        Ok(())
    }
}

fn checksum(bytes: &[u8]) -> [u8; 4] { crc32c::crc32c(bytes).rotate_right(15).wrapping_add(0xa282_ead8).to_le_bytes() }

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

impl Event
{
    fn at(step: i64) -> Self
    {
        Self {
            wall_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64(),
            step,
            file_version: None,
            summary: None,
        }
    }
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
