use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use bullet_trainer::optimiser::adam::AdamWParams;
use serde::{Deserialize, Serialize};

use self::schedule::{BATCHES_PER_SUPERBATCH, LrSchedule, LrSpec, Phase, WdlSchedule, WdlSpec};
use crate::{architecture::Architecture, data::Format};

pub mod schedule;

#[derive(Clone, Debug)]
pub struct Config
{
    pub name: String,
    pub architecture: String,
    pub seed: u64,
    pub device: i32,
    pub tensorboard: TensorboardConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
    pub optimizer: OptimizerConfig,
    pub learning_rate: LrSchedule,
    pub wdl: WdlSchedule,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingConfig
{
    pub superbatches: usize,
    pub log_every: usize,
    pub checkpoint: CheckpointConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorboardConfig
{
    pub enabled: bool,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(from = "CheckpointFile")]
pub struct CheckpointConfig
{
    pub location: PathBuf,
    pub policy: CheckpointPolicy,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum CheckpointPolicy
{
    Periodic
    {
        superbatch: usize
    },
    Single
    {
        superbatch: usize
    },
}

impl CheckpointPolicy
{
    pub fn should_save(self, superbatch: usize) -> bool
    {
        match self {
            Self::Periodic { superbatch: period } => superbatch > 0 && period > 0 && superbatch.is_multiple_of(period),
            Self::Single { superbatch: target } => superbatch == target,
        }
    }

    pub fn due(self, superbatch: usize, final_superbatch: usize, stopping: bool) -> bool
    {
        self.should_save(superbatch)
            || matches!(self, Self::Periodic { .. }) && (superbatch == final_superbatch || stopping)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(try_from = "DataFile")]
pub struct DataConfig
{
    pub buffer_size_mb: usize,
    pub loader_threads: usize,
    pub mapping_threads: u8,
    pub batch_size: usize,
    pub sources: Vec<DataSource>,
    pub filter: Option<Filter>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataSource
{
    pub format: Format,
    pub path: PathBuf,
    #[serde(default)]
    pub separator: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Filter
{
    pub minimum_ply: u16,
    pub maximum_ply: Option<u16>,
    pub minimum_pieces: u32,
    pub minimum_eval: u32,
    pub maximum_eval: u32,
    pub exclude_tactical: bool,
    pub exclude_check: bool,
}

impl Default for Filter
{
    fn default() -> Self
    {
        Self {
            minimum_ply: 0,
            maximum_ply: None,
            minimum_pieces: 0,
            minimum_eval: 0,
            maximum_eval: 32768,
            exclude_tactical: false,
            exclude_check: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(from = "OptimizerFile")]
pub struct OptimizerConfig
{
    pub minimum_weight: f32,
    pub maximum_weight: f32,
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
}

impl OptimizerConfig
{
    pub fn params(&self) -> AdamWParams
    {
        AdamWParams {
            min_weight: self.minimum_weight,
            max_weight: self.maximum_weight,
            decay: self.decay,
            beta1: self.beta1,
            beta2: self.beta2,
        }
    }

    fn validate(&self) -> Result<()>
    {
        ensure!(self.decay.is_finite() && self.decay >= 0.0, "Weight decay must be finite and nonnegative");
        ensure!(
            (0.0..1.0).contains(&self.beta1) && (0.0..1.0).contains(&self.beta2),
            "AdamW betas must be between 0 and 1"
        );
        ensure!(
            self.minimum_weight.is_finite()
                && self.maximum_weight.is_finite()
                && self.minimum_weight < self.maximum_weight
                && self.minimum_weight >= -2.0
                && self.maximum_weight <= 2.0,
            "Weight limits must be ordered within [-2, 2]"
        );

        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile
{
    schema: Schema,
    config: Identity,
    random: Random,
    hardware: Hardware,
    tensorboard: TensorboardConfig,
    training: TrainingFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Schema
{
    version: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity
{
    name: String,
    arch: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Random
{
    seed: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Hardware
{
    gpu: i32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingFile
{
    superbatches: usize,
    #[serde(default = "default_log_interval")]
    log_every: usize,
    checkpoint: CheckpointConfig,
    data: DataConfig,
    optimizer: OptimizerConfig,
    lr: BTreeMap<String, Phase<LrSpec>>,
    wdl: BTreeMap<String, Phase<WdlSpec>>,
}

fn default_log_interval() -> usize { 32 }

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum CheckpointFile
{
    Periodic
    {
        location: PathBuf, periodic: Interval
    },
    Single
    {
        location: PathBuf, single: Interval
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Interval
{
    superbatch: usize,
}

impl From<CheckpointFile> for CheckpointConfig
{
    fn from(value: CheckpointFile) -> Self
    {
        match value {
            CheckpointFile::Periodic { location, periodic } => {
                Self { location, policy: CheckpointPolicy::Periodic { superbatch: periodic.superbatch } }
            },
            CheckpointFile::Single { location, single } => {
                Self { location, policy: CheckpointPolicy::Single { superbatch: single.superbatch } }
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DataFile
{
    buffer: Buffer,
    threads: Threads,
    batch: Batch,
    source: BTreeMap<String, DataSource>,
    #[serde(default)]
    filter: Option<Filter>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Buffer
{
    size_mb: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Threads
{
    loader: usize,
    mapper: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Batch
{
    size: usize,
}

impl TryFrom<DataFile> for DataConfig
{
    type Error = anyhow::Error;

    fn try_from(value: DataFile) -> Result<Self>
    {
        Ok(Self {
            buffer_size_mb: value.buffer.size_mb,
            loader_threads: value.threads.loader,
            mapping_threads: value.threads.mapper,
            batch_size: value.batch.size,
            sources: schedule::indexed(value.source, "Data source")?,
            filter: value.filter,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OptimizerFile
{
    minimum_weight: f32,
    maximum_weight: f32,
    adamw: AdamW,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AdamW
{
    decay: f32,
    beta1: f32,
    beta2: f32,
}

impl From<OptimizerFile> for OptimizerConfig
{
    fn from(value: OptimizerFile) -> Self
    {
        Self {
            minimum_weight: value.minimum_weight,
            maximum_weight: value.maximum_weight,
            decay: value.adamw.decay,
            beta1: value.adamw.beta1,
            beta2: value.adamw.beta2,
        }
    }
}

impl Config
{
    pub fn load(path: &Path) -> Result<Self>
    {
        let source = fs::read_to_string(path).with_context(|| format!("Cannot read {}", path.display()))?;
        Self::from_toml(&source)
    }

    pub fn from_toml(source: &str) -> Result<Self>
    {
        let file: ConfigFile = toml::from_str(source).context("Invalid training configuration")?;
        ensure!(file.schema.version == 1, "Unsupported configuration version");

        let training = file.training;
        ensure!(training.superbatches > 0 && training.superbatches < usize::MAX, "Training counts must be positive");

        let config = Self {
            name: file.config.name,
            architecture: file.config.arch,
            seed: file.random.seed,
            device: file.hardware.gpu,
            tensorboard: file.tensorboard,
            training: TrainingConfig {
                superbatches: training.superbatches,
                log_every: training.log_every,
                checkpoint: training.checkpoint,
            },
            data: training.data,
            optimizer: training.optimizer,
            learning_rate: LrSchedule::new(training.lr, training.superbatches)?,
            wdl: WdlSchedule::new(training.wdl, training.superbatches)?,
        };
        config.validate()?;

        Ok(config)
    }

    pub fn checkpoint_value(&self) -> serde_json::Value
    {
        numerical_configuration(serde_json::json!({
            "name": self.name,
            "architecture": self.architecture,
            "seed": self.seed,
            "device": self.device,
            "training": self.training,
            "data": self.data,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "wdl": self.wdl,
        }))
    }

    pub fn validate(&self) -> Result<()>
    {
        ensure!(
            !self.name.is_empty()
                && self.name.len() <= 100
                && self.name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "Name must use 1 to 100 letters, numbers, '-' or '_'"
        );
        self.architecture.parse::<Architecture>()?;

        ensure!(self.device >= 0, "GPU number cannot be negative");

        ensure!(
            self.training.superbatches > 0 && self.training.superbatches < usize::MAX && self.training.log_every > 0,
            "Training counts must be positive"
        );
        self.data
            .batch_size
            .checked_mul(BATCHES_PER_SUPERBATCH)
            .and_then(|count| count.checked_mul(self.training.superbatches))
            .context("Training is too large")?;

        ensure!(!self.training.checkpoint.location.as_os_str().is_empty(), "Checkpoint location cannot be empty");
        match self.training.checkpoint.policy {
            CheckpointPolicy::Periodic { superbatch } => {
                ensure!(superbatch > 0, "Checkpoint interval must be positive")
            },
            CheckpointPolicy::Single { superbatch } => ensure!(
                superbatch > 0 && superbatch <= self.training.superbatches,
                "Checkpoint superbatch must fall within training"
            ),
        }

        ensure!(
            !self.tensorboard.enabled || !self.tensorboard.path.as_os_str().is_empty(),
            "TensorBoard path cannot be empty"
        );

        self.data.validate()?;
        self.optimizer.validate()
    }
}

pub(crate) fn numerical_configuration(mut value: serde_json::Value) -> serde_json::Value
{
    if let Some(training) = value.get_mut("training").and_then(serde_json::Value::as_object_mut) {
        training.remove("log_every");
    }

    value
}

impl DataConfig
{
    pub fn validate(&self) -> Result<()>
    {
        ensure!(self.batch_size > 0, "Batch size must be positive");
        self.batch_size.checked_mul(BATCHES_PER_SUPERBATCH).context("Batch size is too large")?;

        ensure!((1..=1_048_576).contains(&self.buffer_size_mb), "Buffer size must be between 1 and 1048576 MB");
        self.buffer_size_mb.checked_mul(1024 * 1024).context("Buffer is too large")?;

        ensure!(
            (1..=256).contains(&self.loader_threads) && self.mapping_threads > 0,
            "Loader and mapper counts must be positive; at most 256 loaders are supported"
        );
        ensure!(!self.sources.is_empty(), "At least one data source is required");

        if let Some(filter) = &self.filter {
            ensure!(
                filter.minimum_eval < filter.maximum_eval && filter.maximum_eval <= 32768,
                "Evaluation bounds must be ordered between 0 and 32768"
            );
            ensure!(filter.minimum_pieces <= 32, "Minimum piece count cannot exceed 32");
            ensure!(filter.maximum_ply.is_none_or(|maximum| maximum >= filter.minimum_ply), "Ply bounds are reversed");
        }

        for source in &self.sources {
            ensure!(
                !source.path.as_os_str().is_empty() && source.path.to_str().is_some(),
                "Dataset path must be nonempty UTF-8"
            );

            match (&source.format, &source.separator) {
                (Format::Text, Some(separator)) => ensure!(
                    !separator.is_empty() && !separator.contains(['\n', '\r']),
                    "Text separator must be nonempty and stay on one line"
                ),
                (Format::Text, None) => anyhow::bail!("Text sources need a separator"),
                (_, Some(_)) => anyhow::bail!("Separators are only used with text sources"),
                _ => {},
            }

            if let Some(filter) = &self.filter {
                match source.format {
                    Format::Bullet => ensure!(
                        filter.minimum_ply == 0
                            && filter.maximum_ply.is_none()
                            && !filter.exclude_tactical
                            && !filter.exclude_check,
                        "These filters need game metadata unavailable in Bullet data"
                    ),
                    Format::Text => ensure!(
                        !filter.exclude_tactical,
                        "Tactical filtering needs the played move, unavailable in text data"
                    ),
                    _ => {},
                }
            }
        }

        Ok(())
    }
}
