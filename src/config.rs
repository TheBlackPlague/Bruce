use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use bullet_trainer::optimiser::adam::AdamWParams;
use serde::{Deserialize, Deserializer, de::Error};
use serde_json::json;

use crate::{
    data::{Filter, LabelPolicy, Source},
    model::Architecture,
    schedule::{self, LrSchedule, WdlSchedule},
};

pub const DEFAULT_BATCHES_PER_SUPERBATCH: usize = 6104;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config
{
    schema: Schema,
    #[serde(rename = "config")]
    pub run: Run,
    pub random: Random,
    pub hardware: Hardware,
    pub training: Training,
    #[serde(skip)]
    pub source: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Schema
{
    version: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run
{
    pub name: String,
    #[serde(deserialize_with = "architecture")]
    pub arch: Architecture,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Random
{
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hardware
{
    pub gpu: i32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Training
{
    pub superbatches: usize,
    #[serde(default = "default_log_interval")]
    pub log_every: usize,
    #[serde(default)]
    pub metrics: Option<PathBuf>,
    pub checkpoint: Checkpoint,
    pub data: Data,
    pub optimizer: Optimizer,
    pub lr: LrSchedule,
    pub wdl: WdlSchedule,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Checkpoint
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

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval
{
    pub superbatch: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Data
{
    pub buffer: Buffer,
    pub threads: Threads,
    pub batch: Batch,
    #[serde(rename = "source", deserialize_with = "sources")]
    pub sources: Vec<Source>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub labels: LabelPolicy,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer
{
    pub size_mb: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Threads
{
    pub loader: usize,
    pub mapper: u8,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch
{
    pub size: usize,
    #[serde(default = "default_per_superbatch")]
    pub per_superbatch: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Optimizer
{
    pub minimum_weight: f32,
    pub maximum_weight: f32,
    pub adamw: AdamW,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdamW
{
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
}

fn default_log_interval() -> usize { 32 }

fn default_per_superbatch() -> usize { DEFAULT_BATCHES_PER_SUPERBATCH }

fn architecture<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Architecture, D::Error>
{
    String::deserialize(deserializer)?.parse().map_err(D::Error::custom)
}

fn sources<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Source>, D::Error>
{
    schedule::indexed(BTreeMap::deserialize(deserializer)?, "Data source").map_err(D::Error::custom)
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
        let parsed = toml::from_str(source).context("Invalid training configuration")?;
        let config = Self { source: source.to_owned(), ..parsed };
        config.validate()?;

        Ok(config)
    }

    pub fn checkpoint_value(&self) -> serde_json::Value
    {
        let (training, data) = (&self.training, &self.training.data);
        let (policy, interval) = match &training.checkpoint {
            Checkpoint::Periodic { periodic, .. } => ("Periodic", periodic),
            Checkpoint::Single { single, .. } => ("Single", single),
        };

        let mut value = json!({
            "name": self.run.name,
            "architecture": self.run.arch.name(),
            "seed": self.random.seed,
            "device": self.hardware.gpu,
            "training": {
                "superbatches": training.superbatches,
                "checkpoint": {
                    "location": training.checkpoint.location(),
                    "policy": { policy: { "superbatch": interval.superbatch } },
                },
            },
            "data": {
                "buffer_size_mb": data.buffer.size_mb,
                "loader_threads": data.threads.loader,
                "mapping_threads": data.threads.mapper,
                "batch_size": data.batch.size,
                "sources": data.sources,
                "filter": data.filter,
            },
            "optimizer": {
                "minimum_weight": training.optimizer.minimum_weight,
                "maximum_weight": training.optimizer.maximum_weight,
                "decay": training.optimizer.adamw.decay,
                "beta1": training.optimizer.adamw.beta1,
                "beta2": training.optimizer.adamw.beta2,
            },
            "learning_rate": training.lr,
            "wdl": training.wdl,
        });
        if data.batch.per_superbatch != DEFAULT_BATCHES_PER_SUPERBATCH {
            value["data"]["batches_per_superbatch"] = json!(data.batch.per_superbatch);
        }
        if !data.labels.is_keep() {
            value["data"]["labels"] = json!(data.labels);
        }

        value
    }

    pub fn phases(&self) -> Vec<(usize, usize)>
    {
        let mut starts: Vec<usize> = self.training.lr.starts().chain(self.training.wdl.starts()).collect();
        starts.sort_unstable();
        starts.dedup();

        let ends = starts.iter().skip(1).map(|next| next - 1).chain([self.training.superbatches]);
        starts.iter().copied().zip(ends).collect()
    }

    pub fn schedule_points(&self) -> Vec<(f32, f32)>
    {
        let (lr, wdl) = (self.training.lr.curve(), self.training.wdl.curve());
        (1..=self.training.superbatches).map(|superbatch| (lr(0, superbatch), wdl(0, superbatch))).collect()
    }

    pub fn validate(&self) -> Result<()>
    {
        let (name, training) = (&self.run.name, &self.training);
        let (superbatches, batch) = (training.superbatches, &training.data.batch);

        ensure!(self.schema.version == 1, "Unsupported configuration version");
        ensure!(
            !name.is_empty()
                && name.len() <= 100
                && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "Name must use 1 to 100 letters, numbers, '-' or '_'"
        );
        ensure!(self.hardware.gpu >= 0, "GPU number cannot be negative");

        ensure!(
            superbatches > 0 && superbatches < usize::MAX && training.log_every > 0,
            "Training counts must be positive"
        );
        (batch.size.checked_mul(batch.per_superbatch))
            .and_then(|count| count.checked_mul(superbatches))
            .context("Training is too large")?;

        ensure!(!training.checkpoint.location().as_os_str().is_empty(), "Checkpoint location cannot be empty");
        let (valid, message) = match &training.checkpoint {
            Checkpoint::Periodic { periodic, .. } => (periodic.superbatch > 0, "Checkpoint interval must be positive"),
            Checkpoint::Single { single, .. } => (
                single.superbatch > 0 && single.superbatch <= superbatches,
                "Checkpoint superbatch must fall within training",
            ),
        };
        ensure!(valid, message);
        ensure!(
            training.metrics.as_ref().is_none_or(|path| !path.as_os_str().is_empty()),
            "Metrics directory cannot be empty"
        );

        training.data.validate()?;

        let Optimizer { minimum_weight, maximum_weight, adamw: AdamW { decay, beta1, beta2 } } = training.optimizer;
        ensure!(decay.is_finite() && decay >= 0.0, "Weight decay must be finite and nonnegative");
        ensure!((0.0..1.0).contains(&beta1) && (0.0..1.0).contains(&beta2), "AdamW betas must be between 0 and 1");
        ensure!(
            minimum_weight.is_finite()
                && maximum_weight.is_finite()
                && minimum_weight < maximum_weight
                && minimum_weight >= -2.0
                && maximum_weight <= 2.0,
            "Weight limits must be ordered within [-2, 2]"
        );

        training.lr.validate(superbatches, batch.per_superbatch)?;
        training.wdl.validate(superbatches, batch.per_superbatch)
    }
}

impl Checkpoint
{
    pub fn location(&self) -> &Path
    {
        let (Self::Periodic { location, .. } | Self::Single { location, .. }) = self;
        location
    }

    pub fn destination(&self, name: &str, superbatch: usize) -> PathBuf
    {
        self.location().join(format!("{name}-{superbatch}"))
    }

    pub fn reason(&self, superbatch: usize, last: usize, stopping: bool) -> Option<&'static str>
    {
        match self {
            _ if superbatch == last => Some("final"),
            _ if stopping => Some("stop"),
            Self::Periodic { periodic, .. } => {
                (superbatch > 0 && superbatch.is_multiple_of(periodic.superbatch)).then_some("scheduled")
            },
            Self::Single { single, .. } => (superbatch == single.superbatch).then_some("scheduled"),
        }
    }
}

impl Data
{
    pub fn validate(&self) -> Result<()>
    {
        ensure!(self.batch.size > 0, "Batch size must be positive");
        ensure!(self.batch.per_superbatch > 0, "Batches per superbatch must be positive");
        ensure!((1..=1_048_576).contains(&self.buffer.size_mb), "Buffer size must be between 1 and 1048576 MB");
        ensure!(
            (1..=256).contains(&self.threads.loader) && self.threads.mapper > 0,
            "Loader and mapper counts must be positive; at most 256 loaders are supported"
        );
        ensure!(!self.sources.is_empty(), "At least one data source is required");

        if let Some(agreement) = self.labels.agreement {
            ensure!((0.0..=4000.0).contains(&agreement.offset), "Agreement offset must be between 0 and 4000");
            ensure!((20.0..=4000.0).contains(&agreement.spread), "Agreement spread must be between 20 and 4000");
        }

        if let Some(filter) = &self.filter {
            ensure!(
                filter.minimum_eval < filter.maximum_eval && filter.maximum_eval <= 32768,
                "Evaluation bounds must be ordered between 0 and 32768"
            );
            ensure!(filter.minimum_pieces <= 32, "Minimum piece count cannot exceed 32");
            ensure!(filter.maximum_ply.is_none_or(|maximum| maximum >= filter.minimum_ply), "Ply bounds are reversed");
            ensure!((0.0..1.0).contains(&filter.random_skip), "Random skip must be at least 0 and below 1");
        }

        for source in &self.sources {
            ensure!(
                !source.path.as_os_str().is_empty() && source.path.to_str().is_some(),
                "Dataset path must be nonempty UTF-8"
            );
            ensure!((0.1..=10.0).contains(&source.scale), "Source scale must be between 0.1 and 10");
        }

        Ok(())
    }
}

impl Optimizer
{
    pub fn params(&self) -> AdamWParams
    {
        AdamWParams {
            min_weight: self.minimum_weight,
            max_weight: self.maximum_weight,
            decay: self.adamw.decay,
            beta1: self.adamw.beta1,
            beta2: self.adamw.beta2,
        }
    }
}
