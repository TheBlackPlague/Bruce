use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use bullet_trainer::optimiser::adam::AdamWParams;
use serde::{Deserialize, Serialize};

use crate::{
    data::DataConfig,
    schedule::{LrConfig, WdlConfig},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub name: String,
    #[serde(default = "architecture")]
    pub architecture: String,
    pub output_directory: PathBuf,
    #[serde(default = "seed")]
    pub seed: u64,
    #[serde(default)]
    pub device: i32,
    pub training: TrainingConfig,
    pub data: DataConfig,
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    pub learning_rate: LrConfig,
    pub wdl: WdlConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub superbatches: usize,
    pub save_every: usize,
    #[serde(default = "log_every")]
    pub log_every: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct OptimizerConfig {
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub min_weight: f32,
    pub max_weight: f32,
}

fn architecture() -> String {
    "Aurora".into()
}

fn seed() -> u64 {
    42
}

fn log_every() -> usize {
    32
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        let p = AdamWParams::default();
        Self {
            decay: p.decay,
            beta1: p.beta1,
            beta2: p.beta2,
            min_weight: p.min_weight,
            max_weight: p.max_weight,
        }
    }
}

impl OptimizerConfig {
    pub fn params(&self) -> AdamWParams {
        AdamWParams {
            decay: self.decay,
            beta1: self.beta1,
            beta2: self.beta2,
            min_weight: self.min_weight,
            max_weight: self.max_weight,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let source = fs::read_to_string(path)
            .with_context(|| format!("Cannot read configuration {}", path.display()))?;
        let mut config: Self = toml::from_str(&source).context("Invalid training TOML")?;

        let parent = fs::canonicalize(path)?
            .parent()
            .context("Configuration has no parent directory")?
            .to_path_buf();

        if config.output_directory.is_relative() {
            config.output_directory = parent.join(&config.output_directory);
        }

        for path in &mut config.data.paths {
            if path.is_relative() {
                *path = parent.join(&*path);
            }
        }

        config.validate()?;

        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "Unsupported schema_version {}; expected 1",
            self.schema_version
        );
        ensure!(
            self.architecture == "Aurora",
            "Only the Aurora chess architecture is supported"
        );
        ensure!(
            !self.name.is_empty()
                && self.name.len() <= 100
                && self
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "name must use 1–100 ASCII letters, digits, '-' or '_'"
        );
        ensure!(self.device >= 0, "device must be a nonnegative GPU ordinal");

        let t = &self.training;

        ensure!(
            t.batch_size > 0
                && t.batches_per_superbatch > 0
                && t.superbatches > 0
                && t.save_every > 0
                && t.log_every > 0,
            "All training counts must be greater than zero"
        );

        t.batch_size
            .checked_mul(t.batches_per_superbatch)
            .and_then(|n| n.checked_mul(t.superbatches))
            .context("Training position count overflows this platform")?;

        ensure!(t.superbatches < usize::MAX, "superbatches is too large");
        ensure!(
            !self.output_directory.as_os_str().is_empty(),
            "output_directory cannot be empty"
        );
        ensure!(
            !self.data.paths.is_empty(),
            "At least one dataset path is required"
        );
        ensure!(
            self.data.loader_threads > 0
                && self.data.mapping_threads > 0
                && self.data.buffer_size_mb > 0,
            "Loader threads, mapping threads, and buffer_size_mb must be positive"
        );

        self.data
            .buffer_size_mb
            .checked_mul(1024 * 1024)
            .context("buffer_size_mb overflows this platform")?;

        let p = &self.optimizer;

        ensure!(
            p.decay.is_finite() && p.decay >= 0.0,
            "optimizer.decay must be finite and nonnegative"
        );
        ensure!(
            p.beta1.is_finite()
                && (0.0..1.0).contains(&p.beta1)
                && p.beta2.is_finite()
                && (0.0..1.0).contains(&p.beta2),
            "AdamW betas must be finite and in [0, 1)"
        );
        ensure!(
            p.min_weight.is_finite() && p.max_weight.is_finite() && p.min_weight < p.max_weight,
            "Invalid optimizer clipping range"
        );
        ensure!(
            p.min_weight >= -2.0 && p.max_weight <= 2.0,
            "Aurora's i16 output bias requires clipping within [-2, 2]"
        );

        self.data         .validate(              )?;
        self.learning_rate.validate(t.superbatches)?;
        self.wdl          .validate(t.superbatches)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //noinspection SpellCheckingInspection
    #[test]
    fn example_and_strict_toml() {
        let source = include_str!("../config/training.toml");
        let config: Config = toml::from_str(source).unwrap();

        config.validate().unwrap();

        assert!(toml::from_str::<Config>(&format!("{source}\n[training]\nbatch_size=8")).is_err());
        assert!(toml::from_str::<Config>(&source.replace("batch_size =", "batch_szie =")).is_err());
    }

    #[test]
    fn rejects_invalid_numeric_settings() {
        let base: Config = toml::from_str(include_str!("../config/training.toml")).unwrap();
        let mut config = base.clone();

        config.training.batch_size = 0;

        assert!(config.validate().is_err());

        config = base.clone();
        config.optimizer.beta2 = f32::NAN;

        assert!(config.validate().is_err());

        config = base;
        config.training.superbatches = usize::MAX;

        assert!(config.validate().is_err());
    }
}
