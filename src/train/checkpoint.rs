use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Component, Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result, ensure};
use bullet_gpu::runtime::Gpu;
use bullet_trainer::{
    model::{ModelWeights, SavedFormat},
    optimiser::{Optimiser, OptimiserState},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::architecture::Architecture;
use crate::{
    BULLET_REVISION,
    config::{Config, numerical_configuration},
};

const STATE_FILES: [&str; 3] = ["weights.bin", "momentum.bin", "velocity.bin"];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Dataset
{
    path: PathBuf,
    length: u64,
    modified: u128,
}

impl Dataset
{
    fn read(path: &Path) -> Result<Self>
    {
        let path = fs::canonicalize(path).context("Cannot locate dataset")?;
        let metadata = fs::metadata(&path)?;
        Ok(Self { path, length: metadata.len(), modified: metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos() })
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct DigestEntry
{
    length: u64,
    sha256: String,
}

impl DigestEntry
{
    fn read(path: &Path) -> Result<Self>
    {
        let mut file = File::open(path).context("Checkpoint file is missing")?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        let mut length = 0;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }

            hasher.update(&buffer[..count]);
            length += count as u64;
        }

        Ok(Self { length, sha256: format!("{:x}", hasher.finalize()) })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest
{
    version: u32,
    bullet_revision: String,
    architecture: String,
    configuration: serde_json::Value,
    datasets: Vec<Dataset>,
    completed_superbatch: usize,
    files: BTreeMap<String, DigestEntry>,
}

impl Manifest
{
    fn read(path: &Path, architecture: Architecture) -> Result<Self>
    {
        let file = File::open(path.join("checkpoint.json")).context("Cannot read checkpoint")?;

        let manifest: Self = serde_json::from_reader(file).context("Invalid checkpoint")?;
        ensure!(manifest.version == 1, "Unsupported checkpoint version");
        ensure!(manifest.bullet_revision == BULLET_REVISION, "Checkpoint uses a different Bullet version");
        ensure!(
            manifest.architecture.eq_ignore_ascii_case(architecture.name()),
            "Checkpoint uses a different architecture"
        );

        Ok(manifest)
    }

    fn validate_files(&self, path: &Path, architecture: Architecture) -> Result<()>
    {
        for file in STATE_FILES {
            ensure!(self.files.contains_key(&format!("optimiser_state/{file}")), "Checkpoint is incomplete");
        }

        let prefix = format!("{}-", architecture.name().to_ascii_lowercase());
        ensure!(
            self.files.keys().any(|name| name.to_ascii_lowercase().starts_with(&prefix) && name.ends_with(".nnue")),
            "Checkpoint has no exported network"
        );

        for (name, expected) in &self.files {
            ensure!(
                Path::new(name).components().all(|part| matches!(part, Component::Normal(_))),
                "Invalid checkpoint filename"
            );
            ensure!(DigestEntry::read(&path.join(name))? == *expected, "Checkpoint is incomplete or corrupted");
        }

        validate_optimizer_state(&path.join("optimiser_state"), &ModelWeights::zeroed(&architecture.definition()))
    }
}

pub struct Checkpoints
{
    directory: PathBuf,
    name: String,
    architecture: Architecture,
    final_superbatch: usize,
    configuration: serde_json::Value,
    datasets: Vec<Dataset>,
}

impl Checkpoints
{
    pub fn new(config: &Config) -> Result<Self>
    {
        Ok(Self {
            directory: config.training.checkpoint.location.clone(),
            name: config.name.clone(),
            architecture: config.architecture.parse()?,
            final_superbatch: config.training.superbatches,
            configuration: config.checkpoint_value(),
            datasets: config.data.sources.iter().map(|source| Dataset::read(&source.path)).collect::<Result<_>>()?,
        })
    }

    pub fn inspect(&self, path: &Path) -> Result<usize>
    {
        let manifest = Manifest::read(path, self.architecture)?;
        ensure!(
            numerical_configuration(manifest.configuration.clone()) == self.configuration,
            "Configuration differs from the checkpoint; restore the original numerical training settings"
        );
        ensure!(manifest.datasets == self.datasets, "Datasets differ from the checkpoint");
        ensure!(manifest.completed_superbatch > 0, "Checkpoint has no completed training");
        ensure!(manifest.completed_superbatch < self.final_superbatch, "Training is already complete");

        manifest.validate_files(path, self.architecture)?;

        Ok(manifest.completed_superbatch)
    }

    pub fn destination(&self, superbatch: usize) -> PathBuf
    {
        self.directory.join(format!("{}-{superbatch}", self.name))
    }

    pub fn check_destination(&self, superbatch: usize) -> Result<()>
    {
        ensure!(!self.destination(superbatch).exists(), "Checkpoint already exists");

        Ok(())
    }

    pub fn save<G: Gpu, S: OptimiserState<G>>(
        &self,
        optimiser: &Optimiser<G, S>,
        format: &[SavedFormat],
        superbatch: usize,
    ) -> Result<PathBuf>
    {
        ensure!((1..=self.final_superbatch).contains(&superbatch), "Invalid checkpoint step");

        fs::create_dir_all(&self.directory).context("Cannot create checkpoint directory")?;

        let lock = self.directory.join(format!(".{}-{superbatch}.lock", self.name));
        fs::create_dir(&lock).context("Checkpoint is already being saved")?;

        let _reservation = Reservation(lock);

        self.check_destination(superbatch)?;

        let temporary = tempfile::Builder::new().prefix(".checkpoint-").tempdir_in(&self.directory)?;
        let state = temporary.path().join("optimiser_state");
        fs::create_dir(&state)?;

        checkpoint_operation("save", || {
            optimiser
                .write_to_checkpoint(state.to_str().context("Checkpoint path must be UTF-8")?)
                .map_err(|error| anyhow::anyhow!("Cannot save optimizer: {error:?}"))
        })?;

        let weights = optimiser.cpu_weights().map_err(|error| anyhow::anyhow!("Cannot read network: {error:?}"))?;
        validate_optimizer_state(&state, &weights)?;

        let bytes = weights.to_quantised_buffer(format, true).context("Cannot export network")?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let network = format!("{}-{}.nnue", self.architecture, &hash[..10]);
        fs::write(temporary.path().join(&network), bytes)?;

        let mut files = BTreeMap::new();
        for name in STATE_FILES.map(|file| format!("optimiser_state/{file}")).into_iter().chain([network]) {
            let path = temporary.path().join(&name);

            let digest = DigestEntry::read(&path)?;
            ensure!(digest.length > 0, "Checkpoint contains an empty file");

            File::options().write(true).open(path)?.sync_all()?;
            files.insert(name, digest);
        }

        let manifest = Manifest {
            version: 1,
            bullet_revision: BULLET_REVISION.to_owned(),
            architecture: self.architecture.name().to_owned(),
            configuration: self.configuration.clone(),
            datasets: self.datasets.clone(),
            completed_superbatch: superbatch,
            files,
        };
        {
            let mut file = File::create(temporary.path().join("checkpoint.json"))?;

            file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
            file.sync_all()?;
        }

        sync_directory(&state)?;
        sync_directory(temporary.path())?;

        self.check_destination(superbatch)?;
        let destination = self.destination(superbatch);

        fs::rename(temporary.path(), &destination).context("Cannot publish checkpoint")?;
        sync_directory(&self.directory)?;

        Ok(destination)
    }
}

pub fn restore<G: Gpu, S: OptimiserState<G>>(optimiser: &mut Optimiser<G, S>, path: &Path) -> Result<()>
{
    let state = path.join("optimiser_state");

    checkpoint_operation("restore", || {
        optimiser
            .load_from_checkpoint(state.to_str().context("Checkpoint path must be UTF-8")?)
            .map_err(|error| anyhow::anyhow!("Cannot restore optimizer: {error:?}"))
    })
}

pub fn load_weights(path: &Path, architecture: Architecture) -> Result<ModelWeights>
{
    Manifest::read(path, architecture)?.validate_files(path, architecture)?;

    let mut weights = ModelWeights::zeroed(&architecture.definition());

    let file = File::open(path.join("optimiser_state/weights.bin"))?;
    catch_unwind(AssertUnwindSafe(|| weights.load_from(file)))
        .map_err(|_| anyhow::anyhow!("Invalid network weights"))??;
    ensure!(
        weights.iter().all(|(_, tensor)| tensor.values.f32().iter().all(|value| value.is_finite())),
        "Network weights are incomplete or invalid"
    );

    Ok(weights)
}

fn checkpoint_operation<T>(action: &str, operation: impl FnOnce() -> Result<T>) -> Result<T>
{
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| anyhow::anyhow!("Cannot {action} optimizer"))?
}

fn validate_optimizer_state(directory: &Path, weights: &ModelWeights) -> Result<()>
{
    for name in STATE_FILES {
        validate_state(&directory.join(name), weights)?;
    }

    Ok(())
}

fn validate_state(path: &Path, weights: &ModelWeights) -> Result<()>
{
    let mut tensors: BTreeMap<_, _> =
        weights.iter().map(|(name, tensor)| (name.as_str(), tensor.values.size())).collect();

    let expected_size: usize =
        tensors.iter().map(|(name, count)| name.len() + 1 + size_of::<usize>() + count * 4).sum();

    let file = File::open(path)?;
    ensure!(file.metadata()?.len() == expected_size as u64, "Invalid checkpoint state");

    let mut bytes = Vec::with_capacity(expected_size);
    file.take(expected_size as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() == expected_size, "Invalid checkpoint state");

    let mut remaining = bytes.as_slice();
    while !remaining.is_empty() {
        let end = remaining.iter().position(|byte| *byte == b'\n').context("Invalid checkpoint state")?;
        let name = std::str::from_utf8(&remaining[..end]).context("Invalid checkpoint state")?;
        let count = tensors.remove(name).context("Invalid checkpoint tensor")?;
        remaining = &remaining[end + 1..];

        let size_bytes = remaining.get(..size_of::<usize>()).context("Invalid checkpoint state")?;
        ensure!(usize::from_le_bytes(size_bytes.try_into()?) == count, "Invalid checkpoint tensor size");

        remaining = &remaining[size_of::<usize>()..];
        let values = remaining.get(..count * 4).context("Invalid checkpoint state")?;
        ensure!(
            values.as_chunks::<4>().0.iter().all(|bytes| f32::from_le_bytes(*bytes).is_finite()),
            "Checkpoint contains invalid values"
        );

        remaining = &remaining[values.len()..];
    }

    ensure!(tensors.is_empty(), "Checkpoint is missing tensors");

    Ok(())
}

fn sync_directory(path: &Path) -> Result<()>
{
    #[cfg(unix)]
    File::open(path)?.sync_all()?;

    #[cfg(not(unix))]
    let _ = path;

    Ok(())
}

struct Reservation(PathBuf);

impl Drop for Reservation
{
    fn drop(&mut self) { let _ = fs::remove_dir(&self.0); }
}
