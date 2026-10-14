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

use crate::{
    BULLET_REVISION,
    architecture::Architecture,
    config::{Config, numerical_configuration},
};

const STATE_FILES: [&str; 3] = ["weights.bin", "momentum.bin", "velocity.bin"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset
{
    path: PathBuf,
    length: u64,
    modified: u128,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DigestEntry
{
    length: u64,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
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
        let manifest: Self =
            serde_json::from_reader(File::open(path.join("checkpoint.json")).context("Cannot read checkpoint")?)
                .context("Invalid checkpoint")?;

        ensure!(manifest.version == 1, "Unsupported checkpoint version");
        ensure!(manifest.bullet_revision == BULLET_REVISION, "Checkpoint uses a different Bullet version");
        ensure!(
            manifest.architecture.eq_ignore_ascii_case(architecture.name()),
            "Checkpoint uses a different architecture"
        );
        Ok(manifest)
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
        let datasets = config.data.sources.iter().map(|source| dataset(&source.path)).collect::<Result<Vec<_>>>()?;

        Ok(Self {
            directory: config.training.checkpoint.location.clone(),
            name: config.name.clone(),
            architecture: config.architecture.parse()?,
            final_superbatch: config.training.superbatches,
            configuration: config.checkpoint_value(),
            datasets,
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

        validate_files(path, &manifest, self.architecture)?;
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
        ensure!(superbatch > 0 && superbatch <= self.final_superbatch, "Invalid checkpoint step");
        fs::create_dir_all(&self.directory).context("Cannot create checkpoint directory")?;

        let reservation = self.directory.join(format!(".{}-{superbatch}.lock", self.name));
        fs::create_dir(&reservation).context("Checkpoint is already being saved")?;
        let _reservation = Reservation(reservation);
        self.check_destination(superbatch)?;

        let temporary = tempfile::Builder::new().prefix(".checkpoint-").tempdir_in(&self.directory)?;
        let state = temporary.path().join("optimiser_state");

        fs::create_dir(&state)?;

        catch_unwind(AssertUnwindSafe(|| {
            optimiser
                .write_to_checkpoint(state.to_str().context("Checkpoint path must be UTF-8")?)
                .map_err(|error| anyhow::anyhow!("Cannot save optimizer: {error:?}"))
        }))
        .map_err(|_| anyhow::anyhow!("Cannot save optimizer"))??;

        let weights = optimiser.cpu_weights().map_err(|error| anyhow::anyhow!("Cannot read network: {error:?}"))?;
        for name in STATE_FILES {
            validate_state(&state.join(name), &weights)?;
        }
        let bytes = weights.to_quantised_buffer(format, true).context("Cannot export network")?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let network = format!("{}-{}.nnue", self.architecture, &hash[..10]);
        fs::write(temporary.path().join(&network), bytes)?;

        let mut files = BTreeMap::new();
        for name in STATE_FILES.iter().map(|name| format!("optimiser_state/{name}")).chain(std::iter::once(network)) {
            let path = temporary.path().join(&name);
            let entry = digest(&path)?;
            ensure!(entry.length > 0, "Checkpoint contains an empty file");
            File::options().write(true).open(path)?.sync_all()?;
            files.insert(name, entry);
        }

        let manifest = Manifest {
            version: 1,
            bullet_revision: BULLET_REVISION.into(),
            architecture: self.architecture.name().into(),
            configuration: self.configuration.clone(),
            datasets: self.datasets.clone(),
            completed_superbatch: superbatch,
            files,
        };
        let mut file = File::create(temporary.path().join("checkpoint.json"))?;
        file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
        file.sync_all()?;
        drop(file);

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
    let path = path.join("optimiser_state");
    catch_unwind(AssertUnwindSafe(|| {
        optimiser
            .load_from_checkpoint(path.to_str().context("Checkpoint path must be UTF-8")?)
            .map_err(|error| anyhow::anyhow!("Cannot restore optimizer: {error:?}"))
    }))
    .map_err(|_| anyhow::anyhow!("Cannot restore optimizer"))?
}

pub fn load_weights(path: &Path, architecture: Architecture) -> Result<ModelWeights>
{
    let manifest = Manifest::read(path, architecture)?;
    validate_files(path, &manifest, architecture)?;

    let mut weights = ModelWeights::zeroed(&architecture.definition());
    let file = File::open(path.join("optimiser_state/weights.bin"))?;
    catch_unwind(AssertUnwindSafe(|| weights.load_from(file)))
        .map_err(|_| anyhow::anyhow!("Invalid network weights"))??;

    ensure!(
        weights.iter().all(|(_, value)| value.values.f32().iter().all(|value| value.is_finite())),
        "Network weights are incomplete or invalid"
    );
    Ok(weights)
}

fn validate_files(path: &Path, manifest: &Manifest, architecture: Architecture) -> Result<()>
{
    for name in STATE_FILES {
        ensure!(manifest.files.contains_key(&format!("optimiser_state/{name}")), "Checkpoint is incomplete");
    }

    let network_prefix = format!("{}-", architecture.name().to_ascii_lowercase());
    let has_exported_network = manifest
        .files
        .keys()
        .any(|name| name.to_ascii_lowercase().starts_with(&network_prefix) && name.ends_with(".nnue"));
    ensure!(has_exported_network, "Checkpoint has no exported network");

    for (name, expected) in &manifest.files {
        ensure!(
            Path::new(name).components().all(|part| matches!(part, Component::Normal(_))),
            "Invalid checkpoint filename"
        );
        ensure!(digest(&path.join(name))? == *expected, "Checkpoint is incomplete or corrupted");
    }

    let weights = ModelWeights::zeroed(&architecture.definition());
    for name in STATE_FILES {
        validate_state(&path.join("optimiser_state").join(name), &weights)?;
    }

    Ok(())
}

fn validate_state(path: &Path, weights: &ModelWeights) -> Result<()>
{
    let mut expected: BTreeMap<_, _> =
        weights.iter().map(|(name, value)| (name.as_str(), value.values.size())).collect();
    
    let size: usize = expected
        .iter()
        .map(|(name, count)| {
            let name_bytes = name.len() + 1;
            let count_bytes = size_of::<usize>();
            let value_bytes = count * size_of::<f32>();

            name_bytes + count_bytes + value_bytes
        })
        .sum();

    let file = File::open(path)?;
    ensure!(file.metadata()?.len() == size as u64, "Invalid checkpoint state");

    let mut bytes = Vec::with_capacity(size);
    file.take(size as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() == size, "Invalid checkpoint state");

    let mut remaining = bytes.as_slice();
    while !remaining.is_empty() {
        let name_end = remaining.iter().position(|byte| *byte == b'\n').context("Invalid checkpoint state")?;

        let name = std::str::from_utf8(&remaining[..name_end]).context("Invalid checkpoint state")?;
        let count = expected.remove(name).context("Invalid checkpoint tensor")?;

        remaining = &remaining[name_end + 1..];
        let count_bytes = remaining.get(..size_of::<usize>()).context("Invalid checkpoint state")?;
        ensure!(usize::from_le_bytes(count_bytes.try_into()?) == count, "Invalid checkpoint tensor size");

        remaining = &remaining[size_of::<usize>()..];
        let values = remaining.get(..count * size_of::<f32>()).context("Invalid checkpoint state")?;
        let all_finite = values.as_chunks::<4>().0.iter().all(|value| f32::from_le_bytes(*value).is_finite());
        ensure!(all_finite, "Checkpoint contains invalid values");

        remaining = &remaining[values.len()..];
    }

    ensure!(expected.is_empty(), "Checkpoint is missing tensors");
    Ok(())
}

fn dataset(path: &Path) -> Result<Dataset>
{
    let path = fs::canonicalize(path).context("Cannot locate dataset")?;
    let metadata = fs::metadata(&path)?;
    Ok(Dataset { path, length: metadata.len(), modified: metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos() })
}

fn digest(path: &Path) -> Result<DigestEntry>
{
    let mut file = File::open(path).context("Checkpoint file is missing")?;
    let mut hash = Sha256::new();
    let mut length = 0;
    let mut buffer = [0u8; 65536];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        length += count as u64;
    }

    Ok(DigestEntry { length, sha256: format!("{:x}", hash.finalize()) })
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
