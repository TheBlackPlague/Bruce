use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufReader, Read},
    path::{Component, Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use bullet_trainer::model::ModelWeights;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::Trainer;
use crate::{
    config::Config,
    control::contain,
    data::{SourceIdentity, stream::Bookmark},
    model::Architecture,
};

pub const BULLET_REVISION: &str = "1896c0884c7ededb1868068bbf5583d1726b37f6";
pub const MANIFEST: &str = "checkpoint.json";
pub const LEGACY: u32 = 1;
pub const VERSION: u32 = 2;
pub const STATE: &str = "optimiser_state";

const WEIGHTS: &str = "weights.bin";
pub const STATE_FILES: [&str; 3] = [WEIGHTS, "momentum.bin", "velocity.bin"];

const PUBLISH_RETRIES: u32 = 10;
const PUBLISH_PAUSE: Duration = Duration::from_millis(250);

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DigestEntry
{
    pub length: u64,
    pub sha256: String,
}

impl DigestEntry
{
    pub fn read(path: &Path) -> io::Result<Self>
    {
        let mut hasher = Sha256::new();
        let length = io::copy(&mut File::open(path)?, &mut hasher)?;

        Ok(Self { length, sha256: format!("{:x}", hasher.finalize()) })
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest
{
    pub version: u32,
    pub bullet_revision: String,
    pub architecture: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<StepsRecord>,
    pub completed_superbatch: usize,
    pub configuration: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub datasets: Datasets,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<DataRecord>,
    #[serde(default, skip_serializing)]
    pub validation: Option<Value>,
    pub files: BTreeMap<String, DigestEntry>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StepsRecord
{
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Datasets
{
    Identified(Vec<SourceIdentity>),
    Legacy(Vec<DatasetV1>),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetV1
{
    path: PathBuf,
    length: u64,
    modified: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct DataRecord
{
    pub bookmark: Bookmark,
}

impl Manifest
{
    pub fn read(dir: &Path) -> Result<Self>
    {
        let path = dir.join(MANIFEST);
        let bytes = fs::read(&path).map_err(|error| unreadable(&path, &error))?;
        let manifest: Self =
            serde_json::from_slice(&bytes).map_err(|error| invalid(dir, format!("{MANIFEST} is invalid: {error}")))?;
        let version = manifest.version;
        let supported = version == LEGACY || version == VERSION;
        ensure!(supported, invalid(dir, format!("checkpoint version {version} is not supported")));

        Ok(manifest)
    }
}

pub fn check_destinations(config: &Config, first: usize) -> Result<()>
{
    let (checkpoint, last) = (&config.training.checkpoint, config.training.superbatches);
    (first..=last)
        .filter(|&superbatch| checkpoint.reason(superbatch, last, false).is_some())
        .try_for_each(|superbatch| vacant(&checkpoint.destination(&config.run.name, superbatch)))
}

fn vacant(destination: &Path) -> Result<()>
{
    ensure!(
        !destination.exists(),
        "Checkpoint destination {} already exists; choose another run name or checkpoint location, or move the \
         existing checkpoint away",
        destination.display()
    );

    Ok(())
}

pub fn save(
    config: &Config,
    datasets: &[SourceIdentity],
    trainer: &Trainer,
    superbatch: usize,
    data: Option<DataRecord>,
) -> Result<PathBuf>
{
    let location = config.training.checkpoint.location();
    let destination = config.training.checkpoint.destination(&config.run.name, superbatch);
    fs::create_dir_all(location).context("Cannot create checkpoint directory")?;
    vacant(&destination)?;

    let temporary = tempfile::Builder::new().prefix(".checkpoint-").tempdir_in(location)?;
    let state = temporary.path().join(STATE);
    fs::create_dir(&state)?;

    let bytes = contain(|| -> Result<Vec<u8>> {
        trainer
            .write_to_checkpoint(state.to_str().context("Checkpoint path must be UTF-8")?)
            .map_err(|error| anyhow!("Cannot save optimizer: {error:?}"))?;

        let weights = trainer.cpu_weights().map_err(|error| anyhow!("Cannot read network: {error:?}"))?;
        validate_state(&state, &weights)?;

        weights.to_quantised_buffer(&config.run.arch.saved_format(), true).context("Cannot export network")
    })
    .map_err(|panic| anyhow!("Cannot save optimizer or export network: {panic}"))??;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let network = format!("{}-{}.nnue", config.run.arch, &hash[..10]);
    fs::write(temporary.path().join(&network), bytes)?;

    let mut files = BTreeMap::new();
    for name in STATE_FILES.map(|file| format!("{STATE}/{file}")).into_iter().chain([network]) {
        let path = temporary.path().join(&name);
        let digest = DigestEntry::read(&path).context("Checkpoint file is missing")?;

        File::options().write(true).open(path)?.sync_all()?;
        files.insert(name, digest);
    }

    let batch = &config.training.data.batch;
    let manifest = Manifest {
        version: VERSION,
        bullet_revision: BULLET_REVISION.to_owned(),
        architecture: config.run.arch.name().to_owned(),
        steps: Some(StepsRecord { batch_size: batch.size, batches_per_superbatch: batch.per_superbatch }),
        completed_superbatch: superbatch,
        configuration: config.checkpoint_value(),
        source: Some(config.source.clone()),
        datasets: Datasets::Identified(datasets.to_vec()),
        data,
        validation: None,
        files,
    };
    let path = temporary.path().join(MANIFEST);
    fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    File::options().write(true).open(path)?.sync_all()?;

    sync_directory(&state)?;
    sync_directory(temporary.path())?;

    let published =
        vacant(&destination).and_then(|()| retried(|| fs::rename(temporary.path(), &destination)).map_err(Into::into));
    if let Err(error) = published {
        let kept = temporary.keep();
        return Err(error.context(format!("Cannot publish the checkpoint; its files are kept in {}", kept.display())));
    }
    sync_directory(location)?;

    Ok(destination)
}

pub fn restore(trainer: &mut Trainer, path: &Path) -> Result<()>
{
    let state = path.join(STATE);
    let state = state.to_str().context("Checkpoint path must be UTF-8")?;

    contain(|| trainer.load_from_checkpoint(state))
        .map_err(|panic| anyhow!("Cannot restore optimizer: {panic}"))?
        .map_err(|error| anyhow!("Cannot restore optimizer: {error:?}"))
}

pub fn verify(checkpoint: &Path, expected: Option<Architecture>) -> Result<Architecture>
{
    let invalid = |reason: String| invalid(checkpoint, reason);
    let Manifest { bullet_revision, architecture, files, .. } = Manifest::read(checkpoint)?;

    ensure!(
        bullet_revision == BULLET_REVISION,
        invalid(format!("it was written by Bullet revision {bullet_revision}; this build uses {BULLET_REVISION}"))
    );

    let architecture = architecture.parse::<Architecture>().map_err(|_| {
        anyhow!(
            "Checkpoint {} records the unknown architecture '{architecture}'; expected one of {}",
            checkpoint.display(),
            Architecture::names()
        )
    })?;
    if let Some(expected) = expected
        && expected != architecture
    {
        bail!(
            "Checkpoint {} holds a {architecture} network, not {expected}; check that --arch and the configuration \
             name the architecture it was trained with",
            checkpoint.display()
        );
    }

    let listed = STATE_FILES.map(|file| format!("{STATE}/{file}"));
    if let Some(missing) = listed.iter().find(|name| !files.contains_key(*name)) {
        bail!(invalid(format!("its manifest does not list {missing}")));
    }

    let prefix = format!("{}-", architecture.name().to_ascii_lowercase());
    let exported = |name: &String| name.to_ascii_lowercase().starts_with(&prefix) && name.ends_with(".nnue");
    ensure!(files.keys().any(exported), invalid(format!("it has no exported {architecture} network")));

    for (name, expected) in &files {
        let plain = Path::new(name).components().all(|part| matches!(part, Component::Normal(_)));
        ensure!(plain, invalid(format!("its manifest names the invalid file {name}")));

        let path = checkpoint.join(name);
        let digest = DigestEntry::read(&path).map_err(|error| unreadable(&path, &error))?;
        ensure!(digest == *expected, self::invalid(&path, "the file is incomplete or corrupted"));
    }

    validate_state(&checkpoint.join(STATE), &ModelWeights::zeroed(&architecture.definition()))?;

    Ok(architecture)
}

pub fn load(checkpoint: &Path, architecture: Option<Architecture>) -> Result<(Architecture, ModelWeights)>
{
    let architecture = verify(checkpoint, architecture)?;

    let path = checkpoint.join(STATE).join(WEIGHTS);
    let invalid = |reason: &str| invalid(&path, reason);
    let file = File::open(&path).map_err(|error| unreadable(&path, &error))?;

    let mut weights = ModelWeights::zeroed(&architecture.definition());
    contain(|| weights.load_from(BufReader::new(file)))
        .map_err(|_| invalid("the network weights cannot be decoded"))?
        .map_err(|error| unreadable(&path, &error))?;

    let finite = weights.iter().all(|(_, tensor)| tensor.values.f32().iter().all(|value| value.is_finite()));
    ensure!(finite, invalid("the network weights are incomplete or invalid"));

    Ok((architecture, weights))
}

pub fn validate_state(directory: &Path, weights: &ModelWeights) -> Result<()>
{
    STATE_FILES.iter().try_for_each(|name| validate_tensors(&directory.join(name), weights))
}

fn validate_tensors(path: &Path, weights: &ModelWeights) -> Result<()>
{
    let invalid = |reason: String| invalid(path, reason);
    let malformed = || invalid("the optimiser state is malformed".to_owned());

    let mut tensors: BTreeMap<&str, usize> =
        weights.iter().map(|(name, tensor)| (name.as_str(), tensor.values.size())).collect();
    let expected: usize = tensors.iter().map(|(name, count)| name.len() + 1 + size_of::<usize>() + count * 4).sum();

    let file = File::open(path).map_err(|error| unreadable(path, &error))?;
    let length = file.metadata().map_err(|error| unreadable(path, &error))?.len();
    ensure!(
        length == expected as u64,
        invalid(format!("the optimiser state holds {length} bytes where the network needs {expected}"))
    );

    let mut bytes = Vec::with_capacity(expected);
    file.take(expected as u64).read_to_end(&mut bytes).map_err(|error| unreadable(path, &error))?;

    let mut remaining = bytes.as_slice();
    while !remaining.is_empty() {
        let end = remaining.iter().position(|byte| *byte == b'\n').ok_or_else(malformed)?;
        let name = str::from_utf8(&remaining[..end]).map_err(|_| malformed())?;
        let count = tensors
            .remove(name)
            .ok_or_else(|| invalid(format!("the optimiser state names an unknown tensor {name}")))?;

        let (size, rest) = remaining[end + 1..].split_first_chunk().ok_or_else(malformed)?;
        ensure!(usize::from_le_bytes(*size) == count, invalid(format!("tensor {name} has the wrong size")));

        let (values, rest) = rest.split_at_checked(count * 4).ok_or_else(malformed)?;
        let finite = values.as_chunks::<4>().0.iter().all(|bytes| f32::from_le_bytes(*bytes).is_finite());
        ensure!(finite, invalid(format!("tensor {name} holds a value that is not finite")));

        remaining = rest;
    }

    match tensors.keys().next() {
        Some(name) => Err(invalid(format!("the optimiser state lacks tensor {name}"))),
        None => Ok(()),
    }
}

fn retried(mut operation: impl FnMut() -> io::Result<()>) -> io::Result<()>
{
    for _ in 0..PUBLISH_RETRIES {
        if operation().is_ok() {
            return Ok(());
        }

        thread::sleep(PUBLISH_PAUSE);
    }

    operation()
}

fn sync_directory(path: &Path) -> io::Result<()> { if cfg!(unix) { File::open(path)?.sync_all() } else { Ok(()) } }

fn unreadable(path: &Path, error: &io::Error) -> anyhow::Error { invalid(path, format!("cannot be read: {error}")) }

fn invalid(path: &Path, reason: impl std::fmt::Display) -> anyhow::Error
{
    anyhow!("Checkpoint {}: {reason}; check that the checkpoint directory is complete and unchanged", path.display())
}
