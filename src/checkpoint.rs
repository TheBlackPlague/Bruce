use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    architecture::aurora::{self, AuroraOptimiser},
    config::Config,
};

const SCHEMA_VERSION: u32 = 1;
const BULLET_REVISION: &str = "2ea3d2d0f7e597b0d645f6e8040cf37818f51bce";
const OPTIMISER_FILES: [&str; 3] = [
    "optimiser_state/weights.bin",
    "optimiser_state/momentum.bin",
    "optimiser_state/velocity.bin",
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    schema_version: u32,
    bullet_revision: String,
    pub completed_superbatch: usize,
    config: Config,
    datasets: Vec<DatasetIdentity>,
    files: BTreeMap<String, FileIdentity>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetIdentity {
    path: PathBuf,
    bytes: u64,
    modified_nanos: u128,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    bytes: u64,
    sha256: String,
}

fn file_identity(path: &Path) -> Result<FileIdentity> {
    let mut file = File::open(path).with_context(|| format!("Opening {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut bytes = 0;

    loop {
        let count = file.read(&mut buffer)?;

        if count == 0 {
            break;
        }

        bytes += count as u64;
        hash.update(&buffer[..count]);
    }

    Ok(FileIdentity {
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn dataset_identities(config: &Config) -> Result<Vec<DatasetIdentity>> {
    config
        .data
        .paths
        .iter()
        .map(|path| {
            let path = fs::canonicalize(path)
                .with_context(|| format!("Locating training dataset {}", path.display()))?;
            let metadata = fs::metadata(&path)?;
            let modified_nanos = metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos();

            Ok(DatasetIdentity {
                path,
                bytes: metadata.len(),
                modified_nanos,
            })
        })
        .collect()
}

pub fn inspect(path: &Path, config: &Config) -> Result<Checkpoint> {
    let metadata_path = path.join("checkpoint.json");
    let checkpoint: Checkpoint = serde_json::from_slice(
        &fs::read(&metadata_path)
            .with_context(|| format!("Reading {}", metadata_path.display()))?,
    )?;

    ensure!(
        checkpoint.schema_version == SCHEMA_VERSION,
        "Unsupported checkpoint version"
    );
    ensure!(
        checkpoint.bullet_revision == BULLET_REVISION,
        "Checkpoint uses a different Bullet revision"
    );
    ensure!(
        checkpoint.config == *config,
        "Resume configuration differs from the saved configuration; use the original configuration"
    );
    ensure!(
        checkpoint.datasets == dataset_identities(config)?,
        "Training datasets have changed since this checkpoint was saved"
    );
    ensure!(
        checkpoint.completed_superbatch > 0,
        "Checkpoint has no completed superbatch"
    );
    ensure!(
        checkpoint.completed_superbatch < config.training.superbatches,
        "This checkpoint already completed the configured training run"
    );

    for required in OPTIMISER_FILES {
        ensure!(
            checkpoint.files.contains_key(required),
            "Checkpoint is missing {required}"
        );
    }

    ensure!(
        checkpoint
            .files
            .keys()
            .any(|name| name.starts_with("Aurora-") && name.ends_with(".nnue")),
        "Checkpoint is missing its quantized network"
    );

    for (name, expected) in &checkpoint.files {
        let relative = Path::new(name);

        ensure!(
            relative
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_))),
            "Invalid checkpoint filename"
        );
        ensure!(
            file_identity(&path.join(relative))? == *expected,
            "Checkpoint file {name} is incomplete or corrupted"
        );
    }

    Ok(checkpoint)
}

pub fn check_destinations(config: &Config, start_superbatch: usize) -> Result<()> {
    let entries = match fs::read_dir(&config.output_directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).context("Reading checkpoint output directory"),
    };

    let prefix = format!("{}-", config.name);
    for entry in entries {
        let entry = entry?;
        let filename = entry.file_name();

        let Some(filename) = filename.to_str() else {
            continue;
        };

        let Some(superbatch) = filename
            .strip_prefix(&prefix)
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };

        if filename == format!("{prefix}{superbatch}")
            && superbatch >= start_superbatch
            && superbatch <= config.training.superbatches
            && (superbatch.is_multiple_of(config.training.save_every)
                || superbatch == config.training.superbatches)
        {
            bail!(
                "Checkpoint {} already exists; \
                use a new run name/output directory or resume after it",
                entry.path().display()
            );
        }
    }

    Ok(())
}

pub fn restore(optimiser: &mut AuroraOptimiser, path: &Path) -> Result<()> {
    let optimiser_path = path.join("optimiser_state");
    let optimiser_path = optimiser_path
        .to_str()
        .context("Checkpoint path must be UTF-8 for Bullet")?;

    optimiser
        .load_from_checkpoint(optimiser_path)
        .map_err(|error| anyhow::anyhow!("Loading Bullet optimiser: {error:?}"))
}

pub fn save(
    optimiser: &AuroraOptimiser,
    config: &Config,
    completed_superbatch: usize,
) -> Result<PathBuf> {
    fs::create_dir_all(&config.output_directory)?;

    let name = format!("{}-{completed_superbatch}", config.name);
    let destination = config.output_directory.join(&name);

    ensure!(
        !destination.exists(),
        "Checkpoint {} already exists; refusing to overwrite it",
        destination.display()
    );

    let temporary = config
        .output_directory
        .join(format!(".{name}-{}.tmp", std::process::id()));

    fs::create_dir(&temporary)
        .with_context(|| format!("Creating temporary checkpoint {}", temporary.display()))?;

    let result = write_checkpoint(optimiser, config, completed_superbatch, &temporary)
        .and_then(|()| {
            fs::rename(&temporary, &destination).context("Publishing completed checkpoint")
        });

    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }

    #[cfg(unix)]
    File::open(&config.output_directory)?.sync_all()?;

    Ok(destination)
}

fn write_checkpoint(
    optimiser: &AuroraOptimiser,
    config: &Config,
    completed_superbatch: usize,
    directory: &Path,
) -> Result<()> {
    let optimiser_path = directory.join("optimiser_state");
    fs::create_dir(&optimiser_path)?;

    optimiser
        .write_to_checkpoint(
            optimiser_path
                .to_str()
                .context("Checkpoint path must be UTF-8 for Bullet")?,
        )
        .map_err(|error| anyhow::anyhow!("Saving Bullet optimiser: {error:?}"))?;

    let weights = optimiser
        .cpu_weights()
        .map_err(|error| anyhow::anyhow!("Reading Bullet weights: {error:?}"))?;

    let bytes = weights
        .to_quantised_buffer(&aurora::saved_format(), true)
        .context("Quantizing Aurora for MantaRay v2")?;

    let digest = format!("{:x}", Sha256::digest(&bytes));
    let network_name = format!("Aurora-{}.nnue", &digest[..10]);

    fs::write(directory.join(&network_name), bytes)?;

    let mut files = BTreeMap::new();
    for name in OPTIMISER_FILES
        .into_iter()
        .chain(std::iter::once(network_name.as_str()))
    {
        let path = directory.join(name);
        let identity = file_identity(&path)?;

        if identity.bytes == 0 {
            bail!("Bullet produced an empty checkpoint file: {name}");
        }

        fs::OpenOptions::new().write(true).open(&path)?.sync_all()?;
        files.insert(name.to_string(), identity);
    }

    let checkpoint = Checkpoint {
        schema_version: SCHEMA_VERSION,
        bullet_revision: BULLET_REVISION.to_string(),
        completed_superbatch,
        config: config.clone(),
        datasets: dataset_identities(config)?,
        files,
    };

    let mut metadata = File::create(directory.join("checkpoint.json"))?;
    metadata.write_all(&serde_json::to_vec_pretty(&checkpoint)?)?;
    metadata.sync_all()?;

    #[cfg(unix)]
    {
        File::open(&optimiser_path)?.sync_all()?;
        File::open(directory)?.sync_all()?;
    }

    Ok(())
}

#[cfg(all(test, not(any(feature = "cuda", feature = "rocm", feature = "metal"))))]
mod tests {
    use super::*;

    fn fixture(root: &Path) -> Config {
        let mut config: Config = toml::from_str(include_str!("../config/training.toml")).unwrap();
        let dataset = root.join("sample.bullet");
        fs::write(&dataset, [0_u8; 32]).unwrap();

        config.data.paths = vec![dataset];
        config.output_directory = root.join("checkpoints");
        config.training.superbatches = 3;

        config
    }

    #[test]
    fn native_checkpoint_restores_master_weights_and_rejects_corruption() {
        let root = tempfile::tempdir().unwrap();
        let config = fixture(root.path());
        let optimiser = aurora::create(42, 0, config.optimizer.params()).unwrap();
        let path = save(&optimiser, &config, 1).unwrap();
        
        fs::copy(
            path.join("optimiser_state/weights.bin"),
            path.join("optimiser_state/momentum.bin"),
        )
        .unwrap();

        let mut manifest: Checkpoint =
            serde_json::from_slice(&fs::read(path.join("checkpoint.json")).unwrap()).unwrap();
        manifest.files.insert(
            "optimiser_state/momentum.bin".into(),
            file_identity(&path.join("optimiser_state/momentum.bin")).unwrap(),
        );

        fs::write(
            path.join("checkpoint.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let checkpoint = inspect(&path, &config).unwrap();
        assert_eq!(checkpoint.completed_superbatch, 1);

        let network_name = checkpoint
            .files
            .keys()
            .find(|name| name.ends_with(".nnue"))
            .unwrap();

        assert_eq!(
            checkpoint.files[network_name].bytes,
            aurora::NETWORK_BYTES as u64
        );
        assert!(network_name.contains(&checkpoint.files[network_name].sha256[..10]));

        let mut restored = aurora::create(999, 0, config.optimizer.params()).unwrap();
        restore(&mut restored, &path).unwrap();

        let next = save(&restored, &config, 2).unwrap();
        let restored_metadata = inspect(&next, &config).unwrap();
        assert_eq!(checkpoint.files, restored_metadata.files);
        assert!(save(&restored, &config, 2).is_err());

        fs::write(path.join("optimiser_state/momentum.bin"), b"truncated").unwrap();
        assert!(inspect(&path, &config).is_err());
    }

    #[test]
    fn checks_only_future_checkpoint_destinations() {
        let root = tempfile::tempdir().unwrap();
        let mut config = fixture(root.path());

        config.training.save_every = 1;

        fs::create_dir_all(config.output_directory.join(format!("{}-1", config.name))).unwrap();
        assert!(check_destinations(&config, 1).is_err());

        check_destinations(&config, 2).unwrap();

        fs::create_dir(config.output_directory.join(format!("{}-3", config.name))).unwrap();
        assert!(check_destinations(&config, 2).is_err());
    }

    #[test]
    fn refuses_changed_schedule_dataset_and_completed_run() {
        let root = tempfile::tempdir().unwrap();
        let config = fixture(root.path());
        let optimiser = aurora::create(42, 0, config.optimizer.params()).unwrap();
        let path = save(&optimiser, &config, 1).unwrap();

        let mut changed = config.clone();
        changed.optimizer.beta1 = 0.5;
        assert!(inspect(&path, &changed).is_err());

        fs::write(&config.data.paths[0], [0_u8; 64]).unwrap();
        assert!(inspect(&path, &config).is_err());
        
        let completed = save(&optimiser, &config, config.training.superbatches).unwrap();
        assert!(inspect(&completed, &config).is_err());
    }
}
