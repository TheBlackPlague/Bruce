use std::{fs, path::Path};

use bruce::{
    architecture::Architecture,
    checkpoint::{self, Checkpoints},
    config::Config,
};
use bullet_gpu::runtime::{Device, mock::MockGpu};
use bullet_trainer::{
    model::ModelWeights,
    optimiser::{Optimiser, adam::AdamW},
};

fn fixture(root: &Path) -> Config
{
    let mut config = Config::from_toml(include_str!("../config/training.toml.default")).unwrap();
    let data = root.join("data.bin");
    fs::write(&data, [0u8; 32]).unwrap();
    config.data.sources[0].path = data;
    config.training.checkpoint.location = root.join("checkpoints");
    config
}

fn optimiser(architecture: Architecture, seed: u64) -> Optimiser<MockGpu, AdamW<MockGpu>>
{
    let definition = architecture.definition();
    let weights = ModelWeights::new(&definition, seed);
    Optimiser::new(definition, weights, Device::<MockGpu>::new(0).unwrap(), Default::default()).unwrap()
}

#[test]
fn checkpoint_restores_native_weights_and_optimizer_and_rejects_overwrite()
{
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let checkpoints = Checkpoints::new(&config).unwrap();
    let architecture: Architecture = config.architecture.parse().unwrap();
    let original = optimiser(architecture, 42);
    let path = checkpoints.save(&original, &architecture.saved_format(), 1).unwrap();

    assert_eq!(checkpoints.inspect(&path).unwrap(), 1);
    assert!(checkpoints.save(&original, &architecture.saved_format(), 1).is_err());

    let mut restored = optimiser(architecture, 77);
    checkpoint::restore(&mut restored, &path).unwrap();
    let second = checkpoints.save(&restored, &architecture.saved_format(), 2).unwrap();

    for file in ["weights.bin", "momentum.bin", "velocity.bin"] {
        let file = Path::new("optimiser_state").join(file);
        assert_eq!(fs::read(path.join(&file)).unwrap(), fs::read(second.join(&file)).unwrap());
    }

    let loaded = checkpoint::load_weights(&path, architecture).unwrap();
    let expected = original.cpu_weights().unwrap();
    for (name, values) in loaded.iter() {
        assert_eq!(values.values.f32(), expected.get(name).values.f32());
    }

    assert!(checkpoint::load_weights(&path, Architecture::Starshard).is_err());

    fs::write(path.join("optimiser_state/momentum.bin"), b"corrupted").unwrap();

    assert!(checkpoints.inspect(&path).is_err());
    assert!(checkpoint::load_weights(&path, architecture).is_err());
}

#[test]
fn checkpoint_rejects_changed_datasets_completed_runs_and_unsafe_paths()
{
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let checkpoints = Checkpoints::new(&config).unwrap();
    let architecture: Architecture = config.architecture.parse().unwrap();
    let optimiser = optimiser(architecture, 42);
    let path = checkpoints.save(&optimiser, &architecture.saved_format(), 1).unwrap();
    let completed = checkpoints.save(&optimiser, &architecture.saved_format(), config.training.superbatches).unwrap();

    assert!(checkpoints.inspect(&completed).is_err());

    fs::write(&config.data.sources[0].path, [0u8; 64]).unwrap();

    assert!(Checkpoints::new(&config).unwrap().inspect(&path).is_err());

    let manifest_path = path.join("checkpoint.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let digest = manifest["files"]["optimiser_state/weights.bin"].clone();
    manifest["files"]["../outside"] = digest;
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    assert!(checkpoint::load_weights(&path, architecture).is_err());
}

#[test]
fn malformed_native_state_is_rejected_even_with_a_matching_digest()
{
    use sha2::{Digest, Sha256};

    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let checkpoints = Checkpoints::new(&config).unwrap();
    let architecture: Architecture = config.architecture.parse().unwrap();
    let optimiser = optimiser(architecture, 42);
    let path = checkpoints.save(&optimiser, &architecture.saved_format(), 1).unwrap();
    let state_path = path.join("optimiser_state/weights.bin");

    let mut bytes = fs::read(&state_path).unwrap();
    bytes[0] = b'?';

    fs::write(state_path, &bytes).unwrap();
    let manifest_path = path.join("checkpoint.json");

    let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["files"]["optimiser_state/weights.bin"]["sha256"] = format!("{:x}", Sha256::digest(bytes)).into();
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    assert!(checkpoints.inspect(&path).is_err());
    assert!(checkpoint::load_weights(&path, architecture).is_err());
}

#[test]
fn nonfinite_optimizer_state_is_never_published()
{
    let root = tempfile::tempdir().unwrap();
    let config = fixture(root.path());
    let checkpoints = Checkpoints::new(&config).unwrap();
    let architecture: Architecture = config.architecture.parse().unwrap();
    let mut trainer = optimiser(architecture, 42);
    let state = root.path().join("native");
    fs::create_dir(&state).unwrap();
    trainer.write_to_checkpoint(state.to_str().unwrap()).unwrap();

    let momentum = state.join("momentum.bin");
    let mut bytes = fs::read(&momentum).unwrap();
    let value = bytes.iter().position(|byte| *byte == b'\n').unwrap() + 1 + size_of::<usize>();
    bytes[value..value + size_of::<f32>()].copy_from_slice(&f32::INFINITY.to_le_bytes());
    fs::write(&momentum, bytes).unwrap();
    trainer.load_from_checkpoint(state.to_str().unwrap()).unwrap();

    let error = checkpoints.save(&trainer, &architecture.saved_format(), 1).unwrap_err();
    assert!(error.to_string().contains("invalid values"));
    assert!(!checkpoints.destination(1).exists());
    assert_eq!(fs::read_dir(&config.training.checkpoint.location).unwrap().count(), 0);

    trainer.reset_state().unwrap();
    let path = checkpoints.save(&trainer, &architecture.saved_format(), 1).unwrap();
    assert_eq!(checkpoints.inspect(&path).unwrap(), 1);
}
