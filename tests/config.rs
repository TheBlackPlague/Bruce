use std::path::PathBuf;

use bruce::config::{CheckpointPolicy, Config};

const EXAMPLE: &str = include_str!("../config/training.toml.default");

#[test]
fn example_matches_requested_layout()
{
    let config = Config::from_toml(EXAMPLE).unwrap();

    assert_eq!(config.name, "exp_name");
    assert_eq!(config.training.superbatches, 400);
    assert_eq!(config.data.batch_size, 16384);
    assert_eq!(config.data.mapping_threads, 4);
    assert_eq!(config.data.filter.unwrap().minimum_eval, 100);
    assert!(config.training.checkpoint.policy.should_save(30));
    assert!(!config.training.checkpoint.policy.should_save(31));
}

#[test]
fn configuration_location_does_not_change_relative_paths()
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("training.toml");
    std::fs::write(&path, EXAMPLE).unwrap();
    let config = Config::load(&path).unwrap();

    assert_eq!(config.training.checkpoint.location, PathBuf::from("checkpoints"));
    assert_eq!(config.data.sources[0].path, PathBuf::from("data/monty_value_00_06.bin"));
    assert_eq!(config.tensorboard.path, PathBuf::from("runs"));
}

#[test]
fn rejects_unknown_fields_and_invalid_values()
{
    for source in [
        EXAMPLE.replace("superbatches = 400", "superbatches = 400\nunknown = true"),
        EXAMPLE.replace("gamma = 0.3", "gama = 0.3"),
        EXAMPLE.replace("superbatches = 400", "superbatches = 0"),
        EXAMPLE.replace("mapper = 4", "mapper = 256"),
        EXAMPLE.replace("size = 16384", "size = 0"),
        EXAMPLE.replace("beta2 = 0.999", "beta2 = nan"),
        EXAMPLE.replace("minimum_eval = 100", "minimum_eval = 8000"),
        EXAMPLE.replace("minimum_pieces = 0", "minimum_pieces = 33"),
        EXAMPLE.replace("arch = \"Aurora\"", "arch = \"Unknown\""),
        EXAMPLE.replace("superbatch = 30", "superbatch = 0"),
    ] {
        assert!(Config::from_toml(&source).is_err(), "accepted invalid configuration: {source}");
    }
}

#[test]
fn both_stockdory_architectures_are_accepted()
{
    for architecture in ["Starshard", "Aurora"] {
        Config::from_toml(&EXAMPLE.replace("arch = \"Aurora\"", &format!("arch = \"{architecture}\""))).unwrap();
    }
}

#[test]
fn single_checkpoint_saves_only_requested_superbatch()
{
    let policy = CheckpointPolicy::Single { superbatch: 200 };
    assert!(!policy.should_save(199));
    assert!(policy.should_save(200));
    assert!(!policy.should_save(201));
}

#[test]
fn text_source_requires_runtime_separator()
{
    let source = EXAMPLE
        .replace("format = \"monty\"", "format = \"text\"")
        .replace("exclude_tactical = true", "exclude_tactical = false")
        .replace("exclude_check = true", "exclude_check = false")
        .replace("minimum_ply = 14", "minimum_ply = 0");
    assert!(Config::from_toml(&source).is_err());
    Config::from_toml(&source.replace("format = \"text\"", "format = \"text\"\nseparator = \" | \"")).unwrap();
}

#[test]
fn canonical_snapshot_ignores_toml_layout_and_comments()
{
    let first = Config::from_toml(EXAMPLE).unwrap();
    let second = Config::from_toml(&format!("# Another layout\n\n{EXAMPLE}\n")).unwrap();
    assert_eq!(first.checkpoint_value(), second.checkpoint_value());
}

#[test]
fn source_count_is_not_limited_like_schedule_phases()
{
    let mut source = EXAMPLE.to_owned();
    for index in 1..=100 {
        source.push_str(&format!("\n[training.data.source.{index}]\nformat='monty'\npath='data/part{index}.bin'\n"));
    }
    assert_eq!(Config::from_toml(&source).unwrap().data.sources.len(), 101);
}

#[test]
fn checkpoint_snapshot_tracks_runtime_overrides()
{
    let mut config = Config::from_toml(EXAMPLE).unwrap();
    let before = config.checkpoint_value();
    config.optimizer.decay = 0.02;
    assert_ne!(config.checkpoint_value(), before);
}

#[test]
fn periodic_interval_may_exceed_training_but_single_target_must_not()
{
    Config::from_toml(&EXAMPLE.replace("superbatch = 30", "superbatch = 1000")).unwrap();
    let source = EXAMPLE
        .replace("superbatch = 30", "superbatch = 1000")
        .replace("type = \"periodic\"", "type = \"single\"")
        .replace("[training.checkpoint.periodic]", "[training.checkpoint.single]");
    assert!(Config::from_toml(&source).is_err());
}

#[test]
fn reporting_changes_do_not_invalidate_resume()
{
    let mut config = Config::from_toml(EXAMPLE).unwrap();
    let before = config.checkpoint_value();
    config.tensorboard.path = PathBuf::from("another-directory");
    config.tensorboard.enabled = !config.tensorboard.enabled;
    assert_eq!(config.checkpoint_value(), before);
}
