use std::path::Path;

use bruce::{
    config::{Config, DataSource},
    data::Format,
    train,
};

fn configuration(path: &Path) -> Config
{
    let mut config = Config::from_toml(include_str!("../config/training.toml.default")).unwrap();
    config.data.sources = vec![DataSource { format: Format::Text, path: path.into(), separator: Some(" | ".into()) }];
    config.data.filter = None;
    config
}

#[test]
fn preflight_checks_the_whole_dataset_without_a_gpu()
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("positions.txt");
    let position = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 | -120 | 0.5\n";
    std::fs::write(&path, position).unwrap();

    let config = configuration(&path);
    train::check(&config).unwrap();

    std::fs::write(&path, format!("{position}not a position\n")).unwrap();
    assert!(train::check(&config).is_err());
}

#[test]
fn preflight_rejects_empty_data()
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("positions.txt");
    std::fs::write(&path, "").unwrap();
    assert!(train::check(&configuration(&path)).is_err());
}

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "metal")))]
#[test]
fn mock_build_never_claims_to_train()
{
    let directory = tempfile::tempdir().unwrap();
    let config = configuration(&directory.path().join("unused.txt"));
    let error = train::run(config, None, true).unwrap_err();
    assert!(error.to_string().contains("GPU build"));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
}
