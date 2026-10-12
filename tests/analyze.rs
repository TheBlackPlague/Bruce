use bruce::{
    analyze::{self, Options},
    config::DataSource,
    data::Format,
};

const POSITION: &str = "4k3/8/8/8/8/8/8/4K3";

fn options(path: std::path::PathBuf) -> Options
{
    Options {
        source: DataSource { format: Format::Text, path, separator: Some(";".into()) },
        network: None,
        architecture: None,
        gpu: 0,
        limit: None,
    }
}

#[test]
fn finite_analysis_normalizes_black_scores_and_respects_position_limit()
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("positions.txt");
    std::fs::write(
        &path,
        format!("{POSITION} w - - 0 1 ; -100 ; 0\n{POSITION} b - - 0 1 ; -300 ; 0\n{POSITION} w - - 0 1 ; 100 ; 1\n"),
    )
    .unwrap();
    let mut options = options(path);
    let report = analyze::run(&options).unwrap();
    
    assert_eq!(report.positions, 3);
    assert_eq!(report.score_distribution, [(-100, 1), (100, 1), (300, 1)].into());
    assert_eq!((report.results.losses, report.results.draws, report.results.wins), (1, 0, 2));
    assert_eq!(report.material[&2].positions, 3);
    assert!(!report.limit_reached);
    
    options.limit = Some(2);
    let report = analyze::run(&options).unwrap();
    
    assert_eq!(report.positions, 2);
    assert_eq!(report.scores.mean_cp, 100.0);
    assert!(report.limit_reached);
}

#[test]
fn empty_and_truncated_inputs_fail_without_a_partial_report()
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("positions.txt");
    std::fs::write(&path, "").unwrap();
    let options = options(path.clone());
    
    assert!(analyze::run(&options).is_err());
    
    std::fs::write(&path, format!("{POSITION} w - - 0 1 ; 0 ; 0.5\nbroken")).unwrap();
    
    assert!(analyze::run(&options).is_err());
}
