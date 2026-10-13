use super::*;

fn board(score: i16, result: u8) -> ChessBoard { ChessBoard { occ: 3, score, result, ..ChessBoard::default() } }

#[test]
fn signed_score_distribution_and_percentiles_are_exact()
{
    let mut stats = Accumulator::new(None);
    for (score, result) in [(-100, 0), (-100, 1), (0, 1), (300, 2)] {
        stats.observe(board(score, result));
    }
    let report = stats.finish(None);

    assert_eq!(report.score_distribution, BTreeMap::from([(-100, 2), (0, 1), (300, 1)]));
    assert_eq!(report.scores.mean_cp, 25.0);
    assert!((report.scores.standard_deviation_cp - 26875.0_f64.sqrt()).abs() < 1e-10);
    assert_eq!(report.scores.percentiles_cp[&50], -100);
    assert_eq!(report.scores.percentiles_cp[&95], 300);
    assert_eq!((report.results.losses, report.results.draws, report.results.wins), (1, 2, 1));
    assert_eq!(report.material[&2].positions, 4);
}

#[test]
fn network_error_and_calibration_use_stm_scores()
{
    let mut stats = Accumulator::new(Some(Architecture::Aurora));

    for (score, result, prediction) in [(-100, 0, 0.0), (100, 2, 0.0)] {
        let board = board(score, result);
        stats.observe(board);
        stats.predict(board, prediction).unwrap();
    }

    let report = stats.finish(None);
    let network = report.network.unwrap();

    assert_eq!(network.overall.bias_cp, 0.0);
    assert_eq!(network.overall.mean_absolute_error_cp, 100.0);
    assert_eq!(network.overall.root_mean_squared_error_cp, 100.0);
    assert_eq!(network.overall.result_squared_error, 0.25);
    assert_eq!(network.calibration[10].positions, 2);
    assert_eq!(network.calibration[10].mean_prediction, 0.5);
    assert_eq!(network.calibration[10].mean_result, 0.5);
    assert_eq!(network.by_teacher_score_cp[&-100].bias_cp, 100.0);
    assert_eq!(network.by_teacher_score_cp[&100].bias_cp, -100.0);
}

#[test]
fn score_buckets_floor_negative_values_and_nonfinite_outputs_fail()
{
    let mut stats = Accumulator::new(Some(Architecture::Starshard));
    stats.predict(board(-1, 1), 0.0).unwrap();

    assert!(stats.network.as_ref().unwrap().by_teacher_score_cp.contains_key(&-100));
    assert!(stats.predict(board(0, 1), f32::NAN).is_err());
    assert!(stats.predict(board(0, 1), f32::INFINITY).is_err());
}

#[test]
fn report_does_not_overwrite_existing_files()
{
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("report.json");

    let mut stats = Accumulator::new(None);
    stats.observe(board(0, 1));

    let report = stats.finish(Some(1));
    report.write(&path).unwrap();

    let first = std::fs::read(&path).unwrap();

    assert!(report.write(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), first);

    let decoded: serde_json::Value = serde_json::from_slice(&first).unwrap();

    assert_eq!(decoded["limit_reached"], true);
    assert_eq!(decoded["score_distribution"]["0"], 1);
}
