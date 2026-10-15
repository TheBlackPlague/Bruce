use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use bullet_lib::game::formats::bulletformat::ChessBoard;
use serde::Serialize;

use crate::architecture::{Architecture, SCALE};

const SCORE_BUCKET: i32 = 100;
const PROBABILITY_BUCKETS: usize = 20;

#[derive(Serialize)]
pub struct Report
{
    pub perspective: &'static str,
    pub positions: u64,
    pub limit_reached: bool,
    pub scores: ScoreSummary,
    pub score_distribution: BTreeMap<i16, u64>,
    pub results: Results,
    pub material: BTreeMap<u32, Group>,
    pub network: Option<NetworkReport>,
}

#[derive(Default, Serialize)]
pub struct Results
{
    pub losses: u64,
    pub draws: u64,
    pub wins: u64,
}

#[derive(Default, Serialize)]
pub struct ScoreSummary
{
    pub mean_cp: f64,
    pub standard_deviation_cp: f64,
    pub minimum_cp: i16,
    pub maximum_cp: i16,
    pub percentiles_cp: BTreeMap<u8, i16>,
}

#[derive(Default, Serialize)]
pub struct Group
{
    pub positions: u64,
    pub mean_score_cp: f64,
    pub results: Results,
}

#[derive(Serialize)]
pub struct NetworkReport
{
    pub architecture: Architecture,
    pub weights: &'static str,
    pub score_bucket_cp: i32,
    pub overall: ErrorSummary,
    pub by_teacher_score_cp: BTreeMap<i32, ErrorSummary>,
    pub by_piece_count: BTreeMap<u32, ErrorSummary>,
    pub calibration: Vec<Calibration>,
    pub extreme_predictions: u64,
    pub extreme_probability_threshold: f64,
}

#[derive(Default, Serialize)]
pub struct ErrorSummary
{
    pub positions: u64,
    pub mean_prediction_cp: f64,
    pub bias_cp: f64,
    pub mean_absolute_error_cp: f64,
    pub root_mean_squared_error_cp: f64,
    pub result_squared_error: f64,
    pub teacher_probability_squared_error: f64,
    #[serde(skip)]
    squared_error_sum: f64,
}

#[derive(Serialize)]
pub struct Calibration
{
    pub probability_from: f64,
    pub probability_to: f64,
    pub positions: u64,
    pub mean_prediction: f64,
    pub mean_result: f64,
}

pub(super) struct Statistics
{
    report: Report,
    score_counts: Vec<u64>,
    score_deviations: f64,
}

impl Statistics
{
    pub(super) fn new(architecture: Option<Architecture>) -> Self
    {
        Self {
            report: Report {
                perspective: "side_to_move",
                positions: 0,
                limit_reached: false,
                scores: ScoreSummary::default(),
                score_distribution: BTreeMap::new(),
                results: Results::default(),
                material: BTreeMap::new(),
                network: architecture.map(NetworkReport::new),
            },
            score_counts: vec![0; 1 << 16],
            score_deviations: 0.0,
        }
    }

    pub(super) fn positions(&self) -> u64 { self.report.positions }

    pub(super) fn record(&mut self, board: &ChessBoard)
    {
        let report = &mut self.report;
        report.positions += 1;

        let score = f64::from(board.score);
        let previous_delta = score - report.scores.mean_cp;

        update_mean(&mut report.scores.mean_cp, score, report.positions);

        self.score_deviations += previous_delta * (score - report.scores.mean_cp);

        let score_index = (i32::from(board.score) - i32::from(i16::MIN)) as usize;
        self.score_counts[score_index] += 1;

        report.results.record(board.result);

        let group = report.material.entry(board.occ.count_ones()).or_default();
        group.positions += 1;

        update_mean(&mut group.mean_score_cp, score, group.positions);
        group.results.record(board.result);
    }

    pub(super) fn prediction(&mut self, board: &ChessBoard, output: f32) -> Result<()>
    {
        ensure!(output.is_finite(), "Network produced an invalid score");

        let network = self.report.network.as_mut().expect("Network statistics require an architecture");

        let prediction_cp = f64::from(output) * f64::from(SCALE);

        network.overall.record(board, prediction_cp);
        network.by_piece_count.entry(board.occ.count_ones()).or_default().record(board, prediction_cp);
        let score_bucket = i32::from(board.score).div_euclid(SCORE_BUCKET) * SCORE_BUCKET;
        network.by_teacher_score_cp.entry(score_bucket).or_default().record(board, prediction_cp);

        let probability = sigmoid(f64::from(output));

        let bucket = ((probability * PROBABILITY_BUCKETS as f64) as usize).min(PROBABILITY_BUCKETS - 1);

        let calibration = &mut network.calibration[bucket];
        calibration.positions += 1;

        update_mean(&mut calibration.mean_prediction, probability, calibration.positions);
        update_mean(&mut calibration.mean_result, f64::from(board.result) / 2.0, calibration.positions);

        let threshold = network.extreme_probability_threshold;
        if probability < threshold || probability > 1.0 - threshold {
            network.extreme_predictions += 1;
        }

        Ok(())
    }

    pub(super) fn finish(mut self, limit: Option<u64>) -> Report
    {
        let report = &mut self.report;

        report.limit_reached = limit == Some(report.positions);
        report.score_distribution = self
            .score_counts
            .into_iter()
            .enumerate()
            .filter(|(_, count)| *count != 0)
            .map(|(index, count)| ((index as i32 + i32::from(i16::MIN)) as i16, count))
            .collect();
        report.scores.standard_deviation_cp = (self.score_deviations / report.positions.max(1) as f64).sqrt();
        report.scores.minimum_cp = report.score_distribution.first_key_value().map_or(0, |(&score, _)| score);
        report.scores.maximum_cp = report.score_distribution.last_key_value().map_or(0, |(&score, _)| score);

        let mut percentiles = [1_u8, 5, 25, 50, 75, 95, 99].into_iter().peekable();

        let mut count = 0;
        for (&score, &frequency) in &report.score_distribution {
            count += frequency;
            while let Some(&percentile) = percentiles.peek() {
                let rank = (u128::from(report.positions) * u128::from(percentile)).div_ceil(100);

                if u128::from(count) < rank {
                    break;
                }

                report.scores.percentiles_cp.insert(percentile, score);
                percentiles.next();
            }
        }

        self.report
    }
}

impl Results
{
    fn record(&mut self, result: u8)
    {
        match result {
            0 => self.losses += 1,
            1 => self.draws += 1,
            2 => self.wins += 1,
            _ => unreachable!("Dataset results have been validated"),
        }
    }
}

impl NetworkReport
{
    fn new(architecture: Architecture) -> Self
    {
        let calibration = (0..PROBABILITY_BUCKETS)
            .map(|bucket| Calibration {
                probability_from: bucket as f64 / PROBABILITY_BUCKETS as f64,
                probability_to: (bucket + 1) as f64 / PROBABILITY_BUCKETS as f64,
                positions: 0,
                mean_prediction: 0.0,
                mean_result: 0.0,
            })
            .collect();

        Self {
            architecture,
            weights: "native_float_checkpoint",
            score_bucket_cp: SCORE_BUCKET,
            overall: ErrorSummary::default(),
            by_teacher_score_cp: BTreeMap::new(),
            by_piece_count: BTreeMap::new(),
            calibration,
            extreme_predictions: 0,
            extreme_probability_threshold: 0.01,
        }
    }
}

impl ErrorSummary
{
    fn record(&mut self, board: &ChessBoard, prediction: f64)
    {
        self.positions += 1;

        let error = prediction - f64::from(board.score);
        let probability = sigmoid(prediction / f64::from(SCALE));
        let teacher = sigmoid(f64::from(board.score) / f64::from(SCALE));
        let result = f64::from(board.result) / 2.0;

        update_mean(&mut self.mean_prediction_cp, prediction, self.positions);
        update_mean(&mut self.bias_cp, error, self.positions);
        update_mean(&mut self.mean_absolute_error_cp, error.abs(), self.positions);
        update_mean(&mut self.result_squared_error, (probability - result).powi(2), self.positions);
        update_mean(&mut self.teacher_probability_squared_error, (probability - teacher).powi(2), self.positions);

        self.squared_error_sum += error * error;
        self.root_mean_squared_error_cp = (self.squared_error_sum / self.positions as f64).sqrt();
    }
}

fn update_mean(mean: &mut f64, value: f64, count: u64) { *mean += (value - *mean) / count as f64; }

fn sigmoid(value: f64) -> f64 { 1.0 / (1.0 + (-value).exp()) }
