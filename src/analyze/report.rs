use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    data::{Filter, Outcome, is_zero, pipeline::Sampling, select::Counters},
    model::Architecture,
};

pub const SCHEMA: &str = "bruce.analysis/1";

#[derive(Serialize)]
pub struct Report
{
    pub schema: &'static str,
    pub bruce: &'static str,
    pub created: String,
    pub complete: bool,
    pub scale_cp: u16,
    pub sampling: Sampling,
    pub inputs: Vec<InputSummary>,
    pub combined: Combined,
}

#[derive(Default, Serialize)]
pub struct InputSummary
{
    pub path: String,
    pub bytes: u64,
    pub bytes_read: u64,
    pub games: u64,
    pub positions: u64,
    pub complete: bool,
}

#[derive(Serialize)]
pub struct Combined
{
    pub games: GamesReport,
    pub scores: ScoresReport,
    pub calibration: CalibrationReport,
    pub filter: Option<FilterReport>,
    pub network: Option<NetworkReport>,
}

#[derive(Serialize)]
pub struct GamesReport
{
    pub games: u64,
    pub positions: u64,
    pub bytes_per_position: Option<f64>,
    pub results: Results<u64>,
    pub length: Length,
    pub mean_length_by_result: Results<Option<f64>>,
    pub first_fullmove: Values,
    pub black_moves_first: u64,
    pub final_eval_agrees: u64,
}

#[derive(Clone, Copy, Serialize)]
pub struct Results<T>
{
    pub white: T,
    pub draw: T,
    pub black: T,
}

impl<T: Copy> From<[T; 3]> for Results<T>
{
    fn from(counts: [T; 3]) -> Self
    {
        let of = |outcome: Outcome| counts[outcome as usize];
        Self { white: of(Outcome::WhiteWin), draw: of(Outcome::Draw), black: of(Outcome::BlackWin) }
    }
}

#[derive(Serialize)]
pub struct Length
{
    pub mean: Option<f64>,
    pub percentiles: BTreeMap<u8, u64>,
    pub max: u64,
    pub histogram: Bins,
}

#[derive(Serialize)]
pub struct Bins
{
    pub bin_width: usize,
    pub counts: Vec<u64>,
}

#[derive(Serialize)]
pub struct Values
{
    pub values: Vec<u32>,
    pub counts: Vec<u64>,
}

#[derive(Serialize)]
pub struct ScoresReport
{
    pub positions: u64,
    pub histogram_cp: Edges,
    pub forced: Forced,
    #[serde(skip_serializing_if = "is_zero")]
    pub capped: u64,
    pub large_evals: LargeEvals,
    pub win_probability: Probabilities,
    pub ply: Bins,
    pub pieces: Vec<PieceRow>,
}

#[derive(Serialize)]
pub struct Edges
{
    pub edges: Vec<i32>,
    pub counts: Vec<u64>,
}

#[derive(Serialize)]
pub struct Forced
{
    pub minimum: u64,
    pub maximum: u64,
}

#[derive(Serialize)]
pub struct LargeEvals
{
    pub at_least_4000: u64,
    pub at_least_8000: u64,
}

#[derive(Serialize)]
pub struct Probabilities
{
    pub bins: usize,
    pub counts: Vec<u64>,
}

#[derive(Serialize)]
pub struct PieceRow
{
    pub pieces: u8,
    pub positions: u64,
    #[serde(flatten)]
    pub results: Results<u64>,
}

#[derive(Serialize)]
pub struct CalibrationReport
{
    pub positions: u64,
    pub brier: Option<f64>,
    pub mean_absolute: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub buckets: Vec<Bucket>,
    pub by_ply: Vec<PlyPhase>,
}

#[derive(Serialize)]
pub struct Bucket
{
    pub from: f64,
    pub to: f64,
    pub positions: u64,
    pub mean_probability: Option<f64>,
    pub mean_result: Option<f64>,
    #[serde(flatten)]
    pub results: Results<u64>,
}

#[derive(Serialize)]
pub struct PlyPhase
{
    pub from: u16,
    pub to: Option<u16>,
    pub positions: u64,
    pub brier: Option<f64>,
    pub mean_absolute: Option<f64>,
}

#[derive(Serialize)]
pub struct FilterReport
{
    pub source: String,
    pub rules: Option<Filter>,
    pub positions: u64,
    pub accepted: u64,
    pub effects: Vec<Effect>,
    pub policy: Option<PolicyReport>,
    pub schedule: TrainingSchedule,
    pub counters: Counters,
}

#[derive(Serialize)]
pub struct Effect
{
    pub rule: &'static str,
    pub setting: Option<String>,
    pub first: u64,
    pub left_after: u64,
}

#[derive(Serialize)]
pub struct PolicyReport
{
    pub policy: String,
    pub games: u64,
    pub flagged_games: u64,
    pub dropped: u64,
    pub substituted: u64,
    pub distrusted: u64,
}

#[derive(Serialize)]
pub struct TrainingSchedule
{
    pub superbatches: usize,
    pub batches_per_superbatch: usize,
    pub batch_size: usize,
    pub positions: u64,
    pub passes: Option<f64>,
}

#[derive(Serialize)]
pub struct NetworkReport
{
    pub checkpoint: String,
    pub architecture: Option<Architecture>,
    pub evaluated: bool,
    pub config: Option<String>,
    pub filtered: bool,
    pub batch_size: usize,
    pub positions: u64,
    pub score: Option<f64>,
    pub result: Option<f64>,
    pub ece: Option<f64>,
    pub calibration: Vec<ReliabilityBin>,
}

#[derive(Serialize)]
pub struct ReliabilityBin
{
    pub from: f64,
    pub to: f64,
    pub positions: u64,
    pub mean_prediction: Option<f64>,
    pub mean_result: Option<f64>,
}
