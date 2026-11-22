use std::{array, collections::BTreeMap, path::Path};

use super::report::{
    Bins, Bucket, CalibrationReport, Edges, Effect, FilterReport, Forced, GamesReport, LargeEvals, Length, PieceRow,
    PlyPhase, PolicyReport, Probabilities, Results, ScoresReport, TrainingSchedule, Values,
};
use crate::{
    config::Config,
    data::{
        Filter, Game, Outcome,
        select::{Counters, Rule},
    },
    model::sigmoid,
};

pub const SCALE_CP: u16 = 400;
pub const BUCKETS: usize = 20;

const FIXED: u32 = 24;
const ONE: f64 = (1_u64 << FIXED) as f64;
const HALF: u32 = 1 << (FIXED - 1);
const QUARTER: u32 = 1 << (FIXED - 2);
const SCORES: usize = 1 << 16;
const BIN: usize = 10;
const BINS: usize = 101;
const PERCENTILES: [u8; 9] = [1, 5, 10, 25, 50, 75, 90, 95, 99];
const EXACT: usize = 1024;
const EDGES_CP: [i32; 23] = [
    -32768, -8000, -4000, -2000, -1000, -600, -400, -200, -100, -50, -20, 0, 20, 50, 100, 200, 400, 600, 1000, 2000,
    4000, 8000, 32768,
];
const PROBABILITY_BINS: usize = 50;
const LARGE_CP: [i32; 2] = [4000, 8000];
const PLY_BIN: usize = 10;
const PLY_BINS: usize = 101;
const PIECES: usize = 33;
const PLY_EDGES: [u16; 7] = [0, 20, 40, 60, 80, 120, 160];

pub struct Tables
{
    probability: Box<[u32]>,
    squared: Box<[[u32; 3]]>,
    absolute: Box<[[u32; 3]]>,
}

impl Tables
{
    pub fn new() -> Self
    {
        let fixed = |value: f64| (value * ONE).round() as u32;
        let results = [0.0, 0.5, 1.0];
        let curve: Vec<f64> = (0..SCORES).map(|index| sigmoid(f64::from(score(index)) / f64::from(SCALE_CP))).collect();

        Self {
            probability: curve.iter().map(|&p| fixed(p)).collect(),
            squared: curve.iter().map(|&p| results.map(|result| fixed((result - p) * (result - p)))).collect(),
            absolute: curve.iter().map(|&p| results.map(|result| fixed((result - p).abs()))).collect(),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Sums
{
    positions: u64,
    squared: u128,
    absolute: u128,
    probability: u128,
    results: u64,
    counts: [u64; 3],
}

impl Sums
{
    fn add(&mut self, other: &Self)
    {
        self.positions += other.positions;
        self.squared += other.squared;
        self.absolute += other.absolute;
        self.probability += other.probability;
        self.results += other.results;
        add(&mut self.counts, &other.counts);
    }

    fn mean(&self, sum: u128) -> Option<f64> { (self.positions > 0).then(|| sum as f64 / ONE / self.positions as f64) }

    fn mean_result(&self) -> Option<f64>
    {
        (self.positions > 0).then(|| self.results as f64 / 2.0 / self.positions as f64)
    }
}

pub struct Stats
{
    pub games: u64,
    pub positions: u64,
    pub counters: Counters,
    results: [u64; 3],
    lengths: [u64; BINS],
    exact: [u64; EXACT + 1],
    longest: u64,
    length_sums: [u64; 3],
    first_moves: BTreeMap<u32, u64>,
    black_moves_first: u64,
    final_eval_agrees: u64,
    saturated: u64,
    capped: u64,
    plies: [u64; PLY_BINS],
    pieces: [[u64; 3]; PIECES],
    phases: [Sums; PLY_EDGES.len()],
    table: Box<[[u64; 3]]>,
}

impl Stats
{
    pub fn new() -> Self
    {
        Self {
            games: 0,
            positions: 0,
            counters: Counters::default(),
            results: [0; 3],
            lengths: [0; BINS],
            exact: [0; EXACT + 1],
            longest: 0,
            length_sums: [0; 3],
            first_moves: BTreeMap::new(),
            black_moves_first: 0,
            final_eval_agrees: 0,
            saturated: 0,
            capped: 0,
            plies: [0; PLY_BINS],
            pieces: [[0; 3]; PIECES],
            phases: [Sums::default(); PLY_EDGES.len()],
            table: vec![[0; 3]; SCORES].into(),
        }
    }

    pub fn observe(&mut self, game: &Game, tables: &Tables)
    {
        let result = usize::from(game.outcome.relative(false));
        let length = game.plies.len();
        self.games += 1;
        self.positions += length as u64;
        self.results[result] += 1;
        self.lengths[(length / BIN).min(BINS - 1)] += 1;
        self.exact[length.min(EXACT)] += 1;
        self.longest = self.longest.max(length as u64);
        self.length_sums[result] += length as u64;
        self.black_moves_first += u64::from(!game.white_moves_first);
        *self.first_moves.entry(u32::from(game.fen_ply(0)) / 2 + 1).or_default() += 1;
        if let Some(last) = game.plies.last() {
            let probability = tables.probability[index(last.score)];
            self.final_eval_agrees += u64::from(agrees(game.outcome, probability));
        }

        for (ply, entry) in game.plies.iter().enumerate() {
            let index = index(entry.score);
            self.table[index][result] += 1;
            self.pieces[usize::from(entry.pieces).min(PIECES - 1)][result] += 1;
            let game_ply = game.fen_ply(ply);
            self.plies[(usize::from(game_ply) / PLY_BIN).min(PLY_BINS - 1)] += 1;
            let phase = &mut self.phases[PLY_EDGES.partition_point(|&edge| edge <= game_ply) - 1];
            phase.positions += 1;
            phase.squared += u128::from(tables.squared[index][result]);
            phase.absolute += u128::from(tables.absolute[index][result]);
        }
        self.saturated += u64::from(game.saturated);
        self.capped += u64::from(game.capped);
    }

    pub fn merge(&mut self, other: Self)
    {
        self.games += other.games;
        self.positions += other.positions;
        self.counters += other.counters;
        add(&mut self.results, &other.results);
        add(&mut self.lengths, &other.lengths);
        add(&mut self.exact, &other.exact);
        self.longest = self.longest.max(other.longest);
        add(&mut self.length_sums, &other.length_sums);
        for (fullmove, count) in other.first_moves {
            *self.first_moves.entry(fullmove).or_default() += count;
        }
        self.black_moves_first += other.black_moves_first;
        self.final_eval_agrees += other.final_eval_agrees;
        self.saturated += other.saturated;
        self.capped += other.capped;
        add(&mut self.plies, &other.plies);
        add(self.pieces.as_flattened_mut(), other.pieces.as_flattened());
        for (phase, other) in self.phases.iter_mut().zip(&other.phases) {
            phase.add(other);
        }
        add(self.table.as_flattened_mut(), other.table.as_flattened());
    }

    pub fn games(&self, bytes_read: u64) -> GamesReport
    {
        let mean = |sum: u64, count: u64| (count > 0).then(|| sum as f64 / count as f64);
        let means = array::from_fn(|side| mean(self.length_sums[side], self.results[side]));

        GamesReport {
            games: self.games,
            positions: self.positions,
            bytes_per_position: mean(bytes_read, self.positions),
            results: Results::from(self.results),
            length: Length {
                mean: mean(self.length_sums.iter().sum(), self.games),
                percentiles: (PERCENTILES.into_iter())
                    .filter_map(|wanted| Some((wanted, percentile(&self.exact, self.games, wanted)?)))
                    .collect(),
                max: self.longest,
                histogram: Bins { bin_width: BIN, counts: self.lengths.to_vec() },
            },
            mean_length_by_result: Results::from(means),
            first_fullmove: Values {
                values: self.first_moves.keys().copied().collect(),
                counts: self.first_moves.values().copied().collect(),
            },
            black_moves_first: self.black_moves_first,
            final_eval_agrees: self.final_eval_agrees,
        }
    }

    pub fn scores(&self, tables: &Tables) -> ScoresReport
    {
        let mut positions = 0;
        let mut histogram = vec![0; EDGES_CP.len() - 1];
        let mut probabilities = vec![0; PROBABILITY_BINS];
        let mut large = [0; LARGE_CP.len()];
        for (index, counts) in self.table.iter().enumerate() {
            let count: u64 = counts.iter().sum();
            if count == 0 {
                continue;
            }

            let score = i32::from(score(index));
            positions += count;
            histogram[EDGES_CP.partition_point(|&edge| edge <= score) - 1] += count;
            probabilities[probability_bin(tables.probability[index], PROBABILITY_BINS)] += count;
            for (large, threshold) in large.iter_mut().zip(LARGE_CP) {
                *large += if score.abs() >= threshold { count } else { 0 };
            }
        }

        ScoresReport {
            positions,
            histogram_cp: Edges { edges: EDGES_CP.to_vec(), counts: histogram },
            forced: Forced { minimum: self.saturated, maximum: self.table[index(i16::MAX)].iter().sum() },
            capped: self.capped,
            large_evals: LargeEvals { at_least_4000: large[0], at_least_8000: large[1] },
            win_probability: Probabilities { bins: PROBABILITY_BINS, counts: probabilities },
            ply: Bins { bin_width: PLY_BIN, counts: self.plies.to_vec() },
            pieces: ((0..=32_u8).zip(&self.pieces).rev())
                .filter(|(_, counts)| counts.iter().any(|&count| count > 0))
                .map(|(pieces, &counts)| PieceRow {
                    pieces,
                    positions: counts.iter().sum(),
                    results: Results::from(counts),
                })
                .collect(),
        }
    }

    pub fn calibration(&self, tables: &Tables) -> CalibrationReport
    {
        let mut total = Sums::default();
        let mut buckets = [Sums::default(); BUCKETS];
        for (index, counts) in self.table.iter().enumerate() {
            let positions: u64 = counts.iter().sum();
            if positions == 0 {
                continue;
            }

            let probability = tables.probability[index];
            let mut sums = Sums {
                positions,
                probability: u128::from(positions) * u128::from(probability),
                results: counts[Outcome::Draw as usize] + 2 * counts[Outcome::WhiteWin as usize],
                counts: *counts,
                ..Sums::default()
            };
            for (result, &count) in counts.iter().enumerate() {
                sums.squared += u128::from(count) * u128::from(tables.squared[index][result]);
                sums.absolute += u128::from(count) * u128::from(tables.absolute[index][result]);
            }

            total.add(&sums);
            buckets[probability_bin(probability, BUCKETS)].add(&sums);
        }

        let gaps: f64 = (buckets.iter())
            .filter_map(|bucket| {
                Some(bucket.positions as f64 * (bucket.mean(bucket.probability)? - bucket.mean_result()?).abs())
            })
            .sum();

        CalibrationReport {
            positions: total.positions,
            brier: total.mean(total.squared),
            mean_absolute: total.mean(total.absolute),
            expected_calibration_error: (total.positions > 0).then(|| gaps / total.positions as f64),
            buckets: (0..)
                .zip(&buckets)
                .map(|(bucket, sums)| Bucket {
                    from: f64::from(bucket) / BUCKETS as f64,
                    to: f64::from(bucket + 1) / BUCKETS as f64,
                    positions: sums.positions,
                    mean_probability: sums.mean(sums.probability),
                    mean_result: sums.mean_result(),
                    results: Results::from(sums.counts),
                })
                .collect(),
            by_ply: (PLY_EDGES.iter().zip(&self.phases).enumerate())
                .map(|(phase, (&from, sums))| PlyPhase {
                    from,
                    to: PLY_EDGES.get(phase + 1).copied(),
                    positions: sums.positions,
                    brier: sums.mean(sums.squared),
                    mean_absolute: sums.mean(sums.absolute),
                })
                .collect(),
        }
    }

    pub fn filter(&self, path: &Path, config: &Config, passes: bool) -> FilterReport
    {
        let counters = self.counters;
        let data = &config.training.data;
        let positions = counters.plies;
        let mut left = positions;
        let effects = (Rule::ALL.into_iter())
            .map(|(rule, name)| {
                let first = counters.first_rejection[rule as usize];
                let setting = data.filter.as_ref().and_then(|filter| setting(filter, rule, name));
                left -= first;
                Effect { rule: name, setting, first, left_after: left }
            })
            .collect();

        let batches = [config.training.superbatches, data.batch.per_superbatch, data.batch.size];
        let total = batches.into_iter().fold(1_u64, |product, factor| product.saturating_mul(factor as u64));
        FilterReport {
            source: path.display().to_string(),
            rules: data.filter.clone(),
            positions,
            accepted: counters.accepted,
            effects,
            policy: (!data.labels.is_keep()).then(|| PolicyReport {
                policy: data.labels.to_string(),
                games: counters.games,
                flagged_games: counters.flagged_games,
                dropped: counters.dropped_by_label,
                substituted: counters.substituted,
                distrusted: counters.distrusted,
            }),
            schedule: TrainingSchedule {
                superbatches: config.training.superbatches,
                batches_per_superbatch: data.batch.per_superbatch,
                batch_size: data.batch.size,
                positions: total,
                passes: (passes && counters.accepted > 0).then(|| total as f64 / counters.accepted as f64),
            },
            counters,
        }
    }
}

fn setting(filter: &Filter, rule: Rule, name: &str) -> Option<String>
{
    let bound = |value: u32, active: bool| active.then(|| format!("{name} = {value}"));
    let flag = |active: bool| active.then(|| name.to_owned());

    match rule {
        Rule::MinimumPly => bound(filter.minimum_ply.into(), filter.minimum_ply > 0),
        Rule::MaximumPly => filter.maximum_ply.and_then(|maximum| bound(maximum.into(), true)),
        Rule::MinimumGamePly => bound(filter.minimum_game_ply.into(), filter.minimum_game_ply > 0),
        Rule::MinimumPieces => bound(filter.minimum_pieces, filter.minimum_pieces > 0),
        Rule::MinimumEval => bound(filter.minimum_eval, filter.minimum_eval > 0),
        Rule::MaximumEval => bound(filter.maximum_eval, filter.maximum_eval < Filter::default().maximum_eval),
        Rule::Tactical => flag(filter.exclude_tactical),
        Rule::Check => flag(filter.exclude_check),
        Rule::RandomSkip => (filter.random_skip > 0.0).then(|| format!("{name} = {}", filter.random_skip)),
    }
}

fn percentile(exact: &[u64], games: u64, percentile: u8) -> Option<u64>
{
    let rank = (u128::from(games) * u128::from(percentile)).div_ceil(100).max(1);
    let mut seen = 0;
    (0..).zip(exact).find_map(|(length, &count)| {
        seen += u128::from(count);
        (seen >= rank).then_some(length)
    })
}

fn agrees(outcome: Outcome, probability: u32) -> bool
{
    match outcome {
        Outcome::WhiteWin => probability > HALF,
        Outcome::BlackWin => probability < HALF,
        Outcome::Draw => (QUARTER..=3 * QUARTER).contains(&probability),
    }
}

fn probability_bin(probability: u32, bins: usize) -> usize
{
    ((u64::from(probability) * bins as u64) >> FIXED).min(bins as u64 - 1) as usize
}

fn index(score: i16) -> usize { usize::from(score.cast_unsigned() ^ 0x8000) }

fn score(index: usize) -> i16 { (index as u16 ^ 0x8000).cast_signed() }

fn add(total: &mut [u64], other: &[u64])
{
    for (total, other) in total.iter_mut().zip(other) {
        *total += other;
    }
}
