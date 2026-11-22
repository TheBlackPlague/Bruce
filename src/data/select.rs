use std::{array, ops::AddAssign};

use bullet_lib::game::formats::bulletformat::ChessBoard;
use serde::{Deserialize, Serialize};

use super::{
    DataError, Game, Ply, Source,
    game::{narrow, relative},
    is_zero,
    label::{AGREEMENT, Agreement, AgreementAction, Label, LabelAction, LabelPolicy, PlyLabel, outcome_probabilities},
    pipeline::splitmix64,
};

const WHOLE: f64 = 18_446_744_073_709_551_616.0;
const SKIP: u64 = 0x7261_6e64_736b_6970;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Filter
{
    pub minimum_ply: u16,
    pub maximum_ply: Option<u16>,
    #[serde(skip_serializing_if = "is_zero")]
    pub minimum_game_ply: u16,
    pub minimum_pieces: u32,
    pub minimum_eval: u32,
    pub maximum_eval: u32,
    pub exclude_tactical: bool,
    pub exclude_check: bool,
    #[serde(skip_serializing_if = "is_zero")]
    pub random_skip: f64,
}

impl Default for Filter
{
    fn default() -> Self
    {
        Self {
            minimum_ply: 0,
            maximum_ply: None,
            minimum_game_ply: 0,
            minimum_pieces: 0,
            minimum_eval: 0,
            maximum_eval: 32768,
            exclude_tactical: false,
            exclude_check: false,
            random_skip: 0.0,
        }
    }
}

impl Filter
{
    pub fn first_rejection(
        &self,
        ply: Ply,
        fen_ply: u16,
        game_ply: u16,
        check: impl Fn() -> bool,
        draw: impl Fn() -> f64,
    ) -> Option<Rule>
    {
        let magnitude = u32::from(ply.score.unsigned_abs());
        Rule::ALL.into_iter().map(|(rule, _)| rule).find(|rule| match rule {
            Rule::MinimumPly => fen_ply < self.minimum_ply,
            Rule::MaximumPly => self.maximum_ply.is_some_and(|maximum| fen_ply > maximum),
            Rule::MinimumGamePly => game_ply < self.minimum_game_ply,
            Rule::MinimumPieces => u32::from(ply.pieces) < self.minimum_pieces,
            Rule::MinimumEval => magnitude < self.minimum_eval,
            Rule::MaximumEval => magnitude >= self.maximum_eval,
            Rule::Tactical => self.exclude_tactical && ply.tactical,
            Rule::Check => self.exclude_check && check(),
            Rule::RandomSkip => self.random_skip > 0.0 && draw() < self.random_skip,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule
{
    MinimumPly,
    MaximumPly,
    MinimumGamePly,
    MinimumPieces,
    MinimumEval,
    MaximumEval,
    Tactical,
    Check,
    RandomSkip,
}

impl Rule
{
    pub const ALL: [(Rule, &str); 9] = [
        (Self::MinimumPly, "minimum_ply"),
        (Self::MaximumPly, "maximum_ply"),
        (Self::MinimumGamePly, "minimum_game_ply"),
        (Self::MinimumPieces, "minimum_pieces"),
        (Self::MinimumEval, "minimum_eval"),
        (Self::MaximumEval, "maximum_eval"),
        (Self::Tactical, "exclude_tactical"),
        (Self::Check, "exclude_check"),
        (Self::RandomSkip, "random_skip"),
    ];
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Counters
{
    pub games: u64,
    pub plies: u64,
    pub accepted: u64,
    pub first_rejection: [u64; Rule::ALL.len()],
    pub saturated_scores: u64,
    pub flagged_games: u64,
    pub distrusted: u64,
    pub substituted: u64,
    pub dropped_by_label: u64,
}

impl AddAssign for Counters
{
    fn add_assign(&mut self, other: Self)
    {
        self.games += other.games;
        self.plies += other.plies;
        self.accepted += other.accepted;
        self.first_rejection = array::from_fn(|rule| self.first_rejection[rule] + other.first_rejection[rule]);
        self.saturated_scores += other.saturated_scores;
        self.flagged_games += other.flagged_games;
        self.distrusted += other.distrusted;
        self.substituted += other.substituted;
        self.dropped_by_label += other.dropped_by_label;
    }
}

pub fn held(seed: u64, key: u64, offset: u64, ordinal: u64) -> u64
{
    splitmix64(splitmix64(splitmix64(seed ^ key) ^ offset) ^ ordinal)
}

pub fn pass_key(key: u64, pass: u64) -> u64 { if pass == 0 { key } else { held(AGREEMENT, key, pass, 0) } }

#[derive(Clone, Debug, Default)]
pub struct Selection
{
    pub filter: Option<Filter>,
    pub labels: LabelPolicy,
    pub key: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Scratch(Vec<u16>, Vec<PlyLabel>);

impl Selection
{
    pub fn identified(self, source: &Source) -> Result<Self, DataError>
    {
        let skips = self.filter.as_ref().is_some_and(|filter| filter.random_skip > 0.0);
        let key = if self.labels.agreement.is_some() || skips { source.identity()?.key() } else { self.key };
        Ok(Self { key, ..self })
    }

    pub fn select_each(
        &self,
        game: &Game,
        counters: Option<&mut Counters>,
        scratch: &mut Scratch,
        mut keep: impl FnMut(usize, ChessBoard),
    ) -> Result<(), DataError>
    {
        let mut ignored = Counters::default();
        let counters = counters.unwrap_or(&mut ignored);
        let Scratch(events, labels) = scratch;
        self.labels.labels_into(game, events, labels);
        counters.games += 1;
        counters.plies += game.plies.len() as u64;
        counters.saturated_scores += u64::from(game.saturated) + u64::from(game.capped);
        counters.flagged_games += u64::from(self.labels.action != LabelAction::Keep && !events.is_empty());

        game.walk(|ply, position| {
            let rejection = self.filter.as_ref().and_then(|filter| {
                filter.first_rejection(
                    game.plies[ply],
                    game.fen_ply(ply),
                    narrow(ply),
                    || position.in_check(),
                    || held(SKIP, self.key, game.offset, (game.index << 16) | ply as u64) as f64 / WHOLE,
                )
            });
            if let Some(rule) = rejection {
                counters.first_rejection[rule as usize] += 1;
                return Ok(());
            }

            let PlyLabel { keep: kept, label } = labels[ply];
            let Some(label) = kept.then_some(label).and_then(|label| self.agree(game, ply, label)) else {
                counters.dropped_by_label += 1;
                return Ok(());
            };

            let mut board = game.board(ply, position)?;
            label.write(&mut board);
            counters.accepted += 1;
            counters.distrusted += u64::from(label.distrust > 0);
            counters.substituted += u64::from(label.substitute.is_some());
            keep(ply, board);

            Ok(())
        })
    }

    pub fn agreement_skips(&self, game: &Game, scratch: &mut Scratch) -> Result<u64, DataError>
    {
        if !self.labels.agreement.is_some_and(|agreement| agreement.action == AgreementAction::Skip) {
            return Ok(0);
        }

        let open = Self { labels: LabelPolicy { agreement: None, ..self.labels }, ..self.clone() };
        let mut skips = 0;
        open.select_each(game, None, scratch, |ply, _| {
            skips += u64::from(self.agree(game, ply, Label::RESULT).is_none());
        })?;

        Ok(skips)
    }

    fn agree(&self, game: &Game, ply: usize, label: Label) -> Option<Label>
    {
        let Some(Agreement { action, offset, spread }) = self.labels.agreement else { return Some(label) };
        let black = !game.white_to_move(ply);
        let score = f64::from(relative(game.plies[ply].score, black));
        let probability = outcome_probabilities(score, f64::from(offset), f64::from(spread))
            [usize::from(game.outcome.relative(black))];

        match action {
            AgreementAction::Skip => {
                let draw = held(AGREEMENT, self.key, game.offset, (game.index << 16) | ply as u64) as f64 / WHOLE;
                (draw < probability).then_some(label)
            },
            AgreementAction::Distrust => Some(label.distrusted(probability)),
        }
    }
}
