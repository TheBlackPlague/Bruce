use std::fmt;

use bullet_lib::game::formats::bulletformat::ChessBoard;
use serde::{Deserialize, Serialize};

use super::{Game, game::narrow};

const DEFAULT_LOSS: f32 = 0.30;
pub const AGREEMENT: u64 = 0x6167_7265_656d_656e;

fn mover_loss(game: &Game, ply: usize) -> f32
{
    let (before, after) = (game.win_probability(ply), game.win_probability(ply + 1));
    if game.white_to_move(ply) { before - after } else { after - before }
}

pub fn outcome_probabilities(score: f64, offset: f64, spread: f64) -> [f64; 3]
{
    let sigmoid = |value: f64| 1.0 / (1.0 + (-value).exp());
    let magnitude = score.abs();
    let behind = sigmoid((-magnitude - offset) / spread);
    let ahead = sigmoid((magnitude - offset) / spread);
    let draw = sigmoid((offset - magnitude) / spread) - behind;
    if score < 0.0 { [ahead, draw, behind] } else { [behind, draw, ahead] }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Detector
{
    Swing
    {
        loss: f32
    },
}

impl Detector
{
    pub fn assess(&self, game: &Game, events: &mut Vec<u16>)
    {
        let Self::Swing { loss } = *self;
        events.clear();

        let plies = 0..game.plies.len().saturating_sub(1);
        events.extend(plies.filter(|&ply| mover_loss(game, ply) >= loss).map(narrow));
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelAction
{
    #[default]
    Keep,
    DropBefore,
    Relabel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AgreementAction
{
    Skip,
    Distrust,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Agreement
{
    pub action: AgreementAction,
    pub offset: f32,
    pub spread: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(try_from = "Table")]
pub struct LabelPolicy
{
    pub action: LabelAction,
    pub detector: Detector,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agreement: Option<Agreement>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table
{
    #[serde(rename = "type")]
    action: LabelAction,
    #[serde(default)]
    blunder: Option<Blunder>,
    #[serde(default)]
    agreement: Option<Agreement>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Blunder
{
    #[serde(default)]
    swing: Option<f32>,
}

impl TryFrom<Table> for LabelPolicy
{
    type Error = &'static str;

    fn try_from(table: Table) -> Result<Self, Self::Error>
    {
        let mut policy = Self { action: table.action, agreement: table.agreement, ..Self::default() };

        if let Some(Blunder { swing }) = table.blunder {
            let loss = swing.unwrap_or(DEFAULT_LOSS);
            if !(loss > 0.0 && loss <= 1.0) {
                return Err("Blunder swing must be greater than 0 and at most 1");
            }
            if policy.action != LabelAction::Keep {
                policy.detector = Detector::Swing { loss };
            }
        }

        Ok(policy)
    }
}

impl Default for LabelPolicy
{
    fn default() -> Self
    {
        Self { action: LabelAction::Keep, detector: Detector::Swing { loss: DEFAULT_LOSS }, agreement: None }
    }
}

impl LabelPolicy
{
    pub fn is_keep(&self) -> bool { self.action == LabelAction::Keep && self.agreement.is_none() }

    pub fn drops(&self) -> bool
    {
        self.action == LabelAction::DropBefore
            || self.agreement.is_some_and(|agreement| agreement.action == AgreementAction::Skip)
    }

    pub fn labels_into(&self, game: &Game, events: &mut Vec<u16>, labels: &mut Vec<PlyLabel>)
    {
        labels.clear();
        labels.resize(game.plies.len(), PlyLabel::UNCHANGED);
        if self.action == LabelAction::Keep {
            return;
        }

        self.detector.assess(game, events);
        if self.action == LabelAction::DropBefore {
            let trusted_from = events.last().map_or(0, |last| last.saturating_add(1));
            labels[..usize::from(trusted_from)].fill(PlyLabel::DROPPED);
            return;
        }

        let mut first = 0;
        for event in events.iter().map(|&event| usize::from(event)) {
            let white = game.win_probability(event);
            for (ply, label) in labels.iter_mut().enumerate().take(event + 1).skip(first) {
                let probability = if game.white_to_move(ply) { white } else { 1.0 - white };
                let substitute = Some((probability * 255.0).round() as u8);
                *label = PlyLabel { keep: true, label: Label { distrust: 0, substitute } };
            }
            first = event + 1;
        }
    }
}

impl fmt::Display for LabelPolicy
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        let Detector::Swing { loss } = self.detector;
        let action = match self.action {
            LabelAction::Keep => None,
            LabelAction::DropBefore => Some("drop_before"),
            LabelAction::Relabel => Some("relabel"),
        };
        match action {
            None => formatter.write_str("keep")?,
            Some(action) => {
                let short = format!("{loss:.2}");
                let loss: &dyn fmt::Display = if short.parse() == Ok(loss) { &short } else { &loss };
                write!(formatter, "{action} swing:{loss}")?;
            },
        }

        if let Some(Agreement { action, offset, spread }) = self.agreement {
            let action = match action {
                AgreementAction::Skip => "skip",
                AgreementAction::Distrust => "distrust",
            };
            write!(formatter, " + agreement {action} {offset:.0}/{spread:.0}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlyLabel
{
    pub keep: bool,
    pub label: Label,
}

impl PlyLabel
{
    pub const UNCHANGED: Self = Self { keep: true, label: Label::RESULT };
    const DROPPED: Self = Self { keep: false, label: Label::RESULT };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label
{
    pub distrust: u8,
    pub substitute: Option<u8>,
}

impl Label
{
    pub const RESULT: Self = Self { distrust: 0, substitute: None };
    const SUBSTITUTE: u8 = 1;

    pub fn read(board: &ChessBoard) -> Self
    {
        let [distrust, substitute, flags] = board.extra;
        Self { distrust, substitute: (flags & Self::SUBSTITUTE != 0).then_some(substitute) }
    }

    pub fn write(self, board: &mut ChessBoard)
    {
        board.extra = self.substitute.map_or([self.distrust, 0, 0], |target| [self.distrust, target, Self::SUBSTITUTE]);
    }

    pub fn distrusted(self, probability: f64) -> Self
    {
        let agreement = 255 - (255.0 * probability).round() as u8;
        let trust = (255 - u32::from(self.distrust)) * (255 - u32::from(agreement));
        Self { distrust: (255 - (trust + 127) / 255) as u8, ..self }
    }

    pub fn target(self, board: &ChessBoard, blend: f32) -> f32
    {
        let score = 1.0 / (1.0 + (-f32::from(board.score) / 400.0).exp());
        let outcome = self.substitute.map_or(f32::from(board.result) / 2.0, |q| f32::from(q) / 255.0);
        let weight = blend * (1.0 - f32::from(self.distrust) / 255.0);
        weight * outcome + (1.0 - weight) * score
    }
}
