use std::{fmt, str::FromStr};

use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{Chess768, SparseInputType},
    },
    wdl::WdlScheduler,
};
use bullet_trainer::model::{DenseInput, ModelDefinition, ModelInputs, ModelInputsMapper, SavedFormat, SparseInput};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const SCALE: f32 = 400.0;
pub const QA: i16 = 255;
pub const QB: i16 = 64;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Architecture
{
    #[serde(alias = "Starshard")]
    Starshard,
    #[serde(alias = "Aurora")]
    Aurora,
    #[serde(alias = "Flare")]
    Flare,
    #[serde(alias = "Corona")]
    Corona,
}

type Inputs = ModelInputs<((SparseInput, SparseInput), DenseInput<f32>)>;

fn inputs() -> Inputs
{
    ModelInputs::default().add_sparse("stm", (768, 1), 32).add_sparse("ntm", (768, 1), 32).add_dense("targets", (1, 1))
}

impl Architecture
{
    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Starshard => "Starshard",
            Self::Aurora => "Aurora",
            Self::Flare => "Flare",
            Self::Corona => "Corona"
        }
    }

    pub const fn hidden(self) -> usize
    {
        match self {
            Self::Starshard => 256,
            Self::Aurora => 384,
            Self::Flare | Self::Corona => 768,
        }
    }

    pub const fn payload_bytes(self) -> usize { (771 * self.hidden() + 1) * size_of::<i16>() }

    pub fn definition(self) -> ModelDefinition
    {
        ModelDefinition::build(&inputs(), |builder, ((stm, ntm), targets)| {
            let l0 = builder.new_affine("l0", 768, self.hidden());
            let l1 = builder.new_affine("l1", 2 * self.hidden(), 1);

            let y0 = l0.forward(stm);
            let y1 = l0.forward(ntm);

            let (y0, y1) = match self {
                Self::Starshard | Self::Aurora | Self::Flare => (y0.crelu(), y1.crelu()),
                _ => (y0.screlu(), y1.screlu())
            };

            let y0 = y0.concat(y1);

            let y = l1.forward(y0);

            let loss = y.sigmoid().squared_error(targets).reduce_sum_batch();

            (Some(loss), vec![("output".to_owned(), y)])
        })
    }

    pub fn mapper(self, wdl: impl WdlScheduler) -> ModelInputsMapper<ChessBoard>
    {
        ModelInputsMapper::build(&inputs(), move |position: &ChessBoard, step, ((stm, ntm), targets)| {
            let mut count = 0;

            Chess768.map_features(position, |ours, theirs| {
                stm[count] = ours as i32;
                ntm[count] = theirs as i32;
                count += 1;
            });

            if count < Chess768.max_active() {
                stm[count] = -1;
                ntm[count] = -1;
            }

            let blend = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
            let score = 1.0 / (1.0 + (-f32::from(position.score) / SCALE).exp());

            targets[0] = blend * (f32::from(position.result) / 2.0) + (1.0 - blend) * score;
        })
    }

    pub fn saved_format(self) -> Vec<SavedFormat>
    {
        [("l0w", QA), ("l0b", QA), ("l1w", QB), ("l1b", QA * QB)]
            .into_iter()
            .map(|(name, scale)| SavedFormat::id(name).round().quantise::<i16>(scale))
            .collect()
    }
}

impl fmt::Display for Architecture
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.name()) }
}

impl FromStr for Architecture
{
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err>
    {
        match value.to_ascii_lowercase().as_str() {
            "starshard" => Ok(Self::Starshard),
            "aurora" => Ok(Self::Aurora),
            "flare" => Ok(Self::Flare),
            "corona" => Ok(Self::Corona),
            _ => anyhow::bail!("Unknown architecture '{value}'"),
        }
    }
}
