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

type Inputs = ModelInputs<((SparseInput, SparseInput), DenseInput<f32>)>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Architecture
{
    #[serde(alias = "Starshard")]
    Starshard,
    #[serde(alias = "Aurora")]
    Aurora,
}

impl Architecture
{
    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Starshard => "Starshard",
            Self::Aurora => "Aurora",
        }
    }

    pub const fn hidden(self) -> usize
    {
        match self {
            Self::Starshard => 256,
            Self::Aurora => 384,
        }
    }

    pub const fn payload_bytes(self) -> usize { (771 * self.hidden() + 1) * size_of::<i16>() }

    pub fn definition(self) -> ModelDefinition
    {
        ModelDefinition::build(&inputs(), |builder, ((stm, ntm), targets)| {
            let feature = builder.new_affine("l0", 768, self.hidden());
            let output = builder.new_affine("l1", self.hidden() * 2, 1);

            let hidden = feature.forward(stm).crelu().concat(feature.forward(ntm).crelu());
            let prediction = output.forward(hidden);
            let loss = prediction.sigmoid().squared_error(targets).reduce_sum_batch();

            (Some(loss), vec![("output".into(), prediction)])
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
            let result = f32::from(position.result) / 2.0;
            targets[0] = blend * result + (1.0 - blend) * score;
        })
    }

    pub fn saved_format(self) -> Vec<SavedFormat>
    {
        vec![
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ]
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
            _ => anyhow::bail!("Unknown architecture '{value}'"),
        }
    }
}

fn inputs() -> Inputs
{
    ModelInputs::default().add_sparse("stm", (768, 1), 32).add_sparse("ntm", (768, 1), 32).add_dense("targets", (1, 1))
}
