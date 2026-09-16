use std::path::PathBuf;

use anyhow::{Result, ensure};
use bullet_lib::game::formats::{
    montyformat::chess::{Move, Position},
    sfbinpack::{TrainingDataEntry, chess::r#move::MoveType},
    viriformat::dataformat::Filter,
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    Bullet,
    Monty,
    Sf,
    Viri,
    Text,
    Marlin,
    Cudad,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    pub format: DataFormat,
    pub paths: Vec<PathBuf>,
    #[serde(default = "buffer_default")]
    pub buffer_size_mb: usize,
    #[serde(default = "threads_default")]
    pub loader_threads: usize,
    #[serde(default = "mapping_default")]
    pub mapping_threads: u8,
    #[serde(default)]
    pub filter: Option<FilterConfig>,
}

fn  buffer_default() -> usize { 256 }
fn threads_default() -> usize {  2  }
fn mapping_default() -> u8    {  2  }

impl DataConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.paths.is_empty(),
            "data.paths must contain at least one file"
        );
        ensure!(
            self.buffer_size_mb > 0 && self.buffer_size_mb <= 1_048_576,
            "data.buffer_size_mb must be in 1..=1048576"
        );
        ensure!(
            (1..=256).contains(&self.loader_threads),
            "data.loader_threads must be in 1..=256"
        );
        ensure!(
            self.mapping_threads > 0,
            "data.mapping_threads must be positive"
        );

        if let Some(filter) = &self.filter {
            ensure!(
                filter.max_eval > 0 && filter.max_eval <= 32768,
                "filter.max_eval must be in 1..=32768 (exclusive score bound)"
            );
            ensure!(
                filter.min_pieces <= 32,
                "filter.min_pieces cannot exceed 32"
            );
            ensure!(
                matches!(
                    self.format,
                    DataFormat::Monty | DataFormat::Sf | DataFormat::Viri
                ),
                "filters require Monty, SF or Viri metadata; \
                convert/filter before using Bullet, Marlin, CudAD or text"
            );
        }

        ensure!(
            self.format != DataFormat::Text || self.paths.len() == 1,
            "Bullet's in-memory text loader accepts one file; \
            convert multiple text files to Bullet format first"
        );

        Ok(())
    }

    pub fn resume_note(&self) -> &'static str {
        match self.format {
            DataFormat::Bullet | DataFormat::Marlin | DataFormat::Cudad => {
                "Resume restores optimizer state, schedule and the exact next record offset, \
                provided the files and their order are unchanged. \
                GPU arithmetic may not be bitwise deterministic."
            }
            DataFormat::Text => {
                "Resume restores optimizer state and schedule. \
                Bullet's text loader restarts at the first record; \
                exact data-stream continuation is unavailable. The text dataset is loaded into RAM."
            }
            _ => {
                "Resume restores optimizer state and schedule. Bullet's binpack loaders restart \
                reading and reshuffle; file offsets, shuffle buffers and RNG state are not \
                checkpointed, so exact data-stream continuation is unavailable. \
                Shuffle buffers use host RAM, not GPU VRAM."
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FilterConfig {
    pub min_ply: u16,
    pub min_pieces: u32,
    pub max_eval: u32,
    pub exclude_tactical: bool,
    pub exclude_check: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_ply: 14,
            min_pieces: 0,
            max_eval: 8000,
            exclude_tactical: true,
            exclude_check: true,
        }
    }
}

impl FilterConfig {
    pub fn viri(&self) -> Filter {
        Filter {
            min_ply: u32::from(self.min_ply),
            min_pieces: self.min_pieces,
            max_eval: self.max_eval,
            filter_tactical: self.exclude_tactical,
            filter_check: self.exclude_check,
            ..Filter::UNRESTRICTED
        }
    }

    pub fn monty(&self, pos: &Position, mov: Move, score: i16) -> bool {
        let ply = u32::from(pos.fullm().saturating_sub(1)) * 2 + pos.stm() as u32;

        ply >= u32::from(self.min_ply)
            && pos.occ().count_ones() >= self.min_pieces
            && u32::from(score.unsigned_abs()) < self.max_eval
            && (!self.exclude_tactical || (!mov.is_capture() && !mov.is_promo()))
            && (!self.exclude_check || !pos.in_check())
    }

    pub fn sf(&self, entry: &TrainingDataEntry) -> bool {
        let tactical = matches!(entry.mv.mtype(), MoveType::Promotion | MoveType::EnPassant)
            || (entry.mv.mtype() != MoveType::Castle
                && entry.pos.piece_at(entry.mv.to())
                    != bullet_lib::game::formats::sfbinpack::chess::piece::Piece::none());

        entry.ply >= self.min_ply
            && entry.pos.occupied().bits().count_ones() >= self.min_pieces
            && u32::from(entry.score.unsigned_abs()) < self.max_eval
            && (!self.exclude_tactical || !tactical)
            && (!self.exclude_check || !entry.pos.is_checked(entry.pos.side_to_move()))
    }
}
