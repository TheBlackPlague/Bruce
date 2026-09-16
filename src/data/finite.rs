use std::{
    fs::File,
    io::{BufRead, BufReader, Seek},
    path::Path,
};

use super::{DataFormat, FilterConfig};
use anyhow::{Context, Result, anyhow, ensure};
use bullet_lib::game::formats::{
    bulletformat::{
        BulletFormat, ChessBoard, DataLoader,
        chess::{CudADFormat, MarlinFormat},
    },
    montyformat::MontyValueFormat,
    sfbinpack::{CompressedTrainingDataEntryReader, TrainingDataEntry},
    viriformat::dataformat::{Filter, Game},
};

pub fn valid_sf_score(score: i16) -> bool {
    score != 32002 && score != i16::MIN
}

pub fn validate_board(board: &ChessBoard) -> Result<()> {
    ensure!(board.result <= 2, "invalid result in Bullet record");
    ensure!(
        (2..=32).contains(&board.occ.count_ones()),
        "invalid piece count in Bullet record"
    );

    let mut kings = [0u8; 2];

    for (piece, square) in *board {
        ensure!(piece & 7 <= 5, "invalid piece code in Bullet record");

        if piece & 7 == 5 {
            let color = usize::from(piece & 8 != 0);

            kings[color] += 1;

            let expected = if color == 0 {
                board.ksq
            } else {
                board.opp_ksq ^ 56
            };

            ensure!(
                square == expected,
                "invalid cached king square in Bullet record"
            );
        }
    }

    ensure!(
        kings == [1, 1],
        "a chess record must contain one king per side"
    );

    Ok(())
}

pub fn sf_board(entry: &TrainingDataEntry) -> Result<ChessBoard> {
    let black = entry.pos.side_to_move().ordinal() != 0;

    let score = if black {
        -i32::from(entry.score)
    } else {
        i32::from(entry.score)
    };

    let result = if black {
        1 - entry.result
    } else {
        1 + entry.result
    };

    let text = format!(
        "{} | {} | {}",
        entry.pos.fen().map_err(|e| anyhow!("{e:?}"))?,
        score,
        f32::from(result) / 2.0
    );

    text.parse().map_err(|e: String| anyhow!(e))
}

pub fn visit(
    path: &Path,
    format: DataFormat,
    filter: Option<&FilterConfig>,
    mut callback: impl FnMut(ChessBoard) -> Result<bool>,
    mut progress: impl FnMut(u64),
) -> Result<()> {
    match format {
        DataFormat::Bullet => fixed::< ChessBoard >(path, &mut callback, &mut progress),
        DataFormat::Marlin => fixed::<MarlinFormat>(path, &mut callback, &mut progress),
        DataFormat::Cudad  => fixed::< CudADFormat>(path, &mut callback, &mut progress),

        DataFormat::Text => {
            let mut reader = BufReader::new(File::open(path)?);
            let mut line = String::new();
            let mut line_number = 0;
            let mut bytes = 0;

            while reader.read_line(&mut line)? != 0 {
                line_number += 1;
                bytes += line.len() as u64;
                let board = line
                    .trim()
                    .parse::<ChessBoard>()
                    .map_err(|e| anyhow!("{}:{line_number}: {e}", path.display()))?;

                if callback(board)? {
                    break;
                }

                if line_number % 8192 == 0 {
                    progress(bytes);
                }

                line.clear();
            }

            progress(bytes);

            Ok(())
        }

        DataFormat::Monty => {
            let mut reader = BufReader::new(File::open(path)?);
            let mut games = 0;

            while !reader.fill_buf()?.is_empty() {
                let game = MontyValueFormat::deserialise_from(&mut reader, Vec::new())
                    .with_context(|| format!("decode Monty game in {}", path.display()))?;

                let mut pos = game.startpos;
                for entry in game.moves {
                    if filter.is_none_or(|filter| filter.monty(&pos, entry.best_move, entry.score))
                    {
                        let board =
                            ChessBoard::from_raw(pos.bbs(), pos.stm(), entry.score, game.result)
                                .map_err(|e| anyhow!(e))?;

                        if callback(board)? {
                            return Ok(());
                        }
                    }
                    pos.make(entry.best_move, &game.castling);
                }

                games += 1;

                if games % 256 == 0 {
                    progress(reader.stream_position()?);
                }
            }

            progress(reader.stream_position()?);

            Ok(())
        }

        DataFormat::Sf => {
            let mut reader =
                CompressedTrainingDataEntryReader::new(BufReader::new(File::open(path)?))
                    .map_err(|e| anyhow!("{e:?}"))?;

            let mut count = 0;
            while reader.has_next() {
                let entry = reader.next();
                if valid_sf_score(entry.score)
                    && filter.is_none_or(|filter| filter.sf(&entry))
                    && callback(sf_board(&entry)?)?
                {
                    return Ok(());
                }

                count += 1;

                if count % 8192 == 0 {
                    progress(reader.read_bytes());
                }
            }

            progress(reader.read_bytes());

            Ok(())
        }

        DataFormat::Viri => {
            let mut reader = BufReader::new(File::open(path)?);

            let filter = filter.map_or(Filter::UNRESTRICTED, FilterConfig::viri);

            let mut games = 0;

            while !reader.fill_buf()?.is_empty() {
                let game = Game::deserialise_from(&mut reader, Vec::new())
                    .with_context(|| format!("decode Viri game in {}", path.display()))?;
                let mut stopped = false;

                game.splat_to_bulletformat(
                    |board| {
                        if !stopped {
                            stopped = callback(board)?;
                        }

                        Ok(())
                    },
                    &filter,
                )?;

                if stopped {
                    return Ok(());
                }

                games += 1;

                if games % 256 == 0 {
                    progress(reader.stream_position()?);
                }
            }

            progress(reader.stream_position()?);

            Ok(())
        }
    }
}

fn fixed<T>(
    path: &Path,
    callback: &mut impl FnMut(ChessBoard) -> Result<bool>,
    progress: &mut impl FnMut(u64),
) -> Result<()> where T: BulletFormat, ChessBoard: From<T>,
{
    let loader = DataLoader::<T>::new(path, 8)?;
    let mut result = Ok(());
    let mut stopped = false;

    let mut bytes = T::HEADER_SIZE as u64;

    loader.map_batches(8192, |batch| {
        if stopped || result.is_err() {
            return;
        }

        for &record in batch {
            match callback(ChessBoard::from(record)) {
                Ok(true) => {
                    stopped = true;
                    break;
                }
                Ok(false) => {}
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }

        bytes += size_of_val(batch) as u64;

        progress(bytes);
    });

    result
}
