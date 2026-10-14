use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, anyhow, ensure};
use bullet_lib::game::formats::{
    bulletformat::ChessBoard, montyformat::MontyValueFormat, sfbinpack, viriformat::dataformat::Game,
};

use super::{
    Format, Record, accepts, monty_fen, monty_uci, parse_board, sf_board, sf_tactical, validate_board, validate_source,
    viri_fen, viri_result,
};
use crate::config::{DataSource, Filter};

pub fn visit(source: &DataSource, filter: Option<&Filter>, callback: impl FnMut(Record) -> Result<bool>) -> Result<()>
{
    scan(source, true, filter, None, callback).map(|_| ())
}

pub fn visit_boards(
    source: &DataSource,
    filter: Option<&Filter>,
    mut callback: impl FnMut(ChessBoard) -> Result<bool>,
) -> Result<()>
{
    scan(source, false, filter, None, |record| callback(record.board)).map(|_| ())
}

pub fn visit_counted(source: &DataSource, callback: impl FnMut(Record) -> Result<bool>) -> Result<(u64, u64)>
{
    scan(source, true, None, None, callback)
}

pub fn visit_boards_stoppable(
    source: &DataSource,
    filter: Option<&Filter>,
    stop: &AtomicBool,
    mut callback: impl FnMut(ChessBoard) -> Result<bool>,
) -> Result<()>
{
    scan(source, false, filter, Some(stop), |record| callback(record.board)).map(|_| ())
}

pub fn visit_counted_stoppable(
    source: &DataSource,
    stop: &AtomicBool,
    callback: impl FnMut(Record) -> Result<bool>,
) -> Result<(u64, u64)>
{
    scan(source, true, None, Some(stop), callback)
}

pub(super) fn scan(
    source: &DataSource,
    details: bool,
    filter: Option<&Filter>,
    stop: Option<&AtomicBool>,
    mut callback: impl FnMut(Record) -> Result<bool>,
) -> Result<(u64, u64)>
{
    validate_source(source, filter)?;

    let mut positions = 0;
    let mut skipped = 0;

    catch_unwind(AssertUnwindSafe(|| {
        decode(source, details, stop, &mut skipped, &mut |record| {
            validate_board(&record.board)?;

            if !accepts(&record, filter) {
                return Ok(false);
            }

            positions += 1;
            callback(record)
        })
    }))
    .map_err(|_| anyhow!("Could not decode the dataset"))??;

    Ok((positions, skipped))
}

fn check_stop(stop: Option<&AtomicBool>) -> Result<()>
{
    if stop.is_some_and(|stop| stop.load(Ordering::Relaxed)) {
        return Err(crate::cli::Cancelled.into());
    }

    Ok(())
}

pub(super) fn bullet_board(bytes: &[u8; size_of::<ChessBoard>()]) -> ChessBoard
{
    ChessBoard {
        occ: u64::from_ne_bytes(bytes[..8].try_into().unwrap()),
        pcs: bytes[8..24].try_into().unwrap(),
        score: i16::from_ne_bytes(bytes[24..26].try_into().unwrap()),
        result: bytes[26],
        ksq: bytes[27],
        opp_ksq: bytes[28],
        extra: bytes[29..32].try_into().unwrap(),
    }
}

fn decode(
    source: &DataSource,
    details: bool,
    stop: Option<&AtomicBool>,
    skipped: &mut u64,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    let reader = BufReader::new(File::open(&source.path).context("Could not open the dataset")?);
    match source.format {
        Format::Bullet => read_bullet(reader, stop, callback),
        Format::Text => read_text(
            reader,
            source.separator.as_deref().context("Text data needs a separator")?,
            details,
            stop,
            callback,
        ),
        Format::Monty => read_monty(reader, details, stop, callback),
        Format::Viri => read_viri(reader, details, stop, callback),
        Format::Sf => read_sf(reader, details, stop, skipped, callback),
    }
}

fn read_bullet(
    mut reader: BufReader<File>,
    stop: Option<&AtomicBool>,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    let mut bytes = vec![0; 4096 * size_of::<ChessBoard>()];

    let mut remaining = reader.get_ref().metadata()?.len();
    while remaining > 0 {
        check_stop(stop)?;

        let length = bytes.len().min(usize::try_from(remaining).unwrap_or(usize::MAX));
        reader.read_exact(&mut bytes[..length]).context("The dataset is truncated")?;
        remaining -= length as u64;

        for bytes in bytes[..length].as_chunks::<{ size_of::<ChessBoard>() }>().0 {
            check_stop(stop)?;
            if callback(Record {
                board: bullet_board(bytes),
                fen: None,
                next_move: None,
                ply: None,
                in_check: None,
                tactical: None,
            })? {
                return Ok(());
            }
        }
    }

    Ok(())
}

fn read_text(
    reader: BufReader<File>,
    separator: &str,
    details: bool,
    stop: Option<&AtomicBool>,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    for line in reader.lines() {
        check_stop(stop)?;

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields = line.split(separator).map(str::trim).collect::<Vec<_>>();
        ensure!(fields.len() == 3, "Expected FEN, score and result in each text row");

        let position = parse_board(fields[0]).context("Invalid FEN in the dataset")?;
        let score = fields[1].parse::<i32>().context("Invalid score in the dataset")?;
        let score = if fields[0].split_whitespace().nth(1) == Some("b") { score.checked_neg() } else { Some(score) };
        let score = score.and_then(|score| i16::try_from(score).ok()).context("The score cannot be represented")?;
        let result = match fields[2].trim_matches(['[', ']']) {
            "0" | "0.0" => 0,
            "0.5" | "1/2" => 1,
            "1" | "1.0" => 2,
            _ => return Err(anyhow!("Invalid result in the dataset")),
        };

        let mut board = position.to_bulletformat(result, 0)?;
        board.score = score;

        if callback(Record {
            board,
            fen: details.then(|| fields[0].to_owned()),
            next_move: None,
            ply: Some(position.ply().try_into().context("Position ply is too large")?),
            in_check: Some(position.in_check()),
            tactical: None,
        })? {
            return Ok(());
        }
    }

    Ok(())
}

fn read_monty(
    mut reader: BufReader<File>,
    details: bool,
    stop: Option<&AtomicBool>,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    while !reader.fill_buf()?.is_empty() {
        check_stop(stop)?;

        let game =
            MontyValueFormat::deserialise_from(&mut reader, Vec::new()).context("Could not read a Monty game")?;

        let mut position = game.startpos;
        for entry in game.moves {
            check_stop(stop)?;

            let board = ChessBoard::from_raw(position.bbs(), position.stm(), entry.score, game.result)
                .map_err(|_| anyhow!("Invalid Monty position"))?;

            let record = Record {
                board,
                fen: details.then(|| monty_fen(&position, &game.castling)),
                next_move: details.then(|| monty_uci(entry.best_move, &game.castling)),
                ply: Some(position.fullm().saturating_sub(1).saturating_mul(2) + position.stm() as u16),
                in_check: Some(position.in_check()),
                tactical: Some(entry.best_move.is_capture() || entry.best_move.is_promo()),
            };

            if callback(record)? {
                return Ok(());
            }

            position.make(entry.best_move, &game.castling);
        }
    }

    Ok(())
}

fn read_viri(
    mut reader: BufReader<File>,
    details: bool,
    stop: Option<&AtomicBool>,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    while !reader.fill_buf()?.is_empty() {
        check_stop(stop)?;

        let game = Game::deserialise_from(&mut reader, Vec::new()).context("Could not read a Viri game")?;
        let result = viri_result(game.outcome());

        let mut position = game.initial_position();
        for (mov, score) in game.moves {
            check_stop(stop)?;

            let record = Record {
                board: position.to_bulletformat(result, score.get())?,
                fen: details.then(|| viri_fen(&position)),
                next_move: details.then(|| mov.display(true).to_string()),
                ply: Some(position.ply().try_into().context("Position ply is too large")?),
                in_check: Some(position.in_check()),
                tactical: Some(position.is_tactical(mov)),
            };

            if callback(record)? {
                return Ok(());
            }

            ensure!(position.make_move_simple(mov), "Invalid move in a Viri game");
        }
    }

    Ok(())
}

fn read_sf(
    mut reader: BufReader<File>,
    details: bool,
    stop: Option<&AtomicBool>,
    skipped: &mut u64,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    let mut bytes = Vec::new();

    while sfbinpack::read_chunk_into(&mut reader, &mut bytes)? {
        check_stop(stop)?;

        let mut chunk = sfbinpack::ChunkReader::default();
        while chunk.has_next(&bytes) {
            check_stop(stop)?;

            let entry = chunk.next(&bytes);
            if entry.score == 32002 {
                *skipped += 1;
                continue;
            }

            let fen = if details { Some(entry.pos.fen().map_err(|_| anyhow!("Invalid SF position"))?) } else { None };

            let next_move = details.then(|| {
                if entry.mv.mtype() == sfbinpack::chess::r#move::MoveType::Castle {
                    format!("{}{}", entry.mv.from(), entry.mv.to())
                } else {
                    entry.mv.as_uci()
                }
            });

            if callback(Record {
                board: sf_board(&entry)?,
                fen,
                next_move,
                ply: Some(entry.ply),
                in_check: Some(entry.pos.is_checked(entry.pos.side_to_move())),
                tactical: Some(sf_tactical(&entry)),
            })? {
                return Ok(());
            }
        }
    }

    Ok(())
}
