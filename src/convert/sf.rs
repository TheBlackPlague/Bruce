use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    sync::atomic::AtomicBool,
};

use anyhow::{Context, Result, anyhow, ensure};
use bullet_lib::game::formats::{
    bulletformat::{BulletFormat, ChessBoard},
    montyformat::{
        MontyValueFormat, SearchResult,
        chess::{Castling, Flag, Move as MontyMove, Position},
    },
    sfbinpack::{
        self, TrainingDataEntry,
        chess::{
            castling_rights::{CastleType, CastlingTraits},
            color::Color,
            r#move::MoveType,
            piece::Piece,
            piecetype::PieceType,
        },
    },
    viriformat::{
        chess::{
            board::{Board, movegen::MoveList},
            chessmove::{Move as ViriMove, MoveFlags},
            piece::PieceType as ViriPiece,
            types::Square,
        },
        dataformat::Game,
    },
};

use super::{Options, Summary, check_stop, encode::outcome, progress};
use crate::{
    cli::Reporter,
    data::{self, Format},
};

pub(super) fn convert(
    options: &Options,
    threads: usize,
    writer: &mut impl Write,
    reporter: &Reporter,
    stop: &AtomicBool,
    summary: &mut Summary,
) -> Result<()>
{
    let mut reader = BufReader::new(File::open(&options.input)?);
    let mut finished = false;

    while !finished {
        check_stop(stop)?;

        let mut chunks = Vec::with_capacity(threads);
        let mut input_bytes = 0;
        while chunks.len() < threads && input_bytes < 64 * 1024 * 1024 {
            check_stop(stop)?;

            let mut chunk = Vec::new();
            if !sfbinpack::read_chunk_into(&mut reader, &mut chunk)? {
                finished = true;
                break;
            }

            input_bytes += chunk.len();
            chunks.push(chunk);
        }

        if threads == 1 {
            for chunk in chunks {
                let converted = convert_chunk(chunk, options, stop, writer)?;

                check_stop(stop)?;

                summary.positions += converted.positions;
                summary.skipped += converted.skipped;
            }
        } else {
            std::thread::scope(|scope| -> Result<()> {
                let workers = chunks
                    .into_iter()
                    .map(|chunk| {
                        scope.spawn(move || -> Result<_> {
                            let mut output = tempfile::spooled_tempfile(1024 * 1024);
                            let mut writer = BufWriter::with_capacity(64 * 1024, &mut output);
                            let converted = convert_chunk(chunk, options, stop, &mut writer)?;
                            writer.flush()?;
                            drop(writer);

                            Ok((output, converted))
                        })
                    })
                    .collect::<Vec<_>>();

                let mut buffer = [0; 64 * 1024];
                for worker in workers {
                    let (mut output, converted) =
                        worker.join().map_err(|_| anyhow!("Could not convert an SF binpack chunk"))??;
                    output.rewind()?;

                    loop {
                        check_stop(stop)?;

                        let count = output.read(&mut buffer)?;
                        if count == 0 {
                            break;
                        }

                        writer.write_all(&buffer[..count])?;
                    }

                    summary.positions += converted.positions;
                    summary.skipped += converted.skipped;
                }

                Ok(())
            })?;
        }

        progress(reporter, summary.positions, "positions converted");
    }

    check_stop(stop)?;
    ensure!(summary.positions > 0, "No positions could be converted");

    Ok(())
}

struct ChunkSummary
{
    positions: u64,
    skipped: u64,
}

fn convert_chunk(
    mut chunk: Vec<u8>,
    options: &Options,
    stop: &AtomicBool,
    writer: &mut impl Write,
) -> Result<ChunkSummary>
{
    let length = u32::try_from(chunk.len()).context("The SF chunk is too large")?;
    chunk.splice(..0, b"BINP".iter().copied().chain(length.to_le_bytes()));

    let mut reader = sfbinpack::CompressedTrainingDataEntryReader::from_bytes(chunk)?;
    let mut summary = ChunkSummary { positions: 0, skipped: 0 };
    let mut sf_writer = if options.to == Format::Sf {
        Some(sfbinpack::CompressedTrainingDataEntryWriter::new_in_memory()?)
    } else {
        None
    };
    let mut game = None;

    while reader.has_next() {
        check_stop(stop)?;

        let continuation = reader.is_next_entry_continuation();
        let mut entry = reader.next();
        if !continuation {
            finish_game(&mut game, writer)?;

            if let Some(encoder) = sf_writer.as_mut()
                && encoder.written_bytes() > 0
            {
                writer.write_all(&encoder.into_bytes()?)?;
                *encoder = sfbinpack::CompressedTrainingDataEntryWriter::new_in_memory()?;
            }
        }

        if entry.score == 32002 {
            finish_game(&mut game, writer)?;

            summary.skipped += 1;
            continue;
        }

        entry.score = options.score_encoding.convert(entry.score)?;
        validate_position(&entry)?;

        let black = entry.pos.side_to_move() == Color::Black;
        let white_score = if black { -i32::from(entry.score) } else { i32::from(entry.score) };
        let white_result = (if black { 1 - entry.result } else { 1 + entry.result }) as u8;

        match options.to {
            Format::Bullet => {
                let board = data::sf_board(&entry)?;
                data::validate_board(&board)?;
                writer.write_all(ChessBoard::as_bytes_slice(&[board]))?;
            },

            Format::Text => {
                let separator = options.separator.as_deref().context("Text conversion needs --separator")?;
                writeln!(
                    writer,
                    "{}{separator}{white_score}{separator}{}",
                    entry.pos.fen().map_err(|_| anyhow!("Invalid SF position"))?,
                    f32::from(white_result) / 2.0
                )?;
            },

            Format::Sf => sf_writer.as_mut().unwrap().write_entry(&entry)?,

            Format::Monty | Format::Viri => {
                let Ok(score) = i16::try_from(white_score) else {
                    finish_game(&mut game, writer)?;
                    summary.skipped += 1;
                    continue;
                };

                ensure!(entry.pos.rule50_counter() < 255, "The move counter cannot be represented in the output");
                if game.is_none() {
                    game = Some(GameState::new(&entry, white_result, options.to)?);
                }

                game.as_mut().unwrap().push(&entry, score)?;
            },
        }

        summary.positions += 1;
    }

    check_stop(stop)?;
    finish_game(&mut game, writer)?;
    if let Some(mut encoder) = sf_writer {
        writer.write_all(&encoder.into_bytes()?)?;
    }

    Ok(summary)
}

fn validate_position(entry: &TrainingDataEntry) -> Result<()>
{
    ensure!((-1..=1).contains(&entry.result), "Invalid result in the dataset");
    ensure!(entry.pos.occupied().bits().count_ones() <= 32, "Invalid position in the dataset");

    for side in [Color::White, Color::Black] {
        ensure!(
            entry.pos.pieces_bb_color(side, PieceType::King).bits().count_ones() == 1,
            "A position must contain both kings"
        );
    }

    Ok(())
}

enum GameState
{
    Monty
    {
        game: MontyValueFormat, position: Position
    },
    Viri
    {
        game: Game, position: Board
    },
}

impl GameState
{
    fn new(entry: &TrainingDataEntry, result: u8, format: Format) -> Result<Self>
    {
        let fen = entry.pos.fen().map_err(|_| anyhow!("Invalid SF position"))?;
        let fields = fen.split_whitespace().collect::<Vec<_>>();
        ensure!(fields.len() == 6, "Invalid SF position");
        let halfmove = fields[4].parse::<u8>().context("The move counter cannot be represented in the output")?;
        let fullmove = fields[5].parse::<u16>().context("The move counter cannot be represented in the output")?;
        ensure!(fullmove > 0 && halfmove < 255, "The move counter cannot be represented in the output");

        match format {
            Format::Monty => {
                let mut castling = Castling::default();
                let startpos = Position::parse_fen(&fen, &mut castling);
                ensure!(
                    startpos.halfm() == halfmove && startpos.fullm() == fullmove,
                    "The move counter cannot be represented in Monty format"
                );

                Ok(Self::Monty {
                    game: MontyValueFormat { startpos, castling, result: f32::from(result) / 2.0, moves: Vec::new() },
                    position: startpos,
                })
            },

            Format::Viri => {
                let position = data::parse_board(&fen)?;
                let mut game = Game::new(&position);
                game.set_outcome(outcome(result));

                Ok(Self::Viri { game, position })
            },

            _ => unreachable!(),
        }
    }

    fn push(&mut self, entry: &TrainingDataEntry, score: i16) -> Result<()>
    {
        match self {
            Self::Monty { game, position } => {
                let mov = monty_move(entry)?;
                let mut legal = false;
                position.map_legal_moves(&game.castling, |candidate| legal |= candidate == mov);
                ensure!(legal, "The dataset contains an illegal move");

                game.moves.push(SearchResult { best_move: mov, score });
                position.make(mov, &game.castling);
            },

            Self::Viri { game, position } => {
                let mov = viri_move(entry)?;
                let mut moves = MoveList::new();
                position.generate_moves(&mut moves);
                ensure!(moves.iter_moves().any(|&candidate| candidate == mov), "The dataset contains an illegal move");
                ensure!(position.make_move_simple(mov), "The dataset contains an illegal move");

                game.add_move(mov, score);
            },
        }

        Ok(())
    }
}

fn finish_game(game: &mut Option<GameState>, writer: &mut impl Write) -> Result<()>
{
    match game.take() {
        Some(GameState::Monty { game, .. }) => game.serialise_into(writer)?,
        Some(GameState::Viri { game, .. }) => game.serialise_into(writer)?,
        None => {},
    }

    Ok(())
}

fn move_squares(entry: &TrainingDataEntry) -> Result<(u16, u16)>
{
    let from = entry.mv.from().index();
    let to = entry.mv.to().index();
    ensure!(from < 64 && to < 64 && from != to, "Invalid move in the dataset");

    if entry.mv.mtype() == MoveType::Castle {
        let side = entry.pos.side_to_move();
        let rank = if side == Color::White { 0 } else { 56 };
        ensure!(from == rank + 4 && (to == rank || to == rank + 7), "Invalid castling move in the dataset");

        let kind = if to > from { CastleType::Short } else { CastleType::Long };
        ensure!(
            entry.pos.piece_at(entry.mv.from()) == Piece::new(PieceType::King, side)
                && entry.pos.piece_at(entry.mv.to()) == Piece::new(PieceType::Rook, side)
                && entry.pos.castling_rights().contains(CastlingTraits::castling_rights(side, kind)),
            "Invalid castling move in the dataset"
        );
    }

    Ok((from as u16, to as u16))
}

fn promotion(entry: &TrainingDataEntry) -> Result<u16>
{
    let piece = entry.mv.promoted_piece();
    ensure!(piece.color() == entry.pos.side_to_move(), "Invalid promotion in the dataset");

    match piece.piece_type() {
        PieceType::Knight => Ok(0),
        PieceType::Bishop => Ok(1),
        PieceType::Rook => Ok(2),
        PieceType::Queen => Ok(3),
        _ => Err(anyhow!("Invalid promotion in the dataset")),
    }
}

fn monty_move(entry: &TrainingDataEntry) -> Result<MontyMove>
{
    let (from, mut to) = move_squares(entry)?;
    let capture = entry.pos.piece_at(entry.mv.to()) != Piece::none();
    let flag = match entry.mv.mtype() {
        MoveType::Normal => {
            if capture {
                Flag::CAP
            } else if entry.pos.piece_at(entry.mv.from()).piece_type() == PieceType::Pawn && from.abs_diff(to) == 16 {
                Flag::DBL
            } else {
                Flag::QUIET
            }
        },
        MoveType::EnPassant => Flag::ENP,
        MoveType::Promotion => Flag::NPR + promotion(entry)? + if capture { Flag::CAP } else { Flag::QUIET },
        MoveType::Castle => {
            let kingside = to > from;
            to = (from / 8) * 8 + if kingside { 6 } else { 2 };
            if kingside { Flag::KS } else { Flag::QS }
        },
    };

    Ok(MontyMove::new(from, to, flag))
}

fn viri_move(entry: &TrainingDataEntry) -> Result<ViriMove>
{
    let (from, to) = move_squares(entry)?;
    let from = Square::new(from as u8).context("Invalid move square")?;
    let to = Square::new(to as u8).context("Invalid move square")?;

    Ok(match entry.mv.mtype() {
        MoveType::Normal => ViriMove::new(from, to),
        MoveType::EnPassant => ViriMove::new_with_flags(from, to, MoveFlags::EnPassant),
        MoveType::Castle => ViriMove::new_with_flags(from, to, MoveFlags::Castle),
        MoveType::Promotion => {
            let piece = [ViriPiece::Knight, ViriPiece::Bishop, ViriPiece::Rook, ViriPiece::Queen]
                [usize::from(promotion(entry)?)];
            ViriMove::new_with_promo(from, to, piece)
        },
    })
}
