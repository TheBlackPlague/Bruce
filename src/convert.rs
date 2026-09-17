use std::{
    cell::Cell,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Cursor, Seek, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant},
};

use crate::{
    data::{self, DataFormat, finite},
    events::{Event, Reporter},
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use bullet_lib::game::formats::{
    bulletformat::{
        self, BulletFormat, ChessBoard, DataLoader,
        chess::{CudADFormat, MarlinFormat},
    },
    montyformat::{FastDeserialise, MontyValueFormat},
    sfbinpack::{self, ChunkReader, TrainingDataEntry},
    viriformat::{
        chess::{
            board::{Board, DrawType, GameOutcome, WinType},
            piece::{Colour, Piece, PieceType},
            types::Square,
        },
        dataformat::{Filter, Game},
    },
};
use clap::{Args, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputFormat {
    Bullet,
    Viri,
}

#[derive(Debug, Args)]
pub struct ConvertOptions {
    #[arg(long, value_enum)]
    pub from: DataFormat,
    #[arg(long, value_enum)]
    pub to: OutputFormat,
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 2)]
    pub threads: usize,
}

pub fn convert(options: &ConvertOptions, reporter: &Reporter) -> Result<()> {
    ensure!(options.threads > 0, "conversion threads must be positive");
    ensure!(
        !options.output.exists(),
        "output already exists: {}",
        options.output.display()
    );

    data::check_file(&options.input, options.from)?;
    let total = fs::metadata(&options.input)?.len();
    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));

    let temporary =
        tempfile::NamedTempFile::new_in(parent).context("create temporary conversion output")?;

    reporter.emit(Event::DataProgress {
        name: "Converting dataset".into(),
        bytes: 0,
        total: Some(total),
        positions: 0,
        skipped: 0,
    });

    let positions = Cell::new(     0u64     );
    let skipped   = Cell::new(     0u64     );
    let last      = Cell::new(Instant::now());

    let progress = |bytes| {
        if last.get().elapsed() >= Duration::from_millis(200) || bytes == total {
            reporter.emit(Event::DataProgress {
                name: "Converting dataset".into(),
                bytes,
                total: Some(total),
                positions: positions.get(),
                skipped: skipped.get(),
            });

            last.set(Instant::now());
        }
    };

    if options.from == DataFormat::Sf {
        reporter.emit(Event::Note {
            message: "SF VALUE_NONE (32002) is omitted; -32768 is omitted only when \
                      the target requires an unrepresentable sign change.".into(),
        });
    }

    match options.to {
        OutputFormat::Bullet => match options.from {
            DataFormat::Bullet => {
                observe_native_output(temporary.path(), Some(total), reporter, || {
                    fs::copy(&options.input, temporary.path())?;
                    Ok(())
                })?;
            }

            DataFormat::Marlin => {
                let output_bytes =
                    total / size_of::<MarlinFormat>() as u64 * size_of::<ChessBoard>() as u64;

                observe_native_output(temporary.path(), Some(output_bytes), reporter, || {
                    bulletformat::convert_from_bin::<MarlinFormat, ChessBoard>(
                        &options.input,
                        temporary.path(),
                        options.threads,
                    )?;
                    Ok(())
                })?;
            }

            DataFormat::Cudad => {
                let output_bytes = (total - CudADFormat::HEADER_SIZE as u64)
                    / size_of::<CudADFormat>() as u64
                    * size_of::< ChessBoard>() as u64;

                observe_native_output(temporary.path(), Some(output_bytes), reporter, || {
                    bulletformat::convert_from_bin::<CudADFormat, ChessBoard>(
                        &options.input,
                        temporary.path(),
                        options.threads,
                    )?;
                    Ok(())
                })?;
            }

            _ => convert_parallel(options, temporary.path(), |bytes, written, omitted| {
                positions.set(written);
                skipped.set(omitted);
                progress(bytes);
            })?,
        },
        OutputFormat::Viri => {
            if matches!(options.from, DataFormat::Text | DataFormat::Bullet | DataFormat::Marlin | DataFormat::Cudad) {
                reporter.emit(Event::Note {
                    message: "Position-only sources use a synthetic legal move in Viri output; \
                              disable tactical filtering. Terminal positions require Bullet \
                              output. \
                              Binary position-only sources retain Bullet-normalized orientation; \
                              unavailable clocks, castling rights and game history cannot be \
                              recovered.".into(),
                });
            } else if options.from == DataFormat::Sf {
                reporter.emit(Event::Note {
                    message: "SF records become single-position Viri games preserving position \
                              metadata, played move, score and result.".into(),
                });
            }

            convert_parallel(options, temporary.path(), |bytes, written, omitted| {
                positions.set(written);
                skipped.set(omitted);
                progress(bytes);
            })?;
        }
    }

    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(&options.output)
        .map_err(|e| anyhow!("publish conversion: {e}"))?;

    if matches!(options.to, OutputFormat::Bullet) {
        positions.set(fs::metadata(&options.output)?.len() / size_of::<ChessBoard>() as u64);
    }

    if !matches!(
        (options.from, options.to),
        (
            DataFormat::Bullet | DataFormat::Marlin | DataFormat::Cudad,
            OutputFormat::Bullet
        )
    ) {
        progress(total);
    }

    reporter.emit(Event::Note {
        message: format!("Saved {}", options.output.display()),
    });

    reporter.emit(Event::Finished);

    Ok(())
}

fn observe_native_output(
    output: &Path,
    total: Option<u64>,
    reporter: &Reporter,
    convert: impl FnOnce() -> Result<()>,
) -> Result<()> {
    reporter.emit(Event::DataProgress {
        name: "Writing converted data (output bytes)".into(),
        bytes: 0,
        total,
        positions: 0,
        skipped: 0,
    });

    std::thread::scope(|scope| {
        let (done, receiver) = mpsc::channel::<()>();

        scope.spawn(move || {
            let mut previous = 0;
            loop {
                let finished = !matches!(
                    receiver.recv_timeout(Duration::from_millis(200)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );

                let completed = fs::metadata(output).map_or(previous, |metadata| metadata.len());

                if completed != previous || finished {
                    reporter.emit(Event::DataProgress {
                        name: "Writing converted data (output bytes)".into(),
                        bytes: completed,
                        total,
                        positions: completed / size_of::<ChessBoard>() as u64,
                        skipped: 0,
                    });

                    previous = completed;
                }

                if finished {
                    break;
                }
            }
        });

        let result = convert();
        drop(done);

        result
    })
}

fn convert_parallel(
    options: &ConvertOptions,
    output: &Path,
    mut progress: impl FnMut(u64, u64, u64),
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(output)?);
    let mut written = 0;
    let mut skipped = 0;
    let mut complete = |bytes, counts: (u64, u64)| {
        written += counts.0;
        skipped += counts.1;
        progress(bytes, written, skipped);
    };

    if matches!(
        options.from,
        DataFormat::Bullet | DataFormat::Marlin | DataFormat::Cudad
    ) {
        macro_rules! fixed {
            ($source:ty) => {{
                let mut bytes = <$source>::HEADER_SIZE as u64;
                let mut result = Ok(());

                DataLoader::<$source>::new(&options.input, 8)?.map_batches(8192, |batch| {
                    if result.is_err() {
                        return;
                    }

                    result = write_parallel(batch, options.threads, &mut writer, |record, output| {
                        position_game(ChessBoard::from(*record))?.serialise_into(output)?;

                        Ok((1, 0))
                    }).map(|counts| {
                        bytes += size_of_val(batch) as u64;
                        complete(bytes, counts);
                    });
                });

                result?;
            }};
        }

        match options.from {
            DataFormat::Bullet => fixed!( ChessBoard ),
            DataFormat::Marlin => fixed!(MarlinFormat),
            DataFormat::Cudad  => fixed!( CudADFormat),

            _ => unreachable!(),
        }
    } else {
        let mut reader = BufReader::new(File::open(&options.input)?);

        loop {
            let mut batch = Vec::new();
            let mut batch_bytes = 0;

            let limit = if options.from == DataFormat::Sf {
                options.threads
            } else {
                8192
            };

            while batch.len() < limit && batch_bytes < 16 * 1024 * 1024 {
                if reader.fill_buf()?.is_empty() {
                    break;
                }

                let mut bytes = Vec::new();

                match options.from {
                    DataFormat::Sf => {
                        if !sfbinpack::read_chunk_into(&mut reader, &mut bytes)? {
                            break;
                        }
                    }

                    DataFormat::Monty => {
                        MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut bytes)?
                    }

                    DataFormat::Viri => {
                        Game::deserialise_fast_into_buffer(&mut reader, &mut bytes)?
                    }

                    DataFormat::Text => {
                        reader.read_until(b'\n', &mut bytes)?;
                    }

                    _ => unreachable!(),
                }

                batch_bytes += bytes.len();
                batch.push(bytes);
            }

            if batch.is_empty() {
                break;
            }

            let counts = write_parallel(&batch, options.threads, &mut writer, |bytes, output| {
                convert_unit(bytes, options.from, options.to, output)
            })?;

            complete(reader.stream_position()?, counts);
        }
    }

    writer.flush()?;

    Ok(())
}

fn write_parallel<T: Sync>(
    batch: &[T],
    threads: usize,
    writer: &mut impl Write,
    convert: impl Fn(&T, &mut Vec<u8>) -> Result<(u64, u64)> + Sync,
) -> Result<(u64, u64)> {
    std::thread::scope(|scope| {
        let convert = &convert;

        let handles = batch.chunks(batch.len().div_ceil(threads)).map(|part| {
            scope.spawn(move || -> Result<_> {
                let mut output = Vec::new();
                let mut counts = (0, 0);

                for item in part {
                    let (written, skipped) = convert(item, &mut output)?;

                    counts.0 += written;
                    counts.1 += skipped;
                }

                Ok((output, counts))
            })
        }).collect::<Vec<_>>();

        let mut counts = (0, 0);

        for handle in handles {
            let (output, count) = handle.join().map_err(
                |_| anyhow!("conversion worker panicked")
            )??;

            writer.write_all(&output)?;

            counts.0 += count.0;
            counts.1 += count.1;
        }

        Ok(counts)
    })
}

fn convert_unit(
    bytes: &[u8],
    from: DataFormat,
    to: OutputFormat,
    output: &mut Vec<u8>,
) -> Result<(u64, u64)> {
    let mut counts = (0, 0);

    match from {
        DataFormat::Sf => {
            let mut reader = ChunkReader::default();

            while reader.has_next(bytes) {
                let entry = reader.next(bytes);

                let black = entry.pos.side_to_move().ordinal() != 0;

                if  entry.score == 32002 ||
                    (matches!(to, OutputFormat::Viri) && black && entry.score == i16::MIN)
                {
                    counts.1 += 1;
                    continue;
                }

                match to {
                    OutputFormat::Bullet => output.extend_from_slice(ChessBoard::as_bytes_slice(
                        &[finite::sf_board(&entry)?],
                    )),

                    OutputFormat::Viri => {
                        let board = sf_viri_board(&entry)?;
                        let mov = board.parse_uci(&entry.mv.as_uci())?;

                        let score = if black { -entry.score } else { entry.score };

                        let result = f32::from(if black {
                            1 - entry.result
                        } else {
                            1 + entry.result
                        }) / 2.0;

                        let mut game = Game::new(&board);

                        game.set_outcome(outcome(result)?);
                        game.add_move(mov, score);
                        game.serialise_into(output)?;
                    }
                }

                counts.0 += 1;
            }
        }

        DataFormat::Monty => {
            let source = MontyValueFormat::deserialise_from(&mut Cursor::new(bytes), Vec::new())?;

            counts.0 = source.moves.len() as u64;

            match to {
                OutputFormat::Bullet => {
                    let mut pos = source.startpos;

                    for entry in source.moves {
                        let record = ChessBoard::from_raw(
                            pos.bbs(),
                            pos.stm(),
                            entry.score,
                            source.result
                        ).map_err(|e| anyhow!(e))?;

                        output.extend_from_slice(ChessBoard::as_bytes_slice(&[record]));

                        pos.make(entry.best_move, &source.castling);
                    }
                }

                OutputFormat::Viri => {
                    let mut board = monty_viri_board(&source);
                    let mut game = Game::new(&board);

                    game.set_outcome(outcome(source.result)?);

                    for entry in source.moves {
                        let mov = board.parse_uci(&entry.best_move.to_uci(&source.castling))?;

                        game.add_move(mov, entry.score);
                        board.make_move_simple(mov);
                    }

                    game.serialise_into(output)?;
                }
            }
        }

        DataFormat::Viri => {
            let game = Game::deserialise_from(&mut Cursor::new(bytes), Vec::new())?;

            match to {
                OutputFormat::Bullet => game.splat_to_bulletformat(
                    |board| {
                        output.extend_from_slice(ChessBoard::as_bytes_slice(&[board]));

                        counts.0 += 1;

                        Ok(())
                    },
                    &Filter::UNRESTRICTED,
                )?,

                OutputFormat::Viri => {
                    game.serialise_into(output)?;
                    counts.0 = game.len() as u64;
                }
            }
        }

        DataFormat::Text => {
            let line = std::str::from_utf8(bytes)?.trim_end_matches(['\r', '\n']);
            match to {
                OutputFormat::Bullet => {
                    let record = line.parse::<ChessBoard>().map_err(|e| anyhow!(e))?;

                    output.extend_from_slice(ChessBoard::as_bytes_slice(&[record]));
                }

                OutputFormat::Viri => {
                    let mut fields = line.split('|').map(str::trim);
                    let mut board = board_from_fen(fields.next().context("missing FEN")?)?;
                    let score = fields.next().context("missing score")?.parse::<i16>()?;
                    let result = fields.next().context("missing result")?;

                    let result = match result {
                        "1.0" | "[1.0]" | "1"   => 1.0,
                        "0.5" | "[0.5]" | "1/2" => 0.5,
                        "0.0" | "[0.0]" | "0"   => 0.0,
                        _ => bail!("invalid text result {result}"),
                    };

                    let mov = board
                        .legal_moves()
                        .into_iter()
                        .next()
                        .context("terminal position requires Bullet output")?;

                    let mut game = Game::new(&board);

                    game.set_outcome(outcome(result)?);
                    game.add_move(mov, score);

                    game.serialise_into(output)?;
                }
            }

            counts.0 = 1;
        }

        _ => unreachable!(),
    }

    Ok(counts)
}

fn board_from_bitboards(bbs: [u64; 8]) -> Board {
    let mut board = Board::new();

    for (kind, &pieces) in bbs[2..].iter().enumerate() {
        let mut pieces = pieces;

        while pieces != 0 {
            let square = pieces.trailing_zeros() as u8;

            pieces &= pieces - 1;

            board.add_piece(
                Square::new(square).unwrap(),
                Piece::new(
                    Colour::new(bbs[1] & (1 << square) != 0),
                    PieceType::new(kind as u8).unwrap(),
                ),
            );
        }
    }

    board
}

fn finish_board(board: &mut Board) {
    board.regenerate_zobrist();
    board.regenerate_threats();
}

fn sf_viri_board(entry: &TrainingDataEntry) -> Result<Board> {
    use sfbinpack::chess::castling_rights::CastlingRights as SfRights;

    let pos = &entry.pos;
    let mut board = board_from_bitboards(finite::sf_bitboards(entry));

    *board.turn_mut() = Colour::new(pos.side_to_move().ordinal() != 0);
    *board.ep_sq_mut() = Square::new(pos.ep_square().index() as u8);
    *board.halfmove_clock_mut() = pos.rule50_counter().try_into()?;

    board.set_fullmove_clock(pos.ply() / 2 + 1);

    for (colour, king, queen, offset) in [
        (
            Colour::White,
            SfRights::WHITE_KING_SIDE,
            SfRights::WHITE_QUEEN_SIDE,
            0,
        ),
        (
            Colour::Black,
            SfRights::BLACK_KING_SIDE,
            SfRights::BLACK_QUEEN_SIDE,
            56,
        ),
    ] {
        if pos.castling_rights().contains(king ) {
            *board.castling_rights_mut().kingside_mut (colour) = Square::new(offset + 7);
        }

        if pos.castling_rights().contains(queen) {
            *board.castling_rights_mut().queenside_mut(colour) = Square::new(offset    );
        }
    }

    finish_board(&mut board);

    Ok(board)
}

fn monty_viri_board(source: &MontyValueFormat) -> Board {
    let pos = &source.startpos;
    let mut board = board_from_bitboards(pos.bbs());

    *board.turn_mut() = Colour::new(pos.stm() != 0);
    *board.ep_sq_mut() = if pos.enp_sq() == 0 {
        None
    } else {
        Square::new(pos.enp_sq())
    };
    *board.halfmove_clock_mut() = pos.halfm();

    board.set_fullmove_clock(pos.fullm());

    for (side, colour) in [(0, Colour::White), (1, Colour::Black)] {
        let rooks = source.castling.rook_files()[side];

        if pos.rights() & (8 >> (2 * side)) != 0 {
            *board.castling_rights_mut().queenside_mut(colour) = Square::new(
                56 * side as u8 + rooks[0]
            );
        }

        if pos.rights() & (4 >> (2 * side)) != 0 {
            *board.castling_rights_mut().kingside_mut(colour) = Square::new(
                56 * side as u8 + rooks[1]
            );
        }
    }

    finish_board(&mut board);

    board
}

fn board_from_fen(fen: &str) -> Result<Board> {
    let mut board = Board::new();
    board.set_from_fen(fen)?;

    Ok(board)
}

fn outcome(result: f32) -> Result<GameOutcome> {
    match result {
        0.0 => Ok(GameOutcome::BlackWin( WinType::Adjudication)),
        0.5 => Ok(GameOutcome::    Draw(DrawType::Adjudication)),
        1.0 => Ok(GameOutcome::WhiteWin( WinType::Adjudication)),

        _ => bail!("invalid game outcome {result}"),
    }
}

fn position_game(record: ChessBoard) -> Result<Game> {
    let mut board = Board::new();

    for (piece, square) in record {
        board.add_piece(
            Square::new(square).context("invalid square")?,
            Piece::new(
                Colour::new(piece & 8 != 0),
                PieceType::new(piece & 7).context("invalid piece")?,
            ),
        );
    }

    finish_board(&mut board);

    let mov = board.legal_moves().into_iter().next().context(
        "terminal position has no legal move: use Bullet output, since Viri stores scores on moves",
    )?;

    let mut game = Game::new(&board);

    game.set_outcome(outcome(record.result())?);
    game.add_move(mov, record.score());

    Ok(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_lib::game::formats::viriformat::dataformat::Filter;

    #[test]
    fn direct_boards_preserve_position_metadata() {
        use bullet_lib::game::formats::{montyformat::chess::Castling, sfbinpack::chess};

        for fen in [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 17 42",
            "r3k2r/8/8/8/8/8/8/R3K2R b Qk - 17 42",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 42",
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 42",
        ] {
            let expected = board_from_fen(fen).unwrap();
            let entry = TrainingDataEntry {
                pos: chess::position::Position::from_fen(fen).unwrap(),
                mv: chess::r#move::Move::normal(
                    chess::coords::Square::new(4),
                    chess::coords::Square::new(5),
                ),
                score: -250,
                ply: 82,
                result: -1,
            };

            assert_eq!(sf_viri_board(&entry).unwrap(), expected, "SF: {fen}");

            let mut castling = Castling::default();
            let source = MontyValueFormat {
                startpos: bullet_lib::game::formats::montyformat::chess::Position::parse_fen(
                    fen,
                    &mut castling,
                ),
                castling,
                result: 0.0,
                moves: Vec::new(),
            };
            
            assert_eq!(monty_viri_board(&source), expected, "Monty: {fen}");
        }
    }

    #[test]
    fn viri_roundtrip_retains_black_stm_features_score_and_result() {
        let record: ChessBoard =
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1 | 125 | 1.0"
                .parse()
                .unwrap();

        let game = position_game(record).unwrap();

        let mut recovered = Vec::new();

        game.splat_to_bulletformat(
            |board| {
                recovered.push(board);
                Ok(())
            },
            &Filter::UNRESTRICTED,
        )
        .unwrap();

        assert_eq!(recovered, vec![record]);
    }

    #[test]
    fn terminal_position_conversion_is_explicit_error() {
        let record: ChessBoard = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1 | 1000 | 1.0"
            .parse()
            .unwrap();

        assert!(
            position_game(record)
                .unwrap_err()
                .to_string()
                .contains("terminal")
        );
    }

    fn collected(path: &Path, format: DataFormat) -> Vec<ChessBoard> {
        let mut records = Vec::new();
        finite::visit(
            path,
            format,
            None,
            |board| {
                records.push(board);
                Ok(false)
            },
            |_| {},
        )
        .unwrap();

        records
    }

    #[test]
    fn monty_conversion_preserves_both_score_perspectives_and_games() {
        use bullet_lib::game::formats::montyformat::{
            SearchResult,
            chess::{Castling, Position},
        };

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("monty");
        let mut writer = File::create(&input).unwrap();

        for side in ["w", "b"] {
            let mut castling = Castling::default();

            let position = Position::parse_fen(
                &format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {side} KQkq - 0 8"),
                &mut castling,
            );

            let mut moves = Vec::new();
            position.map_legal_moves(&castling, |mov| moves.push(mov));

            MontyValueFormat {
                startpos: position,
                castling,
                result: 1.0,
                moves: vec![SearchResult {
                    best_move: moves[0],
                    score: 125,
                }],
            }
            .serialise_into(&mut writer)
            .unwrap();
        }

        drop(writer);

        let expected = collected(&input, DataFormat::Monty);

        assert_eq!(
            expected.iter().map(|b| b.score).collect::<Vec<_>>(),
            vec![125, -125]
        );

        for (target, format) in [
            (OutputFormat::Bullet, DataFormat::Bullet),
            (OutputFormat::Viri  , DataFormat::Viri  ),
        ] {
            let output = directory.path().join(format!("{target:?}"));

            convert(
                &ConvertOptions {
                    from: DataFormat::Monty,
                    to: target,

                     input:  input.clone(),
                    output: output.clone(),

                    threads: 3,
                },
                &Reporter::silent(),
            )
            .unwrap();

            assert_eq!(collected(&output, format), expected);
        }
    }

    #[test]
    fn stockfish_conversion_preserves_stm_score_result_and_excludes_none() {
        use bullet_lib::game::formats::sfbinpack::{
            CompressedTrainingDataEntryWriter, TrainingDataEntry,
            chess::{coords::Square, r#move::Move, position::Position},
        };

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("sf");
        let mut writer = CompressedTrainingDataEntryWriter::new(
            File::create(&input).unwrap()
        ).unwrap();

        for side in ["w", "b"] {
            let position = Position::from_fen(&format!(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {side} KQkq - 0 8"
            ))
            .unwrap();

            let mov = if side == "w" {
                Move::normal(Square::new(12), Square::new(28))
            } else {
                Move::normal(Square::new(52), Square::new(36))
            };

            writer.write_entry(&TrainingDataEntry {
                pos: position,
                mv: mov,
                score: 125,
                ply: 14,
                result: 1,
            }).unwrap();

            writer.write_entry(&TrainingDataEntry {
                pos: position,
                mv: mov,
                score: 32002,
                ply: 14,
                result: 1,
            }).unwrap();
        }

        writer.flush_and_end();
        drop(writer);

        let expected = collected(&input, DataFormat::Sf);

        assert_eq!(expected.len(), 2);
        assert!(
            expected
                .iter()
                .all(|board| board.score == 125 && board.result == 2)
        );

        for (target, format) in [
            (OutputFormat::Bullet, DataFormat::Bullet),
            (OutputFormat::Viri  , DataFormat::Viri  ),
        ] {
            let output = directory.path().join(format!("{target:?}"));

            convert(
                &ConvertOptions {
                    from: DataFormat::Sf,
                    to: target,

                     input:  input.clone(),
                    output: output.clone(),

                    threads: 3,
                },
                &Reporter::silent(),
            )
            .unwrap();

            assert_eq!(collected(&output, format), expected);
        }
    }

    #[test]
    fn fixed_and_text_formats_convert_to_both_targets() {
        let directory = tempfile::tempdir().unwrap();
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 8";
        let text = format!("{fen} | 125 | 1.0");
        let record: ChessBoard = text.parse().unwrap();

        let board = board_from_fen(fen).unwrap();
        let bullet = directory.path().join("bullet");
        fs::write(&bullet, ChessBoard::as_bytes_slice(&[record])).unwrap();

        let marlin = directory.path().join("marlin");
        fs::write(&marlin, board.to_marlinformat(125, 2, 0).as_bytes()).unwrap();

        let text_path = directory.path().join("text");
        fs::write(&text_path, &text).unwrap();

        let cudad = directory.path().join("cudad");
        let mut bytes = vec![0; CudADFormat::HEADER_SIZE];
        bytes.extend_from_slice(&record.pcs);
        bytes.extend_from_slice(&record.occ.to_le_bytes());
        bytes.extend_from_slice(&[8, 0, 0, 0]);
        bytes.extend_from_slice(&125i16.to_le_bytes());
        bytes.extend_from_slice(&[1, 0]);
        fs::write(&cudad, bytes).unwrap();

        for (input, source) in [
            (bullet   , DataFormat::Bullet),
            (marlin   , DataFormat::Marlin),
            (text_path, DataFormat::Text  ),
            (cudad    , DataFormat::Cudad ),
        ] {
            for (target, format) in [
                (OutputFormat::Bullet, DataFormat::Bullet),
                (OutputFormat::Viri  , DataFormat::Viri  ),
            ] {
                let output = directory.path().join(format!("{source:?}-{target:?}"));

                convert(
                    &ConvertOptions {
                        from: source,
                        to: target,
                        input: input.clone(),
                        output: output.clone(),
                        threads: 3,
                    },
                    &Reporter::silent(),
                )
                .unwrap();

                assert_eq!(
                    collected(&output, format),
                    vec![record],
                    "{source:?} to {target:?}"
                );
            }
        }
    }
}
