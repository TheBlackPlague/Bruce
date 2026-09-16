use std::{
    cell::Cell,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Seek, Write},
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
        self, BulletFormat, ChessBoard,
        chess::{CudADFormat, MarlinFormat},
    },
    montyformat::MontyValueFormat,
    sfbinpack::CompressedTrainingDataEntryReader,
    viriformat::{
        chess::board::{Board, DrawType, GameOutcome, WinType},
        dataformat::Game,
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

    reporter.emit(Event::Phase {
        name: "Converting dataset".into(),
        completed: 0,
        total: Some(total),
    });

    let progress = phase_progress(reporter, "Converting dataset", Some(total));

    if options.from == DataFormat::Sf {
        reporter.emit(Event::Note {
            message: "SF VALUE_NONE (32002) and unrepresentable -32768 scores are omitted because \
                      they are not usable evaluation targets.".into()
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

            DataFormat::Text => {
                reporter.emit(Event::Phase {
                    name: "Validating text".into(),
                    completed: 0,
                    total: Some(total),
                });

                let mut positions = 0u64;
                finite::visit(
                    &options.input,
                    options.from,
                    None,
                    |board| {
                        finite::validate_board(&board)?;
                        positions += 1;
                        Ok(false)
                    },
                    phase_progress(reporter, "Validating text", Some(total)),
                )?;

                observe_native_output(
                    temporary.path(),
                    Some(positions * size_of::<ChessBoard>() as u64),
                    reporter,
                    || {
                        bulletformat::convert_from_text::<ChessBoard>(
                            &options.input,
                            temporary.path(),
                        )?;
                        Ok(())
                    },
                )?;
            }

            _ => {
                let mut writer = BufWriter::new(File::create(temporary.path())?);
                let mut buffer = Vec::with_capacity(8192);
                finite::visit(
                    &options.input,
                    options.from,
                    None,
                    |board| {
                        finite::validate_board(&board)?;
                        buffer.push(board);
                        if buffer.len() == 8192 {
                            ChessBoard::write_to_bin(&mut writer, &buffer)?;
                            buffer.clear();
                        }
                        Ok(false)
                    },
                    &progress,
                )?;
                ChessBoard::write_to_bin(&mut writer, &buffer)?;
                writer.flush()?;
            }
        },

        OutputFormat::Viri => {
            let mut writer = BufWriter::new(File::create(temporary.path())?);

            match options.from {
                DataFormat::Viri => {
                    let mut reader = BufReader::new(File::open(&options.input)?);

                    while !reader.fill_buf()?.is_empty() {
                        Game::deserialise_from(&mut reader, Vec::new())?
                            .serialise_into(&mut writer)?;

                        progress(reader.stream_position()?);
                    }
                }

                DataFormat::Monty => {
                    let mut reader = BufReader::new(File::open(&options.input)?);

                    while !reader.fill_buf()?.is_empty() {
                        let source = MontyValueFormat::deserialise_from(&mut reader, Vec::new())?;
                        let mut board = board_from_fen(&source.startpos.as_fen())?;
                        let mut game = Game::new(&board);

                        game.set_outcome(outcome(source.result)?);

                        for entry in source.moves {
                            let mov = board.parse_uci(&entry.best_move.to_uci(&source.castling))?;
                            game.add_move(mov, entry.score);

                            ensure!(
                                board.make_move_simple(mov),
                                "illegal move while converting Monty game"
                            );
                        }

                        game.serialise_into(&mut writer)?;

                        progress(reader.stream_position()?);
                    }
                }

                DataFormat::Sf => {
                    reporter.emit(Event::Note {
                        message: "SF records become single-position Viri games preserving \
                                  position metadata, played move, score and result. \
                                  SF VALUE_NONE scores are omitted.".into()
                    });

                    let mut reader = CompressedTrainingDataEntryReader::new(BufReader::new(
                        File::open(&options.input)?,
                    ))
                    .map_err(|e| anyhow!("{e:?}"))?;

                    let mut count = 0;
                    while reader.has_next() {
                        let entry = reader.next();
                        if !finite::valid_sf_score(entry.score) {
                            continue;
                        }

                        let board =
                            board_from_fen(&entry.pos.fen().map_err(|e| anyhow!("{e:?}"))?)?;
                        let mov = board.parse_uci(&entry.mv.as_uci())?;
                        let black = entry.pos.side_to_move().ordinal() != 0;

                        let result = f32::from(if black {
                            1 - entry.result
                        } else {
                            1 + entry.result
                        }) / 2.0;

                        let score = if black { -entry.score } else { entry.score };

                        let mut game = Game::new(&board);

                        game.set_outcome(outcome(result)?);
                        game.add_move(mov, score);
                        game.serialise_into(&mut writer)?;

                        count += 1;

                        if count % 8192 == 0 {
                            progress(reader.read_bytes());
                        }
                    }
                }

                DataFormat::Text => {
                    reporter.emit(Event::Note {
                        message: "Text records become single-position Viri games preserving the \
                                  FEN, score and result, with a synthetic legal move. \
                                  Disable tactical-move filtering on these converted files.".into()
                    });

                    let mut reader = BufReader::new(File::open(&options.input)?);
                    let mut line = String::new();
                    let mut bytes = 0;

                    while reader.read_line(&mut line)? != 0 {
                        let record = line.trim().parse::<ChessBoard>().map_err(|e| anyhow!(e))?;

                        finite::validate_board(&record)?;

                        let fen = line.split('|').next().context("missing FEN")?.trim();
                        let mut board = board_from_fen(fen)?;
                        let black = fen.split_whitespace().nth(1) == Some("b");

                        let score = if black {
                            record
                                .score()
                                .checked_neg()
                                .context("unrepresentable score")?
                        } else {
                            record.score()
                        };

                        let result = if black {
                            1.0 - record.result()
                        } else {
                            record.result()
                        };

                        let mov = board
                            .legal_moves()
                            .into_iter()
                            .next()
                            .context("terminal position cannot be represented as a scored Viri \
                                      move; use Bullet output")?;

                        let mut game = Game::new(&board);

                        game.set_outcome(outcome(result)?);
                        game.add_move(mov, score);
                        game.serialise_into(&mut writer)?;

                        bytes += line.len() as u64;

                        progress(bytes);

                        line.clear();
                    }
                }

                _ => {
                    reporter.emit(Event::Note {
                        message: "Position-only sources become single-position Viri games with a \
                                  synthetic legal move. Bullet-normalized board orientation, \
                                  score and result are preserved; unavailable clocks, castling \
                                  rights and original game history cannot be recovered. \
                                  Train these converted files with filters disabled. \
                                  Terminal positions cannot be represented by a scored Viri move \
                                  and cause an explicit error.".into()
                    });

                    finite::visit(
                        &options.input,
                        options.from,
                        None,
                        |board| {
                            position_game(board)?.serialise_into(&mut writer)?;
                            Ok(false)
                        },
                        &progress,
                    )?;
                }
            }

            writer.flush()?;
        }
    }

    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(&options.output)
        .map_err(|e| anyhow!("publish conversion: {e}"))?;

    progress(total);

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
    reporter.emit(Event::Phase {
        name: "Writing converted data".into(),
        completed: 0,
        total,
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
                    reporter.emit(Event::Phase {
                        name: "Writing converted data".into(),
                        completed,
                        total,
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

fn phase_progress<'a>(
    reporter: &'a Reporter,
    name: &'a str,
    total: Option<u64>,
) -> impl Fn(u64) + 'a {
    let last = Cell::new(Instant::now());

    move |completed| {
        if last.get().elapsed() >= Duration::from_millis(200) || total == Some(completed) {
            reporter.emit(Event::Phase {
                name: name.into(),
                completed,
                total,
            });

            last.set(Instant::now());
        }
    }
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
    finite::validate_board(&record)?;

    let mut squares = [' '; 64];
    for (piece, square) in record {
        squares[usize::from(square)] = match piece {
             0 => 'P',
             1 => 'N',
             2 => 'B',
             3 => 'R',
             4 => 'Q',
             5 => 'K',
             8 => 'p',
             9 => 'n',
            10 => 'b',
            11 => 'r',
            12 => 'q',
            13 => 'k',

            _ => unreachable!("validated piece"),
        };
    }

    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let piece = squares[8 * rank + file];

            if piece == ' ' {
                empty += 1;
            } else {
                if empty > 0 {
                    fen.push(char::from(b'0' + empty));
                    empty = 0;
                }

                fen.push(piece);
            }
        }

        if empty > 0 {
            fen.push(char::from(b'0' + empty));
        }

        if rank != 0 {
            fen.push('/');
        }
    }

    fen.push_str(" w - - 0 1");

    let mut board = board_from_fen(&fen)?;
    let mov = board
        .legal_moves()
        .into_iter()
        .next()
        .context(
            "terminal position has no legal move: \
            use Bullet output, since Viri stores scores on moves"
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

    fn collected(path: &std::path::Path, format: DataFormat) -> Vec<ChessBoard> {
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

                    threads: 1,
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

                    threads: 1,
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
                        threads: 1,
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
