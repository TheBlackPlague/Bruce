use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use bullet_lib::game::formats::{
    bulletformat::{BulletFormat, ChessBoard},
    montyformat::{
        MontyValueFormat, SearchResult,
        chess::{Castling, Position},
    },
    sfbinpack::{
        self, TrainingDataEntry,
        chess::{
            coords::Square,
            r#move::{Move as SfMove, MoveType},
            piece::Piece,
            piecetype::PieceType,
        },
    },
    viriformat::{
        chess::{
            CHESS960,
            board::{Board, DrawType, GameOutcome, WinType, movegen::MoveList},
            chessmove::Move,
        },
        dataformat::Game,
    },
};
use clap::Args;

use crate::{
    cli::{self, Progress, Reporter},
    config::DataSource,
    data::{self, Format, Record},
};

#[derive(Debug, Args)]
#[command(after_help = "Example:\n  bruce convert --from text --to bullet --input positions.txt --output \
                        positions.bullet --separator ';'")]
pub struct Options
{
    #[arg(long, value_enum)]
    pub from: Format,

    #[arg(long, value_enum)]
    pub to: Format,

    #[arg(long)]
    pub input: PathBuf,

    #[arg(long)]
    pub output: PathBuf,

    #[arg(long)]
    pub separator: Option<String>,

    #[arg(long)]
    pub threads: Option<usize>,
}

#[derive(Debug)]
pub struct Summary
{
    pub positions: u64,
    pub skipped: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

impl std::fmt::Display for Summary
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(
            formatter,
            "Converted {} positions ({} bytes to {} bytes; {} skipped)",
            self.positions, self.input_bytes, self.output_bytes, self.skipped
        )
    }
}

pub fn run(options: &Options, plain: bool) -> Result<Summary>
{
    let reporter = Reporter::new(cli::OutputOptions {
        display: if plain { cli::DisplayMode::Plain } else { cli::DisplayMode::Auto },
        ..Default::default()
    });

    reporter.start("Convert", &options.input.to_string_lossy());
    reporter.context("Output", options.output.to_string_lossy());

    let result = run_reported(options, &reporter, Arc::new(AtomicBool::new(false)));
    if let Err(error) = &result {
        reporter.error(
            "Dataset conversion failed",
            "Check the input format and destination permissions",
            &format!("{error:#}"),
        );
    }

    let outcome = if result.is_ok() { cli::Outcome::Success } else { cli::Outcome::Failed };
    let finished = reporter.finish(outcome);
    let summary = result?;

    finished?;

    Ok(summary)
}

pub fn run_reported(options: &Options, reporter: &Reporter, stop: Arc<AtomicBool>) -> Result<Summary>
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_inner(options, reporter, &stop))).map_err(|panic| {
        let detail = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("Unknown panic while decoding or encoding the dataset");

        anyhow!("{detail}").context("Could not convert the dataset")
    })?
}

fn check_cancelled(stop: &AtomicBool) -> Result<()>
{
    if stop.load(Ordering::Relaxed) {
        return Err(cli::Cancelled.into());
    }

    Ok(())
}

fn run_inner(options: &Options, reporter: &Reporter, stop: &AtomicBool) -> Result<Summary>
{
    reporter.phase("Preparing conversion");

    check_cancelled(stop)?;

    let threads = options.threads.unwrap_or(1);

    ensure!((1..=256).contains(&threads), "Threads must be between 1 and 256");
    ensure!(!options.output.exists(), "The output file already exists");

    if options.from == Format::Text || options.to == Format::Text {
        let separator = options.separator.as_deref().context("Text conversion needs --separator")?;
        ensure!(!separator.is_empty() && !separator.contains(['\n', '\r']), "Choose a nonempty text separator");
    }

    let source = DataSource { format: options.from, path: options.input.clone(), separator: options.separator.clone() };
    data::validate_source(&source, None)?;
    check_cancelled(stop)?;

    let parent = options
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).context("Could not create the output file")?;

    let mut summary =
        Summary { positions: 0, skipped: 0, input_bytes: std::fs::metadata(&options.input)?.len(), output_bytes: 0 };

    if options.from == options.to {
        summary.positions =
            copy_validated_dataset(&source, temporary.as_file_mut(), summary.input_bytes, reporter, stop)?;
    } else {
        reporter.phase("Converting positions");

        CHESS960.store(true, Ordering::Relaxed);

        let mut writer = BufWriter::new(temporary.as_file_mut());

        if matches!(options.from, Format::Bullet | Format::Text)
            && matches!(options.to, Format::Monty | Format::Viri | Format::Sf)
        {
            reporter
                .warning("Positions without moves use a legal placeholder move; original games cannot be recovered");
        }

        if options.from == Format::Bullet && options.to != Format::Bullet {
            reporter.warning("Bullet data has no original side, castling rights or move counters");
        }

        if options.from == Format::Monty && options.to == Format::Viri {
            summary.positions = convert_games(
                options,
                &mut writer,
                threads,
                reporter,
                stop,
                |reader| {
                    let game = MontyValueFormat::deserialise_from(reader, Vec::new())?;
                    let positions = game.moves.len();
                    Ok((game, positions))
                },
                encode_monty_games,
            )?;
        } else if options.from == Format::Viri && options.to == Format::Monty {
            summary.positions = convert_games(
                options,
                &mut writer,
                threads,
                reporter,
                stop,
                |reader| {
                    let game = Game::deserialise_from(reader, Vec::new())?;
                    let positions = game.moves.len();

                    Ok((game, positions))
                },
                encode_viri_games,
            )?;
        } else {
            let mut records = Vec::with_capacity(8192);

            let (_, skipped) = data::visit_counted_stoppable(&source, stop, |record| {
                check_cancelled(stop)?;
                if options.from == Format::Sf
                    && options.to == Format::Viri
                    && i16::try_from(record.white_score()).is_err()
                {
                    summary.skipped += 1;
                    return Ok(false);
                }

                records.push(record);
                if records.len() == 8192 {
                    write_parallel(&records, threads, |chunk| encode(chunk, options), &mut writer)?;
                    summary.positions += records.len() as u64;
                    records.clear();

                    reporter.progress(Progress::Work {
                        completed: summary.positions,
                        total: None,
                        unit: "positions converted",
                    });
                }

                Ok(false)
            })?;

            summary.skipped += skipped;

            check_cancelled(stop)?;

            write_parallel(&records, threads, |chunk| encode(chunk, options), &mut writer)?;
            summary.positions += records.len() as u64;

            ensure!(summary.positions > 0, "No positions could be converted");
        }

        reporter.progress(Progress::Work { completed: summary.positions, total: None, unit: "positions converted" });
        check_cancelled(stop)?;

        reporter.phase("Flushing converted dataset");
        writer.flush().context("Could not flush the converted dataset")?;
    }

    check_cancelled(stop)?;
    reporter.phase("Saving converted dataset");
    temporary.as_file().sync_all().context("Could not sync the converted dataset")?;

    summary.output_bytes = temporary.as_file().metadata()?.len();
    check_cancelled(stop)?;

    temporary
        .persist_noclobber(&options.output)
        .map_err(|error| anyhow!(error.error))
        .context("Could not publish the converted dataset; check the destination permissions and free space")?;

    if summary.skipped > 0 {
        reporter.warning(&format!(
            "Skipped {} positions with scores that cannot be represented in the output",
            summary.skipped
        ));
    }

    Ok(summary)
}

fn copy_validated_dataset(
    source: &DataSource,
    writer: &mut impl Write,
    input_bytes: u64,
    reporter: &Reporter,
    stop: &AtomicBool,
) -> Result<u64>
{
    reporter.phase("Checking positions before copying");

    let mut validated: u64 = 0;
    let (positions, skipped) = data::visit_counted_stoppable(source, stop, |_| {
        check_cancelled(stop)?;
        validated += 1;
        if validated.is_multiple_of(8192) {
            reporter.progress(Progress::Work { completed: validated, total: None, unit: "positions checked" });
        }
        Ok(false)
    })?;

    let positions = positions + skipped;
    reporter.progress(Progress::Work { completed: positions, total: None, unit: "positions checked" });

    check_cancelled(stop)?;
    reporter.phase("Copying validated dataset");

    let mut reader = File::open(&source.path)?;
    let mut buffer = [0; 64 * 1024];
    let mut copied = 0;

    loop {
        check_cancelled(stop)?;

        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }

        writer.write_all(&buffer[..length])?;
        copied += length as u64;

        reporter.progress(Progress::Work { completed: copied, total: Some(input_bytes), unit: "bytes copied" });
    }

    Ok(positions)
}

fn write_parallel<T: Sync>(
    items: &[T],
    threads: usize,
    encode: impl Fn(&[T]) -> Result<Vec<u8>> + Sync,
    writer: &mut impl Write,
) -> Result<()>
{
    if items.is_empty() {
        return Ok(());
    }

    if threads == 1 {
        writer.write_all(&encode(items)?)?;
        return Ok(());
    }

    let worker_count = threads.min(items.len());
    let chunk_size = items.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        let encode = &encode;
        let handles = items.chunks(chunk_size).map(|chunk| scope.spawn(move || encode(chunk))).collect::<Vec<_>>();
        for handle in handles {
            let bytes = handle.join().map_err(|_| anyhow!("Could not encode the dataset"))??;
            writer.write_all(&bytes)?
        }

        Ok(())
    })
}

fn encode(records: &[Record], options: &Options) -> Result<Vec<u8>>
{
    let mut bytes = Vec::new();

    if options.to == Format::Sf {
        let mut writer = sfbinpack::CompressedTrainingDataEntryWriter::new_in_memory()?;
        for record in records {
            writer.write_entry(&sf_entry(record)?)?
        }

        return Ok(writer.into_bytes()?);
    }

    for record in records {
        match options.to {
            Format::Bullet => bytes.extend_from_slice(ChessBoard::as_bytes_slice(&[record.board])),

            Format::Text => {
                let fen = record_fen(record);

                let separator = options.separator.as_deref().context("Text conversion needs --separator")?;

                writeln!(
                    &mut bytes,
                    "{fen}{separator}{}{separator}{}",
                    record.white_score(),
                    f32::from(record.white_result()) / 2.0
                )?;
            },

            Format::Viri => {
                let (board, mov) = position_move(record)?;
                let mut game = Game::new(&board);

                game.set_outcome(outcome(record.white_result()));
                game.add_move(
                    mov,
                    record.white_score().try_into().context("The score cannot be represented in Viri format")?,
                );

                game.serialise_into(&mut bytes)?;
            },

            Format::Monty => {
                let (board, mov) = position_move(record)?;
                let fen = data::viri_fen(&board);
                let mut castling = Castling::default();
                let position = Position::parse_fen(&fen, &mut castling);
                let candidate = find_monty_move(&position, &castling, &format!("{}", mov.display(true)))?;

                MontyValueFormat {
                    startpos: position,
                    castling,
                    result: f32::from(record.white_result()) / 2.0,
                    moves: vec![SearchResult {
                        best_move: candidate,
                        score: record
                            .white_score()
                            .try_into()
                            .context("The score cannot be represented in Monty format")?,
                    }],
                }
                .serialise_into(&mut bytes)?;
            },

            Format::Sf => unreachable!(),
        }
    }

    Ok(bytes)
}

fn outcome(result: u8) -> GameOutcome
{
    match result {
        0 => GameOutcome::BlackWin(WinType::Adjudication),
        1 => GameOutcome::Draw(DrawType::Adjudication),
        _ => GameOutcome::WhiteWin(WinType::Adjudication),
    }
}

fn record_fen(record: &Record) -> std::borrow::Cow<'_, str>
{
    match &record.fen {
        Some(fen) => std::borrow::Cow::Borrowed(fen),
        None => std::borrow::Cow::Owned(data::bullet_fen(&record.board)),
    }
}

fn position_move(record: &Record) -> Result<(Board, Move)>
{
    let fen = record_fen(record);

    let board = data::parse_board(&fen).context("Could not convert the position")?;
    if let Some(uci) = &record.next_move {
        let mov = board.parse_uci(uci).context("Could not convert the played move")?;
        let mut copy = board.clone();

        ensure!(copy.make_move_simple(mov), "The dataset contains an illegal move");

        return Ok((board, mov));
    }

    let mut list = MoveList::new();
    board.generate_moves(&mut list);

    for &mov in list.iter_moves() {
        let mut copy = board.clone();

        if copy.make_move_simple(mov) {
            return Ok((board, mov));
        }
    }

    bail!("A terminal position has no move; use Bullet or text output")
}

fn sf_entry(record: &Record) -> Result<TrainingDataEntry>
{
    let (board, mov) = position_move(record)?;
    let fen = data::viri_fen(&board);

    let mut fields = fen.split_whitespace().map(str::to_string).collect::<Vec<_>>();

    let mut rights = String::new();
    for right in fields[2].chars() {
        rights.push(match right {
            'A' => 'Q',
            'H' => 'K',
            'a' => 'q',
            'h' => 'k',
            '-' => '-',
            _ => bail!("SF binpack cannot preserve Chess960 castling rights"),
        });
    }
    fields[2] = rights;

    let pos = sfbinpack::chess::position::Position::from_fen(&fields.join(" "))
        .map_err(|_| anyhow!("Could not represent the position in SF binpack"))?;
    let from = Square::new(u32::from(mov.from().inner()));
    let to = Square::new(u32::from(mov.to().inner()));
    let kind = if mov.is_castle() {
        MoveType::Castle
    } else if mov.is_ep() {
        MoveType::EnPassant
    } else if mov.is_promo() {
        MoveType::Promotion
    } else {
        MoveType::Normal
    };

    let promoted = match mov.promotion_type() {
        None => Piece::none(),
        Some(piece) => {
            use bullet_lib::game::formats::viriformat::chess::piece::PieceType as ViriPiece;

            let kind = match piece {
                ViriPiece::Knight => PieceType::Knight,
                ViriPiece::Bishop => PieceType::Bishop,
                ViriPiece::Rook => PieceType::Rook,
                ViriPiece::Queen => PieceType::Queen,
                _ => bail!("Invalid promotion"),
            };
            Piece::new(kind, pos.side_to_move())
        },
    };

    Ok(TrainingDataEntry {
        pos,
        mv: SfMove::new(from, to, kind, promoted),
        score: record.board.score,
        ply: record.ply.unwrap_or(0),
        result: i16::from(record.board.result) - 1,
    })
}

fn find_monty_move(
    position: &Position,
    castling: &Castling,
    uci: &str,
) -> Result<bullet_lib::game::formats::montyformat::chess::Move>
{
    let mut found = None;
    position.map_legal_moves(castling, |mov| {
        if data::monty_uci(mov, castling) == uci {
            found = Some(mov)
        }
    });

    found.context("Could not convert the played move")
}

fn convert_games<T: Sync>(
    options: &Options,
    writer: &mut impl Write,
    threads: usize,
    reporter: &Reporter,
    stop: &AtomicBool,
    decode: impl Fn(&mut BufReader<File>) -> Result<(T, usize)>,
    encode: impl Fn(&[T]) -> Result<Vec<u8>> + Sync,
) -> Result<u64>
{
    let mut reader = BufReader::new(File::open(&options.input)?);
    let games_per_batch = threads * 64;
    let mut games = Vec::with_capacity(games_per_batch);

    let mut count = 0;
    while !reader.fill_buf()?.is_empty() {
        check_cancelled(stop)?;
        let (game, positions) = decode(&mut reader)?;
        count += positions as u64;
        games.push(game);

        if games.len() == games_per_batch {
            write_parallel(&games, threads, &encode, writer)?;
            games.clear();
            reporter.progress(Progress::Work { completed: count, total: None, unit: "positions converted" });
        }
    }

    check_cancelled(stop)?;
    write_parallel(&games, threads, &encode, writer)?;

    reporter.progress(Progress::Work { completed: count, total: None, unit: "positions converted" });
    ensure!(count > 0, "No positions could be converted");

    Ok(count)
}

fn encode_monty_games(games: &[MontyValueFormat]) -> Result<Vec<u8>>
{
    let mut bytes = Vec::new();
    for source in games {
        let mut board = data::parse_board(&data::monty_fen(&source.startpos, &source.castling))?;
        let mut game = Game::new(&board);

        ensure!([0.0, 0.5, 1.0].contains(&source.result), "Invalid result in the dataset");
        game.set_outcome(outcome((2.0 * source.result) as u8));

        for entry in &source.moves {
            let mov = board.parse_uci(&data::monty_uci(entry.best_move, &source.castling))?;
            game.add_move(mov, entry.score);
            ensure!(board.make_move_simple(mov), "Invalid move in a Monty game");
        }

        game.serialise_into(&mut bytes)?;
    }

    Ok(bytes)
}

fn encode_viri_games(games: &[Game]) -> Result<Vec<u8>>
{
    let mut bytes = Vec::new();
    for source in games {
        let board = source.initial_position();
        let mut castling = Castling::default();

        let startpos = Position::parse_fen(&data::viri_fen(&board), &mut castling);

        let mut position = startpos;

        let mut game = MontyValueFormat {
            startpos,
            castling,
            result: f32::from(data::viri_result(source.outcome())) / 2.0,
            moves: Vec::new(),
        };

        for &(mov, score) in &source.moves {
            let mov = find_monty_move(&position, &castling, &format!("{}", mov.display(true)))?;
            game.moves.push(SearchResult { best_move: mov, score: score.get() });
            position.make(mov, &castling);
        }

        game.serialise_into(&mut bytes)?;
    }

    Ok(bytes)
}
