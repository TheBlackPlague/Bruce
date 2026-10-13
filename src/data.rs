use std::{
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, anyhow, ensure};
use bullet_lib::{
    game::formats::{
        bulletformat::ChessBoard,
        montyformat::{
            MontyValueFormat,
            chess::{Castling, Position},
        },
        sfbinpack::{
            self,
            chess::{color::Color, piecetype::PieceType},
        },
        viriformat::{chess::board::Board, dataformat::Game},
    },
    value::loader::{DirectSequentialDataLoader, MontyBinpackLoader, SfBinpackLoader, ViriBinpackLoader},
};
use bullet_trainer::reader::DataReader;
use clap::ValueEnum;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    config::{DataConfig, DataSource, Filter},
    schedule::BATCHES_PER_SUPERBATCH,
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Format
{
    #[value(alias = "sfbinpack")]
    #[serde(alias = "sfbinpack")]
    Sf,
    Monty,
    Bullet,
    Viri,
    Text,
}

pub struct Record
{
    pub board: ChessBoard,
    pub fen: Option<String>,
    pub next_move: Option<String>,
    pub ply: Option<u16>,
    pub in_check: Option<bool>,
    pub tactical: Option<bool>,
}

impl Record
{
    pub fn white_score(&self) -> i32
    {
        if self.black_to_move() { -i32::from(self.board.score) } else { i32::from(self.board.score) }
    }

    pub fn white_result(&self) -> u8 { if self.black_to_move() { 2 - self.board.result } else { self.board.result } }

    fn black_to_move(&self) -> bool { self.fen.as_ref().is_some_and(|fen| fen.split_whitespace().nth(1) == Some("b")) }
}

fn accepted(record: &Record, filter: Option<&Filter>) -> bool
{
    let Some(filter) = filter else { return true };
    accepted_position(filter, record.board.score, record.board.occ().count_ones(), record.ply)
        && (!filter.exclude_check || record.in_check == Some(false))
        && (!filter.exclude_tactical || record.tactical == Some(false))
}

fn accepted_position(filter: &Filter, score: i16, pieces: u32, ply: Option<u16>) -> bool
{
    let score = u32::from(score.unsigned_abs());
    score >= filter.minimum_eval
        && score < filter.maximum_eval
        && pieces >= filter.minimum_pieces
        && ply.is_none_or(|ply| ply >= filter.minimum_ply && filter.maximum_ply.is_none_or(|max| ply <= max))
}

pub fn validate_source(source: &DataSource, filter: Option<&Filter>) -> Result<()>
{
    let size = std::fs::metadata(&source.path).context("Could not open the dataset")?.len();
    ensure!(size > 0, "The dataset is empty");

    if source.format == Format::Bullet {
        ensure!(size.is_multiple_of(size_of::<ChessBoard>() as u64), "The dataset is truncated");
    }

    if source.format == Format::Text {
        let separator = source.separator.as_deref().context("Text data needs a separator")?;
        ensure!(!separator.is_empty() && !separator.contains(['\n', '\r']), "Choose a nonempty text separator");
    }

    if source.format == Format::Bullet
        && let Some(filter) = filter
    {
        ensure!(
            filter.minimum_ply == 0
                && filter.maximum_ply.is_none()
                && !filter.exclude_check
                && !filter.exclude_tactical,
            "Bullet data does not contain move or ply information"
        );
    }

    if source.format == Format::Text
        && let Some(filter) = filter
    {
        ensure!(!filter.exclude_tactical, "Text data does not contain the played move");
    }

    Ok(())
}

pub fn validate(config: &DataConfig) -> Result<()>
{
    for source in &config.sources {
        validate_source(source, config.filter.as_ref())?;

        let mut count = 0;
        visit_boards(source, config.filter.as_ref(), |board| {
            if source.format == Format::Sf && !valid_sf_score(board.score) {
                return Ok(false);
            }
            count += 1;

            Ok(false)
        })?;

        ensure!(count > 0, "No positions pass the filter");
    }

    Ok(())
}

pub fn visit(source: &DataSource, filter: Option<&Filter>, callback: impl FnMut(Record) -> Result<bool>) -> Result<()>
{
    visit_validated(source, true, filter, None, callback).map(|_| ())
}

fn visit_validated(
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
        visit_inner(source, details, stop, &mut skipped, &mut |record| {
            validate_board(&record.board)?;
            if accepted(&record, filter) {
                positions += 1;
                callback(record)
            } else {
                Ok(false)
            }
        })
    }))
    .map_err(|_| anyhow!("Could not decode the dataset"))??;

    Ok((positions, skipped))
}

fn visit_inner(
    source: &DataSource,
    details: bool,
    stop: Option<&AtomicBool>,
    skipped: &mut u64,
    callback: &mut impl FnMut(Record) -> Result<bool>,
) -> Result<()>
{
    match source.format {
        Format::Bullet => {
            let total = std::fs::metadata(&source.path)?.len() as usize / size_of::<ChessBoard>();
            let path = source.path.to_string_lossy();

            let mut seen = 0;
            let mut status = Ok(false);

            DirectSequentialDataLoader::new(&[&path]).read_chunks(0, |boards: &[ChessBoard]| {
                for &board in boards {
                    status = check_cancelled(stop).and_then(|()| {
                        callback(Record {
                            board,
                            fen: None,
                            next_move: None,
                            ply: None,
                            in_check: None,
                            tactical: None,
                        })
                    });

                    seen += 1;

                    if !matches!(status, Ok(false)) || seen == total {
                        return true;
                    }
                }

                false
            });

            status?;
        },

        Format::Text => {
            let separator = source.separator.as_deref().context("Text data needs a separator")?;
            for line in BufReader::new(File::open(&source.path)?).lines() {
                check_cancelled(stop)?;
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                let fields = line.split(separator).map(str::trim).collect::<Vec<_>>();
                ensure!(fields.len() == 3, "Expected FEN, score and result in each text row");

                let position = parse_board(fields[0]).context("Invalid FEN in the dataset")?;
                let score = fields[1].parse::<i32>().map_err(|_| anyhow!("Invalid score in the dataset"))?;
                let normalized =
                    if fields[0].split_whitespace().nth(1) == Some("b") { score.checked_neg() } else { Some(score) };
                let score = normalized
                    .and_then(|score| i16::try_from(score).ok())
                    .context("The score cannot be represented")?;
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
                    fen: details.then(|| fields[0].to_string()),
                    next_move: None,
                    ply: Some(position.ply().try_into().context("Position ply is too large")?),
                    in_check: Some(position.in_check()),
                    tactical: None,
                })? {
                    break;
                }
            }
        },

        Format::Monty => {
            let mut reader = BufReader::new(File::open(&source.path)?);
            while !reader.fill_buf()?.is_empty() {
                check_cancelled(stop)?;

                let game = MontyValueFormat::deserialise_from(&mut reader, Vec::new())
                    .context("Could not read a Monty game")?;
                let mut position = game.startpos;

                for entry in game.moves {
                    check_cancelled(stop)?;

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
        },
        Format::Viri => {
            let mut reader = BufReader::new(File::open(&source.path)?);

            while !reader.fill_buf()?.is_empty() {
                check_cancelled(stop)?;

                let game = Game::deserialise_from(&mut reader, Vec::new()).context("Could not read a Viri game")?;
                let result = viri_result(game.outcome());
                let mut position = game.initial_position();

                for (mov, score) in game.moves {
                    check_cancelled(stop)?;

                    let score = score.get();
                    let record = Record {
                        board: position.to_bulletformat(result, score)?,
                        fen: details.then(|| viri_fen(&position)),
                        next_move: details.then(|| format!("{}", mov.display(true))),
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
        },
        Format::Sf => {
            let mut reader = BufReader::new(File::open(&source.path)?);
            let mut bytes = Vec::new();

            while sfbinpack::read_chunk_into(&mut reader, &mut bytes)? {
                check_cancelled(stop)?;

                let mut chunk = sfbinpack::ChunkReader::default();
                while chunk.has_next(&bytes) {
                    check_cancelled(stop)?;

                    let entry = chunk.next(&bytes);
                    if entry.score == 32002 {
                        *skipped += 1;
                        continue;
                    }

                    let fen =
                        if details { Some(entry.pos.fen().map_err(|_| anyhow!("Invalid SF position"))?) } else { None };

                    let record = Record {
                        board: sf_board(&entry)?,
                        fen,
                        next_move: details.then(|| {
                            if entry.mv.mtype() == sfbinpack::chess::r#move::MoveType::Castle {
                                format!("{}{}", entry.mv.from(), entry.mv.to())
                            } else {
                                entry.mv.as_uci()
                            }
                        }),
                        ply: Some(entry.ply),
                        in_check: Some(entry.pos.is_checked(entry.pos.side_to_move())),
                        tactical: Some(sf_tactical(&entry)),
                    };

                    if callback(record)? {
                        return Ok(());
                    }
                }
            }
        },
    }

    Ok(())
}

pub fn valid_sf_score(score: i16) -> bool { score != 32002 && score != i16::MIN }

pub fn validate_board(board: &ChessBoard) -> Result<()>
{
    ensure!(board.occ().count_ones() <= 32 && board.result <= 2, "Invalid position in the dataset");

    let mut kings = [0; 2];
    for (piece, square) in *board {
        ensure!(piece & 7 < 6, "Invalid piece in the dataset");

        if piece & 7 == 5 {
            kings[usize::from(piece >> 3)] += 1;

            ensure!(
                if piece >> 3 == 0 { square == board.ksq } else { square ^ 56 == board.opp_ksq },
                "Invalid king square in the dataset"
            );
        }
    }

    ensure!(kings == [1, 1], "A position must contain both kings");
    Ok(())
}

pub fn sf_board(entry: &sfbinpack::TrainingDataEntry) -> Result<ChessBoard>
{
    let mut bbs = [0; 8];
    bbs[0] = entry.pos.pieces_bb(Color::White).bits();
    bbs[1] = entry.pos.pieces_bb(Color::Black).bits();

    for (index, piece) in
        [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King]
            .into_iter()
            .enumerate()
    {
        let mut bb = 0u64;

        bb |= entry.pos.pieces_bb_color(Color::White, piece).bits();
        bb |= entry.pos.pieces_bb_color(Color::Black, piece).bits();

        bbs[index + 2] = bb;
    }

    let black = entry.pos.side_to_move() == Color::Black;
    ensure!((-1..=1).contains(&entry.result), "Invalid result in the dataset");

    let mut board =
        ChessBoard::from_raw(bbs, usize::from(black), 0, 0.5).map_err(|_| anyhow!("Invalid SF position"))?;

    board.score = entry.score;
    board.result = (entry.result + 1) as u8;

    Ok(board)
}

fn sf_tactical(entry: &sfbinpack::TrainingDataEntry) -> bool
{
    use sfbinpack::chess::{piece::Piece, r#move::MoveType};

    matches!(entry.mv.mtype(), MoveType::EnPassant | MoveType::Promotion)
        || (entry.mv.mtype() != MoveType::Castle && entry.pos.piece_at(entry.mv.to()) != Piece::none())
}

pub fn monty_fen(position: &Position, castling: &Castling) -> String
{
    let basic = position.as_fen();

    let mut fields = basic.split_whitespace().map(str::to_string).collect::<Vec<_>>();
    let mut rights = String::new();

    for (mask, side, flank) in [(4, 0, 1), (8, 0, 0), (1, 1, 1), (2, 1, 0)] {
        if position.rights() & mask != 0 {
            rights.push((if side == 0 { b'A' } else { b'a' } + castling.rook_file(side, flank) as u8) as char);
        }
    }

    fields[2] = if rights.is_empty() { "-".to_string() } else { rights };
    fields[3] = if position.enp_sq() == 0 { "-".to_string() } else { square_name(position.enp_sq()) };

    fields.join(" ")
}

pub fn monty_uci(mov: bullet_lib::game::formats::montyformat::chess::Move, castling: &Castling) -> String
{
    let to = if [2, 3].contains(&mov.flag()) {
        let side = usize::from(mov.src() >= 56);
        (56 * side + castling.rook_file(side, usize::from(mov.flag() == 2)) as usize) as u8
    } else {
        mov.to() as u8
    };

    let promotion = if mov.is_promo() { ["n", "b", "r", "q"][(mov.flag() & 3) as usize] } else { "" };

    format!("{}{}{promotion}", square_name(mov.src() as u8), square_name(to))
}

fn square_name(square: u8) -> String { format!("{}{}", char::from(b'a' + square % 8), square / 8 + 1) }

pub fn bullet_fen(board: &ChessBoard) -> String
{
    let mut squares = [' '; 64];
    for (piece, square) in *board {
        squares[usize::from(square)] = "PNBRQKpnbrqk".as_bytes()[usize::from((piece & 7) + 6 * (piece >> 3))] as char
    }

    let mut fen = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let piece = squares[rank * 8 + file];
            if piece == ' ' {
                empty += 1
            } else {
                if empty > 0 {
                    fen.push(char::from(b'0' + empty));
                    empty = 0
                }
                fen.push(piece);
            }
        }

        if empty > 0 {
            fen.push(char::from(b'0' + empty))
        }

        if rank != 0 {
            fen.push('/')
        }
    }

    fen.push_str(" w - - 0 1");
    fen
}

#[derive(Clone)]
pub struct DatasetReader
{
    config: DataConfig,
    seed: u64,
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
}

impl DatasetReader
{
    pub fn new(config: &DataConfig, seed: u64, stop: Arc<AtomicBool>) -> Result<Self>
    {
        config.validate()?;
        for source in &config.sources {
            let mut found = false;
            visit_boards_stoppable(source, config.filter.as_ref(), &stop, |board| {
                if source.format == Format::Sf && !valid_sf_score(board.score) {
                    return Ok(false);
                }
                found = true;
                Ok(true)
            })?;

            ensure!(found, "No positions pass the filter");
        }

        Ok(Self { config: config.clone(), seed, stop, error: Arc::default() })
    }

    pub fn error(&self) -> Option<String> { self.error.lock().unwrap_or_else(|e| e.into_inner()).clone() }

    fn read(&self, skip: usize, mut callback: impl FnMut(&[ChessBoard]) -> bool)
    {
        let sources = &self.config.sources;
        let same_format = sources.iter().all(|source| source.format == sources[0].format);

        let paths = sources.iter().map(|source| source.path.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let paths = paths.iter().map(String::as_str).collect::<Vec<_>>();

        let filter = self.config.filter.clone();

        let buffer = self.config.buffer_size_mb;
        let threads = self.config.loader_threads;

        if same_format {
            match sources[0].format {
                Format::Bullet if filter.is_none() => {
                    if let Err(error) = read_bullet(&paths, skip, callback) {
                        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{error:#}"));
                    }
                    return;
                },

                Format::Monty => {
                    MontyBinpackLoader::new_concat_multiple(&paths, buffer, threads, move |pos, mov, score, _| {
                        let Some(filter) = &filter else { return true };
                        let ply = pos.fullm().saturating_sub(1).saturating_mul(2) + pos.stm() as u16;
                        accepted_position(filter, score, pos.occ().count_ones(), Some(ply))
                            && (!filter.exclude_check || !pos.in_check())
                            && (!filter.exclude_tactical || !(mov.is_capture() || mov.is_promo()))
                    })
                    .read_chunks(skip, callback);
                    return;
                },

                Format::Sf => {
                    SfBinpackLoader::new_concat_multiple(&paths, buffer, threads, move |entry| {
                        if !valid_sf_score(entry.score) {
                            return false;
                        }
                        let Some(filter) = &filter else { return true };
                        let pieces = (entry.pos.pieces_bb(Color::White).bits()
                            | entry.pos.pieces_bb(Color::Black).bits())
                        .count_ones();
                        accepted_position(filter, entry.score, pieces, Some(entry.ply))
                            && (!filter.exclude_check || !entry.pos.is_checked(entry.pos.side_to_move()))
                            && (!filter.exclude_tactical || !sf_tactical(entry))
                    })
                    .read_chunks(skip, callback);
                    return;
                },

                Format::Viri if filter.as_ref().is_none_or(|filter| filter.maximum_ply.is_none()) => {
                    use bullet_lib::game::formats::viriformat::dataformat::Filter as ViriFilter;
                    let mut upstream = ViriFilter::UNRESTRICTED;
                    if let Some(filter) = &filter {
                        upstream.min_ply = u32::from(filter.minimum_ply);
                        upstream.min_pieces = filter.minimum_pieces;
                        upstream.max_eval = filter.maximum_eval;
                        upstream.filter_tactical = filter.exclude_tactical;
                        upstream.filter_check = filter.exclude_check;
                    }
                    let mut filtered = Vec::new();
                    ViriBinpackLoader::new_concat_multiple(&paths, buffer, threads, upstream).read_chunks(
                        skip,
                        |boards| {
                            if let Some(filter) = &filter
                                && filter.minimum_eval > 0
                            {
                                filtered.clear();
                                filtered.extend(
                                    boards
                                        .iter()
                                        .copied()
                                        .filter(|board| u32::from(board.score.unsigned_abs()) >= filter.minimum_eval),
                                );
                                if filtered.is_empty() { false } else { callback(&filtered) }
                            } else {
                                callback(boards)
                            }
                        },
                    );
                    return;
                },

                _ => {},
            }
        }

        self.read_fallback(skip, callback);
    }

    fn read_fallback(&self, skip: usize, mut callback: impl FnMut(&[ChessBoard]) -> bool)
    {
        let capacity = self.config.buffer_size_mb * 1024 * 1024 / size_of::<ChessBoard>();

        let mut values = Vec::with_capacity(capacity);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut remaining = skip;

        loop {
            let mut stopped = false;
            for group in self.config.sources.chunks(self.config.loader_threads) {
                let abort = AtomicBool::new(false);
                let result = std::thread::scope(|scope| {
                    let receivers = group
                        .iter()
                        .map(|source| {
                            let (sender, receiver) = std::sync::mpsc::sync_channel(1);

                            let abort = &abort;
                            let filter = self.config.filter.as_ref();

                            scope.spawn(move || {
                                let mut batch = Vec::with_capacity(4096);
                                let result = visit_validated(source, false, None, None, |record| {
                                    if abort.load(Ordering::Relaxed) {
                                        return Ok(true);
                                    }

                                    if !accepted(&record, filter)
                                        || (source.format == Format::Sf && !valid_sf_score(record.board.score))
                                    {
                                        return Ok(false);
                                    }

                                    batch.push(record.board);

                                    if batch.len() == 4096 {
                                        let ready = std::mem::replace(&mut batch, Vec::with_capacity(4096));

                                        if sender.send(Ok(ready)).is_err() {
                                            return Ok(true);
                                        }
                                    }

                                    Ok(false)
                                });

                                if let Err(error) = result {
                                    let _ = sender.send(Err(error));
                                } else if !batch.is_empty() {
                                    let _ = sender.send(Ok(batch));
                                }
                            });
                            receiver
                        })
                        .collect::<Vec<_>>();

                    let mut active = vec![true; receivers.len()];
                    while active.iter().any(|&active| active) {
                        for (index, receiver) in receivers.iter().enumerate() {
                            if !active[index] {
                                continue;
                            }

                            let Ok(batch) = receiver.recv() else {
                                active[index] = false;
                                continue;
                            };

                            let batch = batch.inspect_err(|_| abort.store(true, Ordering::Relaxed))?;
                            for board in batch {
                                if remaining > 0 {
                                    remaining -= 1;
                                    continue;
                                }

                                values.push(board);
                                if values.len() == capacity {
                                    values.shuffle(&mut rng);

                                    stopped = callback(&values);
                                    values.clear();

                                    if stopped {
                                        abort.store(true, Ordering::Relaxed);
                                        return Ok::<_, anyhow::Error>(());
                                    }
                                }
                            }
                        }
                    }

                    Ok(())
                });

                if let Err(error) = result {
                    *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{error:#}"));
                    return;
                }

                if stopped {
                    return;
                }
            }
            if !values.is_empty() {
                values.shuffle(&mut rng);

                if callback(&values) {
                    return;
                }
                values.clear();
            }
        }
    }
}

impl DataReader<ChessBoard> for DatasetReader
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip: usize, mut callback: F)
    {
        if self.stop.load(Ordering::Relaxed) {
            return;
        }

        let boundary = BATCHES_PER_SUPERBATCH * self.config.batch_size;
        let mut emitted = 0usize;
        self.read(skip, |mut values| {
            while !values.is_empty() {
                if emitted.is_multiple_of(boundary) && self.stop.load(Ordering::Relaxed) {
                    return true;
                }

                let count = values.len().min(boundary - emitted % boundary);
                if callback(&values[..count]) {
                    return true;
                }

                emitted += count;
                values = &values[count..];
            }

            false
        });
    }
}

pub fn parse_board(fen: &str) -> Result<Board>
{
    bullet_lib::game::formats::viriformat::chess::CHESS960.store(true, Ordering::Relaxed);

    let mut board = Board::new();

    let mut fields = fen.split_whitespace().map(str::to_string).collect::<Vec<_>>();
    ensure!(fields.len() == 6, "Invalid FEN in the dataset");

    fields[2] = fields[2]
        .chars()
        .map(|right| match right {
            'K' => 'H',
            'Q' => 'A',
            'k' => 'h',
            'q' => 'a',
            other => other,
        })
        .collect();
    ensure!(
        fields[2].chars().all(|right| right == '-' || ('A'..='H').contains(&right) || ('a'..='h').contains(&right)),
        "Invalid castling rights"
    );

    board.set_from_fen(&fields.join(" "))?;

    Ok(board)
}

pub fn viri_fen(board: &Board) -> String
{
    let text = board.to_string();
    let mut fields = text.split_whitespace().map(str::to_string).collect::<Vec<_>>();
    let rights = board.castling_rights();
    let mut value = String::new();
    for (square, base) in [(rights.wk, b'A'), (rights.wq, b'A'), (rights.bk, b'a'), (rights.bq, b'a')] {
        if let Some(square) = square {
            value.push(char::from(base + square.inner() % 8));
        }
    }
    fields[2] = if value.is_empty() { "-".to_string() } else { value };
    fields.join(" ")
}

pub fn viri_result(result: bullet_lib::game::formats::viriformat::dataformat::WDL) -> u8
{
    use bullet_lib::game::formats::viriformat::dataformat::WDL;

    match result {
        WDL::Win => 2,
        WDL::Draw => 1,
        WDL::Loss => 0,
    }
}

pub fn visit_boards(
    source: &DataSource,
    filter: Option<&Filter>,
    mut callback: impl FnMut(ChessBoard) -> Result<bool>,
) -> Result<()>
{
    visit_validated(source, false, filter, None, |record| callback(record.board)).map(|_| ())
}

pub fn visit_counted(source: &DataSource, callback: impl FnMut(Record) -> Result<bool>) -> Result<(u64, u64)>
{
    visit_validated(source, true, None, None, callback)
}

pub fn visit_boards_stoppable(
    source: &DataSource,
    filter: Option<&Filter>,
    stop: &AtomicBool,
    mut callback: impl FnMut(ChessBoard) -> Result<bool>,
) -> Result<()>
{
    visit_validated(source, false, filter, Some(stop), |record| callback(record.board)).map(|_| ())
}

pub fn visit_counted_stoppable(
    source: &DataSource,
    stop: &AtomicBool,
    callback: impl FnMut(Record) -> Result<bool>,
) -> Result<(u64, u64)>
{
    visit_validated(source, true, None, Some(stop), callback)
}

fn check_cancelled(stop: Option<&AtomicBool>) -> Result<()>
{
    if stop.is_some_and(|stop| stop.load(Ordering::Relaxed)) {
        return Err(crate::cli::Cancelled.into());
    }
    Ok(())
}

fn read_bullet(paths: &[&str], skip: usize, mut callback: impl FnMut(&[ChessBoard]) -> bool) -> Result<()>
{
    let sizes = paths
        .iter()
        .map(|path| Ok(std::fs::metadata(path)?.len() as usize / size_of::<ChessBoard>()))
        .collect::<Result<Vec<_>>>()?;
    let positions: usize = sizes.iter().sum();
    ensure!(positions > 0, "The dataset is empty");
    let mut offset = skip % positions;
    let mut first = 0;
    while offset >= sizes[first] {
        offset -= sizes[first];
        first += 1;
    }
    let mut rotated = paths.to_vec();
    rotated.rotate_left(first);
    if offset > 0 {
        if read_bullet_tail(rotated[0], offset, &mut callback)? {
            return Ok(());
        }
        rotated.rotate_left(1);
    }
    DirectSequentialDataLoader::new(&rotated).read_chunks(0, callback);
    Ok(())
}

fn read_bullet_tail(path: &str, offset: usize, callback: &mut impl FnMut(&[ChessBoard]) -> bool) -> Result<bool>
{
    let mut file = File::open(path).context("Could not open the resumed dataset")?;
    file.seek(SeekFrom::Start((offset as u64) * size_of::<ChessBoard>() as u64))?;
    let capacity = (128 * 1024 * 1024) / size_of::<ChessBoard>();
    let mut boards = vec![ChessBoard::default(); capacity];

    loop {
        // SAFETY: pinned bullet format ChessBoard is repr(C), exactly 32 bytes, with integer fields and no padding;
        // every byte pattern is valid
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(boards.as_mut_ptr().cast::<u8>(), boards.len() * size_of::<ChessBoard>())
        };

        let count = file.read(bytes).context("Could not read the resumed dataset")?;

        if count == 0 {
            return Ok(false);
        }

        ensure!(count.is_multiple_of(size_of::<ChessBoard>()), "The resumed dataset is truncated");
        if callback(&boards[..count / size_of::<ChessBoard>()]) {
            return Ok(true);
        }
    }
}

#[cfg(test)]
mod reporting_tests
{
    use bullet_lib::game::formats::bulletformat::BulletFormat;

    use super::*;

    #[test]
    fn silent_resume_adapter_preserves_pinned_loader_sequence_across_files_and_epochs()
    {
        let root = tempfile::tempdir().unwrap();
        let files = [root.path().join("a.bullet"), root.path().join("b.bullet")];

        for (index, path) in files.iter().enumerate() {
            let mut file = std::io::BufWriter::new(File::create(path).unwrap());

            for score in (index * 3)..(index * 3 + 3) {
                let mut board = parse_board("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap().to_bulletformat(1, 0).unwrap();
                board.score = score as i16;

                ChessBoard::write_to_bin(&mut file, &[board]).unwrap();
            }
        }

        let names = files.iter().map(|p| p.to_str().unwrap()).collect::<Vec<_>>();
        for skip in [0, 1, 3, 4, 6, 13] {
            let mut expected = Vec::new();

            DirectSequentialDataLoader::new(&names).read_chunks(skip, |boards: &[ChessBoard]| {
                expected.extend(boards.iter().map(|b| b.score).take(16 - expected.len()));
                expected.len() == 16
            });

            let mut actual = Vec::new();
            read_bullet(&names, skip, |boards| {
                actual.extend(boards.iter().map(|b| b.score).take(16 - actual.len()));
                actual.len() == 16
            })
            .unwrap();

            assert_eq!(actual, expected, "skip {skip}");
        }
    }

    #[test]
    fn cancellation_is_checked_before_filtering()
    {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("positions.txt");
        std::fs::write(&path, "4k3/8/8/8/8/8/8/4K3 w - - 0 1;0;0.5\n").unwrap();

        let source = DataSource { path, format: Format::Text, separator: Some(";".into()) };
        let filter = Filter { minimum_eval: 100, ..Default::default() };

        let error =
            visit_boards_stoppable(&source, Some(&filter), &AtomicBool::new(true), |_| panic!("filtered callback"))
                .unwrap_err();
        assert!(error.is::<crate::cli::Cancelled>());
    }
}
