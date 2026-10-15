mod loader;
mod read;

use std::sync::atomic::Ordering;

use anyhow::{Context, Result, ensure};
use bullet_lib::game::formats::{
    bulletformat::ChessBoard,
    montyformat::chess::{Castling, Move, Position},
    sfbinpack::{
        self,
        chess::{color::Color, piecetype::PieceType},
    },
    viriformat::{chess::board::Board, dataformat::WDL},
};
use clap::ValueEnum;
pub use loader::DatasetReader;
pub use read::{visit, visit_boards, visit_boards_stoppable, visit_counted, visit_counted_stoppable};
use serde::{Deserialize, Serialize};

use crate::config::{DataConfig, DataSource, Filter};

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
    fn black_to_move(&self) -> bool
    {
        self.fen.as_deref().is_some_and(|fen| fen.split_whitespace().nth(1) == Some("b"))
    }

    pub fn white_score(&self) -> i32
    {
        let score = i32::from(self.board.score);
        if self.black_to_move() { -score } else { score }
    }

    pub fn white_result(&self) -> u8 { if self.black_to_move() { 2 - self.board.result } else { self.board.result } }
}

fn accepts_position(filter: &Filter, score: i16, pieces: u32, ply: Option<u16>) -> bool
{
    let magnitude = u32::from(score.unsigned_abs());
    magnitude >= filter.minimum_eval
        && magnitude < filter.maximum_eval
        && pieces >= filter.minimum_pieces
        && ply.is_none_or(|ply| ply >= filter.minimum_ply && filter.maximum_ply.is_none_or(|last| ply <= last))
}

fn accepts(record: &Record, filter: Option<&Filter>) -> bool
{
    filter.is_none_or(|filter| {
        accepts_position(filter, record.board.score, record.board.occ().count_ones(), record.ply)
            && (!filter.exclude_check || record.in_check == Some(false))
            && (!filter.exclude_tactical || record.tactical == Some(false))
    })
}

pub fn validate_source(source: &DataSource, filter: Option<&Filter>) -> Result<()>
{
    let length = std::fs::metadata(&source.path).context("Could not open the dataset")?.len();
    ensure!(length > 0, "The dataset is empty");

    if source.format == Format::Bullet {
        ensure!(length.is_multiple_of(size_of::<ChessBoard>() as u64), "The dataset is truncated");

        if let Some(filter) = filter {
            ensure!(
                filter.minimum_ply == 0
                    && filter.maximum_ply.is_none()
                    && !filter.exclude_check
                    && !filter.exclude_tactical,
                "Bullet data does not contain move or ply information"
            );
        }
    }

    if source.format == Format::Text {
        let separator = source.separator.as_deref().context("Text data needs a separator")?;
        ensure!(!separator.is_empty() && !separator.contains(['\n', '\r']), "Choose a nonempty text separator");

        ensure!(filter.is_none_or(|filter| !filter.exclude_tactical), "Text data does not contain the played move");
    }

    Ok(())
}

pub fn validate(config: &DataConfig) -> Result<()>
{
    for source in &config.sources {
        let mut found = false;
        visit_boards(source, config.filter.as_ref(), |board| {
            found |= source.format != Format::Sf || valid_sf_score(board.score);
            Ok(false)
        })?;

        ensure!(found, "No positions pass the filter");
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
            let side = usize::from(piece >> 3);
            kings[side] += 1;

            let expected = if side == 0 { board.ksq } else { board.opp_ksq ^ 56 };
            ensure!(square == expected, "Invalid king square in the dataset");
        }
    }

    ensure!(kings == [1, 1], "A position must contain both kings");

    Ok(())
}

pub fn sf_board(entry: &sfbinpack::TrainingDataEntry) -> Result<ChessBoard>
{
    let mut bitboards = [0; 8];
    bitboards[0] = entry.pos.pieces_bb(Color::White).bits();
    bitboards[1] = entry.pos.pieces_bb(Color::Black).bits();

    for (index, piece) in
        [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King]
            .into_iter()
            .enumerate()
    {
        bitboards[index + 2] = entry.pos.pieces_bb_color(Color::White, piece).bits()
            | entry.pos.pieces_bb_color(Color::Black, piece).bits();
    }

    ensure!((-1..=1).contains(&entry.result), "Invalid result in the dataset");

    let mut board = ChessBoard::from_raw(bitboards, usize::from(entry.pos.side_to_move() == Color::Black), 0, 0.5)
        .map_err(|_| anyhow::anyhow!("Invalid SF position"))?;

    board.score = entry.score;
    board.result = (entry.result + 1) as u8;

    Ok(board)
}

fn sf_tactical(entry: &sfbinpack::TrainingDataEntry) -> bool
{
    use sfbinpack::chess::{r#move::MoveType, piece::Piece};

    matches!(entry.mv.mtype(), MoveType::EnPassant | MoveType::Promotion)
        || (entry.mv.mtype() != MoveType::Castle && entry.pos.piece_at(entry.mv.to()) != Piece::none())
}

pub fn parse_board(fen: &str) -> Result<Board>
{
    bullet_lib::game::formats::viriformat::chess::CHESS960.store(true, Ordering::Relaxed);

    let mut fields = fen.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
    ensure!(fields.len() == 6, "Invalid FEN in the dataset");

    fields[2] = fields[2]
        .chars()
        .map(|right| match right {
            'K' => 'H',
            'Q' => 'A',
            'k' => 'h',
            'q' => 'a',
            value => value,
        })
        .collect();
    ensure!(
        fields[2].chars().all(|right| right == '-' || ('A'..='H').contains(&right) || ('a'..='h').contains(&right)),
        "Invalid castling rights"
    );

    let mut board = Board::new();
    board.set_from_fen(&fields.join(" "))?;

    Ok(board)
}

pub fn viri_fen(board: &Board) -> String
{
    let mut fields = board.to_string().split_whitespace().map(str::to_owned).collect::<Vec<_>>();

    let rights = board.castling_rights();

    let mut encoded = String::new();
    for (square, base) in [(rights.wk, b'A'), (rights.wq, b'A'), (rights.bk, b'a'), (rights.bq, b'a')] {
        if let Some(square) = square {
            encoded.push(char::from(base + square.inner() % 8));
        }
    }

    fields[2] = if encoded.is_empty() { "-".into() } else { encoded };
    fields.join(" ")
}

pub fn viri_result(result: WDL) -> u8
{
    match result {
        WDL::Win => 2,
        WDL::Draw => 1,
        WDL::Loss => 0,
    }
}

fn square_name(square: u8) -> String { format!("{}{}", char::from(b'a' + square % 8), square / 8 + 1) }

pub fn monty_fen(position: &Position, castling: &Castling) -> String
{
    let mut fields = position.as_fen().split_whitespace().map(str::to_owned).collect::<Vec<_>>();

    let mut rights = String::new();
    for (mask, side, flank) in [(4, 0, 1), (8, 0, 0), (1, 1, 1), (2, 1, 0)] {
        if position.rights() & mask != 0 {
            rights.push(char::from((if side == 0 { b'A' } else { b'a' }) + castling.rook_file(side, flank) as u8));
        }
    }
    fields[2] = if rights.is_empty() { "-".into() } else { rights };

    fields[3] = if position.enp_sq() == 0 { "-".into() } else { square_name(position.enp_sq()) };
    fields.join(" ")
}

pub fn monty_uci(mov: Move, castling: &Castling) -> String
{
    let destination = if matches!(mov.flag(), 2 | 3) {
        let side = usize::from(mov.src() >= 56);
        (56 * side + castling.rook_file(side, usize::from(mov.flag() == 2)) as usize) as u8
    } else {
        mov.to() as u8
    };

    let promotion = if mov.is_promo() { ["n", "b", "r", "q"][(mov.flag() & 3) as usize] } else { "" };
    format!("{}{}{promotion}", square_name(mov.src() as u8), square_name(destination))
}

pub fn bullet_fen(board: &ChessBoard) -> String
{
    let mut squares = [b' '; 64];
    for (piece, square) in *board {
        squares[usize::from(square)] = b"PNBRQKpnbrqk"[usize::from((piece & 7) + 6 * (piece >> 3))];
    }

    let mut fen = String::with_capacity(90);
    for rank in (0..8).rev() {
        if rank < 7 {
            fen.push('/');
        }

        let mut empty = 0;
        for &piece in &squares[rank * 8..rank * 8 + 8] {
            if piece == b' ' {
                empty += 1;
                continue;
            }

            if empty > 0 {
                fen.push(char::from(b'0' + empty));
                empty = 0;
            }

            fen.push(char::from(piece));
        }

        if empty > 0 {
            fen.push(char::from(b'0' + empty));
        }
    }

    fen.push_str(" w - - 0 1");
    fen
}
