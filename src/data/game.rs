use std::mem;

use bullet_lib::game::formats::{
    bulletformat::ChessBoard,
    montyformat::{
        MontyValueFormat, SearchResult,
        chess::{Castling, Position},
    },
};

use super::DataError;

const HEADER: usize = 43;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Outcome
{
    WhiteWin = 2,
    #[default]
    Draw = 1,
    BlackWin = 0,
}

impl Outcome
{
    pub fn relative(self, black: bool) -> u8 { if black { 2 - self as u8 } else { self as u8 } }

    pub fn from_white(result: u8) -> Option<Self>
    {
        [Self::BlackWin, Self::Draw, Self::WhiteWin].get(usize::from(result)).copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ply
{
    pub score: i16,
    pub pieces: u8,
    pub tactical: bool,
}

#[derive(Default)]
pub struct Game
{
    pub source: usize,
    pub index: u64,
    pub offset: u64,
    pub outcome: Outcome,
    pub first_ply: u16,
    pub white_moves_first: bool,
    pub plies: Vec<Ply>,
    start: Position,
    castling: Castling,
    moves: Vec<SearchResult>,
    pub saturated: u32,
    pub capped: u32,
}

pub fn record_len(bytes: &[u8]) -> Option<usize>
{
    let body = bytes.get(HEADER..)?;
    let end = body.as_chunks::<4>().0.iter().position(|record| *record == [0; 4])?;
    Some(HEADER + (end + 1) * 4)
}

impl Game
{
    pub fn read(&mut self, bytes: &mut &[u8], scale: f64) -> Result<(), DataError>
    {
        let MontyValueFormat { startpos, castling, result, moves } =
            MontyValueFormat::deserialise_from(bytes, mem::take(&mut self.moves))
                .map_err(|_| DataError::truncated())?;
        self.outcome = Outcome::from_white((result * 2.0) as u8)
            .ok_or_else(|| DataError::new(format!("Invalid data: unknown game result {result}")))?;
        self.first_ply = startpos.fullm().saturating_sub(1).saturating_mul(2).saturating_add(startpos.stm() as u16);
        self.white_moves_first = startpos.stm() == 0;
        (self.saturated, self.capped) = (0, 0);
        self.plies.clear();

        let mut pieces = startpos.occ().count_ones();
        for entry in &moves {
            let (score, saturated) = clamp(i32::from(entry.score));
            let (score, capped) = scaled(score, scale);
            self.saturated += u32::from(saturated);
            self.capped += u32::from(capped);

            let tactical = entry.best_move.is_capture() || entry.best_move.is_promo();
            self.plies.push(Ply { score, pieces: pieces as u8, tactical });
            pieces = pieces.saturating_sub(u32::from(entry.best_move.is_capture()));
        }

        (self.start, self.castling, self.moves) = (startpos, castling, moves);

        Ok(())
    }

    pub fn white_to_move(&self, ply: usize) -> bool { self.white_moves_first == ply.is_multiple_of(2) }

    pub fn win_probability(&self, ply: usize) -> f32 { 1.0 / (1.0 + (-f32::from(self.plies[ply].score) / 400.0).exp()) }

    pub fn fen_ply(&self, ply: usize) -> u16 { self.first_ply.saturating_add(narrow(ply)) }

    pub fn board(&self, ply: usize, position: &Position) -> Result<ChessBoard, DataError>
    {
        let black = position.stm() == 1;
        let mut board = ChessBoard::from_raw(position.bbs(), usize::from(black), 0, 0.5).map_err(invalid_position)?;
        board.score = relative(self.plies[ply].score, black);
        board.result = self.outcome.relative(black);

        Ok(board)
    }

    pub fn walk(&self, mut visit: impl FnMut(usize, &Position) -> Result<(), DataError>) -> Result<(), DataError>
    {
        let mut position = self.start;
        let walked = self.moves.iter().enumerate().try_for_each(|(ply, entry)| -> Result<(), DataError> {
            validate(&position.bbs()).map_err(|reason| invalid_position(format!("{reason} at ply {ply}")))?;
            visit(ply, &position)?;
            position.make(entry.best_move, &self.castling);
            Ok(())
        });

        walked.map_err(|error| error.offset(self.offset).game(self.index))
    }
}

pub fn narrow(ply: usize) -> u16 { u16::try_from(ply).unwrap_or(u16::MAX) }

pub fn relative(score: i16, black: bool) -> i16 { if black { score.saturating_neg() } else { score } }

fn clamp(score: i32) -> (i16, bool)
{
    let limit = i32::from(i16::MAX);
    let clamped = score.clamp(-limit, limit);
    (clamped as i16, clamped != score)
}

fn scaled(score: i16, scale: f64) -> (i16, bool)
{
    if score.unsigned_abs() == i16::MAX.unsigned_abs() {
        return (score, false);
    }

    clamp((scale * f64::from(score)).round() as i32)
}

fn validate(bitboards: &[u64; 8]) -> Result<(), String>
{
    let [white, black, .., kings] = *bitboards;

    let pieces = (white | black).count_ones();
    if pieces > 32 {
        return Err(format!("{pieces} pieces; at most 32 are supported"));
    }

    if (kings & white).count_ones() != 1 || (kings & black).count_ones() != 1 {
        return Err("each side needs exactly one king".into());
    }

    Ok(())
}

fn invalid_position(reason: String) -> DataError { DataError::new(format!("Invalid position: {reason}")) }
