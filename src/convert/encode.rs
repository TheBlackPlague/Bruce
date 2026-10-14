use std::{borrow::Cow, io::Write};

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
            board::{Board, DrawType, GameOutcome, WinType, movegen::MoveList},
            chessmove::Move,
        },
        dataformat::Game,
    },
};

use super::Options;
use crate::data::{self, Format, Record};

pub(super) fn records(records: &[Record], options: &Options) -> Result<Vec<u8>>
{
    if options.to == Format::Sf {
        let mut output = sfbinpack::CompressedTrainingDataEntryWriter::new_in_memory()?;

        for record in records {
            output.write_entry(&sf_entry(record)?)?;
        }

        return Ok(output.into_bytes()?);
    }

    let mut bytes = Vec::new();
    for record in records {
        match options.to {
            Format::Bullet => bytes.extend_from_slice(ChessBoard::as_bytes_slice(&[record.board])),

            Format::Text => {
                let separator = options.separator.as_deref().context("Text conversion needs --separator")?;
                writeln!(
                    &mut bytes,
                    "{}{separator}{}{separator}{}",
                    fen(record),
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
                let mut castling = Castling::default();
                let startpos = Position::parse_fen(&data::viri_fen(&board), &mut castling);

                let best_move = find_monty_move(&startpos, &castling, &mov.display(true).to_string())?;

                let score =
                    record.white_score().try_into().context("The score cannot be represented in Monty format")?;

                MontyValueFormat {
                    startpos,
                    castling,
                    result: f32::from(record.white_result()) / 2.0,
                    moves: vec![SearchResult { best_move, score }],
                }
                .serialise_into(&mut bytes)?;
            },

            Format::Sf => unreachable!(),
        }
    }

    Ok(bytes)
}

pub(super) fn outcome(result: u8) -> GameOutcome
{
    match result {
        0 => GameOutcome::BlackWin(WinType::Adjudication),
        1 => GameOutcome::Draw(DrawType::Adjudication),
        _ => GameOutcome::WhiteWin(WinType::Adjudication),
    }
}

fn fen(record: &Record) -> Cow<'_, str>
{
    record.fen.as_deref().map(Cow::Borrowed).unwrap_or_else(|| Cow::Owned(data::bullet_fen(&record.board)))
}

fn position_move(record: &Record) -> Result<(Board, Move)>
{
    let board = data::parse_board(&fen(record)).context("Could not convert the position")?;

    if let Some(uci) = &record.next_move {
        let mov = board.parse_uci(uci).context("Could not convert the played move")?;
        ensure!(board.clone().make_move_simple(mov), "The dataset contains an illegal move");

        return Ok((board, mov));
    }

    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);

    for &mov in moves.iter_moves() {
        if board.clone().make_move_simple(mov) {
            return Ok((board, mov));
        }
    }

    bail!("A terminal position has no move; use Bullet or text output")
}

fn sf_entry(record: &Record) -> Result<TrainingDataEntry>
{
    let (board, mov) = position_move(record)?;

    let mut fields = data::viri_fen(&board).split_whitespace().map(str::to_owned).collect::<Vec<_>>();

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

    let kind = if mov.is_castle() {
        MoveType::Castle
    } else if mov.is_ep() {
        MoveType::EnPassant
    } else if mov.is_promo() {
        MoveType::Promotion
    } else {
        MoveType::Normal
    };

    let promotion = match mov.promotion_type() {
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

    let mv = SfMove::new(
        Square::new(u32::from(mov.from().inner())),
        Square::new(u32::from(mov.to().inner())),
        kind,
        promotion,
    );

    Ok(TrainingDataEntry {
        pos,
        mv,
        score: record.board.score,
        ply: record.ply.unwrap_or(0),
        result: i16::from(record.board.result) - 1,
    })
}

pub(super) fn find_monty_move(
    position: &Position,
    castling: &Castling,
    uci: &str,
) -> Result<bullet_lib::game::formats::montyformat::chess::Move>
{
    let mut found = None;
    position.map_legal_moves(castling, |mov| {
        if data::monty_uci(mov, castling) == uci {
            found = Some(mov);
        }
    });

    found.context("Could not convert the played move")
}
