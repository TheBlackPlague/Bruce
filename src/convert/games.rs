use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::atomic::AtomicBool,
};

use anyhow::{Result, ensure};
use bullet_lib::game::formats::{
    montyformat::{
        MontyValueFormat, SearchResult,
        chess::{Castling, Position},
    },
    viriformat::dataformat::Game,
};

use super::{
    Options, check_stop,
    encode::{find_monty_move, outcome},
    progress, write_parallel,
};
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
) -> Result<u64>
{
    let reader = BufReader::new(File::open(&options.input)?);
    match options.from {
        Format::Monty => transcode(
            reader,
            threads,
            writer,
            reporter,
            stop,
            |reader| {
                let game = MontyValueFormat::deserialise_from(reader, Vec::new())?;
                let count = game.moves.len();
                Ok((game, count))
            },
            monty_to_viri,
        ),

        Format::Viri => transcode(
            reader,
            threads,
            writer,
            reporter,
            stop,
            |reader| {
                let game = Game::deserialise_from(reader, Vec::new())?;
                let count = game.moves.len();
                Ok((game, count))
            },
            viri_to_monty,
        ),

        _ => unreachable!(),
    }
}

fn transcode<T: Sync>(
    mut reader: BufReader<File>,
    threads: usize,
    writer: &mut impl Write,
    reporter: &Reporter,
    stop: &AtomicBool,
    decode: impl Fn(&mut BufReader<File>) -> Result<(T, usize)>,
    encode: impl Fn(&[T]) -> Result<Vec<u8>> + Sync,
) -> Result<u64>
{
    let capacity = threads * 64;
    let mut batch = Vec::with_capacity(capacity);

    let mut positions = 0;
    while !reader.fill_buf()?.is_empty() {
        check_stop(stop)?;

        let (game, count) = decode(&mut reader)?;
        positions += count as u64;

        batch.push(game);
        if batch.len() == capacity {
            write_parallel(&batch, threads, &encode, writer)?;
            batch.clear();

            progress(reporter, positions, "positions converted");
        }
    }

    check_stop(stop)?;

    write_parallel(&batch, threads, encode, writer)?;
    progress(reporter, positions, "positions converted");

    ensure!(positions > 0, "No positions could be converted");

    Ok(positions)
}

fn monty_to_viri(games: &[MontyValueFormat]) -> Result<Vec<u8>>
{
    let mut bytes = Vec::new();
    for source in games {
        let mut position = data::parse_board(&data::monty_fen(&source.startpos, &source.castling))?;

        let mut game = Game::new(&position);

        ensure!([0.0, 0.5, 1.0].contains(&source.result), "Invalid result in the dataset");
        game.set_outcome(outcome((source.result * 2.0) as u8));

        for entry in &source.moves {
            let mov = position.parse_uci(&data::monty_uci(entry.best_move, &source.castling))?;

            game.add_move(mov, entry.score);
            ensure!(position.make_move_simple(mov), "Invalid move in a Monty game");
        }

        game.serialise_into(&mut bytes)?;
    }

    Ok(bytes)
}

fn viri_to_monty(games: &[Game]) -> Result<Vec<u8>>
{
    let mut bytes = Vec::new();
    for source in games {
        let mut castling = Castling::default();
        let startpos = Position::parse_fen(&data::viri_fen(&source.initial_position()), &mut castling);

        let mut position = startpos;

        let mut moves = Vec::with_capacity(source.moves.len());
        for &(mov, score) in &source.moves {
            let best_move = find_monty_move(&position, &castling, &mov.display(true).to_string())?;
            moves.push(SearchResult { best_move, score: score.get() });
            position.make(best_move, &castling);
        }

        MontyValueFormat { startpos, castling, result: f32::from(data::viri_result(source.outcome())) / 2.0, moves }
            .serialise_into(&mut bytes)?;
    }

    Ok(bytes)
}
