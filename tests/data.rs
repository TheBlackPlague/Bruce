use std::fs;

use bruce::{
    config::{DataSource, Filter},
    convert::{self, Options},
    data::{self, Format},
};
use tempfile::TempDir;

const WHITE: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const BLACK: &str = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";

fn source(path: &std::path::Path, format: Format) -> DataSource
{
    DataSource { path: path.into(), format, separator: (format == Format::Text).then(|| ";".into()) }
}

fn convert(input: &std::path::Path, from: Format, output: &std::path::Path, to: Format) -> anyhow::Result<()>
{
    convert::run(
        &Options {
            input: input.into(),
            output: output.into(),
            from,
            to,
            separator: Some(";".into()),
            threads: Some(2),
        },
        true,
    )
    .map(|_| ())
}

fn collect(path: &std::path::Path, format: Format) -> Vec<(i16, u8, u64)>
{
    let mut values = Vec::new();
    data::visit(&source(path, format), None, |record| {
        values.push((record.board.score, record.board.result, record.board.occ));
        Ok(false)
    })
    .unwrap();
    values
}

#[test]
fn all_output_formats_preserve_positions_scores_and_results()
{
    let directory = TempDir::new().unwrap();
    let text = directory.path().join("input.txt");
    fs::write(&text, format!("{WHITE};120;1\n{BLACK};-250;0\n")).unwrap();

    let expected = collect(&text, Format::Text);
    assert_eq!(expected.iter().map(|entry| (entry.0, entry.1)).collect::<Vec<_>>(), [(120, 2), (250, 2)]);

    for format in [Format::Bullet, Format::Viri, Format::Monty, Format::Sf] {
        let output = directory.path().join(format!("{format:?}"));
        convert(&text, Format::Text, &output, format).unwrap();
        assert_eq!(collect(&output, format), expected, "{format:?}");

        let roundtrip = directory.path().join(format!("{format:?}.txt"));
        convert(&output, format, &roundtrip, Format::Text).unwrap();
        assert_eq!(collect(&roundtrip, Format::Text), expected, "{format:?} text roundtrip");
    }
}

#[test]
fn game_format_conversion_preserves_moves_and_metadata()
{
    use bullet_lib::game::formats::montyformat::{
        MontyValueFormat, SearchResult,
        chess::{Castling, Position},
    };

    let directory = TempDir::new().unwrap();
    let mut castling = Castling::default();
    let startpos = Position::parse_fen(WHITE, &mut castling);
    let mut position = startpos;
    let mut moves = Vec::new();
    for uci in ["e2e4", "e7e5", "g1f3"] {
        let mut found = None;
        position.map_legal_moves(&castling, |mov| {
            if data::monty_uci(mov, &castling) == uci {
                found = Some(mov)
            }
        });
        let mov = found.unwrap();
        moves.push(SearchResult { best_move: mov, score: 135 });
        position.make(mov, &castling);
    }

    let path = directory.path().join("game.monty");
    MontyValueFormat { startpos, castling, moves, result: 0.5 }
        .serialise_into(&mut fs::File::create(&path).unwrap())
        .unwrap();

    let viri = directory.path().join("game.viri");
    convert(&path, Format::Monty, &viri, Format::Viri).unwrap();
    assert_eq!(collect(&viri, Format::Viri), collect(&path, Format::Monty));

    let restored = directory.path().join("restored.monty");
    convert(&viri, Format::Viri, &restored, Format::Monty).unwrap();
    assert_eq!(fs::read(&restored).unwrap(), fs::read(&path).unwrap());
}

#[test]
fn filters_keep_negative_evaluations_and_enforce_ply()
{
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("input.txt");
    fs::write(&path, format!("{WHITE};-120;0.5\n{BLACK};-250;0.5\n")).unwrap();

    let filter = Filter { minimum_eval: 100, maximum_eval: 200, ..Default::default() };
    let mut values = Vec::new();

    data::visit(&source(&path, Format::Text), Some(&filter), |record| {
        values.push(record.board.score);
        Ok(false)
    })
    .unwrap();
    assert_eq!(values, [-120]);
}

#[test]
fn output_is_not_created_on_failure_or_overwritten()
{
    let directory = TempDir::new().unwrap();
    let input = directory.path().join("input.txt");
    let output = directory.path().join("output.bullet");

    fs::write(&input, "bad input").unwrap();
    assert!(convert(&input, Format::Text, &output, Format::Bullet).is_err());
    assert!(!output.exists());

    fs::write(&output, "keep").unwrap();
    assert!(convert(&input, Format::Text, &output, Format::Bullet).is_err());
    assert_eq!(fs::read_to_string(&output).unwrap(), "keep");
}

#[test]
fn truncated_records_and_missing_text_separator_are_rejected()
{
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("truncated.bullet");
    fs::write(&path, [0; 31]).unwrap();
    assert!(data::visit(&source(&path, Format::Bullet), None, |_| Ok(false)).is_err());

    let path = directory.path().join("data.txt");
    fs::write(&path, format!("{WHITE};0;0.5")).unwrap();
    assert!(data::visit(&DataSource { format: Format::Text, path, separator: None }, None, |_| Ok(false)).is_err());
}

#[test]
fn cancellation_stops_on_a_complete_superbatch()
{
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use bruce::config::DataConfig;
    use bullet_trainer::reader::DataReader;

    let directory = TempDir::new().unwrap();
    let path = directory.path().join("data.txt");
    fs::write(&path, format!("{WHITE};0;0.5\n").repeat(6105)).unwrap();

    let config = DataConfig {
        buffer_size_mb: 1,
        loader_threads: 2,
        mapping_threads: 1,
        batch_size: 1,
        sources: vec![source(&path, Format::Text)],
        filter: None,
    };
    let stop = Arc::new(AtomicBool::new(false));
    let reader = data::DatasetReader::new(&config, 42, stop.clone()).unwrap();

    let mut count = 0;
    reader.read_chunks(0, |boards| {
        count += boards.len();
        stop.store(true, Ordering::Relaxed);

        false
    });
    assert_eq!(count, 6104);
    assert_eq!(reader.error(), None);

    fs::remove_file(&path).unwrap();

    reader.read_chunks(0, |_| panic!("Stopped reader emitted a batch"));
    assert_eq!(reader.error(), None);
}

#[test]
fn sf_sentinel_filter_retains_representable_extreme_scores()
{
    assert!(data::valid_sf_score(i16::MAX));
    assert!(data::valid_sf_score(-32002));
    assert!(!data::valid_sf_score(32002));
    assert!(!data::valid_sf_score(i16::MIN));
}

#[test]
fn game_conversion_preserves_castling_en_passant_and_promotion()
{
    use bullet_lib::game::formats::montyformat::{
        MontyValueFormat, SearchResult,
        chess::{Castling, Position},
    };

    let directory = TempDir::new().unwrap();
    for (index, (fen, uci)) in [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 15", "e1h1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 15", "e8a8"),
        ("4k3/8/8/8/8/8/8/2R1K1R1 w CG - 0 15", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 15", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 15", "a7a8q"),
    ]
    .into_iter()
    .enumerate()
    {
        let mut castling = Castling::default();
        let startpos = Position::parse_fen(fen, &mut castling);
        let mut found = None;
        startpos.map_legal_moves(&castling, |mov| {
            if data::monty_uci(mov, &castling) == uci {
                found = Some(mov)
            }
        });

        let game = MontyValueFormat {
            startpos,
            castling,
            moves: vec![SearchResult { best_move: found.expect(uci), score: -100 }],
            result: 0.0,
        };

        let original = directory.path().join(format!("{index}.monty"));
        game.serialise_into(&mut fs::File::create(&original).unwrap()).unwrap();

        let viri = directory.path().join(format!("{index}.viri"));
        convert(&original, Format::Monty, &viri, Format::Viri).unwrap();

        let restored = directory.path().join(format!("{index}.roundtrip"));
        convert(&viri, Format::Viri, &restored, Format::Monty).unwrap();

        assert_eq!(fs::read(&restored).unwrap(), fs::read(&original).unwrap(), "{fen} {uci}");

        if index != 2 {
            let sf = directory.path().join(format!("{index}.sf"));
            convert(&original, Format::Monty, &sf, Format::Sf).unwrap();

            let viri_sf = directory.path().join(format!("{index}.sf.viri"));
            convert(&sf, Format::Sf, &viri_sf, Format::Viri).unwrap();

            let mut actual_move = None;
            data::visit(&source(&viri_sf, Format::Viri), None, |record| {
                actual_move = record.next_move;
                Ok(true)
            })
            .unwrap();

            assert_eq!(actual_move.as_deref(), Some(uci));
            assert_eq!(collect(&viri_sf, Format::Viri), collect(&original, Format::Monty));
        }
    }
}

#[test]
fn sf_conversion_preserves_minimum_score_without_overflow()
{
    use bullet_lib::game::formats::sfbinpack::{
        CompressedTrainingDataEntryWriter, TrainingDataEntry,
        chess::{coords::Square, r#move::Move, position::Position},
    };

    let directory = TempDir::new().unwrap();
    let input = directory.path().join("minimum.sf");

    let mut writer = CompressedTrainingDataEntryWriter::new_in_memory().unwrap();

    for (fen, mov) in [
        (WHITE, Move::normal(Square::new(12), Square::new(28))),
        (BLACK, Move::normal(Square::new(52), Square::new(36))),
    ] {
        writer
            .write_entry(&TrainingDataEntry {
                pos: Position::from_fen(fen).unwrap(),
                mv: mov,
                score: i16::MIN,
                ply: 1,
                result: 1,
            })
            .unwrap();
    }
    fs::write(&input, writer.into_bytes().unwrap()).unwrap();

    let output = directory.path().join("minimum.bullet");
    let summary = convert::run(
        &Options {
            input: input.clone(),
            output: output.clone(),
            from: Format::Sf,
            to: Format::Bullet,
            separator: None,
            threads: Some(1),
        },
        true,
    )
    .unwrap();
    assert_eq!((summary.positions, summary.skipped), (2, 0));
    assert!(collect(&output, Format::Bullet).iter().all(|entry| entry.0 == i16::MIN));

    let text = directory.path().join("minimum.txt");
    convert(&input, Format::Sf, &text, Format::Text).unwrap();
    assert_eq!(collect(&text, Format::Text), collect(&input, Format::Sf));

    let output = directory.path().join("minimum.viri");
    let summary = convert::run(
        &Options { input, output, from: Format::Sf, to: Format::Viri, separator: None, threads: Some(1) },
        true,
    )
    .unwrap();
    assert_eq!((summary.positions, summary.skipped), (1, 1));
}

#[test]
fn mixed_reader_interleaves_sources_and_cancels_without_deadlock()
{
    use std::sync::{Arc, atomic::AtomicBool};

    use bruce::config::DataConfig;
    use bullet_trainer::reader::DataReader;

    let directory = TempDir::new().unwrap();
    let text = directory.path().join("positive.txt");
    let other = directory.path().join("negative.txt");
    let binary = directory.path().join("negative.bullet");

    fs::write(&text, format!("{WHITE};100;0.5\n").repeat(5000)).unwrap();
    fs::write(&other, format!("{WHITE};-100;0.5\n").repeat(5000)).unwrap();
    convert(&other, Format::Text, &binary, Format::Bullet).unwrap();

    let config = DataConfig {
        buffer_size_mb: 1,
        loader_threads: 2,
        mapping_threads: 1,
        batch_size: 1,
        sources: vec![source(&text, Format::Text), source(&binary, Format::Bullet)],
        filter: None,
    };
    let reader = data::DatasetReader::new(&config, 42, Arc::new(AtomicBool::new(false))).unwrap();

    let mut scores = Vec::new();
    reader.read_chunks(0, |boards| {
        scores.extend(boards.iter().map(|board| board.score));
        scores.len() >= 10000
    });
    assert_eq!(scores.len(), 10000);
    assert_eq!(scores.iter().filter(|&&score| score == 100).count(), 5000);
    assert_eq!(scores.iter().filter(|&&score| score == -100).count(), 5000);
    assert_eq!(reader.error(), None);

    let mut repeated_scores = Vec::new();
    reader.read_chunks(0, |boards| {
        repeated_scores.extend(boards.iter().map(|board| board.score));
        repeated_scores.len() >= 10000
    });
    assert_eq!(repeated_scores, scores, "Worker timing must not change the seeded source order");
    assert_eq!(reader.error(), None);

    let mut first = None;
    reader.read_chunks(9999, |boards| {
        first = boards.first().map(|board| board.score);

        true
    });

    assert!(first.is_some());
    assert_eq!(reader.error(), None);
}
