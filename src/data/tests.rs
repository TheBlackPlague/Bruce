use super::*;
use bullet_lib::game::formats::bulletformat::{BulletFormat, ChessBoard};
use bullet_trainer::reader::DataReader;
use std::{
    fs::File,
    io::{BufWriter, Write},
};

fn config(path: std::path::PathBuf, format: DataFormat) -> DataConfig {
    DataConfig {
        format,
        paths: vec![path],
        buffer_size_mb: 1,
        loader_threads: 1,
        mapping_threads: 1,
        filter: None,
    }
}

#[test]
fn native_reader_resumes_at_the_exact_record_across_files() {
    let directory = tempfile::tempdir().unwrap();
    let mut records = Vec::new();

    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";

    for score in 1..=6 {
        records.push(
            format!("{fen} | {score} | 0.5")
                .parse::<ChessBoard>()
                .unwrap(),
        );
    }

    let paths = [
        directory.path().join("first"),
        directory.path().join("second"),
    ];

    for (path, chunk) in paths.iter().zip(records.chunks(3)) {
        let mut writer = BufWriter::new(File::create(path).unwrap());
        ChessBoard::write_to_bin(&mut writer, chunk).unwrap();
        writer.flush().unwrap();
    }

    let mut config = config(paths[0].clone(), DataFormat::Bullet);
    config.paths.push(paths[1].clone());

    let reader = reader(&config).unwrap();
    let mut resumed = Vec::new();
    reader.read_chunks(4, |chunk| {
        resumed.extend_from_slice(chunk);
        resumed.len() >= 8
    });

    let expected = records
        .iter()
        .cycle()
        .skip(4)
        .take(8)
        .copied()
        .collect::<Vec<_>>();

    assert_eq!(&resumed[..8], &expected);
}

#[test]
fn zeroed_and_truncated_bullet_files_fail_preflight() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad");

    for length in [31, 32] {
        fs::write(&path, vec![0; length]).unwrap();

        assert!(
            validate(
                &config(path.clone(), DataFormat::Bullet),
                &Reporter::silent()
            )
            .is_err()
        );
    }
}

#[test]
fn filter_fullmove_boundary_matches_the_reference_run() {
    use bullet_lib::game::formats::montyformat::chess::{Castling, Move, Position};

    let filter = FilterConfig::default();
    let mut castling = Castling::default();

    let before = Position::parse_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 7", &mut castling);
    let at     = Position::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 8", &mut castling);

    assert!(!filter.monty(&before, Move::new(60, 59, 0), 0));
    assert!( filter.monty(&at    , Move::new( 4,  3, 0), 0));
}
