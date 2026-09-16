use std::{fs, process::Command};

fn convert(plain: bool, fail: bool) -> (tempfile::TempDir, std::process::Output) {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.txt");

    fs::write(
        &input,
        if fail {
            "invalid record\n"
        } else {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5\n"
        },
    ).unwrap();

    let mut command = Command::new(env!("CARGO_BIN_EXE_bruce"));

    command
        .current_dir(directory.path())
        .env_remove("CI")
        .env("TERM", "xterm-256color");

    if plain {
        command.arg("--plain");
    }

    let output = command
        .args(["convert", "--from", "text", "--to", "bullet", "--input"])
        .arg(input)
        .arg("--output")
        .arg(directory.path().join("output.bullet"))
        .output()
        .unwrap();

    (directory, output)
}

#[test]
fn redirected_and_explicit_plain_preserve_summary_and_no_ansi() {
    for plain in [false, true] {
        let (directory, output) = convert(plain, false);
        let text = String::from_utf8(output.stderr).unwrap();

        assert!(output.status.success(), "{text}");
        assert!(text.contains("\n\nConverting Text → Bullet"), "{text}");
        assert!(text.contains("output.bullet\n\n"), "{text}");
        assert!(text.contains("1 pos · skipped 0"), "{text}");
        assert!(
            text.contains("100.0%") && text.contains("Complete."),
            "{text}"
        );
        assert!(!text.contains('\x1b') && !text.contains('\r'));
        assert_eq!(
            fs::metadata(directory.path().join("output.bullet")).unwrap().len(),
            32
        );

        for file in fs::read_dir(directory.path().join("logs")).unwrap() {
            let log = fs::read_to_string(file.unwrap().path()).unwrap();

            assert!(!log.contains('\x1b'));
        }
    }
}

#[test]
fn failed_conversion_is_not_reported_as_complete_or_published() {
    let (directory, output) = convert(false, true);
    let text = String::from_utf8(output.stderr).unwrap();

    assert!(!output.status.success());
    assert!(text.contains("Stopped before completion."), "{text}");
    assert!(!text.contains("Complete."));
    assert!(!directory.path().join("output.bullet").exists());
    assert!(!text.contains('\x1b'));
}

#[test]
fn stockfish_conversion_reports_skips_and_preserves_usable_scores() {
    use bullet_lib::game::formats::{
        bulletformat::{BulletFormat, ChessBoard, DataLoader},
        sfbinpack::{
            CompressedTrainingDataEntryWriter, TrainingDataEntry,
            chess::{
                coords::Square,
                r#move::{Move, MoveType},
                piece::Piece,
                position::Position,
            },
        },
    };

    let directory = tempfile::tempdir().unwrap();

    let input = directory.path().join("input.binpack");
    {
        let file = fs::File::create(&input).unwrap();
        let mut writer = CompressedTrainingDataEntryWriter::new(file).unwrap();

        for score in [125, 32002, 32002, -200] {
            writer.write_entry(&TrainingDataEntry {
                pos: Position::from_fen(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                ).unwrap(),
                mv: Move::new(
                    Square::new(12),
                    Square::new(28),
                    MoveType::Normal,
                    Piece::none(),
                ),
                score,
                ply: 0,
                result: 0,
            }).unwrap();
        }
    }

    for to in ["bullet", "viri"] {
        let output = directory.path().join(to);
        let result = Command::new(env!("CARGO_BIN_EXE_bruce"))
            .current_dir(directory.path())
            .args(["convert", "--from", "sf", "--to", to, "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap();

        let text = String::from_utf8(result.stderr).unwrap();
        assert!(result.status.success(), "{text}");
        assert!(text.contains("2 pos · skipped 2"), "{text}");

        if to == "bullet" {
            let mut scores = Vec::new();

            DataLoader::<ChessBoard>::new(output, 1).unwrap().map_batches(10, |batch| {
                scores.extend(batch.iter().map(|b| b.score()));
            });

            assert_eq!(scores, [125, -200]);
        }
    }
}
