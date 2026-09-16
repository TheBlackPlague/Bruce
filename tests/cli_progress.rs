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
    let scores = [125, 32002, 32002, -200, -32002, i16::MIN, i16::MAX];
    {
        let mut file = fs::File::create(&input).unwrap();
        for side in ["w", "b"] {
            for score in scores {
                let mut chunk = Vec::new();
                let mut writer = CompressedTrainingDataEntryWriter::new(&mut chunk).unwrap();

                let (from, to) = if side == "w" { (12, 28) } else { (52, 36) };

                writer.write_entry(&TrainingDataEntry {
                    pos: Position::from_fen(&format!(
                        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {side} KQkq - 0 1"
                    )).unwrap(),
                    mv: Move::new(
                        Square::new(from),
                        Square::new( to ),
                        MoveType::Normal,
                        Piece::none(),
                    ),
                    score: if score == i16::MIN { 0 } else { score },
                    ply: u16::from(side == "b"),
                    result: -1,
                }).unwrap();

                drop(writer);

                if score == i16::MIN {
                    chunk[34..36].copy_from_slice(&u16::MAX.to_be_bytes());
                }

                std::io::Write::write_all(&mut file, &chunk).unwrap();
            }
        }
    }

    for to in ["bullet", "viri"] {
        let mut serial = Vec::new();

        for threads in ["1", "3"] {
            let output = directory.path().join(format!("{to}-{threads}"));
            let result = Command::new(env!("CARGO_BIN_EXE_bruce"))
                .current_dir(directory.path())
                .args([
                    "convert",
                    "--from",
                    "sf",
                    "--to",
                    to,
                    "--threads",
                    threads,
                    "--input",
                ])
                .arg(&input)
                .arg("--output")
                .arg(&output)
                .output()
                .unwrap();

            let text = String::from_utf8(result.stderr).unwrap();
            assert!(result.status.success(), "{text}");

            let summary = if to == "bullet" {
                "10 pos · skipped 4"
            } else {
                "9 pos · skipped 5"
            };
            assert!(text.contains(summary), "{text}");

            let bytes = fs::read(&output).unwrap();
            if threads == "1" {
                serial = bytes;
            } else {
                assert_eq!(bytes, serial);
            }

            if to == "bullet" {
                let mut actual = Vec::new();
                DataLoader::<ChessBoard>::new(output, 1).unwrap().map_batches(10, |batch| {
                    actual.extend(batch.iter().map(|b| (b.score(), b.result_idx())));
                });

                let expected: Vec<_> = scores
                    .into_iter()
                    .filter(|s| *s != 32002)
                    .map(|s| (s, 0))
                    .cycle()
                    .take(10)
                    .collect();
                assert_eq!(actual, expected);
            } else {
                use bullet_lib::game::formats::viriformat::dataformat::Game;

                let mut reader = std::io::BufReader::new(fs::File::open(output).unwrap());

                for black in [false, true] {
                    for score in scores {
                        if score == 32002 || (black && score == i16::MIN) {
                            continue;
                        }

                        let game = Game::deserialise_from(&mut reader, Vec::new()).unwrap();
                        assert_eq!(game.moves[0].1.get(), if black { -score } else { score });
                        assert_eq!(game.initial_position().turn().inner(), u8::from(black));
                    }
                }
            }
        }
    }
}
