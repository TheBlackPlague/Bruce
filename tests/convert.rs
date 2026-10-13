use std::fs;

use bruce::{
    convert::{self, Options},
    data::Format,
};
use bullet_lib::game::formats::montyformat::{
    MontyValueFormat, SearchResult,
    chess::{Castling, Position},
};

#[test]
fn parallel_game_conversion_preserves_order_across_batches_and_tail()
{
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.monty");
    let mut castling = Castling::default();
    let startpos = Position::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &mut castling);
    let mut mov = None;

    startpos.map_legal_moves(&castling, |candidate| {
        if bruce::data::monty_uci(candidate, &castling) == "e2e4" {
            mov = Some(candidate);
        }
    });

    let mut file = fs::File::create(&input).unwrap();

    for score in -65..65 {
        MontyValueFormat {
            startpos,
            castling,
            result: 0.5,
            moves: vec![SearchResult { best_move: mov.unwrap(), score }],
        }
        .serialise_into(&mut file)
        .unwrap();
    }

    drop(file);

    let mut options = Options {
        from: Format::Monty,
        to: Format::Viri,
        input: input.clone(),
        output: directory.path().join("single.viri"),
        separator: None,
        threads: Some(1),
    };

    let single = convert::run(&options, true).unwrap();
    let expected = fs::read(&options.output).unwrap();

    options.threads = Some(2);
    options.output = directory.path().join("parallel.viri");

    let parallel = convert::run(&options, true).unwrap();
    assert_eq!((single.positions, parallel.positions), (130, 130));
    assert_eq!(fs::read(&options.output).unwrap(), expected);

    options.from = Format::Viri;
    options.to = Format::Monty;
    options.input = options.output;
    options.output = directory.path().join("restored.monty");
    assert_eq!(convert::run(&options, true).unwrap().positions, 130);
    assert_eq!(fs::read(&options.output).unwrap(), fs::read(input).unwrap());
}

#[test]
fn cancelled_conversion_preserves_existing_files_and_never_publishes_output()
{
    use std::sync::{Arc, atomic::AtomicBool};

    use bruce::cli::{Cancelled, OutputOptions, Reporter};

    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.txt");
    let output = directory.path().join("output.bullet");
    let content = "4k3/8/8/8/8/8/8/4K3 w - - 0 1;0;0.5\n";
    fs::write(&input, content).unwrap();

    let options = Options {
        from: Format::Text,
        to: Format::Bullet,
        input: input.clone(),
        output: output.clone(),
        separator: Some(";".into()),
        threads: None,
    };
    let reporter = Reporter::new(OutputOptions::default());
    let error = convert::run_reported(&options, &reporter, Arc::new(AtomicBool::new(true))).unwrap_err();

    assert!(error.is::<Cancelled>());
    assert!(!output.exists());
    assert_eq!(fs::read_to_string(input).unwrap(), content);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn same_format_conversion_checks_input_and_preserves_every_byte()
{
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.txt");
    let output = directory.path().join("copy.txt");
    let content = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 ; 0 ; 0.5\n";
    fs::write(&input, content).unwrap();

    let options = Options {
        from: Format::Text,
        to: Format::Text,
        input: input.clone(),
        output: output.clone(),
        separator: Some(";".into()),
        threads: None,
    };

    let summary = convert::run(&options, true).unwrap();
    assert_eq!(summary.positions, 1);
    assert_eq!(summary.input_bytes, summary.output_bytes);
    assert_eq!(fs::read_to_string(&output).unwrap(), content);

    fs::remove_file(&output).unwrap();
    fs::write(&input, format!("{content}broken\n")).unwrap();
    assert!(convert::run(&options, true).is_err());
    assert!(!output.exists());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}
