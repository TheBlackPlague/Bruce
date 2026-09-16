use std::{path::Path, process::Command};

use bruce::architecture::aurora;
use bullet_lib::game::inputs::{Chess768, SparseInputType};
use bullet_trainer::model::ModelWeights;
use bulletformat::ChessBoard;

const POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 6",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 4 6",
    "7k/2p5/8/P7/3Q4/8/6K1/8 w - - 0 40",
    "7k/2p5/8/P7/3Q4/8/6K1/8 b - - 0 40",
];

fn bullet_integer_evaluation(bytes: &[u8], fen: &str) -> i32 {
    assert_eq!(bytes.len(), aurora::NETWORK_BYTES);

    let weights: Vec<_> = bytes[..aurora::PAYLOAD_BYTES]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| i16::from_le_bytes([x[0], x[1]]))
        .collect();

    let bias_offset = 768 * aurora::HIDDEN;
    let output_offset = bias_offset + aurora::HIDDEN;

    let mut ours = weights[bias_offset..output_offset].to_vec();
    let mut theirs = ours.clone();

    let board: ChessBoard = format!("{fen} | 0 | 0.5").parse().unwrap();

    Chess768.map_features(&board, |stm, ntm| {
        for neuron in 0..aurora::HIDDEN {
            ours  [neuron] = ours  [neuron].wrapping_add(weights[stm * aurora::HIDDEN + neuron]);
            theirs[neuron] = theirs[neuron].wrapping_add(weights[ntm * aurora::HIDDEN + neuron]);
        }
    });

    let mut output = i32::from(*weights.last().unwrap());

    for (index, activation) in ours.into_iter().chain(theirs).enumerate() {
        output = output.wrapping_add(
            i32::from(activation.clamp(0, 255)) * i32::from(weights[output_offset + index]),
        );
    }

    output.wrapping_mul(400) / (255 * 64)
}

fn compare_network(executable: &Path, path: &Path, bytes: &[u8]) {
    for fen in POSITIONS {
        let result = Command::new(executable)
            .arg(path)
            .arg(fen)
            .output()
            .unwrap();

        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );

        let cpp: i32 = String::from_utf8(result.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();

        let rust = bullet_integer_evaluation(bytes, fen);

        assert_eq!(rust, cpp, "Perspective or export mismatch: {fen}");
        eprintln!("{fen}: Bullet integer = MantaRay = {cpp}");
    }
}

#[test]
#[ignore = "requires an external MantaRay v2 checkout and a C++20 compiler"]
fn exported_aurora_matches_actual_mantaray_v2() {
    let root = std::env::var_os("MANTARAY_ROOT")
        .expect("Set MANTARAY_ROOT to a MantaRay v2 checkout");

    let temporary = tempfile::tempdir().unwrap();
    let executable = temporary.path().join("mantaray-reference");
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "g++".into());

    let output = Command::new(compiler)
        .args(["-std=c++20", "-O2", "-U__amd64__", "-Wno-attributes"])
        .arg("-I")
        .arg(Path::new(&root).join("include"))
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/mantaray_reference.cpp"))
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut weights = ModelWeights::new(&aurora::definition(), 462);

    for id in ["l0b", "l1b"] {
        let mut values = weights.get(id).values.clone();

        for index in 0..values.size() {
            values.write(index, weights.get("l0w").values.read(index));
        }

        assert!(weights.set(id, values));
    }

    let bytes = weights.to_quantised_buffer(&aurora::saved_format(), true).unwrap();
    let network = temporary.path().join("exported.nnue");

    std::fs::write(&network, &bytes).unwrap();

    compare_network(&executable, &network, &bytes);

    if let Some(path) = std::env::var_os("STOCKDORY_NETWORK") {
        let path = Path::new(&path);
        let bytes = std::fs::read(path).unwrap();

        compare_network(&executable, path, &bytes);

        let raw = bullet_integer_evaluation(&bytes, POSITIONS[0]);

        assert_eq!(raw, 103);

        // Scale evaluation using StockDory's constants
        let material = 78.0;
        let a = ((60.09285654 * material / 58.0 - 67.89307926) * material / 58.0 - 74.69971460)
            * material
            / 58.0
            + 320.80620991;
        
        assert_eq!((100.0 * f64::from(raw) / a).round() as i32, 42);
    }
}
