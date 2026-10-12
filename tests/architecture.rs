use bruce::architecture::Architecture;
use bullet_compiler::tensor::TValue;
use bullet_gpu::runtime::{Device, mock::MockGpu};
use bullet_lib::{game::formats::bulletformat::ChessBoard, wdl::ConstantWDL};
use bullet_trainer::{model::ModelWeights, run::HostPool};

#[test]
fn both_stockdory_architectures_compile_for_training()
{
    for architecture in [Architecture::Starshard, Architecture::Aurora] {
        let definition = architecture.definition();
        let forward = definition.lower_forward(1024).unwrap();
        for id in definition.ir().inputs().keys().chain(definition.ir().weights().keys()) {
            assert!(forward.map().contains_key(id));
        }
        let (output_id, name) = &definition.outputs()[0];

        assert_eq!(name, "output");
        assert_eq!(forward.ir().get_node(forward.map()[output_id]).unwrap().ty().size().get(), 1024);

        definition.lower_backward(&Default::default(), 4).unwrap();
    }
}

#[test]
fn exported_layout_matches_stockdory_mantaray()
{
    for (architecture, hidden, payload, padded) in
        [(Architecture::Starshard, 256, 394_754, 394_816), (Architecture::Aurora, 384, 592_130, 592_192)]
    {
        let mut weights = ModelWeights::zeroed(&architecture.definition());
        let mut feature = vec![0.0; 768 * hidden];
        feature[3 * hidden + 17] = -0.5;
        let mut bias = vec![0.0; hidden];
        bias[23] = 0.25;
        let mut output = vec![0.0; hidden * 2];
        output[11] = 1.5;
        output[hidden + 19] = -1.25;

        assert!(weights.set("l0w", TValue::F32(feature)));
        assert!(weights.set("l0b", TValue::F32(bias)));
        assert!(weights.set("l1w", TValue::F32(output)));
        assert!(weights.set("l1b", TValue::F32(vec![0.125])));

        let bytes = weights.to_quantised_buffer(&architecture.saved_format(), true).unwrap();
        let word = |index| i16::from_le_bytes(bytes[index * 2..index * 2 + 2].try_into().unwrap());

        assert_eq!(architecture.payload_bytes(), payload);
        assert_eq!(bytes.len(), padded);
        assert_eq!(word(3 * hidden + 17), -128);
        assert_eq!(word(768 * hidden + 23), 64);
        assert_eq!(word(769 * hidden + 11), 96);
        assert_eq!(word(770 * hidden + 19), -80);
        assert_eq!(word(771 * hidden), 2040);
    }
}

#[test]
fn forward_graph_has_shared_clipped_features_and_centipawn_scale()
{
    use std::collections::BTreeMap;

    for architecture in [Architecture::Starshard, Architecture::Aurora] {
        let definition = architecture.definition();
        let batch_size = 1024;
        let forward = definition.lower_forward(batch_size).unwrap();
        let hidden = architecture.hidden();
        let mut weights = ModelWeights::zeroed(&definition);
        let mut feature = vec![0.0; hidden * 768];
        feature[0] = 2.0;
        feature[hidden] = -1.0;
        let mut output = vec![0.0; hidden * 2];
        output[0] = 2.0;
        output[hidden] = -1.0;
        assert!(weights.set("l0w", TValue::F32(feature)));
        assert!(weights.set("l1w", TValue::F32(output)));
        assert!(weights.set("l1b", TValue::F32(vec![0.125])));

        let mut inputs = BTreeMap::new();
        for (id, (name, _)) in definition.ir().weights() {
            inputs.insert(forward.map()[id], weights.get(name).values.clone());
        }
        for (id, name) in definition.ir().inputs() {
            let mapped = forward.map()[id];
            if forward.ir().get_node(mapped).is_err() {
                continue;
            }
            let value = match name.as_str() {
                "stm" | "ntm" => {
                    let mut features = vec![-1; 32 * batch_size];
                    for position in 0..batch_size {
                        features[32 * position] = i32::from((position % 2 == 0) == (name == "ntm"));
                    }
                    TValue::I32(features)
                },
                "targets" => TValue::F32(vec![0.0; batch_size]),
                _ => panic!("Unknown model input"),
            };
            inputs.insert(mapped, value);
        }
        let values = forward.ir().evaluate(inputs).unwrap().unwrap();
        let output_id = forward.map()[&definition.outputs()[0].0];
        let predictions = values[&output_id].f32();
        assert!(predictions.iter().enumerate().all(|(index, value)| *value == [2.125, -0.875][index % 2]));
        assert_eq!(predictions[0] * bruce::architecture::SCALE, 850.0);
        assert_eq!(predictions[1] * bruce::architecture::SCALE, -350.0);
    }
}

#[test]
fn mapper_preserves_perspectives_signed_scores_and_results()
{
    let device = Device::<MockGpu>::new(0).unwrap();
    let pool = HostPool::new(device);

    for (side, our, their, expected_score, expected_result) in [
        ("w", vec![0, 324, 764], vec![440, 764, 324], 0.268_941_43, 1.0),
        ("b", vec![324, 440, 764], vec![764, 0, 324], 0.731_058_6, 0.0),
    ] {
        let position: ChessBoard = format!("4k3/8/8/8/8/8/8/P3K3 {side} - - 0 1 | -400 | 1.0").parse().unwrap();

        for (blend, expected_target) in [(0.0, expected_score), (1.0, expected_result)] {
            let mapper = Architecture::Aurora.mapper(ConstantWDL { value: blend });
            let batch = mapper.map(&pool, &[position], Default::default(), 1).unwrap();
            let stm = batch.inputs["stm"].as_slice::<i32>().unwrap();
            let ntm = batch.inputs["ntm"].as_slice::<i32>().unwrap();
            let target = batch.inputs["targets"].as_slice::<f32>().unwrap()[0];

            assert_eq!(&stm[..3], our);
            assert_eq!(&ntm[..3], their);
            assert_eq!(stm[3], -1);
            assert_eq!(ntm[3], -1);
            assert!((target - expected_target).abs() < 1e-6);
        }
    }
}

#[test]
#[ignore = "requires MANTARAY_ROOT and a C++20 compiler"]
fn exported_networks_match_actual_mantaray()
{
    use std::{path::Path, process::Command};

    use bullet_lib::game::inputs::{Chess768, SparseInputType};

    let root = std::env::var_os("MANTARAY_ROOT").expect("Set MANTARAY_ROOT to StockDory's pinned MantaRay");
    let temporary = tempfile::tempdir().unwrap();
    let executable = temporary.path().join("mantaray");
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "g++".into());
    let output = Command::new(compiler)
        .args(["-std=c++20", "-O2", "-U__amd64__", "-Wno-attributes", "-I"])
        .arg(Path::new(&root).join("include"))
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/mantaray.cpp"))
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

    for architecture in [Architecture::Starshard, Architecture::Aurora] {
        let definition = architecture.definition();
        let mut weights = ModelWeights::new(&definition, 743_592);
        let mut bias = vec![0.0; architecture.hidden()];
        bias.iter_mut().enumerate().for_each(|(i, value)| *value = (i % 11) as f32 * 0.03125);
        assert!(weights.set("l0b", TValue::F32(bias)));
        assert!(weights.set("l1b", TValue::F32(vec![-0.0625])));

        let bytes = weights.to_quantised_buffer(&architecture.saved_format(), true).unwrap();
        let network = temporary.path().join(format!("{architecture}.nnue"));
        std::fs::write(&network, &bytes).unwrap();
        let words: Vec<i16> = bytes[..architecture.payload_bytes()]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| i16::from_le_bytes([word[0], word[1]]))
            .collect();

        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "7k/2p5/8/P7/3Q4/8/6K1/8 w - - 0 40",
        ] {
            let hidden = architecture.hidden();
            let mut ours = words[768 * hidden..769 * hidden].to_vec();
            let mut theirs = ours.clone();
            let position: ChessBoard = format!("{fen} | 0 | 0.5").parse().unwrap();
            Chess768.map_features(&position, |stm, ntm| {
                for i in 0..hidden {
                    ours[i] = ours[i].wrapping_add(words[stm * hidden + i]);
                    theirs[i] = theirs[i].wrapping_add(words[ntm * hidden + i]);
                }
            });
            let sum =
                ours.into_iter().chain(theirs).enumerate().fold(i32::from(words[771 * hidden]), |sum, (i, value)| {
                    sum.wrapping_add(i32::from(value.clamp(0, 255)) * i32::from(words[769 * hidden + i]))
                });
            let expected = sum.wrapping_mul(400) / 16_320;
            let actual = Command::new(&executable).arg(hidden.to_string()).arg(&network).arg(fen).output().unwrap();
            assert!(actual.status.success(), "{}", String::from_utf8_lossy(&actual.stderr));
            let actual: i32 = String::from_utf8(actual.stdout).unwrap().parse().unwrap();
            assert_eq!(actual, expected, "{architecture}: {fen}");
        }
    }
}
