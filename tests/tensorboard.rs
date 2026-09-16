use bruce::{events::Event, tensorboard::TensorBoard};

#[test]
#[ignore = "requires Python tensorboard; exercised by CI"]
fn official_tensorboard_reads_live_final_and_partial_runs() {
    let root = tempfile::tempdir().unwrap();
    let mut logger = TensorBoard::start(root.path(), "roundtrip").unwrap();

    let event = Event::Metric {
        superbatch: 3,
        batch: 10,
        batches_per_superbatch: 100,
        final_superbatch: 5,
        loss: 0.125,
        learning_rate: 0.001,
        positions: 1000,
        total_positions: 21000,
        elapsed_seconds: 2.0,
    };

    logger.observe(&event);

    std::thread::sleep(std::time::Duration::from_millis(5200));

    let verify = |path: &std::path::Path| {
        let result = std::process::Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/verify_tensorboard.py"
            ))
            .arg(path)
            .output()
            .unwrap();

        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    };

    verify(&logger.directory);
    logger.finish().unwrap();
    verify(&logger.directory);

    let path;
    {
        let mut partial = TensorBoard::start(root.path(), "roundtrip").unwrap();
        path = partial.directory.clone();
        partial.observe(&event);
    }

    verify(&path);
}
