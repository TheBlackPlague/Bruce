use std::process::{Command, Output};

fn bruce(arguments: &[&str]) -> Output { Command::new(env!("CARGO_BIN_EXE_bruce")).args(arguments).output().unwrap() }

#[test]
fn exposes_three_workloads_and_resume()
{
    let help = bruce(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();

    for command in ["train", "convert", "analyze"] {
        assert!(text.contains(command));
    }

    assert!(text.contains("Train a network"));
    assert!(text.contains("Convert a dataset"));
    assert!(text.contains("Inspect dataset statistics"));

    let help = bruce(&["train", "--help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();

    for option in ["--check", "--resume", "--tensorboard-dir", "--plain"] {
        assert!(text.contains(option));
    }

    assert!(text.contains("Validate configuration and datasets without training"));
    assert!(text.contains("Examples:"));
}

#[test]
fn rejects_conflicting_preflight_and_resume()
{
    let output = bruce(&["train", "--check", "--resume", "checkpoint"]);

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr).unwrap().contains("cannot be used"));
}

#[test]
fn reports_missing_configuration_without_a_panic()
{
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.toml");
    let output = bruce(&["train", "--check", "--config", missing.to_str().unwrap()]);

    assert!(!output.status.success());

    let text = String::from_utf8(output.stderr).unwrap();

    assert!(text.contains("Error:"));
    assert!(text.contains("Help:"));
    assert!(text.contains("Failed:"));
    assert!(!text.contains("panicked"));
}

#[test]
fn conversion_and_analysis_commands_write_reports_and_session_logs()
{
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("input.txt"),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;125;0.5\n",
    )
    .unwrap();

    let convert = Command::new(env!("CARGO_BIN_EXE_bruce"))
        .current_dir(directory.path())
        .args([
            "convert",
            "--from",
            "text",
            "--to",
            "bullet",
            "--input",
            "input.txt",
            "--output",
            "output.bullet",
            "--separator",
            ";",
        ])
        .output()
        .unwrap();
    assert!(convert.status.success(), "{}", String::from_utf8_lossy(&convert.stderr));

    let input_bytes = std::fs::metadata(directory.path().join("input.txt")).unwrap().len();
    let output_bytes = std::fs::metadata(directory.path().join("output.bullet")).unwrap().len();
    assert_eq!(
        String::from_utf8(convert.stdout).unwrap(),
        format!("Converted 1 positions ({input_bytes} bytes to {output_bytes} bytes; 0 skipped)\n")
    );

    let analyze = Command::new(env!("CARGO_BIN_EXE_bruce"))
        .current_dir(directory.path())
        .args(["analyze", "--format", "bullet", "--input", "output.bullet", "--output", "report.json"])
        .output()
        .unwrap();
    assert!(analyze.status.success(), "{}", String::from_utf8_lossy(&analyze.stderr));
    assert_eq!(String::from_utf8(analyze.stdout).unwrap(), "1 positions; score range 125 to 125 cp; mean 125.0 cp\n");

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("report.json")).unwrap()).unwrap();
    assert_eq!(report["positions"], 1);
    assert_eq!(report["score_distribution"]["125"], 1);
    assert_eq!(std::fs::read_dir(directory.path().join("logs")).unwrap().count(), 2);
}

#[test]
fn redirected_display_modes_are_plain_and_quiet_keeps_results_and_warnings()
{
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("input.txt"), "4k3/8/8/8/8/8/8/4K3 w - - 0 1;125;0.5\n").unwrap();

    for (index, flags) in [
        vec!["--display", "live"],
        vec!["--plain", "--display", "live", "--color", "always"],
        vec!["--quiet", "--display", "live"],
        vec!["--verbosity", "verbose", "--plain"],
    ]
    .into_iter()
    .enumerate()
    {
        let output_path = format!("output-{index}.monty");
        let output = Command::new(env!("CARGO_BIN_EXE_bruce"))
            .current_dir(directory.path())
            .args(&flags)
            .args([
                "convert",
                "--from",
                "text",
                "--to",
                "monty",
                "--input",
                "input.txt",
                "--output",
                &output_path,
                "--separator",
                ";",
            ])
            .output()
            .unwrap();

        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();

        assert!(stdout.starts_with("Converted 1 positions"));
        assert!(!stdout.contains("Warning"));
        assert!(stderr.contains("Warning: Positions without moves"));
        assert!(!stderr.contains('\u{1b}'));
        assert!(!stderr.contains(">^>"));

        if flags.contains(&"--quiet") {
            assert!(!stderr.contains("Phase:"));
            assert!(!stderr.contains("Complete:"));
        } else {
            assert!(stderr.contains("Phase:"));
            assert!(stderr.contains("Complete:"));
        }
    }

    for file in std::fs::read_dir(directory.path().join("logs")).unwrap() {
        let text = std::fs::read_to_string(file.unwrap().path()).unwrap();

        assert!(text.contains("Warning: Positions without moves"));
        assert!(text.contains("Command outcome: Complete"));
        assert!(!text.contains('\u{1b}'));
    }
}

#[test]
fn failed_setup_is_logged_before_finalization_and_never_claims_completion()
{
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bruce"))
        .current_dir(directory.path())
        .args(["--quiet", "train", "--config", "missing.toml"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());

    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(stderr.contains("Error:"));
    assert!(stderr.contains("Failed:"));
    assert!(!stderr.contains("Complete:"));

    let logs = std::fs::read_dir(directory.path().join("logs")).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(logs.len(), 1);

    let text = std::fs::read_to_string(logs[0].path()).unwrap();

    assert!(text.contains("missing.toml"));
    assert!(text.find("Error:").unwrap() < text.find("Command outcome: Failed").unwrap());
    assert!(text.find("Command outcome: Failed").unwrap() < text.find("closed").unwrap());
}
