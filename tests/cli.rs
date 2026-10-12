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

    let help = bruce(&["train", "--help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();

    for option in ["--check", "--resume", "--tensorboard-dir", "--plain"] {
        assert!(text.contains(option));
    }
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
    
    assert!(text.starts_with("Bruce: "));
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
    
    let analyze = Command::new(env!("CARGO_BIN_EXE_bruce"))
        .current_dir(directory.path())
        .args(["analyze", "--format", "bullet", "--input", "output.bullet", "--output", "report.json"])
        .output()
        .unwrap();
    assert!(analyze.status.success(), "{}", String::from_utf8_lossy(&analyze.stderr));
    
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("report.json")).unwrap()).unwrap();
    assert_eq!(report["positions"], 1);
    assert_eq!(report["score_distribution"]["125"], 1);
    assert_eq!(std::fs::read_dir(directory.path().join("logs")).unwrap().count(), 2);
}
