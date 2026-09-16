use std::{
    path::PathBuf,
    process::{Command as ProcessCommand, ExitCode},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use bruce::{
    config::Config,
    convert::{self, ConvertOptions},
    data,
    events::{Event, Reporter},
    training, ui,
};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "bruce",
    version,
    about = "🦈 Bruce — Neural Network Trainer Interface for StockDory",
    styles = cli_styles()
)]
struct Cli {
    #[arg(long, global = true)]
    plain: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Train(TrainOptions),

    Check(CheckOptions),

    Convert(ConvertOptions),

    #[command(name = "__worker", hide = true)]
    Worker {
        #[command(subcommand)]
        command: WorkCommand,
    },
}

#[derive(Subcommand)]
enum WorkCommand {
    Train(TrainOptions),
    Check(CheckOptions),
    Convert(ConvertOptions),
}

#[derive(Args)]
struct TrainOptions {
    #[arg(short, long, default_value = "config/training.toml")]
    config: PathBuf,

    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(Args)]
struct CheckOptions {
    #[arg(short, long, default_value = "config/training.toml")]
    config: PathBuf,
}

fn cli_styles() -> clap::builder::Styles {
    use clap::builder::styling::RgbColor;
    clap::builder::Styles::styled()
        .header     (RgbColor(227, 195, 139).on_default().bold())
        .usage      (RgbColor(227, 195, 139).on_default().bold())
        .literal    (RgbColor( 69, 202, 255).on_default()       )
        .placeholder(RgbColor(173, 176, 178).on_default()       )
        .error      (RgbColor(255, 117, 138).on_default().bold())
        .valid      (RgbColor( 85, 220, 133).on_default()       )
        .invalid    (RgbColor(255, 117, 138).on_default()       )
}

fn main() -> ExitCode {
    match execute(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Bruce: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn execute(cli: Cli) -> Result<()> {
    if let Command::Worker { command } = cli.command {
        let reporter = Reporter::new();

        return match command {
            WorkCommand::Train(options) => {
                training::run(Config::load(&options.config)?, options.resume, reporter)
            }

            WorkCommand::Check(options) => {
                reporter.emit(Event::Phase {
                    name: "Checking configuration".into(),
                    completed: 0,
                    total: None,
                });

                let config = Config::load(&options.config)?;
                data::validate(&config.data, &reporter)?;

                reporter.emit(Event::Note {
                    message: config.data.resume_note().to_string(),
                });

                reporter.emit(Event::Note {
                    message: "Configuration and dataset preflight passed.".into(),
                });

                reporter.emit(Event::Finished);

                Ok(())
            }

            WorkCommand::Convert(options) => convert::convert(&options, &reporter),
        };
    }

    let log_directory = match &cli.command {
        Command::Train(options) => Config::load(&options.config)?.output_directory,
        _ => std::env::current_dir()?.join("logs"),
    };

    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let log_path = log_directory.join(format!("bruce-{stamp}.log"));

    let mut child = ProcessCommand::new(
        std::env::current_exe().context("Locating Bruce executable")?
    );
    child.arg("__worker").args(std::env::args_os().skip(1));

    ui::run_child(child, cli.plain, &log_path)
}
