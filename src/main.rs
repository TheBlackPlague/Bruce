use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::ExitCode,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use bruce::{
    analyze,
    architecture::Architecture,
    config::{Config, DataSource},
    convert,
    data::Format,
    train,
};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bruce", version, about = "Neural Network Training Interface & Tools for StockDory")]
struct Cli
{
    /// Print progress without live updates
    #[arg(long, global = true)]
    plain: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command
{
    /// Train a network or check a training configuration
    Train(TrainOptions),
    /// Convert a dataset
    Convert(convert::Options),
    /// Analyze a dataset and optional network checkpoint
    Analyze(AnalyzeOptions),
}

#[derive(Args)]
struct TrainOptions
{
    #[arg(short, long, default_value = "config/training.toml")]
    config: PathBuf,
    #[arg(long, conflicts_with = "check")]
    resume: Option<PathBuf>,
    /// Check the configuration and datasets without training
    #[arg(long)]
    check: bool,
    /// Override the TensorBoard output directory and enable reporting
    #[arg(long)]
    tensorboard_dir: Option<PathBuf>,
}

#[derive(Args)]
struct AnalyzeOptions
{
    #[arg(long, value_enum)]
    format: Format,
    #[arg(long)]
    input: PathBuf,
    /// Separator used by text records
    #[arg(long)]
    separator: Option<String>,
    /// Native Bruce checkpoint to evaluate
    #[arg(long)]
    network: Option<PathBuf>,
    #[arg(long, value_enum, requires = "network")]
    arch: Option<Architecture>,
    #[arg(long, default_value_t = 0)]
    gpu: i32,
    /// Maximum positions to analyze
    #[arg(long)]
    limit: Option<u64>,
    /// Save the full report as JSON
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> ExitCode
{
    std::panic::set_hook(Box::new(|_| eprintln!("Bruce: An operation failed unexpectedly")));

    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Bruce: {error}");
            ExitCode::FAILURE
        },
    }
}

fn run(cli: Cli) -> Result<()>
{
    match cli.command {
        Command::Train(options) => {
            let mut config = Config::load(&options.config)?;

            if let Some(path) = options.tensorboard_dir {
                config.tensorboard.enabled = true;
                config.tensorboard.path = path;
            }

            if options.check {
                logged("Checking configuration and datasets", || {
                    train::check(&config)?;
                    Ok("Configuration and datasets checked".into())
                })
            } else {
                train::run(config, options.resume, cli.plain)
            }
        },

        Command::Convert(options) => logged("Converting dataset", || {
            let summary = convert::run(&options, cli.plain)?;
            Ok(summary.to_string())
        }),

        Command::Analyze(options) => logged("Analyzing dataset", || {
            let report = analyze::run(&analyze::Options {
                source: DataSource { format: options.format, path: options.input, separator: options.separator },
                network: options.network,
                architecture: options.arch,
                gpu: options.gpu,
                limit: options.limit,
            })?;

            if let Some(path) = options.output {
                report.write(&path)?;
            }

            Ok(report.summary())
        }),
    }
}

fn logged(label: &str, operation: impl FnOnce() -> Result<String>) -> Result<()>
{
    fs::create_dir_all("logs")?;

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let mut log = File::create_new(format!("logs/bruce-{timestamp}-{}.log", std::process::id()))?;

    writeln!(log, "{label}")?;
    log.flush()?;

    eprintln!("{label}");

    match operation() {
        Ok(summary) => {
            writeln!(log, "{summary}")?;
            println!("{summary}");
            Ok(())
        },
        Err(error) => {
            let _ = writeln!(log, "{error}");
            Err(error)
        },
    }
}
