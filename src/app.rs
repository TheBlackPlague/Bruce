use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    process::ExitCode,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use bruce::{
    analyze,
    architecture::Architecture,
    cli::{Cancelled, ColorMode, CommandOutput, DisplayMode, Outcome, OutputOptions, Reporter, Verbosity},
    config::{Config, DataSource},
    convert,
    data::Format,
    logging::Logger,
    train,
};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "bruce",
    version,
    about = "Neural network training and dataset tools for StockDory",
    after_help = "Start with: bruce train --check\nUse bruce <COMMAND> --help for options and examples."
)]
struct Arguments
{
    #[arg(long, global = true, help_heading = "Display options")]
    plain: bool,
    #[arg(long, global = true, value_enum, default_value = "auto", help_heading = "Display options")]
    display: DisplayMode,
    #[arg(long, global = true, value_enum, default_value = "normal", help_heading = "Display options")]
    verbosity: Verbosity,
    #[arg(short, long, global = true, conflicts_with = "verbose", help_heading = "Display options")]
    quiet: bool,
    #[arg(short, long, global = true, conflicts_with = "quiet", help_heading = "Display options")]
    verbose: bool,
    #[arg(long, global = true, value_enum, default_value = "auto", help_heading = "Display options")]
    color: ColorMode,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command
{
    #[command(about = "Train a network")]
    Train(TrainOptions),
    #[command(about = "Convert a dataset between supported formats")]
    Convert(convert::Options),
    #[command(about = "Inspect dataset statistics and optional network predictions")]
    Analyze(AnalyzeOptions),
}

#[derive(Args)]
#[command(after_help = "Examples:\n  bruce train --check\n  bruce train --config config/training.toml\n  bruce train \
                        --resume checkpoints/run-10")]
struct TrainOptions
{
    #[arg(short, long, default_value = "config/training.toml")]
    config: PathBuf,
    #[arg(long, conflicts_with = "check")]
    resume: Option<PathBuf>,
    #[arg(long, help = "Validate configuration and datasets without training")]
    check: bool,
    #[arg(long)]
    tensorboard_dir: Option<PathBuf>,
    #[arg(long)]
    log_every: Option<usize>,
}

#[derive(Args)]
#[command(after_help = "Examples:\n  bruce analyze --format bullet --input positions.bullet\n  bruce analyze \
                        --format bullet --input positions.bullet --limit 100000 --output report.json")]
struct AnalyzeOptions
{
    #[arg(long, value_enum)]
    format: Format,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    separator: Option<String>,
    #[arg(long)]
    network: Option<PathBuf>,
    #[arg(long, value_enum, requires = "network")]
    arch: Option<Architecture>,
    #[arg(long, default_value_t = 0)]
    gpu: i32,
    #[arg(long)]
    limit: Option<u64>,
    #[arg(long)]
    output: Option<PathBuf>,
}

pub fn run() -> ExitCode
{
    let args = Arguments::parse();
    let options = OutputOptions {
        display: if args.plain { DisplayMode::Plain } else { args.display },
        verbosity: if args.quiet {
            Verbosity::Quiet
        } else if args.verbose {
            Verbosity::Verbose
        } else {
            args.verbosity
        },
        color: args.color,
    };

    let task = match &args.command {
        Command::Train(options) if options.check => "check",
        Command::Train(_) => "train",
        Command::Convert(_) => "convert",
        Command::Analyze(_) => "analyze",
    };

    let reporter = Reporter::new(options);
    reporter.start(task, "");

    let stop = Arc::new(AtomicBool::new(false));
    reporter.set_stop_flag(stop.clone());

    let panic_details = Arc::new(Mutex::new(None));
    let captured = panic_details.clone();
    std::panic::set_hook(Box::new(move |info| {
        *captured.lock().unwrap_or_else(|error| error.into_inner()) = Some(info.to_string());
    }));

    let signal = stop.clone();
    let result = ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .context("Could not install Ctrl+C handling")
        .and_then(|()| {
            catch_unwind(AssertUnwindSafe(|| execute(args.command, &reporter, stop))).unwrap_or_else(|_| {
                Err(anyhow::anyhow!("Operation failed unexpectedly; inspect the log and report this issue"))
            })
        });

    if let Some(details) = panic_details.lock().unwrap_or_else(|error| error.into_inner()).take() {
        reporter.verbose(&format!("Underlying panic: {details}"));
    }

    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) if error.is::<Cancelled>() => Outcome::Cancelled,
        Err(error) => {
            reporter.error(&error.to_string(), recovery(task), &format!("{error:#}"));
            Outcome::Failed
        },
    };

    if reporter.finish(outcome).is_err() {
        return ExitCode::FAILURE;
    }

    match outcome {
        Outcome::Success => ExitCode::SUCCESS,
        Outcome::Cancelled => ExitCode::from(130),
        Outcome::Failed => ExitCode::FAILURE,
    }
}

fn execute(command: Command, reporter: &Reporter, stop: Arc<AtomicBool>) -> Result<Outcome>
{
    match command {
        Command::Train(options) => training(options, reporter, stop),
        Command::Convert(options) => {
            attach_log(reporter, None, None)?;

            reporter.context("Input", options.input.to_string_lossy());
            reporter.context("Output", options.output.to_string_lossy());
            reporter.context("Format", format!("{} → {}", format_name(options.from), format_name(options.to)));

            let summary = convert::run_reported(&options, reporter, stop)?;

            reporter.verbose(&summary.to_string());
            reporter.set_output(CommandOutput::conversion(&summary));

            Ok(Outcome::Success)
        },
        Command::Analyze(options) => {
            attach_log(reporter, None, None)?;

            reporter.context("Input", options.input.to_string_lossy());
            reporter.context("Format", format_name(options.format));

            if let Some(network) = &options.network {
                reporter.context("Network", network.to_string_lossy());
            }

            if let Some(limit) = options.limit {
                reporter.context("Limit", format!("{limit} positions"));
            }

            let report = analyze::run_reported(
                &analyze::Options {
                    source: DataSource { format: options.format, path: options.input, separator: options.separator },
                    network: options.network,
                    architecture: options.arch,
                    gpu: options.gpu,
                    limit: options.limit,
                },
                reporter,
                stop,
            )?;

            if let Some(path) = options.output {
                reporter.phase("Writing analysis report");

                report.write(&path)?;

                reporter.context("Report", path.to_string_lossy());
            }

            reporter.verbose(&report.summary());
            reporter.set_output(CommandOutput::analysis(&report));

            Ok(Outcome::Success)
        },
    }
}

fn training(options: TrainOptions, reporter: &Reporter, stop: Arc<AtomicBool>) -> Result<Outcome>
{
    reporter.phase("Reading configuration");
    reporter.context("Config", options.config.to_string_lossy());

    let prepared = (|| {
        let mut config = Config::load(&options.config)?;

        if let Some(path) = options.tensorboard_dir {
            config.tensorboard.enabled = true;
            config.tensorboard.path = path;
        }

        if let Some(interval) = options.log_every {
            config.training.log_every = interval;
        }

        config.validate()?;

        Ok::<_, anyhow::Error>(config)
    })();

    let config = match prepared {
        Ok(config) => config,
        Err(error) => {
            attach_log(reporter, None, None)?;
            return Err(error);
        },
    };

    let log = PathBuf::from("logs").join(format!("{}.log", config.name));
    let tensorboard = (config.tensorboard.enabled && !options.check)
        .then_some((config.tensorboard.path.as_path(), config.name.as_str()));
    attach_log(reporter, Some(log), tensorboard)?;

    reporter.context("Run", &config.name);
    reporter.verbose(&format!("Configuration: {config:#?}"));

    if stop.load(Ordering::Relaxed) {
        return Err(Cancelled.into());
    }

    if options.check {
        let outcome = train::check_reported(&config, reporter, stop)?;
        if outcome == Outcome::Success {
            reporter.set_output(CommandOutput::check());
        }

        Ok(outcome)
    } else {
        train::run_reported(config, options.resume, reporter, stop)
    }
}

fn attach_log(reporter: &Reporter, path: Option<PathBuf>, tensorboard: Option<(&std::path::Path, &str)>) -> Result<()>
{
    let path = path.unwrap_or_else(|| {
        let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        PathBuf::from(format!("logs/bruce-{time}-{}.log", std::process::id()))
    });

    reporter.attach_log(
        Logger::open(&path, tensorboard)
            .context("Could not open the diagnostic log; check permissions and free space")?,
    );
    reporter.context("Log", path.to_string_lossy());

    if let Some((root, name)) = tensorboard {
        reporter.context("TensorBoard", root.join(name).to_string_lossy());
    }

    reporter.verbose("Detailed diagnostics are saved in the log. Use --verbose or --help for more detail.");

    Ok(())
}

fn format_name(format: Format) -> String
{
    format.to_possible_value().expect("Dataset formats have CLI names").get_name().to_owned()
}

fn recovery(task: &str) -> &'static str
{
    match task {
        "train" | "check" => {
            "Check configuration and dataset paths with train --check; use --verbose and the log for details."
        },
        "convert" => {
            "Check input format, separator, output permissions and free space; choose a new output path if it exists."
        },
        _ => "Check input format, paths and checkpoint architecture; use --verbose and the log for details.",
    }
}
