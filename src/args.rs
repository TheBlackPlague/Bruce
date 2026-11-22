use std::{fs, num::NonZero, path::PathBuf, thread};

use anyhow::{Context, Result};
use clap::{
    Args, ColorChoice, Parser, Subcommand,
    builder::{NonEmptyStringValueParser, RangedU64ValueParser, TypedValueParser},
};
use time::OffsetDateTime;

use crate::{
    analyze::{self, NetworkOptions, REPORTS, check::CheckMode},
    config::Config,
    data::{Source, is_unit, pipeline::Sampling},
    model::Architecture,
};

#[derive(Parser)]
#[command(
    name = "bruce",
    version,
    about = "Neural network training and dataset tools for StockDory",
    after_help = "Start with: bruce train --check\nUse bruce <COMMAND> --help for options and examples."
)]
pub struct Arguments
{
    #[arg(
        short,
        long,
        global = true,
        conflicts_with = "verbose",
        help_heading = "Display options",
        help = "Print only the result, without the live progress"
    )]
    pub quiet: bool,
    #[arg(
        short,
        long,
        global = true,
        help_heading = "Display options",
        help = "Print the backtrace of an unexpected failure"
    )]
    pub verbose: bool,
    #[arg(
        long,
        global = true,
        value_enum,
        default_value = "auto",
        help_heading = "Display options",
        help = "Color status output; auto honours NO_COLOR"
    )]
    pub color: ColorChoice,
    #[arg(
        long,
        global = true,
        value_name = "DIR",
        default_value = "logs",
        conflicts_with = "no_log",
        help_heading = "Log options",
        help = "Directory that receives <name>.log for training and bruce.log for other commands"
    )]
    pub log_dir: PathBuf,
    #[arg(long, global = true, help_heading = "Log options", help = "Write no text log")]
    pub no_log: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command
{
    #[command(about = "Train a network")]
    Train(TrainArguments),
    #[command(about = "Inspect dataset statistics and optional network predictions")]
    Analyze(AnalyzeArguments),
}

#[derive(Args)]
#[command(after_help = "Examples:\n  bruce train --check\n  bruce train --config config/training.toml\n  bruce train \
                        --resume checkpoints/run-10\n  bruce train --resume checkpoints/run-10 --config \
                        config/longer.toml\n  bruce train --benchmark 60")]
pub struct TrainArguments
{
    #[arg(
        short,
        long,
        value_name = "PATH",
        help = "Training configuration [default: config/training.toml; with --resume, the configuration stored in the \
                checkpoint]"
    )]
    pub config: Option<PathBuf>,
    #[arg(
        long,
        value_name = "DIR",
        conflicts_with = "check",
        help = "Checkpoint directory to continue from; with --config it may change the run: a new name forks it, more \
                superbatches extend it"
    )]
    pub resume: Option<PathBuf>,
    #[arg(
        long,
        value_name = "MODE",
        num_args = 0..=1,
        default_missing_value = "quick",
        value_enum,
        help = "Validate the configuration and measure the filter effect; no GPU. quick samples the first 200,000 \
                games of each source, full scans everything [default when given: quick]"
    )]
    pub check: Option<CheckMode>,
    #[arg(
        long,
        value_name = "SECONDS",
        num_args = 0..=1,
        default_missing_value = "20",
        value_parser = RangedU64ValueParser::<u64>::new().range(1..),
        conflicts_with_all = ["check", "resume"],
        help = "Measure loader throughput for SECONDS per measurement: the Bruce reader alone and with the mapper; \
                no GPU [default when given: 20]"
    )]
    pub benchmark: Option<u64>,
    #[arg(
        long,
        value_name = "DIR",
        help = "Directory that receives <name>/metrics.jsonl; default [training] metrics, else runs"
    )]
    pub metrics_dir: Option<PathBuf>,
    #[arg(
        long,
        value_name = "BATCHES",
        value_parser = RangedU64ValueParser::<usize>::new().range(1..),
        help = "Metric sample interval in batches [default: the configuration's log_every]"
    )]
    pub log_every: Option<usize>,
    #[arg(long, help = "Do not open the bruce-plot viewer for this run")]
    pub no_viewer: bool,
}

#[derive(Args)]
#[command(after_help = "Examples:\n  bruce analyze --input games.binpack\n  bruce analyze --input a.binpack --input \
                        b.binpack --sample head:200000 --output report.json\n  bruce analyze --config \
                        config/training.toml --sample head:200000")]
pub struct AnalyzeArguments
{
    #[arg(
        short = 'i',
        long = "input",
        value_name = "PATH",
        required_unless_present = "config",
        num_args = 1..,
        help = "Monty dataset file; repeat for several"
    )]
    pub inputs: Vec<PathBuf>,
    #[arg(
        short,
        long,
        value_name = "PATH",
        help = "Take sources, filter and label policy from a training configuration; --input then replaces the \
                sources only"
    )]
    pub config: Option<PathBuf>,
    #[arg(
        long,
        value_name = "SPEC",
        value_parser = NonEmptyStringValueParser::new().try_map(|value| value.parse::<Sampling>()),
        help = "all or head:<GAMES> [default: all]"
    )]
    pub sample: Option<Sampling>,
    #[arg(
        short,
        long,
        value_name = "N",
        value_parser = RangedU64ValueParser::<usize>::new().range(1..=256),
        help = "Worker threads, from 1 to 256 [default: logical CPUs]"
    )]
    pub threads: Option<usize>,
    #[arg(long, value_name = "DIR", help = "Checkpoint directory to evaluate [default: no network]")]
    pub network: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        ignore_case = true,
        requires = "network",
        help = "Architecture of the --network checkpoint [default: the one its manifest records]"
    )]
    pub arch: Option<Architecture>,
    #[arg(long, value_name = "INDEX", default_value_t = 0, requires = "network", help = "GPU for network inference")]
    pub gpu: u32,
    #[arg(
        long,
        value_name = "N",
        default_value_t = NetworkOptions::BATCH_SIZE,
        value_parser = RangedU64ValueParser::<usize>::new().range(1..),
        requires = "network",
        help = "Inference batch size"
    )]
    pub batch_size: usize,
    #[arg(
        short,
        long,
        value_name = "PATH",
        help = "Write the JSON report to this path, which must not exist [default: reports/analysis-<UTC \
                date>-<time>.json]"
    )]
    pub output: Option<PathBuf>,
}

impl AnalyzeArguments
{
    pub fn options(self) -> Result<analyze::Options>
    {
        let now = OffsetDateTime::now_utc();
        let (month, day, hour, minute, second) =
            (u8::from(now.month()), now.day(), now.hour(), now.minute(), now.second());
        let created = format!("{:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z", now.year());
        let threads = self.threads.unwrap_or_else(|| thread::available_parallelism().map_or(1, NonZero::get).min(256));
        let network = self.network.map(|checkpoint| NetworkOptions {
            checkpoint,
            architecture: self.arch,
            gpu: self.gpu,
            batch_size: self.batch_size,
        });

        let mut config = self.config.map(|path| Config::load(&path).map(|config| (path, config))).transpose()?;
        let given = !self.inputs.is_empty();
        let inputs: Vec<Source> = match &config {
            Some((_, config)) if !given => config.training.data.sources.clone(),
            _ => self.inputs.into_iter().map(Source::new).collect(),
        };
        let replaced = given
            && config
                .as_ref()
                .is_some_and(|(_, config)| config.training.data.sources.iter().any(|source| !is_unit(&source.scale)));
        let warnings =
            replaced.then(|| "--input replaces the configuration's sources; their scale is not used".to_owned());
        if let Some((_, config)) = &mut config {
            config.training.data.sources.clone_from(&inputs);
            config.training.data.validate()?;
        }
        if self.output.is_none() {
            fs::create_dir_all(REPORTS).with_context(|| format!("Could not create {REPORTS}"))?;
        }

        Ok(analyze::Options {
            inputs,
            sampling: self.sample.unwrap_or(Sampling::All),
            threads,
            output: self.output,
            created,
            network,
            config,
            warnings: warnings.into_iter().collect(),
        })
    }
}
