mod encode;
mod games;

use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, anyhow, ensure};
use clap::Args;

use crate::{
    cli::{self, Progress, Reporter},
    config::DataSource,
    data::{self, Format},
};

#[derive(Debug, Args)]
#[command(after_help = "Example:\n  bruce convert --from text --to bullet --input positions.txt --output \
                        positions.bullet --separator ';'")]
pub struct Options
{
    #[arg(long, value_enum)]
    pub from: Format,
    #[arg(long, value_enum)]
    pub to: Format,
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub separator: Option<String>,
    #[arg(long)]
    pub threads: Option<usize>,
}

#[derive(Debug)]
pub struct Summary
{
    pub positions: u64,
    pub skipped: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

impl std::fmt::Display for Summary
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(
            formatter,
            "Converted {} positions ({} bytes to {} bytes; {} skipped)",
            self.positions, self.input_bytes, self.output_bytes, self.skipped
        )
    }
}

pub fn run(options: &Options, plain: bool) -> Result<Summary>
{
    let reporter = Reporter::new(cli::OutputOptions {
        display: if plain { cli::DisplayMode::Plain } else { cli::DisplayMode::Auto },
        ..Default::default()
    });

    reporter.start("Convert", &options.input.to_string_lossy());
    reporter.context("Output", options.output.to_string_lossy());

    let result = run_reported(options, &reporter, Arc::new(AtomicBool::new(false)));
    if let Err(error) = &result
        && !error.is::<cli::Cancelled>()
    {
        reporter.error(
            "Dataset conversion failed",
            "Check the input format and destination permissions",
            &format!("{error:#}"),
        );
    }

    let outcome = match &result {
        Ok(_) => cli::Outcome::Success,
        Err(error) if error.is::<cli::Cancelled>() => cli::Outcome::Cancelled,
        Err(_) => cli::Outcome::Failed,
    };
    let finished = reporter.finish(outcome);

    let summary = result?;
    finished?;

    Ok(summary)
}

pub fn run_reported(options: &Options, reporter: &Reporter, stop: Arc<AtomicBool>) -> Result<Summary>
{
    reporter.set_stop_flag(stop.clone());

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| convert(options, reporter, &stop))).map_err(|panic| {
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("Unknown panic while decoding or encoding the dataset");
        anyhow!("{message}").context("Could not convert the dataset")
    })?
}

fn check_stop(stop: &AtomicBool) -> Result<()>
{
    if stop.load(Ordering::Relaxed) {
        return Err(cli::Cancelled.into());
    }
    Ok(())
}

fn progress(reporter: &Reporter, completed: u64, unit: &'static str)
{
    reporter.progress(Progress::Work { completed, total: None, unit });
}

fn convert(options: &Options, reporter: &Reporter, stop: &AtomicBool) -> Result<Summary>
{
    reporter.phase("Preparing conversion");
    check_stop(stop)?;

    let threads = options.threads.unwrap_or(1);
    ensure!((1..=256).contains(&threads), "Threads must be between 1 and 256");

    ensure!(!options.output.exists(), "The output file already exists");
    if options.from == Format::Text || options.to == Format::Text {
        let separator = options.separator.as_deref().context("Text conversion needs --separator")?;
        ensure!(!separator.is_empty() && !separator.contains(['\n', '\r']), "Choose a nonempty text separator");
    }

    let source = DataSource { path: options.input.clone(), format: options.from, separator: options.separator.clone() };
    data::validate_source(&source, None)?;

    check_stop(stop)?;

    let parent =
        options.output.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).context("Could not create the output file")?;

    let mut summary =
        Summary { positions: 0, skipped: 0, input_bytes: std::fs::metadata(&options.input)?.len(), output_bytes: 0 };
    if options.from == options.to {
        summary.positions = copy(&source, temporary.as_file_mut(), summary.input_bytes, reporter, stop)?;
    } else {
        reporter.phase("Converting positions");

        bullet_lib::game::formats::viriformat::chess::CHESS960.store(true, Ordering::Relaxed);

        if matches!(options.from, Format::Bullet | Format::Text)
            && matches!(options.to, Format::Monty | Format::Viri | Format::Sf)
        {
            reporter
                .warning("Positions without moves use a legal placeholder move; original games cannot be recovered");
        }

        if options.from == Format::Bullet {
            reporter.warning("Bullet data has no original side, castling rights or move counters");
        }

        let mut writer = BufWriter::new(temporary.as_file_mut());
        if matches!((options.from, options.to), (Format::Monty, Format::Viri) | (Format::Viri, Format::Monty)) {
            summary.positions = games::convert(options, threads, &mut writer, reporter, stop)?;
        } else {
            convert_records(&source, options, threads, &mut writer, reporter, stop, &mut summary)?;
        }

        progress(reporter, summary.positions, "positions converted");
        check_stop(stop)?;

        reporter.phase("Flushing converted dataset");
        writer.flush().context("Could not flush the converted dataset")?;
    }

    check_stop(stop)?;
    reporter.phase("Saving converted dataset");

    temporary.as_file().sync_all().context("Could not sync the converted dataset")?;
    summary.output_bytes = temporary.as_file().metadata()?.len();

    check_stop(stop)?;

    temporary
        .persist_noclobber(&options.output)
        .map_err(|error| anyhow!(error.error))
        .context("Could not publish the converted dataset; check the destination permissions and free space")?;

    if summary.skipped > 0 {
        reporter.warning(&format!(
            "Skipped {} positions with scores that cannot be represented in the output",
            summary.skipped
        ));
    }

    Ok(summary)
}

fn convert_records(
    source: &DataSource,
    options: &Options,
    threads: usize,
    writer: &mut impl Write,
    reporter: &Reporter,
    stop: &AtomicBool,
    summary: &mut Summary,
) -> Result<()>
{
    let mut records = Vec::with_capacity(8192);
    let (_, skipped) = data::visit_counted_stoppable(source, stop, |record| {
        check_stop(stop)?;

        if options.from == Format::Sf && options.to == Format::Viri && i16::try_from(record.white_score()).is_err() {
            summary.skipped += 1;
            return Ok(false);
        }

        records.push(record);
        if records.len() == 8192 {
            write_parallel(&records, threads, |records| encode::records(records, options), writer)?;
            summary.positions += records.len() as u64;

            records.clear();

            progress(reporter, summary.positions, "positions converted");
        }

        Ok(false)
    })?;

    summary.skipped += skipped;

    check_stop(stop)?;

    write_parallel(&records, threads, |records| encode::records(records, options), writer)?;
    summary.positions += records.len() as u64;

    ensure!(summary.positions > 0, "No positions could be converted");

    Ok(())
}

fn copy(source: &DataSource, writer: &mut impl Write, total: u64, reporter: &Reporter, stop: &AtomicBool)
-> Result<u64>
{
    reporter.phase("Checking positions before copying");

    let mut checked: u64 = 0;
    let (positions, skipped) = data::visit_counted_stoppable(source, stop, |_| {
        check_stop(stop)?;

        checked += 1;
        if checked.is_multiple_of(8192) {
            progress(reporter, checked, "positions checked");
        }

        Ok(false)
    })?;

    progress(reporter, positions + skipped, "positions checked");

    check_stop(stop)?;
    reporter.phase("Copying validated dataset");

    let mut input = File::open(&source.path)?;
    let mut buffer = [0; 64 * 1024];
    let mut copied = 0;
    loop {
        check_stop(stop)?;

        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }

        writer.write_all(&buffer[..count])?;
        copied += count as u64;

        reporter.progress(Progress::Work { completed: copied, total: Some(total), unit: "bytes copied" });
    }

    Ok(positions + skipped)
}

fn write_parallel<T: Sync>(
    items: &[T],
    threads: usize,
    encode: impl Fn(&[T]) -> Result<Vec<u8>> + Sync,
    writer: &mut impl Write,
) -> Result<()>
{
    if items.is_empty() {
        return Ok(());
    }

    if threads == 1 {
        return Ok(writer.write_all(&encode(items)?)?);
    }

    let chunk_size = items.len().div_ceil(threads.min(items.len()));
    std::thread::scope(|scope| {
        let encode = &encode;

        let workers = items.chunks(chunk_size).map(|items| scope.spawn(move || encode(items))).collect::<Vec<_>>();
        for worker in workers {
            let bytes = worker.join().map_err(|_| anyhow!("Could not encode the dataset"))??;
            writer.write_all(&bytes)?;
        }

        Ok(())
    })
}
