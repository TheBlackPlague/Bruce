mod inference;
mod statistics;

use std::{
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, ensure};
use bullet_lib::game::formats::bulletformat::ChessBoard;
use inference::{BATCH_SIZE, Inference};
use statistics::Statistics;
pub use statistics::{Calibration, ErrorSummary, Group, NetworkReport, Report, Results, ScoreSummary};

use crate::{
    architecture::Architecture,
    checkpoint,
    cli::{self, Progress, Reporter},
    config::DataSource,
    data,
};

pub struct Options
{
    pub source: DataSource,
    pub network: Option<PathBuf>,
    pub architecture: Option<Architecture>,
    pub gpu: i32,
    pub limit: Option<u64>,
}

pub fn run(options: &Options) -> Result<Report>
{
    let reporter = Reporter::new(cli::OutputOptions::default());
    reporter.start("Analyze", &options.source.path.to_string_lossy());

    let result = run_reported(options, &reporter, Arc::new(AtomicBool::new(false)));

    let outcome = match &result {
        Ok(_) => cli::Outcome::Success,
        Err(error) if error.is::<cli::Cancelled>() => cli::Outcome::Cancelled,
        Err(error) => {
            reporter.error(
                "Dataset analysis failed",
                "Check the input format and checkpoint settings",
                &format!("{error:#}"),
            );
            cli::Outcome::Failed
        },
    };

    let finished = reporter.finish(outcome);
    let report = result?;

    finished?;
    Ok(report)
}

pub fn run_reported(options: &Options, reporter: &Reporter, stop: Arc<AtomicBool>) -> Result<Report>
{
    reporter.set_stop_flag(stop.clone());
    reporter.phase("Preparing analysis");
    check_cancelled(&stop)?;

    ensure!(options.limit != Some(0), "Position limit must be positive");
    ensure!(options.gpu >= 0, "GPU index must not be negative");
    ensure!(
        options.network.is_some() == options.architecture.is_some(),
        "Network analysis needs both a checkpoint and an architecture"
    );

    data::validate_source(&options.source, None)?;

    let mut inference = if let (Some(path), Some(architecture)) = (&options.network, options.architecture) {
        ensure!(cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")), "Network analysis needs a GPU build");

        reporter.phase("Loading checkpoint for analysis");

        let weights = checkpoint::load_weights(path, architecture)?;

        check_cancelled(&stop)?;
        reporter.phase("Preparing network inference");

        Some(Inference::new(architecture, &weights, options.gpu)?)
    } else {
        None
    };

    check_cancelled(&stop)?;
    reporter.phase(if inference.is_some() {
        "Analyzing positions and network predictions"
    } else {
        "Analyzing positions"
    });

    let mut statistics = Statistics::new(options.architecture);
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    data::visit_boards_stoppable(&options.source, None, &stop, |board| {
        check_cancelled(&stop)?;

        statistics.record(&board);
        if let Some(inference) = &mut inference {
            batch.push(board);
            if batch.len() == BATCH_SIZE {
                predict(inference, &mut batch, &mut statistics)?;
            }
        }

        if statistics.positions().is_multiple_of(BATCH_SIZE as u64) {
            report_progress(reporter, statistics.positions());
        }

        Ok(options.limit == Some(statistics.positions()))
    })?;

    check_cancelled(&stop)?;
    if let Some(inference) = &mut inference {
        predict(inference, &mut batch, &mut statistics)?;
    }
    report_progress(reporter, statistics.positions());

    ensure!(statistics.positions() > 0, "Dataset contains no positions");

    check_cancelled(&stop)?;
    reporter.phase("Finalizing analysis");

    Ok(statistics.finish(options.limit))
}

fn predict(inference: &mut Inference, batch: &mut Vec<ChessBoard>, statistics: &mut Statistics) -> Result<()>
{
    if !batch.is_empty() {
        let predictions = inference.predict(batch)?;
        for (board, prediction) in batch.iter().zip(predictions) {
            statistics.prediction(board, prediction)?;
        }

        batch.clear();
    }

    Ok(())
}

fn report_progress(reporter: &Reporter, completed: u64)
{
    reporter.progress(Progress::Work { completed, total: None, unit: "positions analyzed" });
}

fn check_cancelled(stop: &AtomicBool) -> Result<()>
{
    if stop.load(Ordering::Relaxed) {
        return Err(cli::Cancelled.into());
    }

    Ok(())
}

impl Report
{
    pub fn summary(&self) -> String
    {
        let mut summary = format!(
            "{} positions; score range {} to {} cp; mean {:.1} cp",
            self.positions, self.scores.minimum_cp, self.scores.maximum_cp, self.scores.mean_cp
        );

        if let Some(network) = &self.network {
            summary.push_str(&format!(
                "; network MAE {:.1} cp, bias {:+.1} cp",
                network.overall.mean_absolute_error_cp, network.overall.bias_cp
            ));
        }

        if self.limit_reached {
            summary.push_str(" (position limit reached)");
        }

        summary
    }

    pub fn write(&self, path: &Path) -> Result<()>
    {
        let directory = path.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));

        let mut file = tempfile::NamedTempFile::new_in(directory).context("Cannot create analysis report")?;
        {
            let mut writer = BufWriter::new(file.as_file_mut());

            serde_json::to_writer_pretty(&mut writer, self).context("Cannot write analysis report")?;

            writer.write_all(b"\n")?;
            writer.flush().context("Cannot finish analysis report")?;
        }

        file.as_file().sync_all().context("Cannot sync analysis report")?;
        file.persist_noclobber(path)
            .map_err(|error| anyhow::anyhow!(error.error))
            .context("Cannot publish analysis report; choose an unused path in a writable directory")?;

        Ok(())
    }
}
