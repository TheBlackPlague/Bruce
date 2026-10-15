pub mod architecture;
pub mod checkpoint;
pub mod metrics;

use std::{
    cell::RefCell,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, bail, ensure};
use bullet_gpu::runtime::Device;
use bullet_lib::{nn::ExecutionContext, trainer::schedule::lr::LrScheduler};
use bullet_trainer::{
    model::{ModelWeights, SavedFormat},
    optimiser::{Optimiser, adam::AdamW},
    reader::ReadMapLoader,
    run::{DataLoadingError, TrainingError, TrainingEvent, TrainingSchedule, TrainingSteps, train_with_observer},
};

use self::{architecture::Architecture, checkpoint::Checkpoints, metrics::TrainingMetrics};
use crate::{
    cli::{Cancelled, DisplayMode, Outcome, OutputOptions, Progress, Reporter},
    config::{Config, schedule::BATCHES_PER_SUPERBATCH},
    data::{self, DatasetReader, Format},
    logging::Logger,
};

pub type Trainer = Optimiser<ExecutionContext, AdamW<ExecutionContext>>;

pub fn check(config: &Config) -> Result<()>
{
    config.validate()?;
    data::validate(&config.data)
}

pub fn check_reported(config: &Config, reporter: &Reporter, stop: Arc<AtomicBool>) -> Result<Outcome>
{
    reporter.set_stop_flag(stop.clone());

    config.validate()?;
    
    let mut positions = 0u64;
    for source in &config.data.sources {
        if stop.load(Ordering::Relaxed) {
            return Ok(Outcome::Cancelled);
        }
        
        reporter.context("Dataset", source.path.to_string_lossy());
        reporter.phase("Validating dataset");
        
        let mut accepted = 0u64;
        data::visit_boards_stoppable(source, config.data.filter.as_ref(), &stop, |board| {
            if source.format != Format::Sf || data::valid_sf_score(board.score) {
                positions += 1;
                accepted += 1;
                
                if positions.is_multiple_of(4096) {
                    reporter.progress(Progress::Work { completed: positions, total: None, unit: "positions checked" });
                }
            }
            
            Ok(false)
        })?;
        
        reporter.progress(Progress::Work { completed: positions, total: None, unit: "positions checked" });
        
        if stop.load(Ordering::Relaxed) {
            return Ok(Outcome::Cancelled);
        }
        
        ensure!(accepted > 0, "No positions pass the filter; check the dataset and filter settings");
    }
    
    Ok(Outcome::Success)
}

pub fn run(config: Config, resume: Option<PathBuf>, plain: bool) -> Result<()>
{
    let reporter = Reporter::new(OutputOptions {
        display: if plain { DisplayMode::Plain } else { DisplayMode::Auto },
        ..OutputOptions::default()
    });
    
    let stop = Arc::new(AtomicBool::new(false));
    
    reporter.start("Train", &config.name);
    reporter.set_stop_flag(stop.clone());
    
    let result = (|| {
        config.validate()?;
        
        let path = PathBuf::from("logs").join(format!("{}.log", config.name));
        let tensorboard =
            config.tensorboard.enabled.then_some((config.tensorboard.path.as_path(), config.name.as_str()));
        
        reporter.attach_log(Logger::open(&path, tensorboard)?);
        reporter.context("Log", path.to_string_lossy());
        
        if cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")) {
            let signal = stop.clone();
            ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
                .context("Could not handle stop requests")?;
        }
        
        run_reported(config, resume, &reporter, stop)
    })();
    match result {
        Ok(outcome) => {
            reporter.finish(outcome)?;
            if outcome == Outcome::Cancelled {
                bail!("Training cancelled; resume from the latest saved checkpoint");
            }
            
            Ok(())
        },
        Err(error) => {
            if error.is::<Cancelled>() {
                reporter.finish(Outcome::Cancelled)?;
            } else {
                reporter.error(
                    &error.to_string(),
                    "Review the log before retrying or resuming.",
                    &format!("{error:#}"),
                );
                reporter.finish(Outcome::Failed).context("Could not finalize reporting after training failed")?;
            }
            
            Err(error)
        },
    }
}

pub fn run_reported(
    config: Config,
    resume: Option<PathBuf>,
    reporter: &Reporter,
    stop: Arc<AtomicBool>,
) -> Result<Outcome>
{
    reporter.set_stop_flag(stop.clone());

    config.validate()?;
    
    if !cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")) {
        bail!("Training needs a GPU build; enable cuda, rocm, or metal");
    }
    
    if stop.load(Ordering::Relaxed) {
        return Ok(Outcome::Cancelled);
    }
    
    reporter.context("Device", format!("GPU {}", config.device));
    reporter.context("Checkpoints", config.training.checkpoint.location.to_string_lossy());
    reporter.verbose(&format!("Configuration: {}", config.checkpoint_value()));
    reporter.phase("Checking training data");
    
    let architecture: Architecture = config.architecture.parse()?;
    
    let reader = DatasetReader::new(&config.data, config.seed, stop.clone())?;
    
    if stop.load(Ordering::Relaxed) {
        return Ok(Outcome::Cancelled);
    }
    
    reporter.phase("Checking dataset identity for checkpoints");
    let checkpoints = Checkpoints::new(&config)?;
    
    reporter.phase("Checking checkpoint destinations");
    let completed = resume.as_ref().map(|path| checkpoints.inspect(path)).transpose()?.unwrap_or(0);
    for superbatch in completed + 1..=config.training.superbatches {
        if config.training.checkpoint.policy.due(superbatch, config.training.superbatches, false) {
            checkpoints.check_destination(superbatch)?;
        }
    }
    
    if stop.load(Ordering::Relaxed) {
        return Ok(Outcome::Cancelled);
    }
    
    reporter.phase("Preparing network and GPU");
    
    let definition = architecture.definition();
    let weights = ModelWeights::new(&definition, config.seed);
    
    let device = Device::<ExecutionContext>::new(config.device)
        .map_err(|error| anyhow::anyhow!("{error:?}"))
        .context("Could not open the GPU; check the device number and GPU driver")?;
    
    let mut trainer = Trainer::new(definition, weights, device, config.optimizer.params())
        .map_err(|error| anyhow::anyhow!("{error:?}"))
        .context("Could not prepare training; check GPU memory and the network configuration")?;
    
    if stop.load(Ordering::Relaxed) {
        return Ok(Outcome::Cancelled);
    }
    
    if let Some(path) = &resume {
        reporter.phase("Restoring checkpoint");
        checkpoint::restore(&mut trainer, path)?;
        reporter.context("Resumed checkpoint", path.to_string_lossy());
        
        let exact =
            config.data.filter.is_none() && config.data.sources.iter().all(|source| source.format == Format::Bullet);
        reporter.notice(if exact {
            "Resuming weights, schedule and dataset position"
        } else {
            "Resuming weights and schedule; data order may change"
        });
    }
    
    if stop.load(Ordering::Relaxed) {
        return Ok(Outcome::Cancelled);
    }
    
    let steps = TrainingSteps {
        batch_size: config.data.batch_size,
        batches_per_superbatch: BATCHES_PER_SUPERBATCH,
        start_superbatch: completed + 1,
        end_superbatch: config.training.superbatches,
    };
    
    let mut metrics = TrainingMetrics::new(steps, config.training.log_every);
    
    let loader =
        ReadMapLoader::new(reader.clone(), architecture.mapper(config.wdl.clone()), config.data.mapping_threads);
    
    let schedule = TrainingSchedule {
        steps,
        lr_schedule: config.learning_rate.clone().boxed(),
        log_rate: config.training.log_every,
    };
    
    let format = architecture.saved_format();
    let state =
        RefCell::new(TrainingState { latest_checkpoint: resume, checkpoint_error: None, nonfinite_loss: false });
    
    reporter.phase("Compiling training and loading first batch");
    
    let result = train_with_observer(
        &mut trainer,
        schedule,
        loader,
        |_, _, loss| state.borrow_mut().observe_loss(loss, reporter, &stop),
        |trainer, step| {
            let mut state = state.borrow_mut();
            if config.training.checkpoint.policy.due(
                step.superbatch(),
                config.training.superbatches,
                stop.load(Ordering::Relaxed),
            ) {
                state.save(trainer, step.superbatch(), &checkpoints, &format, reporter, &stop);
            }
        },
        &mut |event| report_training_event(event, &mut metrics, reporter),
    );
    
    if let Some(sample) = metrics.finish() {
        reporter.sample(sample);
    }
    
    if let Some(progress) = metrics.progress() {
        reporter.progress(Progress::Training(progress));
    }
    
    if let Some(loss) = metrics.run_loss() {
        reporter.set_training_loss(loss);
    }
    
    let state = state.into_inner();
    if let Some(path) = &state.latest_checkpoint {
        reporter.context("Latest checkpoint", path.to_string_lossy());
    }
    
    let outcome = resolve_outcome(
        result,
        state.checkpoint_error,
        reader.error(),
        state.nonfinite_loss,
        stop.load(Ordering::Relaxed),
    )?;
    
    if outcome == Outcome::Cancelled && state.latest_checkpoint.is_none() {
        reporter.warning("No checkpoint was saved; the next training invocation will start from the beginning.");
    }
    
    Ok(outcome)
}

struct TrainingState
{
    latest_checkpoint: Option<PathBuf>,
    checkpoint_error: Option<anyhow::Error>,
    nonfinite_loss: bool,
}

impl TrainingState
{
    fn observe_loss(&mut self, loss: f32, reporter: &Reporter, stop: &AtomicBool)
    {
        if !loss.is_finite() {
            if !self.nonfinite_loss {
                reporter.warning(
                    "Loss is not finite; stopping without saving invalid weights. Check the data and learning rate.",
                );
            }
            
            self.nonfinite_loss = true;
            stop.store(true, Ordering::Relaxed);
        }
    }

    fn save(
        &mut self,
        trainer: &Trainer,
        superbatch: usize,
        checkpoints: &Checkpoints,
        format: &[SavedFormat],
        reporter: &Reporter,
        stop: &AtomicBool,
    )
    {
        if self.nonfinite_loss || self.checkpoint_error.is_some() {
            return;
        }
        
        reporter.phase("Saving checkpoint");
        match checkpoints.save(trainer, format, superbatch) {
            Ok(path) => {
                reporter.context("Latest checkpoint", path.to_string_lossy());
                self.latest_checkpoint = Some(path);
            },
            Err(error) => {
                reporter.warning(
                    "Checkpoint saving failed; stopping at a safe boundary. Check disk space and permissions.",
                );
                self.checkpoint_error = Some(error.context(format!(
                    "Could not save checkpoint for superbatch {superbatch}; check disk space and destination \
                     permissions"
                )));
                stop.store(true, Ordering::Relaxed);
            },
        }
        
        if stop.load(Ordering::Relaxed) {
            reporter.stopping();
        } else {
            reporter.phase("Training");
        }
    }
}

fn report_training_event(event: &TrainingEvent, metrics: &mut TrainingMetrics, reporter: &Reporter)
{
    match event {
        TrainingEvent::RunStarted { device_name, .. } => reporter.context("Device", device_name),
        TrainingEvent::TrainingReady { .. } => reporter.phase("Training"),
        TrainingEvent::SuperbatchCompleted { step, loss, .. } => {
            reporter.verbose(&format!("Superbatch {} complete: mean loss {loss:.6}", step.superbatch()))
        },
        TrainingEvent::RunCompleted { .. } => reporter.phase("Finalizing training"),
        _ => {},
    }
    
    let update = metrics.observe(event);
    
    if let Some(progress) = update.progress {
        reporter.progress(Progress::Training(progress));
    }
    
    if let Some(sample) = update.sample {
        reporter.sample(sample);
    }
}

fn resolve_outcome(
    result: Result<(), TrainingError<ExecutionContext>>,
    checkpoint_error: Option<anyhow::Error>,
    dataset_error: Option<String>,
    nonfinite_loss: bool,
    stopping: bool,
) -> Result<Outcome>
{
    if let Some(error) = checkpoint_error {
        return Err(error);
    }
    
    if let Some(error) = dataset_error {
        return Err(anyhow::anyhow!(error).context("Could not read the dataset; validate it with train --check"));
    }
    
    if nonfinite_loss {
        bail!("Training stopped because the loss is not finite; check the data and learning rate before resuming");
    }
    
    match result {
        Ok(()) if stopping => Ok(Outcome::Cancelled),
        Ok(()) => Ok(Outcome::Success),
        Err(error) if stopping && ended_early(&error) => Ok(Outcome::Cancelled),
        Err(error) => Err(training_error(error)),
    }
}

fn ended_early(error: &TrainingError<ExecutionContext>) -> bool
{
    match error {
        TrainingError::DataLoadingError(DataLoadingError::NoBatchesReceived) => true,
        TrainingError::DataLoadingError(DataLoadingError::Message(message)) => {
            message == "Data loader ended before all scheduled batches completed"
        },
        _ => false,
    }
}

fn training_error(error: TrainingError<ExecutionContext>) -> anyhow::Error
{
    let summary = match &error {
        TrainingError::DataLoadingError(_) => "Could not load training data; validate it with train --check",
        TrainingError::GradientCalculationError(_) | TrainingError::OptimiserUpdateError(_) => {
            "The GPU could not complete a training step; check available GPU memory and driver diagnostics"
        },
        TrainingError::CompilingBackwards(_) => {
            "Could not prepare the network for training; check the GPU backend and network configuration"
        },
        TrainingError::InvalidSchedule => "The training schedule is invalid; check the training configuration",
        TrainingError::IoError => "Could not read or write training files; check paths, permissions and free space",
        TrainingError::Unexpected(_) => "The GPU could not complete the operation; check GPU driver diagnostics",
    };
    anyhow::anyhow!("{error:?}").context(summary)
}
