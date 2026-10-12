use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use bullet_gpu::runtime::Device;
use bullet_lib::{nn::ExecutionContext, trainer::schedule::lr::LrScheduler};
use bullet_trainer::{
    model::ModelWeights,
    optimiser::{Optimiser, adam::AdamW},
    reader::ReadMapLoader,
    run::{DataLoadingError, TrainingError, TrainingSchedule, TrainingSteps, train_with_observer},
};

use crate::{
    architecture::Architecture,
    checkpoint::{self, Checkpoints},
    config::{CheckpointPolicy, Config},
    data::{self, DatasetReader, Format},
    report::Reporter,
    schedule::BATCHES_PER_SUPERBATCH,
};

pub type Trainer = Optimiser<ExecutionContext, AdamW<ExecutionContext>>;

pub fn check(config: &Config) -> Result<()>
{
    config.validate()?;
    let _ = config.architecture.parse::<Architecture>()?;
    data::validate(&config.data)?;
    Ok(())
}

pub fn run(config: Config, resume: Option<PathBuf>, plain: bool) -> Result<()>
{
    config.validate()?;
    if !cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")) {
        bail!("Training needs a GPU build; enable cuda, rocm, or metal");
    }

    let architecture = config.architecture.parse::<Architecture>()?;
    eprintln!("Preparing training");
    let stop = Arc::new(AtomicBool::new(false));
    let reader = DatasetReader::new(&config.data, config.seed, stop.clone())?;
    let checkpoints = Checkpoints::new(&config)?;
    let completed = resume.as_ref().map(|path| checkpoints.inspect(path)).transpose()?.unwrap_or(0);

    if completed >= config.training.superbatches {
        bail!("This checkpoint has already finished the configured training");
    }

    for superbatch in completed + 1..=config.training.superbatches {
        if should_save(&config, superbatch) {
            checkpoints.check_destination(superbatch)?;
        }
    }

    let definition = architecture.definition();
    let weights = ModelWeights::new(&definition, config.seed);
    let device =
        Device::<ExecutionContext>::new(config.device).map_err(|_| anyhow::anyhow!("Could not open the GPU"))?;
    let mut trainer = Trainer::new(definition, weights, device, config.optimizer.params())
        .map_err(|_| anyhow::anyhow!("Could not prepare training"))?;

    if let Some(path) = &resume {
        checkpoint::restore(&mut trainer, path)?;
    }

    let steps = TrainingSteps {
        batch_size: config.data.batch_size,
        batches_per_superbatch: BATCHES_PER_SUPERBATCH,
        start_superbatch: completed + 1,
        end_superbatch: config.training.superbatches,
    };
    let reporter = RefCell::new(Reporter::new(&config, steps, plain)?);

    if resume.is_some() {
        let exact_position =
            config.data.filter.is_none() && config.data.sources.iter().all(|source| source.format == Format::Bullet);
        let message = if exact_position {
            "Resuming weights, schedule and dataset position"
        } else {
            "Resuming weights and schedule; data order may change"
        };
        reporter.borrow_mut().message(message);
    }

    let mapper = architecture.mapper(config.wdl.clone());
    let loader = ReadMapLoader::new(reader.clone(), mapper, config.data.mapping_threads);
    let schedule = TrainingSchedule {
        steps,
        lr_schedule: config.learning_rate.clone().boxed(),
        log_rate: config.training.log_every,
    };
    let saved_format = architecture.saved_format();
    let checkpoint_error = RefCell::new(None);
    let nonfinite_loss = Cell::new(false);
    let saved = Cell::new(completed);

    let signal = stop.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed)).context("Could not handle stop requests")?;

    let result = train_with_observer(
        &mut trainer,
        schedule,
        loader,
        |_, _, loss| {
            if !loss.is_finite() {
                nonfinite_loss.set(true);
                stop.store(true, Ordering::Relaxed);
            }
        },
        |trainer, step| {
            let superbatch = step.superbatch();
            let stopping = stop.load(Ordering::Relaxed)
                && matches!(config.training.checkpoint.policy, CheckpointPolicy::Periodic { .. });

            if !nonfinite_loss.get()
                && checkpoint_error.borrow().is_none()
                && (should_save(&config, superbatch) || stopping)
            {
                reporter.borrow_mut().message("Saving checkpoint");
                match checkpoints.save(trainer, &saved_format, superbatch) {
                    Ok(path) => {
                        saved.set(superbatch);
                        reporter.borrow_mut().message(&format!("Checkpoint saved: {}", path.display()));
                    },
                    Err(error) => {
                        *checkpoint_error.borrow_mut() = Some(error);
                        stop.store(true, Ordering::Relaxed);
                    },
                }
            }
        },
        &mut |event| reporter.borrow_mut().on_event(event),
    );

    let reporting = reporter.into_inner().finish();

    if let Some(error) = checkpoint_error.into_inner() {
        return Err(error);
    }

    if let Some(error) = reader.error() {
        bail!("Could not read the dataset: {error}");
    }

    if nonfinite_loss.get() {
        bail!("Training stopped because the loss is not finite");
    }

    match result {
        Ok(()) => {},
        Err(error) if stop.load(Ordering::Relaxed) && ended_early(&error) => {
            if saved.get() == 0 {
                bail!("Training stopped without a checkpoint");
            }

            bail!("Training stopped; latest checkpoint is superbatch {}", saved.get());
        },
        Err(error) => return Err(training_error(error)),
    }

    reporting
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

fn should_save(config: &Config, superbatch: usize) -> bool
{
    config.training.checkpoint.policy.should_save(superbatch)
        || (matches!(config.training.checkpoint.policy, CheckpointPolicy::Periodic { .. })
            && superbatch == config.training.superbatches)
}

fn training_error(error: TrainingError<ExecutionContext>) -> anyhow::Error
{
    match error {
        TrainingError::DataLoadingError(_) => anyhow::anyhow!("Could not load training data"),
        TrainingError::GradientCalculationError(_) | TrainingError::OptimiserUpdateError(_) => {
            anyhow::anyhow!("The GPU could not complete a training step")
        },
        TrainingError::CompilingBackwards(_) => {
            anyhow::anyhow!("Could not prepare the network for training")
        },
        TrainingError::InvalidSchedule => anyhow::anyhow!("The training schedule is invalid"),
        TrainingError::IoError => anyhow::anyhow!("Could not read or write training files"),
        TrainingError::Unexpected(_) => anyhow::anyhow!("The GPU could not complete the operation"),
    }
}
