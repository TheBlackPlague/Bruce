use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, Result, bail};
use bullet_lib::trainer::schedule::lr::LrScheduler;
use bullet_trainer::{
    reader::ReadMapLoader,
    run::{TrainingSchedule, TrainingSteps, train},
};

use crate::{
    architecture::aurora,
    checkpoint,
    config::Config,
    data,
    events::{Event, Reporter},
};

struct CallbackFailure(anyhow::Error);

pub fn run(config: Config, resume: Option<PathBuf>, reporter: Reporter) -> Result<()> {
    anyhow::ensure!(
        cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")),
        "Training requires a GPU backend; rebuild Bruce with --features cuda, rocm, or metal"
    );

    config.validate()?;

    reporter.emit(Event::Phase {
        name: "Checking configuration and data".into(),
        completed: 0,
        total: None,
    });

    data::validate(&config.data, &reporter)?;

    reporter.emit(Event::Note {
        message: config.data.resume_note().to_string(),
    });

    let start_superbatch = if let Some(path) = &resume {
        reporter.emit(Event::Phase {
            name: "Verifying checkpoint".into(),
            completed: 0,
            total: None,
        });

        checkpoint::inspect(path, &config)?.completed_superbatch + 1
    } else {
        1
    };

    checkpoint::check_destinations(&config, start_superbatch)?;

    reporter.emit(Event::Phase {
        name: "Initializing device and Aurora".into(),
        completed: 0,
        total: None,
    });

    let mut optimiser = aurora::create(config.seed, config.device, config.optimizer.params())?;

    if let Some(path) = &resume {
        reporter.emit(Event::Phase {
            name: "Restoring weights and optimizer".into(),
            completed: 0,
            total: None,
        });

        checkpoint::restore(&mut optimiser, path)?;
    }

    reporter.emit(Event::Phase {
        name: "Preparing data".into(),
        completed: 0,
        total: None,
    });

    let reader = data::reader(&config.data)?;
    let mapper = aurora::mapper(config.wdl.clone());
    let loader = ReadMapLoader::new(reader, mapper, config.data.mapping_threads);

    let steps = TrainingSteps {
        batch_size: config.training.batch_size,
        batches_per_superbatch: config.training.batches_per_superbatch,
        start_superbatch,
        end_superbatch: config.training.superbatches,
    };
    let schedule = TrainingSchedule {
        steps,
        lr_schedule: config.learning_rate.clone().boxed(),
        log_rate: config.training.log_every,
    };

    reporter.emit(Event::Phase {
        name: "Compiling kernels and filling Bullet's data queue".into(),
        completed: 0,
        total: None,
    });

    let training_started = Instant::now();

    let mut positions = 0_u64;
    let mut loss_sum = 0.0_f64;
    let mut loss_batches = 0_usize;

    let result = catch_unwind(AssertUnwindSafe(|| {
        train(
            &mut optimiser,
            schedule,
            loader,

            |_, step, loss| {
                if !loss.is_finite() {
                    resume_unwind(Box::new(CallbackFailure(anyhow::anyhow!(
                        "Training loss became non-finite at superbatch {}, batch {}",
                        step.superbatch(),
                        step.batch() + 1
                    ))));
                }

                positions += config.training.batch_size as u64;
                loss_sum += f64::from(loss);
                loss_batches += 1;

                let batch = step.batch() + 1;

                if  positions == config.training.batch_size as u64  ||
                    batch.is_multiple_of(config.training.log_every) ||
                    batch == step.batches_per_superbatch()
                {
                    reporter.emit(Event::Metric {
                        superbatch: step.superbatch(),
                        batch,
                        batches_per_superbatch: step.batches_per_superbatch(),
                        final_superbatch: step.final_superbatch(),
                        loss: (loss_sum / loss_batches as f64) as f32,
                        learning_rate: config.learning_rate.lr(step.batch(), step.superbatch()),
                        positions,
                        total_positions: ((step.superbatch() - 1) * step.batches_per_superbatch()
                            + batch) as u64
                            * config.training.batch_size as u64,
                        elapsed_seconds: training_started.elapsed().as_secs_f64(),
                    });

                    loss_sum = 0.0;
                    loss_batches = 0;
                }
            },

            |optimiser, step| {
                if  step.superbatch().is_multiple_of(config.training.save_every) ||
                    step.superbatch() == step.final_superbatch()
                {
                    reporter.emit(Event::Phase {
                        name: "Saving checkpoint and quantizing Aurora".into(),
                        completed: 0,
                        total: None,
                    });

                    match checkpoint::save(optimiser, &config, step.superbatch()) {
                        Ok(path) => reporter.emit(Event::Checkpoint { path }),
                        Err(error) => resume_unwind(Box::new(CallbackFailure(error))),
                    }
                }
            },
        )
    }));

    match result {
        Ok(result) => {
            result.map_err(|error| anyhow::anyhow!("Bullet training failed: {error:?}"))?
        }

        Err(payload) => match payload.downcast::<CallbackFailure>() {
            Ok(failure) => return Err(failure.0).context("Training stopped"),

            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown native training panic");
                bail!("Bullet training failed: {message}");
            }
        },
    }

    reporter.emit(Event::Finished);

    Ok(())
}
