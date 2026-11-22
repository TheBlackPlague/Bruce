pub mod bench;
pub mod checkpoint;
pub mod resume;

use std::{cell::RefCell, fs, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Result, anyhow, bail};
use bullet_gpu::runtime::{Device, Gpu};
use bullet_lib::nn::ExecutionContext;
use bullet_trainer::{
    model::ModelWeights,
    optimiser::{Optimiser, adam::AdamW},
    reader::ReadMapLoader,
    run::{
        DataLoader, DataLoadingError, HostPool, PreparedBatchHost, Step, TrainingError, TrainingEvent,
        TrainingSchedule, TrainingSteps, train_with_observer,
    },
};

use self::checkpoint::DataRecord;
use crate::{
    config::{Config, Data},
    control::{Control, Outcome, Stop},
    data::{
        self, DataError, SourceIdentity,
        pipeline::accepts_any,
        select::Selection,
        stream::{LoaderStats, TrainingStream},
    },
    events::{Batch, Checkpoint, Ending, Event, Level, Resumed, RunInfo, Stage, Superbatch},
    schedule::Curve,
    telemetry::Hub,
};

pub type Trainer = Optimiser<ExecutionContext, AdamW<ExecutionContext>>;

const BACKEND: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "rocm") {
    "rocm"
} else if cfg!(feature = "metal") {
    "metal"
} else {
    "mock"
};

const NON_FINITE: &str =
    "Loss is not finite; stopping without saving invalid weights. Check the data and learning rate.";
const SAVE_FAILED: &str = "Checkpoint saving failed; stopping without it. Check disk space and permissions.";
const UNSAVED: &str = "No checkpoint was saved; the next training invocation will start from the beginning.";

pub fn run(config: Config, resume: Option<PathBuf>, control: &Control, hub: &Hub) -> Result<Ending>
{
    let halted = || control.halted().then(|| Ending::of(control.outcome()));

    let architecture = config.run.arch;

    if let Some(ending) = halted() {
        return Ok(ending);
    }

    located(&config.training.data)?;

    if !cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")) {
        bail!("Training needs a GPU build; enable cuda, rocm, or metal");
    }

    hub.message(Level::Detail, "Checking dataset identity for checkpoints");
    let identities = data::identities(&config.training.data.sources)?;

    hub.message(Level::Detail, "Checking checkpoint destinations");
    let resuming = resume.as_deref().map(|path| resume::inspect(path, &config, &identities, hub)).transpose()?;
    let (resumed, saved) = resuming.unzip();
    let completed = resumed.as_ref().map_or(0, |resumed| resumed.completed);
    checkpoint::check_destinations(&config, completed + 1)?;

    if let Some(ending) = halted() {
        return Ok(ending);
    }

    hub.message(Level::Detail, "Preparing network and GPU");
    let device = Device::<ExecutionContext>::new(config.hardware.gpu)
        .map_err(|error| anyhow!("Could not open GPU {}: {error:?}", config.hardware.gpu))?;
    let definition = architecture.definition();
    let weights = ModelWeights::new(&definition, config.random.seed);
    let mut trainer = Trainer::new(definition, weights, device, config.training.optimizer.params())
        .map_err(|error| anyhow!("Could not prepare the network on the GPU: {error:?}"))?;

    if let Some(path) = &resume {
        hub.message(Level::Detail, format!("Restoring checkpoint {}", path.display()));
        checkpoint::restore(&mut trainer, path)?;
    }

    if let Some(ending) = halted() {
        return Ok(ending);
    }

    let data = &config.training.data;
    hub.message(Level::Detail, "Checking training data");
    probe(data, control)?;
    let stream = TrainingStream::new(data, config.random.seed, saved.flatten(), Some(&identities))?;
    if let Some(resumed) = &resumed {
        resume::announce(resumed, hub);
    }

    if let Some(ending) = halted() {
        return Ok(ending);
    }

    let mapper = architecture.mapper(blend(&config));
    let loader =
        Timed { inner: ReadMapLoader::new(stream.clone(), mapper, data.threads.mapper), stats: stream.stats() };

    let lr = config.training.lr.curve();
    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: data.batch.size,
            batches_per_superbatch: data.batch.per_superbatch,
            start_superbatch: completed + 1,
            end_superbatch: config.training.superbatches,
        },
        lr_schedule: Box::new(move |step| lr(step.batch(), step.superbatch())),
        log_rate: config.training.log_every,
    };

    let run = RefCell::new(Run {
        config: &config,
        control,
        hub,
        identities: &identities,
        stream: &stream,
        wdl: config.training.wdl.curve(),
        resumed,
        rates: (0.0, 0.0),
        completed: 0,
        saved: false,
        ended: None,
    });

    hub.message(Level::Detail, "Compiling training and loading first batch");

    let result = train_with_observer(
        &mut trainer,
        schedule,
        loader,
        |_, step, loss| run.borrow_mut().batch(step, loss),
        |trainer, step| run.borrow_mut().superbatch(trainer, step),
        &mut |event| run.borrow_mut().observe(event),
    );
    stream.finish();
    hub.emit(Event::Stage(Stage::Finalizing));

    run.into_inner().resolve(result)
}

fn blend(config: &Config) -> impl Fn(Step) -> f32 + Send + Sync + 'static
{
    let wdl = config.training.wdl.curve();
    let table = config.training.wdl.table(config.training.superbatches);

    move |step| {
        let superbatch = step.superbatch();
        table
            .as_deref()
            .and_then(|table| table.get(superbatch).copied())
            .unwrap_or_else(|| wdl(step.batch(), superbatch))
    }
}

fn located(data: &Data) -> Result<(), DataError>
{
    data.sources.iter().try_for_each(|source| match fs::metadata(&source.path) {
        Ok(_) => Ok(()),
        Err(error) => Err(DataError::io(&error).at(source)),
    })
}

fn probe(data: &Data, control: &Control) -> Result<(), DataError>
{
    let selection = Selection { filter: data.filter.clone(), labels: data.labels, ..Selection::default() };
    for source in &data.sources {
        let selection = selection.clone().identified(source)?;
        if !accepts_any(source, &selection, control)? && control.stop() == Stop::Running {
            return Err(DataError::no_positions().at(source));
        }
    }

    Ok(())
}

struct Run<'a>
{
    config: &'a Config,
    control: &'a Control,
    hub: &'a Hub,
    identities: &'a [SourceIdentity],
    stream: &'a TrainingStream,
    wdl: Curve,
    resumed: Option<Resumed>,
    rates: (f32, f32),
    completed: usize,
    saved: bool,
    ended: Option<Result<Outcome>>,
}

impl Run<'_>
{
    fn observe(&mut self, event: &TrainingEvent)
    {
        let None = self.ended else { return };

        match event {
            TrainingEvent::RunStarted { steps, device_name, device_arch, .. } => {
                let config = self.config;
                let info = RunInfo {
                    name: config.run.name.clone(),
                    architecture: config.run.arch.name(),
                    device: device_name.clone(),
                    device_architecture: device_arch.clone(),
                    backend: BACKEND,
                    steps: *steps,
                    schedule: config.schedule_points(),
                    phases: config.phases(),
                    labels: config.training.data.labels.to_string(),
                    configuration: config.checkpoint_value(),
                    resumed: self.resumed.clone(),
                };
                self.hub.emit(Event::Started(Box::new(info)));
            },
            TrainingEvent::TrainingReady { setup_time, .. } => {
                self.hub.emit(Event::Loader(self.stream.stats().snapshot()));
                self.hub.emit(Event::Ready(*setup_time));
                self.hub.emit(Event::Stage(Stage::Training));
            },
            TrainingEvent::BatchCompleted { step, learning_rate, .. } => {
                self.rates = (*learning_rate, (self.wdl)(step.batch(), step.superbatch()));
            },
            TrainingEvent::SuperbatchCompleted { step, loss, superbatch_elapsed, .. } => {
                let ((learning_rate, wdl), superbatch, duration) = (self.rates, step.superbatch(), *superbatch_elapsed);
                self.hub.emit(Event::Superbatch(Superbatch { superbatch, loss: *loss, duration, learning_rate, wdl }));
            },
            _ => {},
        }
    }

    fn batch(&mut self, step: Step, loss: f32)
    {
        let None = self.ended else { return };

        let (superbatch, batch) = (step.superbatch(), step.batch() + 1);
        if !loss.is_finite() {
            self.hub.message(Level::Warning, NON_FINITE);
            self.end(Err(anyhow!("The loss is not finite at superbatch {superbatch}, batch {batch}")));
        } else if self.control.outcome() == Outcome::Aborted {
            self.end(Ok(Outcome::Aborted));
        } else {
            let (learning_rate, wdl) = self.rates;
            self.hub.emit(Event::Batch(Batch { superbatch, batch, loss, learning_rate, wdl }));
            self.hub.emit(Event::Loader(self.stream.stats().snapshot()));
        }
    }

    fn superbatch(&mut self, trainer: &Trainer, step: Step)
    {
        let None = self.ended else { return };

        let superbatch = step.superbatch();
        self.completed = superbatch;

        let committed = self.control.commit();
        if committed {
            self.ended = Some(Ok(Outcome::Stopped));
            self.hub.emit(Event::StopCommitted(superbatch));
        } else if self.control.outcome() == Outcome::Aborted {
            return self.end(Ok(Outcome::Aborted));
        }

        let reason = self.config.training.checkpoint.reason(superbatch, self.config.training.superbatches, committed);
        if let Some(reason) = reason {
            self.hub.emit(Event::Stage(Stage::SavingCheckpoint));
            let started = Instant::now();
            let data = self.stream.bookmark(superbatch).map(|bookmark| DataRecord { bookmark });
            let path = match checkpoint::save(self.config, self.identities, trainer, superbatch, data) {
                Ok(path) => path,
                Err(error) => {
                    self.hub.message(Level::Warning, SAVE_FAILED);
                    let error = error.context(format!("Could not save the checkpoint of superbatch {superbatch}"));
                    return self.end(Err(error));
                },
            };
            self.hub.emit(Event::Checkpoint(Checkpoint { superbatch, path, reason, duration: started.elapsed() }));
            self.saved = true;
        }

        if committed {
            self.stream.finish();
        } else if reason.is_some() {
            self.hub.emit(Event::Stage(Stage::Training));
        }
    }

    fn resolve(self, result: Result<(), TrainingError<ExecutionContext>>) -> Result<Ending>
    {
        let last = self.config.training.superbatches;
        let outcome = match (result, self.stream.failure(), self.ended) {
            (Ok(()), _, None) => Outcome::Finished,
            (Ok(()), _, Some(Ok(Outcome::Stopped))) if self.completed == last => Outcome::Finished,
            (Ok(()), _, Some(ended)) => ended?,
            (Err(TrainingError::DataLoadingError(_)), None, Some(ended)) if self.stream.asked() => ended?,
            (Err(TrainingError::DataLoadingError(_)), Some(error), _) => return Err(error.into()),
            (Err(error), ..) => bail!("Training failed inside Bullet: {error:?}"),
        };

        if outcome != Outcome::Finished && !self.saved && self.resumed.is_none() {
            self.hub.message(Level::Warning, UNSAVED);
        }

        Ok(Ending::of(outcome))
    }

    fn end(&mut self, ended: Result<Outcome>)
    {
        self.ended = Some(ended);
        self.stream.finish();
    }
}

struct Timed<L>
{
    inner: L,
    stats: Arc<LoaderStats>,
}

impl<L: DataLoader> DataLoader for Timed<L>
{
    fn map_batches<G: Gpu, F: FnMut(PreparedBatchHost<G>) -> bool>(
        self,
        pool: &Arc<HostPool<G>>,
        start: Step,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), DataLoadingError>
    {
        let stats = self.stats;
        self.inner.map_batches(pool, start, batch_size, |batch| {
            let handed = Instant::now();
            let finished = f(batch);
            LoaderStats::add(&stats.handoff, handed.elapsed());
            finished
        })
    }
}
