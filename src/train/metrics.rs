use std::time::Duration;

use bullet_trainer::run::{TrainingEvent, TrainingSteps};

pub use crate::logging::TensorBoard;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricSample
{
    pub step: i64,
    pub loss: f32,
    pub learning_rate: f32,
    pub speed: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingProgress
{
    pub superbatch: usize,
    pub final_superbatch: usize,
    pub batch: usize,
    pub batches_per_superbatch: usize,
    pub global_batch: usize,
    pub session_batches: usize,
    pub loss: Option<f64>,
    pub loss_batches: usize,
    pub learning_rate: f32,
    pub elapsed: Duration,
    pub positions_per_second: Option<f64>,
    pub superbatch_eta: Option<Duration>,
    pub run_eta: Option<Duration>,
}

impl TrainingProgress
{
    fn update_rates(&mut self, batch_size: usize)
    {
        self.positions_per_second = None;
        self.superbatch_eta = None;
        self.run_eta = None;

        let rate = self.session_batches as f64 / self.elapsed.as_secs_f64();
        if !rate.is_finite() || rate <= 0.0 {
            return;
        }

        let positions = rate * batch_size as f64;
        self.positions_per_second = positions.is_finite().then_some(positions);

        let batches_left = self.batches_per_superbatch.saturating_sub(self.batch);
        let run_left = self
            .final_superbatch
            .saturating_sub(self.superbatch)
            .saturating_mul(self.batches_per_superbatch)
            .saturating_add(batches_left);
        self.superbatch_eta = Duration::try_from_secs_f64(batches_left as f64 / rate).ok();
        self.run_eta = Duration::try_from_secs_f64(run_left as f64 / rate).ok();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Observation
{
    pub progress: Option<TrainingProgress>,
    pub sample: Option<MetricSample>,
}

#[derive(Debug, Default)]
struct Average
{
    sum: f64,
    count: usize,
}

impl Average
{
    fn add(&mut self, value: f32)
    {
        self.sum += f64::from(value);
        self.count += 1;
    }

    fn value(&self) -> Option<f64> { (self.count > 0).then(|| self.sum / self.count as f64) }
}

#[derive(Debug)]
pub struct TrainingMetrics
{
    batch_size: usize,
    log_every: usize,
    setup: Duration,
    latest: Option<TrainingProgress>,
    window: Average,
    session: Average,
}

impl TrainingMetrics
{
    pub fn new(steps: TrainingSteps, log_every: usize) -> Self
    {
        Self {
            batch_size: steps.batch_size,
            log_every: log_every.max(1),
            setup: Duration::ZERO,
            latest: None,
            window: Average::default(),
            session: Average::default(),
        }
    }

    pub fn observe(&mut self, event: &TrainingEvent) -> Observation
    {
        match event {
            TrainingEvent::TrainingReady { setup_time, .. } => self.setup = *setup_time,
            TrainingEvent::BatchCompleted { step, loss, learning_rate, elapsed, .. } => {
                self.window.add(*loss);
                self.session.add(*loss);

                let batch = step.batch() + 1;

                let mut progress = TrainingProgress {
                    superbatch: step.superbatch(),
                    final_superbatch: step.final_superbatch(),
                    batch,
                    batches_per_superbatch: step.batches_per_superbatch(),
                    global_batch: step.total_batches() + 1,
                    session_batches: self.session.count,
                    loss: self.window.value(),
                    loss_batches: self.window.count,
                    learning_rate: *learning_rate,
                    elapsed: elapsed.saturating_sub(self.setup),
                    positions_per_second: None,
                    superbatch_eta: None,
                    run_eta: None,
                };
                progress.update_rates(self.batch_size);

                self.latest = Some(progress);

                let sample_due =
                    batch == 1 || batch == step.batches_per_superbatch() || batch.is_multiple_of(self.log_every);
                return Observation { progress: self.latest, sample: if sample_due { self.finish() } else { None } };
            },
            TrainingEvent::RunCompleted { elapsed, .. } => {
                if let Some(progress) = &mut self.latest {
                    progress.elapsed = elapsed.saturating_sub(self.setup);
                    progress.update_rates(self.batch_size);
                }

                return Observation { progress: self.latest, sample: None };
            },
            _ => {},
        }
        Observation::default()
    }

    pub fn progress(&self) -> Option<TrainingProgress> { self.latest }

    pub fn run_loss(&self) -> Option<f64> { self.session.value() }

    pub fn finish(&mut self) -> Option<MetricSample>
    {
        let progress = self.latest?;
        let loss = self.window.value()? as f32;

        self.window = Average::default();

        Some(MetricSample {
            step: i64::try_from(progress.global_batch).ok()?,
            loss,
            learning_rate: progress.learning_rate,
            speed: (progress.positions_per_second.unwrap_or(0.0) / 1e6) as f32,
        })
    }
}
