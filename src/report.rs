use std::time::Duration;

use bullet_trainer::run::{Step, TrainingEvent, TrainingSteps};

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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Observation
{
    pub progress: Option<TrainingProgress>,
    pub sample: Option<MetricSample>,
}

#[derive(Debug)]
pub struct TrainingMetrics
{
    steps: TrainingSteps,
    log_every: usize,
    setup: Duration,
    latest: Option<TrainingProgress>,
    window_loss_sum: f64,
    window_batches: usize,
    run_loss_sum: f64,
    session_batches: usize,
}

impl TrainingMetrics
{
    pub fn new(steps: TrainingSteps, log_every: usize) -> Self
    {
        Self {
            steps,
            log_every: log_every.max(1),
            setup: Duration::ZERO,
            latest: None,
            window_loss_sum: 0.0,
            window_batches: 0,
            run_loss_sum: 0.0,
            session_batches: 0,
        }
    }

    pub fn observe(&mut self, event: &TrainingEvent) -> Observation
    {
        match event {
            TrainingEvent::TrainingReady { setup_time, .. } => {
                self.setup = *setup_time;
                Observation::default()
            },
            TrainingEvent::BatchCompleted { step, loss, learning_rate, elapsed, .. } => {
                self.batch_completed(*step, *loss, *learning_rate, *elapsed)
            },
            TrainingEvent::RunCompleted { elapsed, .. } => {
                if let Some(mut progress) = self.latest {
                    progress.elapsed = elapsed.saturating_sub(self.setup);
                    self.update_rates(&mut progress);
                    self.latest = Some(progress);
                }

                Observation { progress: self.latest, sample: None }
            },
            _ => Observation::default(),
        }
    }

    pub fn progress(&self) -> Option<TrainingProgress> { self.latest }

    pub fn run_loss(&self) -> Option<f64>
    {
        (self.session_batches > 0).then(|| self.run_loss_sum / self.session_batches as f64)
    }

    pub fn finish(&mut self) -> Option<MetricSample> { self.take_sample() }

    fn batch_completed(&mut self, step: Step, loss: f32, learning_rate: f32, elapsed: Duration) -> Observation
    {
        self.window_loss_sum += f64::from(loss);
        self.window_batches += 1;
        self.run_loss_sum += f64::from(loss);
        self.session_batches += 1;

        let batch = step.batch() + 1;
        let mut progress = TrainingProgress {
            superbatch: step.superbatch(),
            final_superbatch: step.final_superbatch(),
            batch,
            batches_per_superbatch: step.batches_per_superbatch(),
            global_batch: step.total_batches() + 1,
            session_batches: self.session_batches,
            loss: Some(self.window_loss_sum / self.window_batches as f64),
            loss_batches: self.window_batches,
            learning_rate,
            elapsed: elapsed.saturating_sub(self.setup),
            positions_per_second: None,
            superbatch_eta: None,
            run_eta: None,
        };
        self.update_rates(&mut progress);
        self.latest = Some(progress);

        let superbatch_boundary = batch == 1 || batch == step.batches_per_superbatch();
        let sample =
            if superbatch_boundary || batch.is_multiple_of(self.log_every) { self.take_sample() } else { None };

        Observation { progress: Some(progress), sample }
    }

    fn update_rates(&self, progress: &mut TrainingProgress)
    {
        let seconds = progress.elapsed.as_secs_f64();
        let batches_per_second = progress.session_batches as f64 / seconds;
        if seconds <= 0.0 || !batches_per_second.is_finite() || batches_per_second <= 0.0 {
            progress.positions_per_second = None;
            progress.superbatch_eta = None;
            progress.run_eta = None;
            return;
        }

        let positions_per_second = batches_per_second * self.steps.batch_size as f64;
        progress.positions_per_second = positions_per_second.is_finite().then_some(positions_per_second);

        let superbatch_remaining = progress.batches_per_superbatch.saturating_sub(progress.batch);
        let run_remaining = progress
            .final_superbatch
            .saturating_sub(progress.superbatch)
            .saturating_mul(progress.batches_per_superbatch)
            .saturating_add(superbatch_remaining);
        progress.superbatch_eta = Duration::try_from_secs_f64(superbatch_remaining as f64 / batches_per_second).ok();
        progress.run_eta = Duration::try_from_secs_f64(run_remaining as f64 / batches_per_second).ok();
    }

    fn take_sample(&mut self) -> Option<MetricSample>
    {
        if self.window_batches == 0 {
            return None;
        }

        let progress = self.latest?;
        let loss = (self.window_loss_sum / self.window_batches as f64) as f32;

        self.window_loss_sum = 0.0;
        self.window_batches = 0;

        Some(MetricSample {
            step: i64::try_from(progress.global_batch).ok()?,
            loss,
            learning_rate: progress.learning_rate,
            speed: (progress.positions_per_second.unwrap_or(0.0) / 1e6) as f32,
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/report_unit.rs"]
mod tests;
