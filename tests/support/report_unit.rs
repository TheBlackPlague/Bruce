use std::time::Duration;

use bullet_trainer::run::{Step, TrainingSteps};

use super::TrainingMetrics;

fn steps() -> TrainingSteps
{
    TrainingSteps { batch_size: 100, batches_per_superbatch: 10, start_superbatch: 3, end_superbatch: 4 }
}

fn batch(metrics: &mut TrainingMetrics, superbatch: usize, batch: usize, loss: f32, seconds: u64)
-> super::Observation
{
    metrics.batch_completed(Step::new(superbatch, batch, 4, 10), loss, 0.002, Duration::from_secs(seconds))
}

#[test]
fn resumed_counts_keep_global_steps_but_rates_and_etas_use_only_this_session()
{
    let mut metrics = TrainingMetrics::new(steps(), 4);
    metrics.setup = Duration::from_secs(10);

    let first = batch(&mut metrics, 3, 0, 0.25, 12);
    let progress = first.progress.unwrap();
    assert_eq!(progress.global_batch, 21);
    assert_eq!(progress.session_batches, 1);
    assert_eq!(progress.elapsed, Duration::from_secs(2));
    assert_eq!(progress.positions_per_second, Some(50.0));
    assert_eq!(progress.superbatch_eta, Some(Duration::from_secs(18)));
    assert_eq!(progress.run_eta, Some(Duration::from_secs(38)));
    assert_eq!(first.sample.unwrap().step, 21);

    let second = batch(&mut metrics, 3, 1, 0.5, 14).progress.unwrap();
    assert_eq!(second.global_batch, 22);
    assert_eq!(second.session_batches, 2);
    assert_eq!(second.positions_per_second, Some(50.0));
    assert_eq!(second.superbatch_eta, Some(Duration::from_secs(16)));
    assert_eq!(second.run_eta, Some(Duration::from_secs(36)));
}

#[test]
fn loss_windows_have_equal_batch_weight_and_flush_partial_window_once()
{
    let mut metrics = TrainingMetrics::new(steps(), 4);
    assert_eq!(batch(&mut metrics, 3, 0, 1.0, 1).sample.unwrap().loss, 1.0);
    assert!(batch(&mut metrics, 3, 1, 2.0, 2).sample.is_none());
    assert!(batch(&mut metrics, 3, 2, 4.0, 3).sample.is_none());
    let fourth = batch(&mut metrics, 3, 3, 6.0, 4);
    assert_eq!(fourth.sample.unwrap().loss, 4.0);
    assert_eq!(fourth.progress.unwrap().loss_batches, 3);
    assert_eq!(metrics.progress().unwrap().loss, Some(4.0));
    assert!(metrics.finish().is_none());
    assert_eq!(metrics.progress().unwrap().loss, Some(4.0));

    assert!(batch(&mut metrics, 3, 4, 8.0, 5).sample.is_none());
    assert!(batch(&mut metrics, 3, 5, 10.0, 6).sample.is_none());
    let partial = metrics.finish().unwrap();
    assert_eq!(partial.step, 26);
    assert_eq!(partial.loss, 9.0);
    assert_eq!(metrics.progress().unwrap().loss_batches, 2);
    assert_eq!(metrics.progress().unwrap().loss, Some(9.0));
    assert!(metrics.finish().is_none());
    assert_eq!(metrics.run_loss(), Some(31.0 / 6.0));
}

#[test]
fn snapshots_are_current_even_between_disk_sample_intervals()
{
    let mut metrics = TrainingMetrics::new(steps(), 8);
    batch(&mut metrics, 3, 0, 1.0, 1);
    let observation = metrics.batch_completed(Step::new(3, 1, 4, 10), 3.0, 0.0005, Duration::from_secs(2));
    assert!(observation.sample.is_none());
    let progress = observation.progress.unwrap();
    assert_eq!(progress.learning_rate, 0.0005);
    assert_eq!(progress.global_batch, 22);
    assert_eq!(progress.loss, Some(3.0));
    assert_eq!(metrics.finish().unwrap().learning_rate, 0.0005);
}

#[test]
fn superbatch_boundaries_sample_without_losing_or_repeating_batches()
{
    let mut metrics = TrainingMetrics::new(steps(), 4);
    let mut samples = Vec::new();
    for superbatch in 3..=4 {
        for index in 0..10 {
            if let Some(sample) =
                batch(&mut metrics, superbatch, index, 0.25, 1 + (superbatch - 3) as u64 * 10 + index as u64).sample
            {
                samples.push(sample.step);
            }
        }
    }
    assert_eq!(samples, [21, 24, 28, 30, 31, 34, 38, 40]);
    let final_progress = metrics.progress().unwrap();
    assert_eq!(final_progress.session_batches, 20);
    assert_eq!(final_progress.global_batch, 40);
    assert_eq!(final_progress.run_eta, Some(Duration::ZERO));
    assert_eq!(final_progress.superbatch_eta, Some(Duration::ZERO));
    assert_eq!(final_progress.loss, Some(0.25));
    assert!(metrics.finish().is_none());
}

#[test]
fn zero_elapsed_has_unknown_rates_and_etas_instead_of_infinite_values()
{
    let mut metrics = TrainingMetrics::new(steps(), 0);
    metrics.setup = Duration::from_secs(2);
    let progress = batch(&mut metrics, 3, 0, 0.5, 1).progress.unwrap();
    assert_eq!(progress.elapsed, Duration::ZERO);
    assert_eq!(progress.positions_per_second, None);
    assert_eq!(progress.superbatch_eta, None);
    assert_eq!(progress.run_eta, None);
    assert!(metrics.finish().is_none());
}

#[test]
fn no_completed_batches_produces_no_metrics_or_final_sample()
{
    let mut metrics = TrainingMetrics::new(steps(), 4);
    assert!(metrics.progress().is_none());
    assert!(metrics.run_loss().is_none());
    assert!(metrics.finish().is_none());
}
