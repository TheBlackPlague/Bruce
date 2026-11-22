use std::{
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use bullet_gpu::runtime::{Device, mock::MockGpu};
use bullet_trainer::{
    reader::{DataReader, ReadMapLoader},
    run::{DataLoader, HostPool, Step, TrainingSteps},
};

use super::{Timed, blend, probe};
use crate::{
    config::Config,
    control::{Control, Outcome, Stop},
    data::stream::TrainingStream,
    events::{Ending, Event, Input, Level, LoaderSample, Progress, ScanInfo, Task},
    telemetry::Hub,
    text,
};

const POLL: Duration = Duration::from_millis(20);
const PROGRESS_MILLIS: u128 = 250;
pub const MEASUREMENTS: [&str; 2] = ["Bruce reader", "Bruce reader and mapper"];

pub fn loader(config: &Config, duration: Duration, control: &Control, hub: &Hub) -> Result<Ending>
{
    let data = &config.training.data;

    hub.message(Level::Detail, "Checking training data");
    probe(data, control)?;

    let inputs =
        MEASUREMENTS.iter().map(|name| Input { path: name.into(), total: duration.as_millis() as u64 }).collect();
    let sampling = format!("{} measurements of {}", MEASUREMENTS.len(), text::span(duration));
    let info = ScanInfo { task: Task::Benchmark, inputs, threads: data.threads.loader, sampling, head: None };
    hub.emit(Event::Scan(info));

    let mut steady = [None; 2];
    let mut shares = None;
    let measurements = MEASUREMENTS.into_iter().enumerate();
    for (index, name) in measurements.take_while(|_| control.stop() == Stop::Running) {
        hub.message(Level::Info, format!("Measuring the {name} for {}", text::span(duration)));
        let stream = TrainingStream::new(data, config.random.seed, None, None)?;

        let progress = |elapsed: Duration, positions| {
            let done = elapsed.min(duration).as_millis() as u64;
            hub.emit(Event::Progress(Progress { input: index, done, games: 0, positions }));
        };
        progress(Duration::ZERO, 0);

        let started = Instant::now();
        let (mut warm, mut last, mut positions) = (None, Duration::ZERO, 0);
        let mut deliver = |count: u64| {
            let elapsed = started.elapsed();
            let ticked = elapsed.as_millis() / PROGRESS_MILLIS > last.as_millis() / PROGRESS_MILLIS;
            warm.get_or_insert((elapsed, count));
            (last, positions) = (elapsed, positions + count);
            if ticked {
                progress(last, positions);
            }
            elapsed >= duration || control.stop() != Stop::Running
        };

        let measured = watched(&stream, control, duration, || match index {
            0 => {
                stream.read_chunks(0, |boards| deliver(boards.len() as u64));
                Ok(())
            },
            _ => map(config, &stream, deliver),
        });
        progress(last, positions);
        stream.failure().map_or(Ok(()), Err)?;
        measured?;

        let rate = |count: u64, time: Duration| match time.as_secs_f64() {
            seconds if seconds > 0.0 => count as f64 / seconds / 1e6,
            _ => 0.0,
        };
        let (at, first) = warm.unwrap_or_default();
        let rates = (rate(positions - first, last - at), rate(positions, last));
        let counted = text::count(positions);
        hub.message(
            Level::Info,
            format!("{name}: {:.2} M positions/s steady, {:.2} M overall, {counted} positions", rates.0, rates.1),
        );

        steady[index] = (positions > 0).then_some(rates.0);
        if index == 1 {
            shares = Some(stream.stats().snapshot());
        }
    }

    let outcome = control.outcome();
    if outcome == Outcome::Aborted {
        return Ok(Ending::of(outcome));
    }

    let measured = steady.iter().any(Option::is_some);
    let [reader, mapped] =
        steady.map(|rate| rate.map_or_else(|| "not measured".to_owned(), |rate| format!("{rate:.2}")));
    let grid =
        [("Mpos/s, steady", "Reader".to_owned()), ("", "Reader + mapper".to_owned()), ("Bruce", reader), ("", mapped)]
            .into_iter()
            .filter(|_| measured)
            .map(|(label, value)| (label.to_owned(), value))
            .collect();

    let rows = shares
        .and_then(|LoaderSample { reading, mapping, blocked }| {
            let total = (reading + mapping + blocked).as_secs_f64();
            let shares = [("reading", reading), ("mapping", mapping), ("blocked", blocked)]
                .map(|(name, time)| format!("{name} {:.1}%", 100.0 * time.as_secs_f64() / total));
            (total > 0.0).then(|| ("Bruce loader thread".to_owned(), shares.join(" · ")))
        })
        .into_iter()
        .collect();

    Ok(Ending { grid, rows, ..Ending::of(outcome) })
}

fn map(config: &Config, stream: &TrainingStream, mut deliver: impl FnMut(u64) -> bool) -> Result<()>
{
    let batch = &config.training.data.batch;
    let mapper = config.run.arch.mapper(blend(config));
    let loader = ReadMapLoader::new(stream.clone(), mapper, config.training.data.threads.mapper);
    let steps = TrainingSteps {
        batch_size: batch.size,
        batches_per_superbatch: batch.per_superbatch,
        start_superbatch: 1,
        end_superbatch: usize::MAX - 1,
    };
    let device =
        Device::<MockGpu>::new(0).map_err(|error| anyhow!("Could not open the host-side batch pool: {error:?}"))?;

    Timed { inner: loader, stats: stream.stats() }
        .map_batches(&HostPool::new(device), Step::from(steps), batch.size, |_| deliver(batch.size as u64))
        .map_err(|error| anyhow!("Could not map the training data: {error:?}"))
}

fn watched<T>(stream: &TrainingStream, control: &Control, duration: Duration, measure: impl FnOnce() -> T) -> T
{
    let (finished, done) = mpsc::channel::<()>();
    let deadline = Instant::now() + duration;

    thread::scope(|scope| {
        scope.spawn(move || {
            while done.recv_timeout(POLL) == Err(RecvTimeoutError::Timeout) {
                if control.stop() != Stop::Running || Instant::now() >= deadline {
                    return stream.finish();
                }
            }
        });

        let measured = measure();
        drop(finished);
        measured
    })
}
