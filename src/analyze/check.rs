use std::{num::NonZero, ops::ControlFlow, path::Path, thread, time::Instant};

use anyhow::{Result, bail};
use clap::ValueEnum;

use super::PREVIEW;
use crate::{
    config::Config,
    control::{Control, Outcome, Stop},
    data::{
        Source,
        label::{AgreementAction, LabelAction},
        pipeline::{self, BLOCK, Sampling, ScanOptions},
        select::{Counters, Selection},
    },
    events::{Ending, Event, Level, Progress, ScanInfo, Task},
    telemetry::Hub,
    text,
    train::checkpoint,
};

const QUICK_GAMES: u64 = 200_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CheckMode
{
    Quick,
    Full,
}

pub fn run(config: &Config, path: &Path, mode: CheckMode, control: &Control, hub: &Hub) -> Result<Ending>
{
    let started = Instant::now();
    hub.message(Level::Info, "Configuration is valid");

    let training = &config.training;
    hub.message(Level::Detail, "Checking checkpoint destinations");
    checkpoint::check_destinations(config, 1)?;

    let data = &training.data;
    let sources = &data.sources;
    let inputs: Vec<_> = sources.iter().map(Source::input).collect();
    let bytes: u64 = inputs.iter().map(|input| input.total).sum();
    let threads = thread::available_parallelism().map_or(1, NonZero::get);
    let (scope, sampling) = match mode {
        CheckMode::Quick => (
            format!("quick: the first {} games of each source", text::count(QUICK_GAMES)),
            Sampling::Head { games: QUICK_GAMES },
        ),
        CheckMode::Full => ("full: every game of each source".to_owned(), Sampling::All),
    };
    hub.emit(Event::Scan(ScanInfo { task: Task::Check, inputs, threads, sampling: scope, head: sampling.head() }));

    let selection = Selection { filter: data.filter.clone(), labels: data.labels, ..Selection::default() };
    let selections =
        sources.iter().map(|source| selection.clone().identified(source)).collect::<Result<Vec<_>, _>>()?;
    let options = ScanOptions { sampling, threads, in_flight: None, block: BLOCK };
    let mut seen: Vec<(Counters, u64, u64)> = vec![(Counters::default(), 0, 0); sources.len()];
    let (mut finished, mut since, mut previewed) = (0, Instant::now(), None::<Instant>);
    let counts = |(counters, ..): &(Counters, u64, u64)| (counters.games, counters.plies);
    let stopped = pipeline::scan(
        sources,
        &options,
        control,
        |scratch, block| -> Result<(Counters, u64)> {
            let selection = &selections[block.source];
            let (mut counters, mut skips) = (Counters::default(), 0);
            for game in block.games {
                selection.select_each(game, Some(&mut counters), scratch, |_, _| {})?;
                skips += selection.agreement_skips(game, scratch)?;
            }
            Ok((counters, skips))
        },
        |info, (counters, skips)| {
            hub.finished_inputs(seen[..info.source].iter().map(counts), &mut finished, &mut since);
            let (total, skipped, read) = &mut seen[info.source];
            *total += counters;
            *skipped += skips;
            *read = info.end;
            let progress = Progress { input: info.source, done: info.end, games: total.games, positions: total.plies };
            hub.emit(Event::Progress(progress));
            if previewed.is_none_or(|previewed| previewed.elapsed() >= PREVIEW) {
                let (accepted, positions) = share_of(&seen);
                hub.emit(Event::Preview(vec![("Accepted".to_owned(), text::percent(accepted, positions))]));
                previewed = Some(Instant::now());
            }
            Ok(ControlFlow::Continue(()))
        },
    )?;

    let complete = !stopped;
    if complete {
        hub.finished_inputs(seen.iter().map(counts), &mut finished, &mut since);
        warn(config, path, &seen, hub);
    }

    let empty = seen.iter().position(|(counters, ..)| counters.accepted == 0).filter(|_| complete);
    if let Some(empty) = empty
        && control.stop() == Stop::Running
    {
        let empty = sources[empty].path.display();
        match mode {
            CheckMode::Quick => bail!(
                "No sampled position in {empty} passes the filter; loosen the filter, or run train --check full to \
                 look past the sampled games"
            ),
            CheckMode::Full => bail!(
                "No positions in {empty} pass the filter; loosen the filter, or check that the file holds Monty data"
            ),
        }
    }

    let outcome = if complete && empty.is_none() { Outcome::Finished } else { control.outcome() };
    if outcome == Outcome::Aborted {
        return Ok(Ending::of(outcome));
    }

    let (accepted, positions) = share_of(&seen);
    let plies: u64 = seen.iter().map(|(counters, ..)| counters.plies).sum();
    let read: u64 = seen.iter().map(|(.., read)| read).sum();
    let share = text::percent(accepted, positions);
    let what = match (complete, mode) {
        (false, _) => "positions scanned",
        (true, CheckMode::Quick) => "sampled positions",
        (true, CheckMode::Full) => "positions",
    };
    let scanned = (share, format!(" of {} {what}", text::scaled(positions as f64)));

    let training = (training.superbatches as u64)
        .saturating_mul(data.batch.per_superbatch as u64)
        .saturating_mul(data.batch.size as u64) as f64;
    let exact = (complete && mode == CheckMode::Full && accepted > 0).then(|| training / accepted as f64);
    let estimate = (plies > 0)
        .then(|| read as f64 / plies as f64)
        .map(|per_position| bytes as f64 / per_position * accepted as f64 / positions.max(1) as f64)
        .filter(|accepted| *accepted > 0.0)
        .map(|accepted| training / accepted);
    let passes = match (exact, estimate) {
        (Some(passes), _) => (format!("{passes:.1}"), format!(" over {} training positions", text::scaled(training))),
        (None, Some(passes)) if complete => (format!("about {passes:.1}"), ", estimated from the sample".to_owned()),
        (None, _) => ("needs a complete scan".to_owned(), String::new()),
    };

    let elapsed = (text::span(started.elapsed()), String::new());
    let rows = [("Accepted", scanned), ("Passes", passes), ("Elapsed", elapsed)];
    let rows = rows.map(|(label, (value, qualifier))| (label.to_owned(), value, qualifier));
    Ok(Ending { rows: rows.into(), ..Ending::of(outcome) })
}

fn share_of(seen: &[(Counters, u64, u64)]) -> (u64, u64)
{
    seen.iter().fold((0, 0), |(accepted, positions), (counters, ..)| {
        (accepted + counters.accepted, positions + counters.plies)
    })
}

fn warn(config: &Config, path: &Path, seen: &[(Counters, u64, u64)], hub: &Hub)
{
    let training = &config.training;
    let labels = training.data.labels;
    let superbatches = training.superbatches;
    let closing = training.data.batch.per_superbatch.saturating_sub(1);
    let wdl = training.wdl.curve();
    let zero =
        (1..=superbatches).filter(|&superbatch| wdl(0, superbatch) == 0.0 && wdl(closing, superbatch) == 0.0).count();

    let (plies, dropped) = seen.iter().fold((0, 0), |(plies, dropped), (counters, skips, _)| {
        (plies + counters.plies, dropped + counters.dropped_by_label - skips)
    });
    if labels.action == LabelAction::DropBefore && plies > 0 && zero > 0 {
        let dropped = 100.0 * (dropped as f64 / plies as f64);
        let warning = if zero == superbatches {
            "drop_before only deletes data: the WDL blend is 0 in every superbatch".to_owned()
        } else {
            format!(
                "drop_before removes {dropped:.2}% of positions, also in {zero} of {superbatches} superbatches with \
                 WDL blend 0"
            )
        };
        hub.message(Level::Warning, warning);
    }

    let skipping = labels.agreement.is_some_and(|agreement| agreement.action == AgreementAction::Skip);
    if skipping && zero < superbatches {
        hub.message(Level::Warning, format!("{}: the blended target leans towards the score", path.display()));
    }
}
