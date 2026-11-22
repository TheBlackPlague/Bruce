use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use serde_json::Value;

use super::checkpoint::{Datasets, LEGACY, Manifest, verify};
use crate::{
    config::{Config, DEFAULT_BATCHES_PER_SUPERBATCH},
    data::{
        SourceIdentity,
        stream::{self, Bookmark, READER_VERSION},
    },
    events::{DataResume, Level, Resumed},
    model::Architecture,
    telemetry::Hub,
};

const LEGACY_RESTARTED: &str =
    "Version-1 checkpoint: resuming weights and schedule; the data restarts at the beginning of an epoch";
const RESTARTED: &str = "Resuming weights and schedule; the data restarts at the beginning of an epoch";
const SOURCES_CHANGED: &str =
    "The data sources changed: resuming weights and schedule; the data restarts at the beginning of an epoch";
const MISFIT: &str = "The saved data position does not fit the data sources: resuming weights and schedule; the data \
                      restarts at the beginning of an epoch";
const APPROXIMATE: &str = "The data selection changed: resuming within one shuffle window of the saved data position";
const UNSTORED: &str = "This checkpoint does not store its configuration; pass --config";

const KNOBS: [(&str, &str); 3] = [("optimizer", "optimizer"), ("learning_rate", "learning rate"), ("wdl", "wdl")];

pub fn stored(checkpoint: &Path) -> Result<Config>
{
    let source = Manifest::read(checkpoint)?.source.ok_or_else(|| anyhow!(UNSTORED))?;

    parsed(&source).map_err(|error| {
        anyhow!("The configuration stored in this checkpoint cannot be used ({error:#}); pass --config")
    })
}

fn parsed(source: &str) -> Result<Config>
{
    let mut table = source.parse::<toml::Table>().unwrap_or_default();
    let training = table.get_mut("training").and_then(toml::Value::as_table_mut);
    match training.and_then(|training| training.remove("validation")) {
        Some(_) => Config::from_toml(&toml::to_string(&table)?),
        None => Config::from_toml(source),
    }
}

pub fn announce(resumed: &Resumed, hub: &Hub)
{
    let Resumed { checkpoint, completed, data, .. } = resumed;
    let from =
        format!("Resumed from {} after superbatch {completed}, data position {}", checkpoint.display(), data.name());
    hub.message(Level::Info, from);
}

pub fn inspect(
    checkpoint: &Path,
    new: &Config,
    identities: &[SourceIdentity],
    hub: &Hub,
) -> Result<(Resumed, Option<Bookmark>)>
{
    let manifest = Manifest::read(checkpoint)?;
    let old = manifest.source.as_deref().and_then(|source| parsed(source).ok());

    let count = |value: &Value| value.as_u64().and_then(|count| usize::try_from(count).ok());
    let saved = &manifest.configuration["data"];
    let (recorded, per_superbatch) = match manifest.steps {
        Some(steps) => (Some(steps.batch_size), Some(steps.batches_per_superbatch)),
        None => (
            count(&saved["batch_size"]),
            saved.get("batches_per_superbatch").map_or(Some(DEFAULT_BATCHES_PER_SUPERBATCH), count),
        ),
    };

    let batch = &new.training.data.batch;
    let architecture = manifest.architecture.parse::<Architecture>().ok();
    ensure!(architecture == Some(new.run.arch), "Architecture differs from the checkpoint");
    ensure!(recorded.is_none_or(|size| size == batch.size), "Batch size differs from the checkpoint");
    ensure!(per_superbatch == Some(batch.per_superbatch), "Batches per superbatch differs from the checkpoint");

    let completed = manifest.completed_superbatch;
    ensure!(completed > 0, "Checkpoint has no completed training");
    ensure!(
        completed < new.training.superbatches,
        "Training is complete at superbatch {completed}; raise superbatches to extend it"
    );

    let stored = old.as_ref().map(Config::checkpoint_value);
    let before = stored.as_ref().unwrap_or(&manifest.configuration);
    let after = new.checkpoint_value();
    let (data, bookmark, notice) = position(&manifest, new, identities);

    let labels = (before["data"]["labels"] != after["data"]["labels"]).then_some("labels");
    let knobs = KNOBS.iter().filter(|(key, _)| before[*key] != after[*key]).map(|&(_, name)| name);
    let changed: Vec<&str> = labels.into_iter().chain(knobs).collect();

    let assumed = recorded
        .is_none()
        .then(|| format!("The checkpoint does not record its batch size; assuming the configuration's {}", batch.size));

    verify(checkpoint, None)?;
    for warning in assumed.into_iter().chain(notice.map(str::to_owned)) {
        hub.message(Level::Warning, warning);
    }
    if !changed.is_empty() {
        let (names, next) = (changed.join(", "), completed + 1);
        let settings = format!("Changed since the checkpoint: {names}; the new settings apply from superbatch {next}");
        hub.message(Level::Info, settings);
    }

    let resumed = Resumed {
        checkpoint: checkpoint.to_owned(),
        completed,
        data,
        run: manifest.configuration["name"].as_str().unwrap_or_default().to_owned(),
        extended: old.is_some_and(|old| old.training.superbatches == completed),
    };
    Ok((resumed, bookmark))
}

fn position(
    manifest: &Manifest,
    new: &Config,
    identities: &[SourceIdentity],
) -> (DataResume, Option<Bookmark>, Option<&'static str>)
{
    let restarted = |notice| (DataResume::Restarted, None, Some(notice));
    if manifest.version == LEGACY {
        return restarted(LEGACY_RESTARTED);
    }
    let Some(record) = &manifest.data else { return restarted(RESTARTED) };

    let sources = identities.len();
    let unmoved = matches!(&manifest.datasets, Datasets::Identified(saved) if saved == identities);
    if !unmoved {
        return restarted(SOURCES_CHANGED);
    }

    let bookmark = &record.bookmark;
    let cursor = &bookmark.start;
    let batch = &new.training.data.batch;
    let emitted = (manifest.completed_superbatch * batch.size * batch.per_superbatch) as u64;
    let fits = bookmark.reader_version == READER_VERSION
        && bookmark.emitted == emitted
        && cursor.offsets.len() == sources
        && cursor.games.len() == sources
        && cursor.offsets.iter().zip(identities).all(|(&offset, identity)| offset <= identity.length);
    if !fits {
        return restarted(MISFIT);
    }

    if bookmark.selection == stream::fingerprints(&new.training.data, new.random.seed, identities).0 {
        (DataResume::Exact, Some(bookmark.clone()), None)
    } else {
        (DataResume::Approximate, Some(bookmark.clone()), Some(APPROXIMATE))
    }
}
