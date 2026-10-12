use std::{collections::BTreeMap, fmt, sync::Arc};

use anyhow::{Result, ensure};
use bullet_lib::trainer::schedule::{
    lr::{self, LrScheduler},
    wdl::{self, WdlScheduler},
};
use serde::{
    Deserialize, Serialize,
    de::{DeserializeOwned, Error},
};

pub const BATCHES_PER_SUPERBATCH: usize = 6104;

type LrFunction = dyn Fn(usize, usize) -> f32 + Send + Sync;
type WdlFunction = dyn Fn(usize, usize, usize) -> f32 + Send + Sync;

#[derive(Clone)]
pub struct LrSchedule(Arc<LrFunction>);

#[derive(Clone)]
pub struct WdlSchedule(Arc<WdlFunction>);

impl fmt::Debug for LrSchedule
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("LrSchedule") }
}

impl fmt::Debug for WdlSchedule
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("WdlSchedule") }
}

impl LrSchedule
{
    fn new(scheduler: impl LrScheduler + Send + Sync) -> Self
    {
        Self(Arc::new(move |batch, superbatch| scheduler.lr(batch, superbatch)))
    }
}

impl WdlSchedule
{
    fn new(scheduler: impl WdlScheduler) -> Self
    {
        Self(Arc::new(move |batch, superbatch, max| scheduler.blend(batch, superbatch, max)))
    }
}

impl LrScheduler for LrSchedule
{
    fn lr(&self, batch: usize, superbatch: usize) -> f32 { self.0(batch, superbatch) }

    fn colourful(&self) -> String { "configured learning rate".into() }
}

impl WdlScheduler for WdlSchedule
{
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 { self.0(batch, superbatch, max) }

    fn colourful(&self) -> String { "configured WDL blend".into() }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Phase<T>
{
    pub superbatch_range: [usize; 2],
    #[serde(flatten)]
    pub schedule: T,
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for Phase<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error>
    {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        let range = value
            .as_object_mut()
            .and_then(|object| object.remove("superbatch_range"))
            .ok_or_else(|| D::Error::missing_field("superbatch_range"))?;
        
        let superbatch_range = serde_json::from_value(range).map_err(D::Error::custom)?;
        let schedule = serde_json::from_value(value).map_err(D::Error::custom)?;
        
        Ok(Self { superbatch_range, schedule })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum LrSpec
{
    Constant
    {
        constant: Constant
    },
    Step
    {
        step: Step
    },
    Linear
    {
        linear: Decay
    },
    Cosine
    {
        cosine: Decay
    },
    Exponential
    {
        exponential: Decay
    },
    Warmup
    {
        warmup: Warmup<LrSpec>
    },
    Sequence
    {
        sequence: Sequence<LrSpec>
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WdlSpec
{
    Constant
    {
        constant: Constant
    },
    Linear
    {
        linear: Linear
    },
    Warmup
    {
        warmup: Warmup<WdlSpec>
    },
    Sequence
    {
        sequence: Sequence<WdlSpec>
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Constant
{
    value: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Step
{
    start: f32,
    gamma: f32,
    size: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Decay
{
    initial: f32,
    final_value: f32,
    #[serde(default)]
    final_superbatch: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Linear
{
    start: f32,
    end: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Warmup<T>
{
    batches: usize,
    inner: Box<T>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Sequence<T>
{
    switch_after: usize,
    first: Box<T>,
    second: Box<T>,
}

pub(crate) fn indexed<T>(entries: BTreeMap<String, T>, label: &str) -> Result<Vec<T>>
{
    ensure!(!entries.is_empty(), "{label} needs at least one entry");
    let mut result = BTreeMap::new();

    for (key, entry) in entries {
        let index = key.parse::<usize>().ok();
        ensure!(
            index.is_some_and(|index| index.to_string() == key),
            "{label} indexes must be nonnegative whole numbers"
        );
        result.insert(index.expect("checked index"), entry);
    }

    ensure!(result.keys().copied().eq(0..result.len()), "{label} indexes must start at 0 without gaps");
    Ok(result.into_values().collect())
}

fn phases<T>(entries: BTreeMap<String, Phase<T>>, duration: usize, label: &str) -> Result<Vec<Phase<T>>>
{
    let entries = indexed(entries, label)?;
    ensure!(entries.len() <= 100, "{label} supports indexes from 0 to 99");
    
    let mut next = 1;
    for phase in &entries {
        let [start, end] = phase.superbatch_range;
        
        ensure!(
            start == next && end >= start && end <= duration,
            "{label} ranges must cover training without gaps or overlaps"
        );
        next = end.checked_add(1).ok_or_else(|| anyhow::anyhow!("Training is too long"))?;
    }

    ensure!(next == duration + 1, "{label} ranges must cover every superbatch");
    Ok(entries)
}

pub(crate) fn learning_rate(entries: BTreeMap<String, Phase<LrSpec>>, duration: usize) -> Result<LrSchedule>
{
    let phases = phases(entries, duration, "Learning rate")?
        .into_iter()
        .map(|phase| {
            let [start, end] = phase.superbatch_range;
            Ok((start, end, phase.schedule.build(end - start + 1)?))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LrSchedule(Arc::new(move |batch, superbatch| {
        let index = phases.partition_point(|(_, end, _)| *end < superbatch).min(phases.len() - 1);
        let (start, end, schedule) = &phases[index];
        schedule.lr(batch, superbatch.clamp(*start, *end) - start + 1)
    })))
}

pub(crate) fn wdl(entries: BTreeMap<String, Phase<WdlSpec>>, duration: usize) -> Result<WdlSchedule>
{
    let phases = phases(entries, duration, "WDL")?
        .into_iter()
        .map(|phase| {
            let [start, end] = phase.superbatch_range;
            Ok((start, end, phase.schedule.build(end - start + 1)?))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(WdlSchedule(Arc::new(move |batch, superbatch, _| {
        let index = phases.partition_point(|(_, end, _)| *end < superbatch).min(phases.len() - 1);
        let (start, end, schedule) = &phases[index];
        schedule.blend(batch, superbatch.clamp(*start, *end) - start + 1, end - start + 1)
    })))
}

fn positive(value: f32) -> Result<()>
{
    ensure!(value.is_finite() && value > 0.0, "Learning rates must be positive and finite");
    Ok(())
}

fn proportion(value: f32) -> Result<()>
{
    ensure!(value.is_finite() && (0.0..=1.0).contains(&value), "WDL values must be between 0 and 1");
    Ok(())
}

fn warmup(batches: usize) -> Result<()>
{
    ensure!((1..=BATCHES_PER_SUPERBATCH).contains(&batches), "Warmup must fit in the first superbatch");
    Ok(())
}

fn split(midpoint: usize, duration: usize) -> Result<()>
{
    ensure!(midpoint > 0 && midpoint < duration, "Schedule switch must fall inside its phase");
    Ok(())
}

impl LrSpec
{
    fn build(self, duration: usize) -> Result<LrSchedule>
    {
        Ok(match self {
            Self::Constant { constant } => {
                positive(constant.value)?;
                LrSchedule::new(lr::ConstantLR { value: constant.value })
            },
            Self::Step { step } => {
                positive(step.start)?;
                ensure!(
                    step.gamma.is_finite() && step.gamma > 0.0 && step.gamma <= 1.0 && step.size > 0,
                    "Step needs a positive size and gamma in (0, 1]"
                );
                ensure!((duration - 1) / step.size <= i32::MAX as usize, "Step schedule is too long");
                
                let scheduler = lr::StepLR { start: step.start, gamma: step.gamma, step: step.size };
                positive(scheduler.lr(0, duration))?;
                
                LrSchedule::new(scheduler)
            },
            Self::Linear { linear: decay } => {
                let end = decay.validate(duration)?;
                LrSchedule::new(lr::LinearDecayLR {
                    initial_lr: decay.initial,
                    final_lr: decay.final_value,
                    final_superbatch: end,
                })
            },
            Self::Cosine { cosine: decay } => {
                let end = decay.validate(duration)?;
                LrSchedule::new(lr::CosineDecayLR {
                    initial_lr: decay.initial,
                    final_lr: decay.final_value,
                    final_superbatch: end,
                })
            },
            Self::Exponential { exponential: decay } => {
                let end = decay.validate(duration)?;
                positive(decay.final_value / decay.initial)?;
                LrSchedule::new(lr::ExponentialDecayLR {
                    initial_lr: decay.initial,
                    final_lr: decay.final_value,
                    final_superbatch: end,
                })
            },
            Self::Warmup { warmup: spec } => {
                warmup(spec.batches)?;
                let scheduler = lr::Warmup { inner: spec.inner.build(duration)?, warmup_batches: spec.batches };
                positive(scheduler.lr(0, 1))?;
                LrSchedule::new(scheduler)
            },
            Self::Sequence { sequence } => {
                split(sequence.switch_after, duration)?;
                LrSchedule::new(lr::Sequence {
                    first: sequence.first.build(sequence.switch_after)?,
                    second: sequence.second.build(duration - sequence.switch_after)?,
                    first_scheduler_final_superbatch: sequence.switch_after,
                })
            },
        })
    }
}

impl Decay
{
    fn validate(&self, duration: usize) -> Result<usize>
    {
        positive(self.initial)?;
        positive(self.final_value)?;
        let end = self.final_superbatch.unwrap_or(duration);
        ensure!(end > 0, "Decay duration must be positive");
        Ok(end)
    }
}

impl WdlSpec
{
    fn build(self, duration: usize) -> Result<WdlSchedule>
    {
        Ok(match self {
            Self::Constant { constant } => {
                proportion(constant.value)?;
                WdlSchedule::new(wdl::ConstantWDL { value: constant.value })
            },
            Self::Linear { linear } => {
                proportion(linear.start)?;
                proportion(linear.end)?;
                WdlSchedule::new(wdl::LinearWDL { start: linear.start, end: linear.end })
            },
            Self::Warmup { warmup: spec } => {
                warmup(spec.batches)?;
                WdlSchedule::new(wdl::Warmup { inner: spec.inner.build(duration)?, warmup_batches: spec.batches })
            },
            Self::Sequence { sequence } => {
                split(sequence.switch_after, duration)?;
                WdlSchedule::new(wdl::Sequence {
                    first: sequence.first.build(sequence.switch_after)?,
                    second: sequence.second.build(duration - sequence.switch_after)?,
                    first_scheduler_final_superbatch: sequence.switch_after,
                })
            },
        })
    }
}
