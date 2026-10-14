use std::{collections::BTreeMap, sync::Arc};

use anyhow::{Result, ensure};
use bullet_lib::trainer::schedule::{
    lr::{self, LrScheduler},
    wdl::{self, WdlScheduler},
};
use serde::{
    Deserialize, Serialize, Serializer,
    de::{DeserializeOwned, Error},
};

pub const BATCHES_PER_SUPERBATCH: usize = 6104;

#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct LrSchedule(Arc<Phases<LrSpec>>);

#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct WdlSchedule(Arc<Phases<WdlSpec>>);

#[derive(Clone, Debug)]
struct Phases<T>(Vec<Phase<T>>);

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Phase<T>
{
    superbatch_range: [usize; 2],
    #[serde(flatten)]
    schedule: T,
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for Phase<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error>
    {
        let mut fields = serde_json::Value::deserialize(deserializer)?;
        let range = fields
            .as_object_mut()
            .and_then(|fields| fields.remove("superbatch_range"))
            .ok_or_else(|| D::Error::missing_field("superbatch_range"))?;

        Ok(Self {
            superbatch_range: serde_json::from_value(range).map_err(D::Error::custom)?,
            schedule: serde_json::from_value(fields).map_err(D::Error::custom)?,
        })
    }
}

impl<T: Serialize> Serialize for Phases<T>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    {
        serializer.collect_map(self.0.iter().enumerate().map(|(index, phase)| (index.to_string(), phase)))
    }
}

impl<T> Phases<T>
{
    fn new(
        entries: BTreeMap<String, Phase<T>>,
        duration: usize,
        label: &str,
        validate: impl Fn(&T, usize) -> Result<()>,
    ) -> Result<Self>
    {
        let entries = indexed(entries, label)?;
        ensure!(entries.len() <= 100, "{label} supports indexes from 0 to 99");

        let mut next = 1;
        let mut phases = Vec::with_capacity(entries.len());

        for phase in entries {
            let [start, end] = phase.superbatch_range;
            ensure!(
                start == next && end >= start && end <= duration,
                "{label} ranges must cover training without gaps or overlaps"
            );

            next = end.checked_add(1).ok_or_else(|| anyhow::anyhow!("Training is too long"))?;
            validate(&phase.schedule, end - start + 1)?;

            phases.push(phase);
        }

        ensure!(duration.checked_add(1) == Some(next), "{label} ranges must cover every superbatch");

        Ok(Self(phases))
    }

    fn at(&self, superbatch: usize) -> (&T, usize, usize)
    {
        let index = self.0.partition_point(|phase| phase.superbatch_range[1] < superbatch).min(self.0.len() - 1);
        let phase = &self.0[index];

        let [start, end] = phase.superbatch_range;
        (&phase.schedule, superbatch.clamp(start, end) - start + 1, end - start + 1)
    }
}

pub(crate) fn indexed<T>(entries: BTreeMap<String, T>, label: &str) -> Result<Vec<T>>
{
    ensure!(!entries.is_empty(), "{label} needs at least one entry");
    let mut ordered = BTreeMap::new();

    for (key, entry) in entries {
        let index =
            key.parse::<usize>().map_err(|_| anyhow::anyhow!("{label} indexes must be nonnegative whole numbers"))?;
        ensure!(index.to_string() == key, "{label} indexes must be nonnegative whole numbers");

        ordered.insert(index, entry);
    }

    ensure!(ordered.keys().copied().eq(0..ordered.len()), "{label} indexes must start at 0 without gaps");
    Ok(ordered.into_values().collect())
}

impl LrSchedule
{
    pub(crate) fn new(entries: BTreeMap<String, Phase<LrSpec>>, duration: usize) -> Result<Self>
    {
        Ok(Self(Arc::new(Phases::new(entries, duration, "Learning rate", LrSpec::validate)?)))
    }
}

impl WdlSchedule
{
    pub(crate) fn new(entries: BTreeMap<String, Phase<WdlSpec>>, duration: usize) -> Result<Self>
    {
        Ok(Self(Arc::new(Phases::new(entries, duration, "WDL", WdlSpec::validate)?)))
    }
}

impl LrScheduler for LrSchedule
{
    fn lr(&self, batch: usize, superbatch: usize) -> f32
    {
        let (spec, local, duration) = self.0.at(superbatch);
        spec.value(batch, local, duration)
    }

    fn colourful(&self) -> String { "configured learning rate".into() }
}

impl WdlScheduler for WdlSchedule
{
    fn blend(&self, batch: usize, superbatch: usize, _: usize) -> f32
    {
        let (spec, local, duration) = self.0.at(superbatch);
        spec.value(batch, local, duration)
    }

    fn colourful(&self) -> String { "configured WDL blend".into() }
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

impl LrSpec
{
    fn value(&self, batch: usize, superbatch: usize, duration: usize) -> f32
    {
        match self {
            Self::Constant { constant } => lr::ConstantLR { value: constant.value }.lr(batch, superbatch),
            Self::Step { step } => {
                lr::StepLR { start: step.start, gamma: step.gamma, step: step.size }.lr(batch, superbatch)
            },
            Self::Linear { linear } => lr::LinearDecayLR {
                initial_lr: linear.initial,
                final_lr: linear.final_value,
                final_superbatch: linear.final_superbatch.unwrap_or(duration),
            }
            .lr(batch, superbatch),
            Self::Cosine { cosine } => lr::CosineDecayLR {
                initial_lr: cosine.initial,
                final_lr: cosine.final_value,
                final_superbatch: cosine.final_superbatch.unwrap_or(duration),
            }
            .lr(batch, superbatch),
            Self::Exponential { exponential } => lr::ExponentialDecayLR {
                initial_lr: exponential.initial,
                final_lr: exponential.final_value,
                final_superbatch: exponential.final_superbatch.unwrap_or(duration),
            }
            .lr(batch, superbatch),
            Self::Warmup { warmup } => warmup.value(warmup.inner.value(batch, superbatch, duration), batch, superbatch),
            Self::Sequence { sequence } => {
                let (spec, local, length) = sequence.at(superbatch, duration);
                spec.value(batch, local, length)
            },
        }
    }
}

impl WdlSpec
{
    fn value(&self, batch: usize, superbatch: usize, duration: usize) -> f32
    {
        match self {
            Self::Constant { constant } => {
                wdl::ConstantWDL { value: constant.value }.blend(batch, superbatch, duration)
            },
            Self::Linear { linear } => {
                wdl::LinearWDL { start: linear.start, end: linear.end }.blend(batch, superbatch, duration)
            },
            Self::Warmup { warmup } => warmup.value(warmup.inner.value(batch, superbatch, duration), batch, superbatch),
            Self::Sequence { sequence } => {
                let (spec, local, length) = sequence.at(superbatch, duration);
                spec.value(batch, local, length)
            },
        }
    }
}

fn positive_rate(value: f32) -> Result<()>
{
    ensure!(value.is_finite() && value > 0.0, "Learning rates must be positive and finite");
    Ok(())
}

fn blend_value(value: f32) -> Result<()>
{
    ensure!(value.is_finite() && (0.0..=1.0).contains(&value), "WDL values must be between 0 and 1");
    Ok(())
}

impl<T> Warmup<T>
{
    fn value(&self, base: f32, batch: usize, superbatch: usize) -> f32
    {
        if superbatch == 1 && batch < self.batches { base / (self.batches - batch) as f32 } else { base }
    }

    fn validate(&self) -> Result<()>
    {
        ensure!((1..=BATCHES_PER_SUPERBATCH).contains(&self.batches), "Warmup must fit in the first superbatch");
        Ok(())
    }
}

impl<T> Sequence<T>
{
    fn at(&self, superbatch: usize, duration: usize) -> (&T, usize, usize)
    {
        if superbatch <= self.switch_after {
            (&self.first, superbatch, self.switch_after)
        } else {
            (&self.second, superbatch - self.switch_after, duration - self.switch_after)
        }
    }

    fn validate(&self, duration: usize) -> Result<()>
    {
        ensure!(self.switch_after > 0 && self.switch_after < duration, "Schedule switch must fall inside its phase");
        Ok(())
    }
}

impl Decay
{
    fn validate(&self, duration: usize) -> Result<()>
    {
        positive_rate(self.initial)?;
        positive_rate(self.final_value)?;

        ensure!(self.final_superbatch.unwrap_or(duration) > 0, "Decay duration must be positive");

        Ok(())
    }
}

impl LrSpec
{
    fn validate(&self, duration: usize) -> Result<()>
    {
        match self {
            Self::Constant { constant } => positive_rate(constant.value),
            Self::Step { step } => {
                positive_rate(step.start)?;
                ensure!(
                    step.gamma.is_finite() && step.gamma > 0.0 && step.gamma <= 1.0 && step.size > 0,
                    "Step needs a positive size and gamma in (0, 1]"
                );
                ensure!((duration - 1) / step.size <= i32::MAX as usize, "Step schedule is too long");
                positive_rate(lr::StepLR { start: step.start, gamma: step.gamma, step: step.size }.lr(0, duration))
            },
            Self::Linear { linear } => linear.validate(duration),
            Self::Cosine { cosine } => cosine.validate(duration),
            Self::Exponential { exponential } => {
                exponential.validate(duration)?;
                positive_rate(exponential.final_value / exponential.initial)
            },
            Self::Warmup { warmup } => {
                warmup.validate()?;
                warmup.inner.validate(duration)?;
                positive_rate(self.value(0, 1, duration))
            },
            Self::Sequence { sequence } => {
                sequence.validate(duration)?;
                sequence.first.validate(sequence.switch_after)?;
                sequence.second.validate(duration - sequence.switch_after)
            },
        }
    }
}

impl WdlSpec
{
    fn validate(&self, duration: usize) -> Result<()>
    {
        match self {
            Self::Constant { constant } => blend_value(constant.value),
            Self::Linear { linear } => {
                blend_value(linear.start)?;
                blend_value(linear.end)
            },
            Self::Warmup { warmup } => {
                warmup.validate()?;
                warmup.inner.validate(duration)
            },
            Self::Sequence { sequence } => {
                sequence.validate(duration)?;
                sequence.first.validate(sequence.switch_after)?;
                sequence.second.validate(duration - sequence.switch_after)
            },
        }
    }
}
