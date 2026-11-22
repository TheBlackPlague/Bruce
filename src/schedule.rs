use std::{collections::BTreeMap, fmt, iter, sync::Arc};

use anyhow::{Result, anyhow, ensure};
use bullet_lib::trainer::schedule::{
    lr::{self, LrScheduler},
    wdl::{self, WdlScheduler},
};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{DeserializeOwned, Error},
};

pub type Curve = Arc<dyn Fn(usize, usize) -> f32 + Send + Sync>;
pub type LrSchedule = Schedule<LrSpec>;
pub type WdlSchedule = Schedule<WdlSpec>;

#[derive(Clone, Debug)]
pub struct Schedule<T>(Vec<Phase<T>>);

#[derive(Clone, Debug, Serialize)]
pub struct Phase<T>
{
    superbatch_range: [usize; 2],
    #[serde(flatten)]
    schedule: T,
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for Phase<T>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>
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

impl<'de, T: Spec> Deserialize<'de> for Schedule<T>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>
    {
        indexed(BTreeMap::deserialize(deserializer)?, T::LABEL).map(Self).map_err(D::Error::custom)
    }
}

impl<T: Serialize> Serialize for Schedule<T>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>
    {
        serializer.collect_map(self.0.iter().enumerate().map(|(index, phase)| (index.to_string(), phase)))
    }
}

pub fn indexed<T>(entries: BTreeMap<String, T>, label: &str) -> Result<Vec<T>>
{
    ensure!(!entries.is_empty(), "{label} needs at least one entry");
    let mut ordered = BTreeMap::new();

    for (key, entry) in entries {
        let index = key.parse::<usize>().map_err(|_| anyhow!("{label} indexes must be nonnegative whole numbers"))?;
        ensure!(index.to_string() == key, "{label} indexes must be nonnegative whole numbers");

        ordered.insert(index, entry);
    }

    ensure!(ordered.keys().copied().eq(0..ordered.len()), "{label} indexes must start at 0 without gaps");
    Ok(ordered.into_values().collect())
}

impl<T: Spec> Schedule<T>
{
    pub fn validate(&self, superbatches: usize, batches_per_superbatch: usize) -> Result<()>
    {
        let label = T::LABEL;
        ensure!(self.0.len() <= 100, "{label} supports indexes from 0 to 99");

        let mut next = 1;
        for phase in &self.0 {
            let [start, end] = phase.superbatch_range;
            ensure!(
                start == next && end >= start && end <= superbatches,
                "{label} ranges must cover training without gaps or overlaps"
            );

            next = end.checked_add(1).ok_or_else(|| anyhow::anyhow!("Training is too long"))?;
            phase.schedule.validate(end - start + 1, batches_per_superbatch)?;
        }

        ensure!(superbatches.checked_add(1) == Some(next), "{label} ranges must cover every superbatch");

        Ok(())
    }

    #[expect(clippy::expect_used, reason = "a schedule deserialises only with at least one phase")]
    pub fn curve(&self) -> Curve
    {
        let phases = self.0.iter().rev().map(|phase| {
            let [start, end] = phase.superbatch_range;
            (phase.schedule.build(end - start + 1), end - start + 1)
        });
        let (Built(curve), _) = phases
            .reduce(|(second, _), (first, length)| (join(first, second, length), length))
            .expect("a schedule has at least one phase");
        curve
    }

    pub fn starts(&self) -> impl Iterator<Item = usize> + '_ { self.0.iter().map(|phase| phase.superbatch_range[0]) }
}

impl Schedule<WdlSpec>
{
    pub fn table(&self, superbatches: usize) -> Option<Vec<f32>>
    {
        if self.0.iter().any(|phase| phase.schedule.warms_up()) {
            return None;
        }

        let blend = self.curve();
        Some(iter::once(0.0).chain((1..=superbatches).map(|superbatch| blend(0, superbatch))).collect())
    }
}

pub trait Spec: DeserializeOwned
{
    const LABEL: &'static str;

    fn validate(&self, duration: usize, batches: usize) -> Result<()>;

    fn build(&self, duration: usize) -> Built;
}

#[derive(Clone)]
pub struct Built(Curve);

fn join(first: Built, second: Built, switch_after: usize) -> Built
{
    Built::from_lr(lr::Sequence { first, second, first_scheduler_final_superbatch: switch_after })
}

impl Built
{
    fn from_lr(scheduler: impl LrScheduler + Send + Sync) -> Self
    {
        Self(Arc::new(move |batch, superbatch| scheduler.lr(batch, superbatch)))
    }

    fn from_wdl(scheduler: impl WdlScheduler, duration: usize) -> Self
    {
        Self(Arc::new(move |batch, superbatch| scheduler.blend(batch, superbatch, duration)))
    }
}

impl LrScheduler for Built
{
    fn lr(&self, batch: usize, superbatch: usize) -> f32 { (self.0)(batch, superbatch) }

    fn colourful(&self) -> String { String::new() }
}

impl WdlScheduler for Built
{
    fn blend(&self, batch: usize, superbatch: usize, _: usize) -> f32 { (self.0)(batch, superbatch) }

    fn colourful(&self) -> String { String::new() }
}

impl fmt::Debug for Built
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result { formatter.write_str("Built") }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LrSpec
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
pub enum WdlSpec
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
pub struct Constant
{
    value: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Step
{
    start: f32,
    gamma: f32,
    size: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Decay
{
    initial: f32,
    final_value: f32,
    #[serde(default)]
    final_superbatch: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Linear
{
    start: f32,
    end: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Warmup<T>
{
    batches: usize,
    inner: Box<T>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Sequence<T>
{
    switch_after: usize,
    first: Box<T>,
    second: Box<T>,
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

impl<T: Spec> Warmup<T>
{
    fn validate(&self, duration: usize, batches_per_superbatch: usize) -> Result<()>
    {
        ensure!(
            (1..=batches_per_superbatch).contains(&self.batches),
            "Warmup must fit in the first superbatch of {batches_per_superbatch} batches"
        );
        self.inner.validate(duration, batches_per_superbatch)
    }
}

impl<T: Spec> Sequence<T>
{
    fn build(&self, duration: usize) -> Built
    {
        join(self.first.build(self.switch_after), self.second.build(duration - self.switch_after), self.switch_after)
    }

    fn validate(&self, duration: usize, batches_per_superbatch: usize) -> Result<()>
    {
        ensure!(self.switch_after > 0 && self.switch_after < duration, "Schedule switch must fall inside its phase");
        self.first.validate(self.switch_after, batches_per_superbatch)?;
        self.second.validate(duration - self.switch_after, batches_per_superbatch)
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

impl Spec for LrSpec
{
    const LABEL: &'static str = "Learning rate";

    fn validate(&self, duration: usize, batches_per_superbatch: usize) -> Result<()>
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
                warmup.validate(duration, batches_per_superbatch)?;
                positive_rate(self.build(duration).lr(0, 1))
            },
            Self::Sequence { sequence } => sequence.validate(duration, batches_per_superbatch),
        }
    }

    fn build(&self, duration: usize) -> Built
    {
        match self {
            Self::Constant { constant } => Built::from_lr(lr::ConstantLR { value: constant.value }),
            Self::Step { step } => Built::from_lr(lr::StepLR { start: step.start, gamma: step.gamma, step: step.size }),
            Self::Linear { linear } => Built::from_lr(lr::LinearDecayLR {
                initial_lr: linear.initial,
                final_lr: linear.final_value,
                final_superbatch: linear.final_superbatch.unwrap_or(duration),
            }),
            Self::Cosine { cosine } => Built::from_lr(lr::CosineDecayLR {
                initial_lr: cosine.initial,
                final_lr: cosine.final_value,
                final_superbatch: cosine.final_superbatch.unwrap_or(duration),
            }),
            Self::Exponential { exponential } => Built::from_lr(lr::ExponentialDecayLR {
                initial_lr: exponential.initial,
                final_lr: exponential.final_value,
                final_superbatch: exponential.final_superbatch.unwrap_or(duration),
            }),
            Self::Warmup { warmup } => {
                Built::from_lr(lr::Warmup { inner: warmup.inner.build(duration), warmup_batches: warmup.batches })
            },
            Self::Sequence { sequence } => sequence.build(duration),
        }
    }
}

impl Spec for WdlSpec
{
    const LABEL: &'static str = "WDL";

    fn validate(&self, duration: usize, batches_per_superbatch: usize) -> Result<()>
    {
        match self {
            Self::Constant { constant } => blend_value(constant.value),
            Self::Linear { linear } => {
                blend_value(linear.start)?;
                blend_value(linear.end)
            },
            Self::Warmup { warmup } => warmup.validate(duration, batches_per_superbatch),
            Self::Sequence { sequence } => sequence.validate(duration, batches_per_superbatch),
        }
    }

    fn build(&self, duration: usize) -> Built
    {
        match self {
            Self::Constant { constant } => Built::from_wdl(wdl::ConstantWDL { value: constant.value }, duration),
            Self::Linear { linear } => {
                Built::from_wdl(wdl::LinearWDL { start: linear.start, end: linear.end }, duration)
            },
            Self::Warmup { warmup } => {
                let inner = warmup.inner.build(duration);
                Built::from_wdl(wdl::Warmup { inner, warmup_batches: warmup.batches }, duration)
            },
            Self::Sequence { sequence } => sequence.build(duration),
        }
    }
}

impl WdlSpec
{
    fn warms_up(&self) -> bool
    {
        match self {
            Self::Constant { .. } | Self::Linear { .. } => false,
            Self::Warmup { .. } => true,
            Self::Sequence { sequence } => sequence.first.warms_up() || sequence.second.warms_up(),
        }
    }
}
