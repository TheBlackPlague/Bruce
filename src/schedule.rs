use anyhow::{Result, ensure};
use bullet_lib::trainer::schedule::{
    lr::{self, LrScheduler},
    wdl::{self, WdlScheduler},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LrConfig {
    Constant {
        value: f32,
    },
    Step {
        start: f32,
        gamma: f32,
        step: usize,
    },
    Linear {
        initial: f32,
        final_value: f32,
        final_superbatch: usize,
    },
    Cosine {
        initial: f32,
        final_value: f32,
        final_superbatch: usize,
    },
    Exponential {
        initial: f32,
        final_value: f32,
        final_superbatch: usize,
    },
    Warmup {
        batches: usize,
        inner: Box<LrConfig>,
    },
    Sequence {
        switch_after: usize,

        first : Box<LrConfig>,
        second: Box<LrConfig>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WdlConfig {
    Constant {
        value: f32,
    },
    Linear {
        start: f32,
        end: f32,
    },
    Warmup {
        batches: usize,
        inner: Box<WdlConfig>,
    },
    Sequence {
        switch_after: usize,

        first : Box<WdlConfig>,
        second: Box<WdlConfig>,
    },
}

impl LrConfig {
    pub fn validate(&self, duration: usize) -> Result<()> {
        let positive = |v: f32| -> Result<()> {
            ensure!(
                v.is_finite() && v > 0.0,
                "Learning rates must be finite and positive"
            );

            Ok(())
        };

        match self {
            Self::Constant { value } => positive(*value)?,

            Self::Step { start, gamma, step } => {
                positive(*start)?;
                ensure!(
                    gamma.is_finite() && *gamma > 0.0 && *gamma <= 1.0 && *step > 0,
                    "Step LR requires gamma in (0,1] and step > 0"
                );
            }

            Self::Linear {
                initial,
                final_value,
                final_superbatch,
            } |

            Self::Cosine {
                initial,
                final_value,
                final_superbatch,
            } |

            Self::Exponential {
                initial,
                final_value,
                final_superbatch,
            } => {
                positive(*  initial  )?;
                positive(*final_value)?;

                ensure!(*final_superbatch > 0, "final_superbatch must be positive");
            }

            Self::Warmup { batches, inner } => {
                ensure!(*batches > 0, "Warmup batches must be positive");

                inner.validate(duration)?;
            }

            Self::Sequence {
                switch_after,
                first,
                second,
            } => {
                ensure!(
                    *switch_after > 0 && *switch_after < duration,
                    "Sequence switch_after must fall inside its training phase"
                );

                first .validate(*          switch_after)?;
                second.validate(duration - switch_after)?;
            }
        }

        Ok(())
    }
}

impl LrScheduler for LrConfig {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        match self {
            Self::Constant { value } => lr::ConstantLR { value: *value }.lr(batch, superbatch),

            Self::Step { start, gamma, step } => lr::StepLR {
                start: *start,
                gamma: *gamma,
                step: *step,
            }.lr(batch, superbatch),

            Self::Linear {
                initial,
                final_value,
                final_superbatch,
            } => lr::LinearDecayLR {
                initial_lr: *initial,
                final_lr: *final_value,
                final_superbatch: *final_superbatch,
            }.lr(batch, superbatch),

            Self::Cosine {
                initial,
                final_value,
                final_superbatch,
            } => lr::CosineDecayLR {
                initial_lr: *initial,
                final_lr: *final_value,
                final_superbatch: *final_superbatch,
            }.lr(batch, superbatch),

            Self::Exponential {
                initial,
                final_value,
                final_superbatch,
            } => lr::ExponentialDecayLR {
                initial_lr: *initial,
                final_lr: *final_value,
                final_superbatch: *final_superbatch,
            }.lr(batch, superbatch),

            Self::Warmup { batches, inner } => lr::Warmup {
                warmup_batches: *batches,
                inner: (**inner).clone(),
            }.lr(batch, superbatch),

            Self::Sequence {
                switch_after,
                first,
                second,
            } => lr::Sequence {
                first: (**first).clone(),
                second: (**second).clone(),
                first_scheduler_final_superbatch: *switch_after,
            }.lr(batch, superbatch),
        }
    }

    fn colourful(&self) -> String {
        format!("{self:?}")
    }
}

impl WdlConfig {
    pub fn validate(&self, duration: usize) -> Result<()> {
        let proportion = |v: f32| -> Result<()> {
            ensure!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "WDL proportions must be finite and in [0,1]"
            );

            Ok(())
        };

        match self {
            Self::Constant { value } => proportion(*value)?,

            Self::Linear { start, end } => {
                proportion(*start)?;
                proportion(*  end)?;
            }

            Self::Warmup { batches, inner } => {
                ensure!(*batches > 0, "Warmup batches must be positive");
                inner.validate(duration)?;
            }

            Self::Sequence {
                switch_after,
                first,
                second,
            } => {
                ensure!(
                    *switch_after > 0 && *switch_after < duration,
                    "WDL switch_after must fall inside its training phase"
                );

                first .validate(*          switch_after)?;
                second.validate(duration - switch_after)?;
            }
        }

        Ok(())
    }
}

impl WdlScheduler for WdlConfig {
    fn blend(&self, batch: usize, superbatch: usize, max: usize) -> f32 {
        match self {
            Self::Constant { value } => {
                wdl::ConstantWDL { value: *value }.blend(batch, superbatch, max)
            }

            Self::Linear { start, end } => wdl::LinearWDL {
                start: *start,
                  end: *  end,
            }.blend(batch, superbatch, max),

            Self::Warmup { batches, inner } => wdl::Warmup {
                warmup_batches: *batches,
                inner: (**inner).clone(),
            }.blend(batch, superbatch, max),

            Self::Sequence {
                switch_after,
                 first,
                second,
            } => wdl::Sequence {
                 first: (** first).clone(),
                second: (**second).clone(),

                first_scheduler_final_superbatch: *switch_after,
            }.blend(batch, superbatch, max),
        }
    }
    fn colourful(&self) -> String {
        format!("{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_boundaries_match_reference_run() {
        let s = LrConfig::Step {
            start: 0.001,
            gamma: 0.3,
            step: 60,
        };

        assert_eq!(s.lr(0,  60), 0.001                  );
        assert_eq!(s.lr(0,  61), 0.001 * 0.3            );
        assert_eq!(s.lr(0, 121), 0.001 * 0.3_f32.powi(2));
    }

    #[test]
    fn sequence_uses_phase_local_steps() {
        let s = LrConfig::Sequence {
            switch_after: 120,

             first: Box::new(LrConfig::Constant { value: 0.001 }),
            second: Box::new(LrConfig::Step {
                start: 0.0001,
                gamma: 0.3,
                step: 60,
            }),
        };

        assert_eq!(s.lr(0, 120), 0.001       );
        assert_eq!(s.lr(0, 121), 0.0001      );
        assert_eq!(s.lr(0, 181), 0.0001 * 0.3);
    }
}
