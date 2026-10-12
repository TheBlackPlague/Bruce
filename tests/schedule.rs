use bruce::config::Config;
use bullet_lib::trainer::schedule::{
    lr::{self, LrScheduler},
    wdl::{self, WdlScheduler},
};

const EXAMPLE: &str = include_str!("../config/training.toml.default");

fn configured(lr: &str, wdl: &str) -> Config
{
    let prefix = EXAMPLE.split("[training.lr.0]").next().unwrap();
    Config::from_toml(&format!("{prefix}\n{lr}\n{wdl}")).unwrap()
}

const CONSTANT_WDL: &str =
    "[training.wdl.0]\nsuperbatch_range = [1, 400]\ntype = 'constant'\n[training.wdl.0.constant]\nvalue = 0.5";
const CONSTANT_LR: &str =
    "[training.lr.0]\nsuperbatch_range = [1, 400]\ntype = 'constant'\n[training.lr.0.constant]\nvalue = 0.001";

#[test]
fn indexed_phases_restart_local_steps()
{
    let config = Config::from_toml(EXAMPLE).unwrap();
    assert_eq!(config.learning_rate.lr(0, 1), 0.001);
    assert_eq!(config.learning_rate.lr(0, 30), 0.001);
    assert_eq!(config.learning_rate.lr(0, 31), 0.001 * 0.3);
    assert_eq!(config.learning_rate.lr(0, 271), 0.0001);
    assert_eq!(config.learning_rate.lr(0, 290), 0.0001);
    assert_eq!(config.learning_rate.lr(0, 291), 0.0001 * 0.3);
}

#[test]
fn schedule_ranges_and_indexes_are_strict()
{
    for source in [
        EXAMPLE.replace("[271, 400]", "[270, 400]"),
        EXAMPLE.replace("[271, 400]", "[272, 400]"),
        EXAMPLE.replace("[271, 400]", "[271, 399]"),
        EXAMPLE.replace("training.lr.1", "training.lr.2"),
        EXAMPLE.replace("training.lr.1", "training.lr.100"),
        EXAMPLE.replace("training.lr.1", "training.lr.01"),
        EXAMPLE.replace("size = 30", "size = 0"),
    ] {
        assert!(Config::from_toml(&source).is_err());
    }
}

#[test]
fn decay_schedulers_match_bullet_at_boundaries()
{
    for kind in ["linear", "cosine", "exponential"] {
        let config = configured(
            &format!(
                "[training.lr.0]\nsuperbatch_range = [1,400]\ntype = '{kind}'\n[training.lr.0.{kind}]\ninitial = \
                 0.001\nfinal_value = 0.00001\nfinal_superbatch = 200"
            ),
            CONSTANT_WDL,
        );
        for superbatch in [1, 2, 100, 199, 200, 400] {
            let expected =
                match kind {
                    "linear" => lr::LinearDecayLR { initial_lr: 0.001, final_lr: 0.00001, final_superbatch: 200 }
                        .lr(0, superbatch),
                    "cosine" => lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.00001, final_superbatch: 200 }
                        .lr(0, superbatch),
                    _ => lr::ExponentialDecayLR { initial_lr: 0.001, final_lr: 0.00001, final_superbatch: 200 }
                        .lr(0, superbatch),
                };
            assert_eq!(config.learning_rate.lr(0, superbatch), expected);
        }
    }
}

#[test]
fn nested_sequence_and_warmup_delegate_to_bullet()
{
    let config = configured(
        "[training.lr.0]\nsuperbatch_range=[1,400]\ntype='sequence'\n[training.lr.0.sequence]\\
         nswitch_after=200\n[training.lr.0.sequence.first]\ntype='constant'\n[training.lr.0.sequence.first.constant]\\
         nvalue=0.001\n[training.lr.0.sequence.second]\ntype='warmup'\\
         n[training.lr.0.sequence.second.warmup]\nbatches=32\n[training.lr.0.sequence.second.warmup.inner]\ntype='\
         constant'\n[training.lr.0.sequence.second.warmup.inner.constant]\\
         nvalue=0.0001",
        CONSTANT_WDL,
    );
    let reference = lr::Sequence {
        first: lr::ConstantLR { value: 0.001 },
        second: lr::Warmup { warmup_batches: 32, inner: lr::ConstantLR { value: 0.0001 } },
        first_scheduler_final_superbatch: 200,
    };
    for (batch, superbatch) in [(0, 1), (0, 200), (0, 201), (31, 201), (32, 201), (0, 202)] {
        assert_eq!(config.learning_rate.lr(batch, superbatch), reference.lr(batch, superbatch));
    }
}

#[test]
fn wdl_linear_uses_phase_duration()
{
    let config = configured(
        CONSTANT_LR,
        "[training.wdl.0]\nsuperbatch_range=[1,200]\ntype='constant'\n[training.wdl.0.constant]\\
         nvalue=0.0\n[training.wdl.1]\nsuperbatch_range=[201,400]\ntype='linear'\n[training.wdl.1.linear]\nstart=0.1\\
         nend=1.0",
    );
    let reference = wdl::LinearWDL { start: 0.1, end: 1.0 };
    assert_eq!(config.wdl.blend(0, 200, 400), 0.0);
    for superbatch in [201, 202, 300, 400] {
        assert_eq!(config.wdl.blend(0, superbatch, 400), reference.blend(0, superbatch - 200, 200));
    }
}

#[test]
fn wdl_nested_sequence_and_warmup_match_bullet()
{
    let config = configured(
        CONSTANT_LR,
        "[training.wdl.0]\nsuperbatch_range=[1,400]\ntype='sequence'\n[training.wdl.0.sequence]\\
         nswitch_after=200\n[training.wdl.0.sequence.first]\ntype='constant'\n[training.wdl.0.sequence.first.constant]\\
         \
         nvalue=0.0\n[training.wdl.0.sequence.second]\ntype='warmup'\\
         n[training.wdl.0.sequence.second.warmup]\nbatches=32\n[training.wdl.0.sequence.second.warmup.inner]\ntype='\
         linear'\n[training.wdl.0.sequence.second.warmup.inner.linear]\\
         nstart=0.5\nend=1.0",
    );
    let reference = wdl::Sequence {
        first: wdl::ConstantWDL { value: 0.0 },
        second: wdl::Warmup { warmup_batches: 32, inner: wdl::LinearWDL { start: 0.5, end: 1.0 } },
        first_scheduler_final_superbatch: 200,
    };
    for (batch, superbatch) in [(0, 200), (0, 201), (31, 201), (32, 201), (0, 202), (0, 400)] {
        assert_eq!(config.wdl.blend(batch, superbatch, 400), reference.blend(batch, superbatch, 400));
    }
}

#[test]
fn rejects_numeric_schedule_overflow_and_underflow()
{
    let prefix = EXAMPLE.split("[training.lr.0]").next().unwrap();
    for parameters in [
        "type='step'\n[training.lr.0.step]\nstart=0.001\ngamma=0.001\nsize=1",
        "type='exponential'\n[training.lr.0.exponential]\ninitial=1e-35\nfinal_value=1e35",
        "type='exponential'\n[training.lr.0.exponential]\ninitial=1e35\nfinal_value=1e-35",
        "type='warmup'\n[training.lr.0.warmup]\nbatches=6105\n[training.lr.0.warmup.inner]\ntype='constant'\\
         n[training.lr.0.warmup.inner.constant]\nvalue=0.001",
    ] {
        let source = format!("{prefix}\n[training.lr.0]\nsuperbatch_range=[1,400]\n{parameters}\n{CONSTANT_WDL}");
        assert!(Config::from_toml(&source).is_err());
    }

    let source = format!(
        "{}\n[training.lr.0]\nsuperbatch_range=[1,2147483650]\ntype='step'\n[training.lr.0.step]\\
         nstart=0.001\ngamma=1.0\nsize=1\n{}",
        prefix.replace("superbatches = 400", "superbatches = 2147483650"),
        CONSTANT_WDL.replace("400", "2147483650")
    );
    assert!(Config::from_toml(&source).is_err());
}
