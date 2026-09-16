use anyhow::{Result, anyhow};
use bullet_gpu::runtime::Device;
use bullet_lib::{
    game::inputs::{Chess768, SparseInputType},
    nn::ExecutionContext,
    trainer::schedule::wdl::WdlScheduler,
};
use bullet_trainer::{
    model::{
        DenseInput, ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat,
        SparseInput,
    },
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
};
use bulletformat::{BulletFormat, ChessBoard};

use crate::schedule::WdlConfig;

pub const HIDDEN: usize = 384;
pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub const SCALE: f32 = 400.0;
pub const PAYLOAD_BYTES: usize = (768 * HIDDEN + HIDDEN + HIDDEN * 2 + 1) * 2;
pub const NETWORK_BYTES: usize = PAYLOAD_BYTES.div_ceil(64) * 64;
pub type AuroraOptimiser = Optimiser<ExecutionContext, AdamW<ExecutionContext>>;
type Inputs = ModelInputs<((SparseInput, SparseInput), DenseInput<f32>)>;

fn inputs() -> Inputs {
    ModelInputs::default()
        .add_sparse("stm", (768, 1), 32)
        .add_sparse("ntm", (768, 1), 32)

        .add_dense("targets", (1, 1))
}

pub fn definition() -> ModelDefinition {
    ModelDefinition::build(&inputs(), |builder, ((stm, ntm), target)| {
        let l0 = builder.new_affine("l0", 768, HIDDEN);
        let l1 = builder.new_affine("l1", 2 * HIDDEN, 1);

        let hidden = l0.forward(stm).crelu().concat(l0.forward(ntm).crelu());

        let output = l1.forward(hidden);

        let loss = output.sigmoid().squared_error(target).reduce_sum_batch();

        (Some(loss), vec![("output".into(), output)])
    })
}

pub fn create(seed: u64, device: i32, params: AdamWParams) -> Result<AuroraOptimiser> {
    let definition = definition();
    let weights = ModelWeights::new(&definition, seed);

    let device = Device::<ExecutionContext>::new(device)
        .map_err(|e| anyhow!("Cannot initialize GPU: {e:?}"))?;

    Optimiser::new(definition, weights, device, params)
        .map_err(|e| anyhow!("Cannot initialize Aurora: {e:?}"))
}

pub fn mapper(wdl: WdlConfig) -> ModelInputsMapper<ChessBoard> {
    ModelInputsMapper::build(
        &inputs(),
        move |pos: &ChessBoard, step, ((stm, ntm), targets)| {
            stm.fill(-1);
            ntm.fill(-1);

            let mut index = 0;

            Chess768.map_features(pos, |ours, theirs| {
                stm[index] = ours as i32;
                ntm[index] = theirs as i32;
                index += 1;
            });

            let blend = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
            targets[0] = target(pos.score(), pos.result(), blend);
        },
    )
}

fn target(score: i16, result: f32, blend: f32) -> f32 {
    blend * result + (1.0 - blend) / (1.0 + (-f32::from(score) / SCALE).exp())
}

pub fn saved_format() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("l0w").round().quantise::<i16>(QA     ),
        SavedFormat::id("l0b").round().quantise::<i16>(QA     ),
        SavedFormat::id("l1w").round().quantise::<i16>(     QB),
        SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_have_the_required_score_scale_and_blend() {
        assert_eq!(target(  0, 0.0, 0.0), 0.5);
        assert_eq!(target(800, 1.0, 1.0), 1.0);

        assert!((target(400, 0.0, 0.0) - 0.7310586).abs() < 1e-7);
        assert!((target(400, 0.0, 0.5) - 0.3655293).abs() < 1e-7);
    }

    #[test]
    fn quantised_layout_matches_mantaray_v2() {
        let def = definition();
        let mut weights = ModelWeights::zeroed(&def);

        let source = ModelWeights::new(&def, 42);

        for (id, value) in source.iter() {
            assert!(weights.set(id, value.values.clone()));
        }

        let bytes = weights.to_quantised_buffer(&saved_format(), true).unwrap();
        assert_eq!(PAYLOAD_BYTES, 592130);
        assert_eq!(bytes.len()  , 592192);

        let offsets = [
            ("l0w",             0              , QA     ),
            ("l0b",  768 * HIDDEN           * 2, QA     ),
            ("l1w", (768 * HIDDEN + HIDDEN) * 2,      QB),
            ("l1b",      PAYLOAD_BYTES - 2     , QA * QB),
        ];

        for (name, offset, scale) in offsets {
            let actual = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            let expected = (f64::from(source.get(name).values.f32()[0]) *
                            f64::from(scale)).round() as i16;

            assert_eq!(actual, expected, "{name}");
        }
    }

    #[test]
    fn graph_compiles_forward_and_backwards() {
        let def = definition();
        
        def.lower_forward(4).unwrap();
        def.lower_backward(&Default::default(), 4).unwrap();
    }
}
