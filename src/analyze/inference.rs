use std::{collections::BTreeMap, sync::Arc};

use anyhow::{Context, Result, ensure};
use bullet_compiler::ir::NodeId;
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Stream},
};
use bullet_lib::{game::formats::bulletformat::ChessBoard, nn::ExecutionContext, wdl::ConstantWDL};
use bullet_trainer::{
    model::{ModelInputsMapper, ModelWeights, TensorMap},
    run::HostPool,
};

use crate::architecture::Architecture;

pub(super) const BATCH_SIZE: usize = 1024;

pub(super) struct Inference
{
    device: Arc<Device<ExecutionContext>>,
    stream: Arc<Stream<ExecutionContext>>,
    function: Function<ExecutionContext>,
    bindings: BTreeMap<NodeId, Arc<Buffer<ExecutionContext>>>,
    input_nodes: BTreeMap<String, NodeId>,
    input_buffers: TensorMap<ExecutionContext>,
    output_buffer: Arc<Buffer<ExecutionContext>>,
    mapper: ModelInputsMapper<ChessBoard>,
    host_pool: Arc<HostPool<ExecutionContext>>,
}

impl Inference
{
    pub(super) fn new(architecture: Architecture, weights: &ModelWeights, gpu: i32) -> Result<Self>
    {
        let device = Device::<ExecutionContext>::new(gpu).map_err(gpu_error).context("Cannot open GPU")?;

        let model = architecture.definition();

        let forward = model.lower_forward(BATCH_SIZE).map_err(gpu_error).context("Cannot prepare network")?;

        let mut bindings = BTreeMap::new();

        let weights = weights.to_device(&device).map_err(gpu_error).context("Cannot load network onto GPU")?;
        for (node, (name, _)) in model.ir().weights() {
            let buffer = weights.get(name).with_context(|| format!("Network weight '{name}' is missing"))?;
            bindings.insert(forward.map()[node], buffer.clone());
        }

        let input_nodes = model.ir().inputs().iter().map(|(node, name)| (name.clone(), forward.map()[node])).collect();
        let (output_node, _) =
            model.outputs().iter().find(|(_, name)| name == "output").context("Network has no output")?;
        let output_node = forward.map()[output_node];
        let output_type =
            forward.ir().get_node(output_node).map_err(gpu_error).context("Cannot prepare network output")?.ty();
        let output_buffer = Buffer::zeroed(&device, output_type.dtype(), output_type.size().get())
            .map_err(gpu_error)
            .context("Cannot allocate network output")?;
        bindings.insert(output_node, output_buffer.clone());

        let stream = device.new_stream().map_err(gpu_error).context("Cannot open GPU stream")?;
        let mut function = Function::new(device.clone(), forward.ir().clone())
            .map_err(gpu_error)
            .context("Cannot prepare network inference")?;
        function.prealloc().map_err(gpu_error).context("Cannot allocate network buffers")?;

        let host_pool = HostPool::new(device.clone());

        Ok(Self {
            device,
            stream,
            function,
            bindings,
            input_nodes,
            input_buffers: BTreeMap::new(),
            output_buffer,
            mapper: architecture.mapper(ConstantWDL { value: 0.0 }),
            host_pool,
        })
    }

    pub(super) fn predict(&mut self, positions: &[ChessBoard]) -> Result<Vec<f32>>
    {
        ensure!(!positions.is_empty() && positions.len() <= BATCH_SIZE, "Invalid inference batch size");

        let mut padded;
        let batch = if positions.len() == BATCH_SIZE {
            positions
        } else {
            padded = Vec::with_capacity(BATCH_SIZE);

            padded.extend_from_slice(positions);
            padded.resize(BATCH_SIZE, positions[0]);

            &padded
        };

        let inputs = self
            .mapper
            .map(&self.host_pool, batch, Default::default(), 1)
            .map_err(gpu_error)
            .context("Cannot prepare network inputs")?;

        if self.input_buffers.is_empty() {
            self.input_buffers =
                inputs.to_device(&self.device).map_err(gpu_error).context("Cannot transfer network inputs")?;

            for (name, buffer) in &self.input_buffers {
                let node = self.input_nodes.get(name).with_context(|| format!("Unknown network input '{name}'"))?;
                self.bindings.insert(*node, buffer.clone());
            }
        } else {
            let transfers = inputs
                .copy_to_device_async(&self.stream, &self.input_buffers)
                .map_err(gpu_error)
                .context("Cannot transfer network inputs")?;

            for transfer in transfers {
                transfer.value().map_err(gpu_error).context("Cannot transfer network inputs")?;
            }
        }

        self.function
            .execute(self.stream.clone(), &self.bindings)
            .and_then(|completion| completion.value())
            .map_err(gpu_error)
            .context("Network inference failed")?;
        let output = self.output_buffer.to_host().map_err(gpu_error).context("Cannot read network output")?;

        ensure!(output.f32().len() == BATCH_SIZE, "Network returned the wrong number of scores");

        Ok(output.f32()[..positions.len()].to_vec())
    }
}

fn gpu_error(error: impl std::fmt::Debug) -> anyhow::Error { anyhow::anyhow!("{error:?}") }
