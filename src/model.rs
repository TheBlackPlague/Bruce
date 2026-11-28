use std::{collections::BTreeMap, fmt, fmt::Debug, num::NonZero, path::Path, str::FromStr, sync::Arc, thread};

use anyhow::{Result, anyhow};
use bullet_compiler::{ir::NodeId, tensor::TValue};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Stream},
};
use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{Chess768, ChessBucketsMirrored, SparseInputType},
    },
    nn::ExecutionContext,
};
use bullet_trainer::{
    model::{DenseInput, ModelDefinition, ModelInputs, ModelInputsMapper, SavedFormat, SparseInput, TensorMap},
    run::{HostPool, Step},
};
use clap::ValueEnum;
use serde::Serialize;

use crate::{data::label::Label, train::checkpoint};

pub const SCALE: f32 = 400.0;
pub const QA: i16 = 255;
pub const QB: i16 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Architecture
{
    Starshard,
    Aurora,
    Flare,
    Corona,
    Eclipse,
    Parhelion,
}

fn inputs() -> ModelInputs<((SparseInput, SparseInput), DenseInput<f32>)>
{
    ModelInputs::default().add_sparse("stm", (768, 1), 32).add_sparse("ntm", (768, 1), 32).add_dense("targets", (1, 1))
}

impl Architecture
{
    pub fn names() -> String
    {
        Self::value_variants().iter().map(|each| each.name().to_lowercase()).collect::<Vec<_>>().join(", ")
    }

    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Starshard => "Starshard",
            Self::Aurora => "Aurora",
            Self::Flare => "Flare",
            Self::Corona => "Corona",
            Self::Eclipse => "Eclipse",
            Self::Parhelion => "Parhelion",
        }
    }

    pub const fn hidden(self) -> usize
    {
        match self {
            Self::Starshard => 256,
            Self::Aurora => 384,
            Self::Flare | Self::Corona => 768,
            Self::Eclipse | Self::Parhelion => 1536,
        }
    }

    pub fn definition(self) -> ModelDefinition
    {
        ModelDefinition::build(&inputs(), |builder, ((stm, ntm), targets)| {
            let l0 = builder.new_affine("l0", 768, self.hidden());
            let l1 = builder.new_affine("l1", 2 * self.hidden(), 1);

            let y0 = l0.forward(stm);
            let y1 = l0.forward(ntm);

            let (y0, y1) = match self {
                Self::Starshard | Self::Aurora | Self::Flare => (y0.crelu(), y1.crelu()),
                Self::Corona | Self::Eclipse | Self::Parhelion => (y0.screlu(), y1.screlu()),
            };

            let y = l1.forward(y0.concat(y1));

            let loss = y.sigmoid().squared_error(targets).reduce_sum_batch();

            (Some(loss), vec![("output".to_owned(), y)])
        })
    }

    pub fn mapper(self, blend: impl Fn(Step) -> f32 + Send + Sync + 'static) -> ModelInputsMapper<ChessBoard>
    {
        ModelInputsMapper::build(&inputs(), move |position: &ChessBoard, step, ((stm, ntm), targets)| {
            let mut features = stm.iter_mut().zip(ntm.iter_mut());
            let write = |ours, theirs| {
                if let Some((stm, ntm)) = features.next() {
                    *stm = ours as i32;
                    *ntm = theirs as i32;
                }
            };

            match self {
                Self::Starshard | Self::Aurora | Self::Flare | Self::Corona | Self::Eclipse => {
                    Chess768.map_features(position, write)
                },
                Self::Parhelion => ChessBucketsMirrored::default().map_features(position, write),
            }

            if let Some((stm, ntm)) = features.next() {
                *stm = -1;
                *ntm = -1;
            }

            targets[0] = Label::read(position).target(position, blend(step));
        })
    }

    pub fn saved_format(self) -> Vec<SavedFormat>
    {
        [("l0w", QA), ("l0b", QA), ("l1w", QB), ("l1b", QA * QB)]
            .into_iter()
            .map(|(name, scale)| SavedFormat::id(name).round().quantise::<i16>(scale))
            .collect()
    }
}

impl fmt::Display for Architecture
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.name()) }
}

impl FromStr for Architecture
{
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err>
    {
        Self::value_variants()
            .iter()
            .copied()
            .find(|architecture| architecture.name().eq_ignore_ascii_case(value))
            .ok_or_else(|| format!("Unknown architecture '{value}'; expected one of {}", Self::names()))
    }
}

pub struct Evaluator
{
    architecture: Architecture,
    batch_size: usize,
    device: Arc<Device<ExecutionContext>>,
    stream: Arc<Stream<ExecutionContext>>,
    function: Function<ExecutionContext>,
    bindings: BTreeMap<NodeId, Arc<Buffer<ExecutionContext>>>,
    inputs: BTreeMap<String, NodeId>,
    staged: TensorMap<ExecutionContext>,
    output: Arc<Buffer<ExecutionContext>>,
    mapper: ModelInputsMapper<ChessBoard>,
    mappers: u8,
    pool: Arc<HostPool<ExecutionContext>>,
}

impl Evaluator
{
    pub fn open(checkpoint: &Path, architecture: Option<Architecture>, gpu: u32, batch_size: usize) -> Result<Self>
    {
        let (architecture, weights) = checkpoint::load(checkpoint, architecture)?;

        let ordinal = i32::try_from(gpu).map_err(|_| anyhow!("GPU {gpu} does not exist"))?;
        let device = Device::<ExecutionContext>::new(ordinal).map_err(failed(&format!("Could not open GPU {gpu}")))?;
        let weights = weights.to_device(&device).map_err(failed("Could not load the network onto the GPU"))?;

        let model = architecture.definition();
        let forward = model.lower_forward(batch_size).map_err(failed("Could not prepare the network"))?;
        let lowered = |node: &NodeId| {
            forward.map().get(node).copied().ok_or_else(|| anyhow!("The prepared network is incomplete"))
        };

        let mut bindings = BTreeMap::new();
        for (node, (name, _)) in model.ir().weights() {
            let buffer = weights.get(name).ok_or_else(|| anyhow!("Network weight {name} is missing"))?;
            bindings.insert(lowered(node)?, Arc::clone(buffer));
        }

        let inputs =
            model.ir().inputs().iter().map(|(node, name)| Ok((name.clone(), lowered(node)?))).collect::<Result<_>>()?;

        let (output, _) = model
            .outputs()
            .iter()
            .find(|(_, name)| name == "output")
            .ok_or_else(|| anyhow!("The network has no output"))?;
        let output = lowered(output)?;
        let shape = forward.ir().get_node(output).map_err(failed("Could not prepare the network output"))?.ty();
        let buffer = Buffer::zeroed(&device, shape.dtype(), shape.size().get())
            .map_err(failed("Could not allocate the network output"))?;
        bindings.insert(output, Arc::clone(&buffer));

        let stream = device.new_stream().map_err(failed("Could not open a GPU stream"))?;
        let mut function = Function::new(Arc::clone(&device), forward.ir().clone())
            .map_err(failed("Could not compile the network"))?;
        function.prealloc().map_err(failed("Could not allocate the network buffers"))?;

        let available = thread::available_parallelism().map_or(1, NonZero::get);

        Ok(Self {
            architecture,
            batch_size,
            pool: HostPool::new(Arc::clone(&device)),
            device,
            stream,
            function,
            bindings,
            inputs,
            staged: TensorMap::new(),
            output: buffer,
            mapper: architecture.mapper(|_| 0.0),
            mappers: u8::try_from(available.clamp(4, 8)).unwrap_or(8),
        })
    }

    pub fn architecture(&self) -> Architecture { self.architecture }

    pub fn evaluate(&mut self, positions: &[ChessBoard], outputs: &mut Vec<f32>) -> Result<()>
    {
        let batch_size = self.batch_size;
        let entry = outputs.len();
        outputs.reserve(positions.len());

        positions
            .chunks(batch_size)
            .try_for_each(|chunk| {
                let scores = if chunk.len() == batch_size {
                    self.forward(chunk)?
                } else {
                    let mut padded = chunk.to_vec();
                    padded.resize(batch_size, chunk[0]);
                    self.forward(&padded)?
                };
                outputs.extend_from_slice(&scores[..chunk.len()]);
                Ok(())
            })
            .inspect_err(|_| outputs.truncate(entry))
    }

    fn forward(&mut self, batch: &[ChessBoard]) -> Result<Vec<f32>>
    {
        let mapped = self
            .mapper
            .map(&self.pool, batch, Step::default(), self.mappers)
            .map_err(failed("Could not prepare the network inputs"))?;

        if self.staged.is_empty() {
            self.staged = mapped.to_device(&self.device).map_err(failed("Could not copy the network inputs"))?;

            for (name, buffer) in &self.staged {
                let node = self.inputs.get(name).ok_or_else(|| anyhow!("Unknown network input {name}"))?;
                self.bindings.insert(*node, Arc::clone(buffer));
            }
        } else {
            let copies = mapped
                .copy_to_device_async(&self.stream, &self.staged)
                .map_err(failed("Could not copy the network inputs"))?;

            for copy in copies {
                copy.value().map_err(failed("Could not copy the network inputs"))?;
            }
        }

        self.function
            .execute(Arc::clone(&self.stream), &self.bindings)
            .and_then(|completion| completion.value())
            .map_err(failed("Network evaluation failed"))?;

        match self.output.to_host().map_err(failed("Could not read the network output"))? {
            TValue::F32(scores) if scores.len() == self.batch_size => Ok(scores),
            _ => Err(anyhow!("The network returned the wrong number of scores")),
        }
    }
}

fn failed<E: Debug>(context: &str) -> impl Fn(E) -> anyhow::Error + '_ { move |error| anyhow!("{context}: {error:?}") }

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Losses
{
    pub positions: u64,
    pub score: f64,
    pub result: f64,
}

impl Losses
{
    pub fn add(&mut self, x: f32, board: &ChessBoard)
    {
        let prediction = sigmoid(f64::from(x));
        let score = sigmoid(f64::from(board.score) / f64::from(SCALE));
        let result = f64::from(board.result) / 2.0;

        self.positions += 1;
        self.score += (prediction - score).powi(2);
        self.result += (prediction - result).powi(2);
    }
}

pub fn sigmoid(value: f64) -> f64 { 1.0 / (1.0 + (-value).exp()) }
