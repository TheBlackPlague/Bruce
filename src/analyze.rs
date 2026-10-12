use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use bullet_compiler::ir::NodeId;
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Gpu, Stream},
};
use bullet_lib::{game::formats::bulletformat::ChessBoard, nn::ExecutionContext, wdl::ConstantWDL};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputsMapper, ModelWeights, TensorMap},
    run::HostPool,
};
use serde::Serialize;

use crate::{
    architecture::{Architecture, SCALE},
    checkpoint,
    config::DataSource,
    data,
};

const BATCH_SIZE: usize = 1024;
const SCORE_BUCKET_CP: i32 = 100;
const CALIBRATION_BUCKETS: usize = 20;

pub struct Options
{
    pub source: DataSource,
    pub network: Option<PathBuf>,
    pub architecture: Option<Architecture>,
    pub gpu: i32,
    pub limit: Option<u64>,
}

#[derive(Serialize)]
pub struct Report
{
    pub perspective: &'static str,
    pub positions: u64,
    pub limit_reached: bool,
    pub scores: ScoreSummary,
    pub score_distribution: BTreeMap<i16, u64>,
    pub results: Results,
    pub material: BTreeMap<u32, Group>,
    pub network: Option<NetworkReport>,
}

#[derive(Default, Serialize)]
pub struct Results
{
    pub losses: u64,
    pub draws: u64,
    pub wins: u64,
}

#[derive(Default, Serialize)]
pub struct ScoreSummary
{
    pub mean_cp: f64,
    pub standard_deviation_cp: f64,
    pub minimum_cp: i16,
    pub maximum_cp: i16,
    pub percentiles_cp: BTreeMap<u8, i16>,
}

#[derive(Default, Serialize)]
pub struct Group
{
    pub positions: u64,
    pub mean_score_cp: f64,
    pub results: Results,
}

#[derive(Serialize)]
pub struct NetworkReport
{
    pub architecture: Architecture,
    pub weights: &'static str,
    pub score_bucket_cp: i32,
    pub overall: ErrorSummary,
    pub by_teacher_score_cp: BTreeMap<i32, ErrorSummary>,
    pub by_piece_count: BTreeMap<u32, ErrorSummary>,
    pub calibration: Vec<Calibration>,
    pub extreme_predictions: u64,
    pub extreme_probability_threshold: f64,
}

#[derive(Default, Serialize)]
pub struct ErrorSummary
{
    pub positions: u64,
    pub mean_prediction_cp: f64,
    pub bias_cp: f64,
    pub mean_absolute_error_cp: f64,
    pub root_mean_squared_error_cp: f64,
    pub result_squared_error: f64,
    pub teacher_probability_squared_error: f64,
    #[serde(skip)]
    squared_error_sum: f64,
}

#[derive(Serialize)]
pub struct Calibration
{
    pub probability_from: f64,
    pub probability_to: f64,
    pub positions: u64,
    pub mean_prediction: f64,
    pub mean_result: f64,
}

pub fn run(options: &Options) -> Result<Report>
{
    ensure!(options.limit != Some(0), "Position limit must be positive");
    ensure!(options.gpu >= 0, "GPU index must not be negative");
    ensure!(
        options.network.is_some() == options.architecture.is_some(),
        "Network analysis needs both a checkpoint and an architecture"
    );

    let mut report = Accumulator::new(options.architecture);
    let mut evaluator = match (&options.network, options.architecture) {
        (Some(path), Some(architecture)) => {
            ensure!(
                cfg!(any(feature = "cuda", feature = "rocm", feature = "metal")),
                "Network analysis needs a GPU build"
            );

            let weights = checkpoint::load_weights(path, architecture)?;
            let device =
                Device::<ExecutionContext>::new(options.gpu).map_err(|_| anyhow::anyhow!("Cannot open GPU"))?;

            Some(Evaluator::new(architecture, &weights, device)?)
        },
        _ => None,
    };

    let mut batch = Vec::with_capacity(BATCH_SIZE);
    data::visit_boards(&options.source, None, |board| {
        report.observe(board);

        if evaluator.is_some() {
            batch.push(board);
        }

        if batch.len() == BATCH_SIZE {
            evaluate_batch(evaluator.as_mut(), &mut batch, &mut report)?;
        }

        Ok(options.limit.is_some_and(|limit| report.count == limit))
    })?;

    evaluate_batch(evaluator.as_mut(), &mut batch, &mut report)?;

    ensure!(report.count > 0, "Dataset contains no positions");
    Ok(report.finish(options.limit))
}

fn evaluate_batch<G: Gpu>(
    evaluator: Option<&mut Evaluator<G>>,
    batch: &mut Vec<ChessBoard>,
    report: &mut Accumulator,
) -> Result<()>
{
    if let Some(evaluator) = evaluator
        && !batch.is_empty()
    {
        let length = batch.len();
        batch.resize(BATCH_SIZE, batch[0]);

        for (board, value) in batch.iter().zip(evaluator.evaluate(batch)?).take(length) {
            report.predict(*board, value)?;
        }

        batch.clear();
    }
    Ok(())
}

impl Report
{
    pub fn summary(&self) -> String
    {
        let mut text = format!(
            "{} positions; score range {} to {} cp; mean {:.1} cp",
            self.positions, self.scores.minimum_cp, self.scores.maximum_cp, self.scores.mean_cp
        );

        if let Some(network) = &self.network {
            text.push_str(&format!(
                "; network MAE {:.1} cp, bias {:+.1} cp",
                network.overall.mean_absolute_error_cp, network.overall.bias_cp
            ));
        }

        if self.limit_reached {
            text.push_str(" (position limit reached)");
        }

        text
    }

    pub fn write(&self, path: &Path) -> Result<()>
    {
        let file = File::options().write(true).create_new(true).open(path).context("Cannot create analysis report")?;
        let mut writer = BufWriter::new(file);

        serde_json::to_writer_pretty(&mut writer, self).context("Cannot write analysis report")?;

        writer.write_all(b"\n")?;
        writer.flush().context("Cannot finish analysis report")
    }
}

struct Accumulator
{
    count: u64,
    mean: f64,
    variance_sum: f64,
    distribution: Vec<u64>,
    results: Results,
    material: BTreeMap<u32, Group>,
    network: Option<NetworkReport>,
}

impl Accumulator
{
    fn new(architecture: Option<Architecture>) -> Self
    {
        Self {
            count: 0,
            mean: 0.0,
            variance_sum: 0.0,
            distribution: vec![0; 1 << 16],
            results: Results::default(),
            material: BTreeMap::new(),
            network: architecture.map(|architecture| NetworkReport {
                architecture,
                weights: "native_float_checkpoint",
                score_bucket_cp: SCORE_BUCKET_CP,
                overall: ErrorSummary::default(),
                by_teacher_score_cp: BTreeMap::new(),
                by_piece_count: BTreeMap::new(),
                calibration: (0..CALIBRATION_BUCKETS)
                    .map(|index| Calibration {
                        probability_from: index as f64 / CALIBRATION_BUCKETS as f64,
                        probability_to: (index + 1) as f64 / CALIBRATION_BUCKETS as f64,
                        positions: 0,
                        mean_prediction: 0.0,
                        mean_result: 0.0,
                    })
                    .collect(),
                extreme_predictions: 0,
                extreme_probability_threshold: 0.01,
            }),
        }
    }

    fn observe(&mut self, board: ChessBoard)
    {
        self.count += 1;

        let score = f64::from(board.score);
        let delta = score - self.mean;

        self.mean += delta / self.count as f64;
        self.variance_sum += delta * (score - self.mean);

        self.distribution[(i32::from(board.score) - i32::from(i16::MIN)) as usize] += 1;

        self.results.observe(board.result);

        let material = self.material.entry(board.occ.count_ones()).or_default();
        material.positions += 1;
        material.mean_score_cp += (score - material.mean_score_cp) / material.positions as f64;
        material.results.observe(board.result);
    }

    fn predict(&mut self, board: ChessBoard, output: f32) -> Result<()>
    {
        ensure!(output.is_finite(), "Network produced an invalid score");

        let network = self.network.as_mut().expect("Prediction requires a network");

        let predicted = f64::from(output) * f64::from(SCALE);
        network.overall.observe(board, predicted);

        let bucket = i32::from(board.score).div_euclid(SCORE_BUCKET_CP) * SCORE_BUCKET_CP;
        network.by_teacher_score_cp.entry(bucket).or_default().observe(board, predicted);
        network.by_piece_count.entry(board.occ.count_ones()).or_default().observe(board, predicted);

        let probability = sigmoid(f64::from(output));
        let index = ((probability * CALIBRATION_BUCKETS as f64) as usize).min(CALIBRATION_BUCKETS - 1);

        let calibration = &mut network.calibration[index];
        calibration.positions += 1;
        calibration.mean_prediction += (probability - calibration.mean_prediction) / calibration.positions as f64;
        calibration.mean_result +=
            (f64::from(board.result) / 2.0 - calibration.mean_result) / calibration.positions as f64;

        if !(0.01..=0.99).contains(&probability) {
            network.extreme_predictions += 1;
        }

        Ok(())
    }

    fn finish(self, limit: Option<u64>) -> Report
    {
        let distribution: BTreeMap<_, _> = self
            .distribution
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(index, &count)| ((index as i32 + i32::from(i16::MIN)) as i16, count))
            .collect();

        let minimum = distribution.first_key_value().map_or(0, |(&score, _)| score);
        let maximum = distribution.last_key_value().map_or(0, |(&score, _)| score);

        let percentiles = [1, 5, 25, 50, 75, 95, 99]
            .into_iter()
            .map(|percentile| {
                let target = (u128::from(self.count) * u128::from(percentile)).div_ceil(100) as u64;
                let mut cumulative = 0;
                let score = distribution
                    .iter()
                    .find_map(|(&score, &count)| {
                        cumulative += count;
                        (cumulative >= target).then_some(score)
                    })
                    .unwrap_or(0);
                (percentile, score)
            })
            .collect();

        Report {
            perspective: "side_to_move",
            positions: self.count,
            limit_reached: limit.is_some_and(|limit| self.count == limit),
            scores: ScoreSummary {
                mean_cp: self.mean,
                standard_deviation_cp: (self.variance_sum / self.count.max(1) as f64).sqrt(),
                minimum_cp: minimum,
                maximum_cp: maximum,
                percentiles_cp: percentiles,
            },
            score_distribution: distribution,
            results: self.results,
            material: self.material,
            network: self.network,
        }
    }
}

impl Results
{
    fn observe(&mut self, result: u8)
    {
        match result {
            0 => self.losses += 1,
            1 => self.draws += 1,
            2 => self.wins += 1,

            _ => unreachable!("Dataset reader validates results"),
        }
    }
}

impl ErrorSummary
{
    fn observe(&mut self, board: ChessBoard, predicted: f64)
    {
        self.positions += 1;

        let count = self.positions as f64;
        let error = predicted - f64::from(board.score);
        let probability = sigmoid(predicted / f64::from(SCALE));
        let teacher = sigmoid(f64::from(board.score) / f64::from(SCALE));

        self.mean_prediction_cp += (predicted - self.mean_prediction_cp) / count;
        self.bias_cp += (error - self.bias_cp) / count;
        self.mean_absolute_error_cp += (error.abs() - self.mean_absolute_error_cp) / count;
        self.squared_error_sum += error * error;
        self.root_mean_squared_error_cp = (self.squared_error_sum / count).sqrt();
        self.result_squared_error +=
            ((probability - f64::from(board.result) / 2.0).powi(2) - self.result_squared_error) / count;
        self.teacher_probability_squared_error +=
            ((probability - teacher).powi(2) - self.teacher_probability_squared_error) / count;
    }
}

fn sigmoid(value: f64) -> f64 { 1.0 / (1.0 + (-value).exp()) }

struct Evaluator<G: Gpu>
{
    device: Arc<Device<G>>,
    stream: Arc<Stream<G>>,
    function: Function<G>,
    buffers: BTreeMap<NodeId, Arc<Buffer<G>>>,
    inputs: BTreeMap<String, NodeId>,
    input_buffers: TensorMap<G>,
    output: Arc<Buffer<G>>,
    mapper: ModelInputsMapper<ChessBoard>,
    pool: Arc<HostPool<G>>,
}

impl<G: Gpu> Evaluator<G>
{
    fn new(architecture: Architecture, weights: &ModelWeights, device: Arc<Device<G>>) -> Result<Self>
    {
        let definition: ModelDefinition = architecture.definition();

        let forward = definition.lower_forward(BATCH_SIZE).map_err(|_| anyhow::anyhow!("Cannot prepare network"))?;

        let device_weights = weights.to_device(&device).map_err(|_| anyhow::anyhow!("Cannot load network onto GPU"))?;

        let mut buffers = BTreeMap::new();
        for (id, (name, _)) in definition.ir().weights() {
            buffers.insert(forward.map()[id], device_weights[name].clone());
        }

        let inputs = definition.ir().inputs().iter().map(|(id, name)| (name.clone(), forward.map()[id])).collect();

        let (output_node, _) =
            definition.outputs().iter().find(|(_, name)| name == "output").context("Network has no output")?;
        let output_id = forward.map()[output_node];
        let output_type =
            forward.ir().get_node(output_id).map_err(|_| anyhow::anyhow!("Cannot prepare network output"))?.ty();
        let output = Buffer::zeroed(&device, output_type.dtype(), output_type.size().get())
            .map_err(|_| anyhow::anyhow!("Cannot allocate network output"))?;

        buffers.insert(output_id, output.clone());

        let stream = device.new_stream().map_err(|_| anyhow::anyhow!("Cannot open GPU stream"))?;

        let mut function = Function::new(device.clone(), forward.ir().clone())
            .map_err(|_| anyhow::anyhow!("Cannot prepare network inference"))?;
        function.prealloc().map_err(|_| anyhow::anyhow!("Cannot allocate network buffers"))?;

        let pool = HostPool::new(device.clone());

        Ok(Self {
            device,
            stream,
            function,
            buffers,
            inputs,
            input_buffers: BTreeMap::new(),
            output,
            mapper: architecture.mapper(ConstantWDL { value: 0.0 }),
            pool,
        })
    }

    fn evaluate(&mut self, batch: &[ChessBoard]) -> Result<Vec<f32>>
    {
        let prepared = self
            .mapper
            .map(&self.pool, batch, Default::default(), 1)
            .map_err(|_| anyhow::anyhow!("Cannot prepare network inputs"))?;

        if self.input_buffers.is_empty() {
            self.input_buffers =
                prepared.to_device(&self.device).map_err(|_| anyhow::anyhow!("Cannot transfer network inputs"))?;

            for (name, buffer) in &self.input_buffers {
                self.buffers.insert(self.inputs[name], buffer.clone());
            }
        } else {
            let transfers = prepared
                .copy_to_device_async(&self.stream, &self.input_buffers)
                .map_err(|_| anyhow::anyhow!("Cannot transfer network inputs"))?;

            for transfer in transfers {
                transfer.value().map_err(|_| anyhow::anyhow!("Cannot transfer network inputs"))?;
            }
        }
        self.function
            .execute(self.stream.clone(), &self.buffers)
            .and_then(|value| value.value())
            .map_err(|_| anyhow::anyhow!("Network inference failed"))?;

        let values = self.output.to_host().map_err(|_| anyhow::anyhow!("Cannot read network output"))?;
        ensure!(values.f32().len() == batch.len(), "Network returned the wrong number of scores");

        Ok(values.f32().to_vec())
    }
}

#[cfg(test)]
#[path = "../tests/unit/analyze.rs"]
mod tests;
