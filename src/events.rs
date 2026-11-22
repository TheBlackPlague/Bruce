use std::{path::PathBuf, time::Duration};

use bullet_trainer::run::TrainingSteps;

use crate::control::Outcome;

pub type Rows = Vec<(String, String)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level
{
    Error,
    Warning,
    Info,
    Detail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task
{
    Train,
    Check,
    Benchmark,
    Analyze,
}

impl Task
{
    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Train => "Training",
            Self::Check => "Check",
            Self::Benchmark => "Benchmark",
            Self::Analyze => "Analysis",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage
{
    Training,
    SavingCheckpoint,
    Finalizing,
}

impl Stage
{
    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Training => "training",
            Self::SavingCheckpoint => "saving checkpoint",
            Self::Finalizing => "finalizing",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataResume
{
    Exact,
    Approximate,
    Restarted,
}

impl DataResume
{
    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Exact => "exact",
            Self::Approximate => "approximate",
            Self::Restarted => "restarted",
        }
    }
}

pub enum Event
{
    Started(Box<RunInfo>),
    Ready(Duration),
    Stage(Stage),
    Batch(Batch),
    Loader(LoaderSample),
    Superbatch(Superbatch),
    Checkpoint(Checkpoint),
    StopCommitted(usize),
    Scan(ScanInfo),
    Progress(Progress),
    Preview(Rows),
    Message(Level, String),
}

#[derive(Clone, Debug)]
pub struct RunInfo
{
    pub name: String,
    pub architecture: &'static str,
    pub device: String,
    pub device_architecture: Option<String>,
    pub backend: &'static str,
    pub steps: TrainingSteps,
    pub schedule: Vec<(f32, f32)>,
    pub phases: Vec<(usize, usize)>,
    pub labels: String,
    pub configuration: serde_json::Value,
    pub resumed: Option<Resumed>,
}

#[derive(Clone, Debug)]
pub struct Resumed
{
    pub checkpoint: PathBuf,
    pub completed: usize,
    pub data: DataResume,
    pub run: String,
    pub extended: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Batch
{
    pub superbatch: usize,
    pub batch: usize,
    pub loss: f32,
    pub learning_rate: f32,
    pub wdl: f32,
}

#[derive(Clone, Debug)]
pub struct LoaderSample
{
    pub blocked: Duration,
    pub reading: Duration,
    pub mapping: Duration,
}

#[derive(Clone, Debug)]
pub struct Superbatch
{
    pub superbatch: usize,
    pub loss: f32,
    pub duration: Duration,
    pub learning_rate: f32,
    pub wdl: f32,
}

#[derive(Clone, Debug)]
pub struct Checkpoint
{
    pub superbatch: usize,
    pub path: PathBuf,
    pub reason: &'static str,
    pub duration: Duration,
}

#[derive(Clone, Debug)]
pub struct ScanInfo
{
    pub task: Task,
    pub inputs: Vec<Input>,
    pub threads: usize,
    pub sampling: String,
    pub head: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Input
{
    pub path: PathBuf,
    pub total: u64,
}

#[derive(Clone, Debug)]
pub struct Progress
{
    pub input: usize,
    pub done: u64,
    pub games: u64,
    pub positions: u64,
}

#[derive(Debug)]
pub struct Ending
{
    pub outcome: Outcome,
    pub grid: Rows,
    pub rows: Rows,
    pub files: Rows,
}

impl Ending
{
    pub const fn of(outcome: Outcome) -> Self
    {
        Self { outcome, grid: Vec::new(), rows: Vec::new(), files: Vec::new() }
    }
}
