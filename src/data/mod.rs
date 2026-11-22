pub mod game;
pub mod label;
pub mod pipeline;
pub mod select;
pub mod stream;

use std::{
    array, fmt,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::PathBuf,
};

pub use game::{Game, Outcome, Ply};
pub use label::LabelPolicy;
pub use select::Filter;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::events::Input;

const IDENTITY_SPAN: u64 = 1 << 20;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source
{
    pub path: PathBuf,
    #[serde(default = "unit", skip_serializing_if = "is_unit")]
    pub scale: f64,
}

fn unit() -> f64 { 1.0 }

pub fn is_unit(scale: &f64) -> bool { *scale == unit() }

pub fn is_zero<T: Default + PartialEq>(value: &T) -> bool { *value == T::default() }

impl Source
{
    pub fn new(path: impl Into<PathBuf>) -> Self { Self { path: path.into(), scale: unit() } }

    pub fn input(&self) -> Input
    {
        Input { path: self.path.clone(), total: fs::metadata(&self.path).map_or(0, |metadata| metadata.len()) }
    }

    pub fn identity(&self) -> Result<SourceIdentity, DataError>
    {
        let identify = || -> io::Result<SourceIdentity> {
            let file = File::open(&self.path)?;
            let length = file.metadata()?.len();
            let span = length.min(IDENTITY_SPAN);

            Ok(SourceIdentity { length, head: digest(&file, 0, span)?, tail: digest(&file, length - span, span)? })
        };

        identify().map_err(|error| DataError::io(&error).at(self))
    }
}

pub fn identities(sources: &[Source]) -> Result<Vec<SourceIdentity>, DataError>
{
    sources.iter().map(Source::identity).collect()
}

fn digest(mut file: &File, start: u64, span: u64) -> io::Result<String>
{
    file.seek(SeekFrom::Start(start))?;

    let mut hasher = Sha256::new();
    if io::copy(&mut file.take(span), &mut hasher)? != span {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }

    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct SourceIdentity
{
    pub length: u64,
    pub head: String,
    pub tail: String,
}

impl SourceIdentity
{
    pub fn key(&self) -> u64
    {
        let mut hasher = Sha256::new();
        hasher.update(self.length.to_le_bytes());
        hasher.update(&self.head);
        hasher.update(&self.tail);

        let digest = hasher.finalize();
        u64::from_be_bytes(array::from_fn(|index| digest[index]))
    }
}

#[derive(Clone, Debug)]
pub struct DataError
{
    message: String,
    path: Option<PathBuf>,
    offset: Option<u64>,
    game: Option<u64>,
}

impl DataError
{
    pub fn new(message: impl Into<String>) -> Self
    {
        Self { message: message.into(), path: None, offset: None, game: None }
    }

    pub fn io(error: &io::Error) -> Self { Self::new(format!("Could not read the dataset: {error}")) }

    pub fn truncated() -> Self { Self::new("The dataset is truncated") }

    pub fn no_positions() -> Self { Self::new("No position was selected in a whole pass over the data") }

    pub fn at(self, source: &Source) -> Self { Self { path: Some(source.path.clone()), ..self } }

    pub fn offset(self, offset: u64) -> Self { Self { offset: Some(offset), ..self } }

    pub fn game(self, index: u64) -> Self { Self { game: Some(index), ..self } }
}

impl fmt::Display for DataError
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        if let Some(path) = &self.path {
            write!(formatter, "{}: ", path.display())?;
        }

        write!(formatter, "{}", self.message)?;

        if let Some(offset) = self.offset {
            write!(formatter, " at byte {offset}")?;
        }

        if let Some(game) = self.game {
            write!(formatter, " (game {game})")?;
        }

        Ok(())
    }
}

impl std::error::Error for DataError {}
