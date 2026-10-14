pub mod analyze;
pub mod cli;
pub mod config;
pub mod convert;
pub mod data;
pub mod logging;
pub mod train;

pub use config::schedule;
pub use train::{architecture, checkpoint, metrics as report};

pub const BULLET_REVISION: &str = "1896c0884c7ededb1868068bbf5583d1726b37f6";
