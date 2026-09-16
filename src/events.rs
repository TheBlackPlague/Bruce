use std::{io::Write, path::PathBuf};

use serde::{Deserialize, Serialize};

pub const EVENT_PREFIX: &str = "BRUCE_EVENT ";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Phase {
        name: String,
        completed: u64,
        total: Option<u64>,
    },

    DataProgress {
        name: String,
        bytes: u64,
        total: Option<u64>,
        positions: u64,
        skipped: u64,
    },

    Metric {
        superbatch: usize,
        batch: usize,
        batches_per_superbatch: usize,
        final_superbatch: usize,
        loss: f32,
        learning_rate: f32,
        positions: u64,
        total_positions: u64,
        elapsed_seconds: f64,
    },

    Checkpoint {
        path: PathBuf,
    },

    Note {
        message: String,
    },
    
    Finished,
}

#[derive(Clone, Default)]
pub struct Reporter {
    silent: bool,
}

impl Reporter {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn silent() -> Self {
        Self { silent: true }
    }

    pub fn emit(&self, event: Event) {
        if self.silent {
            return;
        }

        if let Ok(json) = serde_json::to_string(&event) {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "{EVENT_PREFIX}{json}");
        }
    }
}
