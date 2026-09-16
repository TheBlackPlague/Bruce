use std::time::{Duration, Instant};

use crate::events::Event;

pub struct ProgressState {
    pub name: String,
    pub completed: u64,
    pub total: Option<u64>,
    pub bytes: bool,
    pub positions: u64,
    pub skipped: u64,
    pub detail: String,
    started: Instant,
    elapsed: Option<f64>,
    rate: Option<f64>,
    run_remaining: Option<u64>,
}

impl Default for ProgressState {
    fn default() -> Self {
        Self {
            name: "Starting".into(),
            completed: 0,
            total: None,
            bytes: false,
            positions: 0,
            skipped: 0,
            detail: String::new(),
            started: Instant::now(),
            elapsed: None,
            rate: None,
            run_remaining: None,
        }
    }
}

impl ProgressState {
    pub fn update(&mut self, event: &Event) -> Option<String> {
        match event {
            Event::Phase {
                name,
                completed,
                total,
            } => {
                let changed = self.phase(name, false);
                self.completed = *completed;
                self.total = *total;
                changed.then(|| clean(name))
            }

            Event::DataProgress {
                name,
                bytes,
                total,
                positions,
                skipped,
            } => {
                let changed = self.phase(name, true);
                self.completed = *bytes;
                self.total = *total;
                self.positions = *positions;
                self.skipped = *skipped;
                changed.then(|| clean(name))
            }

            Event::Metric {
                superbatch,
                batch,
                batches_per_superbatch,
                final_superbatch,
                loss,
                learning_rate,
                positions,
                elapsed_seconds,
                total_positions,
            } => {
                let width = final_superbatch.to_string().len();
                let name = format!("Training · superbatch {superbatch:>width$}/{final_superbatch}");
                let changed = self.phase(&name, false);
                self.completed = *batch as u64;
                self.total = Some(*batches_per_superbatch as u64);
                self.positions = *positions;

                let throughput = rate(*positions, *elapsed_seconds);

                self.detail = format!(
                    "loss {loss:.5} · lr {learning_rate:.2e} · {} pos/s",
                    throughput.map_or_else(|| "--".into(), compact)
                );
                self.elapsed = Some(*elapsed_seconds);

                let step = (superbatch.saturating_sub(1) * batches_per_superbatch + batch) as u64;
                let size = total_positions.checked_div(step).unwrap_or(0);

                self.rate = throughput.filter(|_| size > 0).map(|r| r / size as f64);

                self.run_remaining = final_superbatch.saturating_sub(*superbatch)
                    .checked_mul(*batches_per_superbatch)
                    .and_then(|remaining| remaining.checked_add(
                        batches_per_superbatch.saturating_sub(*batch)
                    ))
                    .map(|remaining| remaining as u64);

                changed.then_some(name)
            }

            Event::Checkpoint { path } => Some(format!(
                "Checkpoint saved: {}",
                clean(&path.display().to_string())
            )),

            Event::Note { message } => Some(clean(message)),

            Event::Finished => None,
        }
    }

    fn phase(&mut self, name: &str, bytes: bool) -> bool {
        let name = clean(name);
        if self.name == name && self.bytes == bytes {
            return false;
        }

        *self = Self {
            name,
            bytes,
            ..Self::default()
        };

        true
    }

    pub fn elapsed(&self) -> f64 {
        self.elapsed.unwrap_or_else(|| self.started.elapsed().as_secs_f64())
    }

    pub fn fraction(&self) -> Option<f64> {
        fraction(self.completed, self.total)
    }

    pub fn count(&self) -> String {
        if self.bytes {
            format!(
                "{} / {}",
                bytes(self.completed),
                self.total.map_or_else(|| "?".into(), bytes)
            )
        } else {
            format!(
                "{} / {}",
                self.completed,
                self.total.map_or_else(|| "?".into(), |v| v.to_string())
            )
        }
    }

    pub fn live_message(&self) -> String {
        let elapsed = self.elapsed();

        let speed = if self.run_remaining.is_some() {
            self.rate
        } else {
            self.rate.or_else(|| rate(self.completed, elapsed))
        };

        let batch_eta = eta(
            self.completed,
            self.total,
            speed
        ).map_or_else(|| "--:--".into(), duration);

        let metrics = if self.bytes {
            format!(
                "{:.1} MiB/s · {} pos · skipped {}",
                speed.unwrap_or(0.0) / 1_048_576.0,
                compact(self.positions as f64),
                self.skipped
            )
        } else {
            self.detail.clone()
        };

        let mut timing = format!(
            "time {} · ETA {batch_eta}",
            duration(Duration::from_secs_f64(elapsed.max(0.0)))
        );

        if let Some(remaining) = self.run_remaining {
            let total_eta = eta(
                0,
                Some(remaining),
                self.rate
            ).map_or_else(|| "--:--".into(), duration);
            timing.push_str(&format!(" · Total ETA {total_eta}"));
        }

        if metrics.is_empty() {
            timing
        } else {
            format!("{metrics} · {timing}")
        }
    }

    pub fn summary(&self) -> String {
        if self.total.is_none() {
            return format!("{} · {}", self.name, self.live_message());
        }

        let percent = self
            .fraction()
            .map_or_else(|| "--".into(), |p| format!("{:.1}%", p * 100.0));

        format!(
            "{} · {percent} · {} · {}",
            self.name,
            self.count(),
            self.live_message()
        )
    }
}

pub fn fraction(completed: u64, total: Option<u64>) -> Option<f64> {
    total.filter(|&v| v > 0).map(|v| (completed as f64 / v as f64).min(1.0))
}

pub fn rate(completed: u64, elapsed: f64) -> Option<f64> {
    (elapsed.is_finite() && elapsed > 0.0 && completed > 0).then(|| completed as f64 / elapsed)
}

pub fn eta(completed: u64, total: Option<u64>, speed: Option<f64>) -> Option<Duration> {
    let remaining = total?.saturating_sub(completed);
    if remaining == 0 {
        return Some(Duration::ZERO);
    }

    let speed = speed.filter(|r| r.is_finite() && *r > 0.0)?;

    Duration::try_from_secs_f64(remaining as f64 / speed).ok()
}

fn bytes(n: u64) -> String {
    if        n >= 1 << 30 {
        format!("{:.2} GiB", n as f64 / (1u64 << 30) as f64)
    } else if n >= 1 << 20 {
        format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KiB", n as f64 / (1u64 << 10) as f64)
    } else {
        format!("{n} B")
    }
}

fn compact(n: f64) -> String {
    if        n >= 1e9 {
        format!("{:.2}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.1}k", n / 1e3)
    } else {
        format!("{n:.0}")
    }
}

fn duration(d: Duration) -> String {
    let seconds = d.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        seconds /    3600,
        seconds / 60 % 60,
        seconds %      60
    )
}

pub fn clean(value: &str) -> String {
    let mut result = String::new();
    let mut chars = value.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.next() {
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('@'..='~').contains(&c) {
                            break;
                        }
                    }
                }

                Some(']') => {
                    while let Some(c) = chars.next() {
                        if c == '\x07' || (c == '\x1b' && chars.next() == Some('\\')) {
                            break;
                        }
                    }
                }

                _ => {}
            }
        } else if !c.is_control() {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentage_and_eta_boundaries() {
        assert_eq!(fraction( 5, Some(10)), Some(0.5));
        assert_eq!(fraction(11, Some(10)), Some(1.0));
        assert_eq!(fraction( 0, Some( 0)), None     );

        assert_eq!(
            eta(5, Some(10), Some(2.0)),
            Some(Duration::from_millis(2500))
        );

        assert_eq!(eta(11, Some(10), None          ), Some(Duration::ZERO));
        assert_eq!(eta( 0, None    , Some(1.0     )), None                );
        assert_eq!(eta( 1, Some(10), Some(f64::NAN)), None                );

        assert_eq!(rate(100, 0.0), None);
    }
    #[test]
    fn resumed_training_eta_uses_session_work_and_global_batch_size() {
        let mut state = ProgressState::default();
        state.update(&Event::Metric {
            superbatch: 3,
            batch: 10,
            batches_per_superbatch: 100,
            final_superbatch: 5,
            loss: 0.1,
            learning_rate: 0.001,
            positions: 1000,
            total_positions: 21000,
            elapsed_seconds: 2.0,
        });

        assert_eq!(state.rate      , Some(5.0));
        assert_eq!(state.fraction(), Some(0.1));

        assert!(state.summary().contains("ETA 00:00:18 · Total ETA 00:00:58"));

        state.update(&Event::Phase {
            name: "Saving checkpoint".into(),
            completed: 0,
            total: None,
        });

        assert_eq!(state.rate, None);
        assert!(!state.live_message().contains("Total ETA"));
        assert!(state.detail.is_empty());
    }

    #[test]
    fn superbatch_alignment_and_run_eta_boundaries() {
        let mut state = ProgressState::default();

        for (superbatch, batch, positions, elapsed, expected_name, expected_eta) in [
            (  8,  50, 1000, 2.0, "Training · superbatch   8/240", "Total ETA 01:17:30"),
            (  9, 100, 1000, 2.0, "Training · superbatch   9/240", "Total ETA 01:17:00"),
            ( 10,  50,    0, 0.0, "Training · superbatch  10/240", "Total ETA --:--"   ),
            (240, 100, 1000, 2.0, "Training · superbatch 240/240", "Total ETA 00:00:00"),
        ] {
            state.update(&Event::Metric {
                superbatch, batch, batches_per_superbatch: 100, final_superbatch: 240,
                loss: 0.1, learning_rate: 0.001, positions,
                total_positions: ((superbatch - 1) * 100 + batch) as u64 * 100,
                elapsed_seconds: elapsed,
            });

            assert_eq!(state.name, expected_name);
            assert!(state.live_message().contains(expected_eta), "{}", state.live_message());
        }
    }

    #[test]
    fn plain_output_strips_controls_and_keeps_last_progress() {
        let mut state = ProgressState::default();

        state.update(&Event::DataProgress {
            name: "Converting".into(),
            bytes: 100,
            total: Some(200),
            positions: 3,
            skipped: 2,
        });

        state.update(&Event::Finished);

        assert_eq!(state.completed, 100);

        assert!(state.summary().contains("50.0%"    ));
        assert!(state.summary().contains("skipped 2"));

        assert_eq!(clean("\x1b[31mwarning\x1b[0m\r\n"), "warning");
        assert_eq!(clean("\x1b]0;title\x07safe")      , "safe"   );
        
        assert!(!state.summary().contains('\x1b'));
    }
}
