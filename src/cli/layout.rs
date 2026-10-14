use std::time::Duration;

use console::{Style, measure_text_width, truncate_str};

use super::{Outcome, Progress, State, clean_display_text, wrap};

const SHARK: [&str; 17] = [
    "                                        ██▄",
    "                                        ███▄",
    "                                       ▄████▄",
    "                               ▄▄▄▄▄▄███████████▄▄▄▄▄",
    "                        ▄▄██████████████████████████████▄",
    "█▄                 ▄▄████████████████████████████████▪█████▄",
    " ███▄          ▄▄████████████████████████████████████████████▄",
    "  █████▄▄▄▄▄▄██████████████████████████████████████████████████",
    "   ▀██████████████████████████████████████████████▀▀███▀▀███▀▀",
    "   ▄██████▀▀▀▀███████████████████████████████████   █▀   █▀",
    "   ████▀         ▀▀▀▀████████████████████████████",
    " ███▀                  ▀▀▀███████████████████████▄  █▄",
    " █▀                          ▀▀▀███████████████████▄███▄▄▄",
    "                                   ▀▀██████████████████▀▀",
    "                                       ▄██████▀▀▀",
    "                                     ▄████▀",
    "                                     █▀▀",
];

struct Layout
{
    width: usize,
    color: bool,
    rows: Vec<String>,
}

impl Layout
{
    fn new(width: usize, color: bool) -> Self { Self { width: width.max(2), color, rows: Vec::new() } }

    fn accent(&self, text: impl AsRef<str>) -> String
    {
        Style::new().cyan().for_stderr().force_styling(self.color).apply_to(text.as_ref()).to_string()
    }

    fn muted(&self, text: impl AsRef<str>) -> String
    {
        Style::new().dim().for_stderr().force_styling(self.color).apply_to(text.as_ref()).to_string()
    }

    fn strong(&self, text: impl AsRef<str>) -> String
    {
        Style::new().bold().for_stderr().force_styling(self.color).apply_to(text.as_ref()).to_string()
    }

    fn text(&mut self, text: &str) { self.rows.extend(wrap(&clean_display_text(text), self.width)); }

    fn blank(&mut self)
    {
        if self.rows.last().is_some_and(|row| !row.is_empty()) {
            self.rows.push(String::new());
        }
    }

    fn header(&mut self, state: &State, mascot: bool, compact: bool, live: bool)
    {
        let title = format!("Bruce {}", env!("CARGO_PKG_VERSION"));
        let task = task_title(&state.task);
        let run = state
            .contexts
            .iter()
            .find(|(label, _)| label == "Run")
            .map_or(state.run.as_str(), |(_, value)| value.as_str());
        let identity = if run.is_empty() { task.to_owned() } else { format!("{task} · {run}") };
        let device = state.contexts.iter().find(|(label, _)| label == "Device").map(|(_, value)| value.as_str());

        if mascot {
            let shark_width = mascot_width();
            let column_start = shark_width + 4;
            let beside = self.width >= column_start + 20;
            let identity_width = if beside { self.width - column_start } else { self.width };
            let mut identity_rows = vec![self.strong(title)];
            if live {
                identity_rows.push(truncate_str(&identity, identity_width, "…").into_owned());
            } else {
                identity_rows.extend(wrap(&identity, identity_width));
            }
            if let Some(device) = device {
                if live {
                    identity_rows.push(self.muted(truncate_str(device, identity_width, "…")));
                } else {
                    identity_rows.extend(wrap(device, identity_width).into_iter().map(|line| self.muted(line)));
                }
            }

            if !beside {
                self.rows.extend(identity_rows);
                self.blank();
                for row in SHARK {
                    self.rows.push(self.accent(row));
                }
                return;
            }

            let count = SHARK.len().max(identity_rows.len());
            for index in 0..count {
                let shark = SHARK.get(index).copied().unwrap_or("");
                let right = identity_rows.get(index).map(String::as_str).unwrap_or("");
                let padding = " ".repeat(column_start.saturating_sub(measure_text_width(shark)));
                self.rows.push(format!("{}{padding}{right}", self.accent(shark)));
            }
        } else if compact {
            self.text(&format!("{title} · {task}"));
        } else {
            for row in wrap(&title, self.width) {
                self.rows.push(self.strong(row));
            }
            if live {
                self.text(&truncate_str(&identity, self.width, "…"));
            } else {
                self.text(&identity);
            }
            if let Some(device) = device {
                if live {
                    self.text(&truncate_str(device, self.width, "…"));
                } else {
                    self.text(device);
                }
            }
        }
    }

    fn field(&mut self, label: &str, value: &str)
    {
        let label = clean_display_text(label);
        let value = clean_display_text(value);

        let gap = self.width.saturating_sub(measure_text_width(&label) + measure_text_width(&value));

        if gap >= 2 {
            self.rows.push(format!("{}{}{}", self.muted(label), " ".repeat(gap), self.strong(value)));
        } else {
            self.text(&format!("{label}  {value}"));
        }
    }

    fn fields(&mut self, fields: &[(&str, String)])
    {
        if self.width < 70 {
            for (label, value) in fields {
                self.field(label, value);
            }
            return;
        }

        let column_width = (self.width - 4) / 2;
        for pair in fields.chunks(2) {
            let mut columns = Vec::new();
            for (label, value) in pair {
                let mut column = Self::new(column_width, self.color);
                column.field(label, value);
                columns.push(column.rows);
            }

            let count = columns.iter().map(Vec::len).max().unwrap_or(0);
            for index in 0..count {
                let left = columns[0].get(index).map(String::as_str).unwrap_or("");
                let right = columns.get(1).and_then(|column| column.get(index)).map(String::as_str).unwrap_or("");
                let gap = " ".repeat(column_width.saturating_sub(measure_text_width(left)) + 4);
                self.rows.push(format!("{left}{gap}{right}"));
            }
        }
    }

    fn progress(&mut self, label: &str, count: &str, fraction: f64, compact: bool)
    {
        let percent = format!("{:.1}%", fraction.clamp(0.0, 1.0) * 100.0);

        if compact || self.width < 10 {
            self.text(&format!("{label} {count} · {percent}"));
            return;
        }

        self.field(label, count);

        let bar_width = self.width.saturating_sub(percent.len() + 2);
        let filled = (fraction.clamp(0.0, 1.0) * bar_width as f64).round() as usize;
        let fill = self.accent("━".repeat(filled));
        let rest = self.muted("─".repeat(bar_width - filled));
        self.rows.push(format!("{fill}{rest}  {}", self.muted(percent)));
    }

    fn context(&mut self, label: &str, value: &str)
    {
        let label = clean_display_text(label);
        let value = clean_display_text(value);

        let prefix = format!("{label}  ");
        let prefix_width = measure_text_width(&prefix);

        if prefix_width + 12 < self.width {
            for (index, line) in wrap(&value, self.width - prefix_width).iter().enumerate() {
                let prefix = if index == 0 { self.muted(&prefix) } else { " ".repeat(prefix_width) };
                self.rows.push(format!("{prefix}{line}"));
            }
        } else {
            self.text(&format!("{label}: {value}"));
        }
    }
}

pub(super) fn active(state: &State, width: usize, height: usize) -> Vec<String>
{
    let budget = height.saturating_sub(1).max(1);
    let mascot = width >= mascot_width() && height > SHARK.len();
    let mut rows = active_rows(state, width, height, mascot);
    if mascot && rows.len() > budget {
        rows = active_rows(state, width, height, false);
    }
    rows.truncate(budget);
    rows
}

fn active_rows(state: &State, width: usize, height: usize, mascot: bool) -> Vec<String>
{
    let budget = height.saturating_sub(1).max(1);
    let compact = height < 20;
    let mut layout = Layout::new(width, state.renderer.policy.color);
    layout.header(state, mascot, compact, true);

    if !compact {
        layout.blank();
    }

    let phase = if state.phase.is_empty() { "Preparing" } else { &state.phase };
    let marker = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let tick = (state.started.elapsed().as_millis() / 100) as usize % marker.len();
    let status = format!("{} {phase}", marker[tick]);
    let status = if compact { truncate_str(&status, width, "…").into_owned() } else { status };
    for row in wrap(&status, width) {
        let row = layout.accent(row);
        layout.rows.push(row);
    }

    let training = match &state.progress {
        Some(Progress::Training(training)) => Some(training),
        _ => state.last_training.as_ref(),
    };
    if let Some(training) = training {
        let total = training.final_superbatch.saturating_mul(training.batches_per_superbatch);
        let completed = training
            .superbatch
            .saturating_sub(1)
            .saturating_mul(training.batches_per_superbatch)
            .saturating_add(training.batch);
        let overall = if total > 0 { completed as f64 / total as f64 } else { 0.0 };
        let current = if training.batches_per_superbatch > 0 {
            training.batch as f64 / training.batches_per_superbatch as f64
        } else {
            0.0
        };
        let superbatch = format!("{}/{}", training.superbatch, training.final_superbatch);
        let batches = format!("{}/{}", number(training.batch as u64), number(training.batches_per_superbatch as u64));
        if compact {
            layout.progress("Overall", &superbatch, overall, true);
        } else {
            layout.progress("Overall training", &format!("Superbatch {superbatch}"), overall, false);
        }
        layout.progress("Batches", &batches, current, compact);
        if !compact {
            layout.blank();
        }

        let age = if matches!(state.progress, Some(Progress::Training(_))) {
            state.updated.elapsed()
        } else {
            Duration::ZERO
        };
        let fields = [
            ("Loss", training.loss.map_or_else(|| "—".into(), |loss| format!("{loss:.6}"))),
            (
                "Throughput",
                training
                    .positions_per_second
                    .filter(|rate| rate.is_finite() && *rate > 0.0)
                    .map_or_else(|| "—".into(), |rate| format!("{:.2} Mpos/s", rate / 1e6)),
            ),
            ("Learning rate", format!("{:.3e}", training.learning_rate)),
            ("Elapsed", duration(training.elapsed.saturating_add(age))),
            ("Superbatch ETA", training.superbatch_eta.map_or_else(|| "—".into(), duration)),
            ("Run ETA", training.run_eta.map_or_else(|| "—".into(), duration)),
        ];
        layout.fields(&fields);
    } else {
        if let Some(Progress::Work { completed, total, unit }) = &state.progress {
            if let Some(total) = total.filter(|total| *total > 0 && completed <= total) {
                layout.progress(
                    unit,
                    &format!("{}/{}", number(*completed), number(total)),
                    *completed as f64 / total as f64,
                    compact,
                );
            } else {
                layout.text(&format!("{} {unit}", number(*completed)));
            }
        }
        layout.field("Elapsed", &duration(state.started.elapsed()));

        if !compact {
            layout.blank();
        }
        for (label, value) in &state.contexts {
            if matches!(label.as_str(), "Input" | "Output" | "Network" | "Limit" | "Format" | "Dataset") {
                let mut context = Layout::new(width, state.renderer.policy.color);
                context.context(label, value);
                if layout.rows.len() + context.rows.len() + 2 <= budget {
                    layout.rows.extend(context.rows);
                }
            }
        }
    }

    if !state.stopping {
        if !compact {
            layout.blank();
        }
        let hint = if state.task.eq_ignore_ascii_case("train") { "ctrl+c  stop safely" } else { "ctrl+c  cancel" };
        for row in wrap(hint, width) {
            layout.rows.push(layout.muted(row));
        }
    }

    layout.rows
}

pub(super) fn completed(state: &State, outcome: Outcome, width: usize, rich_output: bool) -> Vec<String>
{
    let mut layout = Layout::new(width, state.renderer.policy.color);
    layout.header(state, width >= mascot_width(), false, false);
    layout.blank();

    let outcome_label = match outcome {
        Outcome::Success => "complete",
        Outcome::Cancelled => "cancelled",
        Outcome::Failed => "failed",
    };
    let status = format!("{} {outcome_label}", task_title(&state.task));
    let style = match outcome {
        Outcome::Success => Style::new().green(),
        Outcome::Cancelled => Style::new().yellow(),
        Outcome::Failed => Style::new().red(),
    };
    for row in wrap(&status, width) {
        layout.rows.push(style.clone().bold().for_stderr().force_styling(layout.color).apply_to(row).to_string());
    }

    if outcome == Outcome::Failed
        && let Some((summary, recovery)) = &state.failure_message
    {
        layout.blank();
        layout.text(summary);
        layout.text(&format!("Help: {recovery}"));
    }

    if outcome == Outcome::Success
        && rich_output
        && let Some(output) = &state.output
    {
        for section in &output.sections {
            layout.blank();
            if !section.title.is_empty() {
                for row in wrap(&clean_display_text(section.title), width) {
                    layout.rows.push(layout.strong(row));
                }
            }
            for (label, value) in &section.fields {
                layout.field(label, value);
            }
        }
    }

    if let Some(training) = state.last_training {
        layout.blank();
        if let Some(loss) = state.training_loss {
            layout.field("Run loss", &format!("{loss:.6}"));
        }
        layout.field("Batches", &number(training.session_batches as u64));
        layout.field("Elapsed", &duration(training.elapsed));
        if outcome != Outcome::Success {
            layout.field("Superbatch", &format!("{}/{}", training.superbatch, training.final_superbatch));
        }
    } else if state.output.is_none()
        && let Some(Progress::Work { completed, unit, .. }) = state.progress
    {
        layout.field(unit, &number(completed));
    }

    let destinations: Vec<_> = state
        .contexts
        .iter()
        .filter(|(label, _)| {
            matches!(label.as_str(), "Log" | "TensorBoard" | "Latest checkpoint" | "Report")
                || (label == "Output" && outcome == Outcome::Success)
        })
        .collect();
    if !destinations.is_empty() {
        layout.blank();
        for (label, value) in destinations {
            layout.context(label, value);
        }
    }
    layout.rows
}

fn mascot_width() -> usize { SHARK.iter().map(|row| measure_text_width(row)).max().unwrap_or(0) }

fn number(value: u64) -> String
{
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

fn task_title(task: &str) -> &str
{
    match task {
        "train" | "Train" => "Training",
        "convert" | "Convert" => "Conversion",
        "analyze" | "Analyze" => "Analysis",
        "check" | "Check" => "Configuration check",
        _ => task,
    }
}

fn duration(value: Duration) -> String
{
    let seconds = value.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, seconds / 60 % 60)
    }
}
