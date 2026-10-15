use std::time::Duration;

use console::{Style as ConsoleStyle, measure_text_width, truncate_str};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::{Outcome, Progress, TrainingProgress, View, clean};

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

struct Frame
{
    width: usize,
    color: bool,
    rows: Vec<Line<'static>>,
}

impl Frame
{
    fn new(width: usize, color: bool) -> Self { Self { width: width.max(2), color, rows: Vec::new() } }

    fn style(&self, text: impl AsRef<str>, style: Style) -> Span<'static>
    {
        Span::styled(text.as_ref().to_owned(), if self.color { style } else { Style::default() })
    }

    fn text(&mut self, text: &str) { self.rows.extend(wrap(&clean(text), self.width).into_iter().map(Line::from)); }

    fn styled(&mut self, text: &str, style: Style)
    {
        for row in wrap(&clean(text), self.width) {
            self.rows.push(Line::from(self.style(row, style)));
        }
    }

    fn gap(&mut self)
    {
        if self.rows.last().is_some_and(|row| row.width() > 0) {
            self.rows.push(Line::default());
        }
    }

    fn field(&mut self, label: &str, value: &str)
    {
        let label = clean(label);
        let value = clean(value);
        
        let space = self.width.saturating_sub(measure_text_width(&label) + measure_text_width(&value));
        
        if space < 2 {
            self.text(&format!("{label}  {value}"));
        } else {
            self.rows.push(Line::from(vec![
                self.style(label, Style::default().dim()),
                Span::raw(" ".repeat(space)),
                self.style(value, Style::default().bold()),
            ]));
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
        
        let width = (self.width - 4) / 2;
        for pair in fields.chunks(2) {
            let columns: Vec<_> = pair
                .iter()
                .map(|(label, value)| {
                    let mut column = Self::new(width, self.color);
                    column.field(label, value);
                    column.rows
                })
                .collect();
            
            let height = columns.iter().map(Vec::len).max().unwrap_or(0);
            for row in 0..height {
                let left = columns[0].get(row).cloned().unwrap_or_default();
                let right = columns.get(1).and_then(|column| column.get(row)).cloned().unwrap_or_default();

                let padding = width.saturating_sub(left.width()) + 4;

                let mut spans = left.spans;
                spans.push(Span::raw(" ".repeat(padding)));
                spans.extend(right.spans);

                self.rows.push(Line::from(spans));
            }
        }
    }

    fn bar(&mut self, label: &str, count: &str, fraction: f64, compact: bool)
    {
        let fraction = fraction.clamp(0.0, 1.0);
        let percent = format!("{:.1}%", fraction * 100.0);
        
        if compact || self.width < 10 {
            self.text(&format!("{label} {count} · {percent}"));
            return;
        }
        
        self.field(label, count);
        
        let length = self.width.saturating_sub(percent.len() + 2);
        let filled = (fraction * length as f64).round() as usize;
        
        self.rows.push(Line::from(vec![
            self.style("━".repeat(filled), Style::default().cyan()),
            self.style("─".repeat(length - filled), Style::default().dim()),
            Span::raw("  "),
            self.style(percent, Style::default().dim()),
        ]));
    }

    fn context(&mut self, label: &str, value: &str)
    {
        let label = clean(label);
        let value = clean(value);
        
        let prefix = format!("{label}  ");
        let indent = measure_text_width(&prefix);
        if indent + 12 >= self.width {
            self.text(&format!("{label}: {value}"));
            return;
        }
        
        for (index, row) in wrap(&value, self.width - indent).into_iter().enumerate() {
            let prefix = if index == 0 {
                self.style(&prefix, Style::default().dim())
            } else {
                Span::raw(" ".repeat(indent))
            };

            self.rows.push(Line::from(vec![prefix, Span::raw(row)]));
        }
    }

    fn header(&mut self, view: &View, mascot: bool, compact: bool, active: bool)
    {
        let title = format!("Bruce {}", env!("CARGO_PKG_VERSION"));
        
        let task = task_title(&view.task);
        if compact && !mascot {
            self.text(&format!("{title} · {task}"));
            return;
        }
        
        let run = context(view, "Run").unwrap_or(&view.run);
        let identity = if run.is_empty() { task.to_owned() } else { format!("{task} · {run}") };
        
        let column = mascot_width() + 4;
        let beside = mascot && self.width >= column + 20;
        let width = if beside { self.width - column } else { self.width };
        let identity_rows = |text: &str| {
            let text = clean(text);

            if active { vec![truncate_str(&text, width, "…").into_owned()] } else { wrap(&text, width) }
        };
        
        let mut rows: Vec<Line<'static>> =
            wrap(&title, width).into_iter().map(|row| Line::from(self.style(row, Style::default().bold()))).collect();
        rows.extend(identity_rows(&identity).into_iter().map(Line::from));
        
        if let Some(device) = context(view, "Device") {
            rows.extend(
                identity_rows(device).into_iter().map(|row| Line::from(self.style(row, Style::default().dim()))),
            );
        }
        
        if mascot {
            self.rows.push(Line::default());
        }
        
        if beside {
            let height = rows.len().max(SHARK.len());
            
            let art_start = (height - SHARK.len()) / 2;
            let text_start = (height - rows.len()) / 2;
            
            for row in 0..height {
                let art = row.checked_sub(art_start).and_then(|i| SHARK.get(i)).copied().unwrap_or("");
                
                let mut spans = vec![self.style(art, Style::default().cyan())];
                if let Some(text) = row.checked_sub(text_start).and_then(|i| rows.get(i)) {
                    spans.push(Span::raw(" ".repeat(column.saturating_sub(measure_text_width(art)))));
                    spans.extend(text.spans.iter().cloned());
                }

                self.rows.push(Line::from(spans));
            }
        } else {
            self.rows.extend(rows);
            
            if mascot {
                self.gap();
                
                for row in SHARK {
                    self.rows.push(Line::from(self.style(row, Style::default().cyan())));
                }
            }
        }
    }
}

pub(super) fn live(view: &View, width: u16, height: u16, keyboard: bool) -> Vec<Line<'static>>
{
    let width = usize::from(width).saturating_sub(2).clamp(2, 120);
    let height = usize::from(height);

    let budget = height.saturating_sub(1).max(1);
    let mut compact = height < 20;
    
    let (mut frame, mut header) = running_frame(view, width, budget, compact, keyboard);
    if !compact && frame.rows.len() > budget {
        compact = true;
        (frame, header) = running_frame(view, width, budget, true, keyboard);
    }
    
    if !compact && width >= mascot_width() {
        let mut large = Frame::new(width, view.color);
        large.header(view, true, false, true);
        
        if large.rows.len() + frame.rows.len() - header <= budget {
            frame.rows.splice(..header, large.rows);
        }
    }
    
    if frame.rows.len() > budget && !view.stopping {
        let hint_count = wrap(hint(view, keyboard), width).len();
        
        if hint_count < budget {
            let hint = frame.rows.split_off(frame.rows.len() - hint_count);
            
            frame.rows.truncate(budget - hint_count);
            frame.rows.extend(hint);
        }
    }
    
    frame.rows.truncate(budget);
    frame.rows
}

fn running_frame(view: &View, width: usize, budget: usize, compact: bool, keyboard: bool) -> (Frame, usize)
{
    let mut frame = Frame::new(width, view.color);
    frame.header(view, false, compact, true);
    
    let header = frame.rows.len();
    
    if !compact {
        frame.gap();
    }
    
    let phase = if view.phase.is_empty() { "Preparing" } else { &view.phase };
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let tick = (view.started.elapsed().as_millis() / 100) as usize % spinner.len();
    let status = format!("{} {phase}", spinner[tick]);
    let status = if compact { truncate_str(&status, width.max(2), "…").into_owned() } else { status };
    frame.styled(&status, Style::default().cyan());
    
    let training = match &view.progress {
        Some(Progress::Training(value)) => Some(value),
        _ => view.training.as_ref(),
    };
    
    if let Some(training) = training {
        let total = training.final_superbatch.saturating_mul(training.batches_per_superbatch);
        let completed = training
            .superbatch
            .saturating_sub(1)
            .saturating_mul(training.batches_per_superbatch)
            .saturating_add(training.batch);
        let superbatch = format!("{}/{}", training.superbatch, training.final_superbatch);
        frame.bar(
            if compact { "Overall" } else { "Overall training" },
            &if compact { superbatch } else { format!("Superbatch {superbatch}") },
            fraction(completed, total),
            compact,
        );
        frame.bar(
            "Batches",
            &format!("{}/{}", number(training.batch as u64), number(training.batches_per_superbatch as u64)),
            fraction(training.batch, training.batches_per_superbatch),
            compact,
        );
        
        if !compact {
            frame.gap();
        }
        
        let age =
            if matches!(view.progress, Some(Progress::Training(_))) { view.updated.elapsed() } else { Duration::ZERO };
        
        frame.fields(&[
            ("Loss", training.loss.map(|loss| format!("{loss:.6}")).unwrap_or_else(|| "—".into())),
            (
                "Throughput",
                rate(training).map(|value| format!("{:.2} Mpos/s", value / 1e6)).unwrap_or_else(|| "—".into()),
            ),
            ("Learning rate", format!("{:.3e}", training.learning_rate)),
            ("Elapsed", duration(training.elapsed.saturating_add(age))),
            ("Superbatch ETA", training.superbatch_eta.map(duration).unwrap_or_else(|| "—".into())),
            ("Run ETA", training.run_eta.map(duration).unwrap_or_else(|| "—".into())),
        ]);
    } else {
        if let Some(Progress::Work { completed, total, unit }) = &view.progress {
            if let Some(total) = total.filter(|total| *total > 0 && completed <= total) {
                frame.bar(
                    unit,
                    &format!("{}/{}", number(*completed), number(total)),
                    *completed as f64 / total as f64,
                    compact,
                );
            } else {
                frame.text(&format!("{} {unit}", number(*completed)));
            }
        }
        
        frame.field("Elapsed", &duration(view.started.elapsed()));
        
        if !compact {
            frame.gap();
        }
        
        for (label, value) in &view.contexts {
            if matches!(label.as_str(), "Input" | "Output" | "Network" | "Limit" | "Format" | "Dataset") {
                let mut item = Frame::new(width, view.color);
                item.context(label, value);
                
                if frame.rows.len() + item.rows.len() + 2 <= budget {
                    frame.rows.extend(item.rows);
                }
            }
        }
    }
    
    if !view.stopping {
        if !compact {
            frame.gap();
        }
        
        frame.styled(hint(view, keyboard), Style::default().dim());
    }
    
    (frame, header)
}

pub(super) fn completed(view: &View, outcome: Outcome, width: usize, rich: bool) -> Vec<String>
{
    let mut frame = Frame::new(width, view.color);
    frame.header(view, width >= mascot_width(), false, false);
    frame.gap();
    
    let (label, style) = match outcome {
        Outcome::Success => ("complete", Style::default().green()),
        Outcome::Cancelled => ("cancelled", Style::default().yellow()),
        Outcome::Failed => ("failed", Style::default().red()),
    };
    frame.styled(&format!("{} {label}", task_title(&view.task)), style.bold());
    
    if outcome == Outcome::Failed
        && let Some((summary, recovery)) = &view.failure
    {
        frame.gap();
        frame.text(summary);
        frame.text(&format!("Help: {recovery}"));
    }
    
    if outcome == Outcome::Success
        && rich
        && let Some(output) = &view.output
    {
        for section in &output.sections {
            frame.gap();
            
            if !section.title.is_empty() {
                frame.styled(section.title, Style::default().bold());
            }
            
            for (label, value) in &section.fields {
                frame.field(label, value);
            }
        }
    }
    
    if let Some(training) = &view.training {
        frame.gap();
        
        if let Some(loss) = view.training_loss {
            frame.field("Run loss", &format!("{loss:.6}"));
        }
        
        frame.field("Batches", &number(training.session_batches as u64));
        frame.field("Elapsed", &duration(training.elapsed));
        
        if outcome != Outcome::Success {
            frame.field("Superbatch", &format!("{}/{}", training.superbatch, training.final_superbatch));
        }
    } else if view.output.is_none()
        && let Some(Progress::Work { completed, unit, .. }) = &view.progress
    {
        frame.field(unit, &number(*completed));
    }
    
    let mut first = true;
    for (label, value) in &view.contexts {
        if matches!(label.as_str(), "Log" | "TensorBoard" | "Latest checkpoint" | "Report")
            || (label == "Output" && outcome == Outcome::Success)
        {
            if first {
                frame.gap();
                first = false;
            }
            
            frame.context(label, value);
        }
    }
    
    frame.rows.into_iter().map(ansi).collect()
}

fn ansi(line: Line<'_>) -> String
{
    line.spans
        .into_iter()
        .map(|span| {
            let mut style = ConsoleStyle::new();
            style = match span.style.fg {
                Some(Color::Cyan) => style.cyan(),
                Some(Color::Green) => style.green(),
                Some(Color::Yellow) => style.yellow(),
                Some(Color::Red) => style.red(),
                _ => style,
            };

            if span.style.add_modifier.contains(Modifier::BOLD) {
                style = style.bold();
            }

            if span.style.add_modifier.contains(Modifier::DIM) {
                style = style.dim();
            }
            
            style.for_stderr().force_styling(span.style != Style::default()).apply_to(span.content).to_string()
        })
        .collect()
}

pub(super) fn plain(view: &View, final_snapshot: bool) -> Vec<String>
{
    if final_snapshot && let Some(training) = &view.training {
        let mut rows = plain_training(training, Duration::ZERO);
        rows.push(format!("Session batches completed: {}", training.session_batches));
        return rows;
    }
    
    let mut rows = Vec::new();
    if !view.phase.is_empty() {
        rows.push(if view.progress.is_none() {
            format!("{}  elapsed {}", view.phase, clock(view.phase_started.elapsed()))
        } else {
            view.phase.clone()
        });
    }
    
    match &view.progress {
        Some(Progress::Training(training)) => {
            rows.extend(plain_training(training, if final_snapshot { Duration::ZERO } else { view.updated.elapsed() }))
        },
        Some(Progress::Work { completed, total, unit }) => {
            rows.push(match total {
                Some(total) if *total > 0 && completed <= total => {
                    format!("{completed}/{total} {unit} ({:.1}%)", *completed as f64 * 100.0 / *total as f64)
                },
                _ => format!("{completed} {unit}"),
            });
        },
        None => {},
    }
    
    rows
}

fn plain_training(training: &TrainingProgress, age: Duration) -> Vec<String>
{
    let mut metrics = Vec::new();
    
    if let Some(loss) = training.loss {
        metrics.push(format!("Loss {loss:.6}"));
    }
    
    metrics.push(format!("LR {:.3e}", training.learning_rate));
    
    let mut timing = vec![format!("Elapsed {}", clock(training.elapsed.saturating_add(age)))];
    
    if let Some(rate) = rate(training) {
        timing.push(format!("{:.2} Mpos/s", rate / 1e6));
    }
    if let Some(eta) = training.superbatch_eta {
        timing.push(format!("Superbatch ETA {}", clock(eta)));
    }
    if let Some(eta) = training.run_eta {
        timing.push(format!("Run ETA {}", clock(eta)));
    }
    
    vec![
        format!(
            "Superbatch {}/{}  Batches {}/{}  Global step {}",
            training.superbatch,
            training.final_superbatch,
            training.batch,
            training.batches_per_superbatch,
            training.global_batch
        ),
        metrics.join("  "),
        timing.join("  "),
    ]
}

pub(super) fn wrap(text: &str, width: usize) -> Vec<String>
{
    let width = width.max(2);
    
    let mut rows = Vec::new();
    
    for mut line in text.split('\n') {
        while measure_text_width(line) > width {
            let mut end = 0;
            
            let mut space = None;
            for (index, ch) in line.char_indices() {
                let next = index + ch.len_utf8();
                
                if measure_text_width(&line[..next]) > width {
                    break;
                }
                
                end = next;
                
                if ch.is_whitespace() {
                    space = Some(next);
                }
            }
            
            let cut = space.unwrap_or(end);
            if cut == 0 {
                break;
            }
            
            rows.push(line[..cut].to_owned());
            line = &line[cut..];
        }
        
        rows.push(line.to_owned());
    }
    
    rows
}

fn context<'a>(view: &'a View, key: &str) -> Option<&'a str>
{
    view.contexts.iter().find(|(label, _)| label == key).map(|(_, value)| value.as_str())
}

fn hint(view: &View, keyboard: bool) -> &'static str
{
    match (view.task.eq_ignore_ascii_case("train"), keyboard) {
        (true, true) => "q / ctrl+c  stop safely",
        (true, false) => "ctrl+c  stop safely",
        (false, true) => "q / ctrl+c  cancel",
        (false, false) => "ctrl+c  cancel",
    }
}

fn task_title(task: &str) -> &str
{
    match task {
        "train" | "Train" => "Training",
        "check" | "Check" => "Configuration check",
        "convert" | "Convert" => "Conversion",
        "analyze" | "Analyze" => "Analysis",
        _ => task,
    }
}

fn mascot_width() -> usize { SHARK.iter().map(|row| measure_text_width(row)).max().unwrap_or(0) }

fn fraction(completed: usize, total: usize) -> f64 { if total == 0 { 0.0 } else { completed as f64 / total as f64 } }

fn rate(training: &TrainingProgress) -> Option<f64>
{
    training.positions_per_second.filter(|rate| rate.is_finite() && *rate > 0.0)
}

fn number(value: u64) -> String
{
    let digits = value.to_string();
    
    let mut result = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            result.push(',');
        }
        
        result.push(digit);
    }
    
    result
}

fn duration(value: Duration) -> String
{
    let seconds = value.as_secs();
    match seconds {
        0..60 => format!("{seconds}s"),
        60..3600 => format!("{}m {}s", seconds / 60, seconds % 60),
        _ => format!("{}h {}m", seconds / 3600, seconds / 60 % 60),
    }
}

fn clock(value: Duration) -> String
{
    let seconds = value.as_secs();
    format!("{:02}:{:02}:{:02}", seconds / 3600, seconds / 60 % 60, seconds % 60)
}
