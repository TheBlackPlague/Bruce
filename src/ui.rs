use std::{
    collections::VecDeque,
    fs::File,
    io::{self, BufRead, BufReader, IsTerminal, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use crossterm::{
    event::{self, Event as TerminalEvent, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Wrap},
};

use crate::events::{EVENT_PREFIX, Event};

const CANVAS : Color = Color::Rgb( 14,  16,  18);
const SURFACE: Color = Color::Rgb( 30,  34,  38);
const RAISED : Color = Color::Rgb( 44,  49,  54);
const INK    : Color = Color::Rgb(240, 238, 233);
const MUTED  : Color = Color::Rgb(173, 176, 178);
const ACCENT : Color = Color::Rgb(227, 195, 139);
const SUCCESS: Color = Color::Rgb( 85, 220, 133);
const DANGER : Color = Color::Rgb(255, 117, 138);
const WARNING: Color = Color::Rgb(243, 182,  77);
const INFO   : Color = Color::Rgb( 69, 202, 255);
const PENDING: Color = Color::Rgb(185, 154, 245);

const HISTORY_LIMIT: usize = 240;
const MESSAGE_LIMIT: usize = 8;
const REFRESH: Duration = Duration::from_millis(100);

pub fn run_child(mut command: Command, plain: bool, log_path: &Path) -> Result<()> {
    if let Some(parent) = log_path.parent().filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).context("creating log directory")?;
    }

    let log = File::create(log_path).context("opening worker log")?;

    let mut child = command
        .stdout(Stdio::from(log))
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .context("starting Bruce worker")?;

    let stderr = child.stderr.take().context("opening worker event stream")?;

    let (sender, receiver) = mpsc::sync_channel(256);

    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let event = match line {
                Ok(line) => line
                    .strip_prefix(EVENT_PREFIX)
                    .and_then(|json| serde_json::from_str(json).ok())
                    .unwrap_or_else(|| Event::Note {
                        message: strip_controls(&line),
                    }),

                Err(error) => Event::Note {
                    message: format!("Worker output: {error}"),
                },
            };

            if sender.send(event).is_err() {
                break;
            }
        }
    });

    let plain =
        plain                                                  ||
        !io::stdout().is_terminal()                            ||
        !io::stdin ().is_terminal()                            ||
        std::env::var("TERM").is_ok_and(|term| term == "dumb")  ;

    let mut state = DisplayState::new();
    let result = if plain {
        run_plain   (&mut child, &receiver, &mut state, log_path)
    } else {
        run_terminal(&mut child, &receiver, &mut state, log_path)
    };

    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }

    result
}

fn run_plain(
    child: &mut Child,
    receiver: &Receiver<Event>,
    state: &mut DisplayState,
    log_path: &Path,
) -> Result<()> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&interrupted);

    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))
        .context("installing interrupt handler")?;

    println!("Bruce | worker log: {}", log_path.display());

    let mut last_progress = Instant::now() - Duration::from_secs(5);

    loop {
        for event in receiver.try_iter().take(512) {
            write_plain_event(state, event, &mut io::stdout())?;
        }

        if last_progress.elapsed() >= Duration::from_secs(5) {
            println!("{}", state.summary());
            last_progress = Instant::now();
        }

        if interrupted.load(Ordering::Relaxed) {
            return interrupt(child, state);
        }

        if let Some(status) = child.try_wait()? {
            drain_final_plain(receiver, state, &mut io::stdout())?;

            if !status.success() {
                bail!(
                    "Worker exited with {status}. {} Log: {}",
                    state.last_message(),
                    log_path.display()
                );
            }

            println!("{}", state.summary());

            return Ok(());
        }

        thread::sleep(REFRESH);
    }
}

fn write_plain_event(
    state: &mut DisplayState,
    event: Event,
    output: &mut impl Write,
) -> io::Result<()> {
    let message = match &event {
        Event::Note { message } => Some(message.clone()),
        Event::Checkpoint { path } => Some(format!("Checkpoint saved: {}", path.display())),
        Event::Finished => Some("Complete.".to_owned()),
        Event::Phase { name, .. } if name != &state.phase => Some(format!("{name}...")),

        _ => None,
    };
    state.update(event);

    if let Some(message) = message {
        writeln!(output, "{}", strip_controls(&message))?;
    }

    Ok(())
}

fn drain_final_plain(
    receiver: &Receiver<Event>,
    state: &mut DisplayState,
    output: &mut impl Write,
) -> io::Result<()> {
    while let Ok(event) = receiver.recv_timeout(REFRESH) {
        write_plain_event(state, event, output)?;
    }

    Ok(())
}

fn run_terminal(
    child: &mut Child,
    receiver: &Receiver<Event>,
    state: &mut DisplayState,
    log_path: &Path,
) -> Result<()> {
    enable_raw_mode().context("enabling terminal input")?;

    let _restore = RestoreTerminal;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    loop {
        drain(receiver, state);
        terminal.draw(|frame| draw(frame, state))?;

        if let Some(status) = child.try_wait()? {
            drain_final(receiver, state);
            drop(terminal);
            drop(_restore);

            if !status.success() {
                bail!(
                    "Worker exited with {status}. {} Log: {}",
                    state.last_message(),
                    log_path.display()
                );
            }

            println!("Bruce complete. Worker log: {}", log_path.display());

            if let Some(path) = &state.checkpoint {
                println!("Checkpoint: {}", path.display());
            }

            return Ok(());
        }

        if  event::poll(REFRESH)?                        &&
            let TerminalEvent::Key(key) = event::read()? &&
            key.kind == KeyEventKind::Press              &&
            (
                key.code == KeyCode::Char('q') ||
                (
                    key.code == KeyCode::Char('c') &&
                    key.modifiers.contains(KeyModifiers::CONTROL)
                )
            )
        {
            return interrupt(child, state);
        }
    }
}

struct RestoreTerminal;

impl Drop for RestoreTerminal {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}

fn drain(receiver: &Receiver<Event>, state: &mut DisplayState) {
    for event in receiver.try_iter().take(512) {
        state.update(event);
    }
}

fn drain_final(receiver: &Receiver<Event>, state: &mut DisplayState) {
    while let Ok(event) = receiver.recv_timeout(REFRESH) {
        state.update(event);
    }
}

fn interrupt(child: &mut Child, state: &DisplayState) -> Result<()> {
    let _ = child.kill();
    let _ = child.wait();

    match &state.checkpoint {
        Some(path) => bail!(
            "Interrupted. Resume from the last completed checkpoint: {}. \
            Work since that checkpoint was not saved.",
            path.display()
        ),

        None => bail!(
            "Interrupted. Resume from an existing completed checkpoint, if available. \
            Work since the last checkpoint was not saved."
        ),
    }
}

struct DisplayState {
    started: Instant,
    phase: String,
    phase_started: Instant,
    phase_completed: u64,
    phase_total: Option<u64>,
    superbatch: usize,
    batch: usize,
    batches_per_superbatch: usize,
    final_superbatch: usize,
    first_completed_batch: Option<u64>,
    training_started: Option<Instant>,
    positions: u64,
    last_metric: Option<(Instant, u64)>,
    loss: f32,
    learning_rate: f32,
    throughput: f64,
    loss_history: VecDeque<(f64, f64)>,
    speed_history: VecDeque<(f64, f64)>,
    messages: VecDeque<String>,
    checkpoint: Option<std::path::PathBuf>,
    finished: bool,
}

impl DisplayState {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            phase: "Starting".to_owned(),
            phase_started: Instant::now(),
            phase_completed: 0,
            phase_total: None,
            superbatch: 0,
            batch: 0,
            batches_per_superbatch: 0,
            final_superbatch: 0,
            first_completed_batch: None,
            training_started: None,
            positions: 0,
            last_metric: None,
            loss: 0.0,
            learning_rate: 0.0,
            throughput: 0.0,
            loss_history: VecDeque::new(),
            speed_history: VecDeque::new(),
            messages: VecDeque::new(),
            checkpoint: None,
            finished: false,
        }
    }

    fn update(&mut self, event: Event) {
        match event {
            Event::Phase {
                name,
                completed,
                total,
            } => {
                if self.phase != name {
                    self.phase_started = Instant::now();
                }
                self.phase = strip_controls(&name);
                self.phase_completed = completed;
                self.phase_total = total;
            }

            Event::Metric {
                superbatch,
                batch,
                batches_per_superbatch,
                final_superbatch,
                loss,
                learning_rate,
                positions,
            } => {
                let now = Instant::now();

                if self.training_started.is_none() {
                    self.training_started = Some(now);
                    self.first_completed_batch =
                        Some(completed_batches(superbatch, batch, batches_per_superbatch));
                }

                if let Some((previous, previous_positions)) = self.last_metric {
                    let seconds = now.duration_since(previous).as_secs_f64();
                    if seconds >= 0.2 && positions >= previous_positions {
                        let speed = (positions - previous_positions) as f64 / seconds;

                        self.throughput = if self.throughput == 0.0 {
                            speed
                        } else {
                            self.throughput * 0.8 + speed * 0.2
                        };

                        self.last_metric = Some((now, positions));
                    }
                }

                if self.last_metric.is_none() {
                    self.last_metric = Some((now, positions));
                }

                self.superbatch = superbatch;
                self.batch = batch;
                self.batches_per_superbatch = batches_per_superbatch;
                self.final_superbatch = final_superbatch;
                self.loss = loss;
                self.learning_rate = learning_rate;
                self.positions = positions;

                if self.phase != "Training" {
                    self.phase = "Training".to_owned();
                    self.phase_started = now;
                }

                self.phase_completed = self.completed_batches();
                self.phase_total = Some(self.total_batches());

                let elapsed = self.started.elapsed().as_secs_f64();

                if loss.is_finite() {
                    push_bounded(
                        &mut self.loss_history,
                        (elapsed, f64::from(loss)),
                        HISTORY_LIMIT,
                    );
                }

                push_bounded(
                    &mut self.speed_history,
                    (elapsed, self.throughput),
                    HISTORY_LIMIT,
                );
            }

            Event::Checkpoint { path } => {
                self.note(format!("Checkpoint saved: {}", path.display()));
                self.checkpoint = Some(path);
            }

            Event::Note { message } => self.note(strip_controls(&message)),

            Event::Finished => {
                self.finished = true;
                self.phase = "Complete".to_owned();
                self.phase_completed = 1;
                self.phase_total = Some(1);
            }
        }
    }

    fn note(&mut self, message: String) {
        let message = message.chars().take(2000).collect::<String>();

        if !message.trim().is_empty() {
            push_bounded(&mut self.messages, message, MESSAGE_LIMIT);
        }
    }

    fn last_message(&self) -> &str {
        self.messages.back().map_or("", String::as_str)
    }

    fn total_batches(&self) -> u64 {
        (self.final_superbatch as u64).saturating_mul(self.batches_per_superbatch as u64)
    }

    fn completed_batches(&self) -> u64 {
        completed_batches(self.superbatch, self.batch, self.batches_per_superbatch)
    }

    fn total_eta(&self) -> String {
        match (self.training_started, self.first_completed_batch) {
            (Some(start), Some(first)) => eta(
                self.completed_batches().saturating_sub(first),
                self.total_batches().saturating_sub(first),

                start.elapsed(),
            ),

            _ => "unknown".to_owned(),
        }
    }

    fn summary(&self) -> String {
        if self.superbatch > 0 {
            format!(
                "{} | superbatch {}/{} | batch {}/{} | loss {:.6} | {:.0} positions/s | ETA {}",
                self.phase,
                self.superbatch,
                self.final_superbatch,
                self.batch,
                self.batches_per_superbatch,
                self.loss,
                self.throughput,
                self.total_eta()
            )
        } else {
            format!(
                "{} | {} | elapsed {}",
                self.phase,
                progress_label(
                    self.phase_completed,
                    self.phase_total,
                    self.phase_started.elapsed()
                ),
                duration(self.started.elapsed().as_secs())
            )
        }
    }
}

fn completed_batches(superbatch: usize, batch: usize, batches_per_superbatch: usize) -> u64 {
    (superbatch.saturating_sub(1) as u64)
        .saturating_mul(batches_per_superbatch as u64)
        .saturating_add(batch as u64)
}

fn push_bounded<T>(history: &mut VecDeque<T>, item: T, limit: usize) {
    if history.len() == limit {
        history.pop_front();
    }

    history.push_back(item);
}

fn draw(frame: &mut Frame, state: &DisplayState) {
    let area = frame.area();

    frame.render_widget(
        Block::default().style(Style::default().bg(CANVAS).fg(INK)),
        area,
    );

    if area.width < 45 || area.height < 23 {
        frame.render_widget(
            Paragraph::new(format!(
                "🦈 Bruce\n\n{}\n\nEnlarge the terminal for graphs.\nq / Ctrl-C: stop",
                state.summary()
            )).wrap(Wrap { trim: true }),
            area,
        );

        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(4),
            Constraint::Length(1),
        ])
        .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("🦈 Bruce", Style::default().fg(ACCENT).bold()),
            Span::styled("  /  ", Style::default().fg(RAISED)),
            Span::styled(
                &state.phase,
                Style::default().fg(if state.finished { SUCCESS } else { INFO }),
            ),
            Span::styled(
                format!("   elapsed {}", duration(state.started.elapsed().as_secs())),
                Style::default().fg(MUTED),
            ),
        ])),
        rows[0],
    );

    let phase_ratio = ratio(state.phase_completed, state.phase_total.unwrap_or(0));
    let phase_label = if state.phase == "Training" {
        format!(
            "{} / {} batches  ·  ETA {}",
            state.phase_completed,
            state.total_batches(),
            state.total_eta()
        )
    } else {
        progress_label(
            state.phase_completed,
            state.phase_total,
            state.phase_started.elapsed(),
        )
    };

    let phase_label = if state.phase_total.is_none() {
        format!(
            "{}  {phase_label}",
            activity_spinner(state.phase_started.elapsed())
        )
    } else {
        phase_label
    };

    gauge(frame, rows[1], &state.phase, phase_ratio, phase_label, INFO);

    let batch_ratio = ratio(state.batch as u64, state.batches_per_superbatch as u64);
    let batch_eta = match (state.training_started, state.first_completed_batch) {
        (Some(start), Some(first)) => {
            let observed = state.completed_batches().saturating_sub(first);

            eta(
                observed,
                observed.saturating_add(
                    state.batches_per_superbatch.saturating_sub(state.batch) as u64
                ),
                start.elapsed()
            )
        }

        _ => "unknown".to_owned(),
    };

    gauge(
        frame,
        rows[2],
        &format!(
            "Superbatch {} / {}",
            state.superbatch, state.final_superbatch
        ),
        batch_ratio,
        format!(
            "{} / {} batches  ·  ETA {batch_eta}",
            state.batch, state.batches_per_superbatch
        ),
        ACCENT,
    );

    gauge(
        frame,
        rows[3],
        "Training total",
        ratio(state.completed_batches(), state.total_batches()),
        format!(
            "{:.1}%  ·  ETA {}  ·  LR {:.6}",
            100.0 * ratio(state.completed_batches(), state.total_batches()),
            state.total_eta(),
            state.learning_rate
        ),
        PENDING,
    );

    let charts = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[4]);

    chart(
        frame,
        charts[0],
        &format!("Loss  {:.6}", state.loss),
        &state.loss_history,
        INFO,
    );

    chart(
        frame,
        charts[1],
        &format!("Throughput  {:.0} positions/s", state.throughput),
        &state.speed_history,
        SUCCESS,
    );

    let messages: Vec<Line> = state
        .messages
        .iter()
        .rev()
        .take(2)
        .rev()
        .map(|message| {
            let lower = message.to_ascii_lowercase();
            let color = if lower.contains("error") || lower.contains("panicked") {
                DANGER
            } else if lower.contains("warning") || lower.contains("limit") {
                WARNING
            } else {
                MUTED
            };
            Line::styled(message.as_str(), Style::default().fg(color))
        })
        .collect();

    frame.render_widget(
        Paragraph::new(messages)
            .block(panel("Activity"))
            .wrap(Wrap { trim: true }),
        rows[5],
    );

    frame.render_widget(
        Paragraph::new("q / Ctrl-C  Stop · resume from the last completed checkpoint")
            .style(Style::default().fg(MUTED)),
        rows[6],
    );
}

fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::default().fg(RAISED))
        .title_style(Style::default().fg(INK))
        .style(Style::default().bg(SURFACE))
        .title(title)
}

fn gauge(frame: &mut Frame, area: Rect, title: &str, value: f64, label: String, color: Color) {
    frame.render_widget(
        Gauge::default()
            .block(panel(title))
            .ratio(value)
            .label(label)
            .gauge_style(Style::default().fg(color).bg(SURFACE))
            .use_unicode(true),
        area,
    );
}

fn chart(frame: &mut Frame, area: Rect, title: &str, history: &VecDeque<(f64, f64)>, color: Color) {
    let points: Vec<_> = history.iter().copied().collect();
    let first = points.first().map_or(0.0, |point| point.0);
    let last = points.last().map_or(1.0, |point| point.0).max(first + 1.0);
    let minimum = points
        .iter()
        .map(|point| point.1)
        .reduce(f64::min)
        .unwrap_or(0.0);

    let maximum = points
        .iter()
        .map(|point| point.1)
        .reduce(f64::max)
        .unwrap_or(1.0);

    let padding = ((maximum - minimum) * 0.1)
        .max(maximum.abs() * 0.01)
        .max(0.000001);

    let datasets = vec![
        Dataset::default()
            .graph_type(GraphType::Line)
            .marker(ratatui::symbols::Marker::Braille)
            .style(Style::default().fg(color))
            .data(&points),
    ];

    frame.render_widget(
        Chart::new(datasets)
            .block(panel(title))
            .x_axis(
                Axis::default()
                    .bounds([first, last])
                    .style(Style::default().fg(RAISED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([(minimum - padding).max(0.0), maximum + padding])
                    .labels([format!("{minimum:.3}"), format!("{maximum:.3}")])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}

fn ratio(completed: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (completed as f64 / total as f64).clamp(0.0, 1.0)
    }
}

fn activity_spinner(elapsed: Duration) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

    FRAMES[(elapsed.as_millis() / 100 % FRAMES.len() as u128) as usize]
}

fn progress_label(completed: u64, total: Option<u64>, elapsed: Duration) -> String {
    match total {
        Some(total) => format!(
            "{completed} / {total}  ·  ETA {}",
            eta(completed, total, elapsed)
        ),

        None => format!("{completed} processed  ·  ETA unknown"),
    }
}

fn eta(completed: u64, total: u64, elapsed: Duration) -> String {
    if total == 0 || completed == 0 {
        return "unknown".to_owned();
    }

    let seconds = elapsed.as_secs_f64() * total.saturating_sub(completed) as f64 / completed as f64;
    duration(seconds as u64)
}

fn duration(seconds: u64) -> String {
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds % 3600) / 60,
        seconds % 60
    )
}

fn strip_controls(input: &str) -> String {
    let mut result = String::new();
    let mut chars = input.chars().peekable();

    while let Some(character) = chars.next() {
        if character == '\u{1b}' {
            match chars.next() {
                Some('[') => {
                    for character in chars.by_ref() {
                        if ('@'..='~').contains(&character) {
                            break;
                        }
                    }
                }

                Some(']') => {
                    while let Some(character) = chars.next() {
                        if character == '\u{7}'
                            || (character == '\u{1b}' && chars.peek() == Some(&'\\'))
                        {
                            if character == '\u{1b}' {
                                chars.next();
                            }
                            break;
                        }
                    }
                }

                _ => {}
            }
        } else if !character.is_control() || character == '\t' {
            result.push(character);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    //noinspection SpellCheckingInspection
    #[test]
    fn removes_native_terminal_sequences() {
        assert_eq!(strip_controls("\x1b[31merror\x1b[0m\r"), "error");
        assert_eq!(
            strip_controls("\x1b]8;;https://example.com\x07link\x1b]8;;\x07"),
            "link"
        );
    }

    #[test]
    fn resumed_eta_counts_only_observed_work() {
        let mut state = DisplayState::new();

        state.update(Event::Metric {
            superbatch: 100,
            batch: 50,
            batches_per_superbatch: 100,
            final_superbatch: 101,
            loss: 0.1,
            learning_rate: 0.001,
            positions: 500,
        });

        assert_eq!(state.total_eta(), "unknown");

        state.training_started = Some(Instant::now() - Duration::from_secs(10));

        state.update(Event::Metric {
            superbatch: 100,
            batch: 60,
            batches_per_superbatch: 100,
            final_superbatch: 101,
            loss: 0.09,
            learning_rate: 0.001,
            positions: 600,
        });

        assert_eq!(state.total_eta(), "00:02:20");
    }

    #[test]
    fn renders_training_and_small_terminal() {
        let mut state = DisplayState::new();

        state.update(Event::Phase {
            name: "Training".to_owned(),
            completed: 1,
            total: Some(100),
        });

        state.update(Event::Metric {
            superbatch: 1,
            batch: 50,
            batches_per_superbatch: 100,
            final_superbatch: 2,
            loss: 0.1,
            learning_rate: 0.001,
            positions: 500,
        });

        for (width, height) in [(100, 30), (44, 10), (1, 1)] {
            let backend = ratatui::backend::TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();

            terminal.draw(|frame| draw(frame, &state)).unwrap();
        }
    }

    #[test]
    fn unknown_phase_shows_activity_without_invented_progress() {
        assert_ne!(activity_spinner(Duration::ZERO), activity_spinner(REFRESH));

        let label = progress_label(0, None, Duration::from_secs(3));

        assert!(label.contains("ETA unknown"));
        assert!(!label.contains('%'));

        let mut state = DisplayState::new();
        state.phase = "Filling Bullet's queue".to_owned();

        let backend = ratatui::backend::TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| draw(frame, &state)).unwrap();

        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();

        assert!(text.contains("ETA unknown"));
        assert!(text.contains("Filling Bullet's queue"));
        assert!(
            text.chars()
                .any(|character| ('\u{2800}'..='\u{28ff}').contains(&character))
        );
    }

    #[test]
    fn late_plain_messages_preserve_results_and_completion() {
        let (sender, receiver) = mpsc::channel();

        sender
            .send(Event::Note {
                message: "Saved converted.bullet".to_owned(),
            })
            .unwrap();

        sender
            .send(Event::Note {
                message: "\x1b[32mCheck passed\x1b[0m".to_owned(),
            })
            .unwrap();

        sender.send(Event::Finished).unwrap();

        drop(sender);

        let mut state = DisplayState::new();
        let mut output = Vec::new();
        drain_final_plain(&receiver, &mut state, &mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "Saved converted.bullet\nCheck passed\nComplete.\n"
        );
        assert!(state.finished);
    }

    #[test]
    fn histories_and_diagnostics_are_bounded() {
        let mut state = DisplayState::new();

        for _ in 0..1000 {
            state.note("note".to_owned());
        }

        assert_eq!(state.messages.len(), MESSAGE_LIMIT);
        assert_eq!(ratio(200, 100), 1.0);
        assert_eq!(eta(0, 100, Duration::from_secs(1)), "unknown");
    }
}
