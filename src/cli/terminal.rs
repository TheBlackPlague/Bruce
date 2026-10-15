use std::{
    collections::VecDeque,
    io::{self, IsTerminal, Stderr},
    time::Duration,
};

use crossterm::{
    cursor::MoveTo,
    event::{self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{
        BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate, disable_raw_mode, enable_raw_mode,
        is_raw_mode_enabled,
    },
};
use ratatui::{
    TerminalOptions, Viewport,
    backend::CrosstermBackend,
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Paragraph, Widget},
};

use super::{Level, View, render};

pub(super) struct Terminal
{
    screen: ratatui::Terminal<CrosstermBackend<Stderr>>,
    input: Input,
    pending: VecDeque<(String, Level, bool)>,
    height: u16,
    closed: bool,
}

struct Input
{
    keyboard: bool,
    raw: bool,
    paste: bool,
}

impl Terminal
{
    pub(super) fn new() -> io::Result<Self>
    {
        let input = Input { keyboard: false, raw: false, paste: false };
        let backend = CrosstermBackend::new(io::stderr());
        let screen = ratatui::Terminal::with_options(backend, TerminalOptions { viewport: Viewport::Inline(1) })?;

        Ok(Self { screen, input, pending: VecDeque::new(), height: 1, closed: false })
    }

    pub(super) fn keyboard(&self) -> bool { self.input.keyboard }

    pub(super) fn enable_keyboard(&mut self) -> io::Result<()>
    {
        if !self.closed && !self.keyboard() && io::stdin().is_terminal() {
            self.input = Input::new()?;
        }

        Ok(())
    }

    pub(super) fn draw(&mut self, view: &View) -> io::Result<()>
    {
        if self.closed {
            return Ok(());
        }

        execute!(self.screen.backend_mut(), BeginSynchronizedUpdate)?;

        let result = (|| {
            let size = self.screen.size()?;
            let lines = render::live(view, size.width, size.height, self.keyboard());
            let height = lines.len().clamp(1, usize::from(u16::MAX)) as u16;

            self.screen.autoresize()?;

            if height != self.height {
                let origin = self.screen.get_frame().area().as_position();
                execute!(self.screen.backend_mut(), MoveTo(origin.x, origin.y), Clear(ClearType::FromCursorDown))?;
                self.screen.set_cursor_position(origin)?;

                let backend = CrosstermBackend::new(io::stderr());
                let screen =
                    ratatui::Terminal::with_options(backend, TerminalOptions { viewport: Viewport::Inline(height) })?;

                self.screen = screen;
                self.height = height;
            }

            self.flush_messages()?;
            self.screen.draw(|frame| frame.render_widget(Paragraph::new(lines), frame.area()))?;

            let origin = self.screen.get_frame().area().as_position();
            self.screen.set_cursor_position(origin)
        })();

        let end = execute!(self.screen.backend_mut(), EndSynchronizedUpdate);
        result.and(end)
    }

    pub(super) fn write(&mut self, text: &str, level: Level, color: bool) -> io::Result<()>
    {
        if self.closed {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "The live display is closed"));
        }

        self.pending.push_back((super::clean(text), level, color));

        Ok(())
    }

    fn flush_messages(&mut self) -> io::Result<()>
    {
        let area = self.screen.get_frame().area();
        if area.is_empty() {
            return Ok(());
        }

        while let Some((text, level, color)) = self.pending.pop_front() {
            let style = if color {
                match level {
                    Level::Warning => Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    Level::Error => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    _ => Style::default(),
                }
            } else {
                Style::default()
            };

            let lines = render::wrap(&text, usize::from(area.width));
            for chunk in lines.chunks(1024) {
                let lines: Vec<Line<'_>> = chunk.iter().map(|line| Line::styled(line.as_str(), style)).collect();
                self.screen.insert_before(lines.len() as u16, |buffer| {
                    Paragraph::new(lines).render(buffer.area, buffer);
                })?;
            }
        }

        Ok(())
    }

    pub(super) fn poll_stop(&mut self) -> io::Result<bool>
    {
        if !self.keyboard() || self.closed {
            return Ok(false);
        }

        let mut stop = false;
        for _ in 0..64 {
            if !event::poll(Duration::ZERO)? {
                break;
            }

            if let Event::Key(key) = event::read()?
                && matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
            {
                stop |= matches!(key.code, KeyCode::Char('q' | 'Q'))
                    || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL));
            }
        }

        Ok(stop)
    }

    pub(super) fn close(&mut self) -> io::Result<()>
    {
        if self.closed {
            return self.input.close();
        }

        self.closed = true;

        let mut result = execute!(self.screen.backend_mut(), BeginSynchronizedUpdate);
        let flush = self.screen.autoresize().and_then(|()| self.flush_messages());
        result = result.and(flush);

        let origin = self.screen.get_frame().area().as_position();
        let clear = execute!(self.screen.backend_mut(), MoveTo(origin.x, origin.y), Clear(ClearType::FromCursorDown));

        result = result.and(clear);
        result = result.and(self.screen.set_cursor_position(origin));
        result = result.and(self.screen.show_cursor());
        result = result.and(execute!(self.screen.backend_mut(), EndSynchronizedUpdate));
        result = result.and(self.input.close());

        result
    }
}

impl Drop for Terminal
{
    fn drop(&mut self) { let _ = self.close(); }
}

impl Input
{
    fn new() -> io::Result<Self>
    {
        let mut input = Self { keyboard: io::stdin().is_terminal(), raw: false, paste: false };

        if input.keyboard {
            if !is_raw_mode_enabled()? {
                enable_raw_mode()?;
                input.raw = true;
            }

            input.paste = true;
            execute!(io::stderr(), EnableBracketedPaste)?;
        }

        Ok(input)
    }

    fn close(&mut self) -> io::Result<()>
    {
        let mut result = Ok(());

        if self.paste {
            let restore = execute!(io::stderr(), DisableBracketedPaste);
            if restore.is_ok() {
                self.paste = false;
            }

            result = result.and(restore);
        }

        if self.raw {
            let restore = disable_raw_mode();
            if restore.is_ok() {
                self.raw = false;
            }

            result = result.and(restore);
        }

        result
    }
}

impl Drop for Input
{
    fn drop(&mut self) { let _ = self.close(); }
}
