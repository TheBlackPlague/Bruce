use std::{
    ffi::c_int,
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    mem, ptr,
    sync::{Mutex, mpsc::Sender},
    thread,
};

use anstream::adapter::strip_str;
use crossterm::{
    Command,
    cursor::{Hide, MoveToPreviousLine, Show},
    terminal::{self, BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate},
};

use crate::control::lock;

static TYPING: Mutex<Option<platform::Mode>> = Mutex::new(None);

pub struct Screen
{
    console: File,
    input: File,
    saved: Vec<(c_int, c_int)>,
    block: Block,
    closed: bool,
}

impl Screen
{
    pub fn open(stray: Sender<String>) -> io::Result<Self>
    {
        let (console, input) = (device(platform::OUTPUT)?, device(platform::INPUT)?);
        let mode = platform::open(&console, &input)?;
        let mut screen = Self { console, input, saved: Vec::new(), block: Block::default(), closed: false };
        *lock(&TYPING) = Some(mode);
        platform::set_mode(&screen.input, platform::quiet(mode))?;
        flush();
        for (stream, name) in [(1, "stdout"), (2, "stderr")] {
            let (reader, writer) = io::pipe()?;
            let stray = stray.clone();
            thread::Builder::new().name(format!("bruce-{name}")).spawn(move || {
                for line in BufReader::new(reader).split(b'\n').map_while(Result::ok) {
                    let text = String::from_utf8_lossy(&line);
                    let text = text.trim_end();
                    if !text.is_empty() {
                        let _ = stray.send(format!("{name}: {text}"));
                    }
                }
            })?;
            let pipe = platform::descriptor(writer)?;
            screen.saved.push((stream, duplicate(stream).inspect_err(|_| close(pipe))?));
            let swapped = swap(stream, pipe);
            close(pipe);
            swapped?;
        }
        platform::write(&screen.console, &code(Hide))?;
        Ok(screen)
    }

    pub fn size(&self) -> (u16, u16) { terminal::size().unwrap_or((80, 24)) }

    pub fn draw(&mut self, lines: &[String], size: (u16, u16)) -> io::Result<()>
    {
        flush();
        platform::write(&self.console, &self.block.redraw(lines, size))
    }

    pub fn close(&mut self) -> io::Result<()>
    {
        if mem::replace(&mut self.closed, true) {
            return Ok(());
        }

        let cleared = platform::write(&self.console, &self.block.erase(self.size().0));
        let mode = lock(&TYPING).take();
        let restored = platform::close(&self.input, mode);
        flush();
        for (stream, saved) in self.saved.drain(..) {
            let _ = swap(stream, saved);
            close(saved);
        }
        cleared.and(restored)
    }
}

impl Drop for Screen
{
    fn drop(&mut self) { let _ = self.close(); }
}

pub fn restore()
{
    let Some(mode) = *lock(&TYPING) else { return };
    let _ = device(platform::INPUT).and_then(|input| platform::set_mode(&input, mode));
    let _ = device(platform::OUTPUT).and_then(|console| platform::write(&console, &code(Show)));
}

fn device(path: &str) -> io::Result<File> { OpenOptions::new().read(true).write(true).open(path) }

fn code(command: impl Command) -> String
{
    let mut text = String::new();
    let _ = command.write_ansi(&mut text);
    text
}

#[derive(Default)]
struct Block
{
    size: (u16, u16),
    drawn: Vec<usize>,
    scrolled: usize,
}

impl Block
{
    fn redraw(&mut self, lines: &[String], size: (u16, u16)) -> String
    {
        let (columns, visible) = (usize::from(size.0).max(1), usize::from(size.1).saturating_sub(1));
        let lines = &lines[lines.len().saturating_sub(visible)..];

        let mut frame = code(BeginSynchronizedUpdate);
        let above = self.rewind(&mut frame, columns);
        if size != self.size {
            frame += &code(Clear(ClearType::FromCursorDown));
            let mut hidden = above.saturating_sub(visible);
            let scrolled = self.drawn.iter().take_while(|&&width| {
                let kept = hidden > 0;
                hidden = hidden.saturating_sub(rows(width, columns));
                kept
            });
            self.scrolled = scrolled.count();
        }
        for line in lines {
            frame += line;
            frame += &code(Clear(ClearType::UntilNewLine));
            frame += "\r\n";
        }
        frame += &code(Clear(ClearType::FromCursorDown));
        frame += &code(EndSynchronizedUpdate);

        self.drawn.truncate(self.scrolled);
        self.drawn.extend(lines.iter().map(|line| strip_str(line).map(|text| text.chars().count()).sum::<usize>()));
        self.size = size;
        frame
    }

    fn erase(&self, columns: u16) -> String
    {
        let mut frame = String::new();
        self.rewind(&mut frame, usize::from(columns).max(1));
        frame + &code(Clear(ClearType::FromCursorDown)) + &code(Show)
    }

    fn rewind(&self, frame: &mut String, columns: usize) -> usize
    {
        let above: usize = self.drawn.iter().map(|&width| rows(width, columns)).sum();
        frame.push('\r');
        if above > 0 {
            *frame += &code(MoveToPreviousLine(u16::try_from(above).unwrap_or(u16::MAX)));
        }
        above
    }
}

fn rows(width: usize, columns: usize) -> usize { width.div_ceil(columns).max(1) }

#[expect(unsafe_code)]
fn flush()
{
    let _ = io::stdout().flush();
    // SAFETY: a null stream asks the C runtime to flush every output stream it has open.
    unsafe { libc::fflush(ptr::null_mut()) };
}

#[expect(unsafe_code)]
fn duplicate(stream: c_int) -> io::Result<c_int>
{
    // SAFETY: the descriptor is open; dup adds a new entry to the descriptor table and changes no existing one.
    let copy = unsafe { libc::dup(stream) };
    check(copy >= 0).map(|()| copy)
}

#[expect(unsafe_code)]
fn swap(stream: c_int, source: c_int) -> io::Result<()>
{
    // SAFETY: both descriptors are open, and no Rust value owns the standard stream being replaced.
    check(unsafe { libc::dup2(source, stream) } >= 0)?;
    platform::adopt(stream)
}

#[expect(unsafe_code)]
fn close(descriptor: c_int)
{
    // SAFETY: the descriptor belongs to this module and is not used after this call.
    unsafe { libc::close(descriptor) };
}

fn check(succeeded: bool) -> io::Result<()> { succeeded.then_some(()).ok_or_else(io::Error::last_os_error) }

#[cfg(windows)]
#[expect(unsafe_code)]
mod platform
{
    use std::{
        ffi::c_int,
        fs::File,
        io::{self, PipeWriter},
        os::windows::io::{AsRawHandle, IntoRawHandle},
        ptr,
    };

    use windows_sys::Win32::{
        Foundation::HANDLE,
        System::Console::{
            ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING,
            FlushConsoleInputBuffer, GetConsoleMode, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE, SetConsoleMode, SetStdHandle,
            WriteConsoleW,
        },
    };

    use super::check;

    pub type Mode = u32;

    pub const OUTPUT: &str = "CONOUT$";
    pub const INPUT: &str = "CONIN$";

    pub const fn quiet(mode: Mode) -> Mode
    {
        (mode & !(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT)) | ENABLE_PROCESSED_INPUT
    }

    pub fn open(console: &File, input: &File) -> io::Result<Mode>
    {
        mode(&io::stderr()).map_err(|_| io::Error::other("stderr is not a console"))?;
        set_mode(console, mode(console)? | ENABLE_VIRTUAL_TERMINAL_PROCESSING)?;
        // SAFETY: looking up a descriptor's handle only reads the C runtime's descriptor table.
        if unsafe { libc::get_osfhandle(1) == libc::get_osfhandle(2) } {
            return Err(io::Error::other("stdout and stderr share one handle"));
        }
        mode(input)
    }

    pub fn close(input: &File, typing: Option<Mode>) -> io::Result<()>
    {
        // SAFETY: the handle is open.
        unsafe { FlushConsoleInputBuffer(input.as_raw_handle()) };
        typing.map_or(Ok(()), |typing| set_mode(input, typing))
    }

    fn mode(console: &impl AsRawHandle) -> io::Result<Mode>
    {
        let mut mode = 0;
        // SAFETY: the handle is open and `mode` outlives the call.
        check(unsafe { GetConsoleMode(console.as_raw_handle(), &raw mut mode) } != 0).map(|()| mode)
    }

    pub fn set_mode(console: &File, mode: Mode) -> io::Result<()>
    {
        // SAFETY: the handle is open.
        check(unsafe { SetConsoleMode(console.as_raw_handle(), mode) } != 0)
    }

    pub fn write(console: &File, text: &str) -> io::Result<()>
    {
        let (handle, units) = (console.as_raw_handle(), text.encode_utf16().collect::<Vec<_>>());
        let mut rest = &units[..];
        while !rest.is_empty() {
            let mut written = 0;
            // SAFETY: the handle is open and `rest` is valid for the length passed with it.
            let wrote =
                unsafe { WriteConsoleW(handle, rest.as_ptr(), rest.len() as u32, &raw mut written, ptr::null()) };
            check(wrote != 0 && written != 0)?;
            rest = &rest[(written as usize).min(rest.len())..];
        }
        Ok(())
    }

    pub fn descriptor(writer: PipeWriter) -> io::Result<c_int>
    {
        // SAFETY: the handle is open; the descriptor owns it from here on, so the writer must not close it.
        let descriptor = unsafe { libc::open_osfhandle(writer.as_raw_handle() as isize, 0) };
        check(descriptor >= 0)?;
        let _ = writer.into_raw_handle();
        Ok(descriptor)
    }

    pub fn adopt(stream: c_int) -> io::Result<()>
    {
        let standard = if stream == 1 { STD_OUTPUT_HANDLE } else { STD_ERROR_HANDLE };
        // SAFETY: the descriptor is open, so the C runtime returns the handle behind it.
        let handle = unsafe { libc::get_osfhandle(stream) };
        // SAFETY: the handle stays open for as long as the descriptor does.
        check(unsafe { SetStdHandle(standard, handle as HANDLE) } != 0)
    }
}

#[cfg(unix)]
#[expect(unsafe_code)]
mod platform
{
    use std::{
        ffi::c_int,
        fs::File,
        io::{self, PipeWriter, Write},
        mem::MaybeUninit,
        os::fd::{AsRawFd, IntoRawFd},
    };

    use super::check;

    pub type Mode = libc::termios;

    pub const OUTPUT: &str = "/dev/tty";
    pub const INPUT: &str = "/dev/tty";

    pub const fn quiet(mut mode: Mode) -> Mode
    {
        mode.c_lflag = (mode.c_lflag & !(libc::ECHO | libc::ICANON)) | libc::ISIG;
        mode
    }

    pub fn open(_: &File, terminal: &File) -> io::Result<Mode>
    {
        let mut mode = MaybeUninit::uninit();
        // SAFETY: the descriptor is open and `mode` is valid for one termios.
        check(unsafe { libc::tcgetattr(terminal.as_raw_fd(), mode.as_mut_ptr()) } == 0)?;
        // SAFETY: a successful tcgetattr initialised the whole structure.
        Ok(unsafe { mode.assume_init() })
    }

    pub fn close(terminal: &File, typing: Option<Mode>) -> io::Result<()>
    {
        // SAFETY: the descriptor is open.
        unsafe { libc::tcflush(terminal.as_raw_fd(), libc::TCIFLUSH) };
        typing.map_or(Ok(()), |typing| set_mode(terminal, typing))
    }

    pub fn set_mode(terminal: &File, mode: Mode) -> io::Result<()>
    {
        // SAFETY: the descriptor is open and `mode` is a valid termios.
        check(unsafe { libc::tcsetattr(terminal.as_raw_fd(), libc::TCSANOW, &raw const mode) } == 0)
    }

    pub fn write(mut terminal: &File, text: &str) -> io::Result<()> { terminal.write_all(text.as_bytes()) }

    pub fn descriptor(writer: PipeWriter) -> io::Result<c_int> { Ok(writer.into_raw_fd()) }

    pub fn adopt(_: c_int) -> io::Result<()> { Ok(()) }
}
