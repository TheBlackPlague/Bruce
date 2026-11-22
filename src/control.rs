use std::{
    any::Any,
    backtrace::Backtrace,
    cell::Cell,
    io,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

static PANIC: Mutex<Option<(String, String)>> = Mutex::new(None);

thread_local! {
    static CONTAINING: Cell<u32> = const { Cell::new(0) };
}

const LADDER: [Stop; 6] = [Stop::Running, Stop::AtBoundary, Stop::Committed, Stop::Armed, Stop::Now, Stop::Exit];
#[cfg(windows)]
const FLUSH_WAIT: std::time::Duration = std::time::Duration::from_secs(4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stop
{
    Running,
    AtBoundary,
    Committed,
    Armed,
    Now,
    Exit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome
{
    Finished,
    Stopped,
    Aborted,
    Failed,
}

impl Outcome
{
    pub const fn name(self) -> &'static str
    {
        match self {
            Self::Finished => "finished",
            Self::Stopped => "stopped",
            Self::Aborted => "aborted",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Default)]
pub struct Control
{
    state: AtomicU8,
    closing: AtomicBool,
    flushed: Mutex<bool>,
    released: Condvar,
}

impl Control
{
    pub fn stop(&self) -> Stop { LADDER[usize::from(self.state.load(Ordering::Acquire))] }

    pub fn halted(&self) -> bool
    {
        let halted = self.stop() != Stop::Running;
        if halted {
            self.commit();
        }

        halted
    }

    pub fn commit(&self) -> bool
    {
        let (from, to) = (Stop::AtBoundary as u8, Stop::Committed as u8);
        self.state.compare_exchange(from, to, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    pub fn interrupt(&self) { self.step(true); }

    pub fn abort(&self) { self.step(false); }

    pub fn outcome(&self) -> Outcome
    {
        match self.stop() {
            Stop::Running => Outcome::Finished,
            Stop::AtBoundary | Stop::Committed | Stop::Armed => Outcome::Stopped,
            Stop::Now | Stop::Exit => Outcome::Aborted,
        }
    }

    pub fn closing(&self) -> bool { self.closing.load(Ordering::Acquire) }

    pub fn flushed(&self)
    {
        *lock(&self.flushed) = true;
        self.released.notify_all();
    }

    fn step(&self, interrupt: bool)
    {
        self.state.update(Ordering::AcqRel, Ordering::Acquire, |state| {
            let next = match (LADDER[usize::from(state)], interrupt) {
                (Stop::Running, true) => Stop::AtBoundary,
                (Stop::Committed, true) => Stop::Armed,
                (Stop::Exit, _) | (Stop::Armed | Stop::Now, true) => Stop::Exit,
                (Stop::AtBoundary, true) | (_, false) => Stop::Now,
            };
            next as u8
        });
    }

    #[cfg(windows)]
    fn close(&self)
    {
        self.abort();
        self.closing.store(true, Ordering::Release);
        drop(self.released.wait_timeout_while(lock(&self.flushed), FLUSH_WAIT, |flushed| !*flushed));
    }
}

pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> { mutex.lock().unwrap_or_else(PoisonError::into_inner) }

pub fn hook()
{
    panic::set_hook(Box::new(|info| {
        if !CONTAINING.try_with(Cell::get).is_ok_and(|depth| depth > 0) {
            crate::screen::restore();
        }

        let mut place = panic_message(info.payload());
        if let Some(at) = info.location() {
            place = format!("{place} at {}:{}:{}", at.file().replace('\\', "/"), at.line(), at.column());
        }
        *lock(&PANIC) = Some((place, Backtrace::force_capture().to_string()));
    }));
}

pub fn panicked() -> Option<(String, String)> { lock(&PANIC).take() }

pub fn contain<T>(f: impl FnOnce() -> T) -> Result<T, String>
{
    CONTAINING.with(|depth| depth.set(depth.get() + 1));
    let result = panic::catch_unwind(AssertUnwindSafe(f)).map_err(|payload| panic_message(payload.as_ref()));
    CONTAINING.with(|depth| depth.set(depth.get() - 1));
    result
}

pub fn panic_message(payload: &(dyn Any + Send)) -> String
{
    let text = payload.downcast_ref::<&str>().copied();
    payload.downcast_ref::<String>().cloned().unwrap_or_else(|| text.unwrap_or("unknown panic").to_owned())
}

#[cfg(unix)]
pub fn install(control: &Arc<Control>) -> io::Result<()>
{
    use signal_hook::{
        consts::{SIGHUP, SIGINT, SIGTERM},
        iterator::Signals,
    };

    let control = Arc::clone(control);
    let mut signals = Signals::new([SIGINT, SIGTERM, SIGHUP])?;
    std::thread::Builder::new().name("bruce-signals".into()).spawn(move || {
        for signal in signals.forever() {
            if signal == SIGINT { control.interrupt() } else { control.abort() }
        }
    })?;

    Ok(())
}

#[cfg(windows)]
pub fn install(control: &Arc<Control>) -> io::Result<()>
{
    use windows_sys::Win32::{
        Foundation::{FALSE, TRUE},
        System::Console::{
            CTRL_BREAK_EVENT, CTRL_C_EVENT, CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT,
            SetConsoleCtrlHandler,
        },
    };

    static TARGET: std::sync::OnceLock<Arc<Control>> = std::sync::OnceLock::new();

    extern "system" fn handler(event: u32) -> windows_sys::core::BOOL
    {
        match (event, TARGET.get()) {
            (CTRL_C_EVENT | CTRL_BREAK_EVENT, Some(control)) => control.interrupt(),
            (CTRL_CLOSE_EVENT | CTRL_LOGOFF_EVENT | CTRL_SHUTDOWN_EVENT, Some(control)) => control.close(),
            _ => return FALSE,
        }

        TRUE
    }

    TARGET.set(Arc::clone(control)).map_err(|_| io::Error::other("console control handling is already installed"))?;

    // SAFETY: a null handler only clears an inherited ignore flag; `handler` lives for the whole program.
    #[expect(unsafe_code)]
    let installed =
        unsafe { SetConsoleCtrlHandler(None, FALSE) != FALSE && SetConsoleCtrlHandler(Some(handler), TRUE) != FALSE };

    if installed { Ok(()) } else { Err(io::Error::last_os_error()) }
}
