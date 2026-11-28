use std::{
    collections::BTreeMap,
    fmt,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    mem,
    ops::ControlFlow,
    panic, slice,
    str::FromStr,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel},
    },
    thread::{self, ScopedJoinHandle},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{
    DataError, Game, Source,
    game::record_len,
    select::{Counters, Scratch, Selection},
};
use crate::control::{Control, Stop, contain, lock};

pub const BLOCK: u64 = 4 << 20;

const PROBE: ScanOptions = ScanOptions { sampling: Sampling::All, threads: 4, in_flight: None, block: BLOCK };
pub(super) const CREDITS: usize = 4;
pub(super) const TICK: Duration = Duration::from_millis(20);
const RECORD_LIMIT: u64 = 128 << 20;
const AHEAD: u64 = 64 << 20;
const REQUEST: usize = 1 << 20;

type Stopping<'a> = &'a (dyn Fn() -> bool + Sync);
type Decoded<T, E> = (u64, usize, Option<Result<(BlockMeta, T), E>>);

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum Sampling
{
    All,
    Head
    {
        games: u64,
    },
}

impl Sampling
{
    pub fn head(self) -> Option<u64> { if let Self::Head { games } = self { Some(games) } else { None } }
}

impl FromStr for Sampling
{
    type Err = DataError;

    fn from_str(value: &str) -> Result<Self, Self::Err>
    {
        let fields: Vec<&str> = value.split(':').collect();
        let games = match fields[..] {
            ["all"] => return Ok(Self::All),
            ["head", games] if !games.starts_with('+') => games.parse().ok().filter(|&games| games > 0),
            _ => None,
        };

        games.map(|games| Self::Head { games }).ok_or_else(|| {
            DataError::new(format!("Invalid sampling '{value}'; expected all or head:<GAMES> with GAMES at least 1"))
        })
    }
}

impl fmt::Display for Sampling
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match *self {
            Self::All => write!(formatter, "all"),
            Self::Head { games } => write!(formatter, "head:{games}"),
        }
    }
}

pub fn splitmix64(mut x: u64) -> u64
{
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub struct ScanOptions
{
    pub sampling: Sampling,
    pub threads: usize,
    pub in_flight: Option<usize>,
    pub block: u64,
}

pub struct Block<'a>
{
    pub source: usize,
    pub epoch: u64,
    pub games: &'a [Game],
}

pub fn scan<T: Send, E: From<DataError> + Send>(
    sources: &[Source],
    options: &ScanOptions,
    control: &Control,
    work: impl Fn(&mut Scratch, &Block<'_>) -> Result<T, E> + Sync,
    sink: impl FnMut(&BlockMeta, T) -> Result<ControlFlow<()>, E>,
) -> Result<bool, E>
{
    let mut plan = Plan::new(sources.iter().map(length).collect::<Result<_, _>>()?, false);
    run(sources, options, &mut plan, &|| control.stop() != Stop::Running, work, sink)
}

pub fn accepts_any(source: &Source, selection: &Selection, control: &Control) -> Result<bool, DataError>
{
    let mut accepted = false;
    scan(
        slice::from_ref(source),
        &PROBE,
        control,
        |scratch, block| -> Result<bool, DataError> {
            let mut counters = Counters::default();
            for game in block.games {
                selection.select_each(game, Some(&mut counters), scratch, |_, _| {})?;
                if counters.accepted > 0 {
                    return Ok(true);
                }
            }

            Ok(false)
        },
        |_, found| {
            accepted = found;
            Ok(if found { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
        },
    )?;

    Ok(accepted)
}

fn length(source: &Source) -> Result<u64, DataError>
{
    fs::metadata(&source.path).map(|metadata| metadata.len()).map_err(|error| DataError::io(&error).at(source))
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Cursor
{
    pub epoch: u64,
    pub offsets: Vec<u64>,
    pub games: Vec<u64>,
}

pub struct Plan
{
    cursor: Cursor,
    lengths: Vec<u64>,
    interleave: bool,
}

impl Plan
{
    pub fn new(lengths: Vec<u64>, interleave: bool) -> Self
    {
        let zeros = vec![0; lengths.len()];
        Self { cursor: Cursor { epoch: 0, offsets: zeros.clone(), games: zeros }, lengths, interleave }
    }

    pub fn resume(lengths: Vec<u64>, cursor: &Cursor) -> Self
    {
        Self { cursor: cursor.clone(), lengths, interleave: true }
    }

    fn next(&mut self) -> Option<usize>
    {
        let source = self.behind();
        if source.is_some() || !self.interleave || self.lengths.iter().all(|&length| length == 0) {
            return source;
        }

        self.cursor.epoch += 1;
        self.cursor.offsets.fill(0);
        self.cursor.games.fill(0);
        self.behind()
    }

    fn behind(&self) -> Option<usize>
    {
        let progress = |source: usize| (u128::from(self.cursor.offsets[source]), u128::from(self.lengths[source]));
        let order = |&left: &usize, &right: &usize| {
            let ((consumed_left, total_left), (consumed_right, total_right)) = (progress(left), progress(right));
            (consumed_left * total_right).cmp(&(consumed_right * total_left))
        };

        let mut open = (0..self.lengths.len()).filter(|&source| self.cursor.offsets[source] < self.lengths[source]);
        if self.interleave { open.min_by(order) } else { open.next() }
    }

    fn advance(&mut self, source: usize, bytes: u64, games: u64)
    {
        self.cursor.offsets[source] += bytes;
        self.cursor.games[source] += games;
    }

    pub fn cursor(&self) -> Cursor { self.cursor.clone() }
}

pub struct BlockMeta
{
    pub source: usize,
    pub sequence: u64,
    pub offset: u64,
    pub end: u64,
    pub cursor: Cursor,
    pub games: u32,
}

pub fn run<T: Send, E: From<DataError> + Send>(
    sources: &[Source],
    options: &ScanOptions,
    plan: &mut Plan,
    stop: Stopping<'_>,
    work: impl Fn(&mut Scratch, &Block<'_>) -> Result<T, E> + Sync,
    mut sink: impl FnMut(&BlockMeta, T) -> Result<ControlFlow<()>, E>,
) -> Result<bool, E>
{
    let threads = options.threads;
    let credits = options.in_flight.map_or(CREDITS * threads, |cap| cap.max(1));
    let head = options.sampling.head();
    let inputs = sources.iter().map(|source| SourceFile::open(source, head)).collect::<Result<Vec<_>, _>>()?;

    let halt = AtomicBool::new(false);
    let stopping = || {
        let stopped = halt.load(Ordering::Acquire) || stop();
        if stopped {
            halt.store(true, Ordering::Release);
        }

        stopped
    };
    let stages = Stages {
        sources,
        options,
        reached: sources.iter().map(|_| AtomicBool::new(false)).collect(),
        pool: Pool::default(),
        stopping: &stopping,
    };

    let (blocks, queue) = sync_channel(credits);
    let (results, ordered) = sync_channel(credits);
    let (returned, available) = sync_channel(credits);
    for _ in 0..credits {
        let _ = returned.try_send(());
    }

    let queue = Mutex::new(queue);

    thread::scope(|scope| {
        let halted = OnDrop(&halt);

        let reader = {
            let (stages, results) = (&stages, results.clone());
            scope.spawn(move || stages.read(plan, inputs, &available, &blocks, &results))
        };
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let (stages, queue, work, results) = (&stages, &queue, &work, results.clone());

                scope.spawn(move || stages.process(queue, work, &results))
            })
            .collect();
        drop(results);

        let released = stages.order(&ordered, &returned, &mut sink);
        drop(halted);
        drop((ordered, returned));

        let sent = join(reader);
        workers.into_iter().for_each(join);
        let released = released?;

        Ok(released.is_none() || released != sent)
    })
}

fn join<T>(handle: ScopedJoinHandle<'_, T>) -> T
{
    handle.join().unwrap_or_else(|payload| panic::resume_unwind(payload))
}

fn receive<T>(channel: &Receiver<T>, stopping: Stopping<'_>) -> Option<T>
{
    while !stopping() {
        match channel.recv_timeout(TICK) {
            Err(RecvTimeoutError::Timeout) => {},
            received => return received.ok(),
        }
    }

    None
}

pub(super) struct OnDrop<'a>(pub &'a AtomicBool);

impl Drop for OnDrop<'_>
{
    fn drop(&mut self) { self.0.store(true, Ordering::Release); }
}

pub(super) struct Pool<T>(Mutex<Vec<Vec<T>>>);

impl<T> Default for Pool<T>
{
    fn default() -> Self { Self(Mutex::default()) }
}

impl<T> Pool<T>
{
    pub(super) fn take(&self) -> Vec<T> { lock(&self.0).pop().unwrap_or_default() }

    pub(super) fn put(&self, mut buffer: Vec<T>)
    {
        buffer.clear();
        lock(&self.0).push(buffer);
    }
}

struct Raw
{
    meta: BlockMeta,
    bytes: Vec<u8>,
}

struct SourceFile
{
    file: File,
    limit: Option<u64>,
    length: u64,
    position: u64,
    records: u64,
    carry: Vec<u8>,
    ahead: Vec<u8>,
    taken: usize,
}

impl SourceFile
{
    fn open(source: &Source, limit: Option<u64>) -> Result<Self, DataError>
    {
        File::open(&source.path)
            .and_then(|file| {
                let length = file.metadata()?.len();
                Ok(Self {
                    file,
                    limit,
                    length,
                    position: 0,
                    records: 0,
                    carry: Vec::new(),
                    ahead: Vec::new(),
                    taken: 0,
                })
            })
            .map_err(|error| DataError::io(&error).at(source))
    }

    fn admits(&self, index: u64) -> bool { self.limit.is_none_or(|limit| index < limit) }

    fn cut(&mut self, source: &Source, start: u64, block: u64, pool: &Pool<u8>) -> Result<(Vec<u8>, u64), DataError>
    {
        let located = |error: DataError, offset: u64| error.at(source).offset(offset);
        if self.carry.capacity() == 0 {
            self.carry = pool.take();
        }

        let (length, count) = loop {
            let boundary = (self.position / block + 1).saturating_mul(block);
            let to = boundary.min(self.length.max(self.position));
            self.append(to).map_err(|error| located(DataError::io(&error), self.position))?;

            let at_end = to >= self.length;
            let (mut length, mut count) = (0, 0);
            while self.admits(self.records + count)
                && let Some(record) = record_len(&self.carry[length..])
            {
                length += record;
                count += 1;
            }

            let limited = !self.admits(self.records + count);
            if at_end && !limited && (length == 0 || length < self.carry.len()) {
                return Err(located(DataError::truncated(), start + length as u64).game(self.records + count));
            }

            if length > 0 {
                break (length, count);
            }

            if self.carry.len() as u64 >= RECORD_LIMIT {
                let reason = format!("Invalid data: no record ends within {} bytes", self.carry.len());
                return Err(located(DataError::new(reason), start));
            }
        };

        let mut bytes = mem::take(&mut self.carry);
        if length < bytes.len() {
            self.carry = pool.take();
            self.carry.extend_from_slice(&bytes[length..]);
            bytes.truncate(length);
        }

        self.records += count;
        Ok((bytes, count))
    }

    fn append(&mut self, to: u64) -> io::Result<()>
    {
        let wanted = usize::try_from(to - self.position).map_err(io::Error::other)?;
        if self.ahead.len() - self.taken < wanted {
            self.ahead.drain(..self.taken);
            self.taken = 0;

            let (held, from) = (self.ahead.len(), self.position + self.ahead.len() as u64);
            let end = (from / AHEAD + 1).saturating_mul(AHEAD).min(self.length).max(to);
            self.ahead.resize(held + usize::try_from(end - from).map_err(io::Error::other)?, 0);

            let mut file = &self.file;
            file.seek(SeekFrom::Start(from))?;
            for piece in self.ahead[held..].chunks_mut(REQUEST) {
                file.read_exact(piece)?;
            }
        }

        self.carry.reserve_exact(wanted);
        self.carry.extend_from_slice(&self.ahead[self.taken..self.taken + wanted]);
        self.taken += wanted;
        self.position = to;
        if to >= self.length {
            (self.ahead, self.taken) = (Vec::new(), 0);
        }

        Ok(())
    }
}

#[derive(Default)]
struct Games
{
    games: Vec<Game>,
    len: usize,
}

impl Games
{
    fn clear(&mut self) { self.len = 0; }

    fn fill(&mut self, write: impl FnOnce(&mut Game) -> Result<(), DataError>) -> Result<(), DataError>
    {
        if self.len == self.games.len() {
            self.games.push(Game::default());
        }

        write(&mut self.games[self.len])?;
        self.len += 1;

        Ok(())
    }

    fn as_slice(&self) -> &[Game] { &self.games[..self.len] }
}

struct Stages<'a>
{
    sources: &'a [Source],
    options: &'a ScanOptions,
    reached: Vec<AtomicBool>,
    pool: Pool<u8>,
    stopping: Stopping<'a>,
}

impl Stages<'_>
{
    fn read<T, E: From<DataError>>(
        &self,
        plan: &mut Plan,
        mut inputs: Vec<SourceFile>,
        credits: &Receiver<()>,
        blocks: &SyncSender<Raw>,
        results: &SyncSender<Decoded<T, E>>,
    ) -> Option<u64>
    {
        let (mut issued, mut epoch) = (0, None);

        while let Some(source) = plan.next() {
            let cursor = plan.cursor();
            if epoch != Some(cursor.epoch) {
                for (input, (&offset, &games)) in inputs.iter_mut().zip(cursor.offsets.iter().zip(&cursor.games)) {
                    (input.position, input.records, input.taken) = (offset, games, 0);
                    input.carry.clear();
                    input.ahead.clear();
                }
                epoch = Some(cursor.epoch);
            }

            let (input, offset) = (&mut inputs[source], cursor.offsets[source]);
            if input.admits(input.records) && !self.reached[source].load(Ordering::Acquire) {
                receive(credits, self.stopping)?;
                let sequence = issued;
                issued += 1;

                match input.cut(&self.sources[source], offset, self.options.block, &self.pool) {
                    Ok((bytes, records)) => {
                        let end = offset + bytes.len() as u64;
                        let meta = BlockMeta { source, sequence, offset, end, cursor, games: 0 };
                        blocks.send(Raw { meta, bytes }).ok()?;
                        plan.advance(source, end - offset, records);
                        continue;
                    },
                    Err(error) => results.send((sequence, source, Some(Err(error.into())))).ok()?,
                }
            } else if (self.stopping)() {
                return None;
            }

            plan.advance(source, plan.lengths[source].saturating_sub(offset), 0);
            self.pool.put(mem::take(&mut input.carry));
            (input.ahead, input.taken) = (Vec::new(), 0);
        }

        Some(issued)
    }

    fn process<T, E: From<DataError>>(
        &self,
        queue: &Mutex<Receiver<Raw>>,
        work: &impl Fn(&mut Scratch, &Block<'_>) -> Result<T, E>,
        results: &SyncSender<Decoded<T, E>>,
    )
    {
        let next = || receive(&lock(queue), self.stopping);
        let (mut games, mut scratch) = (Games::default(), Scratch::default());

        while let Some(Raw { mut meta, bytes }) = next() {
            let (sequence, source) = (meta.sequence, meta.source);
            let result = if self.reached[source].load(Ordering::Acquire) {
                None
            } else {
                let worked = contain(|| -> Result<Option<T>, E> {
                    if !self.decode(&meta, &bytes, &mut games)? {
                        return Ok(None);
                    }

                    let block = Block { source, epoch: meta.cursor.epoch, games: games.as_slice() };
                    work(&mut scratch, &block).map(Some)
                });
                let worked = worked.unwrap_or_else(|contained| {
                    let panic = DataError::new(format!("Block processing failed unexpectedly: {contained}"));
                    Err(panic.at(&self.sources[source]).offset(meta.offset).into())
                });
                let Some(value) = worked.transpose() else { return };

                meta.games = u32::try_from(games.as_slice().len()).unwrap_or(u32::MAX);
                Some(value.map(|value| (meta, value)))
            };

            self.pool.put(bytes);
            if results.send((sequence, source, result)).is_err() {
                return;
            }
        }
    }

    fn decode(&self, meta: &BlockMeta, mut bytes: &[u8], games: &mut Games) -> Result<bool, DataError>
    {
        let source = &self.sources[meta.source];
        games.clear();

        let (mut offset, mut index) = (meta.offset, meta.cursor.games[meta.source]);
        while !bytes.is_empty() {
            if (self.stopping)() {
                return Ok(false);
            }

            let length = bytes.len();
            let read = games.fill(|game| {
                game.read(&mut bytes, source.scale)?;
                (game.source, game.index, game.offset) = (meta.source, index, offset);
                Ok(())
            });
            read.map_err(|error| error.at(source).offset(offset).game(index))?;

            offset += (length - bytes.len()) as u64;
            index += 1;
        }

        Ok(true)
    }

    fn order<T, E>(
        &self,
        ordered: &Receiver<Decoded<T, E>>,
        returned: &SyncSender<()>,
        sink: &mut impl FnMut(&BlockMeta, T) -> Result<ControlFlow<()>, E>,
    ) -> Result<Option<u64>, E>
    {
        let mut pending = BTreeMap::<u64, Decoded<T, E>>::new();
        let mut counted = vec![0; self.reached.len()];
        let mut next = 0;

        loop {
            while let Some((_, source, result)) = pending.remove(&next) {
                if (self.stopping)() {
                    return Ok(None);
                }

                if !self.reached[source].load(Ordering::Acquire)
                    && let Some(result) = result
                {
                    let (meta, value) = result?;
                    counted[source] += u64::from(meta.games);
                    if self.options.sampling.head().is_some_and(|games| counted[source] >= games) {
                        self.reached[source].store(true, Ordering::Release);
                    }

                    if sink(&meta, value)?.is_break() {
                        return Ok(None);
                    }
                }

                next += 1;
                let _ = returned.try_send(());
            }

            match ordered.recv_timeout(TICK) {
                Ok(decoded) => {
                    pending.insert(decoded.0, decoded);
                },
                Err(RecvTimeoutError::Timeout) if (self.stopping)() => return Ok(None),
                Err(RecvTimeoutError::Timeout) => {},
                Err(RecvTimeoutError::Disconnected) => return Ok(Some(next)),
            }
        }
    }
}
