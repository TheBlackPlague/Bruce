use std::{
    collections::BTreeMap,
    mem,
    ops::ControlFlow,
    panic,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel},
    },
    thread,
    time::{Duration, Instant},
};

use bullet_lib::game::formats::bulletformat::ChessBoard;
use bullet_trainer::reader::DataReader;
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{RngCore, SeedableRng},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    DataError, SourceIdentity, is_unit,
    pipeline::{self, BLOCK, Cursor, OnDrop, Plan, Pool, Sampling, ScanOptions, TICK},
    select::{Selection, pass_key},
};
use crate::{config::Data, control::lock, events::LoaderSample};

pub const READER_VERSION: u32 = 1;

const SLICE_BATCHES: usize = 64;
const SHUFFLE_STRIDE: usize = 1 << 16;

#[derive(Default)]
pub struct LoaderStats
{
    pub reading: AtomicU64,
    pub callback: AtomicU64,
    pub handoff: AtomicU64,
}

impl LoaderStats
{
    pub fn add(total: &AtomicU64, duration: Duration)
    {
        total.fetch_add(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> LoaderSample
    {
        let elapsed = |total: &AtomicU64| Duration::from_nanos(total.load(Ordering::Relaxed));
        let (callback, handoff) = (elapsed(&self.callback), elapsed(&self.handoff));
        LoaderSample { blocked: handoff, reading: elapsed(&self.reading), mapping: callback.saturating_sub(handoff) }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Bookmark
{
    pub reader_version: u32,
    pub selection: String,
    pub labelling: String,
    pub window: u64,
    pub start: Cursor,
    pub consumed: u64,
    pub emitted: u64,
}

pub fn fingerprints(data: &Data, seed: u64, identities: &[SourceIdentity]) -> (String, String)
{
    let sources: Vec<Value> = data
        .sources
        .iter()
        .zip(identities)
        .map(|(source, identity)| {
            let mut entry = json!({ "identity": identity });
            if !is_unit(&source.scale) {
                entry["scale"] = json!(source.scale);
            }
            entry
        })
        .collect();
    let dropping = data.labels.drops();

    let selection = json!({
        "reader": READER_VERSION,
        "sources": sources,
        "filter": data.filter,
        "labels": dropping.then_some(data.labels),
        "buffer_size_mb": data.buffer.size_mb,
        "seed": seed,
        "holdout": null,
    });
    let labelling = json!({ "labels": (!dropping).then_some(data.labels) });

    (digest(&selection), digest(&labelling))
}

fn digest(value: &Value) -> String { format!("{:x}", Sha256::digest(value.to_string())) }

fn shuffle(boards: &mut [ChessBoard], seed: u64, window: u64, stop: &dyn Fn() -> bool) -> ControlFlow<()>
{
    let mut random = ChaCha8Rng::seed_from_u64(seed);
    random.set_stream(window);

    let mut end = boards.len();
    while end > 1 {
        if stop() {
            return ControlFlow::Break(());
        }

        let start = end.saturating_sub(SHUFFLE_STRIDE).max(1);
        for index in (start..end).rev() {
            let bound = index as u64 + 1;
            boards.swap(index, ((u128::from(random.next_u64()) * u128::from(bound)) >> 64) as usize);
        }

        end = start;
    }

    ControlFlow::Continue(())
}

struct Window
{
    index: u64,
    start: Cursor,
    next: Cursor,
    boards: Vec<ChessBoard>,
}

struct Shared
{
    data: Data,
    lengths: Vec<u64>,
    keys: Vec<u64>,
    capacity: usize,
    slice: usize,
    superbatch: u64,
    seed: u64,
    resume: Option<Bookmark>,
    fingerprints: (String, String),
    stats: Arc<LoaderStats>,
    finished: AtomicBool,
    asked: AtomicBool,
    failure: Mutex<Option<DataError>>,
    marks: Mutex<BTreeMap<u64, Bookmark>>,
}

#[derive(Clone)]
pub struct TrainingStream(Arc<Shared>);

impl TrainingStream
{
    pub fn new(
        data: &Data,
        seed: u64,
        resume: Option<Bookmark>,
        identities: Option<&[SourceIdentity]>,
    ) -> Result<Self, DataError>
    {
        let identities = identities.map_or_else(|| super::identities(&data.sources), |known| Ok(known.to_vec()))?;

        Ok(Self(Arc::new(Shared {
            data: data.clone(),
            lengths: identities.iter().map(|identity| identity.length).collect(),
            keys: identities.iter().map(SourceIdentity::key).collect(),
            capacity: data.buffer.size_mb * 1024 * 1024 / size_of::<ChessBoard>() / 2,
            slice: SLICE_BATCHES.saturating_mul(data.batch.size),
            superbatch: (data.batch.size * data.batch.per_superbatch) as u64,
            seed,
            resume,
            fingerprints: fingerprints(data, seed, &identities),
            stats: Arc::default(),
            finished: AtomicBool::new(false),
            asked: AtomicBool::new(false),
            failure: Mutex::default(),
            marks: Mutex::default(),
        })))
    }

    pub fn stats(&self) -> Arc<LoaderStats> { Arc::clone(&self.0.stats) }

    pub fn bookmark(&self, superbatch: usize) -> Option<Bookmark>
    {
        let mut marks = lock(&self.0.marks);
        *marks = marks.split_off(&(superbatch as u64));
        marks.remove(&(superbatch as u64))
    }

    pub fn finish(&self) { self.0.finished.store(true, Ordering::Release); }

    pub fn asked(&self) -> bool { self.0.asked.load(Ordering::Acquire) }

    pub fn failure(&self) -> Option<DataError> { lock(&self.0.failure).clone() }
}

impl DataReader<ChessBoard> for TrainingStream
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip: usize, f: F)
    {
        let failure = self.0.run(skip as u64, f).err();
        *lock(&self.0.failure) = failure;
    }
}

impl Shared
{
    fn run(&self, skip: u64, mut f: impl FnMut(&[ChessBoard]) -> bool) -> Result<(), DataError>
    {
        let mut plan = match &self.resume {
            Some(bookmark) => Plan::resume(self.lengths.clone(), &bookmark.start),
            None => Plan::new(self.lengths.clone(), true),
        };

        let (windows, ready) = sync_channel(0);
        let (spares, spare) = sync_channel(2);
        let done = AtomicBool::new(false);

        thread::scope(|scope| {
            let producer = scope.spawn(|| self.produce(&mut plan, windows, spare, &done));

            let ended = OnDrop(&done);
            let asked = self.consume(skip, ready, &spares, &mut f);
            drop(ended);

            let produced = producer.join().unwrap_or_else(|payload| panic::resume_unwind(payload));
            self.asked.store(asked.unwrap_or(produced.is_ok()), Ordering::Release);
            if asked.is_some() { Ok(()) } else { produced }
        })
    }

    fn produce(
        &self,
        plan: &mut Plan,
        windows: SyncSender<Window>,
        spares: Receiver<Vec<ChessBoard>>,
        done: &AtomicBool,
    ) -> Result<(), DataError>
    {
        let data = &self.data;
        let stopped = || done.load(Ordering::Acquire) || self.finishing();
        let options =
            ScanOptions { sampling: Sampling::All, threads: data.threads.loader, in_flight: None, block: BLOCK };
        let buffers = Pool::default();

        let mut start = plan.cursor();
        let (mut epoch, mut whole, mut positions) = (start.epoch, start.offsets.iter().all(|&offset| offset == 0), 0);
        let mut index = self.resume.as_ref().map_or(0, |bookmark| bookmark.window);
        let mut boards = Vec::new();

        let halted = pipeline::run(
            &data.sources,
            &options,
            plan,
            &stopped,
            |scratch, block| -> Result<Vec<ChessBoard>, DataError> {
                let source = &data.sources[block.source];
                let key = pass_key(self.keys[block.source], block.epoch);
                let selection = Selection { filter: data.filter.clone(), labels: data.labels, key };
                let mut found = buffers.take();
                for game in block.games {
                    if stopped() {
                        break;
                    }

                    let kept = selection.select_each(game, None, scratch, |_, board| found.push(board));
                    kept.map_err(|error| error.at(source))?;
                }

                Ok(found)
            },
            |meta, found| {
                if meta.cursor.epoch != epoch {
                    if whole && positions == 0 {
                        return Err(DataError::no_positions());
                    }

                    (epoch, whole, positions) = (meta.cursor.epoch, true, 0);
                }
                positions += found.len() as u64;

                if !boards.is_empty() && boards.len() + found.len() > self.capacity {
                    let broken = shuffle(&mut boards, self.seed, index, &stopped).is_break();
                    let next = meta.cursor.clone();
                    let window = Window { index, start: start.clone(), next, boards: mem::take(&mut boards) };
                    if broken || windows.send(window).is_err() {
                        return Ok(ControlFlow::Break(()));
                    }

                    index += 1;
                    boards = spares.try_recv().unwrap_or_default();
                }

                if boards.is_empty() {
                    start.clone_from(&meta.cursor);
                    boards.try_reserve_exact(self.capacity.max(found.len())).map_err(|_| {
                        DataError::new("The shuffle buffer does not fit in memory; lower the buffer size")
                    })?;
                }

                boards.extend_from_slice(&found);
                buffers.put(found);
                Ok(ControlFlow::Continue(()))
            },
        )?;

        if halted { Ok(()) } else { Err(DataError::no_positions()) }
    }

    fn consume(
        &self,
        skip: u64,
        windows: Receiver<Window>,
        spares: &SyncSender<Vec<ChessBoard>>,
        f: &mut impl FnMut(&[ChessBoard]) -> bool,
    ) -> Option<bool>
    {
        let mut emitted = skip;
        let mut skipped = self.resume.as_ref().map(|bookmark| bookmark.consumed);

        loop {
            let waiting = Instant::now();
            let mut received = Err(RecvTimeoutError::Timeout);
            while matches!(received, Err(RecvTimeoutError::Timeout)) && !self.finishing() {
                received = windows.recv_timeout(TICK);
            }
            LoaderStats::add(&self.stats.reading, waiting.elapsed());
            let window = match received {
                Ok(window) => window,
                Err(RecvTimeoutError::Timeout) => return Some(true),
                Err(RecvTimeoutError::Disconnected) => return None,
            };

            let length = window.boards.len();
            let mut consumed =
                skipped.take().map_or(0, |skipped| usize::try_from(skipped).unwrap_or(usize::MAX).min(length));

            while consumed < length {
                if self.finishing() {
                    return Some(true);
                }

                let room = self.superbatch - emitted % self.superbatch;
                let slice = (length - consumed).min(self.slice).min(usize::try_from(room).unwrap_or(usize::MAX));

                let delivered = &window.boards[consumed..consumed + slice];
                consumed += slice;
                emitted += slice as u64;
                if emitted.is_multiple_of(self.superbatch) {
                    self.mark(emitted, &window, consumed);
                }

                let called = Instant::now();
                let satisfied = f(delivered);
                LoaderStats::add(&self.stats.callback, called.elapsed());

                if satisfied {
                    return Some(false);
                }
            }

            let mut boards = window.boards;
            boards.clear();
            let _ = spares.try_send(boards);
        }
    }

    fn mark(&self, emitted: u64, window: &Window, consumed: usize)
    {
        let (index, start, consumed) = if consumed == window.boards.len() {
            (window.index + 1, window.next.clone(), 0)
        } else {
            (window.index, window.start.clone(), consumed as u64)
        };
        let (selection, labelling) = self.fingerprints.clone();
        let bookmark =
            Bookmark { reader_version: READER_VERSION, selection, labelling, window: index, start, consumed, emitted };

        lock(&self.marks).insert(emitted / self.superbatch, bookmark);
    }

    fn finishing(&self) -> bool { self.finished.load(Ordering::Acquire) }
}
