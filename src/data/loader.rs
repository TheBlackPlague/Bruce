use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
};

use anyhow::{Context, Result, ensure};
use bullet_lib::{
    game::formats::{bulletformat::ChessBoard, sfbinpack::chess::color::Color},
    value::loader::{DirectSequentialDataLoader, MontyBinpackLoader, SfBinpackLoader, ViriBinpackLoader},
};
use bullet_trainer::reader::DataReader;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use super::{Format, accepts, accepts_position, read, sf_tactical, valid_sf_score, visit_boards_stoppable};
use crate::{
    config::{DataConfig, DataSource, Filter},
    schedule::BATCHES_PER_SUPERBATCH,
};

#[derive(Clone)]
pub struct DatasetReader
{
    config: DataConfig,
    seed: u64,
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
}

impl DatasetReader
{
    pub fn new(config: &DataConfig, seed: u64, stop: Arc<AtomicBool>) -> Result<Self>
    {
        config.validate()?;

        for source in &config.sources {
            let mut found = false;

            visit_boards_stoppable(source, config.filter.as_ref(), &stop, |board| {
                found = source.format != Format::Sf || valid_sf_score(board.score);
                Ok(found)
            })?;
            ensure!(found, "No positions pass the filter");
        }

        Ok(Self { config: config.clone(), seed, stop, error: Arc::default() })
    }

    pub fn error(&self) -> Option<String> { self.error.lock().unwrap_or_else(|poison| poison.into_inner()).clone() }

    fn load(&self, skip: usize, mut callback: impl FnMut(&[ChessBoard]) -> bool) -> Result<()>
    {
        let sources = &self.config.sources;
        if !sources.iter().all(|source| source.format == sources[0].format) {
            return self.load_mixed(skip, callback);
        }

        let paths = sources.iter().map(|source| source.path.to_str().unwrap()).collect::<Vec<_>>();

        let filter = self.config.filter.as_ref();
        let memory = self.config.buffer_size_mb;
        let threads = self.config.loader_threads;
        match sources[0].format {
            Format::Bullet if filter.is_none() => load_bullet(&paths, skip, callback)?,

            Format::Monty => {
                let filter = filter.cloned();
                MontyBinpackLoader::new_concat_multiple(&paths, memory, threads, move |position, mov, score, _| {
                    filter.as_ref().is_none_or(|filter| {
                        let ply = position.fullm().saturating_sub(1).saturating_mul(2) + position.stm() as u16;
                        accepts_position(filter, score, position.occ().count_ones(), Some(ply))
                            && (!filter.exclude_check || !position.in_check())
                            && (!filter.exclude_tactical || !(mov.is_capture() || mov.is_promo()))
                    })
                })
                .read_chunks(skip, callback);
            },

            Format::Sf => {
                let filter = filter.cloned();
                SfBinpackLoader::new_concat_multiple(&paths, memory, threads, move |entry| {
                    valid_sf_score(entry.score)
                        && filter.as_ref().is_none_or(|filter| {
                            let pieces = (entry.pos.pieces_bb(Color::White).bits()
                                | entry.pos.pieces_bb(Color::Black).bits())
                            .count_ones();
                            accepts_position(filter, entry.score, pieces, Some(entry.ply))
                                && (!filter.exclude_check || !entry.pos.is_checked(entry.pos.side_to_move()))
                                && (!filter.exclude_tactical || !sf_tactical(entry))
                        })
                })
                .read_chunks(skip, callback);
            },

            Format::Viri if filter.is_none_or(|filter| filter.maximum_ply.is_none()) => {
                use bullet_lib::game::formats::viriformat::dataformat::Filter as ViriFilter;

                let mut native = ViriFilter::UNRESTRICTED;
                if let Some(filter) = filter {
                    native.min_ply = u32::from(filter.minimum_ply);
                    native.min_pieces = filter.minimum_pieces;
                    native.max_eval = filter.maximum_eval;
                    native.filter_tactical = filter.exclude_tactical;
                    native.filter_check = filter.exclude_check;
                }

                let minimum = filter.map_or(0, |filter| filter.minimum_eval);

                let mut selected = Vec::new();
                ViriBinpackLoader::new_concat_multiple(&paths, memory, threads, native).read_chunks(skip, |boards| {
                    if minimum == 0 {
                        return callback(boards);
                    }

                    selected.clear();
                    selected.extend(
                        boards.iter().copied().filter(|board| u32::from(board.score.unsigned_abs()) >= minimum),
                    );

                    !selected.is_empty() && callback(&selected)
                });
            },

            _ => self.load_mixed(skip, callback)?,
        }

        Ok(())
    }

    fn load_mixed(&self, mut skip: usize, mut callback: impl FnMut(&[ChessBoard]) -> bool) -> Result<()>
    {
        let capacity = self.config.buffer_size_mb * 1024 * 1024 / size_of::<ChessBoard>();
        let mut shuffle = Vec::with_capacity(capacity);

        let mut random = StdRng::seed_from_u64(self.seed);
        loop {
            for sources in self.config.sources.chunks(self.config.loader_threads) {
                let abort = AtomicBool::new(false);
                let stopped = std::thread::scope(|scope| -> Result<bool> {
                    let receivers = sources
                        .iter()
                        .map(|source| {
                            let (sender, receiver) = sync_channel(1);
                            let abort = &abort;
                            let filter = self.config.filter.as_ref();
                            scope.spawn(move || produce(source, filter, abort, sender));
                            receiver
                        })
                        .collect::<Vec<_>>();

                    let mut active = vec![true; receivers.len()];
                    while active.iter().any(|active| *active) {
                        for (index, receiver) in receivers.iter().enumerate() {
                            if !active[index] {
                                continue;
                            }

                            let batch = match receiver.recv() {
                                Ok(batch) => batch.inspect_err(|_| abort.store(true, Ordering::Relaxed))?,
                                Err(_) => {
                                    active[index] = false;
                                    continue;
                                },
                            };

                            for board in batch {
                                if skip > 0 {
                                    skip -= 1;
                                    continue;
                                }

                                shuffle.push(board);
                                if shuffle.len() == capacity {
                                    shuffle.shuffle(&mut random);

                                    let stopped = callback(&shuffle);

                                    shuffle.clear();

                                    if stopped {
                                        abort.store(true, Ordering::Relaxed);
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    }

                    Ok(false)
                })?;

                if stopped {
                    return Ok(());
                }
            }

            if !shuffle.is_empty() {
                shuffle.shuffle(&mut random);

                if callback(&shuffle) {
                    return Ok(());
                }

                shuffle.clear();
            }
        }
    }
}

impl DataReader<ChessBoard> for DatasetReader
{
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip: usize, mut callback: F)
    {
        if self.stop.load(Ordering::Relaxed) {
            return;
        }

        let boundary = BATCHES_PER_SUPERBATCH * self.config.batch_size;

        let mut emitted = 0usize;
        let result = self.load(skip, |mut boards| {
            while !boards.is_empty() {
                if emitted.is_multiple_of(boundary) && self.stop.load(Ordering::Relaxed) {
                    return true;
                }

                let length = boards.len().min(boundary - emitted % boundary);
                if callback(&boards[..length]) {
                    return true;
                }

                emitted += length;
                boards = &boards[length..];
            }

            false
        });

        if let Err(error) = result {
            *self.error.lock().unwrap_or_else(|poison| poison.into_inner()) = Some(format!("{error:#}"));
        }
    }
}

fn produce(
    source: &DataSource,
    filter: Option<&Filter>,
    abort: &AtomicBool,
    sender: SyncSender<Result<Vec<ChessBoard>>>,
)
{
    let mut batch = Vec::with_capacity(4096);
    let result = read::scan(source, false, None, Some(abort), |record| {
        if !accepts(&record, filter) || (source.format == Format::Sf && !valid_sf_score(record.board.score)) {
            return Ok(false);
        }

        batch.push(record.board);
        if batch.len() < 4096 {
            return Ok(false);
        }

        let ready = std::mem::replace(&mut batch, Vec::with_capacity(4096));

        Ok(sender.send(Ok(ready)).is_err())
    });

    if abort.load(Ordering::Relaxed) {
        return;
    }

    match result {
        Err(error) => {
            let _ = sender.send(Err(error));
        },
        Ok(_) if !batch.is_empty() => {
            let _ = sender.send(Ok(batch));
        },
        _ => {},
    }
}

fn load_bullet(paths: &[&str], skip: usize, mut callback: impl FnMut(&[ChessBoard]) -> bool) -> Result<()>
{
    let lengths = paths
        .iter()
        .map(|path| Ok(std::fs::metadata(path)?.len() as usize / size_of::<ChessBoard>()))
        .collect::<Result<Vec<_>>>()?;
    let total = lengths.iter().sum::<usize>();
    ensure!(total > 0, "The dataset is empty");

    let mut offset = skip % total;
    let mut first = 0;
    while offset >= lengths[first] {
        offset -= lengths[first];
        first += 1;
    }

    let mut ordered = paths.to_vec();
    ordered.rotate_left(first);
    if offset > 0 {
        if load_bullet_tail(ordered[0], offset, &mut callback)? {
            return Ok(());
        }

        ordered.rotate_left(1);
    }

    DirectSequentialDataLoader::new(&ordered).read_chunks(0, callback);

    Ok(())
}

fn load_bullet_tail(path: &str, offset: usize, callback: &mut impl FnMut(&[ChessBoard]) -> bool) -> Result<bool>
{
    let mut file = File::open(path).context("Could not open the resumed dataset")?;

    let byte_offset = offset as u64 * size_of::<ChessBoard>() as u64;
    file.seek(SeekFrom::Start(byte_offset))?;

    let mut remaining = file.metadata()?.len().saturating_sub(byte_offset);
    ensure!(remaining.is_multiple_of(size_of::<ChessBoard>() as u64), "The resumed dataset is truncated");

    let capacity = usize::try_from(remaining).unwrap_or(usize::MAX).min(128 * 1024 * 1024) / size_of::<ChessBoard>();
    let mut boards = vec![ChessBoard::default(); capacity];

    while remaining > 0 {
        let length = usize::try_from(remaining).unwrap_or(usize::MAX).min(size_of_val(boards.as_slice()));
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(boards.as_mut_ptr().cast::<u8>(), size_of_val(boards.as_slice())) };
        file.read_exact(&mut bytes[..length]).context("Could not read the resumed dataset")?;

        if callback(&boards[..length / size_of::<ChessBoard>()]) {
            return Ok(true);
        }

        remaining -= length as u64;
    }

    Ok(false)
}
