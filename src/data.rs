mod config;
pub(crate) mod finite;

pub use config::{DataConfig, DataFormat, FilterConfig};

use std::{fs, path::Path, sync::Arc};

use crate::events::{Event, Reporter};
use anyhow::{Context, Result, ensure};
use bullet_lib::{
    game::formats::{
        bulletformat::{
            self, BulletFormat, ChessBoard,
            chess::{CudADFormat, MarlinFormat},
        },
        viriformat::dataformat::Filter,
    },
    value::loader::{
        DirectSequentialDataLoader, InMemoryTextLoader, MontyBinpackLoader, SfBinpackLoader,
        ViriBinpackLoader,
    },
};
use bullet_trainer::reader::DataReader;

#[derive(Clone)]
pub struct DatasetReader {
    config: DataConfig,
    _converted: Option<Arc<tempfile::TempDir>>,
}

pub fn reader(config: &DataConfig) -> Result<DatasetReader> {
    config.validate()?;
    let mut config = config.clone();
    let mut converted = None;

    if config.format == DataFormat::Cudad {
        let directory = tempfile::tempdir().context("create CudAD conversion directory")?;
        let mut paths = Vec::new();

        for (index, input) in config.paths.iter().enumerate() {
            let output = directory.path().join(format!("{index}.bullet"));
            bulletformat::convert_from_bin::<CudADFormat, ChessBoard>(
                input,
                &output,
                config.loader_threads,
            )?;

            validate_first(&output, DataFormat::Bullet)?;

            paths.push(output);
        }

        config.paths = paths;
        config.format = DataFormat::Bullet;

        converted = Some(Arc::new(directory));
    }

    Ok(DatasetReader {
        config,
        _converted: converted,
    })
}

impl DataReader<ChessBoard> for DatasetReader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, skip_count: usize, mut f: F) {
        let paths = self
            .config
            .paths
            .iter()
            .map(|p| p.to_str().expect("validated UTF-8 dataset path"))
            .collect::<Vec<_>>();

        let buffer = self.config.buffer_size_mb;
        let threads = self.config.loader_threads;
        let filter = self.config.filter.clone();

        match self.config.format {
            DataFormat::Bullet => {
                DirectSequentialDataLoader::new(&paths).read_chunks(skip_count, f)
            }

            DataFormat::Marlin => DirectSequentialDataLoader::new(&paths).read_chunks(
                skip_count,
                |chunk: &[MarlinFormat]| {
                    let converted = chunk
                        .iter()
                        .copied()
                        .map(ChessBoard::from)
                        .collect::<Vec<_>>();
                    f(&converted)
                },
            ),

            DataFormat::Text => InMemoryTextLoader::new(paths[0]).read_chunks(skip_count, f),

            DataFormat::Monty => MontyBinpackLoader::new_concat_multiple(
                &paths,
                buffer,
                threads,
                move |pos, mov, score, _| {
                    filter
                        .as_ref()
                        .is_none_or(|filter| filter.monty(pos, mov, score))
                },
            ).read_chunks(skip_count, f),

            DataFormat::Sf => {
                SfBinpackLoader::new_concat_multiple(&paths, buffer, threads, move |entry| {
                    finite::valid_sf_score(entry.score)
                        && filter.as_ref().is_none_or(|filter| filter.sf(entry))
                })
                .read_chunks(skip_count, f)
            }

            DataFormat::Viri => ViriBinpackLoader::new_concat_multiple(
                &paths,
                buffer,
                threads,
                filter.map_or(Filter::UNRESTRICTED, |f| f.viri()),
            ).read_chunks(skip_count, f),

            DataFormat::Cudad => {
                unreachable!("CudAD is converted with Bullet's native header-aware converter")
            }
        }
    }
}

pub fn validate(config: &DataConfig, reporter: &Reporter) -> Result<()> {
    config.validate()?;
    if config.format == DataFormat::Cudad {
        reporter.emit(Event::Note {
            message: "CudAD is converted to a temporary Bullet file before training because \
                      Bullet's direct reader does not handle its 1288-byte header. \
                      This needs additional disk space and repeats on resume.".into()
        });
    }

    for (index, path) in config.paths.iter().enumerate() {
        reporter.emit(Event::Phase {
            name: "Checking datasets".into(),
            completed: index as u64,
            total: Some(config.paths.len() as u64),
        });

        ensure!(
            path.to_str().is_some(),
            "dataset paths must be UTF-8: {}",
            path.display()
        );

        check_file(path, config.format)?;
        if matches!(config.format, DataFormat::Bullet | DataFormat::Marlin) {
            validate_first(path, config.format)?;
        }

        if matches!(
            config.format,
            DataFormat::Monty | DataFormat::Sf | DataFormat::Viri | DataFormat::Text
        ) {
            let mut accepted = false;

            finite::visit(
                path,
                config.format,
                config.filter.as_ref(),
                |board| {
                    finite::validate_board(&board)?;
                    accepted = true;
                    Ok(true)
                },
                |bytes| {
                    reporter.emit(Event::Phase {
                        name: format!("Checking {}", path.display()),
                        completed: bytes,
                        total: fs::metadata(path).ok().map(|m| m.len()),
                    })
                },
            )?;

            ensure!(
                accepted,
                "{} contains no positions accepted by the configured filter",
                path.display()
            );
        }
    }

    reporter.emit(Event::Phase {
        name: "Checking datasets".into(),
        completed: config.paths.len() as u64,
        total: Some(config.paths.len() as u64),
    });

    Ok(())
}

pub(crate) fn check_file(path: &Path, format: DataFormat) -> Result<()> {
    let metadata = fs::metadata(path).with_context(|| format!("open dataset {}", path.display()))?;

    ensure!(
        metadata.is_file() && metadata.len() > 0,
        "dataset must be a nonempty regular file: {}",
        path.display()
    );

    let (header, size) = match format {
        DataFormat::Bullet => (0                       , size_of::< ChessBoard >()),
        DataFormat::Marlin => (0                       , size_of::<MarlinFormat>()),
        DataFormat::Cudad  => (CudADFormat::HEADER_SIZE, size_of::< CudADFormat>()),

        _ => return Ok(()),
    };

    ensure!(
         metadata.len() > header as u64 &&
        (metadata.len() - header as u64).is_multiple_of(size as u64),
        "{} has a truncated or invalid fixed-size record layout",
        path.display()
    );

    Ok(())
}

fn validate_first(path: &Path, format: DataFormat) -> Result<()> {
    let loader = DirectSequentialDataLoader::new(
        &[path.to_str().context("dataset path must be UTF-8")?]
    );

    let mut result = Ok(());

    match format {
        DataFormat::Bullet => loader.read_chunks(0, |chunk: &[ChessBoard]| {
            result = chunk
                .first()
                .context("empty Bullet dataset")
                .and_then(finite::validate_board);

            true
        }),

        DataFormat::Marlin => loader.read_chunks(0, |chunk: &[MarlinFormat]| {
            result = chunk
                .first()
                .context("empty Marlin dataset")
                .and_then(|record| {
                    ensure!(
                        (2..=32).contains(&record.occ().count_ones()),
                        "invalid Marlin piece count"
                    );
                    finite::validate_board(&ChessBoard::from(*record))
                });

            true
        }),

        _ => unreachable!("only headerless fixed-size records"),
    }

    result.with_context(|| format!("invalid first record in {}", path.display()))
}

#[cfg(test)]
mod tests;
