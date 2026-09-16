#[cfg(any(
    all(feature = "cuda", feature = "rocm"),
    all(feature = "cuda", feature = "metal"),
    all(feature = "rocm", feature = "metal")
))]
compile_error!("Enable exactly one GPU backend: cuda, rocm, or metal");

pub mod architecture;
pub mod checkpoint;
pub mod config;
pub mod convert;
pub mod data;
pub mod events;
pub mod schedule;
pub mod training;
pub mod ui;

pub mod progress;
pub mod tensorboard;
