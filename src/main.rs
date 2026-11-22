mod analyze;
mod args;
mod cli;
mod config;
mod control;
mod data;
mod events;
mod model;
mod render;
mod schedule;
mod screen;
mod telemetry;
mod text;
mod train;

fn main() -> std::process::ExitCode { cli::main() }
