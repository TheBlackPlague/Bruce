use std::{path::Path, time::Duration};

pub fn count(value: u64) -> String
{
    let digits = value.to_string();
    let groups: Vec<_> = digits.as_bytes().rchunks(3).rev().map(String::from_utf8_lossy).collect();
    groups.join(",")
}

pub fn file_name(path: &Path) -> String
{
    path.file_name().map_or_else(|| path.display().to_string(), |name| name.to_string_lossy().into_owned())
}

pub fn bytes(value: u64) -> String
{
    match unit(value) {
        (_, "B") => format!("{value} B"),
        (divisor, unit) => format!("{:.1} {unit}", value as f64 / divisor),
    }
}

pub fn bytes_of(read: u64, total: u64) -> String
{
    let read = read as f64 / unit(total).0;
    let decimals = if read < 9.995 { 2 } else { 1 };
    format!("{read:.decimals$}/{}", bytes(total))
}

fn unit(value: u64) -> (f64, &'static str)
{
    let mut divisor = 1.0;
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"] {
        if (value as f64 / divisor * 10.0).round() < 10_000.0 {
            return (divisor, unit);
        }
        divisor *= 1000.0;
    }
    (divisor, "EB")
}

pub fn scaled(value: f64) -> String
{
    match value {
        value if value >= 1e9 => format!("{:.2} B", value / 1e9),
        value if value >= 1e6 => format!("{:.1} M", value / 1e6),
        value if value >= 1e3 => format!("{:.1} k", value / 1e3),
        value => format!("{value:.0}"),
    }
}

pub fn span(value: Duration) -> String
{
    match value.as_secs() {
        seconds @ 0..60 => format!("{seconds}s"),
        seconds @ 60..3600 => format!("{}m {:02}s", seconds / 60, seconds % 60),
        seconds => format!("{}h {:02}m", seconds / 3600, seconds / 60 % 60),
    }
}

pub fn finished(input: usize, games: u64, positions: u64, elapsed: Duration) -> String
{
    format!("input {input} finished: {} games, {} positions in {}", count(games), count(positions), span(elapsed))
}

pub fn rate(positions_per_second: f64) -> String
{
    let millions = positions_per_second / 1e6;
    let decimals = if millions < 9.995 { 2 } else { 1 };
    format!("{millions:.decimals$} Mpos/s")
}

pub fn percent(part: u64, whole: u64) -> String
{
    let share = if whole == 0 { 0.0 } else { 100.0 * part as f64 / whole as f64 };
    format!("{share:.1}%")
}
