import json
import math
from pathlib import Path

import numpy as np

from . import SCHEMA

JOURNAL = "metrics.jsonl"


class SchemaError(ValueError):
    pass


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return math.nan
    return float(value)


def decode(line: bytes) -> dict | None:
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


class Run:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.name = self.directory.name
        self.path = self.directory / JOURNAL
        self.reset()
        self.poll()

    def reset(self) -> None:
        self.per_superbatch = 1
        self.rows: list[tuple[float, float, float]] = []
        self.x = np.empty(0)
        self.loss = np.empty(0)
        self.offset = 0
        self.identity: tuple[int, int] | None = None

    def poll(self) -> bool:
        replaced = False
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if stat.st_size < self.offset or (self.identity is not None and identity != self.identity):
                self.reset()
                replaced = True
            self.identity = identity
            if stat.st_size == self.offset:
                return replaced
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read(stat.st_size - self.offset)
        except OSError:
            return replaced
        complete = chunk.rfind(b"\n") + 1
        if not complete:
            return replaced
        self.offset += complete
        for line in chunk[:complete].splitlines():
            record = decode(line)
            if record is not None:
                self.read(record)
        rows = np.array(self.rows, dtype=float).reshape(-1, 3)
        self.x, self.loss = rows[:, 1], rows[:, 2]
        return True

    def read(self, record: dict) -> None:
        kind = record.get("k")
        if kind == "batch":
            sb = number(record.get("sb"))
            x = sb - 1 + number(record.get("b")) / self.per_superbatch
            self.rows.append((sb, x, number(record.get("loss"))))
        elif kind == "session":
            schema = record.get("schema")
            if not isinstance(schema, str) or not (schema == SCHEMA or schema.startswith(f"{SCHEMA}.")):
                raise SchemaError(f"{self.path}: unsupported schema {schema!r}; this viewer reads {SCHEMA}")
            steps = record.get("steps")
            steps = steps if isinstance(steps, dict) else {}
            first = number(steps.get("first_superbatch"))
            if math.isfinite(first):
                self.rows = [row for row in self.rows if row[0] < first]
            per_superbatch = number(steps.get("batches_per_superbatch"))
            if math.isfinite(per_superbatch) and per_superbatch > 0:
                self.per_superbatch = int(per_superbatch)


def discover(root: Path) -> list[str]:
    return sorted(path.parent.name for path in Path(root).glob(f"*/{JOURNAL}") if path.is_file())


def match(root: Path, fragment: str) -> list[str]:
    names = discover(root)
    return [fragment] if fragment in names else [name for name in names if fragment in name]
