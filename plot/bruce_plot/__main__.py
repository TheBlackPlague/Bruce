import argparse
import sys
from pathlib import Path

import matplotlib

from . import journal
from .viewer import Viewer


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="bruce-plot", description="Loss curves of Bruce runs")
    commands = root.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="open the viewer")
    view.add_argument("runs", nargs="*", metavar="RUN", help="run names or fragments; every run under the root if none")
    view.add_argument("--root", type=Path, default=Path("runs"), metavar="DIR", help="directory of <run>/metrics.jsonl")
    view.add_argument("--wait", action="store_true", help="open one run, named in full, before its journal exists")
    return root


def fail(message: str) -> int:
    print(f"bruce-plot: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = arguments.root
    if arguments.wait:
        if len(arguments.runs) != 1:
            return fail("--wait takes exactly one full run name")
        names = arguments.runs
    elif arguments.runs:
        names = []
        for fragment in arguments.runs:
            found = journal.match(root, fragment)
            if not found:
                return fail(f"no run under {root} matches {fragment!r}")
            names += [name for name in found if name not in names]
    else:
        names = journal.discover(root)
        if not names:
            return fail(f"no runs under {root}")
    try:
        matplotlib.use("TkAgg", force=True)
    except ImportError as error:
        return fail(f"cannot open a window: tk needs tkinter, which this Python lacks ({error})")
    try:
        Viewer([journal.Run(root / name) for name in names]).show()
    except journal.SchemaError as error:
        return fail(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
