import matplotlib as mpl

BACKGROUND = "#000000"
TEXT = "#a2a4a6"
LABEL = "#eeefef"
SPINE = "#7a7a7a"
GRID = "#2b2b2b"
FONT = "DejaVu Sans Mono"
COLOURS = ("#5ccfc6", "#e8b86a", "#b8743a", "#407770", "#fee6aa", "#a1f3ed", "#885945")
REPEAT_DASH = (0, (5, 4))


def style(index: int) -> tuple[str, str | tuple]:
    cycle, slot = divmod(index, len(COLOURS))
    return COLOURS[slot], "solid" if cycle == 0 else REPEAT_DASH


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.family": FONT,
            "font.size": 11,
            "text.color": TEXT,
            "axes.labelcolor": LABEL,
            "axes.labelsize": 11,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linestyle": (0, (2.4, 3.8)),
            "grid.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "axes.formatter.useoffset": False,
        }
    )
