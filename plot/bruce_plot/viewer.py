import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import nonsingular

from . import style
from .journal import Run, SchemaError

SIZE = (14.2, 8.6)
LEFT, RIGHT, BOTTOM = 0.075, 0.975, 0.085
NAME_SIZE = 12
CHARACTER = NAME_SIZE * 0.602
ROW = NAME_SIZE * 1.6
TOP_MARGIN = 14
NAME_GAP = 3


def ema(values: np.ndarray, span: int = 50) -> np.ndarray:
    alpha = 2 / (span + 1)
    smoothed = np.empty_like(values)
    level = math.nan
    for index, value in enumerate(values.tolist()):
        if not math.isnan(value):
            level = value if math.isnan(level) else level + alpha * (value - level)
        smoothed[index] = level
    return smoothed


class Viewer:
    def __init__(self, runs: list[Run]) -> None:
        style.apply()
        self.runs = runs
        self.error: SchemaError | None = None
        self.fig = plt.figure(figsize=SIZE)
        self.fig.canvas.manager.set_window_title("bruce-plot")
        self.ax = self.fig.add_axes((LEFT, BOTTOM, RIGHT - LEFT, 0.8))
        self.ax.set_xlabel("Superbatch")
        self.ax.set_ylabel("Training loss")
        self.ax.format_coord = lambda _x, _y: ""
        self.lines = []
        self.names = []
        for index, run in enumerate(runs):
            colour, dash = style.style(index)
            raw = self.ax.plot([], [], color=colour, linestyle=dash, linewidth=0.8, alpha=0.35)[0]
            smooth = self.ax.plot([], [], color=colour, linestyle=dash, linewidth=2.0)[0]
            self.lines.append((raw, smooth))
            self.names.append(self.fig.text(0, 0, run.name, color=colour, fontsize=NAME_SIZE, va="top"))
        self.place()
        self.fig.canvas.mpl_connect("resize_event", self.place)
        self.timer = self.fig.canvas.new_timer(interval=2000)
        self.timer.add_callback(self.tick)
        self.refresh(changed=True)

    def place(self, _event=None) -> None:
        width, height = self.fig.get_figwidth() * 72, self.fig.get_figheight() * 72
        limit = width * (RIGHT - LEFT)
        offset, row = 0.0, 0
        for text in self.names:
            size = len(text.get_text()) * CHARACTER
            if offset and offset + size > limit:
                offset, row = 0.0, row + 1
            text.set_position((LEFT + offset / width, 1 - (TOP_MARGIN + row * ROW) / height))
            offset += size + NAME_GAP * CHARACTER
        top = 1 - (TOP_MARGIN + (row + 1) * ROW + 10) / height
        self.ax.set_position((LEFT, BOTTOM, RIGHT - LEFT, top - BOTTOM))

    def refresh(self, changed: bool = False) -> None:
        for run, (raw, smooth) in zip(self.runs, self.lines, strict=True):
            if run.poll() or changed:
                raw.set_data(run.x, run.loss)
                smooth.set_data(run.x, ema(run.loss))
                changed = True
        view = self.ax.get_xlim(), self.ax.get_ylim()
        self.fit()
        if changed or view != (self.ax.get_xlim(), self.ax.get_ylim()):
            self.fig.canvas.draw_idle()

    def fit(self) -> None:
        x = np.concatenate([run.x for run in self.runs])
        y = np.concatenate([smooth.get_ydata() for _, smooth in self.lines])
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if self.ax.get_autoscalex_on() and x.size:
            self.ax.set_xlim(nonsingular(x.min(), x.max()), auto=None)
        if self.ax.get_autoscaley_on() and y.size:
            pad = 0.05 * (y.max() - y.min())
            self.ax.set_ylim(nonsingular(y.min() - pad, y.max() + pad), auto=None)

    def tick(self) -> None:
        try:
            self.refresh()
        except SchemaError as error:
            self.error = error
            plt.close(self.fig)

    def show(self) -> None:
        self.timer.start()
        plt.show()
        if self.error is not None:
            raise self.error
