"""The one house style every paper's figures share (DESIGN.md §8).

Figures are never hand-made. A figure is the output of a script reading
``results/``, and this module is the only place that decides what one looks
like — so every paper's plots are consistent and a journal's column width is
changed in one line rather than in forty scripts.

Decisions, and why
------------------
**No LaTeX.** ``text.usetex`` is off. SciencePlots' ``science`` style turns it
on, which makes every figure depend on a working TeX installation with the
right packages; a figure that renders on the author's laptop and dies in CI or
on Kaggle is not reproducible. matplotlib's ``mathtext`` with the ``cm`` font
set gives Computer Modern math out of the box, which is what the usetex path
was wanted for. If a specific submission really needs `\\SI{}` or a custom
macro, `use_style(usetex=True)` is there — opt in per figure, never by default.

**Okabe-Ito, reordered, not SciencePlots' cycle.** Measured 2026-08-08 with the
Machado-Oliveira-Fernandes 2009 CVD model (ΔE = Euclidean distance in OKLab
×100): SciencePlots' default cycle collapses under protanopia, its orange
``#FF9500`` and green ``#00B945`` landing ΔE 2.8 apart — indistinguishable, on
the two colors a "ours vs baseline" plot reaches for first. Paul Tol's
``bright`` fails on lightness and chroma. :data:`PALETTE` is Okabe-Ito ordered
so that the worst *adjacent* pair is ΔE 9.6 on paper white, above the 8 target
for all six slots. The order is fixed and colors are assigned in sequence,
never cycled: a condition keeps its color when another is added or dropped, or
a reader who learned "ours is blue" is misled by the next figure.

**Color is never the only channel.** Every series also gets a linestyle and a
marker (:data:`LINESTYLES`, :data:`MARKERS`). Three reasons, any one of which
would be enough: IEEE requires figures to be readable in black and white; three
of the six slots sit below 3:1 contrast on white, which obliges a second
channel; and a printed figure gets photocopied.

**Fields are perceptually uniform, and diverging maps are centred on zero.**
See :data:`SEQUENTIAL`, :data:`DIVERGING` and :func:`symmetric_norm`. A rainbow
(``jet``) introduces up to ~8% visual error through its uneven lightness and is
unreadable in grayscale; it is never used here.

**Light surface only.** These figures go on white paper. A dark variant would
be a second thing to keep correct for no reader.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from pinnslab.utils.logging import get_logger

log = get_logger(__name__)

# -- palette -------------------------------------------------------------------

#: Categorical hues for *identity* — which condition a line belongs to.
#: Okabe-Ito, ordered by measured adjacent CVD separation (worst pair ΔE 9.6 on
#: ``#ffffff`` under deuteranopia). Assigned in sequence and never cycled.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
)

#: Secondary encoding, paired slot-for-slot with :data:`PALETTE`.
LINESTYLES: tuple[Any, ...] = (
    "-",
    (0, (4, 1.5)),
    (0, (1, 1.2)),
    (0, (5, 1.2, 1, 1.2)),
    (0, (7, 1.5)),
    (0, (3, 1, 1, 1, 1, 1)),
)

#: Third channel, for series sparse enough to carry markers.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")

#: Ink. Text never wears a series color — a colored mark beside it carries
#: identity, so the label stays readable for everyone.
INK = "#1a1a1a"
MUTED_INK = "#5c5c5c"
#: Grid and spines: solid hairlines one shade off the surface, never dashed.
CHROME = "#c8c8c8"
SURFACE = "#ffffff"

#: Magnitude (error fields, |u_pred - u_exact|, residual density). Perceptually
#: uniform, CVD-safe, and monotone in grayscale so it survives a B&W print.
SEQUENTIAL = "viridis"

#: Polarity (a signed solution field u(x, t), a signed difference). Always with
#: :func:`symmetric_norm` — a diverging map whose midpoint is not at zero
#: reports a sign change where the data has none.
DIVERGING = "RdBu_r"


def color(index: int) -> str:
    """The hue for series ``index``. Raises past the palette rather than cycling."""
    return _slot(PALETTE, index, "color")


def linestyle(index: int) -> Any:
    return _slot(LINESTYLES, index, "linestyle")


def marker(index: int) -> str:
    return _slot(MARKERS, index, "marker")


def series_kwargs(index: int, **overrides: Any) -> dict[str, Any]:
    """Everything that identifies series ``index``, as plot kwargs.

    Color plus linestyle, because color alone fails in black and white and for
    a colorblind reader on the tighter pairs.
    """
    return {"color": color(index), "linestyle": linestyle(index), **overrides}


def _slot(values: Sequence[Any], index: int, what: str) -> Any:
    if not 0 <= index < len(values):
        raise IndexError(
            f"no {what} for series {index}: the palette has {len(values)} slots "
            "and is never cycled, because a 7th series would repeat a 1st and "
            "two conditions would share an identity. Fold the tail into one "
            '"other" series, or split the figure into small multiples.'
        )
    return values[index]


# -- figure geometry -----------------------------------------------------------

#: Column widths in inches. A figure is drawn at its final printed size and
#: never scaled in LaTeX: `\includegraphics[width=...]` rescales the fonts too,
#: which is how a paper ends up with six different label sizes.
WIDTHS: dict[str, float] = {
    "single": 3.35,  # Elsevier 1-col (90 mm); also fits IEEE's 3.5 in
    "onehalf": 5.51,  # Elsevier 1.5-col (140 mm)
    "double": 7.48,  # Elsevier 2-col (190 mm); IEEE full width is 7.16
}


def figsize(width: str = "single", aspect: float = 0.68) -> tuple[float, float]:
    """``(w, h)`` in inches. ``aspect`` is height/width; 0.68 ~ 3:2."""
    if width not in WIDTHS:
        raise ValueError(f"unknown width {width!r}; have {sorted(WIDTHS)}")
    w = WIDTHS[width]
    return (w, w * aspect)


# -- the style itself ----------------------------------------------------------


def rc_params(*, usetex: bool = False) -> dict[str, Any]:
    """The house rcParams as a plain dict, so they can be inspected and tested."""
    params: dict[str, Any] = {
        # Type. cmr10 is Computer Modern and ships with matplotlib; the fallback
        # keeps this working on a machine that somehow lacks it. unicode_minus
        # is off because cmr10 has no U+2212 glyph and would draw a box.
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "text.usetex": usetex,
        # Sizes are absolute, because the figure is drawn at final size.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        # Chrome: recessive, hairline, solid. Never dashed — dashing on a grid
        # reads as "threshold" when it is just a grid.
        "axes.edgecolor": CHROME,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": False,
        "grid.color": CHROME,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.alpha": 0.6,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks point out; inward ticks collide with data near the axes.
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        # Marks: thin lines, markers big enough to see at print size.
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "lines.markeredgewidth": 0.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE))
        + mpl.cycler(linestyle=list(LINESTYLES)),
        # Legend: no frame, tight.
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3,
        # Colour maps.
        "image.cmap": SEQUENTIAL,
        # Output. PDF is the deliverable — vector, so the journal's typesetter
        # cannot resample it. Type 42 embeds real fonts rather than Type 3
        # bitmaps, which some publishers reject outright.
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.figsize": figsize("single"),
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    }
    if usetex:
        params["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    return params


@contextmanager
def use_style(*, usetex: bool = False, **overrides: Any) -> Iterator[None]:
    """Apply the house style for one figure, then put rcParams back.

    A context manager rather than a global ``plt.style.use`` so that importing
    a pinnslab module never mutates a notebook's plotting state — the same
    no-global-side-effects rule the package follows for torch.
    """
    with mpl.rc_context({**rc_params(usetex=usetex), **overrides}):
        yield


# -- helpers that keep specific figures honest ---------------------------------


def symmetric_norm(values: Any, *, robust: bool = False) -> Normalize:
    """A diverging norm centred on zero, spanning ``±max|values|``.

    Not optional, and not cosmetic. Matplotlib's default norm maps a field's
    own min and max onto the ends of the colormap, so a field spanning
    ``[-0.2, 1.0]`` renders with its zero crossing at 17% of a red-white-blue
    ramp: the white band lands somewhere in the positive data and the figure
    shows a sign change the solution does not have. Always pair
    :data:`DIVERGING` with this.

    ``robust=True`` clips to the 99.5th percentile of ``|values|`` instead, for
    a field with a small number of extreme cells that would otherwise flatten
    everything else to the midpoint.
    """
    magnitudes = np.abs(np.asarray(values, dtype=float))
    finite = magnitudes[np.isfinite(magnitudes)]
    if finite.size == 0:
        raise ValueError("cannot build a norm from values that are all non-finite")
    limit = float(np.percentile(finite, 99.5) if robust else finite.max())
    if limit == 0.0:
        limit = 1.0  # an identically-zero field is legal; a zero-width norm is not
    return Normalize(vmin=-limit, vmax=limit)


def log_axis(ax: plt.Axes, which: str = "y") -> None:
    """Log scale, a recessive grid on the decades, and minor ticks.

    PINN errors span orders of magnitude and are compared as ratios, so a
    linear axis hides everything that matters below the first decade.

    The minor ticks are not decoration: without them a reader cannot place a
    curve between two decade labels, and "is that 3e-4 or 7e-4" is exactly the
    judgement a convergence plot is asking for. The grid stays on the decades
    only — a minor grid on a log axis is a moiré.
    """
    if which in ("y", "both"):
        ax.set_yscale("log")
        ax.tick_params(axis="y", which="minor", left=True)
    if which in ("x", "both"):
        ax.set_xscale("log")
        ax.tick_params(axis="x", which="minor", bottom=True)
    ax.grid(True, which="major", axis=which if which != "both" else "both")
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: str | Path, *, also_png: bool = True) -> Path:
    """Write the figure as PDF (and a PNG for slides and quick looks).

    Returns the PDF path. Parent directories are created: figures are derived
    artifacts regenerated from ``results/``, so unlike a run directory there is
    nothing here to protect from being overwritten (DESIGN.md §11 — derived
    files are disposable, and a figure you cannot regenerate is a bug).
    """
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    log.info("wrote %s", path)
    return path


__all__ = [
    "CHROME",
    "DIVERGING",
    "INK",
    "LINESTYLES",
    "MARKERS",
    "MUTED_INK",
    "PALETTE",
    "SEQUENTIAL",
    "SURFACE",
    "WIDTHS",
    "color",
    "figsize",
    "linestyle",
    "log_axis",
    "marker",
    "rc_params",
    "save",
    "series_kwargs",
    "symmetric_norm",
    "use_style",
]
