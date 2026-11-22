#!/usr/bin/env python3
"""Concept sheets for notebooks/3-calculus-i.ipynb and notebooks/4-calculus-ii.ipynb (one per segment).

Calculus I: Segment 1 (limits), Segment 2 (the delta method / differentiation), Segment 3 (autodiff).
Calculus II: Segment 2 (partial derivatives) and Segment 3 (integrals); Segment 1 is a one-cell review
pointing to the slides, so it gets no sheet.

Every number is computed here from the same functions, points, step sizes and tensors the notebooks use,
with asserts against the values they print. TensorFlow is not installed: the GradientTape result of
3-calculus-i is reproduced with PyTorch autograd (the notebook's printed TF value is quoted as such).
Images are written to notebooks/img/<notebook stem>_seg<N>.png.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_nb3_4_concept_images.py
"""
import math
from functools import partial

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.integrate import quad
from sklearn.metrics import auc

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid)
from viz_style import save as _save

save = partial(_save, folder=REPO / "notebooks" / "img")

RAMP = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab"]     # light -> dark blues
PURPLE = "#7c5cc4"


# ================================================================ helpers
def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """viz_style.panel, but a line holding a fraction is set in display size (\\dfrac) and gets
    a taller line, so stacked fractions stay readable and never touch the next line."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor=CARD, edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5, weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        tall = r"\frac" in text or r"\int" in text
        if tall:
            text = text.replace(r"\frac", r"\dfrac")
            y -= spacing * 0.15
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0), family="monospace" if is_code else None)
        y -= spacing * (1.75 if tall else 1.0)


def axes_cross(ax):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)


def finish(ax, title, xlabel="x", ylabel="y", legend_loc=None, legend_size=9.5):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=legend_size)


def line_through(ax, xs, x0, y0, m, color, label=None, lw=1.8, ls="--", zorder=3):
    ax.plot(xs, y0 + m * (xs - x0), color=color, linewidth=lw, linestyle=ls, label=label, zorder=zorder)


def box(ax, xy, w, h, text, color, fontsize=12):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=SURFACE, edgecolor=color, linewidth=2.2, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def flow_arrow(ax, start, end, color, text=None, text_xy=None, fontsize=10.5):
    ax.annotate("", xy=end, xytext=start, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2, mutation_scale=16))
    if text:
        ax.text(*text_xy, text, ha="center", va="center", fontsize=fontsize, color=color)


# ================================================================ notebook 3, segment 1
def sheet_nb3_seg1_limits():
    x = np.linspace(-10, 10, 10000)          # the notebook's grid

    def quad_fxn(v):
        return v ** 2 + 2 * v + 2

    def my_fxn(my_x):
        return (my_x ** 2 - 1) / (my_x - 1)

    def sin_fxn(my_x):
        return np.sin(my_x) / my_x

    def inf_fxn(my_x):
        return 25 / my_x

    approach = {2: 3.0, 0.9: 1.9, 0.999: 1.9989999999999712, 1.1: 2.1, 1.001: 2.0009999999999177}
    assert all(my_fxn(k) == v for k, v in approach.items())
    sin_printed = {0.1: 0.9983341664682815, 0.001: 0.9999998333333416,
                   -0.1: 0.9983341664682815, -0.001: 0.9999998333333416}
    assert all(float(sin_fxn(k)) == v for k, v in sin_printed.items())
    assert inf_fxn(1e3) == 0.025 and inf_fxn(1e6) == 2.5e-05
    assert quad_fxn(5) == 37 and 0 not in x            # the dashed guides of cell 19; no x = 0 on the grid

    # exercises (limits of the factored forms)
    ex = [(lambda v: (v ** 2 - 1) / (v - 1), 0, 1.0), (lambda v: (v ** 2 - 25) / (v + 5), -5, -10.0),
          (lambda v: (v ** 2 - 2 * v - 8) / (v - 4), 4, 6.0)]
    for fn, a, lim in ex:
        assert abs(fn(a + 1e-7) - lim) < 1e-5 and abs(fn(a - 1e-7) - lim) < 1e-5
    # exercise 1 evaluates directly: (0 - 1)/(0 - 1) = 1, no hole at x = 0
    assert ex[0][0](0) == 1.0
    assert inf_fxn(-1e6) == -2.5e-05 and inf_fxn(-1e-6) < -1e6 and inf_fxn(1e-6) > 1e6

    fig = new_figure("A limit is the value a function approaches, even where it is undefined",
                     r"$\lim_{x\to 1}\ \frac{x^2-1}{x-1} = 2,\qquad \lim_{x\to 0}\ \frac{\sin x}{x} = 1,"
                     r"\qquad \lim_{x\to\infty}\ \frac{25}{x} = 0$"
                     "      (approach from both sides; the point itself never has to be evaluated)")
    gs = grid(fig, top=0.82)

    # -- zooming in on y = x^2 + 2x + 2 around (-1, 1): curves approach lines
    ax = fig.add_subplot(gs[0, 0])
    windows = [((-2, 0), (0, 2)), ((-1.5, -0.5), (0.5, 1.5)), ((-1.1, -0.9), (0.9, 1.1)),
               ((-1.01, -0.99), (0.99, 1.01))]
    u = np.linspace(0, 1, 400)
    for ((x0, x1), (y0, y1)), color in zip(windows, RAMP):
        xx = x0 + u * (x1 - x0)
        v = (quad_fxn(xx) - y0) / (y1 - y0)
        ax.plot(u, v, color=color, linewidth=2.4, label=f"x in [{x0:g}, {x1:g}]: width {x1 - x0:g}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1], ["left edge", "x = −1", "right edge"])
    ax.set_yticks([0, 0.5, 1], ["bottom", "y = 1", "top"])
    finish(ax, "Zoom in on y = x² + 2x + 2 at (−1, 1)", "", "", "upper center", 9)
    ax.text(0.5, 0.36, "each window 2–10× tighter:\nthe curve flattens into its tangent (m = 0)",
            ha="center", fontsize=9.5, color=INK2)

    # -- (x^2 - 1)/(x - 1) with a hole at x = 1
    ax = fig.add_subplot(gs[0, 1])
    y = my_fxn(x)
    ax.plot(x, y, color=BLUE, linewidth=2.4, label="(x² − 1)/(x − 1)", zorder=2)
    ax.axvline(1, color=PURPLE, linestyle="--", linewidth=1.2)
    ax.axhline(2, color=PURPLE, linestyle="--", linewidth=1.2)
    ax.plot(1, 2, "o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.8,
            zorder=5, label="x = 1: 0/0, division by zero")
    for k, color, off in [(0.9, AQUA, (22, -22)), (1.1, ORANGE, (-24, 12))]:
        ax.plot(k, my_fxn(k), "o", color=color, markersize=6, zorder=4)
        ax.annotate(f"f({k:g}) = {my_fxn(k):g}", (k, my_fxn(k)), textcoords="offset points", xytext=off,
                    ha="right" if off[0] < 0 else "left", fontsize=9.5, color=color)
    ax.text(1.3, 0.9, "f(0.999) = 1.998999…\nf(1.001) = 2.000999…\nf(2) = 3", fontsize=9.5, color=INK2)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0.8, 3.4)
    axes_cross(ax)
    finish(ax, "A hole at x = 1, yet the limit is 2", legend_loc="upper left")

    # -- sin x / x
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x, sin_fxn(x), color=BLUE, linewidth=2.4, label="sin x / x")
    ax.axhline(1, color=PURPLE, linestyle="--", linewidth=1.2)
    ax.plot(0, 1, "o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.8,
            zorder=5, label="x = 0: 0/0, undefined")
    ax.text(0.5, 1.48, "f(±0.1) = 0.998334…\nf(±0.001) = 0.99999983…", fontsize=9.5, color=INK2)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1, 2)
    axes_cross(ax)
    finish(ax, "Same from both sides: sin x / x → 1", legend_loc="lower left")

    # -- 25/x near 0: two branches
    ax = fig.add_subplot(gs[1, 0])
    left_x, right_x = x[x < 0], x[x > 0]
    ax.plot([left_x[-1], right_x[0]], [inf_fxn(left_x[-1]), inf_fxn(right_x[0])], color=MUTED,
            linestyle=":", linewidth=1.6, label="plotting x and y joins the branches")
    ax.plot(left_x, inf_fxn(left_x), color=BLUE, linewidth=2.4, label="25/x, left branch → −∞")
    ax.plot(right_x, inf_fxn(right_x), color=ORANGE, linewidth=2.4, label="25/x, right branch → +∞")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-300, 300)
    axes_cross(ax)
    finish(ax, "lim at x → 0 of 25/x does not exist", legend_loc="upper left", legend_size=9)

    # -- 25/x as x grows
    ax = fig.add_subplot(gs[1, 1])
    big = np.logspace(0, 6.3, 300)
    ax.plot(big, inf_fxn(big), color=BLUE, linewidth=2.4, label="25/x")
    for k in (1e3, 1e6):
        ax.plot(k, inf_fxn(k), "o", color=ORANGE, markersize=7, zorder=4)
        ax.annotate(f"25/{k:.0e} = {inf_fxn(k):g}".replace("e+0", "e"), (k, inf_fxn(k)),
                    textcoords="offset points", xytext=(-10, -16), ha="right", fontsize=10, color=ORANGE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    finish(ax, "x → ∞: 25/x shrinks toward 0 (log–log)", "x (log scale)", "25/x (log scale)", "upper right")

    panel(fig.add_subplot(gs[1, 2]), "The exercises, solved", [
        r"1.  $\lim_{x\to 0}\ \frac{x^2-1}{x-1} = \frac{-1}{-1} = 1$   (no hole there)",
        r"2.  $\frac{x^2-25}{x+5} = x-5\ \Rightarrow\ \lim_{x\to -5} = -10$",
        r"3.  $\frac{x^2-2x-8}{x-4} = x+2\ \Rightarrow\ \lim_{x\to 4} = 6$",
        "4.  x → −∞:  25/x → 0 (from below)",
        "5.  x → 0:  25/x has no limit,",
        "     −∞ from the left, +∞ from the right",
        "",
        "Recipe: factor and cancel the 0/0 term.",
    ], spacing=0.08, fontsize=12)

    save(fig, name="3-calculus-i_seg1.png")


# ================================================================ notebook 3, segment 2
def sheet_nb3_seg2_delta_method():
    x = np.linspace(-10, 10, 10000)

    def f(my_x):
        return my_x ** 2 + 2 * my_x + 2

    def diff_demo(my_f, my_x, my_delta):
        return (my_f(my_x + my_delta) - my_f(my_x)) / my_delta

    assert f(2) == 10 and f(5) == 37 and abs(f(2.1) - 10.61) < 1e-12
    m1 = (37 - 10) / (5 - 2)
    b1 = 37 - m1 * 5
    assert m1 == 9.0 and b1 == -8.0
    m2 = (10.61 - 10) / (2.1 - 2)
    b2 = 10.61 - m2 * 2.1
    assert m2 == 6.099999999999989 and b2 == -2.199999999999978
    delta_x = 0.000001
    x2 = 2 + delta_x
    y2 = f(x2)
    m3 = (y2 - 10) / (x2 - 2)
    assert x2 == 2.000001 and y2 == 10.000006000001001 and m3 == 6.000001000088901
    # exercise at x = -1
    xe1, ye1 = -1, f(-1)
    xe2 = xe1 + delta_x
    ye2 = f(xe2)
    me = (ye2 - ye1) / (xe2 - xe1)
    be = ye2 - me * xe2
    assert ye1 == 1 and xe2 == -0.999999 and ye2 == 1.000000000001
    assert me == 1.0000889005535828e-06 and be == 1.0000010000889006

    deltas = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    at2 = [diff_demo(f, 2, d) for d in deltas]
    atm1 = [diff_demo(f, -1, d) for d in deltas]
    assert at2 == [7.0, 6.099999999999994, 6.009999999999849, 6.000999999999479, 6.000100000012054,
                   6.000009999951316, 6.000001000927568]
    assert atm1 == [1.0, 0.10000000000000009, 0.009999999999998899, 0.001000000000139778,
                    9.99999993922529e-05, 1.000000082740371e-05, 1.000088900582341e-06]
    # algebra: (f(x+d) - f(x))/d = 2x + 2 + d, so the error is exactly d (up to round-off)
    assert all(abs((s - 6) - d) < 1e-9 for s, d in zip(at2, deltas))
    assert all(abs(s - d) < 1e-9 for s, d in zip(atm1, deltas))

    fig = new_figure("The derivative is the slope the secant settles on as Δx → 0",
                     r"$\frac{dy}{dx} = \lim_{\Delta x\to 0}\ \frac{f(x+\Delta x)-f(x)}{\Delta x}$"
                     "      For y = x² + 2x + 2: slope 6 at P = (2, 10), slope 0 at (−1, 1).")
    gs = grid(fig, top=0.82)

    # -- secants through P
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1.5, 6, 300)
    ax.plot(xs, f(xs), color=INK, linewidth=2.4, label="y = x² + 2x + 2", zorder=2)
    line_through(ax, xs, 5, 37, m1, RAMP[1], "Q = (5, 37): slope 27/3 = 9")
    line_through(ax, xs, 2.1, 10.61, m2, RAMP[3], "Q = (2.1, 10.61): slope 6.1")
    line_through(ax, xs, 2, 10, 6.0, ORANGE, "tangent at P: slope 6", lw=2.2, ls="-", zorder=2)
    ax.plot(2, 10, "o", color=INK, markersize=8, zorder=5)
    ax.annotate("P", (2, 10), textcoords="offset points", xytext=(-14, 4), fontsize=11, weight="bold")
    ax.plot(5, 37, "o", color=RAMP[1], markersize=7, zorder=5)
    ax.plot(2.1, 10.61, "o", color=RAMP[3], markersize=6, zorder=5)
    ax.set_xlim(-1.5, 6)
    ax.set_ylim(-5, 50)
    finish(ax, "Delta method: slope Δy/Δx between P and Q", legend_loc="upper left", legend_size=9)

    # -- the slope at every point; the exercise at x = -1
    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-4, 3, 300)
    ax.plot(xs, f(xs), color=INK, linewidth=2.4, label="y = x² + 2x + 2")
    ax.plot(xs, 2 * xs + 2, color=BLUE, linewidth=2, linestyle="--", label="dy/dx = 2x + 2")
    for x0, color in [(-1, AQUA), (2, ORANGE)]:
        t = np.array([x0 - 1.2, x0 + 1.2])
        ax.plot(t, f(x0) + (2 * x0 + 2) * (t - x0), color=color, linewidth=2.4, zorder=4)
        ax.plot(x0, f(x0), "o", color=INK, markersize=6, zorder=5)
        ax.plot(x0, 2 * x0 + 2, "s", color=color, markersize=7, zorder=5)
    ax.annotate("exercise: x = −1, slope 0\n(the minimum)", (-1, 1), textcoords="offset points",
                xytext=(-8, 26), ha="center", fontsize=9.5, color=AQUA)
    ax.annotate("x = 2: slope 6", (2, 6), textcoords="offset points", xytext=(6, -22), ha="center",
                fontsize=9.5, color=ORANGE)
    axes_cross(ax)
    ax.set_xlim(-4, 3)
    ax.set_ylim(-6, 17)
    finish(ax, "A different slope at each x", legend_loc="upper left")

    panel(fig.add_subplot(gs[0, 2]), "Make Δx tiny: Δx = 0.000001", [
        "P = (2, 10),  Q = (x₁ + Δx, f(x₁ + Δx)):",
        "`x2 = 2.000001`",
        "`y2 = 10.000006000001001`",
        r"$m = \frac{y_2-y_1}{x_2-x_1}$",
        "`m  = 6.000001000088901   → 6`",
        "",
        "Exercise, P = (−1, 1):",
        "`y2 = 1.000000000001`",
        "`m  = 1.0000889005535828e-06 → 0`",
    ], spacing=0.08, fontsize=12)

    # -- diff_demo at x = 2
    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(6, color=ORANGE, linestyle=":", linewidth=1.6, label="dy/dx at x = 2: 6")
    ax.plot(deltas, at2, "-o", color=BLUE, linewidth=2, markersize=6, zorder=3)
    for k, (d, s) in enumerate(zip(deltas, at2)):
        if k <= 3 or k == len(deltas) - 1:
            off = (-4, 10) if k == len(deltas) - 1 else (6, 4)
            ax.annotate(f"{s:.6g}" if k else "7.0", (d, s), textcoords="offset points", xytext=off,
                        ha="right" if k == len(deltas) - 1 else "left", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(5.8, 7.25)
    finish(ax, "diff_demo(f, 2, Δx) for Δx = 1 … 10⁻⁶", "Δx (log scale)  ← shrinking",
           "(f(2 + Δx) − f(2)) / Δx", "upper right")

    # -- error = Δx exactly
    ax = fig.add_subplot(gs[1, 1])
    d = np.array(deltas)
    ax.plot(d, d, color=MUTED, linewidth=1.4, linestyle="--", label="error = Δx (the algebra)")
    ax.plot(d, np.abs(np.array(at2) - 6), "o", color=ORANGE, markersize=9, label="x = 2: |slope − 6|")
    ax.plot(d, np.abs(np.array(atm1)), "s", color=BLUE, markersize=5, label="x = −1: |slope − 0|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    finish(ax, "Error = Δx: 10× smaller step, 10× closer", "Δx (log scale)  ← shrinking",
           "error of the secant slope", "upper right")
    ax.text(0.03, 0.06, "(f(x+Δx) − f(x))/Δx = 2x + 2 + Δx", transform=ax.transAxes, fontsize=10,
            color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "From delta method to derivative", [
        r"$\frac{dy}{dx} = \lim_{\Delta x\to 0}\frac{\Delta y}{\Delta x}"
        r" = \lim_{\Delta x\to 0}\frac{f(x+\Delta x)-f(x)}{\Delta x}$",
        r"Here: $\frac{(x+\Delta x)^2+2(x+\Delta x)+2-f(x)}{\Delta x}$",
        r"$= 2x + 2 + \Delta x \ \to\ 2x + 2$",
        "Notation: dy/dx = f′(x) = d f(x)/dx",
        "Rules (on the slides) give it directly:",
        r"$(x^n)' = nx^{n-1}$,  $(cf)' = cf'$,  $(f+g)' = f'+g'$",
        r"$(fg)' = f'g + fg'$,  $(f/g)' = (f'g - fg')/g^2$",
        r"chain: $\frac{dy}{dx} = \frac{dy}{du}\,\frac{du}{dx}$",
    ], spacing=0.078, fontsize=12)

    save(fig, name="3-calculus-i_seg2.png")


# ================================================================ notebook 3, segment 3
def sheet_nb3_seg3_autodiff():
    x = torch.tensor(5.0)
    x.requires_grad_()
    y = x ** 2
    y.backward()
    assert x.grad.item() == 10.0 and y.item() == 25.0
    # TensorFlow is not installed; the notebook prints <tf.Tensor: ... numpy=10.0> for t.gradient(y, x).
    # The same tape reproduced with PyTorch (float32, like tf.Variable(5.0)):
    tf_like = torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
    assert torch.autograd.grad(tf_like ** 2, tf_like)[0].item() == 10.0

    # autograd over a whole grid of x values
    grid_x = torch.linspace(-6, 6, 25, requires_grad=True)
    (grid_x ** 2).sum().backward()
    assert torch.allclose(grid_x.grad, 2 * grid_x.detach())

    # finite differences at x = 5 in float64: truncation error h, round-off ~ eps*25/h
    hs = np.logspace(-15, 0, 61)
    fd = ((5.0 + hs) ** 2 - 25.0) / hs
    fd_err = np.abs(fd - 10.0)
    best = hs[np.argmin(fd_err)]
    assert 1e-10 < best < 1e-6 and fd_err.min() < 1e-6

    fig = new_figure("Autodiff: record the forward pass, run it backward, read off dy/dx",
                     r"$y = x^2$ at $x = 5$:  $\frac{dy}{dx} = 2x = 10$.      "
                     "PyTorch x.grad → tensor(10.)   TensorFlow t.gradient(y, x) → 10.0")
    gs = grid(fig, top=0.82)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1, 8, 300)
    ax.plot(xs, xs ** 2, color=INK, linewidth=2.4, label="y = x²")
    t = np.array([2.5, 7.5])
    ax.plot(t, 25 + 10 * (t - 5), color=ORANGE, linewidth=2.4, label="tangent at x = 5: slope 10")
    ax.plot(5, 25, "o", color=INK, markersize=7, zorder=5)
    ax.annotate("(5, 25)", (5, 25), textcoords="offset points", xytext=(10, -12), fontsize=10)
    axes_cross(ax)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-5, 60)
    finish(ax, "The number autodiff returns: slope at x = 5", legend_loc="upper left")

    # -- computational graph
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("What .backward() does")
    box(ax, (0.02, 0.56), 0.25, 0.2, "x = 5\nleaf", BLUE)
    box(ax, (0.375, 0.56), 0.25, 0.2, "pow(·, 2)\nsaves x", INK2)
    box(ax, (0.73, 0.56), 0.25, 0.2, "y = 25", BLUE)
    flow_arrow(ax, (0.28, 0.69), (0.37, 0.69), BLUE)
    flow_arrow(ax, (0.63, 0.69), (0.725, 0.69), BLUE)
    ax.text(0.5, 0.88, "forward pass: requires_grad_() records each op", ha="center", fontsize=10.5,
            color=BLUE)
    flow_arrow(ax, (0.855, 0.54), (0.855, 0.34), ORANGE)
    ax.annotate("", xy=(0.145, 0.24), xytext=(0.82, 0.24),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=2, mutation_scale=16))
    ax.text(0.86, 0.24, "dy/dy = 1", ha="left", va="center", fontsize=10.5, color=ORANGE)
    ax.text(0.5, 0.29, "× local derivative 2x = 10", ha="center", fontsize=10.5, color=ORANGE)
    flow_arrow(ax, (0.145, 0.25), (0.145, 0.54), ORANGE)
    ax.text(0.5, 0.12, "backward pass (chain rule) → x.grad = 10", ha="center", fontsize=10.5,
            color=ORANGE, weight="bold")

    panel(fig.add_subplot(gs[0, 2]), "The notebook's two versions", [
        "PyTorch",
        "`x = torch.tensor(5.0)`",
        "`x.requires_grad_()`",
        "`y = x**2;  y.backward()`",
        "`x.grad  # tensor(10.)`",
        "TensorFlow 2",
        "`x = tf.Variable(5.0)`",
        "`with tf.GradientTape() as t:`",
        "`    t.watch(x); y = x**2`",
        "`t.gradient(y, x)  # numpy=10.0`",
    ], spacing=0.079, fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    gx = grid_x.detach().numpy()
    xs = np.linspace(-6, 6, 300)
    ax.plot(xs, 2 * xs, color=ORANGE, linewidth=2, label="exact dy/dx = 2x")
    ax.plot(gx, grid_x.grad.numpy(), "o", color=BLUE, markersize=6, label="x.grad for 25 x values")
    ax.plot(5, 10, "o", color=INK, markersize=9, markerfacecolor="none", markeredgewidth=2, zorder=5)
    ax.annotate("x = 5: 10", (5, 10), textcoords="offset points", xytext=(-10, 6), ha="right", fontsize=10)
    axes_cross(ax)
    finish(ax, "One backward pass, a whole tensor of slopes", "x", "dy/dx", "upper left")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(hs, np.maximum(fd_err, 1e-17), color=BLUE, linewidth=2, label="delta method (float64)")
    ax.plot(hs, hs, color=MUTED, linestyle="--", linewidth=1.2, label="truncation error = Δx")
    ax.axvline(1e-6, color=AQUA, linestyle=":", linewidth=1.6, label="Δx = 10⁻⁶ used in Segment 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 10)
    ax.invert_xaxis()
    finish(ax, "Delta method at x = 5 vs autodiff (error 0)", "Δx (log scale)  ← shrinking",
           "|slope − 10|", "upper left", 9)
    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=SURFACE,
              framealpha=1)
    ax.text(0.03, 0.12, "round-off wins\nbelow Δx ≈ 10⁻⁸;\nautodiff: exactly 10.0",
            transform=ax.transAxes, ha="left", fontsize=9.5, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "Why ML needs it", [
        "Training = minimise a cost (loss) C(θ),",
        "e.g. mean squared error of predictions.",
        "Autodiff gives ∂C/∂θ for millions of θ",
        "in one backward pass (backpropagation);",
        "gradient descent steps θ ← θ − η ∂C/∂θ.",
        "Worked example: the Regression in",
        "PyTorch notebook. Differentiable",
        "programming: any code built from these",
        "ops can be differentiated this way.",
        "",
        "TensorFlow isn't installed here: its tape is",
        "reproduced with torch.autograd; 10.0 matches.",
    ], spacing=0.072, fontsize=11.5)

    save(fig, name="3-calculus-i_seg3.png")


# ================================================================ notebook 4, segment 2
def sheet_nb4_seg2_partial_derivatives():
    def f(my_x, my_y):
        return my_x ** 2 - my_y ** 2

    def cylinder_vol(my_r, my_l):
        return math.pi * my_r ** 2 * my_l

    # autodiff at (0, 0)
    x = torch.tensor(0.).requires_grad_()
    y = torch.tensor(0.).requires_grad_()
    z = f(x, y)
    z.backward()
    assert z.item() == 0 and x.grad.item() == 0 and y.grad.item() == 0
    assert str(y.grad) == "tensor(-0.)"
    # the pencil-and-paper exercises, repeated with PyTorch as the notebook asks
    exercises = []
    for px, py in [(3, 0), (2, 3), (-2, -3)]:
        tx, ty = torch.tensor(float(px), requires_grad=True), torch.tensor(float(py), requires_grad=True)
        tz = f(tx, ty)
        tz.backward()
        assert tx.grad.item() == 2 * px and ty.grad.item() == -2 * py and tz.item() == px ** 2 - py ** 2
        exercises.append((px, py, int(tz.item()), int(tx.grad.item()), int(ty.grad.item())))

    # cylinder
    r = torch.tensor(3.).requires_grad_()
    l = torch.tensor(5.).requires_grad_()
    v = cylinder_vol(r, l)
    v.backward()
    assert f"{v.item():.4f}" == "141.3717" and f"{l.grad.item():.4f}" == "28.2743"
    assert f"{r.grad.item():.4f}" == "94.2478"
    assert math.pi * 3 ** 2 == 28.274333882308138 and 2 * math.pi * 3 * 5 == 94.24777960769379
    assert cylinder_vol(3, 6) == 169.64600329384882
    assert cylinder_vol(3, 6) - cylinder_vol(3, 5) == 28.274333882308127
    assert cylinder_vol(3, 7) - cylinder_vol(3, 6) == 28.274333882308156
    delta = 1e-6
    fd_r = (cylinder_vol(3 + delta, 5) - cylinder_vol(3, 5)) / delta
    assert fd_r == 94.24779531741478

    fig = new_figure("A partial derivative is a slope with every other input held still",
                     r"$z = x^2 - y^2$:  $\partial z/\partial x = 2x$,  $\partial z/\partial y = -2y$."
                     r"      Cylinder $v = \pi r^2 l$:  $\partial v/\partial l = \pi r^2 = 28.27$,"
                     r"  $\partial v/\partial r = 2\pi r l = 94.25$  at r = 3, l = 5")
    gs = grid(fig, top=0.82)

    # -- the saddle with the two slices
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    g = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(g, g)
    ax.plot_surface(X, Y, f(X, Y), cmap="Blues", alpha=0.55, linewidth=0, antialiased=True,
                    rstride=2, cstride=2)
    ax.plot(g, np.zeros_like(g), f(g, 0), color=BLUE, linewidth=3, label="slice y = 0: z = x²")
    ax.plot(np.full_like(g, 2), g, f(2, g), color=ORANGE, linewidth=3, label="slice x = 2: z = 4 − y²")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(-0.05, 1.02))
    ax.set_title("z = x² − y²: hold one input, get a curve", pad=4)

    xs = np.linspace(-3, 3, 1000)
    samples = [-2, -1, 0, 1, 2]
    colors = [RAMP[0], RAMP[1], AQUA, RAMP[2], RAMP[3]]
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, f(xs, 0), color=INK, linewidth=2.2, label="z = f(x, 0) = x²")
    for s, c in zip(samples, colors):
        t = np.array([s - 0.8, s + 0.8])
        ax.plot(t, f(s, 0) + 2 * s * (t - s), color=c, linewidth=2, linestyle="--")
        ax.plot(s, f(s, 0), "o", color=c, markersize=7, zorder=4)
        off, ha = ((-10, -12), "right") if s < 0 else (((10, -12), "left") if s > 0 else ((0, -16), "center"))
        ax.annotate(f"slope {2 * s:+d}" if s else "slope 0", (s, f(s, 0)), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=9.5, color=INK2)
    axes_cross(ax)
    ax.set_ylim(-1, 9)
    ax.set_xlim(-3, 3)
    finish(ax, "Vary x, hold y = 0: slopes ∂z/∂x = 2x", "x", "z", "upper center")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(xs, f(0, xs), color=MUTED, linewidth=1.6, linestyle="-", label="x = 0: z = −y²")
    ax.plot(xs, f(2, xs), color=INK, linewidth=2.2, label="x = 2: z = 4 − y²")
    for s, c in zip(samples, colors):
        t = np.array([s - 0.6, s + 0.6])
        ax.plot(t, f(2, s) - 2 * s * (t - s), color=c, linewidth=2, linestyle="--")
        ax.plot(s, f(2, s), "o", color=c, markersize=7, zorder=4)
        off, ha = ((-10, 6), "right") if s < 0 else (((10, 6), "left") if s > 0 else ((0, 9), "center"))
        ax.annotate(f"slope {-2 * s:+d}" if s else "slope 0", (s, f(2, s)), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=9.5, color=INK2)
    axes_cross(ax)
    ax.set_ylim(-5, 5.8)
    ax.set_xlim(-3, 3)
    finish(ax, "Vary y, hold x = 2: slopes ∂z/∂y = −2y", "y", "z", "lower center")

    # -- gradient field with the exercise points
    ax = fig.add_subplot(gs[1, 0])
    g = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(g, g)
    cs = ax.contour(X, Y, f(X, Y), levels=np.arange(-14, 15, 2), cmap="RdBu_r", linewidths=1)
    q = np.arange(-4, 4.1, 1.0)
    QX, QY = np.meshgrid(q, q)
    ax.quiver(QX, QY, 2 * QX, -2 * QY, color=MUTED, angles="xy", scale_units="xy", scale=12, width=0.004)
    for (px, py, pz, gxv, gyv), c in zip(exercises, [BLUE, ORANGE, AQUA]):
        ax.annotate("", xy=(px + gxv / 6, py + gyv / 6), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color=c, linewidth=2.4, mutation_scale=15))
        ax.plot(px, py, "o", color=INK, markersize=6, zorder=5)
        ax.plot([], [], color=c, linewidth=2.4,
                label=f"({px}, {py}): z = {pz}, ∇z = ({gxv}, {gyv})")
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    finish(ax, "Gradient ∇z = (2x, −2y): exercises via autograd", "x", "y")
    ax.legend(loc="lower right", fontsize=8.5, bbox_to_anchor=(1.02, -0.02), framealpha=0.9,
              frameon=True, facecolor=SURFACE, edgecolor=SURFACE)

    # -- cylinder: v is linear in l, curved in r
    ax = fig.add_subplot(gs[1, 1])
    dd = np.linspace(0, 2, 200)
    v0 = cylinder_vol(3, 5)
    ax.plot(dd, cylinder_vol(3, 5 + dd), color=BLUE, linewidth=2.4, label="vary l (r = 3): slope πr² = 28.27")
    ax.plot(dd, cylinder_vol(3 + dd, 5), color=ORANGE, linewidth=2.4,
            label="vary r (l = 5): curve, slope 2πrl = 94.25")
    ax.plot(dd, v0 + 94.24777960769379 * dd, color=ORANGE, linestyle="--", linewidth=1.4,
            label="tangent ∂v/∂r at r = 3")
    for k in (1, 2):
        ax.plot(k, cylinder_vol(3, 5 + k), "o", color=BLUE, markersize=6, zorder=4)
    ax.annotate("v(3, 6) = 169.65\nv(3, 7) = 197.92", (2, cylinder_vol(3, 7)), textcoords="offset points",
                xytext=(-6, -36), ha="right", fontsize=9.5, color=BLUE)
    ax.plot(0, v0, "o", color=INK, markersize=7, zorder=5)
    ax.set_xlim(0, 2)
    ax.set_ylim(100, 520)
    finish(ax, "Cylinder volume from v(3, 5) = 141.37", "change in the input (m)", "v (m³)",
           "upper left", 9)

    panel(fig.add_subplot(gs[1, 2]), "PyTorch gets every partial at once", [
        "`r = torch.tensor(3.).requires_grad_()`",
        "`l = torch.tensor(5.).requires_grad_()`",
        "`v = cylinder_vol(r, l)  # 141.3717`",
        "`v.backward()`",
        "`l.grad  # 28.2743 = π·3²`",
        "`r.grad  # 94.2478 = 2π·3·5`",
        "Each extra metre of l adds exactly 28.27 m³;",
        r"$\partial v/\partial r$ holds only for tiny Δr:",
        "Δr = 10⁻⁶ gives 94.24779531741478.",
        "At (0, 0): x.grad = 0, y.grad = −0.",
    ], spacing=0.078, fontsize=12)

    save(fig, name="4-calculus-ii_seg2.png")


# ================================================================ notebook 4, segment 3
def sheet_nb4_seg3_integrals():
    def g(x):
        return x / 2

    def h(x):
        return 2 * x

    ig, eg = quad(g, 1, 2)
    ih, eh = quad(h, 3, 4)
    assert ig == 0.75 and eg < 1e-12 and ih == 7.0 and eh < 1e-12      # printed: (0.75, 8.3e-15), (7.0, 7.8e-14)
    assert abs((2 ** 2 / 4 - 1 ** 2 / 4) - ig) < 1e-15 and (4 ** 2 - 3 ** 2) == ih
    roc_x = [0, 0, 0.5, 0.5, 1]
    roc_y = [0, 0.5, 0.5, 1, 1]
    area = auc(roc_x, roc_y)
    assert area == 0.75

    ns = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    left = np.array([sum(h(3 + k / n) for k in range(n)) / n for n in ns])
    right = np.array([sum(h(3 + (k + 1) / n) for k in range(n)) / n for n in ns])
    assert np.allclose(left, 7 - 1 / ns) and np.allclose(right, 7 + 1 / ns)

    fig = new_figure("An integral is the area under a curve: quad for functions, auc for points",
                     r"$\int_1^2 \frac{x}{2}\,dx = \frac{3}{4}$,     $\int_3^4 2x\,dx = 7$,"
                     "     ROC curve from five (x, y) points: AUC = 0.75 (trapezoidal rule)")
    gs = grid(fig, top=0.82)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, 3, 300)
    ax.plot(xs, g(xs), color=INK, linewidth=2.4, label="g(x) = x/2")
    sx = np.linspace(1, 2, 100)
    ax.fill_between(sx, 0, g(sx), color=BLUE_LIGHT, label=f"area = {ig:g}")
    ax.text(1.5, 0.3, "3/4", ha="center", fontsize=16, weight="bold", color=BLUE)
    axes_cross(ax)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.6)
    finish(ax, "quad(g, 1, 2) → (0.75, 8.3e-15)", "x", "g(x)", "upper left")

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0, 5, 300)
    ax.plot(xs, h(xs), color=INK, linewidth=2.4, label="h(x) = 2x")
    sx = np.linspace(3, 4, 100)
    ax.fill_between(sx, 0, h(sx), color=BLUE_LIGHT, label=f"area = {ih:g}")
    ax.text(3.5, 3, "7", ha="center", fontsize=16, weight="bold", color=BLUE)
    axes_cross(ax)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 11)
    finish(ax, "quad(h, 3, 4) → (7.0, 7.8e-14)", "x", "h(x)", "upper left")

    # -- the antiderivative view
    ax = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0, 5, 300)
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.4, label="F(x) = x²  (F′ = h = 2x)")
    ax.plot([3, 4], [9, 9], color=MUTED, linewidth=1.2, linestyle=":")
    ax.annotate("", xy=(4, 16), xytext=(4, 9),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, linewidth=2.2, mutation_scale=14))
    ax.text(4.1, 12.5, "F(4) − F(3)\n= 16 − 9 = 7", fontsize=10.5, color=ORANGE, va="center")
    ax.plot([3, 4], [9, 16], "o", color=INK, markersize=6, zorder=4)
    axes_cross(ax)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 26)
    finish(ax, "By hand: area = rise of the antiderivative", "x", "F(x)", "upper left")

    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(7, color=ORANGE, linestyle=":", linewidth=1.6, label="quad: 7.0")
    ax.plot(ns, left, "-o", color=BLUE, linewidth=2, markersize=5, label="left rectangles: 7 − 1/n")
    ax.plot(ns, right, "-s", color=RAMP[3], linewidth=2, markersize=5, label="right rectangles: 7 + 1/n")
    ax.set_xscale("log", base=2)
    ax.set_ylim(5.4, 8.1)
    ax.set_xticks(ns, [str(n) for n in ns])
    finish(ax, "Thinner slices of ∫₃⁴ 2x dx converge to 7", "number of rectangles n", "area estimate",
           "upper right")
    ins = ax.inset_axes([0.55, 0.05, 0.42, 0.26])
    xx = np.linspace(3, 4, 50)
    ins.plot(xx, h(xx), color=INK, linewidth=1.6)
    for k in range(4):
        ins.add_patch(Rectangle((3 + k / 4, 0), 0.25, h(3 + k / 4), facecolor=BLUE_LIGHT,
                                edgecolor=BLUE, linewidth=0.8))
    ins.set_xlim(3, 4)
    ins.set_ylim(5, 8.2)
    ins.set_xticks([3, 4])
    ins.set_yticks([])
    ins.tick_params(labelsize=8)
    ins.set_title("n = 4, left", fontsize=8.5, pad=2)

    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(roc_x, 0, roc_y, color=BLUE_LIGHT, step=None, label=f"AUC = {area:g}")
    ax.plot(roc_x, roc_y, "-o", color=BLUE, linewidth=2.4, markersize=7, label="hot-dog ROC")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="chance (0.5)")
    ax.text(0.3, 0.07, "0.5 × 0.5 = 0.25", ha="center", fontsize=10, color=INK2)
    ax.text(0.8, 0.58, "0.5 × 1\n= 0.5", ha="center", fontsize=10, color=INK2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_aspect("equal")
    finish(ax, "auc(xs, ys) → 0.75", "false positive rate", "true positive rate", "upper left", 9)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95, frameon=True, facecolor=SURFACE,
              edgecolor=SURFACE)

    panel(fig.add_subplot(gs[1, 2]), "Three ways to get the area", [
        "By hand (antiderivative, then subtract):",
        r"$\int_1^2 \frac{x}{2}dx = \left[\frac{x^2}{4}\right]_1^2 = 1 - \frac{1}{4} = \frac{3}{4}$",
        r"$\int_3^4 2x\,dx = \left[x^2\right]_3^4 = 16 - 9 = 7$",
        "Function known → numeric quadrature:",
        "`quad(h, 3, 4)  # (7.0, 7.8e-14)`",
        "  2nd value: estimated absolute error",
        "Only points known → trapezoidal rule:",
        "`auc([0, 0, .5, .5, 1],`",
        "`    [0, .5, .5, 1, 1])  # 0.75`",
    ], spacing=0.075, fontsize=12)

    save(fig, name="4-calculus-ii_seg3.png")


if __name__ == "__main__":
    sheet_nb3_seg1_limits()
    sheet_nb3_seg2_delta_method()
    sheet_nb3_seg3_autodiff()
    sheet_nb4_seg2_partial_derivatives()
    sheet_nb4_seg3_integrals()
