#!/usr/bin/env python3
"""Concept sheets for notebooks/1-intro-to-linear-algebra.ipynb and notebooks/2-linear-algebra-ii.ipynb.

One page per segment ("## Segment N: ..."), three per notebook. Every number is computed here from
the same vectors, matrices and data the notebook uses and checked against its printed outputs with
asserts. TensorFlow is not installed: its cells print the same values as the NumPy / PyTorch cells,
which are what is reproduced here.

Notebook 2's SVD image compression downloads a photo (oboe-with-book.jpg, 4032 x 3024) that is not
on disk; the sheet runs the same code on the course's own drawing img/Oboe.jpg instead and quotes
the photo's storage numbers from the notebook's printed output.

Images are written to notebooks/img/<notebook stem>_seg<N>.png.

Run from anywhere:  python scripts/make_nb1_2_concept_images.py
"""
from functools import partial

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from PIL import Image
from sklearn import datasets
from sklearn.decomposition import PCA

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, GRID, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       SEQ, new_figure, grid, panel, frac_label)
from viz_style import save as _save

NOTEBOOKS = REPO / "notebooks"
save = partial(_save, folder=NOTEBOOKS / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
DIV = LinearSegmentedColormap.from_list(
    "div_orange_blue", ["#a8461d", ORANGE, "#f8c9b3", "#f6f5f1", "#cde2fb", "#3987e5", "#0d366b"])
CIRCLE_T = np.linspace(0, 2 * np.pi, 400)
UNIT_CIRCLE = np.vstack([np.cos(CIRCLE_T), np.sin(CIRCLE_T)])


# ================================================================ helpers
def mat(ax, M, title, labels=None, vmax=None, fontsize=11, highlight_cells=(), box_color=ORANGE,
        title_pad=8, row_names=None, col_names=None):
    """Signed matrix as an annotated heatmap: blue > 0, orange < 0, square cells."""
    M = np.asarray(M, float)
    vmax = vmax or (np.abs(M).max() or 1.0)
    n_rows, n_cols = M.shape
    ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1), M, cmap=DIV, vmin=-vmax,
                  vmax=vmax, edgecolors=SURFACE, linewidth=2)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelsize=9)
    ax.set_xticks(np.arange(n_cols) + 0.5, col_names if col_names else [""] * n_cols)
    ax.set_yticks(np.arange(n_rows) + 0.5, row_names if row_names else [""] * n_rows)
    if col_names:
        ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=title_pad, fontsize=12)
    for i in range(n_rows):
        for j in range(n_cols):
            value = M[i, j]
            text = labels[i][j] if labels is not None else frac_label(value, 24)
            if abs(value) < 1e-12:
                color = MUTED
            elif abs(value) / vmax > 0.6:
                color = "white"
            else:
                color = INK
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=fontsize, color=color)
    for i, j in highlight_cells:
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor=box_color, linewidth=3,
                               clip_on=False))


def labels_of(M, fmt):
    return [[fmt(v) for v in row] for row in np.asarray(M, float)]


def dec(digits):
    def fmt(v):
        return "0" if abs(v) < 0.5 * 10 ** -digits else f"{v:.{digits}f}"
    return fmt


def integer(v):
    return f"{v:.0f}"


def sub_mats(fig, cell, mats, wspace=0.35, width_ratios=None, title=None, ops=None, title_dy=0.05):
    """Several small heatmaps side by side inside one grid cell; ops are drawn between them."""
    sub = cell.subgridspec(1, len(mats), wspace=wspace, width_ratios=width_ratios)
    axes = []
    for k, spec in enumerate(mats):
        ax = fig.add_subplot(sub[0, k])
        mat(ax, *spec[:2], **(spec[2] if len(spec) > 2 else {}))
        axes.append(ax)
    if ops:
        for k, op in enumerate(ops):
            left, right = axes[k].get_position(), axes[k + 1].get_position()
            fig.text((left.x1 + right.x0) / 2, (left.y0 + left.y1) / 2, op, ha="center",
                     va="center", fontsize=16, color=INK2, weight="bold")
    if title:
        first, last = axes[0].get_position(), axes[-1].get_position()
        fig.text((first.x0 + last.x1) / 2, max(a.get_position().y1 for a in axes) + title_dy, title,
                 ha="center", va="bottom", fontsize=13, weight="bold")
    return axes


def plane(ax, title, xlim, ylim, xlabel="x", ylabel="y"):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
    ax.grid(True, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def arrow(ax, v, color, label=None, start=(0, 0), offset=(6, 4), width=2.2, fontsize=10.5,
          ha="left", box=False):
    ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=width, mutation_scale=14,
                                shrinkA=0, shrinkB=0), zorder=4)
    if label:
        bbox = dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5) if box else None
        ax.annotate(label, (start[0] + v[0], start[1] + v[1]), textcoords="offset points",
                    xytext=offset, color=color, fontsize=fontsize, weight="bold", ha=ha, zorder=5,
                    bbox=bbox)


def boxes(ax, x0, y0, rows, cols, values=None, size=0.42, color=BLUE_LIGHT, fontsize=9):
    """A little grid of cells (top-left corner at x0, y0) used for shape diagrams."""
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x0 + j * size, y0 - (i + 1) * size), size, size,
                                   facecolor=color, edgecolor=SURFACE, linewidth=2))
            if values is not None:
                ax.text(x0 + (j + 0.5) * size, y0 - (i + 0.5) * size, str(values[i][j]),
                        ha="center", va="center", fontsize=fontsize, color=INK)


def style_3d(ax, xlabel="x", ylabel="y", zlabel="z"):
    ax.set_xlabel(xlabel, labelpad=0)
    ax.set_ylabel(ylabel, labelpad=0)
    ax.set_zlabel(zlabel, labelpad=0)
    ax.tick_params(labelsize=8, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.97, 0.97, 0.96, 1.0))


def two_lines(ax, slope_intercepts, colors, labels, xs):
    for (m, c), color, label in zip(slope_intercepts, colors, labels):
        ax.plot(xs, m * xs + c, color=color, linewidth=2.2, label=label)


# ================================================================ notebook 1, segment 1
def sheet_nb1_seg1_data_structures():
    # What linear algebra is: the bank robber
    t = np.linspace(0, 40, 1000)
    d_r = 2.5 * t
    d_s = 3 * (t - 5)
    catch_t, catch_d = np.linalg.solve([[2.5, -1], [3, -1]], [0, 15])
    assert np.allclose([catch_t, catch_d], [30, 75])   # the notebook marks t = 30, d = 75

    # vectors and norms
    x = np.array([25, 2, 5])
    l2, l1, l2_sq, l_max = np.linalg.norm(x), np.abs(x).sum(), x @ x, np.abs(x).max()
    assert l2 == 25.573423705088842 and l1 == 32 and l2_sq == 654 and l_max == 25
    y = np.array([[25, 2, 5]])
    assert x.shape == (3,) and x.T.shape == (3,) and y.shape == (1, 3) and y.T.shape == (3, 1)
    i, j = np.array([1, 0]), np.array([0, 1])
    assert i @ j == 0

    # matrices and higher-rank tensors (TensorFlow cells print the same values)
    X = np.array([[25, 2], [5, 26], [3, 7]])
    assert X.shape == (3, 2) and X.size == 6
    assert X[:, 0].tolist() == [25, 5, 3] and X[1, :].tolist() == [5, 26]
    assert X[0:2, 0:2].tolist() == [[25, 2], [5, 26]]
    X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
    assert tuple(X_pt.shape) == (3, 2) and X_pt[1, :].tolist() == [5, 26]
    images_pt = torch.zeros([32, 28, 28, 3])
    n_numbers = images_pt.numel()
    assert n_numbers == 75264 and images_pt.dim() == 4

    fig = new_figure("A tensor is an array of numbers; its rank is how many indices it takes",
                     "Scalar (rank 0) → vector (rank 1) → matrix (rank 2) → rank-4 image batch.   "
                     "A norm measures a vector's size:  for x = (25, 2, 5),  "
                     "‖x‖₁ = 32,  ‖x‖₂ = 25.57,  ‖x‖∞ = 25.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, d_r, color=AQUA, linewidth=2.4, label="robber:  d = 2.5 t")
    ax.plot(t, d_s, color=ORANGE, linewidth=2.4, label="sheriff:  d = 3 (t − 5)")
    ax.axvline(catch_t, color=MUTED, linestyle="--", linewidth=1)
    ax.axhline(catch_d, color=MUTED, linestyle="--", linewidth=1)
    ax.plot(catch_t, catch_d, "o", color=INK, markersize=8, zorder=5)
    ax.annotate("caught at t = 30 min, d = 75 km", (catch_t, catch_d), xytext=(-12, 12),
                textcoords="offset points", ha="right", fontsize=10.5, color=INK)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 100)
    ax.set_xlabel("time (in minutes)")
    ax.set_ylabel("distance (in km)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95, edgecolor="none")
    ax.set_title("Two unknowns, two equations: one solution")

    ax = fig.add_subplot(gs[0, 1])
    names = ["$L^1$\nΣ|xᵢ|", "$L^2$\n√(Σxᵢ²)", "max norm\nmax|xᵢ|"]
    values = [l1, l2, l_max]
    colors = [AQUA, BLUE, ORANGE]
    pos = np.arange(3)
    ax.bar(pos, values, width=0.6, color=colors, zorder=2)
    for p, v in zip(pos, values):
        ax.text(p, v + 0.8, f"{v:.2f}" if v % 1 else f"{v:.0f}", ha="center", va="bottom",
                fontsize=11)
    ax.set_xticks(pos, names)
    ax.set_ylim(0, 40)
    ax.set_ylabel("size of x")
    ax.grid(axis="y", zorder=0)
    ax.text(2.45, 37.5, "squared $L^2$ = x·x = 654\n(no square root: cheap,\nused in ML losses)",
            ha="right", va="top", fontsize=10, color=INK2)
    ax.set_title("Norms of x = (25, 2, 5)")

    ax = fig.add_subplot(gs[0, 2])
    s = np.linspace(-1, 1, 5)
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    square = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    ax.plot(diamond[:, 0], diamond[:, 1], color=AQUA, linewidth=2, label="‖u‖₁ = 1")
    ax.plot(*UNIT_CIRCLE, color=BLUE, linewidth=2, label="‖u‖₂ = 1")
    ax.plot(square[:, 0], square[:, 1], color=ORANGE, linewidth=2, label="‖u‖∞ = 1")
    arrow(ax, i, INK, "i = (1, 0)", offset=(4, -14), width=2.6)
    arrow(ax, j, INK, "j = (0, 1)", offset=(6, 2), width=2.6)
    ax.text(0.12, 0.12, "i·j = 0", fontsize=10.5, color=INK2)
    plane(ax, "Unit vectors under each norm; i ⊥ j", (-1.6, 1.6), (-1.4, 1.75),
          xlabel="x₁", ylabel="x₂")
    ax.set_xticks(s)
    ax.set_yticks(s)
    ax.legend(loc="upper left", fontsize=9, ncol=3, handlelength=1.2, columnspacing=0.8)

    ax = fig.add_subplot(gs[1, 0])
    ax.pcolormesh(np.arange(3), np.arange(4), X, cmap=SEQ, vmin=0, vmax=40, edgecolors=SURFACE,
                  linewidth=2)
    for r in range(3):
        for c in range(2):
            ax.text(c + 0.5, r + 0.5, str(X[r, c]), ha="center", va="center", fontsize=13,
                    color=INK)
    ax.add_patch(Rectangle((0, 0), 1, 3, fill=False, edgecolor=ORANGE, linewidth=3, clip_on=False))
    ax.add_patch(Rectangle((0.06, 1.06), 1.88, 0.88, fill=False, edgecolor=AQUA, linewidth=3,
                           clip_on=False))
    ax.set_xlim(0, 2)
    ax.set_ylim(3, 0)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5], ["col 0", "col 1"])
    ax.set_yticks([0.5, 1.5, 2.5], ["row 0", "row 1", "row 2"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(2.25, 0.35, "X.shape = (3, 2), X.size = 6", fontsize=10.5, va="center")
    ax.text(2.25, 1.0, "X[:, 0] = [25, 5, 3]", fontsize=10.5, color=ORANGE, va="center",
            weight="bold")
    ax.text(2.25, 1.6, "X[1, :] = [5, 26]", fontsize=10.5, color=AQUA, va="center", weight="bold")
    ax.text(2.25, 2.3, "zero-indexed in code,\none-indexed on paper", fontsize=10, color=INK2,
            va="center")
    ax.set_title("A matrix: rows and columns by index", pad=26, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Rank = number of indices")
    boxes(ax, 0.2, 4.4, 1, 1, [[25]], size=0.6, fontsize=10)
    ax.text(0.5, 2.9, "rank 0\nscalar\nshape ()", ha="center", va="top", fontsize=10)
    boxes(ax, 1.6, 4.4, 3, 1, [[25], [2], [5]], size=0.6, fontsize=10)
    ax.text(1.9, 2.3, "rank 1\nvector\n(3,)", ha="center", va="top", fontsize=10)
    boxes(ax, 3.1, 4.4, 3, 2, X.tolist(), size=0.6, fontsize=10)
    ax.text(3.7, 2.3, "rank 2\nmatrix\n(3, 2)", ha="center", va="top", fontsize=10)
    for k, shade in enumerate(["#e3eefb", "#cde2fb", BLUE_LIGHT]):
        boxes(ax, 5.6 + 0.35 * k, 4.75 - 0.35 * k, 3, 3, size=0.45, color=shade)
    ax.text(6.3, 2.3, "rank 3\none image\n(28, 28, 3)", ha="center", va="top", fontsize=10)
    for k in range(4):
        boxes(ax, 8.0 + 0.22 * k, 4.75 - 0.22 * k, 2, 2, size=0.36,
              color=["#e3eefb", "#cde2fb", BLUE_LIGHT, "#86b6ef"][k])
    ax.text(8.75, 2.3, "rank 4\nbatch of 32\n(32, 28, 28, 3)", ha="center", va="top", fontsize=10)
    ax.text(5.0, 0.35, f"torch.zeros([32, 28, 28, 3]) holds {n_numbers:,} numbers",
            ha="center", fontsize=10.5, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "Same tensors in three libraries", [
        "`np.array([25, 2, 5])      # shape (3,)",
        "`torch.tensor([25, 2, 5])  # torch.Size([3])",
        "`tf.Variable([25, 2, 5])   # TensorShape([3])",
        "",
        "Transposing a 1-D array does nothing;",
        "with nested brackets [[25, 2, 5]] the shape",
        "is (1, 3) and .T gives the column (3, 1).",
        "PyTorch tensors feel like NumPy arrays and",
        "run on a GPU. (TensorFlow is not installed",
        "here; its cells print the same values.)",
    ], spacing=0.085, fontsize=12)
    save(fig, name="1-intro-to-linear-algebra_seg1.png")


# ================================================================ notebook 1, segment 2
def sheet_nb1_seg2_tensor_operations():
    X = np.array([[25, 2], [5, 26], [3, 7]])
    assert X.T.tolist() == [[25, 5, 3], [2, 26, 7]]
    assert (X * 2 + 2).tolist() == [[52, 6], [12, 54], [8, 16]]
    A = X + 2
    assert A.tolist() == [[27, 4], [7, 28], [5, 9]]
    assert (A * X).tolist() == [[675, 8], [35, 728], [15, 63]]
    X_pt = torch.tensor(X)
    assert ((X_pt + 2) * X_pt).tolist() == (A * X).tolist()
    assert X.sum() == 68 and X.sum(axis=0).tolist() == [33, 35] and X.sum(axis=1).tolist() == [27, 31, 10]

    x, y = np.array([25, 2, 5]), np.array([0, 1, 2])
    products = x * y
    assert products.tolist() == [0, 2, 10] and x @ y == 12
    assert torch.dot(torch.tensor([25, 2, 5.]), torch.tensor([0, 1, 2.])).item() == 12.0

    # Substitution: y = 3x, -5x + 2y = 2.   Elimination: 2x - 3y = 15, 4x + 10y = 14.
    sub = np.linalg.solve([[-3, 1], [-5, 2]], [0, 2])
    eli = np.linalg.solve([[2, -3], [4, 10]], [15, 14])
    assert np.allclose(sub, [2, 6]) and np.allclose(eli, [6, -1])   # the notebook's dashed lines

    fig = new_figure("Tensor operations: element-wise, reductions, and the dot product",
                     "Same-shape tensors combine cell by cell (A ⊙ X);  a reduction sums cells away;  "
                     "x · y = Σ xᵢyᵢ does both and leaves one number.")
    gs = grid(fig, top=0.80, wspace=0.3, hspace=0.55)

    sub_mats(fig, gs[0, 0], [
        (X, "X  (3 × 2)", dict(labels=labels_of(X, integer), vmax=40, fontsize=12,
                               highlight_cells=[(0, 0), (1, 0), (2, 0)])),
        (X.T, "Xᵀ  (2 × 3)", dict(labels=labels_of(X.T, integer), vmax=40, fontsize=12,
                                  highlight_cells=[(0, 0), (0, 1), (0, 2)])),
    ], ops=["→"], wspace=0.45, title="Transpose: rows become columns", title_dy=0.045)

    hp = A * X
    sub_mats(fig, gs[0, 1], [
        (A, "A = X + 2", dict(labels=labels_of(A, integer), vmax=40, fontsize=10.5)),
        (X, "X", dict(labels=labels_of(X, integer), vmax=40, fontsize=10.5)),
        (hp, "A ⊙ X", dict(labels=labels_of(hp, integer), vmax=800, fontsize=10.5,
                           highlight_cells=[(0, 0)])),
    ], ops=["⊙", "="], wspace=0.5, title="Hadamard product: cell by cell", title_dy=0.045)

    ax = fig.add_subplot(gs[0, 2])
    full = np.zeros((4, 3))
    full[:3, :2] = X
    ax.pcolormesh(np.arange(3), np.arange(4), X, cmap=SEQ, vmin=0, vmax=40, edgecolors=SURFACE,
                  linewidth=2)
    for r in range(3):
        for c in range(2):
            ax.text(c + 0.5, r + 0.5, str(X[r, c]), ha="center", va="center", fontsize=12)
        ax.text(2.5, r + 0.5, str(X.sum(axis=1)[r]), ha="center", va="center", fontsize=12,
                color=AQUA, weight="bold")
    for c in range(2):
        ax.text(c + 0.5, 3.5, str(X.sum(axis=0)[c]), ha="center", va="center", fontsize=12,
                color=ORANGE, weight="bold")
    ax.text(2.5, 3.5, str(X.sum()), ha="center", va="center", fontsize=13, color=INK, weight="bold")
    ax.add_patch(Rectangle((2.1, 3.1), 0.8, 0.8, fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_xlim(0, 5.4)
    ax.set_ylim(4, 0)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5, 2.5], ["", "", "axis=1"])
    ax.set_yticks([3.5], ["axis=0"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(3.2, 1.5, "X.sum(axis=1)\n→ [27, 31, 10]", color=AQUA, fontsize=10.5, va="center")
    ax.text(3.2, 3.5, "X.sum(axis=0)\n→ [33, 35]", color=ORANGE, fontsize=10.5, va="center")
    ax.set_title("Reduction: sum an axis away (or all)", pad=26)

    ax = fig.add_subplot(gs[1, 0])
    pos = np.arange(3)
    w = 0.26
    ax.bar(pos - w, x, width=w - 0.02, color=BLUE, label="x = (25, 2, 5)", zorder=2)
    ax.bar(pos, y, width=w - 0.02, color=AQUA, label="y = (0, 1, 2)", zorder=2)
    ax.bar(pos + w, products, width=w - 0.02, color=ORANGE, label="xᵢ yᵢ", zorder=2)
    for p, v in zip(pos, products):
        ax.text(p + w, v + 0.5, str(v), ha="center", fontsize=11, color=ORANGE, weight="bold")
    ax.set_xticks(pos, ["i = 1", "i = 2", "i = 3"])
    ax.set_ylim(0, 30)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=10)
    ax.text(2.4, 15.5, "x · y = 0 + 2 + 10 = 12", ha="right", fontsize=12, weight="bold")
    ax.set_title("Dot product: multiply, then reduce")

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, 3 * xs, color=AQUA, linewidth=2.2, label="y = 3x")
    ax.plot(xs, 1 + 5 * xs / 2, color=BLUE, linewidth=2.2, label="−5x + 2y = 2")
    ax.plot(xs, -5 + 2 * xs / 3, color=ORANGE, linewidth=2.2, label="2x − 3y = 15")
    ax.plot(xs, (7 - 2 * xs) / 5, color="#a8461d", linewidth=2.2, label="4x + 10y = 14")
    ax.plot(*sub, "o", color=INK, markersize=8, zorder=5)
    ax.plot(*eli, "o", color=INK, markersize=8, zorder=5)
    ax.annotate("substitution:\n(2, 6)", sub, xytext=(-12, 4), textcoords="offset points",
                ha="right", fontsize=10.5)
    ax.annotate("elimination:\n(6, −1)", eli, xytext=(8, -32), textcoords="offset points",
                fontsize=10.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-2, 10)
    ax.set_ylim(-6, 13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title("Linear systems: the lines cross at the solution")

    panel(fig.add_subplot(gs[1, 2]), "The same operations in each library", [
        "`X.T        torch: X_pt.T   tf.transpose(X_tf)",
        "`X*2+2      overloaded in all three",
        "`A * X      Hadamard, NOT matrix product",
        "`X.sum()    torch.sum(X_pt)  tf.reduce_sum",
        "`np.dot(x, y)   torch.dot(x, y)  (floats)",
        "",
        "A dot product is what every artificial",
        "neuron computes: inputs · weights.",
        "Solve by substitution (isolate y) or by",
        "elimination (cancel a variable).",
    ], spacing=0.085, fontsize=12)
    save(fig, name="1-intro-to-linear-algebra_seg2.png")


# ================================================================ notebook 1, segment 3
def sheet_nb1_seg3_matrix_properties():
    X_f = np.array([[1, 2], [3, 4]])
    frob = np.linalg.norm(X_f)
    assert frob == 5.477225575051661 and abs(torch.norm(torch.tensor([[1, 2], [3, 4.]])).item() - 5.4772) < 1e-4

    A = np.array([[3, 4], [5, 6], [7, 8]])
    b = np.array([1, 2])
    Ab = A @ b
    assert Ab.tolist() == [11, 17, 23]
    B = np.array([[1, 9], [2, 0]])
    AB = A @ B
    assert AB.tolist() == [[11, 27], [17, 45], [23, 63]]

    X_sym = np.array([[0, 1, 2], [1, 7, 8], [2, 8, 9]])
    assert (X_sym.T == X_sym).all()
    I3 = torch.eye(3, dtype=torch.int64)
    assert torch.matmul(I3, torch.tensor([25, 2, 5])).tolist() == [25, 2, 5]
    M_q = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    V_q = torch.tensor([[-1, 1, -2], [0, 1, 2]]).T
    assert torch.matmul(M_q, V_q).tolist() == [[-3, 5], [-9, 14], [-15, 23]]

    X = np.array([[4, 2], [-5, -3]])
    Xinv = np.linalg.inv(X)
    assert np.allclose(Xinv, [[1.5, 1.0], [-2.5, -2.0]])
    y = np.array([4, -7])
    w = Xinv @ y
    assert np.allclose(w, [-1, 4]) and np.allclose(X @ w, y)
    N = np.array([[-4, 1], [-8, 2]])
    assert np.allclose(N[:, 0], -4 * N[:, 1])   # parallel columns: no inverse

    K = torch.tensor([[2 / 3, 1 / 3, 2 / 3], [-2 / 3, 2 / 3, 1 / 3], [1 / 3, 2 / 3, -2 / 3]])
    KtK = torch.matmul(K.T, K)
    off_diag = (KtK - torch.eye(3)).abs().max().item()
    assert off_diag < 1e-6 and abs(KtK[0, 1].item() + 3.3114e-09) < 1e-12   # printed -3.3114e-09
    for c in range(3):
        assert abs(torch.norm(K[:, c]).item() - 1) < 1e-6
    K = K.numpy().astype(float)

    fig = new_figure("Matrices act on vectors:  y = Xw,  and X⁻¹ undoes it when it exists",
                     "AB = A applied to each column of B.   X⁻¹X = I, so w = X⁻¹y.   "
                     "Orthogonal K: orthonormal columns,  KᵀK = I.   "
                     r"$\|X\|_F = \sqrt{\sum X_{ij}^2}$.")
    gs = grid(fig, top=0.80, wspace=0.3, hspace=0.55)

    sub_mats(fig, gs[0, 0], [
        (A, "A", dict(labels=labels_of(A, integer), vmax=70, highlight_cells=[(0, 0), (0, 1)])),
        (B, "B", dict(labels=labels_of(B, integer), vmax=70,
                      highlight_cells=[(0, 0), (1, 0)], box_color=AQUA)),
        (AB, "AB", dict(labels=labels_of(AB, integer), vmax=70, highlight_cells=[(0, 0)])),
    ], ops=["·", "="], wspace=0.5, width_ratios=[2, 2, 2],
        title="AB: row of A · column of B", title_dy=0.045)

    ax = fig.add_subplot(gs[0, 1])
    bs = np.linspace(-10, 10, 1000)
    ax.plot(bs, 2 - 2 * bs, color=BLUE, linewidth=2.2, label="4b + 2c = 4")
    ax.plot(bs, (7 - 5 * bs) / 3, color=ORANGE, linewidth=2.2, label="−5b − 3c = −7")
    ax.plot(*w, "o", color=INK, markersize=8, zorder=5)
    ax.annotate("w = X⁻¹y = (−1, 4)", w, xytext=(10, 4), textcoords="offset points", fontsize=10.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1, 5)
    ax.set_xlabel("b")
    ax.set_ylabel("c")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Xw = y with X = [[4, 2], [−5, −3]], y = (4, −7)")

    ax = fig.add_subplot(gs[0, 2])
    span = np.outer([-4.75, 2.6], N[:, 1])
    ax.plot(span[:, 0], span[:, 1], color=ORANGE, linestyle=":", linewidth=1.8,
            label="N·w for every w: only this line")
    arrow(ax, X[:, 0], BLUE, "X col 1 = (4, −5)", offset=(-4, -16), fontsize=10)
    arrow(ax, X[:, 1], AQUA, "X col 2 = (2, −3)", offset=(8, -2), fontsize=10)
    arrow(ax, N[:, 0], ORANGE, "N col 1 = (−4, −8)", offset=(8, 2), fontsize=10)
    arrow(ax, N[:, 1], "#a8461d", "N col 2 = (1, 2)", offset=(8, -6), fontsize=10)
    ax.text(-6.7, 4.7, "N: col 1 = −4 · col 2,\nso every N·w lands\non the dotted line",
            fontsize=9.5, color=ORANGE, va="top", ha="left")
    ax.text(6.8, -9.6, "X: two directions → X⁻¹ exists", fontsize=9.5, color=BLUE,
            va="bottom", ha="right")
    plane(ax, "Singular N = [[−4, 1], [−8, 2]]: no inverse", (-7, 7), (-10, 5),
          xlabel="x₁", ylabel="x₂")

    sub_mats(fig, gs[1, 0], [
        (K, "K", dict(labels=labels_of(K, lambda v: frac_label(v, 3)), vmax=1)),
        (KtK.numpy().astype(float), "KᵀK", dict(labels=[["1", "0", "0"], ["0", "1", "0"],
                                                        ["0", "0", "1"]], vmax=1)),
    ], ops=["→"], wspace=0.45, title="Orthogonal: KᵀK = I", title_dy=0.045)

    ax = fig.add_subplot(gs[1, 1], projection="3d")
    for c, color in zip(range(3), [BLUE, ORANGE, AQUA]):
        ax.quiver(0, 0, 0, *K[:, c], color=color, linewidth=2.4, arrow_length_ratio=0.12)
        ax.text(*(K[:, c] * 1.15), f"K col {c + 1}", color=color, fontsize=10, weight="bold")
        e = np.eye(3)[c]
        ax.quiver(0, 0, 0, *e, color=MUTED, linewidth=1, arrow_length_ratio=0.08, alpha=0.6)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 15))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GRID,
                      linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    style_3d(ax, "x₁", "x₂", "x₃")
    for set_ticks in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        set_ticks([-1, 0, 1])
    ax.view_init(elev=22, azim=35)
    ax.set_title("K's columns: unit length, mutually ⊥\n(gray: columns of I₃)", fontsize=12)

    panel(fig.add_subplot(gs[1, 2]), "Matrix properties in one line each", [
        r"Frobenius: $\|X\|_F$ of [[1, 2], [3, 4]] = √30 = 5.477",
        "Matrix · vector: [[3, 4], [5, 6], [7, 8]]·(1, 2)",
        "     = 1·(3, 5, 7) + 2·(4, 6, 8) = (11, 17, 23)",
        "AB ≠ BA in general (here BA isn't even defined)",
        "Symmetric: X_symᵀ = X_sym.   Identity: I x = x",
        "",
        "X⁻¹ = [[1.5, 1], [−2.5, −2]],  X⁻¹X = I",
        "w = X⁻¹(4, −7) = (−1, 4),  check X w = (4, −7)",
        "Inverse needs square and full rank; N's",
        "columns are parallel → \"singular matrix\".",
    ], spacing=0.083, fontsize=11.8)
    save(fig, name="1-intro-to-linear-algebra_seg3.png")


# ================================================================ notebook 2, segment 1
def sheet_nb2_seg1_review():
    x = np.array([25, 2, 5])
    assert x.shape == (3,)
    x2 = np.array([[25, 2, 5]])
    assert x2.shape == (1, 3) and x2.T.shape == (3, 1)
    x_p = torch.tensor([25, 2, 5])
    assert tuple(x_p.T.shape) == (3,) and tuple(x_p.view(3, 1).shape) == (3, 1)
    assert np.linalg.norm(x2) == 25.573423705088842
    assert abs(torch.norm(torch.tensor([25, 2, 5.])).item() - 25.5734) < 1e-4

    X = np.array([[25, 2], [5, 26], [3, 7]])
    assert (X * 3 + 3).tolist() == [[78, 9], [18, 81], [12, 24]]
    A = np.array([[3, 4], [5, 6], [7, 8]])
    XA = X * A
    assert XA.tolist() == [[75, 8], [25, 156], [21, 56]]
    b = np.array([1, 2])
    assert (A @ b).tolist() == [11, 17, 23]
    B = np.array([[1, 9], [2, 0]])
    AB = A @ B
    assert AB.tolist() == [[11, 27], [17, 45], [23, 63]] and (AB[:, 0] == A @ b).all()

    Xi = np.array([[4, 2], [-5, -3]])
    y = np.array([4, -7])
    w = np.linalg.inv(Xi) @ y
    assert np.allclose(w, [-1, 4])
    w_p = torch.matmul(torch.inverse(torch.tensor([[4, 2], [-5, -3.]])), torch.tensor([4, -7.]))
    assert np.allclose(w_p.numpy(), [-1, 4], atol=1e-5)

    fig = new_figure("Review: shape decides which product you get",
                     "Same shape → element-wise (A ⊙ X).   Inner sizes match → matrix product (m×n)(n×p) → m×p.   "
                     "A square, invertible X gives  w = X⁻¹y.")
    gs = grid(fig, top=0.80, wspace=0.3, hspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("One vector, three shapes")
    boxes(ax, 0.3, 5.0, 1, 3, [[25, 2, 5]], size=0.75, fontsize=11)
    ax.text(1.42, 3.9, "np.array([25, 2, 5])\nshape (3,): .T does nothing", ha="center",
            va="top", fontsize=10)
    boxes(ax, 4.6, 5.0, 1, 3, [[25, 2, 5]], size=0.75, color="#cde2fb", fontsize=11)
    ax.text(5.72, 3.9, "np.array([[25, 2, 5]])\nrow, shape (1, 3)", ha="center", va="top",
            fontsize=10)
    ax.annotate("", xy=(8.1, 4.6), xytext=(7.0, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.8))
    ax.text(7.55, 4.75, ".T", ha="center", fontsize=11, color=INK2)
    boxes(ax, 8.4, 5.6, 3, 1, [[25], [2], [5]], size=0.75, color=BLUE_LIGHT, fontsize=11)
    ax.text(8.78, 3.1, "column (3, 1)", ha="center", va="top", fontsize=10)
    ax.text(5.0, 1.6, "PyTorch: x_p.T is still (3,);  x_p.view(3, 1) gives the column\n"
            "without copying memory.   ‖x‖₂ = √(625 + 4 + 25) = 25.57",
            ha="center", va="center", fontsize=10.5, color=INK2)

    sub_mats(fig, gs[0, 1], [
        (X, "X", dict(labels=labels_of(X, integer), vmax=160, fontsize=10.5)),
        (A, "A", dict(labels=labels_of(A, integer), vmax=160, fontsize=10.5)),
        (XA, "X ⊙ A", dict(labels=labels_of(XA, integer), vmax=160, fontsize=10.5,
                           highlight_cells=[(1, 1)])),
    ], ops=["⊙", "="], wspace=0.5, title="X * A: element-wise (3×2 and 3×2)", title_dy=0.045)

    sub_mats(fig, gs[0, 2], [
        (A, "A  (3×2)", dict(labels=labels_of(A, integer), vmax=70, fontsize=10.5)),
        (B, "B  (2×2)", dict(labels=labels_of(B, integer), vmax=70, fontsize=10.5,
                             highlight_cells=[(0, 0), (1, 0)], box_color=AQUA)),
        (AB, "AB  (3×2)", dict(labels=labels_of(AB, integer), vmax=70, fontsize=10.5,
                               highlight_cells=[(0, 0), (1, 0), (2, 0)])),
    ], ops=["·", "="], wspace=0.5, title="np.dot(A, B): first column = A b", title_dy=0.045)

    ax = fig.add_subplot(gs[1, 0])
    c1, c2 = Xi[:, 0], Xi[:, 1]
    arrow(ax, w[0] * c1, BLUE, "−1 · col 1 = (−4, 5)", offset=(4, 6), ha="left")
    arrow(ax, w[1] * c2, AQUA, width=2.2, start=w[0] * c1)
    ax.text(1.2, -0.6, "+ 4 · col 2\n= (8, −12)", color=AQUA, fontsize=10.5, weight="bold")
    arrow(ax, y, ORANGE, "y = Xw = (4, −7)", offset=(-8, -16), width=2.6, ha="center")
    arrow(ax, w, INK, "w = (−1, 4)", offset=(6, -4), width=2.0, ha="left")
    plane(ax, "Xw = y: w says how much of each column", (-7, 9), (-9.5, 7),
          xlabel="x₁", ylabel="x₂")
    ax.text(-6.7, -9.2, "X's columns:\n(4, −5), (2, −3)", ha="left", va="bottom", fontsize=10,
            color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    entries = [("np.linalg.inv(X)", "torch.inverse(X_p)", "needs float"),
               ("np.dot(A, b)", "torch.matmul(A_p, b_p)", "mat · vec"),
               ("X * A", "X_p * A_p", "Hadamard"),
               ("np.linalg.norm(x)", "torch.norm(x_p)", "needs float"),
               ("x.T  (2-D only)", "x_p.view(3, 1)", "column vector")]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("NumPy ↔ PyTorch")
    ax.text(0.0, 0.93, "NumPy", weight="bold", fontsize=11.5)
    ax.text(0.37, 0.93, "PyTorch", weight="bold", fontsize=11.5)
    ax.text(0.83, 0.93, "note", weight="bold", fontsize=11.5)
    for k, (a, t, n) in enumerate(entries):
        yy = 0.8 - k * 0.15
        ax.add_patch(Rectangle((-0.02, yy - 0.055), 1.04, 0.12, facecolor=GRID if k % 2 == 0 else SURFACE,
                               edgecolor="none", transform=ax.transAxes))
        ax.text(0.0, yy, a, family="monospace", fontsize=9.5, va="center")
        ax.text(0.37, yy, t, family="monospace", fontsize=9.5, va="center")
        ax.text(0.83, yy, n, fontsize=9.5, va="center", color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "The three products, side by side", [
        "Scalar:    3X + 3 → every cell, shape kept",
        "             [[78, 9], [18, 81], [12, 24]]",
        "Hadamard: X ⊙ A, same shapes, cell by cell",
        "Matrix:    (3×2)(2×2) → 3×2, row · column",
        "",
        "Inversion solves a square system:",
        "X = [[4, 2], [−5, −3]],  X⁻¹ = [[1.5, 1], [−2.5, −2]]",
        "w = X⁻¹(4, −7) = (−1, 4), and Xw = (4, −7)",
        "(same answer in NumPy and in PyTorch)",
    ], spacing=0.087, fontsize=11.8)
    save(fig, name="2-linear-algebra-ii_seg1.png")


# ================================================================ notebook 2, segment 2
def sheet_nb2_seg2_eigendecomposition():
    v = np.array([3, 1])
    E, F = np.array([[1, 0], [0, -1]]), np.array([[-1, 0], [0, 1]])
    assert (E @ v).tolist() == [3, -1] and (F @ v).tolist() == [-3, 1]
    A = np.array([[-1, 4], [2, -2]])
    assert (A @ v).tolist() == [1, 4]
    V4 = np.array([[3, 2, -3, -1], [1, 1, -1, 1]])
    AV = A @ V4
    assert AV.tolist() == [[1, 2, -1, 5], [4, 2, -4, -4]]

    lambdas, V = np.linalg.eig(A)
    assert np.allclose(lambdas, [1.37228132, -4.37228132])
    assert np.allclose(V, [[0.86011126, -0.76454754], [0.51010647, 0.64456735]])
    for k in range(2):
        assert np.allclose(A @ V[:, k], lambdas[k] * V[:, k])
    assert np.allclose(A @ V[:, 0], [1.18031462, 0.70000958])
    assert np.allclose(A @ V[:, 1], [3.34281692, -2.81822977])
    lam_p, _ = torch.linalg.eig(torch.tensor([[-1, 4], [2, -2.]]))
    assert np.allclose(lam_p.real.numpy(), lambdas, atol=1e-5)

    X3 = np.array([[25, 2, 9], [5, 26, -5], [3, 7, -1]])
    lam_X3, _ = np.linalg.eig(X3)
    assert np.allclose(lam_X3, [29.67623202, 20.62117365, -0.29740567])

    assert abs(np.linalg.det(np.array([[4, 2], [-5, -3]])) + 2) < 1e-12
    Xd = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
    det_Xd = np.linalg.det(Xd)
    lam_Xd = np.linalg.eig(Xd)[0]
    assert abs(det_Xd - 20) < 1e-12 and np.allclose(lam_Xd, [-3.25599251, -1.13863631, 5.39462882])
    assert abs(np.prod(lam_Xd) - 20) < 1e-12

    N = np.array([[-4, 1], [-8, 2]], float)
    I = np.eye(2)
    J = np.array([[-0.5, 0], [0, 2]])
    D = I * 2
    maps = [("N", N, 0.0, [-2, 0]), ("I", I, 1.0, [1, 1]), ("J", J, -1.0, [-0.5, 2]),
            ("D", D, 4.0, [2, 2])]
    for _, M, det, lam in maps:
        assert abs(np.linalg.det(M) - det) < 1e-12 and np.allclose(np.linalg.eig(M)[0], lam)

    A2 = np.array([[4, 2], [-5, -3]])
    lam2, V2 = np.linalg.eig(A2)
    assert np.allclose(lam2, [2, -1]) and np.allclose(V2, [[0.70710678, -0.37139068],
                                                           [-0.70710678, 0.92847669]])
    V2inv = np.linalg.inv(V2)
    assert np.allclose(V2inv, [[2.3570226, 0.94280904], [1.79505494, 1.79505494]])
    assert np.allclose(V2 @ np.diag(lam2) @ V2inv, A2)

    S = np.array([[2, 1], [1, 2]])
    lamS, Q = np.linalg.eig(S)
    assert np.allclose(lamS, [3, 1]) and np.allclose(Q, [[0.70710678, -0.70710678],
                                                         [0.70710678, 0.70710678]])
    assert np.allclose(Q @ np.diag(lamS) @ Q.T, S) and np.allclose(Q.T @ Q, I)

    fig = new_figure("Eigenvectors keep their direction:  Av = λv",
                     "A matrix moves most vectors off their line; eigenvectors only stretch by λ.   "
                     "det(A) = product of the λ = signed area scale.   A = VΛV⁻¹  (symmetric: A = QΛQᵀ).")
    gs = grid(fig, top=0.80, wspace=0.3, hspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    light = [BLUE_LIGHT, AQUA_LIGHT, GRID, ORANGE_LIGHT]
    dark = [BLUE, AQUA, MUTED, ORANGE]
    for k in range(4):
        arrow(ax, V4[:, k], light[k], width=2.0)
        arrow(ax, AV[:, k], dark[k], width=2.2)
    ax.annotate("v = (3, 1)", V4[:, 0], xytext=(4, -12), textcoords="offset points",
                color=BLUE, fontsize=10)
    ax.annotate("Av = (1, 4)", AV[:, 0], xytext=(4, 2), textcoords="offset points", color=BLUE,
                fontsize=10, weight="bold")
    ax.annotate("(−1, 1) → (5, −4)", AV[:, 3], xytext=(-6, -14), textcoords="offset points",
                color=ORANGE, fontsize=10, weight="bold", ha="right")
    plane(ax, "A = [[−1, 4], [2, −2]] applied to V's columns", (-4, 6), (-5, 5),
          xlabel="x₁", ylabel="x₂")
    ax.plot([], [], color=MUTED, alpha=0.5, linewidth=2, label="light: V")
    ax.plot([], [], color=INK, linewidth=2, label="dark: AV")
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    for k, (light_c, dark_c) in enumerate([(BLUE_LIGHT, BLUE), (AQUA_LIGHT, AQUA)]):
        line = np.outer(np.array([-5, 5]), V[:, k])
        ax.plot(line[:, 0], line[:, 1], color=dark_c, linewidth=0.8, linestyle=":")
        arrow(ax, A @ V[:, k], dark_c, width=2.4)
        arrow(ax, V[:, k], light_c, width=3.2)
    ax.annotate(f"v₁, Av₁ = {lambdas[0]:.2f} v₁", A @ V[:, 0], xytext=(-2, 8),
                textcoords="offset points", color=BLUE, fontsize=10.5, weight="bold", ha="center")
    ax.annotate(f"Av₂ = {lambdas[1]:.2f} v₂\n(flipped)", A @ V[:, 1], xytext=(-6, -6),
                textcoords="offset points", color=AQUA, fontsize=10.5, weight="bold", ha="right",
                va="top")
    ax.annotate("v₂", V[:, 1], xytext=(-6, 6), textcoords="offset points", color=AQUA,
                fontsize=10.5, weight="bold", ha="right")
    plane(ax, "Eigenvectors stay on their own line", (-2, 4), (-3.4, 2), xlabel="x₁", ylabel="x₂")

    sub = gs[0, 2].subgridspec(2, 2, hspace=0.55, wspace=0.35)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    for k, (name, M, det, lam) in enumerate(maps):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        ax.add_patch(Polygon(square, closed=True, facecolor=GRID, edgecolor=MUTED, linewidth=1))
        image = square @ M.T
        if abs(det) < 1e-12:
            ax.plot(image[:, 0], image[:, 1], color=ORANGE, linewidth=3, solid_capstyle="round")
        else:
            ax.add_patch(Polygon(image, closed=True, facecolor=BLUE_LIGHT, edgecolor=BLUE,
                                 linewidth=1.5, alpha=0.8))
        lim = {"N": (-6, 2, -9, 3), "I": (-1, 3, -1, 3), "J": (-1, 3, -1, 3), "D": (-1, 3, -1, 3)}[name]
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.tick_params(labelsize=7.5)
        ax.set_title(f"{name}: det {det:g},  λ = {lam[0]:g}, {lam[1]:g}", fontsize=10, pad=4)
    first = fig.axes[-4].get_position()
    fig.text((first.x0 + fig.axes[-3].get_position().x1) / 2, first.y1 + 0.035,
             "|det| = how much area scales", ha="center", va="bottom", fontsize=13,
             weight="bold")

    ax = fig.add_subplot(gs[1, 0])
    labels = ["λ₁", "λ₂", "λ₃", "λ₁λ₂λ₃", "det(X)"]
    vals = list(lam_Xd) + [np.prod(lam_Xd), det_Xd]
    colors = [BLUE, BLUE, BLUE, ORANGE, AQUA]
    pos = np.arange(5)
    ax.bar(pos, vals, width=0.6, color=colors, zorder=2)
    for p, val in zip(pos, vals):
        ax.text(p, val + (0.6 if val >= 0 else -0.6), f"{val:.3f}" if p < 3 else f"{val:.0f}",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=10.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(pos, labels)
    ax.set_ylim(-6, 24)
    ax.grid(axis="y", zorder=0)
    ax.set_title("X = [[1, 2, 4], [2, −1, 3], [0, 5, 1]]: det = Πλ")
    ax.text(1.0, 15, "any λ = 0 ⇒ det = 0 ⇒\nno inverse (like N)", ha="center", fontsize=10,
            color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    ellipse = S @ UNIT_CIRCLE
    ax.plot(*UNIT_CIRCLE, color=MUTED, linestyle="--", linewidth=1.5, label="unit circle")
    ax.plot(*ellipse, color=BLUE, linewidth=2.2, label="S · circle")
    q1, q2 = Q[:, 0], Q[:, 1]
    arrow(ax, q1, ORANGE_LIGHT, width=3)
    arrow(ax, 3 * q1, ORANGE, "3 q₁", offset=(4, 2))
    arrow(ax, 1 * q2, AQUA, "1 · q₂", offset=(-6, 4), ha="right")
    plane(ax, "Symmetric S = [[2, 1], [1, 2]] = QΛQᵀ", (-3.3, 3.3), (-3.3, 3.3),
          xlabel="x₁", ylabel="x₂")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.text(-3.1, 2.9, "λ = 3, 1; Q's columns ⊥:\nthe ellipse's axes", fontsize=10, color=INK2,
            va="top")

    panel(fig.add_subplot(gs[1, 2]), "Eigendecomposition  A = VΛV⁻¹", [
        "A = [[4, 2], [−5, −3]]:  λ = 2, −1",
        "`V   = [[ 0.707, -0.371], [-0.707, 0.928]]",
        "`V⁻¹ = [[ 2.357,  0.943], [ 1.795, 1.795]]",
        "V · diag(2, −1) · V⁻¹ gives back A exactly.",
        "",
        "3 × 3: [[25, 2, 9], [5, 26, −5], [3, 7, −1]] has",
        "λ = 29.68, 20.62, −0.297 (one per column of V).",
        "Real symmetric matrices (common in ML):",
        "real λ, orthogonal Q, so  Q⁻¹ = Qᵀ.",
    ], spacing=0.087, fontsize=11.8)
    save(fig, name="2-linear-algebra-ii_seg2.png")


# ================================================================ notebook 2, segment 3
def sheet_nb2_seg3_svd_pinv_pca():
    A = np.array([[-1, 2], [3, -2], [5, 7]])
    U, d, VT = np.linalg.svd(A)
    assert np.allclose(d, [8.66918448, 4.10429538])
    assert np.allclose(U, [[0.12708324, 0.47409506, 0.87125411], [0.00164602, -0.87847553, 0.47778451],
                           [0.99189069, -0.0592843, -0.11241989]])
    assert np.allclose(VT, [[0.55798885, 0.82984845], [-0.82984845, 0.55798885]])
    D = np.concatenate((np.diag(d), [[0, 0]]), axis=0)
    assert np.allclose(U @ D @ VT, A)
    assert np.allclose(np.sort(np.linalg.eigvalsh(A.T @ A))[::-1], d ** 2)
    Dplus = np.concatenate((np.linalg.inv(np.diag(d)), np.array([[0, 0]]).T), axis=1)
    A_plus = VT.T @ Dplus @ U.T
    assert np.allclose(A_plus, [[-0.08767773, 0.17772512, 0.07582938],
                                [0.07661927, -0.1192733, 0.08688784]])
    assert np.allclose(A_plus, np.linalg.pinv(A))
    assert np.allclose(torch.pinverse(torch.tensor([[-1, 2], [3, -2], [5, 7.]])).numpy(), A_plus,
                       atol=1e-6)

    # Image compression: the notebook's photo (4032 x 3024) is downloaded, so its numbers are quoted;
    # the same code runs here on the drawing that is on disk, img/Oboe.jpg.
    full_representation = 4032 * 3024
    svd64_rep = 64 * 4032 + 64 + 64 * 3024
    assert full_representation == 12192768 and svd64_rep == 451648
    assert abs(svd64_rep / full_representation - 0.037042286050222556) < 1e-15
    img = Image.open(REPO / "img" / "Oboe.jpg")
    imggray = img.convert("LA")
    imgmat = np.array(list(imggray.getdata(band=0)), float)
    imgmat.shape = (imggray.size[1], imggray.size[0])
    rows, cols = imgmat.shape
    Ui, sigma, Vi = np.linalg.svd(imgmat, full_matrices=False)

    def reconstruct(k):
        return Ui[:, :k] @ np.diag(sigma[:k]) @ Vi[:k, :]

    ks_all = np.arange(1, 129)
    errors = [np.linalg.norm(imgmat - reconstruct(k)) / np.linalg.norm(imgmat) for k in ks_all]
    storage = ks_all * (rows + cols + 1) / (rows * cols)
    photo_storage = ks_all * (4032 + 3024 + 1) / full_representation

    # regression with the pseudoinverse
    x1 = np.array([0, 1, 2, 3, 4, 5, 6, 7.])
    y = np.array([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])
    X = np.column_stack([np.ones(8), x1])
    w = np.linalg.pinv(X) @ y
    b, m = w
    assert np.allclose(w, [1.76, -0.46928571]) and abs(b - 1.7599999999999985) < 1e-12

    # trace
    At = np.array([[25, 2], [5, 4]])
    assert np.trace(At) == 29
    A_p = A.astype(float)
    frob_trace = np.sqrt(np.trace(A_p @ A_p.T))
    assert abs(frob_trace - np.linalg.norm(A_p)) < 1e-12 and abs(frob_trace ** 2 - 92) < 1e-9

    # PCA on iris (bundled with sklearn)
    iris = datasets.load_iris()
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(iris.data)
    assert np.allclose(Xp[0:6], [[-2.68412563, 0.31939725], [-2.71414169, -0.17700123],
                                 [-2.88899057, -0.14494943], [-2.74534286, -0.31829898],
                                 [-2.72871654, 0.32675451], [-2.28085963, 0.74133045]])
    ratio = pca.explained_variance_ratio_
    centred = iris.data - iris.data.mean(axis=0)
    _, s_iris, vt_iris = np.linalg.svd(centred, full_matrices=False)
    assert np.allclose(np.abs(centred @ vt_iris[:2].T), np.abs(Xp))   # PCA = SVD of centred data

    fig = new_figure("SVD:  A = UDVᵀ  drives compression, the pseudoinverse and PCA",
                     "Keep the top n singular values for a low-rank image;  A⁺ = VD⁺Uᵀ solves "
                     "overdetermined regression (w = X⁺y);  PCA projects centred data onto the top "
                     "right-singular vectors.")
    gs = grid(fig, top=0.80, wspace=0.3, hspace=0.5)

    shown = [1, 8, 32, 64]
    sub = gs[0, 0:2].subgridspec(1, len(shown) + 1, wspace=0.08)
    for k_i, k in enumerate(shown + [None]):
        ax = fig.add_subplot(sub[0, k_i])
        picture = imgmat if k is None else reconstruct(k)
        ax.imshow(picture, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if k is None:
            ax.set_title(f"original\n{rows} × {cols}", fontsize=11)
        else:
            ax.set_title(f"n = {k}\n{100 * storage[k - 1]:.1f}% of the numbers", fontsize=11)
    left = fig.axes[-5].get_position()
    right = fig.axes[-1].get_position()
    fig.text((left.x0 + right.x1) / 2, left.y1 + 0.055,
             "U[:, :n] diag(σ[:n]) V[:n, :] on the course's drawing img/Oboe.jpg",
             ha="center", va="bottom", fontsize=13, weight="bold")
    fig.text((left.x0 + right.x1) / 2, left.y0 - 0.015,
             "(stand-in: the notebook's 4032 × 3024 photo is downloaded with wget, not on disk)",
             ha="center", va="top", fontsize=9.5, color=INK2)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ks_all, errors, color=BLUE, linewidth=2.2, label=r"relative error $\|M - M_n\|_F\, /\, \|M\|_F$")
    ax.plot(ks_all, storage, color=ORANGE, linewidth=2.2, label=f"storage, drawing {rows}×{cols}")
    ax.plot(ks_all, photo_storage, color=ORANGE, linestyle="--", linewidth=1.6,
            label="storage, notebook's photo 4032×3024")
    ax.plot(64, photo_storage[63], "o", color=ORANGE, markersize=7, zorder=5)
    ax.annotate(f"photo, n = 64: {svd64_rep:,}\nof {full_representation:,} = 3.7%",
                (64, photo_storage[63]), xytext=(66, 0.36), textcoords="data", fontsize=9.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("singular vectors kept, n")
    ax.set_ylabel("fraction")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8.8)
    ax.set_title("Few singular values, most of the picture")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(x1, y, color=BLUE, s=40, zorder=3, label="8 patients")
    xs = np.array([-0.4, 7.4])
    ax.plot(xs, m * xs + b, color=ORANGE, linewidth=2.2,
            label=f"y = {m:.4f} x + {b:.2f}")
    for xi, yi in zip(x1, y):
        ax.plot([xi, xi], [yi, m * xi + b], color=MUTED, linewidth=1, zorder=2)
    ax.set_xlabel("Drug dosage (mL)")
    ax.set_ylabel("Forgetfulness")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Clinical trial: w = X⁺y with X = [1, x₁]")

    ax = fig.add_subplot(gs[1, 1])
    for t, color in zip(range(3), [BLUE, ORANGE, AQUA]):
        sel = iris.target == t
        ax.scatter(Xp[sel, 0], Xp[sel, 1], s=16, color=color, label=iris.target_names[t], alpha=0.85)
    ax.set_xlabel(f"PC 1  ({100 * ratio[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC 2  ({100 * ratio[1]:.1f}%)")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5, ncol=3)
    ax.set_ylim(-1.6, 1.9)
    ax.set_title("PCA: iris 150 × 4 → 150 × 2")

    panel(fig.add_subplot(gs[1, 2]), "The numbers behind the plots", [
        "A = [[−1, 2], [3, −2], [5, 7]]:  d = 8.669, 4.104",
        "d² = eigenvalues of AᵀA (and nonzero of AAᵀ)",
        "D⁺: invert nonzero d, transpose to 2 × 3:",
        "`A⁺ = [[-0.0877,  0.1777, 0.0758],",
        "`      [ 0.0766, -0.1193, 0.0869]]",
        "= np.linalg.pinv(A) = torch.pinverse(A_p)",
        f"Regression: b = {b:.2f}, m = {m:.4f}",
        "",
        "Trace: Tr([[25, 2], [5, 4]]) = 29;",
        r"$\|A\|_F = \sqrt{\mathrm{Tr}(AA^T)}$ = √92 = 9.592  (A = A_p above)",
    ], spacing=0.083, fontsize=11.6)
    save(fig, name="2-linear-algebra-ii_seg3.png")


if __name__ == "__main__":
    sheet_nb1_seg1_data_structures()
    sheet_nb1_seg2_tensor_operations()
    sheet_nb1_seg3_matrix_properties()
    sheet_nb2_seg1_review()
    sheet_nb2_seg2_eigendecomposition()
    sheet_nb2_seg3_svd_pinv_pca()
