#!/usr/bin/env python3
"""Concept sheets for the seven short standalone notebooks in notebooks/.

The regression notebooks build on each other, so their sheets are complementary rather than repetitive:

  regression-in-pytorch             the four-step loop, 1001 steps to the least-squares line
  single-point-regression-gradient  the chain rule on one data point (why one point cannot pin down a line)
  batch-regression-gradient         the batch gradient as an average of per-point gradients, first 9 steps
  gradient-descent-from-scratch     the update θ' = θ − α∇C by hand; curvature sets speed and stability
  SGD-from-scratch                  mini-batch gradients: noisy but unbiased, 32 of 8 million points
  learning-rate-scheduling          lr decay, DataLoader batching (one iterator, a fresh batch each round)
  artificial-neurons                one dense layer a = σ(xW + b) with Xavier initialization

Every number is recomputed here with the notebooks' own data, seeds, parameters and tensor operations
(torch on CPU) and checked against their printed outputs with asserts. Illustrations the notebooks do
not contain (gradient clouds, learning-rate sweeps, a loader-iterating run) are labeled as such on the
sheets and use their own seeds, after the notebooks' random streams have been reproduced.

Images are written to notebooks/img/<notebook stem>.png. Needs torch and scikit-learn; about one minute.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_standalone_concept_images.py
"""
import warnings
from functools import partial

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, GRID, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel)
from viz_style import save as _save

warnings.filterwarnings("ignore", category=UserWarning)  # the notebooks call .item() on grad tensors

NOTEBOOKS = REPO / "notebooks"
save = partial(_save, folder=NOTEBOOKS / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"

# the eight points shared by regression-in-pytorch, single-point, batch and gradient-descent notebooks
XS = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
YS = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])


# ================================================================ helpers
def regression(my_x, my_m, my_b):
    return my_m * my_x + my_b


def mse(my_yhat, my_y):
    sigma = torch.sum((my_yhat - my_y) ** 2)
    return sigma / len(my_y)


def g3(value):
    """The notebooks' '%.3g' formatting."""
    return "%.3g" % value


def arrow(ax, start, end, color, width=2.2, zorder=4, label=None, scale=16):
    ax.annotate("", xy=end, xytext=start, zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, mutation_scale=scale,
                                shrinkA=0, shrinkB=0))
    if label is not None:
        ax.plot([], [], color=color, linewidth=width, label=label)


def stack_diagram(ax, rows, title, tag_color=ORANGE, detail_size=9.5):
    """Vertical flow of rounded boxes: (name, detail, tag, fill) per row, arrows between."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    n = len(rows)
    h = 1.0 / n
    for i, (name, detail, tag, fill) in enumerate(rows):
        y = 1 - (i + 0.5) * h
        ax.add_patch(FancyBboxPatch((0.0, y - 0.34 * h), 0.78, 0.68 * h,
                                    boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=fill, edgecolor="none"))
        ax.text(0.02, y, name, va="center", fontsize=10.5, weight="bold", color=INK)
        ax.text(0.76, y, detail, va="center", ha="right", fontsize=detail_size, color=INK2, family="monospace")
        if tag:
            ax.text(0.81, y, tag, va="center", ha="left", fontsize=10, color=tag_color, weight="bold")
        if i < n - 1:
            ax.annotate("", xy=(0.38, y - 0.66 * h + 0.002), xytext=(0.38, y - 0.34 * h),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.1, mutation_scale=9))


def mse_surface(stats, b_grid, m_grid):
    """Mean squared error of y ≈ m x + b from the moments (n, Σx, Σx², Σy, Σxy, Σy²) of the data."""
    n, sx, sxx, sy, sxy, syy = stats
    B, M = np.meshgrid(b_grid, m_grid)
    C = (M ** 2 * sxx + 2 * M * B * sx + n * B ** 2 - 2 * M * sxy - 2 * B * sy + syy) / n
    return B, M, C


def moments(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (len(x), x.sum(), (x * x).sum(), y.sum(), (x * y).sum(), (y * y).sum())


def least_squares(x, y):
    """(b, m) minimizing the MSE: the pseudoinverse solution of Linear Algebra II."""
    x = np.asarray(x, dtype=np.float64)
    X = np.c_[np.ones_like(x), x]
    return np.linalg.pinv(X) @ np.asarray(y, dtype=np.float64)


def cost_contours(ax, stats, b_range, m_range, levels, title):
    B, M, C = mse_surface(stats, np.linspace(*b_range, 300), np.linspace(*m_range, 300))
    ax.contourf(B, M, C, levels=levels, cmap="Blues", alpha=0.35)
    ax.contour(B, M, C, levels=levels, colors=MUTED, linewidths=0.6, alpha=0.7)
    ax.set_xlim(*b_range)
    ax.set_ylim(*m_range)
    ax.set_xlabel("intercept b")
    ax.set_ylabel("slope m")
    ax.set_title(title)


def line_through(ax, m, b, xlim, **kw):
    xx = np.array(xlim, dtype=float)
    ax.plot(xx, m * xx + b, **kw)


# ================================================================ shared runs
def run_pytorch_sgd():
    """regression-in-pytorch: one step, then 1000 epochs of torch.optim.SGD(lr=0.01).

    Returns parameters θ_k = (b_k, m_k) for k = 0..1001 and, for k = 0..1000, the cost and gradients
    evaluated at θ_k (the notebook's 'Epoch e' line reports k = e + 1)."""
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()
    optimizer = torch.optim.SGD([m, b], lr=0.01)
    thetas, costs, m_grads, b_grads = [(b.item(), m.item())], [], [], []
    for k in range(1001):
        optimizer.zero_grad()
        C = mse(regression(XS, m, b), YS)
        C.backward()
        costs.append(C.item())
        m_grads.append(m.grad.item())
        b_grads.append(b.grad.item())
        optimizer.step()
        thetas.append((b.item(), m.item()))
    final_cost = mse(regression(XS, m, b), YS).item()
    return dict(thetas=np.array(thetas), costs=np.array(costs + [final_cost]), m_grads=np.array(m_grads),
                b_grads=np.array(b_grads))


def run_by_hand():
    """gradient-descent-from-scratch: the same 1001 steps with θ' = θ − lr·∇C on fresh leaf tensors."""
    lr = 0.01
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()
    thetas, prints = [(b.item(), m.item())], []
    for k in range(1001):
        C = mse(regression(XS, m, b), YS)
        C.backward()
        prints.append((C.item(), m.grad.item(), b.grad.item()))
        gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
        theta = torch.tensor([[b, m]]).T
        new_theta = theta - lr * gradient
        b = new_theta[0].requires_grad_()
        m = new_theta[1].requires_grad_()
        thetas.append((b.item(), m.item()))
    return np.array(thetas), prints, mse(regression(XS, m, b), YS).item()


_CACHE = {}


def pytorch_run():
    if "sgd" not in _CACHE:
        r = run_pytorch_sgd()
        # regression-in-pytorch's printed values
        assert f"{r['costs'][0]:.4f}" == "19.6755"
        assert f"{r['m_grads'][0]:.4f}" == "36.3050" and f"{r['b_grads'][0]:.4f}" == "6.2650"
        assert f"{r['thetas'][1][1]:.4f}" == "0.5369" and f"{r['thetas'][1][0]:.4f}" == "0.0374"
        assert f"{r['costs'][1]:.4f}" == "8.5722"
        printed = {0: ("8.57", "23.2", "3.6"), 1: ("4.08", "14.8", "1.91"), 4: ("1.2", "3.96", "-0.28"),
                   8: ("0.962", "0.82", "-0.894"), 999: ("0.0195", "0.000673", "-0.00331")}
        for epoch, values in printed.items():
            k = epoch + 1
            assert (g3(r["costs"][k]), g3(r["m_grads"][k]), g3(r["b_grads"][k])) == values, epoch
        assert f"{r['thetas'][-1][1]:.6f}" == "-0.468126" and f"{r['thetas'][-1][0]:.6f}" == "1.754296"
        _CACHE["sgd"] = r
    return _CACHE["sgd"]


def big_data():
    """SGD-from-scratch and learning-rate-scheduling: 8 million points from torch.manual_seed(42)."""
    if "big" not in _CACHE:
        torch.manual_seed(42)
        n = 8000000
        x = torch.linspace(0., 8., n)
        y = -0.5 * x + 2 + torch.normal(mean=torch.zeros(n), std=1)
        _CACHE["big"] = (n, x, y, moments(x.numpy(), y.numpy()))
    return _CACHE["big"]


# ================================================================ regression-in-pytorch
def sheet_regression_in_pytorch():
    r = pytorch_run()
    thetas, costs = r["thetas"], r["costs"]
    b_ols, m_ols = least_squares(XS.numpy(), YS.numpy())
    c_ols = float(np.mean((m_ols * XS.numpy() + b_ols - YS.numpy()) ** 2))
    assert f"{c_ols:.4f}" == "0.0195" and abs(m_ols - thetas[-1][1]) < 2e-3 and abs(b_ols - thetas[-1][0]) < 1e-2
    gap_b, gap_m = thetas[-1][0] - b_ols, thetas[-1][1] - m_ols
    x_np, y_np = XS.numpy(), YS.numpy()

    fig = new_figure("Machine learning in four steps: autodiff fits a line to eight points",
                     "ŷ = mx + b,  C = (1/n) Σ(ŷᵢ − yᵢ)²,  C.backward(),  optimizer.step():   "
                     f"m 0.9 → {thetas[-1][1]:.3f},  b 0.1 → {thetas[-1][0]:.3f} after 1001 steps at lr = 0.01")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    stack_diagram(ax, [
        ("1  forward pass", "yhat = m*x + b", "[0.1 … 6.4]", BLUE_LIGHT),
        ("2  cost", "C = mse(yhat, y)", "C = 19.68", BLUE_LIGHT),
        ("3  gradient", "C.backward()", "36.31, 6.27", ORANGE_LIGHT),
        ("4  descend", "optimizer.step()", "m 0.537", AQUA_LIGHT),
        ("repeat", "zero_grad(); 1000×", "C 0.0195", CARD),
    ], "One loop, first pass in numbers")

    ax = fig.add_subplot(gs[0, 1])
    xlim = (-0.4, 7.4)
    ax.scatter(x_np, y_np, s=45, color=BLUE, zorder=4, label="data (8 points)")
    line_through(ax, 0.9, 0.1, xlim, color=MUTED, linestyle=":", linewidth=1.6, label="start m = 0.9, b = 0.1")
    line_through(ax, thetas[1][1], thetas[1][0], xlim, color=ORANGE_LIGHT, linewidth=1.8,
                 label=f"after step 1: m = {thetas[1][1]:.3f}")
    line_through(ax, thetas[-1][1], thetas[-1][0], xlim, color=ORANGE, linewidth=2.4,
                 label=f"after 1001: m = {thetas[-1][1]:.3f}, b = {thetas[-1][0]:.3f}")
    line_through(ax, -0.5, 2.0, xlim, color=AQUA, linestyle="--", linewidth=1.6, label="simulated m = −0.5, b = 2")
    ax.set_xlim(*xlim)
    ax.set_ylim(-2.5, 7.0)
    ax.set_xlabel("drug dosage x (mL)")
    ax.set_ylabel("forgetfulness y")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_title("From a guess to the fitted line")

    ax = fig.add_subplot(gs[0, 2])
    steps = np.arange(len(costs))
    ax.plot(steps + 1, costs, color=BLUE, linewidth=2)
    ax.axhline(c_ols, color=AQUA, linestyle="--", linewidth=1.4, label=f"least-squares minimum {c_ols:.4f}")
    for k, txt in ((0, "19.68"), (1, "8.57"), (2, "4.08"), (10, f"{costs[10]:.3g}"), (1001, f"{costs[1001]:.3g}")):
        ax.plot(k + 1, costs[k], "o", color=ORANGE, markersize=5, zorder=4)
        ax.annotate(txt, (k + 1, costs[k]), textcoords="offset points", xytext=(6, 4), fontsize=9.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step k + 1 (log)")
    ax.set_ylabel("MSE cost C (log)")
    ax.grid(True, which="major")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Cost per step: steep at first, then a long tail")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, thetas[:, 1], color=BLUE, linewidth=2, label="slope m")
    ax.plot(steps, thetas[:, 0], color=ORANGE, linewidth=2, label="intercept b")
    ax.axhline(m_ols, color=BLUE, linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(b_ols, color=ORANGE, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(1001, m_ols - 0.12, f"least squares {m_ols:.4f}", ha="right", va="top", fontsize=9, color=BLUE)
    ax.text(1001, b_ols + 0.1, f"least squares {b_ols:.4f}", ha="right", va="bottom", fontsize=9, color=ORANGE)
    ax.set_xlabel("step")
    ax.set_ylabel("parameter value")
    ax.set_ylim(-0.8, 2.3)
    ax.grid(True)
    ax.legend(loc="center right", fontsize=9.5)
    ax.set_title("m drops at once, then both creep")

    panel(fig.add_subplot(gs[1, 1]), "After 1000 epochs (printed)", [
        "`Epoch 999, cost 0.0195, m grad 0.000673,`",
        "`           b grad -0.00331`",
        f"`m.item() -> {thetas[-1][1]:.10f}`",
        f"`b.item() -> {thetas[-1][0]:.10f}`",
        "",
        "The exact minimizer (pseudoinverse, Linear",
        f"Algebra II): m = {m_ols:.4f}, b = {b_ols:.4f}.",
        f"Gradient descent is within {abs(gap_m):.4f} and {abs(gap_b):.4f};",
        "both gradients are almost 0 there.",
    ], spacing=0.087)

    panel(fig.add_subplot(gs[1, 2]), "Why not m = −0.5, b = 2?", [
        "y was simulated as −0.5x + 2 plus noise",
        "(normal, std 0.2; commented-out cell).",
        "With 8 noisy points, the best line for this",
        f"sample is m = {m_ols:.3f}, b = {b_ols:.3f}; more data",
        "would move it closer to the truth.",
        "",
        "Why not just use algebra?  For 2 parameters",
        "and 8 points it is instant, but the same four",
        "steps scale to millions of parameters and",
        "data points; the algebra does not.",
    ], spacing=0.087)
    save(fig, name="regression-in-pytorch.png")


# ================================================================ single-point-regression-gradient
def sheet_single_point():
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()
    i = 7
    x, y = XS[i], YS[i]
    yhat = regression(x, m, b)
    C = (yhat - y) ** 2
    C.backward()
    assert f"{yhat.item():.4f}" == "6.4000" and f"{C.item():.4f}" == "60.3729"
    assert f"{m.grad.item():.4f}" == "108.7800" and f"{b.grad.item():.4f}" == "15.5400"
    dm, db = m.grad.item(), b.grad.item()
    assert f"{(2 * x * (yhat.item() - y)).item():.4f}" == "108.7800"
    assert f"{(2 * (yhat.item() - y)).item():.4f}" == "15.5400"
    xv, yv, u = x.item(), y.item(), yhat.item() - y.item()
    # one step of lr = 0.01 multiplies the residual by 1 − 2·lr·(1 + x²) = 0 for x = 7
    lr = 0.01
    factor = 1 - 2 * lr * (1 + xv ** 2)
    assert abs(factor) < 1e-12
    b1, m1 = 0.1 - lr * db, 0.9 - lr * dm
    assert abs(m1 * xv + b1 - yv) < 1e-4
    m_zero = (yv - 0.1) / xv     # C = 0 along the line b + 7m = −1.37
    b_zero = yv - 0.9 * xv

    fig = new_figure("One data point: the chain rule gives the two slopes of the cost",
                     "C = (ŷ − y)²,  ŷ = mx + b:   ∂C/∂m = 2x(ŷ − y),  ∂C/∂b = 2(ŷ − y).   "
                     f"At point i = 7 (x = 7, y = −1.37) with m = 0.9, b = 0.1:  ∇C = [∂C/∂b, ∂C/∂m]ᵀ = [{db:.2f}, {dm:.2f}]ᵀ")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    x_np, y_np = XS.numpy(), YS.numpy()
    mask = np.arange(8) != i
    ax.scatter(x_np[mask], y_np[mask], s=40, color=BLUE_LIGHT, zorder=3, label="other 7 points (unused)")
    ax.scatter([xv], [yv], s=80, color=BLUE, zorder=4, label="point i = 7: (7, −1.37)")
    line_through(ax, 0.9, 0.1, (-0.4, 7.6), color=ORANGE, linewidth=2, label="ŷ = 0.9x + 0.1")
    ax.plot([xv, xv], [yv, yhat.item()], color=INK2, linestyle="--", linewidth=1.4)
    ax.plot([xv], [yhat.item()], "o", color=ORANGE, markersize=8, zorder=4)
    ax.text(xv - 0.15, (yv + yhat.item()) / 2, f"ŷ − y = {u:.2f}\nC = {u:.2f}² = {C.item():.2f}",
            ha="right", va="center", fontsize=10)
    ax.text(xv - 0.15, yhat.item(), "ŷ = 6.4", ha="right", va="bottom", fontsize=10, color=ORANGE)
    ax.set_xlim(-0.4, 7.6)
    ax.set_ylim(-2.4, 7.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Step 1 and 2: forward pass, squared error")

    ax = fig.add_subplot(gs[0, 1])
    ms = np.linspace(-0.6, 1.2, 200)
    ax.plot(ms, (ms * xv + 0.1 - yv) ** 2, color=BLUE, linewidth=2.2, label="C(m), b fixed at 0.1")
    tm = np.linspace(0.6, 1.2, 10)
    ax.plot(tm, C.item() + dm * (tm - 0.9), color=ORANGE, linewidth=2, label=f"tangent, slope ∂C/∂m = {dm:.2f}")
    ax.plot([0.9], [C.item()], "o", color=ORANGE, markersize=7, zorder=4)
    ax.plot([m_zero], [0], "o", color=AQUA, markersize=7, zorder=4)
    ax.annotate(f"C = 0 at m = {m_zero:.3f}", (m_zero, 0), textcoords="offset points", xytext=(0, -9),
                fontsize=9.5, color=AQUA, ha="center", va="top")
    ax.set_xlabel("slope m")
    ax.set_ylabel("cost C")
    ax.set_ylim(-16, 100)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Slice through m: steep, because x = 7")

    ax = fig.add_subplot(gs[0, 2])
    bs = np.linspace(-9.5, 1.0, 200)
    ax.plot(bs, (0.9 * xv + bs - yv) ** 2, color=BLUE, linewidth=2.2, label="C(b), m fixed at 0.9")
    tb = np.linspace(-1.8, 1.0, 10)
    ax.plot(tb, C.item() + db * (tb - 0.1), color=ORANGE, linewidth=2, label=f"tangent, slope ∂C/∂b = {db:.2f}")
    ax.plot([0.1], [C.item()], "o", color=ORANGE, markersize=7, zorder=4)
    ax.plot([b_zero], [0], "o", color=AQUA, markersize=7, zorder=4)
    ax.annotate(f"C = 0 at b = {b_zero:.2f}", (b_zero, 0), textcoords="offset points", xytext=(0, -9),
                fontsize=9.5, color=AQUA, ha="center", va="top")
    ax.set_xlabel("intercept b")
    ax.set_ylabel("cost C")
    ax.set_ylim(-16, 100)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Slice through b: 7 times flatter")

    ax = fig.add_subplot(gs[1, 0])
    b_range, m_range = (-2.0, 2.0), (-0.6, 1.2)
    B, M = np.meshgrid(np.linspace(*b_range, 300), np.linspace(*m_range, 300))
    Cs = (M * xv + B - yv) ** 2
    levels = [1, 5, 15, 30, 50, 75, 100, 130]
    ax.contourf(B, M, Cs, levels=[0] + levels, cmap="Blues", alpha=0.35)
    ax.contour(B, M, Cs, levels=levels, colors=MUTED, linewidths=0.6)
    bb = np.linspace(*b_range, 10)
    ax.plot(bb, (yv - bb) / xv, color=AQUA, linewidth=2.4, label="C = 0 on the whole line 7m + b = −1.37")
    arrow(ax, (0.1, 0.9), (b1, m1), ORANGE, width=2.4, label=f"−0.01·∇C: lands on that line")
    ax.plot([0.1], [0.9], "o", color=INK, markersize=6, zorder=5)
    ax.text(0.17, 0.93, "(b, m) = (0.1, 0.9)", fontsize=9.5)
    ax.set_xlim(*b_range)
    ax.set_ylim(*m_range)
    ax.set_xlabel("intercept b")
    ax.set_ylabel("slope m")
    ax.legend(loc="lower left", fontsize=9, framealpha=1, facecolor=SURFACE, frameon=True, edgecolor="none")
    ax.set_title("Cost over (b, m): a valley, not a bowl")

    panel(fig.add_subplot(gs[1, 1]), "Chain rule, link by link", [
        "C = u²,  u = ŷ − y,  ŷ = mx + b",
        "∂C/∂u = 2u = 2 · 7.77 = 15.54",
        "∂ŷ/∂m = x = 7,   ∂ŷ/∂b = 1",
        "∂C/∂m = 2u · x = 15.54 · 7 = 108.78",
        "∂C/∂b = 2u · 1 = 15.54",
        "",
        "Same numbers from autodiff (C.backward()):",
        "`m.grad -> tensor([108.7800])`",
        "`b.grad -> tensor([15.5400])`",
        "`2*x*(yhat.item()-y) -> tensor(108.7800)`",
    ], spacing=0.083)

    panel(fig.add_subplot(gs[1, 2]), "What one point can and cannot do", [
        "∇C points along (1, x) = (1, 7): the slope m",
        "gets 7× the push of the intercept b.",
        "",
        "Every line through (7, −1.37) has C = 0, so",
        "one point cannot pin down m and b. A step",
        "with lr 0.01 even lands exactly on that line:",
        "the residual shrinks by 1 − 2·lr·(1 + x²) = 0.",
        "",
        "Next: average this over all 8 points (MSE),",
        "which turns the valley into a bowl.",
    ], spacing=0.083)
    save(fig, name="single-point-regression-gradient.png")


# ================================================================ batch-regression-gradient
def sheet_batch():
    r = pytorch_run()
    thetas, costs, m_grads, b_grads = r["thetas"], r["costs"], r["m_grads"], r["b_grads"]
    x_np, y_np = XS.numpy().astype(np.float64), YS.numpy().astype(np.float64)
    u = 0.9 * x_np + 0.1 - y_np
    n = len(x_np)
    per_m, per_b = 2 / n * u * x_np, 2 / n * u
    assert f"{per_m.sum():.3f}" == "36.305" and f"{per_b.sum():.3f}" == "6.265"
    assert abs(per_m[7] - 108.78 / 8) < 1e-5 and abs(per_b[7] - 15.54 / 8) < 1e-5
    # the notebook's labeled plots: before the step, then 8 loop epochs plotting θ_k with ∇C(θ_k), k = 1..8
    assert (g3(costs[1]), g3(thetas[1][1]), g3(m_grads[1]), g3(thetas[1][0]), g3(b_grads[1])) == \
        ("8.57", "0.537", "23.2", "0.0374", "3.6")
    assert (g3(costs[8]), g3(thetas[8][1]), g3(m_grads[8]), g3(thetas[8][0]), g3(b_grads[8])) == \
        ("0.981", "-0.0814", "1.17", "-0.0119", "-0.829")
    first_neg = int(np.argmax(b_grads < 0))
    assert first_neg == 5 and (m_grads[:9] > 0).all()
    b_ols, m_ols = least_squares(x_np, y_np)

    fig = new_figure("A batch gradient is the average of the per-point gradients",
                     "C = (1/n) Σ(ŷᵢ − yᵢ)²:   ∂C/∂m = (2/n) Σ(ŷᵢ − yᵢ)xᵢ = 36.305,   ∂C/∂b = (2/n) Σ(ŷᵢ − yᵢ) = 6.265;"
                     "   torch.optim.SGD(lr = 0.01) then steps against ∇C")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    pos = np.arange(n)
    ax.bar(pos - 0.2, per_m, width=0.38, color=BLUE, zorder=2, label=f"(2/n)(ŷᵢ − yᵢ)xᵢ  Σ = {per_m.sum():.3f}")
    ax.bar(pos + 0.2, per_b, width=0.38, color=ORANGE, zorder=2, label=f"(2/n)(ŷᵢ − yᵢ)  Σ = {per_b.sum():.3f}")
    ax.annotate("point 7: 108.78/8 = 13.60\n(single-point notebook ÷ n)", (7 - 0.2, per_m[7]),
                textcoords="offset points", xytext=(-150, -8), fontsize=9.2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xticks(pos, [str(k) for k in pos])
    ax.set_xlabel("data point i")
    ax.set_ylabel("contribution to the gradient")
    ax.set_ylim(-0.5, 17)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9.2)
    ax.set_title("∇C summed point by point (m = 0.9, b = 0.1)")

    ax = fig.add_subplot(gs[0, 1])
    stats = moments(x_np, y_np)
    cost_contours(ax, stats, (-0.4, 2.2), (-0.65, 1.0), [0.05, 0.25, 0.6, 1.2, 2.5, 5, 10, 20, 35],
                  "The MSE is a bowl; 9 steps down its steep side")
    path = thetas[:10]
    ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, markersize=4.5, linewidth=1.6, zorder=4,
            label="θ₀ … θ₉ (lr = 0.01)")
    for k in (0, 1, 2, 9):
        ax.annotate(f"θ{k}" if k else "θ₀ (0.1, 0.9)", path[k], textcoords="offset points",
                    xytext=(7, 2), fontsize=9.5)
    ax.plot([b_ols], [m_ols], "*", color=AQUA, markersize=15, zorder=5, label="minimum (b, m) = "
            f"({b_ols:.2f}, {m_ols:.2f})")
    ax.legend(loc="center right", fontsize=9, framealpha=1, facecolor=SURFACE, frameon=True, edgecolor="none")

    ax = fig.add_subplot(gs[0, 2])
    xlim = (-0.4, 7.4)
    ax.scatter(x_np, y_np, s=45, color=BLUE, zorder=4)
    shades = np.linspace(0.25, 1.0, 10)
    for k in range(10):
        color = ORANGE if k == 9 else MUTED
        ax.plot(xlim, [thetas[k][1] * xlim[0] + thetas[k][0], thetas[k][1] * xlim[1] + thetas[k][0]],
                color=color, alpha=float(shades[k]) if k < 9 else 1.0, linewidth=2.2 if k == 9 else 1.2)
        if k in (0, 1, 2, 9):
            ax.text(xlim[1] + 0.05, thetas[k][1] * xlim[1] + thetas[k][0], f"θ{k}: C = {costs[k]:.3g}",
                    fontsize=9, va="center", color=ORANGE if k == 9 else INK2)
    ax.set_xlim(xlim[0], xlim[1] + 2.6)
    ax.set_ylim(-2.2, 7.2)
    ax.set_xticks(range(8))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title("The fitted line at θ₀ … θ₉, overlaid")

    ax = fig.add_subplot(gs[1, 0])
    ks = np.arange(10)
    ax.plot(ks, m_grads[:10], "-o", color=BLUE, markersize=5, linewidth=2, label="∂C/∂m")
    ax.plot(ks, b_grads[:10], "-o", color=ORANGE, markersize=5, linewidth=2, label="∂C/∂b")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.annotate(f"∂C/∂b < 0 from θ{first_neg} on: b now rises", (first_neg, b_grads[first_neg]),
                textcoords="offset points", xytext=(18, 38), fontsize=9.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8))
    ax.set_xlabel("step k")
    ax.set_ylabel("gradient at θₖ (symlog)")
    ax.set_xticks(ks)
    ax.set_ylim(-3, 60)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Gradient components step by step")

    panel(fig.add_subplot(gs[1, 1]), "From one point to the batch", [
        "Each point contributes its own single-point",
        "gradient 2uᵢ(xᵢ, 1), weighted 1/n; points far",
        "right (large xᵢ) dominate ∂C/∂m.",
        "",
        "Each point's cost is a valley tilted by its xᵢ;",
        "adding 8 differently tilted valleys gives a bowl",
        f"with a single minimum (b, m) = ({b_ols:.3f}, {m_ols:.3f}).",
        "",
        "`gradient -> tensor([[ 6.2650], [36.3050]])`",
    ], spacing=0.087)

    panel(fig.add_subplot(gs[1, 2]), "The loop with torch.optim.SGD", [
        "`optimizer = torch.optim.SGD([m, b], lr=0.01)`",
        "`optimizer.zero_grad()  # else grads add up`",
        "`C = mse(regression(xs, m, b), ys)  # 1, 2`",
        "`C.backward()                       # 3`",
        "`optimizer.step()                   # 4`",
        "",
        "Step 1 cuts m from 0.9 to 0.537 (∂C/∂m = 36.3)",
        "and b only from 0.1 to 0.037 (∂C/∂b = 6.26).",
        f"Cost: 19.68 → 8.57 → … → {costs[9]:.3f} after 9 steps.",
    ], spacing=0.087)
    save(fig, name="batch-regression-gradient.png")


# ================================================================ gradient-descent-from-scratch
def sheet_gradient_descent():
    r = pytorch_run()
    thetas_hand, prints, final_cost = run_by_hand()
    # the 992-epoch loop prints steps 9..1000; identical to regression-in-pytorch's epochs 8..999
    assert (g3(prints[9][0]), g3(prints[9][1]), g3(prints[9][2])) == ("0.962", "0.82", "-0.894")
    assert (g3(prints[1000][0]), g3(prints[1000][1]), g3(prints[1000][2])) == ("0.0195", "0.000673", "-0.00331")
    assert (g3(prints[8][0]), g3(thetas_hand[8][1]), g3(prints[8][1])) == ("0.981", "-0.0814", "1.17")
    assert np.allclose(thetas_hand, r["thetas"], atol=2e-6)
    assert (g3(thetas_hand[-1][0]), g3(thetas_hand[-1][1])) == ("1.75", "-0.468")
    theta0 = np.array([0.1, 0.9])
    r0 = pytorch_run()
    scaled = 0.01 * torch.tensor([[float(r0["b_grads"][0]), float(r0["m_grads"][0])]]).T
    assert [f"{v:.4f}" for v in scaled.flatten().tolist()] == ["0.0626", "0.3630"]
    assert [f"{v:.4f}" for v in thetas_hand[1]] == ["0.0374", "0.5369"]

    x_np, y_np = XS.numpy().astype(np.float64), YS.numpy().astype(np.float64)
    stats = moments(x_np, y_np)
    b_ols, m_ols = least_squares(x_np, y_np)
    # curvature: C is quadratic with Hessian H = (2/n) [[n, Σx], [Σx, Σx²]]
    n = len(x_np)
    H = 2 / n * np.array([[n, x_np.sum()], [x_np.sum(), (x_np ** 2).sum()]])
    eigvals, eigvecs = np.linalg.eigh(H)
    lam_min, lam_max = eigvals
    assert np.allclose(H, [[2, 7], [7, 35]])
    lr = 0.01
    lr_max = 2 / lam_max
    offsets = thetas_hand - np.array([b_ols, m_ols])
    along = offsets @ eigvecs          # coordinates along slow (col 0) and fast (col 1) eigenvectors
    steps = np.arange(len(thetas_hand))

    # learning-rate sweep (not in the notebook): 1001 full-batch steps in float64 from (0.1, 0.9)
    def run_lr(rate, n_steps=1001):
        th = theta0.copy()
        for _ in range(n_steps):
            resid = th[1] * x_np + th[0] - y_np
            th = th - rate * np.array([2 * resid.mean(), 2 * (resid * x_np).mean()])
            if not np.isfinite(th).all() or np.abs(th).max() > 1e6:
                return np.inf
        return float(np.mean((th[1] * x_np + th[0] - y_np) ** 2))

    rates = np.linspace(0.001, 0.0575, 120)
    finals = np.array([run_lr(rr) for rr in rates])
    c_ols = float(np.mean((m_ols * x_np + b_ols - y_np) ** 2))

    fig = new_figure("Gradient descent by hand: θ′ = θ − α∇C, and curvature sets the pace",
                     "θ = [b, m]ᵀ = [0.1, 0.9]ᵀ,  α∇C = 0.01·[6.265, 36.305]ᵀ = [0.0626, 0.3630]ᵀ  →  "
                     "θ′ = [0.0374, 0.5369]ᵀ:  the same numbers torch.optim.SGD produces")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    cost_contours(ax, stats, (-0.1, 0.25), (0.45, 1.0), [4, 6, 8.5722, 11, 14, 19.6755, 25],
                  "One update, component by component")
    th1 = thetas_hand[1]
    arrow(ax, theta0, th1, ORANGE, width=2.6, label="−α∇C")
    ax.plot([theta0[0], th1[0]], [theta0[1], theta0[1]], color=INK2, linestyle=":", linewidth=1.3)
    ax.plot([th1[0], th1[0]], [theta0[1], th1[1]], color=INK2, linestyle=":", linewidth=1.3)
    ax.text((theta0[0] + th1[0]) / 2, theta0[1] + 0.012, "b: −0.0626", ha="center", va="bottom", fontsize=9.5)
    ax.text(th1[0] - 0.006, (theta0[1] + th1[1]) / 2, "m: −0.3630", ha="right", va="center", fontsize=9.5)
    ax.plot(*theta0, "o", color=INK, markersize=6, zorder=5)
    ax.plot(*th1, "o", color=ORANGE, markersize=6, zorder=5)
    ax.text(theta0[0] + 0.01, theta0[1] + 0.01, "θ: C = 19.68", fontsize=9.5)
    ax.text(th1[0] + 0.012, th1[1] - 0.012, "θ′: C = 8.57", fontsize=9.5, va="top")
    ax.legend(loc="lower right", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    cost_contours(ax, stats, (-0.3, 2.3), (-0.7, 1.0), [0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16],
                  "All 1001 steps: a fast drop, then a slow crawl")
    ax.plot(thetas_hand[:, 0], thetas_hand[:, 1], color=ORANGE, linewidth=1.5, zorder=4)
    marks = [0, 1, 2, 5, 10, 100, 300, 600, 1001]
    ax.plot(thetas_hand[marks, 0], thetas_hand[marks, 1], "o", color=ORANGE, markersize=4.5, zorder=5)
    for k in (0, 10, 100, 300, 1001):
        ax.annotate(f"{k}", thetas_hand[k], textcoords="offset points", xytext=(4, 5), fontsize=9)
    ax.plot([b_ols], [m_ols], "*", color=AQUA, markersize=15, zorder=6, label="least-squares minimum")
    slow = eigvecs[:, 0]
    tt = np.linspace(-1.5, 1.5, 10)
    ax.plot(b_ols + tt * slow[0], m_ols + tt * slow[1], color=AQUA, linestyle="--", linewidth=1.1,
            label="flat direction of the bowl")
    ax.legend(loc="upper right", fontsize=9, framealpha=1, facecolor=SURFACE, frameon=True, edgecolor="none")

    ax = fig.add_subplot(gs[0, 2])
    fast_f, slow_f = 1 - lr * lam_max, 1 - lr * lam_min
    steep = np.abs(along[:, 1])
    cut = int(np.argmax(steep < 1e-6))          # below this, float32 rounding noise
    ax.plot(steps[:cut], steep[:cut], color=BLUE, linewidth=2,
            label=f"steep direction: × |1 − αλ| = {abs(fast_f):.3f} per step")
    ax.plot(steps, np.abs(along[:, 0]), color=ORANGE, linewidth=2,
            label=f"flat direction: × {slow_f:.4f} per step")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel("step")
    ax.set_ylabel("distance to the minimum (log)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Two directions, two speeds (κ = {lam_max / lam_min:.0f})")

    ax = fig.add_subplot(gs[1, 0])
    shown = np.where(np.isfinite(finals) & (finals < 10), finals, np.nan)
    ax.plot(rates, shown, color=BLUE, linewidth=2)
    ax.axvline(lr_max, color=ORANGE, linestyle="--", linewidth=1.5)
    ax.axvspan(lr_max, rates[-1] + 0.001, color=ORANGE_LIGHT, alpha=0.5)
    ax.text(lr_max - 0.0008, 3, f"diverges for\nα > 2/λmax = {lr_max:.4f}", ha="right", fontsize=9.5,
            color=ORANGE)
    ax.axhline(c_ols, color=AQUA, linestyle=":", linewidth=1.3)
    ax.plot([lr], [final_cost], "o", color=INK, markersize=7, zorder=5)
    ax.annotate(f"notebook: α = 0.01 → C = {final_cost:.4f}", (lr, final_cost), textcoords="offset points",
                xytext=(8, 22), fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_yscale("log")
    ax.set_ylim(0.015, 10)
    ax.set_xlim(0, rates[-1] + 0.001)
    ax.set_xlabel("learning rate α")
    ax.set_ylabel("cost after 1001 steps (log)")
    ax.grid(True)
    ax.set_title("Learning-rate sweep (not in the notebook)")

    panel(fig.add_subplot(gs[1, 1]), "Why b crawls: the curvature", [
        "C(b, m) is quadratic, with Hessian",
        "H = (2/n)[[n, Σx], [Σx, Σx²]] = [[2, 7], [7, 35]].",
        f"Eigenvalues λ = {lam_max:.2f} and {lam_min:.3f}.",
        "",
        "Along each eigenvector the error shrinks by",
        f"(1 − αλ) per step: {fast_f:.3f} on the steep side,",
        f"{slow_f:.4f} on the flat one (mostly b: all x are",
        "positive, so m and b are entangled).",
        "Stable only while α < 2/λmax.",
    ], spacing=0.087)

    panel(fig.add_subplot(gs[1, 2]), "By hand = torch.optim.SGD", [
        "`gradient = torch.tensor(`",
        "`    [[b.grad.item(), m.grad.item()]]).T`",
        "`theta = torch.tensor([[b, m]]).T`",
        "`new_theta = theta - lr*gradient`",
        "`b = new_theta[0].requires_grad_()`",
        "",
        "Fresh leaf tensors each step, so no zero_grad().",
        "1 + 8 + 992 steps reproduce regression-in-",
        "pytorch exactly: cost 0.0195, m −0.468, b 1.75.",
    ], spacing=0.087)
    save(fig, name="gradient-descent-from-scratch.png")


# ================================================================ SGD-from-scratch
def run_sgd_from_scratch():
    n, x, y, stats = big_data()
    np.random.seed(42)
    indices = np.random.choice(n, size=2000, replace=False)
    batch_size, lr = 32, 0.01
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()
    batch_indices = np.random.choice(n, size=batch_size, replace=False)
    assert [f"{v:.4f}" for v in x[batch_indices][:4].tolist()] == ["7.2894", "3.4176", "4.5081", "4.0586"]
    yhat = regression(x[batch_indices], m, b)
    C = mse(yhat, y[batch_indices])
    C.backward()
    assert f"{C.item():.4f}" == "30.6121"
    assert f"{m.grad.item():.4f}" == "53.4475" and f"{b.grad.item():.4f}" == "8.5877"
    first_batch = (batch_indices.copy(), b.grad.item(), m.grad.item())

    def val_cost(m_, b_):
        return mse(regression(x[indices], m_, b_), y[indices]).item()

    thetas, batch_costs, val_costs, used = [(0.1, 0.9)], [C.item()], [val_cost(m, b)], [batch_indices]
    gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
    theta = torch.tensor([[b, m]]).T
    new_theta = theta - lr * gradient
    assert [f"{v:.4f}" for v in new_theta.flatten().tolist()] == ["0.0141", "0.3655"]
    b, m = new_theta[0], new_theta[1]
    assert f"{mse(regression(x[batch_indices], m, b), y[batch_indices]).item():.4f}" == "8.7904"
    b.requires_grad_()
    m.requires_grad_()
    thetas.append((b.item(), m.item()))
    val_costs.append(val_cost(m, b))
    printed = []
    for r in range(100):
        batch_indices = np.random.choice(n, size=batch_size, replace=False)
        used.append(batch_indices)
        C = mse(regression(x[batch_indices], m, b), y[batch_indices])
        C.backward()
        batch_costs.append(C.item())
        if r % 10 == 0:
            printed.append((g3(C.item()), g3(m.grad.item()), g3(b.grad.item())))
        gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
        theta = torch.tensor([[b, m]]).T
        new_theta = theta - lr * gradient
        b = new_theta[0].requires_grad_()
        m = new_theta[1].requires_grad_()
        thetas.append((b.item(), m.item()))
        val_costs.append(val_cost(m, b))
    assert printed[0] == ("8.31", "27", "4.04") and printed[2] == ("1.54", "-2.39", "-1.15")
    assert printed[9] == ("1.52", "1.38", "-0.266")
    # the final plots: batch cost 1.32, validation cost 1.44, m = -0.258, b = 0.702
    assert (g3(C.item()), g3(val_costs[-1]), g3(m.item()), g3(b.item())) == ("1.32", "1.44", "-0.258", "0.702")
    overlap = len(set(np.concatenate(used).tolist()) & set(indices.tolist()))
    return dict(indices=indices, thetas=np.array(thetas), batch_costs=np.array(batch_costs),
                val_costs=np.array(val_costs), first_batch=first_batch, overlap=overlap,
                n_used=len(np.concatenate(used)))


def sheet_sgd_from_scratch():
    n, x, y, stats = big_data()
    r = run_sgd_from_scratch()
    thetas, indices = r["thetas"], r["indices"]
    x_np, y_np = x.numpy(), y.numpy()
    nn_, sx, sxx, sy, sxy, syy = stats
    # full-data gradient at the start (all 8 million points)
    full_db = 2 * (0.9 * sx + 0.1 * nn_ - sy) / nn_
    full_dm = 2 * (0.9 * sxx + 0.1 * sx - sxy) / nn_
    true_val = mse(regression(x[indices], -0.5, 2.0), y[indices]).item()
    # gradient noise at the start: 400 fresh mini-batches (illustration, own seed, not in the notebook)
    rng = np.random.default_rng(0)
    cloud = []
    for _ in range(400):
        idx = rng.integers(0, n, size=32)
        resid = 0.9 * x_np[idx] + 0.1 - y_np[idx]
        cloud.append((2 * resid.mean(), 2 * (resid * x_np[idx]).mean()))
    cloud = np.array(cloud, dtype=np.float64)
    _, fb_db, fb_dm = r["first_batch"]
    b_full, m_full = least_squares(x_np, y_np)

    fig = new_figure("Stochastic gradient descent: estimate ∇C from 32 random points",
                     "Each round: sample a mini-batch of 32 of the 8,000,000 points, take one step θ′ = θ − α∇C_batch "
                     "(α = 0.01).  101 rounds touch 3,232 points (0.04 %)")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    xv, yv = x_np[indices], y_np[indices]
    ax.scatter(xv, yv, s=8, color=BLUE, alpha=0.15, zorder=2, label="2,000 validation points")
    xlim = (0, 8)
    line_through(ax, 0.9, 0.1, xlim, color=MUTED, linestyle=":", linewidth=1.8, label="start: m = 0.9, b = 0.1")
    line_through(ax, thetas[-1][1], thetas[-1][0], xlim, color=ORANGE, linewidth=2.4,
                 label=f"after 101 rounds: m = {thetas[-1][1]:.3f}, b = {thetas[-1][0]:.3f}")
    line_through(ax, -0.5, 2.0, xlim, color=AQUA, linestyle="--", linewidth=1.8, label="simulated: m = −0.5, b = 2")
    ax.set_xlim(*xlim)
    ax.set_ylim(-5.5, 8.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("y = −0.5x + 2 + N(0, 1), 8 million times")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(cloud[:, 0], cloud[:, 1], s=10, color=BLUE_LIGHT, zorder=2,
               label="400 random batches of 32 (illustration)")
    ax.plot([fb_db], [fb_dm], "o", color=ORANGE, markersize=8, zorder=4,
            label=f"notebook's first batch ({fb_db:.2f}, {fb_dm:.2f})")
    ax.plot([full_db], [full_dm], "*", color=INK, markersize=15, zorder=5,
            label=f"all 8 million points ({full_db:.2f}, {full_dm:.2f})")
    ax.plot([cloud[:, 0].mean()], [cloud[:, 1].mean()], "+", color=AQUA, markersize=16, mew=2.5, zorder=5,
            label="mean of the 400 batches")
    ax.set_xlabel("∂C/∂b")
    ax.set_ylabel("∂C/∂m")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_ylim(top=max(cloud[:, 1].max(), fb_dm) + 22)
    ax.set_title("Batch gradients at the start: noisy, unbiased")

    ax = fig.add_subplot(gs[0, 2])
    rounds = np.arange(len(r["batch_costs"]))
    ax.plot(rounds, r["batch_costs"], color=BLUE_LIGHT, linewidth=1.3, label="cost on the round's batch")
    ax.plot(np.arange(len(r["val_costs"])), r["val_costs"], color=ORANGE, linewidth=2.2,
            label=f"validation cost (2,000 points): {r['val_costs'][-1]:.2f}")
    ax.axhline(true_val, color=AQUA, linestyle="--", linewidth=1.4,
               label=f"validation cost of the true line: {true_val:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("MSE (log)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Cost per round")

    ax = fig.add_subplot(gs[1, 0])
    cost_contours(ax, stats, (-0.3, 2.4), (-0.7, 1.0), [1.05, 1.2, 1.5, 2, 3, 5, 9, 16, 28],
                  "101 noisy steps over the full-data cost")
    ax.plot(thetas[:, 0], thetas[:, 1], "-", color=ORANGE, linewidth=1.2, zorder=4)
    ax.plot(thetas[:, 0], thetas[:, 1], "o", color=ORANGE, markersize=2.5, zorder=4)
    ax.plot(*thetas[0], "o", color=INK, markersize=6, zorder=5)
    ax.annotate(f"round 101 ({thetas[-1][0]:.2f}, {thetas[-1][1]:.2f})", thetas[-1], textcoords="offset points",
                xytext=(8, 10), fontsize=9.5)
    ax.plot([b_full], [m_full], "*", color=AQUA, markersize=15, zorder=5,
            label=f"full-data minimum ({b_full:.2f}, {m_full:.2f})")
    ax.legend(loc="upper right", fontsize=9, framealpha=1, facecolor=SURFACE, frameon=True, edgecolor="none")

    panel(fig.add_subplot(gs[1, 1]), "Why it works", [
        "The batch gradient is an unbiased estimate:",
        "its average over random batches is the full",
        "gradient, for 1/250,000 of the work per step.",
        "",
        "First batch (printed):",
        "`C -> tensor(30.6121)`",
        "`gradient -> [[ 8.5877], [53.4475]]`",
        "`new_theta -> [[0.0141], [0.3655]]`",
        f"Full data at the same θ: [{full_db:.2f}, {full_dm:.2f}].",
    ], spacing=0.087)

    panel(fig.add_subplot(gs[1, 2]), "What 100 rounds buy", [
        "Printed batch cost: 8.31 at step 0, 1.52 at 90.",
        f"Validation cost {r['val_costs'][-1]:.2f}; the true line scores {true_val:.2f}.",
        "",
        "m drops within a few rounds, then m and b",
        "crawl along the same flat valley as in",
        "gradient-descent-from-scratch, now with noisy",
        f"steps: b = {thetas[-1][0]:.2f} vs 2 after 101 rounds.",
        "np.random.choice over 8 M indices per batch",
        "is the slow part (see the next notebook).",
    ], spacing=0.087)
    print(f"  full-data gradient at start: {full_db:.4f}, {full_dm:.4f}; overlap {r['overlap']}")
    save(fig, name="SGD-from-scratch.png")
    return r


# ================================================================ learning-rate-scheduling
def sheet_learning_rate_scheduling():
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    n, x, y, stats = big_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.00025, random_state=42)
    assert x_valid.shape == torch.Size([2000]) and x_train.shape == torch.Size([7998000])
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)
    x_batch, y_batch = next(iter(train_loader))
    assert [f"{v:.4f}" for v in x_batch[:3].tolist()] == ["7.9039", "2.5032", "2.1550"]
    assert [f"{v:.4f}" for v in y_batch[:3].tolist()] == ["-1.1923", "0.5152", "0.7512"]
    batch0 = (x_batch.clone(), y_batch.clone())

    def first_step():
        m = torch.tensor([0.9]).requires_grad_()
        b = torch.tensor([0.1]).requires_grad_()
        C = mse(regression(x_batch, m, b), y_batch)
        C.backward()
        gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
        theta = torch.tensor([[b, m]]).T
        return C, gradient, theta - 0.02 * gradient

    C0, gradient, new_theta = first_step()
    assert f"{C0.item():.4f}" == "33.3355"
    assert [f"{v:.4f}" for v in gradient.flatten().tolist()] == ["9.1138", "59.7826"]
    assert [f"{v:.4f}" for v in new_theta.flatten().tolist()] == ["-0.0823", "-0.2957"]
    assert f"{mse(regression(x_batch, new_theta[1], new_theta[0]), y_batch).item():.4f}" == "2.7568"

    lambd = 0.995

    def run(batches):
        """The notebook's loop; batches(r) returns the (x, y) used in round r."""
        _, _, th = first_step()
        b, m = th[0].requires_grad_(), th[1].requires_grad_()
        lr = lambd * 0.02
        thetas, costs, lrs, val, printed = [(0.1, 0.9), (b.item(), m.item())], [C0.item()], [], [], []
        val.append(mse(regression(x_valid, 0.9, 0.1), y_valid).item())
        val.append(mse(regression(x_valid, m, b), y_valid).item())
        for r in range(1000):
            xb, yb = batches(r)
            C = mse(regression(xb, m, b), yb)
            C.backward()
            if r % 50 == 0:
                printed.append((g3(C.item()), g3(m.grad.item()), g3(b.grad.item()), g3(lr)))
            costs.append(C.item())
            lrs.append(lr)
            gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
            theta = torch.tensor([[b, m]]).T
            new_theta = theta - lr * gradient
            b = new_theta[0].requires_grad_()
            m = new_theta[1].requires_grad_()
            lr = lambd * lr
            thetas.append((b.item(), m.item()))
            val.append(mse(regression(x_valid, m, b), y_valid).item())
        return dict(thetas=np.array(thetas), costs=np.array(costs), lrs=np.array(lrs), val=np.array(val),
                    printed=printed, last_C=C.item())

    # the notebook's earlier code: next(iter(train_loader)) built a new iterator every round -> always batch 0
    same = run(lambda r: next(iter(train_loader)))
    assert same["printed"][0] == ("2.76", "-9.63", "-2.6", "0.0199")
    assert same["printed"][12] == ("0.793", "0.0359", "-0.212", "0.000983")
    assert same["printed"][19] == ("0.786", "0.0336", "-0.199", "0.00017")
    assert (g3(same["last_C"]), g3(same["thetas"][-1][1]), g3(same["thetas"][-1][0])) == ("0.785", "-0.449", "1.85")
    assert g3(same["val"][-1]) == "1.05"
    # the notebook now: batches = iter(train_loader) before the loop, so round r trains on batch r
    loader_batches = []
    for k, batch in enumerate(train_loader):
        if k >= 1000:
            break
        loader_batches.append(batch)
    assert torch.equal(loader_batches[0][0], batch0[0])
    nb = run(lambda r: loader_batches[r])
    assert nb["printed"][0] == ("2.76", "-9.63", "-2.6", "0.0199")
    assert nb["printed"][1] == ("1.79", "-3.33", "-1.62", "0.0155")
    assert nb["printed"][12] == ("0.78", "-0.149", "-0.288", "0.000983")
    assert nb["printed"][19] == ("1.16", "-0.0295", "-0.257", "0.00017")

    xb_np, yb_np = batch0[0].numpy(), batch0[1].numpy()
    b_b0, m_b0 = least_squares(xb_np, yb_np)
    c_b0 = float(np.mean((m_b0 * xb_np + b_b0 - yb_np) ** 2))
    xv_np, yv_np = x_valid.numpy(), y_valid.numpy()
    true_val = float(np.mean((-0.5 * xv_np + 2 - yv_np) ** 2))
    b_full, m_full = least_squares(x.numpy(), y.numpy())
    budget_total = 0.0199 / (1 - lambd)
    budget_1000 = float(nb["lrs"].sum())

    fig = new_figure("Learning-rate scheduling: big steps first, then ever smaller ones",
                     "αᵣ = 0.0199 · 0.995ʳ  (α₀ = 0.02 doubled, λ = 0.995 per round);   batches of 32 from "
                     "DataLoader(TensorDataset(x_train, y_train), batch_size=32)")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    rounds = np.arange(1000)
    ax.plot(rounds, 1000 * nb["lrs"], color=BLUE, linewidth=2.2, label="learning rate αᵣ")
    pr = np.arange(0, 1000, 50)
    ax.plot(pr, [1000 * float(p[3]) for p in nb["printed"]], "o", color=BLUE, markersize=4, zorder=4,
            label="printed every 50 rounds")
    ax.set_xlabel("round r")
    ax.set_ylabel("learning rate αᵣ (× 10⁻³)")
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(rounds, np.cumsum(nb["lrs"]), color=ORANGE, linewidth=2, label="Σ α so far (step budget)")
    ax2.axhline(budget_total, color=ORANGE, linestyle=":", linewidth=1.2)
    ax2.text(990, budget_total + 0.08, f"limit 0.0199/(1 − λ) = {budget_total:.2f}", ha="right", va="bottom",
             fontsize=9, color=ORANGE)
    ax2.set_ylabel("cumulative Σ α", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 4.8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax.set_title("Exponential decay: the total step is finite")

    ax = fig.add_subplot(gs[0, 1])
    rr = np.arange(len(nb["costs"]))
    ax.plot(rr, nb["costs"], color=BLUE_LIGHT, linewidth=0.6, alpha=0.7, label="batch cost, notebook (batch r in round r)")
    ax.plot(rr, same["costs"], color=ORANGE, linewidth=2.2,
            label=f"batch cost, earlier code (always batch 0): {same['last_C']:.3f}")
    ax.axhline(c_b0, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.plot(np.arange(len(same["val"])), same["val"], color=INK, linewidth=1.6,
            label=f"validation, earlier code: {same['val'][-1]:.3f}")
    ax.plot(np.arange(len(nb["val"])), nb["val"], color=AQUA, linewidth=1.8,
            label=f"validation, notebook: {nb['val'][-1]:.3f}")
    ax.axhline(true_val, color=AQUA, linestyle="--", linewidth=1.1)
    ax.text(1000, 0.43, f"batch 0's own minimum (dotted): {c_b0:.3f}", ha="right", va="bottom", fontsize=9,
            color=ORANGE)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 40)
    ax.set_xlabel("round")
    ax.set_ylabel("MSE (log)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8.6)
    ax.set_title("Cost per round")

    ax = fig.add_subplot(gs[0, 2])
    cost_contours(ax, stats, (-0.3, 2.6), (-0.8, 1.0), [1.05, 1.2, 1.5, 2, 3, 5, 9, 16, 28],
                  "Paths over the full-data cost")
    ax.plot(nb["thetas"][:, 0], nb["thetas"][:, 1], color=AQUA, linewidth=1.3, zorder=4,
            label=f"notebook → ({nb['thetas'][-1][0]:.2f}, {nb['thetas'][-1][1]:.3f})")
    ax.plot(same["thetas"][:, 0], same["thetas"][:, 1], color=ORANGE, linewidth=1.8, zorder=5,
            label=f"earlier code → ({same['thetas'][-1][0]:.2f}, {same['thetas'][-1][1]:.3f})")
    ax.plot([b_b0], [m_b0], "s", color=ORANGE, markersize=7, zorder=6,
            label=f"batch 0's least squares ({b_b0:.2f}, {m_b0:.3f})")
    ax.plot([b_full], [m_full], "*", color=INK, markersize=14, zorder=6,
            label=f"full-data minimum ({b_full:.2f}, {m_full:.3f})")
    ax.plot([0.1], [0.9], "o", color=INK, markersize=5, zorder=6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=1, facecolor=SURFACE, frameon=True, edgecolor="none")

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(xv_np, yv_np, s=8, color=BLUE, alpha=0.15, zorder=2, label="2,000 validation points")
    line_through(ax, nb["thetas"][-1][1], nb["thetas"][-1][0], (0, 8), color=ORANGE, linewidth=2.4,
                 label=f"notebook: m = {nb['thetas'][-1][1]:.3f}, b = {nb['thetas'][-1][0]:.2f}")
    line_through(ax, -0.5, 2.0, (0, 8), color=AQUA, linestyle="--", linewidth=1.8, label="simulated: m = −0.5, b = 2")
    ax.set_xlim(0, 8)
    ax.set_ylim(-5.5, 7.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Validation cost {nb['val'][-1]:.2f} (SGD-from-scratch: 1.44)")

    panel(fig.add_subplot(gs[1, 1]), "Why decay the learning rate", [
        "Early: θ is far off, gradients are large and",
        "big steps cover ground fast (α doubled to 0.02).",
        "Late: the batch gradient is mostly noise, so a",
        "fixed α keeps θ jittering around the minimum;",
        "a shrinking α averages the noise out.",
        "",
        f"Σα over 1000 rounds = {budget_1000:.2f} of at most {budget_total:.2f}:",
        "after ~600 rounds θ barely moves.",
        "`Step 950, cost 1.16, ..., lr 0.00017`",
    ], spacing=0.087)

    panel(fig.add_subplot(gs[1, 2]), "One iterator, a new batch per round", [
        "`batches = iter(train_loader)`",
        "`x_batch, y_batch = next(batches)`",
        "Round r trains on batch r: 32,000 distinct",
        "points over 1000 rounds.",
        "",
        "The earlier code called next(iter(train_loader))",
        "each round: a new iterator, so always batch 0.",
        f"Validation: {nb['val'][-1]:.3f} now vs {same['val'][-1]:.3f} then;",
        "the decaying α stops θ short either way.",
    ], spacing=0.087)
    save(fig, name="learning-rate-scheduling.png")
    return nb, same, (b_b0, m_b0, c_b0), true_val


# ================================================================ artificial-neurons
def sheet_artificial_neurons():
    torch.manual_seed(42)
    n_input, n_dense = 784, 128
    x = torch.rand(n_input)
    assert [f"{v:.4f}" for v in x[0:6].tolist()] == ["0.8823", "0.9150", "0.3829", "0.9593", "0.3904", "0.6009"]
    b = torch.zeros(n_dense)
    W = torch.empty([n_input, n_dense])
    W = torch.nn.init.xavier_normal_(W)
    assert [f"{v:.4f}" for v in W[0, 0:4].tolist()] == ["0.0008", "0.0038", "0.0349", "0.0630"]
    assert [f"{v:.4f}" for v in W[3, 0:4].tolist()] == ["0.0050", "-0.0144", "0.0405", "-0.0499"]
    z = torch.add(torch.matmul(x, W), b)
    a = torch.sigmoid(z)
    assert a.shape == torch.Size([128])
    counts, edges = np.histogram(a.numpy())
    assert counts.sum() == 128 and 0.1 < edges[0] < 0.15 and 0.85 < edges[-1] < 0.9

    x_np, W_np, z_np, a_np = x.numpy(), W.numpy(), z.numpy(), a.numpy()
    sigma_w = np.sqrt(2 / (n_input + n_dense))
    ex2 = float((x_np ** 2).sum())
    sigma_z = np.sqrt(ex2) * sigma_w                  # std of z_j given this x
    sigma_z_naive = np.sqrt(ex2) * 1.0                # if W ~ N(0, 1)
    from math import erf
    saturated_naive = 1 - erf(4 / (sigma_z_naive * np.sqrt(2)))   # P(|z| > 4)
    assert np.abs(z_np).max() < 2.2
    n_params = n_input * n_dense + n_dense
    assert n_params == 100480

    fig = new_figure("A dense layer: 128 neurons, each a weighted sum of 784 inputs, squashed",
                     "a = σ(xW + b):  x (784)  ·  W (784 × 128)  +  b (128)  →  z (128)  →  σ(z) = 1/(1 + e^(−z))  →  a (128);"
                     f"   {n_params:,} parameters")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    stack_diagram(ax, [
        ("input x", "torch.rand(784)", "(784,)", CARD),
        ("weights W", "xavier_normal_", "(784, 128)", BLUE_LIGHT),
        ("bias b", "torch.zeros(128)", "(128,)", BLUE_LIGHT),
        ("z = xW + b", "matmul + add", "(128,)", ORANGE_LIGHT),
        ("a = σ(z)", "torch.sigmoid", "(128,)", AQUA_LIGHT),
    ], "Shapes through the layer", detail_size=9.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(x_np, bins=20, range=(0, 1), color=BLUE, edgecolor=SURFACE, zorder=2)
    ax.axhline(n_input / 20, color=ORANGE, linestyle="--", linewidth=1.4, label="uniform: 39.2 per bin")
    ax.set_xlabel("pixel value xᵢ")
    ax.set_ylabel("count")
    ax.grid(axis="y", zorder=0)
    ax.set_ylim(0, 64)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("x: 784 'pixels' from U[0, 1), seed 42")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(W_np.ravel(), bins=60, density=True, color=BLUE, edgecolor=SURFACE, linewidth=0.3, zorder=2,
            label=f"all 100,352 weights, std {W_np.std():.4f}")
    ww = np.linspace(-0.2, 0.2, 200)
    ax.plot(ww, np.exp(-ww ** 2 / (2 * sigma_w ** 2)) / (sigma_w * np.sqrt(2 * np.pi)), color=ORANGE,
            linewidth=2, label=f"N(0, σ²), σ = √(2/(784 + 128)) = {sigma_w:.4f}")
    ax.set_xlim(-0.2, 0.2)
    ax.set_xlabel("weight value")
    ax.set_ylabel("density")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_ylim(0, 11.5)
    ax.set_title("W: Xavier (Glorot) normal initialization")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(z_np, bins=np.linspace(-3, 3, 17), color=BLUE, edgecolor=SURFACE, zorder=2, label=f"z, std {z_np.std():.3f}")
    zz = np.linspace(-3, 3, 200)
    ax.plot(zz, 128 * (6 / 16) * np.exp(-zz ** 2 / (2 * sigma_z ** 2)) / (sigma_z * np.sqrt(2 * np.pi)),
            color=ORANGE, linewidth=2, label=f"predicted N(0, {sigma_z:.3f}²)")
    ax.set_xlim(-3, 3)
    ax.set_xlabel("pre-activation zⱼ")
    ax.set_ylabel("count")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("z = xW + b: one number per neuron")

    ax = fig.add_subplot(gs[1, 1])
    zs = np.linspace(-6, 6, 300)
    ax.plot(zs, 1 / (1 + np.exp(-zs)), color=INK2, linewidth=1.6, label="σ(z)")
    ax.axvspan(z_np.min(), z_np.max(), color=BLUE_LIGHT, alpha=0.4, label="range of this layer's z")
    ax.scatter(z_np, a_np, s=16, color=ORANGE, zorder=4, label="128 neurons (z, a)")
    ax.axvspan(-6, -4, color=GRID, alpha=0.6)
    ax.axvspan(4, 6, color=GRID, alpha=0.6, label="saturated: slope σ′ < 0.02")
    ax.set_xlim(-6, 6)
    ax.set_xlabel("z")
    ax.set_ylabel("activation a = σ(z)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"a = σ(z): all 128 land in [{a_np.min():.2f}, {a_np.max():.2f}]")

    panel(fig.add_subplot(gs[1, 2]), "Why Xavier scaling", [
        "zⱼ = Σᵢ xᵢWᵢⱼ sums 784 random terms, so",
        "Var(zⱼ) = Var(W) · Σᵢ xᵢ²  (bias 0).",
        f"Here Σ xᵢ² = {ex2:.1f}, Var(W) = 2/912:",
        f"std(z) ≈ {sigma_z:.3f}, the sigmoid's sensitive range.",
        "",
        f"With W ~ N(0, 1) instead: std(z) ≈ {sigma_z_naive:.1f},",
        f"so {saturated_naive:.0%} of neurons would have |z| > 4: stuck",
        "near 0 or 1, with σ′ ≈ 0 and almost no gradient.",
        "Here none is (|z| < 2.2 for all 128).",
    ], spacing=0.087)
    save(fig, name="artificial-neurons.png")


if __name__ == "__main__":
    sheet_regression_in_pytorch()
    sheet_single_point()
    sheet_batch()
    sheet_gradient_descent()
    sheet_sgd_from_scratch()
    sheet_learning_rate_scheduling()
    sheet_artificial_neurons()
