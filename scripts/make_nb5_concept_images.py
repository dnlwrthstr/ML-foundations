#!/usr/bin/env python3
"""Concept sheets for notebooks/5-probability.ipynb, one per segment.

Segment 1 (Introduction to Probability), Segment 2 (Distributions in Machine Learning) and
Segment 3 (Information Theory). Every number is computed here from the notebook's own data:
the notebook seeds NumPy's global generator once (np.random.seed(42)) and then draws from it
cell after cell, so replay_notebook_rng() repeats every one of those draws in the same order
and the results are asserted against the notebook's printed outputs. The iris measurements
come from scikit-learn's bundled copy (the notebook downloads seaborn's; the columns used here
reproduce its printed covariances and correlations exactly).
Images are written to notebooks/img/5-probability_seg<N>.png.

Run from anywhere:  python scripts/make_nb5_concept_images.py
"""
from functools import partial
from math import comb, factorial

import numpy as np
import scipy.stats as st
from sklearn.datasets import load_iris

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel)
from viz_style import save as _save

save = partial(_save, folder=REPO / "notebooks" / "img")
STEM = "5-probability"

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"


# ================================================================ helpers
def coinflip_prob(n, k):
    """The notebook's helper: probability of k heads in n fair flips."""
    return factorial(n) / (factorial(k) * factorial(n - k)) / 2 ** n


def sample_mean_calculator(input_dist, sample_size, n_samples):
    """The notebook's helper, same draws from the global generator."""
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(input_dist, size=sample_size, replace=False)
        sample_means.append(sample.mean())
    return sample_means


def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def vline(ax, x, label, color, style="-", width=2.0):
    ax.axvline(x, color=color, linestyle=style, linewidth=width, label=label, zorder=3)


def replay_notebook_rng():
    """Repeat every draw the notebook makes from np.random, in notebook order."""
    d = {}
    np.random.seed(42)
    d["single_flip"] = np.random.binomial(1, 0.5)
    ns = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    d["ns"] = ns
    d["lln_heads"] = np.array([np.random.binomial(n, 0.5) for n in ns])
    d["big_flip"] = np.random.binomial(100000, 0.5)
    d["five_flips"] = np.random.binomial(5, 0.5, 1000)
    d["skew0"] = st.skewnorm.rvs(0, size=1000)
    d["skew10"] = st.skewnorm.rvs(10, size=1000)
    # ---- Segment 2
    d["u"] = np.random.uniform(size=10000)
    d["x_normal"] = np.random.normal(size=10000)
    d["hundred_flips"] = np.array([np.random.binomial(100, 0.5) for _ in range(10000)])
    d["x_sample"] = np.random.choice(d["x_normal"], size=10, replace=False)
    x = d["x_normal"]
    d["means_10x10"] = sample_mean_calculator(x, 10, 10)
    d["means_n10"] = np.array(sample_mean_calculator(x, 10, 1000))
    d["means_n100"] = np.array(sample_mean_calculator(x, 100, 1000))
    d["means_n1000"] = np.array(sample_mean_calculator(x, 1000, 1000))
    d["s"] = st.skewnorm.rvs(10, size=10000)
    d["means_s10"] = np.array(sample_mean_calculator(d["s"], 10, 1000))
    d["means_s1000"] = np.array(sample_mean_calculator(d["s"], 1000, 1000))
    d["m"] = np.concatenate((np.random.normal(size=5000), np.random.normal(loc=4.0, size=5000)))
    d["means_m1000"] = np.array(sample_mean_calculator(d["m"], 1000, 1000))
    d["means_u1000"] = np.array(sample_mean_calculator(d["u"], 1000, 1000))
    d["lognormal"] = np.random.lognormal(size=10000)
    d["exponential"] = np.random.exponential(scale=4, size=10000)
    d["laplace"] = np.random.laplace(size=10000)
    d["laplace25"] = np.random.laplace(loc=25, size=10000)
    d["binom_fair"] = np.random.binomial(5, 0.5, 1000)
    d["binom_08"] = np.random.binomial(5, 0.8, 1000)
    d["bernoulli"] = np.random.binomial(1, 0.5)
    d["rolls"] = np.random.multinomial(1000, [1 / 6.] * 6)
    d["multinoulli"] = np.random.multinomial(1, [1 / 6.] * 6)
    d["poisson"] = np.random.poisson(5, 1000)
    # mixture cell
    distributions = [
        {"type": np.random.normal, "kwargs": {"loc": -3, "scale": 2}},
        {"type": np.random.uniform, "kwargs": {"low": 4, "high": 6}},
        {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
    ]
    coefficients = np.array([0.5, 0.2, 0.3])
    coefficients /= coefficients.sum()
    sample_size = 10000
    data = np.zeros((sample_size, len(distributions)))
    for idx, distr in enumerate(distributions):
        data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(len(distributions)), size=(sample_size,), p=coefficients)
    d["mixture"] = data[np.arange(sample_size), random_idx]
    d["mixture_idx"] = random_idx
    return d


# ================================================================ Segment 1
def sheet_seg1_intro_to_probability(d):
    # ---- counting and the binomial PMF
    P = [coinflip_prob(5, k) for k in range(6)]
    assert P == [0.03125, 0.15625, 0.3125, 0.3125, 0.15625, 0.03125]
    E = sum(k * P[k] for k in range(6))
    assert E == 2.5 and comb(5, 3) == 10 and coinflip_prob(5, 3) == 0.3125

    # ---- law of large numbers
    assert d["single_flip"] == 0
    assert d["lln_heads"].tolist() == [2, 3, 4, 6, 13, 27, 54, 136, 258, 509, 1038, 2094]
    assert d["big_flip"] == 50007
    prop = d["lln_heads"] / d["ns"]
    assert np.allclose(prop[[0, 3, 11]], [1.0, 0.375, 0.51123047])

    # ---- 1000 five-flip experiments: mean, median, mode
    heads_count = d["five_flips"]
    heads, event_count = np.unique(heads_count, return_counts=True)
    assert event_count.tolist() == [33, 181, 283, 313, 161, 29]
    event_proba = event_count / 1000
    assert heads_count.mean() == 2.475 and np.median(heads_count) == 3.0
    assert st.mode(heads_count)[0] == 3

    # ---- skewed sample x = skewnorm(10): quantiles, box plot, dispersion
    x = d["skew10"]
    assert f"{np.median(x):.10f}" == "0.6987365121"
    p95, p99 = np.percentile(x, [95, 99])
    q = np.percentile(x, [25, 50, 75])
    assert np.allclose([p95, p99], [2.06249928, 2.51830108])
    assert np.allclose(q, [0.32494046, 0.69873651, 1.19431522])
    xbar, var, sigma = x.mean(), x.var(), x.std()
    assert np.isclose(xbar, 0.8219573880882259) and np.isclose(var, 0.39813585982250543)
    assert np.isclose(sigma, 0.6309800787841923)
    sem = sigma / np.sqrt(len(x))
    assert np.isclose(sem, 0.019953342071505353)
    r = 1.194 - 0.325                       # the notebook's rounded IQR
    low_reach, high_reach = 0.325 - 1.5 * r, 1.194 + 1.5 * r
    assert np.isclose(high_reach, 2.4975) and np.isclose(low_reach, -0.9785)
    outliers = np.sort(x[x > high_reach])
    upper_whisker = x[x <= high_reach].max()
    assert len(outliers) == 11 and f"{upper_whisker:.3f}" == "2.435"
    assert np.isclose(x.min(), -0.17083838831568182)

    # ---- iris: covariance and correlation
    iris = load_iris().data
    sepal_length, sepal_width, petal_length = iris[:, 0], iris[:, 1], iris[:, 2]
    cov_lp = np.cov(sepal_length, petal_length, ddof=0)
    cov_lw = np.cov(sepal_length, sepal_width, ddof=0)
    assert np.allclose(cov_lp, [[0.68112222, 1.26582], [1.26582, 3.09550267]])
    assert np.allclose(cov_lw, [[0.68112222, -0.04215111], [-0.04215111, 0.18871289]])
    rho_lp = st.pearsonr(sepal_length, petal_length)[0]
    rho_lw = st.pearsonr(sepal_length, sepal_width)[0]
    assert np.isclose(rho_lp, 0.8717537758865831) and np.isclose(rho_lw, -0.11756978413300206)

    fig = new_figure("Probability counts outcomes; a sample is summarized by center, spread and relatedness",
                     r"P(event) = # event outcomes / # outcomes in Ω,   $\binom{n}{k} = \frac{n!}{k!(n-k)!}$,   "
                     r"$\mathbb{E} = \sum_x x P(x)$,   $\sigma^2 = \frac{1}{n}\sum (x_i-\bar{x})^2$,   "
                     r"$\rho = \frac{cov(x,y)}{\sigma_x\sigma_y}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # (0,0) exact binomial vs 1000 simulated experiments
    ax = fig.add_subplot(gs[0, 0])
    width = 0.38
    ax.bar(heads - width / 2, P, width=width - 0.04, color=BLUE, label=r"exact $\binom{5}{k}/2^5$", zorder=2)
    ax.bar(heads + width / 2, event_proba, width=width - 0.04, color=ORANGE_LIGHT,
           label="1000 experiments", zorder=2)
    for k in heads:
        ax.text(k - width / 2, P[k] + 0.008, f"{P[k] * 32:.0f}/32", ha="center", fontsize=9, color=INK)
    vline(ax, E, f"E = {E}", INK2, "--", 1.6)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(heads)
    ax.set_xlabel("heads in 5 fair flips")
    ax.set_ylabel("probability")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Five flips: counting vs. simulating")

    # (0,1) law of large numbers
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0.5, color=ORANGE, linewidth=1.8, zorder=1)
    ax.plot(d["ns"], prop, color=BLUE, linewidth=1.2, zorder=2)
    ax.scatter(d["ns"], prop, color=BLUE, s=36, zorder=3)
    for n, h, pr in zip(d["ns"], d["lln_heads"], prop):
        if n in (2, 16, 64, 4096):
            ax.annotate(f"{h}/{n}", (n, pr), textcoords="offset points", xytext=(0, 8 if pr >= 0.5 else -15),
                        ha="center", fontsize=9, color=INK2)
    ax.text(4096, 0.6, f"100,000 flips: {d['big_flip'] / 100000:.5f}", ha="right", fontsize=10.5,
            color=ORANGE, weight="bold")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.25, 1.08)
    ax.set_xlabel("coin flips in experiment (log scale)")
    ax.set_ylabel("proportion heads")
    ax.grid(True)
    ax.set_title("Law of large numbers: proportion → 0.5")

    panel(fig.add_subplot(gs[0, 2]), "With the notebook's numbers", [
        r"$P(HHHHH) = \frac{1}{4} \cdot \frac{1}{8} = \frac{1}{32} \approx 0.031$",
        "",
        r"$\binom{5}{3} = 10$ ways, so $P(3\ of\ 5) = 10/32 = 0.3125$",
        "",
        r"Lottery: $\frac{1}{1000}\cdot 4990 + \frac{999}{1000}\cdot(-10) = -5$",
        "",
        "Center of the 1000 experiments:",
        f"mean {heads_count.mean()}, median {np.median(heads_count):.0f}, mode 3 (313 times)",
        "",
        f"Skewed x (n = 1000): " + r"$\bar{x}$" + f" = {xbar:.3f}, " + r"$\sigma^2$" + f" = {var:.3f},",
        r"$\sigma$" + f" = {sigma:.3f},  standard error " + r"$\sigma/\sqrt{n}$" + f" = {sem:.4f}",
        "",
        "Gambler's fallacy: each flip is still 50/50.",
    ], fontsize=12, spacing=0.08)

    # (1,0) skewed distribution: center and spread
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(x, bins=30, color=BLUE_LIGHT, edgecolor=SURFACE, zorder=2)
    ax.axvspan(xbar - sigma, xbar + sigma, color=AQUA_LIGHT, alpha=0.45, zorder=1,
               label=r"$\bar{x} \pm \sigma$" + f" (σ = {sigma:.3f})")
    vline(ax, xbar, f"mean = {xbar:.3f}", ORANGE)
    vline(ax, np.median(x), f"median = {np.median(x):.3f}", BLUE, "--")
    vline(ax, p95, f"95th pct = {p95:.2f}", INK2, ":", 1.6)
    vline(ax, p99, f"99th pct = {p99:.2f}", MUTED, ":", 1.6)
    ax.set_xlabel("x = skewnorm(a = 10) sample, n = 1000")
    ax.set_ylabel("count")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Skew drags the mean toward the tail")

    # (1,1) box-and-whisker built by hand
    ax = fig.add_subplot(gs[1, 1])
    ax.add_patch(plt_rect((q[0], 0.35), q[2] - q[0], 0.3, BLUE_LIGHT, BLUE))
    ax.plot([q[1], q[1]], [0.35, 0.65], color=ORANGE, linewidth=2.5, zorder=3)
    ax.plot([x.min(), q[0]], [0.5, 0.5], color=BLUE, linewidth=1.6)
    ax.plot([q[2], upper_whisker], [0.5, 0.5], color=BLUE, linewidth=1.6)
    for w in (x.min(), upper_whisker):
        ax.plot([w, w], [0.42, 0.58], color=BLUE, linewidth=1.6)
    ax.scatter(outliers, np.full(len(outliers), 0.5), s=30, facecolor="none", edgecolor=ORANGE, zorder=3)
    ax.axvline(high_reach, color=MUTED, linestyle=":", linewidth=1.4)
    ax.axvline(low_reach, color=MUTED, linestyle=":", linewidth=1.4)
    ax.text(q[0], 0.3, f"Q1\n{q[0]:.3f}", ha="right", va="top", fontsize=9, color=INK2)
    ax.text(q[1], 0.69, f"median {q[1]:.3f}", ha="center", va="bottom", fontsize=9, color=ORANGE)
    ax.text(q[2], 0.3, f"Q3\n{q[2]:.3f}", ha="left", va="top", fontsize=9, color=INK2)
    ax.text((q[0] + q[2]) / 2, 0.12, f"IQR = Q3 − Q1 ≈ {r:.3f}", ha="center", fontsize=10, color=INK)
    ax.text(high_reach, 0.88, f"Q3 + 1.5·IQR\n= {high_reach:.4f}", ha="center", fontsize=9, color=INK2)
    ax.text(low_reach, 0.88, f"Q1 − 1.5·IQR\n= {low_reach:.4f}", ha="center", fontsize=9, color=INK2)
    ax.text(upper_whisker, 0.36, f"{upper_whisker:.3f}", ha="center", va="top", fontsize=9, color=BLUE)
    ax.text(3.3, 0.6, f"{len(outliers)} outliers", ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.text(x.min(), 0.6, f"min\n{x.min():.3f}", ha="center", fontsize=9, color=BLUE)
    ax.set_xlim(-1.3, 4.3)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("x")
    ax.set_title("Box-and-whisker: quartiles and 1.5 × IQR")

    # (1,2) iris covariance / correlation
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(sepal_length, petal_length, s=16, color=BLUE, alpha=0.75,
               label=f"petal length: cov {cov_lp[0, 1]:.3f}, ρ = {rho_lp:.3f}")
    ax.scatter(sepal_length, sepal_width, s=16, color=ORANGE, alpha=0.75,
               label=f"sepal width: cov {cov_lw[0, 1]:.3f}, ρ = {rho_lw:.3f}")
    ax.set_ylim(0, 9.2)
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("cm")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Iris (n = 150): related vs. barely related")
    save(fig, name=f"{STEM}_seg1.png")


def plt_rect(xy, w, h, face, edge):
    from matplotlib.patches import Rectangle
    return Rectangle(xy, w, h, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=2)


# ================================================================ Segment 2
def sheet_seg2_distributions(d):
    x = d["x_normal"]
    assert np.isclose(x.mean(), 0.010746657894433886) and np.isclose(x.std(), 1.006376502514602)
    assert np.allclose(d["x_sample"], [0.32659596, -0.93700698, 2.27851732, 1.21587356, 1.35370984,
                                       -0.4057365, 0.38816803, -0.55343125, -1.6885922, -2.33595362])
    assert np.isclose(d["x_sample"].mean(), -0.03578558368006331)
    assert d["bernoulli"] == 1 and d["multinoulli"].tolist() == [0, 0, 0, 0, 1, 0]
    # The notebook printed rolls [176, 150, 178, 183, 171, 142]; current NumPy's multinomial gives
    # [176, 151, 178, 183, 171, 141] for the same state (the next draws match again), so the sheet
    # shows the notebook's printed counts.
    notebook_rolls = np.array([176, 150, 178, 183, 171, 142])
    assert np.abs(d["rolls"] - notebook_rolls).max() <= 1 and d["rolls"].sum() == 1000
    pois = d["poisson"]
    assert pois[:20].tolist() == [4, 3, 2, 2, 6, 6, 6, 5, 5, 3, 6, 4, 5, 6, 3, 6, 7, 6, 5, 4]
    pk, pc = np.unique(pois, return_counts=True)
    assert pk.tolist() == list(range(14))
    assert pc.tolist() == [6, 43, 82, 146, 166, 162, 141, 132, 54, 33, 20, 8, 5, 2]

    fig = new_figure("Each random process has its own shape, yet sample means come out normal",
                     r"Continuous (PDF): uniform, $\mathcal{N}(\mu,\sigma^2)$, log-normal, exponential, Laplace.   "
                     r"Discrete (PMF): binomial, multinomial, Poisson.   CLT: $\bar{x} \sim \mathcal{N}(\mu, \sigma^2/n)$.")
    gs = grid(fig, top=0.84, wspace=0.3)

    # (0,0) symmetric continuous
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-5, 5, 61)
    t = np.linspace(-5, 5, 500)
    ax.hist(d["u"], bins=bins, density=True, histtype="stepfilled", color=AQUA_LIGHT, alpha=0.7)
    ax.hist(x, bins=bins, density=True, histtype="step", color=BLUE, linewidth=1.6)
    ax.hist(d["laplace"], bins=bins, density=True, histtype="step", color=ORANGE, linewidth=1.6)
    ax.plot(t, normal_pdf(t), color=BLUE, linewidth=1, linestyle="--")
    ax.plot(t, 0.5 * np.exp(-np.abs(t)), color=ORANGE, linewidth=1, linestyle="--")
    ax.plot([], [], color=AQUA_LIGHT, linewidth=8, label="uniform [0, 1)")
    ax.plot([], [], color=BLUE, label=f"standard normal (mean {x.mean():.3f}, sd {x.std():.3f})")
    ax.plot([], [], color=ORANGE, label="Laplace: sharp peak, heavier tails")
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("x  (10,000 draws each; dashed = true PDF)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Symmetric: uniform, Gaussian, Laplace")

    # (0,1) positive, skewed
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 16, 81)
    t = np.linspace(0.01, 16, 500)
    ln, ex = d["lognormal"], d["exponential"]
    ax.hist(ln, bins=bins, density=True, histtype="stepfilled", color=BLUE_LIGHT)
    ax.hist(ex, bins=bins, density=True, histtype="step", color=ORANGE, linewidth=1.8)
    ax.plot(t, normal_pdf(np.log(t)) / t, color=BLUE, linewidth=1, linestyle="--")
    ax.plot(t, np.exp(-t / 4) / 4, color=ORANGE, linewidth=1, linestyle="--")
    ax.plot([], [], color=BLUE_LIGHT, linewidth=8, label=f"log-normal: log x ~ N(0, 1), median {np.median(ln):.2f}")
    ax.plot([], [], color=ORANGE, label=f"exponential, scale 4: mean {ex.mean():.2f}")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("x  (10,000 draws each; dashed = true PDF)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Positive and right-skewed")

    # (0,2) discrete PMFs
    ax = fig.add_subplot(gs[0, 2])
    k = np.arange(14)
    series = [("binomial n=5, p=0.5", d["binom_fair"], BLUE, -0.27),
              ("binomial n=5, p=0.8", d["binom_08"], ORANGE, 0.0),
              (r"Poisson $\lambda$=5", pois, AQUA, 0.27)]
    for name, draws, color, off in series:
        freq = np.bincount(draws, minlength=14)[:14] / len(draws)
        ax.bar(k + off, freq, width=0.26, color=color, label=name, zorder=2)
    ax.set_xticks(k)
    ax.set_xlabel("count k  (1000 experiments each)")
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 0.48)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Discrete: binomial and Poisson PMFs")

    # (1,0) mixture
    ax = fig.add_subplot(gs[1, 0])
    mix = d["mixture"]
    t = np.linspace(-11, 8, 800)
    comps = [0.5 * normal_pdf(t, -3, 2), 0.2 * ((t >= 4) & (t <= 6)) / 2, 0.3 * normal_pdf(t, 2, 1)]
    ax.hist(mix, bins=100, density=True, color=BLUE_LIGHT, edgecolor=BLUE_LIGHT, zorder=1)
    for comp, color, name in zip(comps, [BLUE, ORANGE, AQUA],
                                 ["0.5 · N(−3, 2²)", "0.2 · U(4, 6)", "0.3 · N(2, 1)"]):
        ax.plot(t, comp, color=color, linewidth=1.6, linestyle="--", label=name, zorder=3)
    ax.plot(t, sum(comps), color=INK, linewidth=1.4, label="mixture density", zorder=4)
    shares = np.bincount(d["mixture_idx"], minlength=3) / len(mix)
    ax.text(0.02, 0.97, "component picked\n" + ", ".join(f"{s:.3f}" for s in shares) + " of the time",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("x  (10,000 draws)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Mixture: a multinomial picks the component")

    # (1,1) CLT on skewed, bimodal and uniform inputs: inputs on top, standardized means below
    inner = gs[1, 1].subgridspec(2, 1, hspace=0.35)
    inputs = [("skewed s", d["s"], d["means_s1000"], ORANGE),
              ("bimodal m", d["m"], d["means_m1000"], BLUE),
              ("uniform u", d["u"], d["means_u1000"], AQUA)]
    ax = fig.add_subplot(inner[0])
    for name, data, _, color in inputs:
        ax.hist(data, bins=np.linspace(-4, 8, 97), density=True, histtype="step", color=color,
                linewidth=1.6, label=name)
    ax.set_xlim(-4, 8)
    ax.set_ylim(0, 1.5)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("input")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9, ncol=3, handlelength=1.2, columnspacing=0.8)
    ax.set_title("CLT: odd inputs, bell-shaped sample means")
    ax = fig.add_subplot(inner[1])
    z = np.linspace(-4, 4, 300)
    for name, _, means, color in inputs:
        standardized = (means - means.mean()) / means.std()
        ax.hist(standardized, bins=np.linspace(-4, 4, 33), density=True, histtype="step", color=color,
                linewidth=1.6, label=f"sd of means {means.std():.3f}")
    ax.plot(z, normal_pdf(z), color=INK, linewidth=1.2, linestyle="--", label="N(0, 1)")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("standardized mean of n = 1000 draws (1000 samples each)")
    ax.set_ylabel("means")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, handlelength=1.2, columnspacing=0.8)

    # (1,2) CLT: sample size tightens the means of the normal x
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(-1.5, 1.5, 61)
    for means, n, color in [(d["means_n10"], 10, BLUE_LIGHT), (d["means_n100"], 100, AQUA),
                            (d["means_n1000"], 1000, ORANGE)]:
        ax.hist(means, bins=bins if n < 1000 else np.linspace(-0.15, 0.15, 25), density=True,
                histtype="stepfilled" if n == 10 else "step", color=color, linewidth=1.8,
                edgecolor=BLUE if n == 10 else color,
                label=f"n = {n}: sd {means.std():.3f}  (1/√n = {1 / np.sqrt(n):.3f})")
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("sample mean (1000 samples from the normal x)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(r"Bigger samples: means tighten as $\sigma/\sqrt{n}$")
    save(fig, name=f"{STEM}_seg2.png")
    return notebook_rolls


# ================================================================ Segment 3
def self_info(my_p):
    return -1 * np.log(my_p)


def binary_entropy(my_p):
    return (my_p - 1) * np.log(1 - my_p) - my_p * np.log(my_p)


def cross_entropy(y, a):
    return -1 * (y * np.log(a) + (1 - y) * np.log(1 - a))


def sheet_seg3_information_theory():
    printed_self = {1: 0.0, 0.1: 2.3025850929940455, 0.01: 4.605170185988091, 0.5: 0.6931471805599453}
    for p, v in printed_self.items():
        assert np.isclose(self_info(p), v)
    assert np.isclose(self_info(0.5) + self_info(0.5), 1.3862943611198906)
    assert np.isclose(self_info(0.5) + self_info(0.5), self_info(0.25))
    printed_h = {0.00001: 0.00012512920464949012, 0.99999: 0.00012512920464901166,
                 0.9: 0.3250829733914482, 0.5: 0.6931471805599453}
    for p, v in printed_h.items():
        assert np.isclose(binary_entropy(p), v)
    printed_ce = {0.9997: 0.00030004500900199243, 0.9: 0.10536051565782628,
                  0.6: 0.5108256237659907, 0.3: 1.2039728043259361}
    for a, v in printed_ce.items():
        assert np.isclose(cross_entropy(1, a), v)

    fig = new_figure("Information is surprise: rare events carry more, and cross-entropy prices wrong guesses",
                     r"$I(x) = -\log P(x)$,   $H(P) = \mathbb{E}_{x\sim P}[I(x)]$,   "
                     r"$D_{KL}(P\|Q) = \mathbb{E}_{x\sim P}[\log P(x) - \log Q(x)]$,   "
                     r"$C = -(y \log\hat{y} + (1-y)\log(1-\hat{y}))$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # (0,0) self-information
    ax = fig.add_subplot(gs[0, 0])
    p = np.linspace(0.005, 1, 600)
    ax.plot(p, self_info(p), color=BLUE, linewidth=2.2, label="nats (ln)")
    ax.plot(p, self_info(p) / np.log(2), color=ORANGE, linewidth=1.6, linestyle="--", label="bits (log₂)")
    for pv in (1, 0.5, 0.1, 0.01):
        ax.scatter([pv], [self_info(pv)], color=BLUE, s=40, zorder=4)
        ax.annotate(f"I({pv:g}) = {abs(self_info(pv)):.3f}", (pv, self_info(pv)), textcoords="offset points",
                    xytext=(-6, -20) if pv == 0.5 else ((10, 4) if pv != 1 else (-8, 10)),
                    ha="left" if pv not in (1, 0.5) else "right",
                    fontsize=9.5, color=INK)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.2, 7.8)
    ax.set_xlabel("P(x)")
    ax.set_ylabel("self-information I(x)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Self-information: −log P(x)")

    # (0,1) binary entropy
    ax = fig.add_subplot(gs[0, 1])
    pp = np.linspace(0.001, 0.999, 1000)
    H = binary_entropy(pp)
    ax.plot(pp, H, color=BLUE, linewidth=2.2)
    ax.fill_between(pp, H, color=BLUE_LIGHT, alpha=0.5)
    for pv, off in ((0.5, (0, 10)), (0.9, (12, 6)), (0.00001, (14, 8))):
        hv = binary_entropy(pv)
        ax.scatter([pv], [hv], color=ORANGE, s=40, zorder=4)
        label = f"H({pv:g}) = {hv:.3f}" if pv > 0.001 else f"H(0.00001) = {hv:.6f}"
        ax.annotate(label, (pv, hv), textcoords="offset points", xytext=off,
                    ha="center" if pv == 0.5 else "left", fontsize=9.5, color=INK)
    ax.text(0.5, 0.33, "fair coin: most uncertain\n= ln 2 nats = 1 bit", ha="center", fontsize=10, color=INK2)
    ax.set_ylim(0, 0.82)
    ax.set_xlabel("p = P(heads)")
    ax.set_ylabel("H (nats)")
    ax.grid(True)
    ax.set_title("Shannon entropy of a Bernoulli trial")

    panel(fig.add_subplot(gs[0, 2]), "Nats, bits and expected surprise", [
        "Certain event: I(1) = 0; rarer = more info.",
        f"Two independent heads: {self_info(0.5):.3f} + {self_info(0.5):.3f}",
        f"= {2 * self_info(0.5):.3f} nats = I(0.25) = 2 bits (adds up).",
        "",
        "nats: natural log, typical in ML",
        "bits (shannons): log₂, typical in CS;",
        r"1 nat = $1/\ln 2$ ≈ 1.443 bits",
        "",
        "Entropy = average self-information:",
        "low for a near-certain coin, highest",
        "for the uniform (fair) one.",
        "Differential entropy: the same for a PDF.",
    ], fontsize=12, spacing=0.078)

    # (1,0) cross-entropy cost
    ax = fig.add_subplot(gs[1, 0])
    a = np.linspace(0.001, 0.999, 800)
    ax.plot(a, cross_entropy(1, a), color=BLUE, linewidth=2.2, label="true label y = 1")
    ax.plot(a, cross_entropy(0, a), color=ORANGE, linewidth=1.6, linestyle="--", label="true label y = 0")
    for av in (0.9997, 0.9, 0.6, 0.3):
        ax.scatter([av], [cross_entropy(1, av)], color=BLUE, s=40, zorder=4)
    table = "\n".join(f"ŷ = {av:<6g} → C = {cross_entropy(1, av):.4f}" for av in (0.9997, 0.9, 0.6, 0.3))
    ax.text(0.2, 3.15, "y = 1, the notebook's four calls:\n" + table, va="top", fontsize=9,
            family="monospace", color=INK)
    ax.set_ylim(0, 4)
    ax.set_xlabel("predicted probability ŷ")
    ax.set_ylabel("cost C")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=10)
    ax.set_title("Cross-entropy cost of a binary classifier")

    # (1,1) KL divergence between Bernoullis, built on the notebook's H(0.9)
    ax = fig.add_subplot(gs[1, 1])
    p_true = 0.9
    q = np.linspace(0.01, 0.99, 500)
    h_p = binary_entropy(p_true)
    cross = -(p_true * np.log(q) + (1 - p_true) * np.log(1 - q))
    kl = cross - h_p
    ax.plot(q, cross, color=BLUE, linewidth=2.2, label="cross-entropy H(P, Q)")
    ax.plot(q, kl, color=ORANGE, linewidth=2.0, label=r"$D_{KL}(P\|Q)$")
    ax.axhline(h_p, color=INK2, linestyle=":", linewidth=1.6, label=f"H(P) = H(0.9) = {h_p:.3f}")
    ax.plot([p_true, p_true], [0, 0.6], color=MUTED, linewidth=1)
    q_half = 0.5
    ax.annotate(r"Q = fair coin: $D_{KL}$ = " + f"{binary_kl(p_true, q_half):.3f}", (q_half, binary_kl(p_true, q_half)),
                xytext=(0.66, 1.25), ha="center", fontsize=9.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.text(p_true, 0.66, r"Q = P: $D_{KL}$ = 0", ha="center", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("q = Q(heads), model of a coin with P(heads) = 0.9")
    ax.set_ylabel("nats")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("KL divergence: the extra cost of using Q for P")

    panel(fig.add_subplot(gs[1, 2]), "From KL divergence to the cost", [
        "KL divergence compares two distributions",
        "over the same x: ≥ 0, and 0 only if Q = P.",
        "",
        r"Cross-entropy: $H(P, Q) = H(P) + D_{KL}(P\|Q)$",
        "",
        "For one labelled example the true P is",
        "certain (H(P) = 0), so the cross-entropy",
        "cost C is exactly the KL divergence:",
        "confident and right ≈ 0, confident and",
        f"wrong explodes (ŷ = 0.3 costs {cross_entropy(1, 0.3):.3f}).",
        "",
        "(KL curve: illustration with P = Bernoulli(0.9).)",
    ], fontsize=12, spacing=0.078)
    save(fig, name=f"{STEM}_seg3.png")


def binary_kl(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


if __name__ == "__main__":
    draws = replay_notebook_rng()
    sheet_seg1_intro_to_probability(draws)
    sheet_seg2_distributions(draws)
    sheet_seg3_information_theory()
