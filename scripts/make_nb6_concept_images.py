#!/usr/bin/env python3
"""Concept sheets for notebooks/6-statistics.ipynb, one per segment.

Segment 1 (frequentist statistics), Segment 2 (regression) and Segment 3 (Bayesian statistics).
Every number is computed here from the same data, parameters and random seed the notebook uses
(np.random.seed(42) and the notebook's exact draw order, the iris data bundled with scikit-learn,
the penguins csv already in the local seaborn cache) and is checked against the notebook's
printed outputs with asserts. Two datasets are not available offline (seaborn's 'exercise' and
'titanic'): for those the sheet uses the values the notebook printed and says so.
Images are written to notebooks/img/6-statistics_seg<N>.png.

Run from anywhere:  python scripts/make_nb6_concept_images.py
"""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter
import scipy.stats as st
from sklearn.datasets import load_iris

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, panel)
from viz_style import save as _save

NOTEBOOKS = REPO / "notebooks"
save = partial(_save, folder=NOTEBOOKS / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"


def close(a, b, rel=1e-9, abs_=1e-12):
    return abs(a - b) <= max(rel * abs(b), abs_)


# ================================================================ data
def sample_mean_calculator(input_dist, sample_size, n_samples):
    """The notebook's helper, unchanged (it draws from np.random's global state)."""
    return [np.random.choice(input_dist, size=sample_size, replace=False).mean()
            for _ in range(n_samples)]


def notebook_random_draws():
    """Replay every np.random call of Segment 1 in the notebook's order after np.random.seed(42)."""
    np.random.seed(42)
    out = {}
    x = st.skewnorm.rvs(10, size=1000)                                   # central tendency
    assert close(x.mean(), 0.7719211854654121) and close(np.median(x), 0.651420257427958)
    assert close(x.var(), 0.35528419278021617) and close(x.std(), 0.5960572059628305)
    assert close(st.sem(x, ddof=0), 0.018848983865986414)
    out["skewed_1000"] = x
    normal = np.random.normal(size=10000)                                # Gaussian
    assert close(normal.mean(), -0.01626305454909193) and close(normal.std(), 1.0039756140435032)
    x_sample = np.random.choice(normal, size=10, replace=False)          # CLT
    assert close(x_sample.mean(), -0.35958129882198664)
    for size, count in [(10, 10), (10, 1000), (100, 1000), (1000, 1000)]:
        sample_mean_calculator(normal, size, count)
    s = st.skewnorm.rvs(10, size=10000)                                  # skewed population
    out["skewed_pop"] = s
    out["means_s10"] = np.array(sample_mean_calculator(s, 10, 1000))
    out["means_s1000"] = np.array(sample_mean_calculator(s, 1000, 1000))
    m = np.concatenate((np.random.normal(size=5000), np.random.normal(loc=4.0, size=5000)))
    sample_mean_calculator(m, 1000, 1000)
    u = np.random.uniform(size=10000)
    sample_mean_calculator(u, 1000, 1000)
    exam_a = np.random.normal(60, 10, 10000)                             # z-scores
    assert close((85 - exam_a.mean()) / exam_a.std(), 2.5063103401632336)
    assert (exam_a > 85).sum() == 69 and close(np.percentile(exam_a, 99), 83.35787058294662)
    exam_b = np.random.normal(90, 2, 10000)
    assert close((85 - exam_b.mean()) / exam_b.std(), -2.479852596293303)
    assert (exam_b > 85).sum() == 9933 and close(np.percentile(exam_b, 1), 85.3781445632019)
    out["exam_a"], out["exam_b"] = exam_a, exam_b
    return out


def penguins():
    """sns.load_dataset('penguins').dropna(), read from the local seaborn cache (no download)."""
    import seaborn as sns
    path = Path(sns.get_data_home()) / "penguins.csv"
    assert path.exists(), f"penguins.csv is not in the local seaborn cache ({path})"
    data = pd.read_csv(path)
    data["sex"] = data["sex"].str.title()      # what sns.load_dataset does for penguins
    data = data.dropna()
    assert np.unique(data.species, return_counts=True)[1].tolist() == [146, 68, 119]
    return data


def iris():
    """The iris data bundled with scikit-learn, in the seaborn column layout."""
    bunch = load_iris()
    frame = pd.DataFrame(bunch.data, columns=["sepal_length", "sepal_width",
                                              "petal_length", "petal_width"])
    frame["species"] = np.array(bunch.target_names)[bunch.target]
    return frame


# ================================================================ Segment 1
def sheet_seg1_frequentist():
    draws = notebook_random_draws()
    x = draws["skewed_1000"]
    xbar, med, sigma = x.mean(), np.median(x), x.std()
    se = sigma / np.sqrt(x.size)

    # z-scores and p-values
    p_outside = 2 * st.norm.cdf(-2.5)
    assert close(p_outside, 0.01241933065155229, rel=1e-12)
    z05, z01 = st.norm.ppf(0.975), st.norm.ppf(0.995)
    assert close(z05, 1.959963984540054) and close(z01, 2.5758293035489004)

    # t-tests
    yeast4 = st.ttest_1samp([48, 50, 54, 60], 50)
    assert close(yeast4.statistic, 1.1338934190276817) and close(yeast4.pvalue, 0.3392540508564543)
    yeast8 = np.array([48, 50, 54, 60, 49, 55, 59, 62])
    yeast8_t = st.ttest_1samp(yeast8, 50)
    assert close(yeast8_t.statistic, 2.448085108904054) and close(yeast8_t.pvalue, 0.0442312968303811)

    pg = penguins()
    adelie = pg[pg.species == "Adelie"]
    f = adelie[adelie.sex == "Female"]["body_mass_g"].to_numpy() / 1000
    m = adelie[adelie.sex == "Male"]["body_mass_g"].to_numpy() / 1000
    assert close(f.mean(), 3.368835616438356) and close(m.mean(), 4.043493150684932)
    welch = st.ttest_ind(f, m, equal_var=False)
    assert close(welch.statistic, -13.126285923485879) and close(welch.pvalue, 6.402319748031518e-26, rel=1e-6)
    ff = adelie[(adelie.sex == "Female") & (adelie.island == "Dream")]["flipper_length_mm"].to_numpy()
    mf = adelie[(adelie.sex == "Male") & (adelie.island == "Dream")]["flipper_length_mm"].to_numpy()
    flip = st.ttest_ind(ff, mf, equal_var=False)
    assert close(flip.pvalue, 0.017898599613024185, rel=1e-6)
    islands = [adelie[(adelie.sex == "Female") & (adelie.island == isl)]["body_mass_g"].to_numpy() / 1000
               for isl in ("Torgersen", "Biscoe", "Dream")]
    anova = st.f_oneway(*islands)
    assert close(anova.statistic, 0.22627752438542714, rel=1e-6) and close(anova.pvalue, 0.7980777848719299, rel=1e-6)

    # paired: the walking, no-fat dieters (seaborn's 'exercise' data is not available offline,
    # so these are the arrays the notebook printed)
    min1 = np.array([84, 103, 92, 97, 102])
    min15 = np.array([86, 109, 96, 98, 104])
    paired = st.ttest_rel(min15, min1)
    indep = st.ttest_ind(min15, min1, equal_var=False)
    assert close(paired.statistic, 3.3541019662496847) and close(paired.pvalue, 0.02846020325433834)
    assert close(indep.statistic, 0.5729114862838532) and close(indep.pvalue, 0.5826164805627767)

    # confidence interval (the notebook uses the ddof=0 standard deviation here)
    ci_err = 1.96 * yeast8.std() / np.sqrt(yeast8.size)
    assert close(ci_err, 3.4637403013216796)

    # correlation
    ir = iris()
    sl, pl, sw = ir.sepal_length.to_numpy(), ir.petal_length.to_numpy(), ir.sepal_width.to_numpy()
    cov = np.mean((sl - sl.mean()) * (pl - pl.mean()))
    r = cov / (sl.std() * pl.std())
    assert close(cov, 1.2658200000000006) and close(r, 0.8717537758865835)
    t_r = r * np.sqrt((sl.size - 2) / (1 - r ** 2))
    p_r = 2 * st.t.cdf(-abs(t_r), sl.size - 2)
    assert close(t_r, 21.646019345759807) and close(p_r, 1.0386674194495906e-47, rel=1e-6)
    r_w = st.pearsonr(sl, sw)
    assert close(r_w.pvalue, 0.15189826071144746, rel=1e-6) and close(r ** 2, 0.7599546457725157)

    fig = new_figure("Frequentist statistics: how surprising are the data if nothing is going on?",
                     "Standardize an estimate, (estimate − null value) / standard error, read its tail "
                     "probability p from the z or t distribution, and call p < α = .05 significant.")
    gs = grid(fig, top=0.84, wspace=0.3)

    # --- [0,0] describing one skewed sample
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(x, bins=30, color=BLUE_LIGHT, edgecolor="white", linewidth=0.6, zorder=2)
    ax.axvline(xbar, color=ORANGE, linewidth=2.4, label=f"mean x̄ = {xbar:.3f}", zorder=3)
    ax.axvline(med, color=AQUA, linewidth=2.4, label=f"median = {med:.3f}", zorder=3)
    for k in (-1, 1):
        ax.axvline(xbar + k * sigma, color=INK2, linestyle="--", linewidth=1.3, zorder=3,
                   label=f"x̄ ± σ  (σ = {sigma:.3f})" if k == 1 else None)
    ax.text(0.97, 0.52, f"σ² = {x.var():.3f}\nSE = σ/√n = {se:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color=INK2)
    ax.set_xlabel("value (skew-normal, a = 10, n = 1000)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Skew pulls the mean right of the median")

    # --- [0,1] central limit theorem on the skewed population
    ax = fig.add_subplot(gs[0, 1])
    s = draws["skewed_pop"]
    bins_pop = np.linspace(-0.6, 3.6, 60)
    series = [(s, bins_pop, MUTED, 0.35, f"population (n = 10,000), sd {s.std():.2f}"),
              (draws["means_s10"], np.linspace(-0.6, 3.6, 70), BLUE, 0.75,
               f"1000 means of n = 10, sd {draws['means_s10'].std():.3f}"),
              (draws["means_s1000"], np.linspace(0.70, 0.88, 30), ORANGE, 0.85,
               f"1000 means of n = 1000, sd {draws['means_s1000'].std():.4f}")]
    for values, bins, color, alpha, label in series:
        counts, edges = np.histogram(values, bins=bins)
        ax.stairs(counts / counts.max(), edges, fill=True, color=color, alpha=alpha, label=label)
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("value / sample mean")
    ax.set_ylabel("count (each scaled to peak 1)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y")
    ax.set_title("CLT: sample means are normal, sd = σ/√n")

    # --- [0,2] z-scores and p-values
    ax = fig.add_subplot(gs[0, 2])
    zz = np.linspace(-4, 4, 800)
    dens = st.norm.pdf(zz)
    ax.plot(zz, dens, color=INK2, linewidth=1.8)
    ax.fill_between(zz, dens, where=np.abs(zz) >= 2.5, color=ORANGE, alpha=0.85, linewidth=0,
                    label=f"|z| ≥ 2.5: p = {p_outside:.4f}")
    for zc, style, lab in [(z05, "--", f"α = .05: |z| > {z05:.2f}"), (z01, ":", f"α = .01: |z| > {z01:.3f}")]:
        for sgn in (-1, 1):
            ax.vlines(sgn * zc, 0, 0.16, color=BLUE, linestyle=style, linewidth=1.4,
                      label=lab if sgn == 1 else None)
    ax.annotate("85% when\nμ = 60, σ = 10:\nz = +2.5;\n69 of 10,000\nbeat you", xy=(2.5, 0.025),
                xytext=(3.95, 0.40), fontsize=9.5, color=INK, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.annotate("85% when\nμ = 90, σ = 2:\nz = −2.5;\n9,933 of 10,000\nbeat you", xy=(-2.5, 0.025),
                xytext=(-3.95, 0.40), fontsize=9.5, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(0, 0.72)
    ax.set_xlim(-4, 4)
    ax.set_xlabel("z = (x − μ) / σ")
    ax.set_ylabel("standard normal density")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(axis="y")
    ax.set_title("z-score → two-tailed p-value")

    # --- [1,0] paired vs independent
    ax = fig.add_subplot(gs[1, 0])
    label_y = {}
    for a, b in sorted(zip(min1, min15), key=lambda pair: pair[1]):
        ax.plot([0, 1], [a, b], color=BLUE, linewidth=1.6, marker="o", markersize=6, zorder=3)
        y_text = max(b, max(label_y.values(), default=-np.inf) + 3.4)
        label_y[(a, b)] = y_text
        ax.text(1.08, y_text, f"+{b - a}", va="center", fontsize=10, color=AQUA, weight="bold")
    ax.plot([0, 1], [min1.mean(), min15.mean()], color=ORANGE, linewidth=2.6, marker="D",
            markersize=7, zorder=4, label=f"means {min1.mean():.1f} → {min15.mean():.1f}")
    ax.text(0.5, 74.5, f"paired: t({paired.df}) = {paired.statistic:.2f}, p = {paired.pvalue:.3f}\n"
                       f"independent (Welch): t = {indep.statistic:.2f}, p = {indep.pvalue:.2f}",
            ha="center", fontsize=10, color=INK)
    ax.set_xticks([0, 1], ["after 1 min", "after 15 min"])
    ax.set_xlim(-0.25, 1.3)
    ax.set_ylim(72, 112)
    ax.set_ylabel("pulse (bpm)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y")
    ax.set_title("Paired t: test the differences d = (2, 6, 4, 1, 2)")

    # --- [1,1] all the p-values of the segment, with alpha and Bonferroni
    ax = fig.add_subplot(gs[1, 1].subgridspec(1, 2, width_ratios=[0.12, 1], wspace=0)[1])
    tests = [("exam, z = 2.5", p_outside),
             ("yeast n = 4, t", yeast4.pvalue),
             ("yeast n = 8, t", yeast8_t.pvalue),
             ("pulse, paired t", paired.pvalue),
             ("pulse, unpaired", indep.pvalue),
             ("penguin mass ♀/♂", welch.pvalue),
             ("Dream flipper ♀/♂", flip.pvalue),
             ("♀ mass, ANOVA", anova.pvalue),
             ("iris r, petal", p_r),
             ("iris r, sep. width", r_w.pvalue)]
    cap = 3.6
    alpha, bonf = 0.05, 0.05 / len(tests)
    for k, (name, p) in enumerate(tests):
        score = -np.log10(p)
        color = ORANGE if p < bonf else (BLUE if p < alpha else MUTED)
        ax.plot([0, min(score, cap)], [k, k], color=color, linewidth=2.2, zorder=2)
        ax.plot(min(score, cap), k, "o", color=color, markersize=7, zorder=3)
        label = f"p = {p:.1e} ▸" if score > cap else f"p = {p:.3f}"
        ax.text(min(score, cap) + 0.08, k, label, va="center", fontsize=9, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    ax.axvline(-np.log10(alpha), color=BLUE, linestyle="--", linewidth=1.3)
    ax.axvline(-np.log10(bonf), color=ORANGE, linestyle="--", linewidth=1.3)
    ax.text(-np.log10(alpha) - 0.05, -1.05, "α = .05", ha="right", fontsize=9.5, color=BLUE)
    ax.text(-np.log10(bonf) + 0.05, -1.05, "α/m = .005", ha="left", fontsize=9.5, color=ORANGE)
    ax.set_yticks(range(len(tests)), [t[0] for t in tests], fontsize=9.5)
    ax.set_ylim(len(tests) - 0.4, -1.5)
    ax.set_xlim(0, cap + 1.4)
    ax.set_xlabel("−log₁₀ p   (right = more surprising)")
    ax.grid(axis="x")
    ax.set_title("Ten tests: which survive Bonferroni (m = 10)?")

    # --- [1,2] the toolbox
    panel(fig.add_subplot(gs[1, 2]), "One recipe, many tests", [
        r"$z = (x-\mu)/\sigma$,   $t = (\bar{x}-\mu_0)/s_{\bar{x}}$,  df = n − 1",
        r"Welch: $t = (\bar{x}-\bar{y})\,/\,\sqrt{s_x^2/n_x + s_y^2/n_y}$",
        "",
        "Paired: a one-sample t on d = y − x",
        r"Correlation: $t = r\sqrt{(n-2)/(1-r^2)}$,  df = n − 2",
        "",
        f"95% CI = x̄ ± 1.96 s/√n: yeast {yeast8.mean() - ci_err:.1f} to {yeast8.mean() + ci_err:.1f} L,",
        "which excludes the 50 L baseline.",
        f"Iris: r = {r:.3f}; r² = {r ** 2:.2f} of petal-length variance",
        "is explained (not caused) by sepal length.",
        "Bonferroni: test each of m tests at α/m.",
    ], fontsize=11.5, spacing=0.08)
    save(fig, name="6-statistics_seg1.png")


# ================================================================ Segment 2
def sheet_seg2_regression():
    ir = iris()
    x, y = ir.sepal_length.to_numpy(), ir.petal_length.to_numpy()
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    beta1 = cov / np.var(x)
    beta0 = y.mean() - beta1 * x.mean()
    assert close(beta1, 1.8584329782548417) and close(beta0, -7.101443369602459)
    y55 = beta0 + beta1 * 5.5
    assert close(y55, 3.1199380107991708)
    r2_single = np.corrcoef(x, y)[0, 1] ** 2

    # clinical-trial example
    xd = np.array([0, 1, 2, 3, 4, 5, 6, 7.])
    yd = np.array([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])
    cm = np.cov(xd, yd)
    b1d = cm[0, 1] / cm[0, 0]
    b0d = yd.mean() - b1d * xd.mean()
    assert close(b1d, -0.4692857142857143) and close(b0d, 1.76)
    assert close(b0d + b1d * 4.5, -0.35178571428571437)

    # penguin exercise
    pg = penguins()
    fem = pg[(pg.species == "Adelie") & (pg.sex == "Female")]
    xp, yp = fem.flipper_length_mm.to_numpy(), fem.body_mass_g.to_numpy() / 1000
    cp = np.cov(xp, yp)
    b1p = cp[0, 1] / cp[0, 0]
    b0p = yp.mean() - b1p * xp.mean()
    assert close(b1p, 0.012659083726357767) and close(b0p + b1p * 197, 3.485368551563184)

    # OLS by the normal equations
    xg, yg = np.array([1, 2, 3, 4.]), np.array([6, 5, 7, 10.])
    A = np.array([[8, 20], [20, 60]])
    zvec = np.array([56, 154])
    X = np.column_stack([np.ones_like(xg), xg])
    assert np.allclose(2 * X.T @ X, A) and np.allclose(2 * X.T @ yg, zvec)
    w = np.linalg.inv(A) @ zvec
    assert np.allclose(np.linalg.inv(A), [[0.75, -0.25], [-0.25, 0.1]]) and np.allclose(w, [3.5, 1.4])
    yhat = X @ w
    assert np.allclose(yhat, [4.9, 6.3, 7.7, 9.1])
    sse_min = ((yhat - yg) ** 2).sum()

    def sse(b0, b1):
        return 4 * b0 ** 2 + 30 * b1 ** 2 + 20 * b0 * b1 - 56 * b0 - 154 * b1 + 210

    assert close(sse(*w), sse_min)

    # dummy-variable OLS on iris (statsmodels OLS reproduced with numpy least squares)
    setosa = (ir.species == "setosa").astype(float).to_numpy()
    versicolor = (ir.species == "versicolor").astype(float).to_numpy()
    Xd = np.column_stack([np.ones_like(x), x, setosa, versicolor])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    assert np.allclose(beta, [1.387660, 0.632110, -3.090002, -0.879864], atol=5e-7)
    resid = y - Xd @ beta
    r2_dummy = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert f"{r2_dummy:.3f}" == "0.975"
    y_versi = beta @ [1, 5.5, 0, 1]
    y_virg = beta @ [1, 5.5, 0, 0]
    assert close(y_versi, 3.984400080089945, rel=1e-6)

    # logistic regression: seaborn's 'titanic' data is not available offline, so the
    # coefficients are the ones the notebook printed (statsmodels Logit, 714 complete rows)
    lb = dict(const=-1.326394, First=2.580625, Second=1.270826, female=2.522781, age=-0.036985)

    def expit(v):
        return 1 / (1 + np.exp(-v))

    rose = lb["const"] + lb["First"] + lb["female"] + lb["age"] * 17
    jack = lb["const"] + lb["age"] * 20
    assert abs(rose - 3.148263137776477) < 1e-5 and abs(jack - -2.066098896322207) < 1e-5
    assert f"{expit(rose):.3f}" == "0.959" and f"{expit(jack):.3f}" == "0.112"
    assert close(np.log(0.1 / 0.9), -2.197224577336219)
    assert close(expit(4.59512), 0.9900000014836675)

    fig = new_figure("Regression: pick the β that minimize the sum of squared residuals",
                     "ŷ = β₀ + β₁x₁ + … + βₘxₘ.  One predictor: β₁ = cov(x, y)/σ²ₓ, β₀ = ȳ − β₁x̄.  "
                     "Several: set ∂SSE/∂β = 0 and solve.  Binary y: p = 1/(1 + e^(−ŷ)).")
    gs = grid(fig, top=0.84, wspace=0.3)

    # --- [0,0] simple linear least squares on iris
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x, y, s=16, color=BLUE, alpha=0.6, edgecolor="none", zorder=2)
    xl = np.linspace(4, 8, 100)
    ax.plot(xl, beta0 + beta1 * xl, color=ORANGE, linewidth=2.4, zorder=3,
            label=f"ŷ = {beta0:.2f} + {beta1:.3f}x   (r² = {r2_single:.2f})")
    ax.plot(5.5, y55, "o", color=INK, markersize=8, zorder=4)
    ax.annotate(f"sepal 5.5 cm → petal {y55:.2f} cm", (5.5, y55), textcoords="offset points",
                xytext=(10, -22), fontsize=10)
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("petal length (cm)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("β₁ = cov(x, y) / σ²ₓ = 1.266 / 0.681")

    # --- [0,1] OLS residuals on the four GAN points
    ax = fig.add_subplot(gs[0, 1])
    xl = np.linspace(0.7, 4.3, 50)
    ax.plot(xl, w[0] + w[1] * xl, color=ORANGE, linewidth=2.4, label="ŷ = 3.5 + 1.4x", zorder=2)
    for xi, yi, hi in zip(xg, yg, yhat):
        ax.plot([xi, xi], [yi, hi], color=INK2, linewidth=1.8, zorder=1)
        ax.text(xi + 0.08, (yi + hi) / 2, f"{yi - hi:+.1f}", va="center", fontsize=10, color=INK2)
    ax.scatter(xg, yg, s=60, color=BLUE, zorder=3, label="data (x, y)")
    ax.text(0.97, 0.06, f"SSE = 1.1² + 1.3² + 0.7² + 0.9² = {sse_min:.1f}", transform=ax.transAxes,
            ha="right", fontsize=10.5)
    ax.set_xlabel("number of convolutional layers")
    ax.set_ylabel("image realism (out of 10)")
    ax.set_ylim(3.5, 10.8)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Residuals y − ŷ: OLS squares and sums them")

    # --- [0,2] the SSE bowl and the normal equations
    ax = fig.add_subplot(gs[0, 2])
    b0g, b1g = np.meshgrid(np.linspace(0, 7, 300), np.linspace(0, 2.8, 300))
    levels = [4.5, 5, 6, 8, 12, 20, 35, 60, 100, 200]
    ax.contourf(b0g, b1g, sse(b0g, b1g), levels=[4.2] + levels + [1000], cmap=SEQ, alpha=0.75)
    ax.contour(b0g, b1g, sse(b0g, b1g), levels=levels, colors=INK2, linewidths=0.6)
    b0_line = np.linspace(0, 7, 50)
    ax.plot(b0_line, (56 - 8 * b0_line) / 20, color=ORANGE, linewidth=2, label="∂S/∂β₀ = 0: 8β₀ + 20β₁ = 56")
    ax.plot(b0_line, (154 - 20 * b0_line) / 60, color=AQUA, linewidth=2, label="∂S/∂β₁ = 0: 20β₀ + 60β₁ = 154")
    ax.plot(*w, "*", color=INK, markersize=15, zorder=4)
    ax.annotate(f"w = A⁻¹z = (3.5, 1.4)\nSSE = {sse_min:.1f}", w, textcoords="offset points",
                xytext=(14, 8), fontsize=10, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2.8)
    ax.set_xlabel("β₀ (intercept)")
    ax.set_ylabel("β₁ (slope)")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.set_title("SSE bowl: both partials vanish at the minimum")

    # --- [1,0] dummy variables: one slope, three intercepts
    ax = fig.add_subplot(gs[1, 0])
    xl = np.linspace(4, 8, 100)
    groups = [("setosa", beta[2], BLUE), ("versicolor", beta[3], ORANGE), ("virginica", 0.0, AQUA)]
    for name, shift, color in groups:
        mask = (ir.species == name).to_numpy()
        ax.scatter(x[mask], y[mask], s=14, color=color, alpha=0.55, edgecolor="none")
        ax.plot(xl, beta[0] + beta[1] * xl + shift, color=color, linewidth=2.2,
                label=f"{name}: {beta[0] + shift:+.2f} + {beta[1]:.3f}x")
    ax.plot([5.5, 5.5], [y_versi, y_virg], "o", color=INK, markersize=7, zorder=4)
    for value in (y_versi, y_virg):
        ax.annotate(f"{value:.2f}", (5.5, value), textcoords="offset points", xytext=(-9, 0),
                    ha="right", va="center", fontsize=10, weight="bold",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1))
    ax.text(0.97, 0.04, f"R² = {r2_dummy:.3f}  ({r2_single:.2f} without species)\n"
                        "● predictions at sepal 5.5 cm",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, linespacing=1.4)
    ax.set_xlabel("sepal length (cm)")
    ax.set_ylabel("petal length (cm)")
    ax.set_ylim(0, 8.2)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.grid(True)
    ax.set_title("Dummy (0/1) species columns shift the intercept")

    # --- [1,1] logistic regression on the titanic coefficients
    ax = fig.add_subplot(gs[1, 1])
    ages = np.linspace(0, 80, 200)
    combos = [("female, 1st class", lb["First"] + lb["female"], ORANGE),
              ("female, 3rd class", lb["female"], ORANGE_LIGHT),
              ("male, 1st class", lb["First"], BLUE),
              ("male, 3rd class", 0.0, BLUE_LIGHT)]
    for name, shift, color in combos:
        ax.plot(ages, expit(lb["const"] + shift + lb["age"] * ages), color=color, linewidth=2.3, label=name)
    ax.plot(17, expit(rose), "o", color=INK, markersize=8, zorder=4)
    ax.annotate(f"Rose, 17: ŷ = {rose:.2f} → p = {expit(rose):.1%}", (17, expit(rose)),
                textcoords="offset points", xytext=(12, -16), fontsize=9.5)
    ax.plot(20, expit(jack), "o", color=INK, markersize=8, zorder=4)
    ax.annotate(f"Jack, 20: ŷ = {jack:.2f} → p = {expit(jack):.1%}", (20, expit(jack)),
                textcoords="offset points", xytext=(10, 12), fontsize=9.5)
    ax.text(0.98, 0.46, "coefficients as printed by the notebook\n(titanic data not available offline)",
            transform=ax.transAxes, ha="right", fontsize=8.8, color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("P(survived) = expit(ŷ)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.86), fontsize=8.8)
    ax.grid(True)
    ax.set_title("Logistic: the linear ŷ becomes a probability")

    # --- [1,2] card
    panel(fig.add_subplot(gs[1, 2]), "The regression toolkit", [
        "Expand S(β₀, β₁) = Σ(β₀ + β₁xᵢ − yᵢ)², set both",
        "partials to 0: the normal equations Aw = z,",
        "`[[8, 20], [20, 60]] w = [56, 154]`",
        "w = A⁻¹z = (3.5, 1.4); same as (XᵀX)⁻¹Xᵀy.",
        "",
        f"Drug trial: ŷ = {b0d:.2f} − {-b1d:.3f}x;  4.5 mL → {b0d + b1d * 4.5:.2f}",
        f"Penguins ♀: 197 mm flipper → {b0p + b1p * 197:.3f} kg",
        "",
        r"logit $= \ln\,(p/(1-p))$ maps (0, 1) to (−∞, ∞),",
        r"expit $= 1/(1+e^{-x})$ maps it back;",
        "e.g. logit(0.1) = −2.20, expit(4.595) = 0.99.",
    ], fontsize=12, spacing=0.083)
    save(fig, name="6-statistics_seg2.png")


# ================================================================ Segment 3
def sheet_seg3_bayes():
    p_lie = 1 / 36
    p_yes_given_exploded = 35 / 36
    prior = 1 / 1000
    py = p_yes_given_exploded * prior + p_lie * (1 - prior)
    assert close(py, 0.02872222222222222)
    posterior = prior * p_yes_given_exploded / py
    # the notebook's code cell uses px = .001 and py_givenx = 0.972 (35/36 rounded)
    notebook_value = 0.001 * 0.972 / py
    assert close(notebook_value, 0.033841392649903296)
    p_not_given_yes = p_lie * (1 - prior) / py

    def post(prior_, lie=p_lie):
        return prior_ * (1 - lie) / ((1 - lie) * prior_ + lie * (1 - prior_))

    fig = new_figure("Bayes' theorem: a 'significant' alarm about a rare event is still probably wrong",
                     "P(x | y) = P(x) P(y | x) / P(y),  P(y) = Σₓ P(y | x) P(x).   The xkcd neutrino "
                     "detector lies when two dice both show 6: P(YES | sun fine) = 1/36 < .05.")
    gs = grid(fig, top=0.84, wspace=0.3)

    # --- [0,0] the frequentist test: 36 dice outcomes
    ax = fig.add_subplot(gs[0, 0])
    board = np.zeros((6, 6))
    board[5, 5] = 1
    ax.pcolormesh(np.arange(7), np.arange(7), board, cmap=SEQ, vmin=0, vmax=1.6,
                  edgecolors="white", linewidth=2)
    ax.add_patch(Rectangle((5, 5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=3))
    ax.text(5.5, 5.5, "lie", ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(6, 0)
    ax.set_xticks(np.arange(6) + 0.5, range(1, 7))
    ax.set_yticks(np.arange(6) + 0.5, range(1, 7))
    ax.set_xlabel(f"die 1      (P(lie) = 1/36 = {p_lie:.4f} < α = .05)")
    ax.set_ylabel("die 2")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title("Frequentist: p = 1/36, reject 'sun is fine'")

    # --- [0,1] natural frequencies over 1,000,000 days
    ax = fig.add_subplot(gs[0, 1])
    days = 1_000_000
    yes_exploded = days * prior * p_yes_given_exploded
    yes_fine = days * (1 - prior) * p_lie
    bars_ = [("exploded\n& YES", yes_exploded, ORANGE), ("sun fine\n& YES (lie)", yes_fine, BLUE)]
    for k, (name, val, color) in enumerate(bars_):
        ax.bar(k, val, width=0.6, color=color, zorder=2)
        ax.text(k, val + 500, f"{val:,.0f} days", ha="center", fontsize=10.5)
    ax.set_xticks([0, 1], [b[0] for b in bars_])
    ax.set_ylim(0, 34000)
    ax.set_ylabel("days with a YES (of 1,000,000)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.text(0.04, 0.55, f"P(exploded | YES) =\n{yes_exploded:,.0f} / {yes_exploded + yes_fine:,.0f} = {posterior:.1%}",
            transform=ax.transAxes, ha="left", fontsize=11, color=INK)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Prior 1/1000: most YES days are lies")

    # --- [0,2] posterior vs prior
    ax = fig.add_subplot(gs[0, 2])
    priors = np.logspace(-6, -0.05, 300)
    for lie, color, name in [(1 / 6, MUTED, "one die (lie 1/6)"), (p_lie, ORANGE, "two dice (lie 1/36)"),
                             (1 / 6 ** 4, BLUE, "four dice (lie 1/1296)")]:
        ax.plot(priors, post(priors, lie), color=color, linewidth=2.2, label=name)
    ax.plot(priors, priors, color=INK2, linestyle=":", linewidth=1.2, label="no evidence: posterior = prior")
    ax.plot(prior, posterior, "o", color=INK, markersize=8, zorder=4)
    ax.annotate(f"prior 1/1000 → {posterior:.1%}", (prior, posterior), textcoords="offset points",
                xytext=(10, -14), fontsize=10)
    ax.plot(1e-4, post(1e-4), "o", color=INK, markersize=6, zorder=4)
    ax.annotate(f"prior 1/10,000 → {post(1e-4):.2%}", (1e-4, post(1e-4)), textcoords="offset points",
                xytext=(10, -14), fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("prior P(exploded)")
    ax.set_ylabel("posterior P(exploded | YES)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)
    ax.set_title("One YES multiplies the prior odds by 35")

    # --- [1,0] two probabilities that are easy to confuse
    ax = fig.add_subplot(gs[1, 0].subgridspec(1, 2, width_ratios=[0.3, 1], wspace=0)[1])
    vals = [("P(YES | fine)\n= the p-value", p_lie, BLUE),
            ("P(fine | YES)\n= what we want", p_not_given_yes, ORANGE)]
    for k, (name, v, color) in enumerate(vals):
        ax.barh(k, v, height=0.55, color=color, zorder=2)
        ax.text(v + 0.02, k, f"{v:.3f}", va="center", fontsize=11)
    ax.axvline(0.05, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(0.06, 1.42, "α = .05", fontsize=9.5, color=INK2)
    ax.set_yticks([0, 1], [v[0] for v in vals])
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("probability")
    ax.grid(axis="x", zorder=0)
    ax.set_title("p is P(data | H₀), not P(H₀ | data)")

    # --- [1,1] the arithmetic
    panel(fig.add_subplot(gs[1, 1]), "The arithmetic", [
        "P(YES | exploded) = 35/36 ≈ 0.972",
        "P(exploded) = 1/1000  (generous prior)",
        "P(YES) = (35/36)(1/1000) + (1/36)(999/1000)",
        f"           = {py:.4f}",
        "",
        r"$P(\mathrm{exploded}\mid\mathrm{YES}) = \dfrac{0.001 \times 0.972}{0.0287} \approx 0.034$",
        "",
        "So after the alarm: 3.4% exploded,",
        f"{p_not_given_yes:.1%} the detector lied.",
    ], fontsize=12, spacing=0.083)

    # --- [1,2] frequentist vs Bayesian + note
    panel(fig.add_subplot(gs[1, 2]), "Frequentist vs Bayesian", [
        "Frequentist: how rare are the data if H₀",
        "holds? 1/36 < .05, so reject H₀.",
        "Bayesian: weigh the evidence by the prior;",
        "a strong prior survives a 1-in-36 alarm.",
        "",
        "Note: the notebook rounds P(YES | exploded)",
        "= 35/36 to 0.972 and prints",
        f"{notebook_value:.5f}; the exact 35/36 gives {posterior:.5f}.",
        f"Both are {posterior:.1%}, the value on this sheet.",
    ], fontsize=12, spacing=0.083)
    save(fig, name="6-statistics_seg3.png")


if __name__ == "__main__":
    sheet_seg1_frequentist()
    sheet_seg2_regression()
    sheet_seg3_bayes()
