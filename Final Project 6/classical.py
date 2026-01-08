import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown

def display_title(s, pref="Figure", num=1, center=False):
    ctag = "center" if center else "p"
    s = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    s = f"{s}<br><br>"
    display(Markdown(s))

def _as_float(a):
    return np.asarray(a, dtype=float)

def _mask_valid(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        a = _as_float(a)
        m &= ~np.isnan(a)
    return m

def _jitter(x, scale=0.08, seed=0):
    rng = np.random.default_rng(seed)
    x = _as_float(x)
    return x + rng.normal(0, scale, size=len(x))

def _welch_ttest(y0, y1):
    y0 = _as_float(y0); y1 = _as_float(y1)
    y0 = y0[~np.isnan(y0)]
    y1 = y1[~np.isnan(y1)]
    t, p = stats.ttest_ind(y0, y1, equal_var=False)
    return t, p

def _anova(groups):
    clean = []
    for g in groups:
        g = _as_float(g)
        g = g[~np.isnan(g)]
        if len(g) > 0:
            clean.append(g)
    F, p = stats.f_oneway(*clean)
    return F, p

def _style_boxplot(bp, face_alpha=0.25):
    """Make boxplots translucent; force median/lines to black."""
    for box in bp["boxes"]:
        box.set_alpha(face_alpha)
        box.set_edgecolor("k")

    for line in bp["medians"]:
        line.set_color("k")
        line.set_linewidth(2)

    for key in ["whiskers", "caps"]:
        for line in bp[key]:
            line.set_color("k")
            line.set_linewidth(1.5)

def _box_jitter(ax, groups, tick_labels, title, ylabel="Heating load",
                seed=0, box_alpha=0.25, point_alpha=0.22):
    bp = ax.boxplot(groups, tick_labels=tick_labels, patch_artist=True)
    _style_boxplot(bp, face_alpha=box_alpha)

    for i, g in enumerate(groups, start=1):
        xj = _jitter(np.ones_like(g) * i, scale=0.07, seed=seed + i)
        ax.scatter(xj, g, alpha=point_alpha)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return bp

def _means_line(ax, cats, means, title, xlabel,
                ylabel="Mean heating load",
                flatten=False, pad_fraction=0.10):
    ax.plot(cats, means, marker="o", lw=2, color="k")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

    if flatten:
        means = np.asarray(means, dtype=float)
        lo, hi = np.min(means), np.max(means)
        span = hi - lo
        pad = max(span * pad_fraction, 0.5)
        ax.set_ylim(lo - pad, hi + pad)

def fig_glazing_distribution(show=True, seed=0):
    y = df["heating_load"].to_numpy(dtype=float)
    gld = df["glazing_area_distribution"].to_numpy(dtype=float)

    m = _mask_valid(y, gld)
    y = y[m]; gld = gld[m]

    cats = np.array(sorted(np.unique(gld)))
    groups = [y[gld == c] for c in cats]
    labels = [str(int(c)) if float(c).is_integer() else str(c) for c in cats]
    _, p_a = _anova(groups)

    y0 = y[gld == 0]
    y1 = y[gld != 0]
    _, p_t = _welch_ttest(y0, y1)

    means = [np.mean(g) for g in groups]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    axA = fig.add_subplot(gs[0, :])   # top spans both columns (ANOVA)
    axB = fig.add_subplot(gs[1, 0])   # bottom-left (t-test)
    axC = fig.add_subplot(gs[1, 1])   # bottom-right (means)

    _box_jitter(
        axA,
        groups=groups,
        tick_labels=labels,
        title="(A) One-way ANOVA: Heating load by glazing distribution category (0–5)",
        seed=seed + 20,
        box_alpha=0.25,
        point_alpha=0.20
    )
    axA.text(
        0.02, 0.98,
        f"One-way ANOVA\np = {p_a:.2e}",
        transform=axA.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="0.90", alpha=0.85)
    )

    _box_jitter(
        axB,
        groups=[y0, y1],
        tick_labels=["gld = 0", "gld ≠ 0"],
        title="(B) Welch t-test: 0 vs non-0 glazing distribution",
        seed=seed + 10,
        box_alpha=0.25,
        point_alpha=0.22
    )
    axB.text(
        0.02, 0.98,
        f"Welch t-test\np = {p_t:.2e}",
        transform=axB.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="0.90", alpha=0.85)
    )

    _means_line(
        axC,
        cats=cats,
        means=means,
        title="(C) Mean heating load by gld category",
        xlabel="Glazing distribution category (gld)",
        flatten=False
    )

    if show:
        plt.show()
    return fig

def fig_height_and_orientation(show=True, seed=0):
    y = df["heating_load"].to_numpy(dtype=float)
    h = df["overall_height"].to_numpy(dtype=float)
    ori = df["orientation"].to_numpy(dtype=float)

    m = _mask_valid(y, h, ori)
    y = y[m]; h = h[m]; ori = ori[m]

    ori_cats = np.array(sorted(np.unique(ori)))
    ori_groups = [y[ori == c] for c in ori_cats]
    ori_labels = [str(int(c)) if float(c).is_integer() else str(c) for c in ori_cats]
    _, p_a = _anova(ori_groups)
    ori_means = [np.mean(g) for g in ori_groups]

    heights = np.array(sorted(np.unique(h)))
    if len(heights) != 2:
        raise ValueError(f"Expected exactly 2 unique heights, found: {heights}")
    h_low, h_high = heights[0], heights[1]
    y_low = y[h == h_low]
    y_high = y[h == h_high]
    _, p_t = _welch_ttest(y_low, y_high)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    axA = fig.add_subplot(gs[0, :])   # top spans both columns (ANOVA)
    axB = fig.add_subplot(gs[1, 0])   # bottom-left (t-test)
    axC = fig.add_subplot(gs[1, 1])   # bottom-right (means)

    _box_jitter(
        axA,
        groups=ori_groups,
        tick_labels=ori_labels,
        title="(A) One-way ANOVA: Heating load by orientation",
        seed=seed + 40,
        box_alpha=0.25,
        point_alpha=0.20
    )
    axA.text(
        0.02, 0.98,
        f"One-way ANOVA\np = {p_a:.2e}",
        transform=axA.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="0.90", alpha=0.85)
    )

    _box_jitter(
        axB,
        groups=[y_low, y_high],
        tick_labels=[f"h = {h_low}", f"h = {h_high}"],
        title="(B) Welch t-test: Heating load by height",
        seed=seed + 30,
        box_alpha=0.25,
        point_alpha=0.22
    )
    axB.text(
        0.02, 0.98,
        f"Welch t-test\np = {p_t:.2e}",
        transform=axB.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="0.90", alpha=0.85)
    )

    _means_line(
        axC,
        cats=ori_cats,
        means=ori_means,
        title="(C) Mean heating load by orientation (zoomed scale)",
        xlabel="Orientation category",
        flatten=True,
        pad_fraction=0.10
    )

    if show:
        plt.show()
    return fig

def plot_hypothesis1(num=2):
    display_title(
        "Glazing distribution effects on heating load (one-way ANOVA across categories, plus Welch t-test for 0 vs non-0)",
        pref="Figure", num=num, center=False
    )
    fig_glazing_distribution(show=True)


def plot_hypothesis2(num=3):
    display_title(
        "Orientation and height effects on heating load (one-way ANOVA for orientation, plus Welch t-test for height groups)",
        pref="Figure", num=num, center=False
    )
    fig_height_and_orientation(show=True)
