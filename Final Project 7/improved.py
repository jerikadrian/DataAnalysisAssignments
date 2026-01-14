import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def display_title(s, pref="Figure", num=1, center=False):
    ctag = "center" if center else "p"
    s = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    s = f"{s}<br><br>"
    display(Markdown(s))

TARGET = "heating_load"

FEATURES = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

CAT_FEATURES = ["orientation", "glazing_area_distribution"]
NUM_FEATURES = [c for c in FEATURES if c not in CAT_FEATURES]

TEST_SIZE = 0.20
RANDOM_STATE = 42

KNN_K = 7
SVR_C = 10.0
SVR_EPS = 0.1
SVR_GAMMA = "scale"
SVR_KERNEL = "rbf"

_cache = {}

def _metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def _common_limits(*arrays):
    lo = min([np.min(a) for a in arrays])
    hi = max([np.max(a) for a in arrays])
    return lo, hi

def _make_preprocessor():
    """
    Scaling + encoding only:
      - scale numeric columns
      - one-hot encode categorical columns
    Fitted on training data inside the Pipeline.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ],
        remainder="drop"
    )
    return pre

def _fit_models_if_needed():
    """Fit KNN and SVR once (WITH scaling+encoding), store everything in _cache."""
    global _cache
    if _cache.get("fitted", False):
        return

    X = df[FEATURES].copy()
    y = df[TARGET].to_numpy(dtype=float)

    h = df["overall_height"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test, h_train, h_test = train_test_split(
        X, y, h, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pre = _make_preprocessor()

    knn_model = Pipeline([
        ("pre", pre),
        ("knn", KNeighborsRegressor(n_neighbors=KNN_K))
    ])

    svr_model = Pipeline([
        ("pre", pre),
        ("svr", SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPS, gamma=SVR_GAMMA))
    ])

    knn_model.fit(X_train, y_train)
    svr_model.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test)
    y_pred_svr = svr_model.predict(X_test)

    _cache = {
        "fitted": True,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "h_train": h_train, "h_test": h_test,
        "knn_model": knn_model, "svr_model": svr_model,
        "y_pred_knn": y_pred_knn,
        "y_pred_svr": y_pred_svr,
    }

def fig_observed_vs_predicted(show=True):
    _fit_models_if_needed()
    y_test = _cache["y_test"]
    y_knn  = _cache["y_pred_knn"]
    y_svr  = _cache["y_pred_svr"]

    r2_knn, rmse_knn, mae_knn = _metrics(y_test, y_knn)
    r2_svr, rmse_svr, mae_svr = _metrics(y_test, y_svr)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    lo, hi = _common_limits(y_test, y_knn, y_svr)

    for ax, y_pred, title, stats_txt in [
        (axs[0], y_knn, "KNN Regression (test set) [scaling + encoding]",
         f"R²={r2_knn:.3f}\nRMSE={rmse_knn:.2f}\nMAE={mae_knn:.2f}"),
        (axs[1], y_svr, "SVR Regression (test set) [scaling + encoding]",
         f"R²={r2_svr:.3f}\nRMSE={rmse_svr:.2f}\nMAE={mae_svr:.2f}")
    ]:
        ax.scatter(y_test, y_pred, alpha=0.55)
        ax.plot([lo, hi], [lo, hi], lw=2, color="k", alpha=0.7)  # y=x line
        ax.set_title(title)
        ax.set_xlabel("Observed heating load")
        ax.set_ylabel("Predicted heating load")
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(facecolor="0.90", alpha=0.85))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    if show:
        plt.show()
    return fig

def fig_stratified_by_height(show=True):
    _fit_models_if_needed()
    y_test = _cache["y_test"]
    h_test = _cache["h_test"]
    y_knn  = _cache["y_pred_knn"]
    y_svr  = _cache["y_pred_svr"]

    heights = np.array(sorted(np.unique(h_test)))
    if len(heights) != 2:
        raise ValueError(f"Expected exactly 2 unique heights in test set, found: {heights}")

    h_low, h_high = heights[0], heights[1]
    idx_low  = (h_test == h_low)
    idx_high = (h_test == h_high)

    r2_knn_low,  rmse_knn_low,  mae_knn_low  = _metrics(y_test[idx_low],  y_knn[idx_low])
    r2_knn_high, rmse_knn_high, mae_knn_high = _metrics(y_test[idx_high], y_knn[idx_high])

    r2_svr_low,  rmse_svr_low,  mae_svr_low  = _metrics(y_test[idx_low],  y_svr[idx_low])
    r2_svr_high, rmse_svr_high, mae_svr_high = _metrics(y_test[idx_high], y_svr[idx_high])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    lo, hi = _common_limits(
        y_test[idx_low], y_test[idx_high],
        y_knn[idx_low], y_knn[idx_high],
        y_svr[idx_low], y_svr[idx_high]
    )

    panels = [
        (axs[0,0], y_test[idx_low],  y_knn[idx_low],  f"KNN (test) | height = {h_low} [scaling + encoding]",
         f"n={idx_low.sum()}\nR²={r2_knn_low:.3f}\nRMSE={rmse_knn_low:.2f}\nMAE={mae_knn_low:.2f}"),
        (axs[0,1], y_test[idx_high], y_knn[idx_high], f"KNN (test) | height = {h_high} [scaling + encoding]",
         f"n={idx_high.sum()}\nR²={r2_knn_high:.3f}\nRMSE={rmse_knn_high:.2f}\nMAE={mae_knn_high:.2f}"),
        (axs[1,0], y_test[idx_low],  y_svr[idx_low],  f"SVR (test) | height = {h_low} [scaling + encoding]",
         f"n={idx_low.sum()}\nR²={r2_svr_low:.3f}\nRMSE={rmse_svr_low:.2f}\nMAE={mae_svr_low:.2f}"),
        (axs[1,1], y_test[idx_high], y_svr[idx_high], f"SVR (test) | height = {h_high} [scaling + encoding]",
         f"n={idx_high.sum()}\nR²={r2_svr_high:.3f}\nRMSE={rmse_svr_high:.2f}\nMAE={mae_svr_high:.2f}"),
    ]

    for ax, y_obs, y_pred, title, stats_txt in panels:
        ax.scatter(y_obs, y_pred, alpha=0.55)
        ax.plot([lo, hi], [lo, hi], lw=2, color="k", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Observed heating load")
        ax.set_ylabel("Predicted heating load")
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(facecolor="0.90", alpha=0.85))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    if show:
        plt.show()
    return fig

def plot_improved_figure1(num=6):
    display_title(
        "Observed vs predicted on the held-out test set (KNN vs SVR) [scaling + encoding]",
        pref="Figure", num=num, center=False
    )
    fig_observed_vs_predicted(show=True)

def plot_improved_figure2(num=7):
    display_title(
        "Model performance changes across height regimes (h = 3.5 vs 7.0) for KNN and SVR [scaling + encoding]",
        pref="Figure", num=num, center=False
    )
    fig_stratified_by_height(show=True)
