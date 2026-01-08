import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

TEST_SIZE = 0.20
RANDOM_STATE = 42

KNN_K = 7
SVR_C = 10.0
SVR_EPS = 0.1
SVR_GAMMA = "scale"
SVR_KERNEL = "rbf"

_cache = {}

def _fit_models_if_needed():
    """Fit KNN and SVR once, store everything in _cache."""
    global _cache
    if _cache.get("fitted", False):
        return

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=KNN_K))
    ])

    svr_model = Pipeline([
        ("scaler", StandardScaler()),
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
        "knn_model": knn_model, "svr_model": svr_model,
        "y_pred_knn": y_pred_knn,
        "y_pred_svr": y_pred_svr,
    }


def _metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def fig_observed_vs_predicted(show=True):
    _fit_models_if_needed()
    y_test = _cache["y_test"]
    y_knn  = _cache["y_pred_knn"]
    y_svr  = _cache["y_pred_svr"]

    r2_knn, rmse_knn, mae_knn = _metrics(y_test, y_knn)
    r2_svr, rmse_svr, mae_svr = _metrics(y_test, y_svr)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    lo = min(np.min(y_test), np.min(y_knn), np.min(y_svr))
    hi = max(np.max(y_test), np.max(y_knn), np.max(y_svr))

    for ax, y_pred, title, stats_txt in [
        (axs[0], y_knn, "KNN Regression (test set)",
         f"R²={r2_knn:.3f}\nRMSE={rmse_knn:.2f}\nMAE={mae_knn:.2f}"),
        (axs[1], y_svr, "SVR Regression (test set)",
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

def fig_residuals_vs_predicted(show=True):
    _fit_models_if_needed()
    y_test = _cache["y_test"]
    y_knn  = _cache["y_pred_knn"]
    y_svr  = _cache["y_pred_svr"]

    res_knn = y_test - y_knn
    res_svr = y_test - y_svr

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, y_pred, res, title in [
        (axs[0], y_knn, res_knn, "KNN Residuals (test set)"),
        (axs[1], y_svr, res_svr, "SVR Residuals (test set)")
    ]:
        ax.scatter(y_pred, res, alpha=0.55)
        ax.axhline(0, lw=2, color="k", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Predicted heating load")
        ax.set_ylabel("Residual (observed - predicted)")

        txt = f"Residual mean={np.mean(res):.2f}\nResidual std={np.std(res):.2f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(facecolor="0.90", alpha=0.85))

    if show:
        plt.show()
    return fig

def plot_ml_figure1(num=4):
    display_title(
        "Machine learning regression with validation — Observed vs predicted on the held-out test set (KNN vs SVR)",
        pref="Figure", num=num, center=False
    )
    fig_observed_vs_predicted(show=True)


def plot_ml_figure2(num=5):
    display_title(
        "Machine learning regression with validation — Residual diagnostics on the held-out test set (KNN vs SVR)",
        pref="Figure", num=num, center=False
    )
    fig_residuals_vs_predicted(show=True)
