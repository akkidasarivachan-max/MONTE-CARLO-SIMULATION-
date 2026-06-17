import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# ── Simulation parameters ──────────────────────────────────────────────────
TAX_RATE = 0.25
BASE_GROWTH = 0.02
SCENARIO_WEIGHTS = {"bull": 0.20, "base": 0.60, "bear": 0.20}

# Per-scenario: extra annual growth on top of BASE_GROWTH, noise multiplier,
# and tax-rate shift applied to earnings-type metrics.
SCENARIO_CFG = {
    "bull": {"growth_delta": +0.04, "noise_factor": 0.6,  "tax_adj": -0.03},
    "base": {"growth_delta":  0.00, "noise_factor": 1.0,  "tax_adj":  0.00},
    "bear": {"growth_delta": -0.05, "noise_factor": 1.5,  "tax_adj": +0.03},
}

# Metrics that receive a tax-rate adjustment in forecasts.
EARNINGS_METRICS = {"Earnings ($B)", "EPS ($)", "Market cap ($B)"}
# ──────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


def load_mcd_dataset():
    frame = pd.read_csv(BASE_DIR / "McDonalds_Financial_Statements.csv")
    frame = frame.sort_values("Year").reset_index(drop=True)
    frame["Year"] = frame["Year"].astype(int)
    return frame


def load_sbux_dataset():
    frame = pd.read_excel(BASE_DIR / "financial_data_sbux.xlsx")
    frame = frame.sort_values("Year").reset_index(drop=True)
    frame = frame.rename(
        columns={
            "Revenue": "Revenue ($B)",
            "COGS": "COGS ($B)",
            "Gross_Profit": "Gross Profit ($B)",
            "SGA": "SG&A ($B)",
            "RD": "R&D ($B)",
            "Marketing": "Marketing ($B)",
            "EBITDA": "EBITDA ($B)",
            "Depreciation": "Depreciation ($B)",
            "EBIT": "EBIT ($B)",
            "Interest": "Interest ($B)",
            "EBT": "EBT ($B)",
            "Tax": "Tax ($B)",
            "Net_Income": "Net Income ($B)",
            "Working_Capital": "Working Capital ($B)",
            "CapEx": "CapEx ($B)",
        }
    )
    frame["Year"] = frame["Year"].astype(int)
    for column in frame.columns:
        if column != "Year":
            frame[column] = pd.to_numeric(frame[column], errors="coerce") / 1e9
    return frame


MCD_METRICS = [
    "Market cap ($B)",
    "Revenue ($B)",
    "Earnings ($B)",
    "EPS ($)",
    "Operating Margin (%)",
    "Total assets ($B)",
    "Total debt ($B)",
    "P/E ratio",
    "Cash on Hand ($B)",
    "Dividend Yield (%)",
]

SBUX_METRICS = [
    "Revenue ($B)",
    "COGS ($B)",
    "Gross Profit ($B)",
    "SG&A ($B)",
    "R&D ($B)",
    "Marketing ($B)",
    "EBITDA ($B)",
    "Depreciation ($B)",
    "EBIT ($B)",
    "Interest ($B)",
    "EBT ($B)",
    "Tax ($B)",
    "Net Income ($B)",
    "Working Capital ($B)",
    "CapEx ($B)",
]

DEFAULT_DATASET = "mcd"

DATASETS = {
    "mcd": {
        "label": "McDonald's",
        "frame": load_mcd_dataset(),
        "metrics": MCD_METRICS,
        "overview_metrics": ["Market cap ($B)", "Revenue ($B)", "Earnings ($B)", "EPS ($)"],
        "default_metric": "Revenue ($B)",
    },
    "sbux": {
        "label": "Starbucks (SBUX)",
        "frame": load_sbux_dataset(),
        "metrics": SBUX_METRICS,
        "overview_metrics": ["Revenue ($B)", "EBITDA ($B)", "Net Income ($B)", "CapEx ($B)"],
        "default_metric": "Revenue ($B)",
    },
}


def get_dataset(dataset_key):
    return DATASETS.get(dataset_key, DATASETS[DEFAULT_DATASET])


def normalize_dataset_key(dataset_key):
    return dataset_key if dataset_key in DATASETS else DEFAULT_DATASET


def get_metric_list(dataset_key):
    return get_dataset(dataset_key)["metrics"]


def get_overview_metrics(dataset_key):
    return get_dataset(dataset_key)["overview_metrics"]


def get_common_metrics():
    metric_sets = [set(cfg["metrics"]) for cfg in DATASETS.values()]
    if not metric_sets:
        return []
    common = set.intersection(*metric_sets)
    return sorted(common)


def is_earnings_metric(metric):
    lowered = metric.lower()
    return any(token in lowered for token in ("earnings", "eps", "market cap", "net income", "ebitda", "ebit", "ebt"))


def serialize_datasets():
    return {
        key: {
            "label": cfg["label"],
            "metrics": cfg["metrics"],
            "overview_metrics": cfg["overview_metrics"],
            "default_metric": cfg["default_metric"],
            "start_year": int(cfg["frame"]["Year"].min()),
            "end_year": int(cfg["frame"]["Year"].max()),
        }
        for key, cfg in DATASETS.items()
    }


def build_features(years_series):
    """Build time-based features for regression."""
    base = years_series.reshape(-1, 1).astype(float)
    return base


def monte_carlo_predict(metric, dataset_key=DEFAULT_DATASET, n_future=5, n_simulations=1000, noise_pct=0.05):
    dataset = get_dataset(dataset_key)
    frame = dataset["frame"]
    metric = metric if metric in frame.columns else dataset["default_metric"]

    years = frame["Year"].values.astype(float)
    values = frame[metric].values.astype(float)

    last_year = int(frame["Year"].max())
    future_years = np.arange(last_year + 1, last_year + n_future + 1, dtype=float)

    X_hist = build_features(years)
    X_future = build_features(future_years)

    value_std = np.std(values)
    value_mean = np.abs(np.mean(values))
    noise_scale = noise_pct * value_std if value_std > 0 else noise_pct * value_mean

    scenario_names = list(SCENARIO_WEIGHTS.keys())
    scenario_probs = list(SCENARIO_WEIGHTS.values())
    years_ahead = np.arange(1, n_future + 1, dtype=float)

    simulation_paths = []
    scenario_counts = {s: 0 for s in scenario_names}

    for _ in range(n_simulations):
        # Draw scenario according to probability weights
        scenario = np.random.choice(scenario_names, p=scenario_probs)
        cfg = SCENARIO_CFG[scenario]
        scenario_counts[scenario] += 1

        # Bootstrap resample with scenario-scaled noise
        effective_noise = noise_scale * cfg["noise_factor"]
        idx = np.random.choice(len(years), size=len(years), replace=True)
        X_boot = X_hist[idx]
        y_boot = values[idx] + np.random.normal(0, effective_noise, size=len(idx))

        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=True)),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=10.0)),
            ]
        )
        model.fit(X_boot, y_boot)
        preds = model.predict(X_future)

        # Compound growth overlay: BASE_GROWTH + scenario growth delta
        effective_growth = BASE_GROWTH + cfg["growth_delta"]
        preds = preds * ((1 + effective_growth) ** years_ahead)

        # Tax-rate adjustment for earnings-type metrics
        if is_earnings_metric(metric):
            effective_tax = np.clip(TAX_RATE + cfg["tax_adj"], 0.05, 0.50)
            preds = preds * (1 - effective_tax) / (1 - TAX_RATE)

        simulation_paths.append(preds.tolist())

    paths = np.array(simulation_paths)

    # Fit best model on full data for point estimate
    best_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=True)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0)),
        ]
    )
    best_model.fit(X_hist, values)
    hist_fitted = best_model.predict(X_hist)
    mae = float(mean_absolute_error(values, hist_fitted))
    r2 = float(r2_score(values, hist_fitted))
    sample_size = min(80, len(paths))

    return {
        "future_years": future_years.astype(int).tolist(),
        "mean": np.mean(paths, axis=0).tolist(),
        "median": np.median(paths, axis=0).tolist(),
        "p5": np.percentile(paths, 5, axis=0).tolist(),
        "p25": np.percentile(paths, 25, axis=0).tolist(),
        "p75": np.percentile(paths, 75, axis=0).tolist(),
        "p95": np.percentile(paths, 95, axis=0).tolist(),
        "std": np.std(paths, axis=0).tolist(),
        "sample_paths": paths[np.random.choice(len(paths), sample_size, replace=False)].tolist(),
        "final_year_dist": paths[:, -1].tolist(),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_simulations": n_simulations,
        "scenario_counts": scenario_counts,
        "scenario_weights": SCENARIO_WEIGHTS,
        "tax_rate": TAX_RATE,
        "base_growth": BASE_GROWTH,
        "dataset": dataset_key,
        "dataset_label": dataset["label"],
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        datasets=serialize_datasets(),
        default_dataset=DEFAULT_DATASET,
        comparison_metrics=get_common_metrics(),
    )


@app.route("/api/historical")
def historical():
    dataset_key = normalize_dataset_key(request.args.get("dataset", DEFAULT_DATASET))
    metric = request.args.get("metric", "Revenue ($B)")
    dataset = get_dataset(dataset_key)
    frame = dataset["frame"]
    metric = metric if metric in frame.columns else dataset["default_metric"]
    return jsonify(
        {
            "years": frame["Year"].astype(int).tolist(),
            "values": frame[metric].round(3).tolist(),
            "metric": metric,
            "dataset": dataset_key,
        }
    )


@app.route("/api/predict")
def predict():
    dataset_key = normalize_dataset_key(request.args.get("dataset", DEFAULT_DATASET))
    metric = request.args.get("metric", "Revenue ($B)")
    n_future = min(int(request.args.get("n_future", 5)), 15)
    n_simulations = min(int(request.args.get("n_simulations", 10000)), 3000)
    noise_pct = float(request.args.get("noise_pct", 0.05))

    result = monte_carlo_predict(metric, dataset_key, n_future, n_simulations, noise_pct)
    result["metric"] = metric
    dataset = get_dataset(dataset_key)
    frame = dataset["frame"]
    metric = metric if metric in frame.columns else dataset["default_metric"]
    result["historical_years"] = frame["Year"].astype(int).tolist()
    result["historical_values"] = frame[metric].round(3).tolist()
    result["dataset"] = dataset_key

    return jsonify(result)


@app.route("/api/overview")
def overview():
    dataset_key = normalize_dataset_key(request.args.get("dataset", DEFAULT_DATASET))
    dataset = get_dataset(dataset_key)
    frame = dataset["frame"]
    latest = frame.iloc[-1]
    prev = frame.iloc[-2]
    cards = []
    for col in get_overview_metrics(dataset_key):
        pv = float(prev[col])
        lv = float(latest[col])
        change = ((lv - pv) / abs(pv)) * 100 if pv != 0 else 0
        cards.append(
            {
                "metric": col,
                "value": round(lv, 2),
                "change": round(change, 1),
                "year": int(latest["Year"]),
            }
        )

    return jsonify(cards)



@app.route("/api/comparison")
def comparison():
    dataset_a_key = normalize_dataset_key(request.args.get("dataset_a", "mcd"))
    dataset_b_key = normalize_dataset_key(request.args.get("dataset_b", "sbux"))
    dataset_a = get_dataset(dataset_a_key)
    dataset_b = get_dataset(dataset_b_key)
    shared_metrics = get_common_metrics()
    default_metric = shared_metrics[0] if shared_metrics else dataset_a["default_metric"]
    metric = request.args.get("metric", default_metric)

    if metric not in dataset_a["frame"].columns or metric not in dataset_b["frame"].columns:
        metric = default_metric

    frame_a = dataset_a["frame"].set_index("Year")
    frame_b = dataset_b["frame"].set_index("Year")
    shared_years = sorted(set(frame_a.index).intersection(frame_b.index))

    if not shared_years:
        return jsonify(
            {
                "metric": metric,
                "dataset_a": dataset_a_key,
                "dataset_b": dataset_b_key,
                "labels": [],
                "series_a": [],
                "series_b": [],
                "normalized_a": [],
                "normalized_b": [],
                "rows": [],
                "summary": {},
            }
        )

    series_a = frame_a.loc[shared_years, metric].astype(float)
    series_b = frame_b.loc[shared_years, metric].astype(float)
    diff = series_a - series_b
    pct_gap = np.where(series_b != 0, diff / np.abs(series_b) * 100, 0.0)

    base_a = float(series_a.iloc[0]) if float(series_a.iloc[0]) != 0 else 1.0
    base_b = float(series_b.iloc[0]) if float(series_b.iloc[0]) != 0 else 1.0
    normalized_a = (series_a / base_a * 100).tolist()
    normalized_b = (series_b / base_b * 100).tolist()

    first_year = int(shared_years[0])
    last_year = int(shared_years[-1])
    latest_a = float(series_a.iloc[-1])
    latest_b = float(series_b.iloc[-1])
    latest_gap = latest_a - latest_b
    latest_gap_pct = (latest_gap / abs(latest_b) * 100) if latest_b != 0 else 0.0
    start_gap = float(series_a.iloc[0] - series_b.iloc[0])
    start_gap_pct = (start_gap / abs(series_b.iloc[0]) * 100) if float(series_b.iloc[0]) != 0 else 0.0

    growth_a = ((latest_a - float(series_a.iloc[0])) / abs(float(series_a.iloc[0])) * 100) if float(series_a.iloc[0]) != 0 else 0.0
    growth_b = ((latest_b - float(series_b.iloc[0])) / abs(float(series_b.iloc[0])) * 100) if float(series_b.iloc[0]) != 0 else 0.0

    rows = []
    for idx, year in enumerate(shared_years):
        rows.append(
            {
                "year": int(year),
                "value_a": round(float(series_a.iloc[idx]), 3),
                "value_b": round(float(series_b.iloc[idx]), 3),
                "difference": round(float(diff.iloc[idx]), 3),
                "gap_pct": round(float(pct_gap[idx]), 2),
            }
        )

    # All-metrics snapshot: every metric from either dataset
    metrics_set_a = set(dataset_a["metrics"])
    metrics_set_b = set(dataset_b["metrics"])
    all_metric_names = sorted(metrics_set_a | metrics_set_b)
    frame_a_full = dataset_a["frame"]
    frame_b_full = dataset_b["frame"]

    all_metrics_data = []
    for m in all_metric_names:
        has_a = m in metrics_set_a and m in frame_a_full.columns
        has_b = m in metrics_set_b and m in frame_b_full.columns

        m_val_a = m_cagr_a = m_growth_a = m_year_a = None
        m_val_b = m_cagr_b = m_growth_b = m_year_b = None

        if has_a:
            m_first_a = float(frame_a_full[m].iloc[0])
            m_last_a = float(frame_a_full[m].iloc[-1])
            m_nyrs_a = len(frame_a_full) - 1
            m_val_a = round(m_last_a, 3)
            m_year_a = int(frame_a_full["Year"].iloc[-1])
            if m_first_a != 0 and m_nyrs_a > 0:
                m_cagr_a = round(
                    ((abs(m_last_a) / abs(m_first_a)) ** (1.0 / m_nyrs_a) - 1) * 100, 2
                )
                m_growth_a = round(((m_last_a - m_first_a) / abs(m_first_a)) * 100, 2)

        if has_b:
            m_first_b = float(frame_b_full[m].iloc[0])
            m_last_b = float(frame_b_full[m].iloc[-1])
            m_nyrs_b = len(frame_b_full) - 1
            m_val_b = round(m_last_b, 3)
            m_year_b = int(frame_b_full["Year"].iloc[-1])
            if m_first_b != 0 and m_nyrs_b > 0:
                m_cagr_b = round(
                    ((abs(m_last_b) / abs(m_first_b)) ** (1.0 / m_nyrs_b) - 1) * 100, 2
                )
                m_growth_b = round(((m_last_b - m_first_b) / abs(m_first_b)) * 100, 2)

        all_metrics_data.append(
            {
                "metric": m,
                "val_a": m_val_a,
                "val_b": m_val_b,
                "cagr_a": m_cagr_a,
                "cagr_b": m_cagr_b,
                "growth_a": m_growth_a,
                "growth_b": m_growth_b,
                "year_a": m_year_a,
                "year_b": m_year_b,
                "shared": has_a and has_b,
                "exclusive_a": has_a and not has_b,
                "exclusive_b": not has_a and has_b,
            }
        )

    return jsonify(
        {
            "metric": metric,
            "dataset_a": dataset_a_key,
            "dataset_b": dataset_b_key,
            "label_a": dataset_a["label"],
            "label_b": dataset_b["label"],
            "labels": [int(year) for year in shared_years],
            "series_a": [round(float(v), 3) for v in series_a.tolist()],
            "series_b": [round(float(v), 3) for v in series_b.tolist()],
            "normalized_a": [round(float(v), 3) for v in normalized_a],
            "normalized_b": [round(float(v), 3) for v in normalized_b],
            "rows": rows,
            "all_metrics": all_metrics_data,
            "summary": {
                "first_year": first_year,
                "last_year": last_year,
                "latest_a": round(latest_a, 3),
                "latest_b": round(latest_b, 3),
                "latest_gap": round(latest_gap, 3),
                "latest_gap_pct": round(latest_gap_pct, 2),
                "start_gap": round(start_gap, 3),
                "start_gap_pct": round(start_gap_pct, 2),
                "growth_a": round(growth_a, 2),
                "growth_b": round(growth_b, 2),
            },
        }
    )


@app.route("/api/all_metrics")
def all_metrics():
    dataset_key = normalize_dataset_key(request.args.get("dataset", DEFAULT_DATASET))
    dataset = get_dataset(dataset_key)
    frame = dataset["frame"]
    metrics = get_metric_list(dataset_key)
    records = []
    for _, row in frame.iterrows():
        rec = {"Year": int(row["Year"])}
        for m in metrics:
            rec[m] = round(float(row[m]), 3)
        records.append(rec)
    return jsonify({"columns": ["Year"] + metrics, "rows": records, "dataset": dataset_key})


@app.route("/api/correlation")
def correlation():
    dataset_key = normalize_dataset_key(request.args.get("dataset", DEFAULT_DATASET))
    dataset = get_dataset(dataset_key)
    frame = dataset["frame"]
    cols = get_metric_list(dataset_key)
    sub = frame[cols].astype(float)
    corr = sub.corr().round(3)
    return jsonify({"labels": corr.columns.tolist(), "matrix": corr.values.tolist(), "dataset": dataset_key})


@app.route("/api/multi_metric_forecast")
def multi_metric_forecast():
    dataset_key = normalize_dataset_key(request.args.get("dataset", DEFAULT_DATASET))
    n_future = min(int(request.args.get("n_future", 5)), 15)
    n_simulations = min(int(request.args.get("n_simulations", 500)), 1000)
    results = {}
    for metric in get_overview_metrics(dataset_key):
        mc = monte_carlo_predict(metric, dataset_key, n_future, n_simulations)
        results[metric] = {
            "future_years": mc["future_years"],
            "mean": mc["mean"],
            "p5": mc["p5"],
            "p95": mc["p95"],
        }
    return jsonify(results)


@app.route("/api/sim_params")
def sim_params():
    return jsonify({
        "tax_rate": TAX_RATE,
        "base_growth": BASE_GROWTH,
        "scenario_weights": SCENARIO_WEIGHTS,
        "scenario_cfg": {
            s: {k: v for k, v in cfg.items()}
            for s, cfg in SCENARIO_CFG.items()
        },
        "earnings_metrics": list(EARNINGS_METRICS),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
