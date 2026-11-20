#!/usr/bin/env python3
"""Stage 4 - decision-time yield forecasts (information up to two years before the season).

A plan for the season ending in year t is made with the district statistics known at planning time,
taken here to be those of year t-2. For each target year t in 2023-2025, every forecast is trained only
on outcomes up to t-2 and uses lags of t-2 or older. Four forecasts per district, crop and year
(column names in the outputs):
  EN_cutoff2           Elastic Net on district effects, trend, lag-2 yield and the mean of lag-2..4 yields
  Persistence_cutoff2  yield of year t-2
  Climatology_cutoff2  district mean yield of the years up to t-2
  Trend_cutoff2        district linear trend fitted to the years up to t-2, extrapolated to t

Elastic Net settings are chosen for each crop and target year by rolling-origin validation inside the
training years (each validation year v is predicted from a model trained on years <= v-2, the same gap
as the forecast; pooled RMSE; grid alpha = logspace(-3, 0.7, 18) x L1 ratio {0.1, 0.5, 0.9, 0.95}).
Training rows need only the lag-2 yield, so training starts in 2017. Maize gets the same forecasts.

Each Elastic Net forecast is also split into a baseline and the contribution of each predictor group,
coefficient x (value - mean of the training rows). For a linear model these are its exact SHAP values
with the training rows as background.

Outputs (generated/forecasts/):
  decision_time_forecasts.csv          the four forecasts and the observed yield, 192 rows
  decision_time_forecast_accuracy.csv  RMSE and MAE by crop and forecast
  decision_time_tuning.csv             chosen Elastic Net settings
  en_coefficients.csv                  penalties and coefficients (t/ha per SD of a predictor)
  en_contributions.csv                 baseline + predictor contributions of every Elastic Net forecast
  forecast_run_info.json               input hashes and settings
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

cli = argparse.ArgumentParser(description="Stage 4 - decision-time yield forecasts")
cli.add_argument("--input-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
args = cli.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = args.input_dir / "analysis_panel_full.csv"
MAIZE = args.input_dir / "maize_rabi_16districts_2015_2025.csv"
COLS = ["district", "crop", "year_end", "area_ha", "yield_t_ha"]
D = (
    pd.concat([pd.read_csv(SOURCE)[COLS], pd.read_csv(MAIZE)[COLS]])
    .sort_values(["district", "crop", "year_end"])
    .reset_index(drop=True)
)
C = ["Boro Rice", "Wheat", "Potato", "Maize"]
GRID = [(float(a), l1) for a in np.logspace(-3, 0.7, 18) for l1 in (0.1, 0.5, 0.9, 0.95)]
TRAIN_CUTOFF_GAP = 2
METHODS = ["EN_cutoff2", "Persistence_cutoff2", "Climatology_cutoff2", "Trend_cutoff2"]
for k in (2, 3, 4):
    D[f"lag{k}_yield"] = D.groupby(["district", "crop"]).yield_t_ha.shift(k)
    D[f"lag{k}_area"] = D.groupby(["district", "crop"]).area_ha.shift(k)
D["lag2_3_4_mean"] = D[["lag2_yield", "lag3_yield", "lag4_yield"]].mean(axis=1)
D["trend"] = D.year_end - 2015
X = ["district", "trend", "lag2_yield", "lag2_3_4_mean"]


def fit_model(crop, until, alpha, l1):
    """Elastic Net trained on the rows of one crop up to year `until` that have a lag-2 yield."""
    train = D[(D.crop == crop) & (D.year_end <= until) & D.lag2_yield.notna()].copy()
    if train.year_end.nunique() < 2:
        raise ValueError((crop, until, len(train)))
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["trend", "lag2_yield", "lag2_3_4_mean"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["district"]),
        ]
    )
    model = Pipeline(
        [("proc", pre), ("en", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=50000, random_state=42))]
    )
    model.fit(train[X], train.yield_t_ha)
    return model


def tune(crop, last):
    """Rolling-origin choice of (alpha, L1 ratio) using only years <= last (= target year - 2)."""
    first = int(D[D.lag2_yield.notna()].year_end.min())
    inner = [v for v in range(first + 3, last + 1)]  # fit on >= 2 years (<= v-2), predict v
    best = None
    for alpha, l1 in GRID:
        err = []
        for v in inner:
            m = fit_model(crop, v - 2, alpha, l1)
            te = D[(D.crop == crop) & (D.year_end == v)]
            err.append((te.yield_t_ha - np.maximum(0.01, m.predict(te[X]))) ** 2)
        rmse = float(np.sqrt(pd.concat(err).mean()))
        if best is None or rmse < best[0] - 1e-15:
            best = (rmse, alpha, l1)
    return dict(
        crop=crop,
        last_training_year=last,
        inner_validation_years=f"{inner[0]}-{inner[-1]}",
        inner_rmse_t_ha=best[0],
        alpha=best[1],
        l1_ratio=best[2],
    )


def trend_forecast(district, crop, last, target):
    """District linear trend fitted to the years up to `last`, extrapolated to `target` (at least 0.01 t/ha)."""
    h = D[(D.district == district) & (D.crop == crop) & (D.year_end <= last)]
    slope, intercept = np.polyfit(h.year_end.to_numpy(float), h.yield_t_ha.to_numpy(float), 1)
    return max(0.01, float(intercept + slope * target))


GROUPS = {
    "trend": ["num__trend"],
    "lag2_yield": ["num__lag2_yield"],
    "lag2_4_mean_yield": ["num__lag2_3_4_mean"],
}


def _design(model, rows):
    """Transformed design matrix of the fitted model for these rows."""
    z = model.named_steps["proc"].transform(rows[X])
    return z.toarray() if hasattr(z, "toarray") else np.asarray(z)


def _train(crop, until):
    """Training rows of a crop up to year `until`."""
    return D[(D.crop == crop) & (D.year_end <= until) & D.lag2_yield.notna()]


def describe_en(model, crop, yr, until, hp):
    """Penalties, intercept and coefficients of one fitted Elastic Net."""
    names = list(model.named_steps["proc"].get_feature_names_out())
    en = model.named_steps["en"]
    c = dict(zip(names, en.coef_))
    dist = np.array([v for k, v in c.items() if k.startswith("cat__district_")])
    return dict(
        crop=crop,
        target_year=yr,
        last_training_year=until,
        n_training_rows=len(_train(crop, until)),
        alpha=hp["alpha"],
        l1_ratio=hp["l1_ratio"],
        intercept=float(en.intercept_),
        coef_trend_per_sd=c["num__trend"],
        coef_lag2_yield_per_sd=c["num__lag2_yield"],
        coef_lag2_4_mean_yield_per_sd=c["num__lag2_3_4_mean"],
        district_effects_nonzero=int((np.abs(dist) > 0).sum()),
        district_effect_min=float(dist.min()) + 0.0,
        district_effect_max=float(dist.max()) + 0.0,
    )  # + 0.0 writes -0.0 as 0.0


def contributions(model, crop, yr, until, te):
    """Split each forecast into a baseline and predictor-group contributions (exact linear SHAP)."""
    names = list(model.named_steps["proc"].get_feature_names_out())
    en = model.named_steps["en"]
    mu = _design(model, _train(crop, until)).mean(axis=0)
    z = _design(model, te)
    part = (z - mu) * en.coef_
    out = pd.DataFrame(
        {
            "district": te.district.to_numpy(),
            "crop": crop,
            "year_end": yr,
            "baseline_t_ha": float(en.intercept_ + mu @ en.coef_),
        }
    )
    for g, cols in GROUPS.items():
        out[g] = part[:, [names.index(k) for k in cols]].sum(axis=1)
    out["district_effect"] = part[:, [i for i, k in enumerate(names) if k.startswith("cat__district_")]].sum(
        axis=1
    )
    out["prediction_t_ha"] = out.baseline_t_ha + out[list(GROUPS) + ["district_effect"]].sum(axis=1)
    assert np.allclose(out.prediction_t_ha.to_numpy(), model.predict(te[X]), atol=1e-9)
    return out


# ---------------------------------------------------------------------------------------------------
# Forecasts for 2023-2025
# ---------------------------------------------------------------------------------------------------
coef_rows, contrib_rows = [], []
forecasts, tuning = [], []
for yr in [2023, 2024, 2025]:
    for crop in C:
        tr_max = yr - TRAIN_CUTOFF_GAP
        hp = tune(crop, tr_max)
        hp["target_year"] = yr
        tuning.append(hp)
        model = fit_model(crop, tr_max, hp["alpha"], hp["l1_ratio"])
        hp["first_training_year"] = int(D[(D.crop == crop) & D.lag2_yield.notna()].year_end.min())
        te = D[(D.crop == crop) & (D.year_end == yr)].copy()
        te["EN_cutoff2"] = np.maximum(0.01, model.predict(te[X]))
        coef_rows.append(describe_en(model, crop, yr, tr_max, hp))
        contrib_rows.append(contributions(model, crop, yr, tr_max, te))
        te["Persistence_cutoff2"] = te.lag2_yield
        te["Climatology_cutoff2"] = [
            D[(D.district == d) & (D.crop == crop) & (D.year_end <= tr_max)].yield_t_ha.mean()
            for d in te.district
        ]
        te["Trend_cutoff2"] = [trend_forecast(d, crop, tr_max, yr) for d in te.district]
        te["last_training_outcome_year"] = tr_max
        te["last_lag_input_year"] = yr - 2
        forecasts.append(te)
P = pd.concat(forecasts).sort_values(["year_end", "district", "crop"])
assert len(P) == 192 and P[METHODS + ["lag2_area"]].notna().all().all()
pd.DataFrame(tuning).to_csv(OUT / "decision_time_tuning.csv", index=False)
P[
    [
        "district",
        "year_end",
        "crop",
        "yield_t_ha",
        "lag2_area",
        *METHODS,
        "last_training_outcome_year",
        "last_lag_input_year",
    ]
].to_csv(OUT / "decision_time_forecasts.csv", index=False)
# Accuracy over the 48 district-years of each crop
metrics = []
for crop in C:
    g = P[P.crop == crop]
    for m in METHODS:
        metrics.append(
            dict(
                crop=crop,
                forecast=m,
                RMSE_t_ha=float(np.sqrt(np.mean((g.yield_t_ha - g[m]) ** 2))),
                MAE_t_ha=float(np.mean(abs(g.yield_t_ha - g[m]))),
            )
        )
pd.DataFrame(metrics).to_csv(OUT / "decision_time_forecast_accuracy.csv", index=False)
pd.DataFrame(coef_rows).to_csv(OUT / "en_coefficients.csv", index=False)
_ct = pd.concat(contrib_rows).sort_values(["year_end", "district", "crop"])
assert len(_ct) == 192 and np.allclose(
    np.maximum(0.01, _ct.prediction_t_ha.to_numpy()),
    P.set_index(["year_end", "district", "crop"])
    .loc[list(_ct.set_index(["year_end", "district", "crop"]).index), "EN_cutoff2"]
    .to_numpy(),
)
_ct.to_csv(OUT / "en_contributions.csv", index=False)
assert D.groupby(["district", "crop"]).size().eq(11).all()
assert not D.duplicated(["district", "crop", "year_end"]).any()
assert (P.last_training_outcome_year <= P.year_end - 2).all()
meta = {
    "panel_rows": len(D),
    "test_forecasts": len(P),
    "forecasts": METHODS,
    "elastic_net_grid": "alpha logspace(-3, 0.7, 18) x l1 {0.1, 0.5, 0.9, 0.95}, rolling origin, gap 2",
    "forecast_cutoff_rule": "training outcome and lag input years <= target year minus 2",
    "crop_scope": C,
    "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "maize_input_sha256": hashlib.sha256(MAIZE.read_bytes()).hexdigest(),
}
(OUT / "forecast_run_info.json").write_text(json.dumps(meta, indent=2))
print("Forecasts written:", json.dumps(meta))
print(pd.DataFrame(metrics).to_string(index=False))
print(pd.DataFrame(tuning).to_string(index=False))
