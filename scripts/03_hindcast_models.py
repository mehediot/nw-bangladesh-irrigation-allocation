#!/usr/bin/env python3
"""Stage 3 - hindcast yield models with same-season climate (descriptive).

These models use the climate of the season they predict, so they cannot be used for planning; they
describe which inputs move yields. The forecasts used for planning are made in stage 4.

For Boro, wheat and potato and each test year 2023-2025 (trained on all earlier years):
  * Elastic Net, tuned by nested rolling-origin validation (inner years 2019 to test year - 1,
    pooled RMSE; grid alpha = logspace(-3, 0.7, 18) x L1 ratio {0.1, 0.5, 0.9, 0.95});
  * gradient-boosted trees (GBT) with fixed settings; the same fitted model gives the test predictions,
    the TreeSHAP values and the climate-stress counterfactuals;
  * two-way fixed-effects coefficients (district and year), descriptive associations only.
Predictors: season rain, preceding monsoon rain, terminal Tmax, season Tmin, season RH, a variety or
structure term (Boro hybrid share, wheat post-2018 indicator, potato HYV share), a year trend and
district effects.

Outputs (generated/hindcast/):
  00_elastic_net_tuning.csv          chosen Elastic Net settings per crop and test year
  01_climate_stress_states.csv       baseline, dry, hot and dry-hot climate states per district and crop
  02_fixed_effects_coefficients.csv  two-way fixed-effects coefficients
  03_hindcast_predictions.csv        test predictions of climatology, Elastic Net and GBT
  04_gbt_shap_values.csv             TreeSHAP values of every test prediction
  05_gbt_climate_counterfactuals.csv GBT predictions under the four climate states
  06_fit_record.csv                  training size, RMSE and SHAP additivity error of every fit
  07_hindcast_metrics.csv            RMSE, skill against climatology and R2
  08_shap_summary.csv                climate share of mean |SHAP| per crop
  09_baseline_coefficients.csv       2023-2025 mean areas, published NIR and mean Elastic Net yield
  10_climate_response_brackets.csv   yield under each climate state by fixed effects and by GBT
  11_boro_feasibility.csv            whether all Boro fits the budget, three-crop check
  12_joint_support_diagnostic.csv    distance of each climate state from the observed years
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cli = argparse.ArgumentParser(description="Stage 3 - hindcast yield models")
cli.add_argument("--input-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
args = cli.parse_args()
I, O = args.input_dir, args.output_dir
O.mkdir(parents=True, exist_ok=True)

CROPS = ["Boro Rice", "Wheat", "Potato"]
CLIMATE = ["season_rain_mm", "monsoon_prev_rain_mm", "terminal_tmax_c", "season_tmin_c", "season_rh_pct"]
STRESS_VARS = CLIMATE[:4]  # rain, monsoon rain, terminal Tmax, Tmin: the variables of the stress states
STRUCTURE = {"Boro Rice": ["hybrid_area_share"], "Wheat": ["wheat_postbreak"], "Potato": ["hyv_area_share"]}
TEST_YEARS = (2023, 2024, 2025)
EN_GRID = [(float(a), l1) for a in np.logspace(-3, 0.7, 18) for l1 in (0.1, 0.5, 0.9, 0.95)]

p = (
    pd.read_csv(I / "analysis_panel_full.csv")
    .sort_values(["crop", "year_end", "district"])
    .reset_index(drop=True)
)
assert p.shape == (528, 28)
assert not p.duplicated(["district", "crop", "year_end"]).any()
p["year_trend"] = p.year_end - p.year_end.min()
p["wheat_postbreak"] = (p.year_end >= 2018).astype(int)
assert p[["district", "year_end", "crop", "yield_t_ha", "area_ha", "nir_mm"] + CLIMATE].notna().all().all()
assert set(p.crop) == set(CROPS)


def preprocessor(features):
    """Standardise numeric predictors; one-hot district effects."""
    return ColumnTransformer(
        [("num", StandardScaler(), features), ("id", OneHotEncoder(handle_unknown="ignore"), ["district"])]
    )


def elastic_net(alpha, l1_ratio, features):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=25000, random_state=42)
    return Pipeline([("pre", preprocessor(features)), ("model", model)])


def nested_tuning(panel):
    """Choose (alpha, L1 ratio) per crop and test year by rolling-origin validation on earlier years."""
    rows = []
    for crop in CROPS:
        g = panel[panel.crop == crop].sort_values(["year_end", "district"])
        features = CLIMATE + STRUCTURE[crop] + ["year_trend"]
        X = features + ["district"]
        for year in TEST_YEARS:
            best = None
            for alpha, l1 in EN_GRID:
                errors = []
                for inner in range(2019, year):
                    train, test = g[g.year_end < inner], g[g.year_end == inner]
                    m = elastic_net(alpha, l1, features)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(train[X], train.yield_t_ha)
                    errors.append(test.yield_t_ha.to_numpy() - m.predict(test[X]))
                score = float(np.sqrt(np.mean(np.square(np.concatenate(errors)))))
                if best is None or score < best[0] - 1e-15:
                    best = (score, alpha, l1)
            rows.append(
                dict(
                    crop=crop,
                    outer_year=year,
                    training_through=year - 1,
                    n_train=int((g.year_end < year).sum()),
                    inner_start=2019,
                    inner_end=year - 1,
                    inner_rmse=best[0],
                    EN_alpha=best[1],
                    EN_l1_ratio=best[2],
                )
            )
    return pd.DataFrame(rows)


tune = nested_tuning(p)
assert len(tune) == 9
tune.to_csv(O / "00_elastic_net_tuning.csv", index=False)

# ---------------------------------------------------------------------------------------------------
# Climate-stress states: the 2023-2025 mean climate, with rain set to the district's 10th percentile
# (dry), temperatures to the 90th percentile (hot), or both. Every value stays inside the observed range.
# ---------------------------------------------------------------------------------------------------
stress = []
for (district, crop), g in p.groupby(["district", "crop"]):
    recent = g[g.year_end.isin(TEST_YEARS)]
    mean_climate = recent[STRESS_VARS].mean()
    low = g[STRESS_VARS[:2]].quantile(0.1)
    high = g[STRESS_VARS[2:]].quantile(0.9)
    for state in ["baseline", "dry", "hot", "dryhot"]:
        new = mean_climate.copy()
        if state in ("dry", "dryhot"):
            for f in STRESS_VARS[:2]:
                new[f] = min(new[f], low[f])
        if state in ("hot", "dryhot"):
            for f in STRESS_VARS[2:]:
                new[f] = max(new[f], high[f])
        row = {"district": district, "crop": crop, "state": state}
        for f in STRESS_VARS:
            assert g[f].min() - 1e-8 <= new[f] <= g[f].max() + 1e-8
            row[f] = new[f]
            row["delta_" + f] = new[f] - mean_climate[f]
        stress.append(row)
s = pd.DataFrame(stress)
assert len(s) == 192
s.to_csv(O / "01_climate_stress_states.csv", index=False)

# ---------------------------------------------------------------------------------------------------
# Two-way fixed effects (district and year), by double demeaning; descriptive, not causal
# ---------------------------------------------------------------------------------------------------
betas, fe_rows = {}, []
for crop in CROPS:
    g = p[p.crop == crop].copy()
    features = CLIMATE + ([STRUCTURE[crop][0]] if crop != "Wheat" else [])

    def demean(v):
        z = g[v].astype(float)
        return (
            z
            - g.groupby("district")[v].transform("mean")
            - g.groupby("year_end")[v].transform("mean")
            + z.mean()
        ).to_numpy()

    xx = np.stack([demean(f) for f in features], axis=1)
    yy = demean("yield_t_ha")
    b = np.linalg.lstsq(xx, yy, rcond=None)[0]
    betas[crop] = dict(zip(features, map(float, b)))
    for f, v in zip(features, b):
        fe_rows.append({"crop": crop, "term": f, "FE_coefficient": v})
pd.DataFrame(fe_rows).to_csv(O / "02_fixed_effects_coefficients.csv", index=False)

# ---------------------------------------------------------------------------------------------------
# Elastic Net and GBT per crop and test year. One GBT fit per crop and year is used for the test
# predictions, the SHAP values and the counterfactuals.
# ---------------------------------------------------------------------------------------------------
GBT_SETTINGS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=5, subsample=0.8, random_state=42
)
n_fits = 0
preds, shap_rows, cf_rows, fit_record = [], [], [], []
for crop in CROPS:
    g = p[p.crop == crop].sort_values(["year_end", "district"])
    features = CLIMATE + STRUCTURE[crop] + ["year_trend"]
    X = features + ["district"]
    for year in TEST_YEARS:
        train, test = g[g.year_end < year].copy(), g[g.year_end == year].copy()
        assert len(test) == 16 and train.year_end.max() < year
        cfg = tune[(tune.crop == crop) & (tune.outer_year == year)]
        assert len(cfg) == 1
        cfg = cfg.iloc[0]

        en = elastic_net(float(cfg.EN_alpha), float(cfg.EN_l1_ratio), features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            en.fit(train[X], train.yield_t_ha)
        gbt = Pipeline(
            [("pre", preprocessor(features)), ("model", GradientBoostingRegressor(**GBT_SETTINGS))]
        )
        gbt.fit(train[X], train.yield_t_ha)
        n_fits += 1

        climatology = test.district.map(train.groupby("district").yield_t_ha.mean()).to_numpy()
        en_pred, gbt_pred = en.predict(test[X]), gbt.predict(test[X])
        for i, (_, r) in enumerate(test.iterrows()):
            preds.append(
                {
                    "district": r.district,
                    "crop": crop,
                    "year_end": year,
                    "observed": r.yield_t_ha,
                    "climatology": climatology[i],
                    "ElasticNet": en_pred[i],
                    "GBT": gbt_pred[i],
                }
            )

        # TreeSHAP on the transformed test rows; the values must add up to the prediction
        transformed = np.asarray(gbt["pre"].transform(test[X]))
        explainer = shap.TreeExplainer(
            gbt["model"], feature_perturbation="tree_path_dependent", model_output="raw"
        )
        values = np.asarray(explainer.shap_values(transformed))
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        additivity = float(np.max(np.abs(values.sum(axis=1) + expected - gbt_pred)))
        assert additivity < 1e-6, ("SHAP additivity", crop, year, additivity)
        for i, (_, r) in enumerate(test.iterrows()):
            entry = {
                "district": r.district,
                "crop": crop,
                "year_end": year,
                "expected": expected,
                "gbt_prediction": gbt_pred[i],
                "district_aggregate": values[i, len(features) :].sum(),
            }
            for j, f in enumerate(features):
                entry[f] = values[i, j]
            shap_rows.append(entry)

        # The same GBT under the four climate states
        for state in ["baseline", "dry", "hot", "dryhot"]:
            x_new = test[X].copy()
            spec = s[(s.crop == crop) & (s.state == state)].set_index("district")
            for f in STRESS_VARS:
                x_new[f] = test.district.map(spec[f]).to_numpy()
            pv = gbt.predict(x_new)
            for i, (_, r) in enumerate(test.iterrows()):
                cf_rows.append(
                    {
                        "district": r.district,
                        "crop": crop,
                        "year_end": year,
                        "state": state,
                        "GBT_counterfactual": pv[i],
                    }
                )

        train_keys = "|".join(train.district.astype(str) + "/" + train.year_end.astype(str))
        fit_record.append(
            {
                "crop": crop,
                "year": year,
                "train_n": len(train),
                "fit_identity": f"{crop}-{year}-GBT-42",
                "train_key_digest": hashlib.sha256(train_keys.encode()).hexdigest(),
                "GBTrmse": np.sqrt(mean_squared_error(test.yield_t_ha, gbt_pred)),
                "SHAP_max_additivity_error": additivity,
                "EN_alpha": cfg.EN_alpha,
                "EN_l1_ratio": cfg.EN_l1_ratio,
            }
        )

pred = pd.DataFrame(preds).sort_values(["crop", "year_end", "district"])
sh, cf = pd.DataFrame(shap_rows), pd.DataFrame(cf_rows)
assert len(pred) == len(sh) == 144 and len(cf) == 576
pred.to_csv(O / "03_hindcast_predictions.csv", index=False)
sh.to_csv(O / "04_gbt_shap_values.csv", index=False)
cf.to_csv(O / "05_gbt_climate_counterfactuals.csv", index=False)
pd.DataFrame(fit_record).to_csv(O / "06_fit_record.csv", index=False)

# Accuracy of the test predictions
metrics = []
for crop, g in pred.groupby("crop"):
    rmse_clim = np.sqrt(np.mean(np.square(g.observed - g.climatology)))
    for name in ["climatology", "ElasticNet", "GBT"]:
        rmse = np.sqrt(np.mean(np.square(g.observed - g[name])))
        metrics.append(
            {
                "crop": crop,
                "model": name,
                "RMSE_t_ha": rmse,
                "skill_vs_climatology": 1 - rmse / rmse_clim,
                "R2": r2_score(g.observed, g[name]),
            }
        )
pd.DataFrame(metrics).to_csv(O / "07_hindcast_metrics.csv", index=False)

# Climate share of the mean absolute SHAP value
shap_summary = []
for crop, g in sh.groupby("crop"):
    cols = CLIMATE + STRUCTURE[crop] + ["year_trend", "district_aggregate"]
    mean_abs = g[cols].abs().mean()
    shap_summary.append(
        {
            "crop": crop,
            "climate_SHAP_share_pct": 100 * mean_abs[CLIMATE].sum() / mean_abs.sum(),
            "top_climate_feature": mean_abs[CLIMATE].idxmax(),
        }
    )
pd.DataFrame(shap_summary).to_csv(O / "08_shap_summary.csv", index=False)

# Baseline coefficients for the three-crop checks: 2023-2025 mean area, published NIR, mean EN yield
base = (
    p[p.year_end.isin(TEST_YEARS)]
    .groupby(["district", "crop"])
    .agg(area0=("area_ha", "mean"), nir=("nir_mm", "first"))
    .reset_index()
)
en_yield = pred.groupby(["district", "crop"]).ElasticNet.mean().rename("EN_OOS").reset_index()
base = base.merge(en_yield, on=["district", "crop"], validate="one_to_one")
assert len(base) == 48
base.to_csv(O / "09_baseline_coefficients.csv", index=False)

# Yield under each climate state: fixed-effects shift and GBT relative change applied to the EN yield
avg = cf.groupby(["district", "crop", "state"]).GBT_counterfactual.mean().rename("v").reset_index()
ref = avg[avg.state == "baseline"][["district", "crop", "v"]].rename(columns={"v": "baseline_gbt"})
avg = avg.merge(ref, on=["district", "crop"], validate="many_to_one")
avg["GBT_delta_relative"] = avg.v / avg.baseline_gbt - 1
resp = s.merge(
    avg[["district", "crop", "state", "GBT_delta_relative"]],
    on=["district", "crop", "state"],
    validate="one_to_one",
)
resp = resp.merge(base, on=["district", "crop"], validate="many_to_one")
resp["FE_delta"] = resp.apply(
    lambda x: sum(betas[x["crop"]][f] * x["delta_" + f] for f in STRESS_VARS), axis=1
)
resp["none_yield"] = resp.EN_OOS
resp["FE_yield"] = resp.EN_OOS + resp.FE_delta
resp["GBT_yield"] = resp.EN_OOS * (1 + resp.GBT_delta_relative)
assert (resp[["none_yield", "FE_yield", "GBT_yield"]] > 0).all().all()
resp.to_csv(O / "10_climate_response_brackets.csv", index=False)

# Can a district keep all of its Boro area within the budget? (three crops; does not depend on yields)
feas = []
for district, g in base.groupby("district"):
    z = g.set_index("crop").loc[CROPS]
    a, nir = z.area0.to_numpy(), z.nir.to_numpy()
    w0 = float(np.dot(a, 10 * nir))
    for rho in [1.0, 0.9, 0.8, 0.7]:
        possible = float(a[0] * nir[0] * 10) <= rho * w0 + max(0.01, 1e-10 * w0)
        feas.append({"district": district, "rho": rho, "strict_Boro_100pct_feasible": possible})
feas = pd.DataFrame(feas)
feas.to_csv(O / "11_boro_feasibility.csv", index=False)
assert [int((~feas[feas.rho == r].strict_Boro_100pct_feasible).sum()) for r in [1.0, 0.9, 0.8, 0.7]] == [
    0,
    5,
    12,
    14,
]

# How far each climate state lies from the observed years (standardised distance to the nearest year):
# the states combine marginal extremes that may not have occurred together
joint = []
for (district, crop), g in p.groupby(["district", "crop"]):
    x = g[STRESS_VARS].to_numpy()
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1)
    for _, row in s[(s.district == district) & (s.crop == crop)].iterrows():
        dist = np.linalg.norm((x - row[STRESS_VARS].to_numpy(float)) / sd, axis=1)
        joint.append(
            {
                "district": district,
                "crop": crop,
                "state": row.state,
                "joint_obs_exact": bool((dist < 1e-8).any()),
                "nearest_standardized_distance": float(dist.min()),
            }
        )
pd.DataFrame(joint).to_csv(O / "12_joint_support_diagnostic.csv", index=False)

info = {
    "input_panel_sha256": hashlib.sha256((I / "analysis_panel_full.csv").read_bytes()).hexdigest(),
    "tuning_file_sha256": hashlib.sha256((O / "00_elastic_net_tuning.csv").read_bytes()).hexdigest(),
    "unit_of_analysis": "district x crop x year",
    "panel_shape": list(p.shape),
    "gbt_fits": n_fits,
    "notes": [
        "Same-season climate: a hindcast, not a forecast available before sowing",
        "Fixed-effects coefficients are associations, not causal effects",
        "Climate states combine marginal extremes; see 12_joint_support_diagnostic.csv",
    ],
}
(O / "run_info.json").write_text(json.dumps(info, indent=2))
print("Elastic Net tuning\n", tune.to_string(index=False))
print(pd.DataFrame(metrics).round(4).to_string(index=False))
print("SHAP summary\n", pd.DataFrame(shap_summary).round(3).to_string(index=False))
print("Done:", n_fits, "GBT fits,", len(s), "climate states")
