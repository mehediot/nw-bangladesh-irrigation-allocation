#!/usr/bin/env python3
"""Stage 5 - decision-time planning: decision value of the yield forecasts and Gamma-robust water.

Cases: 16 districts x 3 target years (2023-2025) = 48. Everything the plan uses is known at t-2.

Model of one case (variables: area of Boro, wheat, potato, maize and fallow, as fractions of the pool)
  pool          L = sum of the crop areas of year t-2 (the four crops)
  land          x_B + x_W + x_P + x_M + u = 1
  water         sum_c (n_c + d_c [c in S]) x_c L <= rho W0   for every set S of Gamma crops
                W0 = sum_c n_c a_c, the nominal demand of the year t-2 areas a_c
  Boro floor    x_B L >= 0.8 a_B
  crop limits   x_c L <= largest area of crop c observed in the years up to t-2
  plans         'least' : fallow first set to its smallest feasible value, then the objective
                'free'  : fallow unrestricted
  objectives    food energy (kcal) or net return (Tk; price minus full production cost, called
                gross_margin in the code and outputs), both linear in the areas

Yield sources for the plan: Elastic Net, persistence, climatology, district trend (all t-2 information) and
the oracle (true yields; a ceiling, not a strategy), all from stage 4 for the four crops. Every plan is scored
ex post with the true yields. Regret = shortfall of the plan's true
outcome from the oracle's: % of the oracle's for food energy; Tk per hectare of the pool for gross margin
(the oracle's gross margin can be zero or negative, so a percentage is not defined there).

Outputs (generated/decision/):
  decision_plans.csv                 every case x objective x plan x NIR set x Gamma x budget x yield source
  decision_value_summary.csv         feasibility and regret by source
  decision_value_tests.csv           Elastic Net vs each benchmark: paired regret difference, district-block
                                     bootstrap 95% interval, Diebold-Mariano-type t test over districts
  robust_thresholds.csv              feasible cases and budget needed to hold 80% / 90% of food energy, by NIR set and Gamma
  break_even_forecast_error.csv      smallest error in one crop's yield that changes the plan, per case
  break_even_summary.csv             the same next to the actual forecast errors
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

cli = argparse.ArgumentParser(description="Stage 5 - decision value and robust planning")
cli.add_argument("--input-dir", type=Path, required=True)
cli.add_argument("--generated-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
args = cli.parse_args()
I, G, O = args.input_dir, args.generated_dir, args.output_dir
O.mkdir(parents=True, exist_ok=True)

CROPS = ["Boro Rice", "Wheat", "Potato", "Maize"]
NC = len(CROPS)
YEARS = [2023, 2024, 2025]
# Mainuddin et al. 2014 (CSIRO), Tables 4.1-4.2: nominal NIR (midpoint of the district range; Boro mean),
# spatial deviation = upper end of the range - nominal, temporal CV = year-to-year coefficient of variation
N0 = np.array([676.0, 283.0, 376.5, 354.0])
NUP = np.array([779.0, 339.0, 444.0, 429.0])
CV = np.array([0.072, 0.137, 0.126, 0.148])
DEV = {"temporal_cv": N0 * CV, "spatial_upper": NUP - N0}
SPECS = [("nominal", 0)] + [(s, g) for s in DEV for g in (1, 2, 3, 4)]
KCAL = np.array([3450.0, 3440.0, 554.4, 3440.0])  # husked rice, wheat, potato, maize (FCT 2013)
FLOOR = 0.8
RHOS = (0.8, 0.9, 1.0)
SOURCES = ["EN_cutoff2", "Persistence_cutoff2", "Climatology_cutoff2", "Trend_cutoff2", "Oracle"]
BENCH = ["Persistence_cutoff2", "Climatology_cutoff2", "Trend_cutoff2"]
PLANS = ("least", "free")
OBJECTIVES = ("food_energy", "gross_margin")
SEED, NBOOT = 20261113, 5000

# ---------------------------------------------------------------------------
# Inputs: forecasts from stage 4, crop areas from the yearbook panels, economics
# ---------------------------------------------------------------------------
panel = pd.read_csv(I / "analysis_panel_full.csv")
maize = pd.read_csv(I / "maize_rabi_16districts_2015_2025.csv")
fc = pd.read_csv(G / "forecasts" / "decision_time_forecasts.csv")
econ = pd.read_csv(I / "crop_economics_scenarios.csv").set_index(["crop", "item"]).value
assert len(fc) == 192 and len(maize) == 176
DISTRICTS = sorted(panel.district.unique())
area = pd.concat(
    [
        panel[["district", "crop", "year_end", "area_ha", "yield_t_ha"]],
        maize[["district", "crop", "year_end", "area_ha", "yield_t_ha"]],
    ]
)
A = area.pivot_table(index=["district", "year_end"], columns="crop", values="area_ha")[CROPS]
Y = area.pivot_table(index=["district", "year_end"], columns="crop", values="yield_t_ha")[CROPS]

MILL = float(econ[("Boro Rice", "paddy_to_husked_rice")])
SOLD = np.array([1000 / MILL, 1000.0, 1000.0, 1000.0])  # kg sold per tonne of reported yield
PRICE = np.array(
    [
        econ[("Boro Rice", "price")],
        econ[("Wheat", "price")],
        econ[("Potato", "price_mid")],
        econ[("Maize", "price_mid")],
    ]
)
COST_KG = np.array([econ[(c, "cost")] for c in CROPS])

cases = []
for t in YEARS:
    # cost per hectare: cost per kg x regional mean yield of year t-2 (known at decision time), same in every district
    cost_ha = COST_KG * Y.xs(t - 2, level="year_end").mean().to_numpy() * SOLD
    for d in DISTRICTS:
        g = fc[(fc.district == d) & (fc.year_end == t)].set_index("crop").loc[CROPS]
        a = A.loc[(d, t - 2)].to_numpy(float)
        assert np.allclose(a, g.lag2_area.to_numpy(float))
        ys = {s: (g[s] if s != "Oracle" else g.yield_t_ha).to_numpy(float) for s in SOURCES}
        hist = A.loc[d]
        cap = np.array([hist.loc[hist.index <= t - 2, c].max() for c in CROPS])
        assert (cap >= a - 1e-9).all() and (a > 0).all()
        cases.append(
            dict(
                year_end=t,
                district=d,
                a=a,
                L=float(a.sum()),
                W0=float(a @ N0),
                cap=cap,
                ys=ys,
                cost_ha=cost_ha,
            )
        )
assert len(cases) == 48


# ---------------------------------------------------------------------------
# The linear programme (variables as fractions of the pool)
# ---------------------------------------------------------------------------
def water_rows(c, spec, gamma):
    if gamma == 0:
        return [N0 * c["L"] / c["W0"]]
    rows = []
    for S in combinations(range(NC), gamma):
        n = N0.copy()
        n[list(S)] += DEV[spec][list(S)]
        rows.append(n * c["L"] / c["W0"])
    return rows


def lp(c, spec, gamma, rho, cost, u_max=None):
    rows = [np.r_[w, 0.0] for w in water_rows(c, spec, gamma)]
    rhs = [rho] * len(rows)
    if u_max is not None:
        rows.append(np.r_[np.zeros(NC), 1.0])
        rhs.append(u_max)
    lo = np.r_[FLOOR * c["a"][0] / c["L"], np.zeros(NC - 1)]
    hi = c["cap"] / c["L"]
    bounds = [(lo[k], hi[k]) for k in range(NC)] + [(0, None)]
    s = linprog(
        cost,
        A_ub=np.array(rows),
        b_ub=rhs,
        A_eq=np.ones((1, NC + 1)),
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    return s if s.status == 0 else None


def gain(c, y, objective):
    """Objective per hectare under yields y."""
    if objective == "food_energy":
        return 1000 * y * KCAL
    return PRICE * y * SOLD - c["cost_ha"]


MIN_U = {}


def min_fallow(c, spec, gamma, rho):
    key = (c["year_end"], c["district"], spec, gamma, rho)
    if key not in MIN_U:
        s = lp(c, spec, gamma, rho, np.r_[np.zeros(NC), 1.0])
        MIN_U[key] = None if s is None else float(s.x[NC])
    return MIN_U[key]


def plan(c, spec, gamma, rho, y, objective, kind):
    """Areas (ha) and fallow (ha) of the optimal plan under forecast yields y; None if infeasible."""
    u_max = None
    if kind == "least":
        um = min_fallow(c, spec, gamma, rho)
        if um is None:
            return None
        u_max = um * (1 + 1e-9) + 1e-9
    s = lp(c, spec, gamma, rho, -np.r_[gain(c, y, objective) / 1e6, 0.0], u_max)
    if s is None:
        return None
    x = s.x[:NC] * c["L"]
    w = max(r @ s.x[:NC] for r in water_rows(c, spec, gamma))
    assert w <= rho + 1e-7 and x[0] >= FLOOR * c["a"][0] * (1 - 1e-7) and (x <= c["cap"] * (1 + 1e-7)).all()
    return x, float(s.x[NC] * c["L"]), float(w)


# ---------------------------------------------------------------------------
# 1. Plans for every case, scored ex post with the true yields
# ---------------------------------------------------------------------------
rows = []
for c in cases:
    truth = c["ys"]["Oracle"]
    for spec, gamma in SPECS:
        for rho in RHOS:
            for objective in OBJECTIVES:
                for kind in PLANS:
                    res = {s: plan(c, spec, gamma, rho, c["ys"][s], objective, kind) for s in SOURCES}
                    true_gain = gain(c, truth, objective)
                    oracle = float(res["Oracle"][0] @ true_gain) if res["Oracle"] else np.nan
                    for s in SOURCES:
                        r = dict(
                            year_end=c["year_end"],
                            district=c["district"],
                            uncertainty_set=spec,
                            gamma=gamma,
                            budget_ratio=rho,
                            objective=objective,
                            plan=kind,
                            yield_source=s,
                            feasible=res[s] is not None,
                            pool_ha=c["L"],
                            reference_water_ha_mm=c["W0"],
                        )
                        if res[s]:
                            x, u, w = res[s]
                            realised = float(x @ true_gain)
                            r.update(
                                boro_ha=x[0],
                                wheat_ha=x[1],
                                potato_ha=x[2],
                                maize_ha=x[3],
                                fallow_ha=u,
                                fallow_pct=100 * u / c["L"],
                                robust_water_ratio=w,
                                nominal_water_ratio=float(x @ N0 / c["W0"]),
                                planned_value=float(x @ gain(c, c["ys"][s], objective)),
                                realised_value=realised,
                                oracle_value=oracle,
                                realised_food_energy_Gkcal=float(1000 * x @ (truth * KCAL)) / 1e9,
                                realised_gross_margin_million_Tk=float(x @ gain(c, truth, "gross_margin"))
                                / 1e6,
                                regret_pct=100 * (oracle - realised) / oracle if oracle > 0 else np.nan,
                                regret_per_ha=(oracle - realised) / c["L"],
                                loss=(
                                    100 * (oracle - realised) / oracle
                                    if objective == "food_energy"
                                    else (oracle - realised) / c["L"]
                                ),
                                loss_unit="% of oracle" if objective == "food_energy" else "Tk per ha",
                                plan_differs_from_EN=bool(
                                    res["EN_cutoff2"] is not None
                                    and np.abs(x - res["EN_cutoff2"][0]).max() > 1e-6 * c["L"]
                                ),
                            )
                        rows.append(r)
P = pd.DataFrame(rows)
assert len(P) == 48 * len(SPECS) * len(RHOS) * len(OBJECTIVES) * len(PLANS) * len(SOURCES)
# feasibility depends only on land, water, floor and limits, never on the yields
assert (
    P.groupby(["year_end", "district", "uncertainty_set", "gamma", "budget_ratio"])
    .feasible.nunique()
    .eq(1)
    .all()
)
assert (P.regret_pct.dropna() > -1e-4).all() and (
    P.regret_per_ha.dropna() > -1e-2
).all()  # no plan beats the oracle
assert P[P.feasible].loss.notna().all()
P.to_csv(O / "decision_plans.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Summary by yield source
# ---------------------------------------------------------------------------
KEYS = ["uncertainty_set", "gamma", "budget_ratio", "objective", "plan"]
summ = (
    P.groupby(KEYS + ["yield_source"], sort=False)
    .agg(
        cases=("feasible", "size"),
        feasible_cases=("feasible", "sum"),
        loss_unit=("loss_unit", "first"),
        mean_regret=("loss", "mean"),
        median_regret=("loss", "median"),
        max_regret=("loss", "max"),
        cases_plan_differs_from_EN=("plan_differs_from_EN", "sum"),
        mean_fallow_pct=("fallow_pct", "mean"),
        realised_food_energy_Tkcal=("realised_food_energy_Gkcal", lambda v: v.sum() / 1e3),
        realised_gross_margin_billion_Tk=("realised_gross_margin_million_Tk", lambda v: v.sum() / 1e3),
    )
    .reset_index()
)
summ.to_csv(O / "decision_value_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Elastic Net vs each benchmark: paired regret difference over the common feasible cases
# ---------------------------------------------------------------------------
# loss = regret (food energy: % of oracle; gross margin: Tk per ha). d = loss(EN) - loss(benchmark); negative
# favours the Elastic Net.
# Interval: district-block bootstrap (16 districts resampled with replacement, the 3 years of a district kept
# together). Test: Diebold-Mariano-type t statistic on the 16 district means of d (blocks remove the
# dependence between the years of one district), two-sided, 15 degrees of freedom.
rng = np.random.default_rng(SEED)
test_rows = []
for key, g in P[P.feasible].groupby(KEYS, sort=False):
    wide = g.pivot_table(index=["district", "year_end"], columns="yield_source", values="loss")
    val = g.pivot_table(index=["district", "year_end"], columns="yield_source", values="realised_value")
    for b in BENCH:
        pair = wide[["EN_cutoff2", b]].dropna()
        row = dict(
            zip(KEYS, key), EN_vs=b, n_cases=len(pair), n_districts=pair.index.get_level_values(0).nunique()
        )
        if len(pair):
            dd = (pair["EN_cutoff2"] - pair[b]).groupby(level=0).mean()
            boot = dd.to_numpy()[rng.integers(0, len(dd), size=(NBOOT, len(dd)))].mean(axis=1)
            v = val.loc[pair.index]
            sd = dd.std(ddof=1)
            tstat = float(dd.mean() / (sd / np.sqrt(len(dd)))) if sd > 0 else np.nan
            row.update(
                loss_unit=g.loss_unit.iloc[0],
                mean_regret_EN=float(pair["EN_cutoff2"].mean()),
                mean_regret_benchmark=float(pair[b].mean()),
                mean_difference=float((pair["EN_cutoff2"] - pair[b]).mean()),
                bootstrap95_low=float(np.quantile(boot, 0.025)),
                bootstrap95_high=float(np.quantile(boot, 0.975)),
                dm_t=tstat,
                dm_p_two_sided=(
                    float(2 * stats.t.sf(abs(tstat), len(dd) - 1)) if np.isfinite(tstat) else np.nan
                ),
                cases_EN_better=int((pair["EN_cutoff2"] < pair[b] - 1e-9).sum()),
                cases_benchmark_better=int((pair["EN_cutoff2"] > pair[b] + 1e-9).sum()),
                realised_total_difference_pct=float(100 * (v["EN_cutoff2"].sum() / v[b].sum() - 1)),
            )
        test_rows.append(row)
TESTS = pd.DataFrame(test_rows)
TESTS.to_csv(O / "decision_value_tests.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Robust thresholds: feasible cases, and the budget needed to hold a share of food energy
# ---------------------------------------------------------------------------
# Inverse problem, per case: smallest rho such that the plan (forecast yields of the Elastic Net source)
# keeps planned food energy >= target x food energy of the year t-2 areas under the same forecast.
def budget_needed(c, spec, gamma, target):
    y = c["ys"]["EN_cutoff2"]
    e = 1000 * y * KCAL * c["L"]
    e0 = float(1000 * c["a"] @ (y * KCAL))
    rows = [np.r_[w, 0.0, -1.0] for w in water_rows(c, spec, gamma)]
    rhs = [0.0] * len(rows)
    rows.append(np.r_[-e / e0, 0.0, 0.0])
    rhs.append(-target)
    lo = np.r_[FLOOR * c["a"][0] / c["L"], np.zeros(NC - 1)]
    hi = c["cap"] / c["L"]
    bounds = [(lo[k], hi[k]) for k in range(NC)] + [(0, None), (0, None)]
    s = linprog(
        np.r_[np.zeros(NC + 1), 1.0],
        A_ub=np.array(rows),
        b_ub=rhs,
        A_eq=np.r_[np.ones(NC + 1), 0.0][None],
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    return float(s.x[-1]) if s.status == 0 else np.nan


thr = []
for spec, gamma in SPECS:
    row = dict(uncertainty_set=spec, gamma=gamma)
    f = P[
        (P.uncertainty_set == spec)
        & (P.gamma == gamma)
        & (P.yield_source == "EN_cutoff2")
        & (P.objective == "food_energy")
        & (P.plan == "least")
    ]
    for rho in RHOS:
        row[f"feasible_cases_of_48_at_budget_{rho:.2f}"] = int(
            f[np.isclose(f.budget_ratio, rho)].feasible.sum()
        )
        # necessary condition, independent of the solver: the Boro floor alone must fit the budget
        need = [
            max(r @ np.r_[FLOOR * c["a"][0] / c["L"], np.zeros(NC - 1)] for r in water_rows(c, spec, gamma))
            for c in cases
        ]
        assert row[f"feasible_cases_of_48_at_budget_{rho:.2f}"] == int(sum(n <= rho + 1e-9 for n in need))
    for target in (0.8, 0.9):
        v = np.array([budget_needed(c, spec, gamma, target) for c in cases])
        assert np.isfinite(v).all()
        row[f"budget_needed_to_hold_{int(target*100)}pct_median"] = float(np.median(v))
        row[f"budget_needed_to_hold_{int(target*100)}pct_worst_case"] = float(v.max())
        row[f"cases_holding_{int(target*100)}pct_at_budget_0.80"] = int((v <= 0.8 + 1e-9).sum())
    thr.append(row)
pd.DataFrame(thr).to_csv(O / "robust_thresholds.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Break-even forecast error: how wrong must one crop's yield be before the plan changes?
# ---------------------------------------------------------------------------
# Start from the true yields (the oracle plan). Scale the yield of one crop by (1 + e) or (1 - e), e from
# 0.01 upward, and record the smallest e at which the plan changes, and the regret that plan then has.
# Nominal NIR, both fallow plans, budgets 0.80 and 1.00. e is searched on a 0.02 grid up to 1.00 (up to 0.98
# downward) and refined by bisection to 0.0002; no change within the range is reported as missing (never).
GRID = np.round(np.arange(0.02, 1.0001, 0.02), 2)
be_rows = []
for c in cases:
    truth = c["ys"]["Oracle"]
    for rho, kind, objective in [(r, k, o) for r in (0.8, 1.0) for k in PLANS for o in OBJECTIVES]:
        base = plan(c, "nominal", 0, rho, truth, objective, kind)
        oracle = float(base[0] @ gain(c, truth, objective))
        for k, crop in enumerate(CROPS):
            for sign in (+1, -1):
                found, regret = np.nan, np.nan
                lo_e = 0.0
                for e in GRID:
                    if sign < 0 and e >= 1:
                        break
                    y = truth.copy()
                    y[k] *= 1 + sign * e
                    x = plan(c, "nominal", 0, rho, y, objective, kind)[0]
                    if np.abs(x - base[0]).max() > 1e-6 * c["L"]:
                        # refine between the last unchanged and the first changed grid point
                        hi_e = e
                        for _ in range(7):
                            mid = (lo_e + hi_e) / 2
                            y = truth.copy()
                            y[k] *= 1 + sign * mid
                            xm = plan(c, "nominal", 0, rho, y, objective, kind)[0]
                            if np.abs(xm - base[0]).max() > 1e-6 * c["L"]:
                                hi_e, x = mid, xm
                            else:
                                lo_e = mid
                        found = hi_e
                        regret = (oracle - float(x @ gain(c, truth, objective))) / c["L"]
                        break
                    lo_e = e
                be_rows.append(
                    dict(
                        year_end=c["year_end"],
                        district=c["district"],
                        budget_ratio=rho,
                        plan=kind,
                        objective=objective,
                        crop=crop,
                        direction="over" if sign > 0 else "under",
                        break_even_relative_error=found,
                        regret_at_break_even_per_ha=regret,
                    )
                )
BE = pd.DataFrame(be_rows)
BE.to_csv(O / "break_even_forecast_error.csv", index=False)
# actual relative errors of the decision-time forecasts, same cases
err = []
for c in cases:
    truth = c["ys"]["Oracle"]
    for s in BENCH + ["EN_cutoff2"]:
        for k, crop in enumerate(CROPS):
            err.append(
                dict(
                    year_end=c["year_end"],
                    district=c["district"],
                    crop=crop,
                    yield_source=s,
                    relative_error=float(c["ys"][s][k] / truth[k] - 1),
                )
            )
ERR = pd.DataFrame(err)
bes = []
for (rho, kind, objective, crop), g in BE.groupby(["budget_ratio", "plan", "objective", "crop"], sort=False):
    th = g.pivot(index=["district", "year_end"], columns="direction", values="break_even_relative_error")
    th = th.reindex(columns=["over", "under"])
    assert len(th) == 48
    row = dict(
        budget_ratio=rho,
        plan=kind,
        objective=objective,
        crop=crop,
        cases_plan_never_changes=int(th.isna().all(axis=1).sum()),
        median_break_even_over=float(th["over"].median()),
        median_break_even_under=float(th["under"].median()),
        median_regret_at_break_even_per_ha=float(g.regret_at_break_even_per_ha.median()),
    )
    for s in ["EN_cutoff2", "Persistence_cutoff2"]:
        e = (
            ERR[(ERR.crop == crop) & (ERR.yield_source == s)]
            .set_index(["district", "year_end"])
            .relative_error.reindex(th.index)
        )
        bound = np.where(e > 0, th["over"], th["under"])
        row[f"median_abs_error_{s}"] = float(e.abs().median())
        row[f"cases_error_beyond_break_even_{s}"] = int((np.abs(e) >= np.nan_to_num(bound, nan=np.inf)).sum())
    bes.append(row)
pd.DataFrame(bes).to_csv(O / "break_even_summary.csv", index=False)

pd.set_option("display.width", 250)
print(
    summ[(summ.uncertainty_set == "nominal") & np.isclose(summ.budget_ratio, 0.8)]
    .round(3)
    .to_string(index=False)
)
print(TESTS[(TESTS.uncertainty_set == "nominal")].round(3).to_string(index=False))
print(pd.DataFrame(thr).round(3).to_string(index=False))
print(pd.DataFrame(bes).round(3).to_string(index=False))
print("DECISION COMPLETE")
