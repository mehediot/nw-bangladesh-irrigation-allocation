#!/usr/bin/env python3
"""Stage 6 - district screening model and its scenarios, plus robustness checks of stages 1-3.

The district allocation model (section 4) is a linear program per district: areas of Boro, wheat,
potato and maize and fallow on a fixed land pool (2023-2025 mean harvested area), a water budget equal
to a ratio rho of the baseline demand, a floor on the Boro area and upper limits on each crop. The
sections after it run the model under different budgets, objectives, prices, water requirements,
AWD adoption, crop limits and crop sets. Sections 1-3 and 6 check the hindcast models and RAWSI.

In the code and outputs `gross_margin` is price minus full production cost; the paper calls it net
return.

Outputs (generated/extensions/):
  ext_01_climate_cells.csv            which districts share one NASA POWER series
  ext_02_fixed_effects.csv            two-way fixed-effects coefficients, clustered SE, within-R2
  ext_03_spatial_validation.csv       GBT leave-one-district-out and leave-one-climate-cell-out
  ext_04_minimum_fallow.csv           smallest fallow share each district needs, by Boro floor and budget ratio
  ext_05_pm_deficit_sensitivity.csv   feasibility and allocation with the FAO-56 crop water requirement (daily PM)
  ext_06_pm_deficit_district.csv      district detail behind ext_05
  ext_07_rawsi_sensitivity_ranks.csv  RAWSI rank of every district under every variant
  ext_08_rawsi_sensitivity_summary.csv agreement of every variant with the primary ranking
  ext_09_rawsi_rank_stability.csv     per-district rank range and top-5 membership
  ext_10_district_boro_share.csv      Boro share of water demand next to RAWSI rank
  ext_11_objective_by_yield_source.csv food-energy and gross-margin plans with yields from five sources
  ext_12_gross_margin_objective.csv   food-energy and gross-margin plans (free / least fallow), valued at three potato prices
  ext_13_margin_per_hectare.csv       price, cost and margin per hectare behind ext_12
  ext_14_potato_price_sweep.csv       gross-margin plan as the potato price moves from Tk 6 to Tk 34 per kg
  ext_15_shadow_prices.csv            value of water, land and fallow, cost of the Boro floor, by district
  ext_16_budget_response.csv          outputs of each objective as the budget ratio moves from 0.78 to 1.00
  ext_17_energy_margin_frontier.csv   trade-off frontier between food energy and gross margin
  ext_18_regional_yield_area_by_year.csv  regional yield, district range and area by crop and year
  ext_19_yield_climate_correlation.csv    correlation of yield with the climate predictors
  ext_20_wheat_cost_sensitivity.csv       gross-margin plans with the wheat cost at -20%, reported, +20%
  ext_21_crop_limit_sensitivity.csv       plans with crop limits at the observed maximum vs 1.5 x baseline
  ext_22_maize_price_cost_sensitivity.csv gross-margin plans under three maize prices and three maize costs
  ext_23_three_vs_four_crops.csv          the same plans with and without maize in the model
  ext_24_procurement_price_sensitivity.csv gross-margin plans with Boro and wheat at the procurement price
  ext_25_awd_scenarios.csv                plans when part of the Boro area uses alternate wetting and drying (AWD)
  ext_26_awd_fallow_curve.csv             smallest fallow and full-Boro feasibility as AWD adoption rises
  ext_27_water_value_vs_charges.csv       model value of water (gross margin) next to irrigation charges farmers pay
  ext_28_sensitivity_tornado.csv          one-factor-at-a-time sensitivities of three outcomes at budget 0.80
  ext_29_low_water_crops.csv              six-crop variant: how far lentil and mustard must expand to absorb released land
  ext_30_floor_limit_grid.csv             Boro floor x crop-area limits, crossed (budget 0.80 and 0.90)
  ext_31_awd_yield_grid.csv               AWD water saving x Boro yield effect, crossed (budget 0.80)
  ext_32_land_accounting_check.csv        largest irrigated area with no land balance (budgets 0.80 and 0.90)
  ext_33_rule_baselines.csv               two simple allocation rules against the optimised plans
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

cli = argparse.ArgumentParser(description="Stage 6 - allocation scenarios")
cli.add_argument("--input-dir", type=Path, required=True)
cli.add_argument("--generated-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
args = cli.parse_args()
I, G, O = args.input_dir, args.generated_dir, args.output_dir
O.mkdir(parents=True, exist_ok=True)

CROPS = ["Boro Rice", "Wheat", "Potato"]
KEYS = ["district", "crop", "year_end"]
NIR = {"Boro Rice": 676.0, "Wheat": 283.0, "Potato": 376.5}  # mm, fixed coefficients
KCAL = np.array([3450.0, 3440.0, 554.4])  # kcal per kg: husked rice (BBS yield basis), wheat, potato
CL = ["season_rain_mm", "monsoon_prev_rain_mm", "terminal_tmax_c", "season_tmin_c", "season_rh_pct"]
STRUCT = {"Boro Rice": ["hybrid_area_share"], "Wheat": ["wheat_postbreak"], "Potato": ["hyv_area_share"]}
ALL_CLIMATE = [
    "season_rain_mm",
    "monsoon_prev_rain_mm",
    "season_tmean_c",
    "season_tmax_c",
    "season_tmin_c",
    "terminal_tmax_c",
    "season_rh_pct",
]
RECENT = [2023, 2024, 2025]

panel = pd.read_csv(I / "analysis_panel_full.csv")
cache = pd.read_csv(
    G / "climate" / "pm_seasonal_fao56_16districts_3crops_2015_2025.csv"
)  # FAO-56, daily (stage 1)
wr4 = pd.read_csv(G / "climate" / "crop_water_requirement_fao56.csv")
gw = pd.read_csv(I / "hodgson_2021_groundwater_trends.csv")
base = pd.read_csv(G / "hindcast" / "09_baseline_coefficients.csv")
primary = pd.read_csv(G / "scarcity_index" / "rawsi.csv")
assert panel.shape == (528, 28) and len(cache) == 528 and len(gw) == 16 and len(base) == 48
assert not panel.duplicated(KEYS).any() and not cache.duplicated(KEYS).any()
panel["year_trend"] = panel.year_end - panel.year_end.min()
panel["wheat_postbreak"] = (panel.year_end >= 2018).astype(int)
DISTRICTS = sorted(panel.district.unique())
assert len(DISTRICTS) == 16

# ---------------------------------------------------------------------------
# 1. Climate cells: districts whose NASA POWER series are identical
# ---------------------------------------------------------------------------
signature = {}
for d, g in panel.sort_values(["crop", "year_end"]).groupby("district"):
    signature[d] = tuple(np.round(g[ALL_CLIMATE].to_numpy().ravel(), 6))
members = {}
for d in DISTRICTS:
    members.setdefault(signature[d], []).append(d)
cells = sorted(members.values(), key=lambda m: m[0])
assert len(cells) == 11, len(cells)
cell_of = {d: f"C{i:02d}" for i, m in enumerate(cells, 1) for d in m}
cell_table = pd.DataFrame(
    [
        dict(
            district=d,
            climate_cell=cell_of[d],
            cell_members="; ".join(next(m for m in cells if d in m)),
            cell_size=len(next(m for m in cells if d in m)),
        )
        for d in DISTRICTS
    ]
)
# The effective-rainfall series of the Penman-Monteith cache follows the same grouping.
rain_sig = {
    d: tuple(np.round(g.effective_rain_monthly_mm.to_numpy(), 6))
    for d, g in cache.sort_values(["crop", "year_end"]).groupby("district")
}
cell_table["same_grouping_in_pm_cache_rain"] = [
    all(rain_sig[d] == rain_sig[o] for o in DISTRICTS if cell_of[o] == cell_of[d])
    and all(rain_sig[d] != rain_sig[o] for o in DISTRICTS if cell_of[o] != cell_of[d])
    for d in DISTRICTS
]
assert cell_table.same_grouping_in_pm_cache_rain.all()
cell_table.to_csv(O / "ext_01_climate_cells.csv", index=False)
panel["climate_cell"] = panel.district.map(cell_of)

# ---------------------------------------------------------------------------
# 2. Two-way fixed effects (district + year), SE clustered by district
# ---------------------------------------------------------------------------
fe_rows = []
for crop in CROPS:
    g = panel[panel.crop == crop].sort_values(["year_end", "district"]).reset_index(drop=True)
    feats = CL + (STRUCT[crop] if crop != "Wheat" else [])  # wheat break dummy is absorbed by year effects

    def within(col):
        z = g[col].astype(float)
        return (
            z
            - g.groupby("district")[col].transform("mean")
            - g.groupby("year_end")[col].transform("mean")
            + z.mean()
        ).to_numpy()

    X = np.column_stack([within(f) for f in feats])
    y = within("yield_t_ha")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    within_r2 = 1 - (resid @ resid) / (y @ y)
    n, k = X.shape
    n_d, n_t = g.district.nunique(), g.year_end.nunique()
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for _, idx in g.groupby("district").indices.items():
        score = X[idx].T @ resid[idx]
        meat += np.outer(score, score)
    # small-sample correction: clusters, and parameters including both sets of fixed effects
    dof_resid = n - k - (n_d - 1) - (n_t - 1) - 1
    correction = n_d / (n_d - 1) * (n - 1) / dof_resid
    se = np.sqrt(np.diag(bread @ meat @ bread) * correction)
    pval = 2 * stats.t.sf(np.abs(beta / se), n_d - 1)
    for f, b, s, p in zip(feats, beta, se, pval):
        fe_rows.append(
            dict(crop=crop, predictor=f, coef=b, cluster_se=s, p_cluster_df=p, within_R2=within_r2)
        )
fe = pd.DataFrame(fe_rows)
fe.to_csv(O / "ext_02_fixed_effects.csv", index=False)
# must agree with the coefficients stage 3 uses for the climate stress brackets
stage2 = pd.read_csv(G / "hindcast" / "02_fixed_effects_coefficients.csv")
chk = fe.merge(stage2, left_on=["crop", "predictor"], right_on=["crop", "term"], validate="one_to_one")
assert len(chk) == len(fe) and np.abs(chk.coef - chk.FE_coefficient).max() < 1e-10


# ---------------------------------------------------------------------------
# 3. Spatial validation of the GBT: hold out one district, then one climate cell
# ---------------------------------------------------------------------------
def spatial_cv(crop, group_col):
    """Hold out every group in turn; the model never sees the held-out districts."""
    g = panel[panel.crop == crop].sort_values(["district", "year_end"]).reset_index(drop=True)
    feats = CL + STRUCT[crop] + ["year_trend"]
    pred = np.full(len(g), np.nan)
    for grp in sorted(g[group_col].unique()):
        test = g[group_col] == grp
        pre = ColumnTransformer(
            [("num", "passthrough", feats), ("id", OneHotEncoder(handle_unknown="ignore"), ["district"])]
        )
        gbt = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        gbt.fit(pre.fit_transform(g.loc[~test, feats + ["district"]]), g.loc[~test, "yield_t_ha"])
        pred[test.to_numpy()] = gbt.predict(pre.transform(g.loc[test, feats + ["district"]]))
    assert np.isfinite(pred).all()
    rmse = float(np.sqrt(np.mean((g.yield_t_ha - pred) ** 2)))
    return dict(
        crop=crop,
        n_folds=g[group_col].nunique(),
        n_obs=len(g),
        GBT_spatial_RMSE=rmse,
        GBT_spatial_R2=float(r2_score(g.yield_t_ha, pred)),
    )


sp_rows = []
for scheme, col in [("leave_one_district_out", "district"), ("leave_one_climate_cell_out", "climate_cell")]:
    for crop in CROPS:
        sp_rows.append(dict(scheme=scheme, **spatial_cv(crop, col)))
spatial = pd.DataFrame(sp_rows)
spatial.to_csv(O / "ext_03_spatial_validation.csv", index=False)

# ---------------------------------------------------------------------------
# 4. District crop-allocation model with explicit fallow land
# ---------------------------------------------------------------------------
# Variables per district: area of Boro, wheat, potato and maize (ha) and fallow u (ha).
#   land balance  x_B + x_W + x_P + x_M + u = A     every hectare of the land pool is cropped or fallow;
#                                                    A = baseline area of the four crops (mean 2023-2025)
#   water         10 * sum_c n_c x_c <= rho * W0    W0 = baseline demand of the district
#   Boro floor    x_B >= floor * a_B
#   crop limits   x_c <= U_c                         largest area observed in the district, 2015-2025
#                                                    (1.5 x baseline kept as a sensitivity case)
#   fallow plan   'free'  : u >= 0 only
#                 'least' : u <= u_min, the smallest fallow the water budget allows, solved first
# Yields are the decision-time Elastic Net forecasts (mean of 2023-2025) unless stated otherwise.
# Maize is the fourth crop of the allocation model (not of the hindcast models or SHAP). Its decision-time
# forecasts come from stage 4 like the other crops; the hindcast source has no maize model, so maize takes
# its decision-time Elastic Net forecast there. Sources: evidence/maize/README.md.
CROPS_A = CROPS + ["Maize"]  # crops of the allocation model
NC = len(CROPS_A)
NIR_A = {**NIR, "Maize": 354.0}  # Mainuddin et al. 2014, Table 4.2: 279-429 mm, midpoint
KCAL_A = np.r_[KCAL, 3440.0]  # FCT 2013, 01_0005 maize, yellow, dried: 344 kcal/100 g, EP 1.00
maize = pd.read_csv(I / "maize_rabi_16districts_2015_2025.csv")
assert len(maize) == 176 and maize.district.nunique() == 16 and set(maize.district) == set(panel.district)
fc = pd.read_csv(G / "forecasts" / "decision_time_forecasts.csv")
assert len(fc) == 192 and set(fc.year_end) == set(RECENT)
src = fc.groupby(["district", "crop"])[
    ["EN_cutoff2", "Persistence_cutoff2", "Climatology_cutoff2", "yield_t_ha"]
].mean()
hind = pd.concat(
    [
        base.set_index(["district", "crop"]).EN_OOS,
        src.xs("Maize", level="crop", drop_level=False).EN_cutoff2.rename("EN_OOS"),
    ]
).sort_index()
YIELD_SOURCES = {
    "decision_time_elastic_net": src.EN_cutoff2,
    "decision_time_persistence": src.Persistence_cutoff2,
    "decision_time_climatology": src.Climatology_cutoff2,
    "observed_2023_2025": src.yield_t_ha,
    "hindcast_elastic_net": hind,
}
area0 = {
    d: np.r_[
        base[base.district == d].set_index("crop").loc[CROPS].area0.to_numpy(),
        maize[(maize.district == d) & maize.year_end.isin(RECENT)].area_ha.mean(),
    ]
    for d in DISTRICTS
}
A0 = {d: float(area0[d].sum()) for d in DISTRICTS}
A_TOTAL = sum(A0.values())
y_main = {d: YIELD_SOURCES["decision_time_elastic_net"].loc[d].loc[CROPS_A].to_numpy() for d in DISTRICTS}
HIST_MAX = pd.concat(
    [
        panel.groupby(["district", "crop"]).area_ha.max().unstack()[CROPS],
        maize.groupby("district").area_ha.max().rename("Maize"),
    ],
    axis=1,
)[CROPS_A]
CAPS = {
    "observed_max_2015_2025": {d: np.maximum(HIST_MAX.loc[d].to_numpy(), area0[d]) for d in DISTRICTS},
    "1.5x_baseline": {d: 1.5 * area0[d] for d in DISTRICTS},
}
CAP0 = "observed_max_2015_2025"
PLANS = ("free", "least")
SCOPES = ("four_crops", "three_crops")  # three_crops = the same model without maize
fixed_coef = lambda d: np.array([NIR_A[c] for c in CROPS_A])
E_U = np.r_[np.zeros(NC), 1.0]  # picks the fallow variable


def pool(d, scope="four_crops"):
    """Land pool and baseline areas of the crops in the model (maize area set to 0 for three_crops)."""
    a = area0[d].copy()
    if scope == "three_crops":
        a[3] = 0.0
    return a, float(a.sum())


def _lp(cost, d, coef, rho, floor, cap, u_max, scope="four_crops", bcoef=None):
    # bcoef: coefficients that set the budget (rho x baseline demand); default = coef. Only the AWD
    # scenario (ext_25) passes a different value: the budget stays at the conventional demand while
    # Boro uses less water per hectare.
    w = 10 * coef
    a, land = pool(d, scope)
    wb = w if bcoef is None else 10 * bcoef
    a_ub, b_ub = [np.r_[w, 0.0]], [rho * float(a @ wb)]
    if u_max is not None:
        a_ub.append(E_U)
        b_ub.append(u_max)
    bounds = [(floor * a[0], cap[0])] + [(0, cap[k]) for k in range(1, NC)] + [(0, None)]
    if scope == "three_crops":
        bounds[3] = (0, 0)
    return linprog(
        cost,
        A_ub=np.array(a_ub),
        b_ub=b_ub,
        A_eq=np.ones((1, NC + 1)),
        b_eq=[land],
        bounds=bounds,
        method="highs",
    )


def min_fallow(d, rho, floor=0.8, coef=None, cap=CAP0, scope="four_crops", bcoef=None):
    """Smallest fallow area (ha) that lets district d meet the budget; None if no plan exists."""
    s = _lp(E_U, d, fixed_coef(d) if coef is None else coef, rho, floor, CAPS[cap][d], None, scope, bcoef)
    return float(s.x[NC]) if s.status == 0 else None


def at_tol(v):
    """Tolerance (ha) for 'at its limit' flags. The least-fallow plan is solved with a slack of about 1e-6 ha on the fallow
    (see allocate), which lets the second stage move a crop a few 1e-4 ha off its limit; a test at 1e-9 relative
    would miss it."""
    return max(1e-6 * v, 1e-2)


def allocate(d, gain, rho, plan="free", floor=0.8, coef=None, cap=CAP0, scope="four_crops", bcoef=None):
    """Maximise gain (per ha of each crop; fallow earns nothing). Returns the solver result or None."""
    coef = fixed_coef(d) if coef is None else coef
    u_max = None
    if plan == "least":
        um = min_fallow(d, rho, floor, coef, cap, scope, bcoef)
        if um is None:
            return None
        u_max = um * (1 + 1e-9) + 1e-6
    s = _lp(-np.r_[gain, 0.0], d, coef, rho, floor, CAPS[cap][d], u_max, scope, bcoef)
    return s if s.status == 0 else None


def solve_region(
    gain_of, rho, plan="free", floor=0.8, coef_of=None, cap=CAP0, scope="four_crops", bcoef_of=None
):
    xs, us, sols = {}, {}, {}
    for d in DISTRICTS:
        s = allocate(
            d,
            gain_of(d),
            rho,
            plan,
            floor,
            None if coef_of is None else coef_of(d),
            cap,
            scope,
            None if bcoef_of is None else bcoef_of(d),
        )
        assert s is not None, (d, rho, plan, floor, scope)
        xs[d], us[d], sols[d] = s.x[:NC], float(s.x[NC]), s
    return xs, us, sols


# Gross margin: revenue = price x yield; cost per hectare = cost per kg x regional mean yield, the same in
# every district. Boro prices and costs are per kg paddy; BBS yields are husked rice.
econ = pd.read_csv(I / "crop_economics_scenarios.csv").set_index(["crop", "item"]).value
MILL = float(econ[("Boro Rice", "paddy_to_husked_rice")])
to_sold_kg = np.array([1000 / MILL, 1000.0, 1000.0, 1000.0])  # kg sold per tonne of reported yield
y_mean = np.mean([y_main[d] for d in DISTRICTS], axis=0)
cost_kg = np.array([econ[(c, "cost")] for c in CROPS_A])
cost_ha = cost_kg * y_mean * to_sold_kg  # Tk per hectare
POTATO = {
    "low": float(econ[("Potato", "price_low")]),
    "mid": float(econ[("Potato", "price_mid")]),
    "high": float(econ[("Potato", "price_high")]),
}
MAIZE = {
    "low": float(econ[("Maize", "price_low")]),
    "mid": float(econ[("Maize", "price_mid")]),
    "high": float(econ[("Maize", "price_high")]),
}

BW_FARM = (float(econ[("Boro Rice", "price")]), float(econ[("Wheat", "price")]))  # harvest-time (DAM)
BW_PROC = (float(econ[("Boro Rice", "price_procurement")]), float(econ[("Wheat", "price_procurement")]))


def margin_per_ha(d, potato_price, ys=None, c_ha=None, maize_price=MAIZE["mid"], bw_price=BW_FARM):
    y = y_main[d] if ys is None else ys[d]
    price = np.array([bw_price[0], bw_price[1], potato_price, maize_price])
    return price * y * to_sold_kg - (cost_ha if c_ha is None else c_ha)


GAIN = {
    "food_energy": lambda d: 1000 * y_main[d] * KCAL_A,  # kcal per ha
    "gross_margin": lambda d: margin_per_ha(d, POTATO["mid"]),
}  # Tk per ha


def describe(
    xs, us, potato_price=POTATO["mid"], ys=None, c_ha=None, maize_price=MAIZE["mid"], scope="four_crops"
):
    """Regional outcome of a plan. Areas are % of the land pool of the model, so the five add to 100."""
    yy = y_main if ys is None else ys
    alloc, u = sum(xs.values()), sum(us.values())
    land = sum(pool(d, scope)[1] for d in DISTRICTS)
    prod = sum(xs[d] * yy[d] for d in DISTRICTS)
    return dict(
        boro_Mt=prod[0] / 1e6,
        wheat_Mt=prod[1] / 1e6,
        potato_Mt=prod[2] / 1e6,
        maize_Mt=prod[3] / 1e6,
        food_energy_trillion_kcal=sum(1000 * xs[d] @ (yy[d] * KCAL_A) for d in DISTRICTS) / 1e12,
        gross_margin_billion_Tk=sum(
            xs[d] @ margin_per_ha(d, potato_price, ys, c_ha, maize_price) for d in DISTRICTS
        )
        / 1e9,
        boro_pct_of_land=100 * alloc[0] / land,
        wheat_pct_of_land=100 * alloc[1] / land,
        potato_pct_of_land=100 * alloc[2] / land,
        maize_pct_of_land=100 * alloc[3] / land,
        fallow_pct_of_land=100 * u / land,
        fallow_thousand_ha=u / 1e3,
        land_pool_thousand_ha=land / 1e3,
    )


# ext_04: the smallest fallow share each district needs, by Boro floor and budget ratio
mf_rows = []
for scope in SCOPES:
    for cap in CAPS:
        for floor in (0.8, 0.9, 1.0):
            for rho in np.round(np.arange(0.70, 1.0001, 0.02), 2):
                tot, n_inf, land_tot = 0.0, 0, 0.0
                for d in DISTRICTS:
                    a, land = pool(d, scope)
                    land_tot += land
                    u = min_fallow(d, float(rho), floor, cap=cap, scope=scope)
                    if u is None:
                        n_inf += 1
                    else:
                        tot += u
                        # lower bound: all non-Boro land in wheat (the crop with the smallest NIR), Boro at its floor
                        w0 = float(a @ fixed_coef(d))
                        cf = max(
                            0.0,
                            land
                            - (rho * w0 - floor * a[0] * (NIR["Boro Rice"] - NIR["Wheat"])) / NIR["Wheat"],
                        )
                        assert u >= cf - 1e-6 * land, (d, rho, floor, cap, scope)
                    mf_rows.append(
                        dict(
                            scope=scope,
                            caps=cap,
                            boro_floor=floor,
                            budget_ratio=float(rho),
                            district=d,
                            feasible=u is not None,
                            min_fallow_pct=np.nan if u is None else 100 * u / land,
                        )
                    )
                mf_rows.append(
                    dict(
                        scope=scope,
                        caps=cap,
                        boro_floor=floor,
                        budget_ratio=float(rho),
                        district="REGION",
                        feasible=n_inf == 0,
                        min_fallow_pct=100 * tot / land_tot if n_inf == 0 else np.nan,
                        districts_infeasible=n_inf,
                    )
                )
min_fallow_table = pd.DataFrame(mf_rows)
min_fallow_table.to_csv(O / "ext_04_minimum_fallow.csv", index=False)
# with no limit on fallow, a plan exists exactly when floor x Boro share <= budget ratio

# ---------------------------------------------------------------------------
# 5. Sensitivity: Penman-Monteith climate deficit instead of the fixed NIR
# ---------------------------------------------------------------------------
# The budget is always rho x the district's own baseline demand computed with the same
# coefficients, so only the RELATIVE water weights of the crops matter here.
# The coefficients are the FAO-56 crop water requirement of stage 1 (daily ET0, crop coefficients,
# effective rain, Boro percolation), for all four crops.
assert (wr4.nir_fao56_mm > 0).all() and len(wr4) == 704
deficit = wr4.pivot_table(index=["district", "year_end"], columns="crop", values="nir_fao56_mm")[CROPS_A]
coef_sets = {
    "fixed_NIR": {d: fixed_coef(d) for d in DISTRICTS},
    "PM_deficit_mean_2023_2025": {d: deficit.loc[d].loc[RECENT].mean().to_numpy() for d in DISTRICTS},
    "PM_deficit_mean_2015_2025": {d: deficit.loc[d].mean().to_numpy() for d in DISTRICTS},
}
for yr in sorted(cache.year_end.unique()):
    coef_sets[f"PM_deficit_{yr}"] = {d: deficit.loc[(d, yr)].to_numpy() for d in DISTRICTS}


def boro_share(coefs, d, scope="four_crops"):
    w = pool(d, scope)[0] * coefs[d]
    return float(w[0] / w.sum())


share = pd.DataFrame({name: {d: boro_share(c, d) for d in DISTRICTS} for name, c in coef_sets.items()})
share3 = pd.Series({d: boro_share(coef_sets["fixed_NIR"], d, "three_crops") for d in DISTRICTS})
GRID = [(1.0, 0.9), (1.0, 0.8), (1.0, 0.7), (0.8, 0.8), (0.8, 0.7)]  # (Boro floor, budget ratio)
infeasible = lambda s, floor, rho: floor * s > rho + 1e-12
pm_rows = []
for name, coefs in coef_sets.items():
    row = dict(water_coefficients=name)
    for floor, rho in GRID:
        row[f"infeasible_floor{int(floor*100)}_rho{int(rho*100)}"] = int(
            infeasible(share[name], floor, rho).sum()
        )
    row["status_changes_vs_fixed_floor100"] = int(
        sum(
            (infeasible(share[name], 1.0, rho) != infeasible(share["fixed_NIR"], 1.0, rho)).sum()
            for rho in (0.9, 0.8, 0.7)
        )
    )
    row["spearman_boro_share_vs_fixed"] = float(stats.spearmanr(share[name], share["fixed_NIR"]).statistic)
    row["median_boro_water_share"] = float(share[name].median())
    xs_, us_, _ = solve_region(GAIN["food_energy"], 0.8, "free", coef_of=lambda d, c=coefs: c[d])
    r = describe(xs_, us_)
    row["rho80_floor80_food_energy_trillion_kcal"] = r["food_energy_trillion_kcal"]
    row["rho80_floor80_free_plan_fallow_pct"] = r["fallow_pct_of_land"]
    row["rho80_floor80_min_fallow_pct"] = (
        100 * sum(min_fallow(d, 0.8, 0.8, coef=coefs[d]) for d in DISTRICTS) / A_TOTAL
    )
    pm_rows.append(row)
pm = pd.DataFrame(pm_rows)
single = pm[pm.water_coefficients.str.match(r"PM_deficit_\d{4}$")]
span = dict(water_coefficients="PM_deficit_single_years_min_max")
for col in [c for c in pm.columns if c.startswith("infeasible_")]:
    span[col] = f"{single[col].min()}-{single[col].max()}"
pm_out = pd.concat([pm, pd.DataFrame([span])], ignore_index=True)
pm_out.to_csv(O / "ext_05_pm_deficit_sensitivity.csv", index=False)
# the same rule with the three-crop baseline reproduces the counts of stage 3 (5, 12, 14 districts)
assert [int(infeasible(share3, 1.0, r).sum()) for r in (0.9, 0.8, 0.7)] == [5, 12, 14]
detail = pd.DataFrame(
    dict(
        district=DISTRICTS,
        boro_share_fixed_NIR=share["fixed_NIR"].loc[DISTRICTS].to_numpy(),
        boro_share_fixed_NIR_three_crops=share3.loc[DISTRICTS].to_numpy(),
        boro_share_PM_2023_2025=share["PM_deficit_mean_2023_2025"].loc[DISTRICTS].to_numpy(),
        boro_share_PM_2015_2025=share["PM_deficit_mean_2015_2025"].loc[DISTRICTS].to_numpy(),
        pm_deficit_boro_mm=[coef_sets["PM_deficit_mean_2023_2025"][d][0] for d in DISTRICTS],
        pm_deficit_wheat_mm=[coef_sets["PM_deficit_mean_2023_2025"][d][1] for d in DISTRICTS],
        pm_deficit_potato_mm=[coef_sets["PM_deficit_mean_2023_2025"][d][2] for d in DISTRICTS],
        pm_deficit_maize_mm=[coef_sets["PM_deficit_mean_2023_2025"][d][3] for d in DISTRICTS],
    )
)
for rho in (0.9, 0.8, 0.7):
    detail[f"infeasible_fixed_rho{int(rho*100)}"] = infeasible(detail.boro_share_fixed_NIR, 1.0, rho)
    detail[f"infeasible_PM_2023_2025_rho{int(rho*100)}"] = infeasible(
        detail.boro_share_PM_2023_2025, 1.0, rho
    )
detail.to_csv(O / "ext_06_pm_deficit_district.csv", index=False)

# ---------------------------------------------------------------------------
# 6. RAWSI sensitivity: weights, normalisation, leave-one-year-out
# ---------------------------------------------------------------------------
j = panel.merge(cache, on=KEYS, validate="one_to_one")
j["wd"] = j.area_ha * j.nir_mm
j["cd"] = j.area_ha * j.climate_deficit_pm_mm
dy = j.groupby(["district", "year_end"], as_index=False).agg(
    area=("area_ha", "sum"), wd=("wd", "sum"), cd=("cd", "sum")
)
dy["demand_mm"] = dy.wd / dy.area  # area-weighted crop NIR
dy["deficit_mm"] = dy.cd / dy.area  # area-weighted climate deficit
severity_raw = ((gw.frac_type1 + gw.frac_falling_mindepth) / 2).set_axis(gw.district)

NORMALISERS = {
    "minmax": lambda v: (v - v.min()) / (v.max() - v.min()),
    "zscore": lambda v: (v - v.mean()) / v.std(ddof=0),
    "percentile_rank": lambda v: v.rank(pct=True),
    "robust_minmax_p5_p95": lambda v: (
        (v.clip(v.quantile(0.05), v.quantile(0.95)) - v.quantile(0.05))
        / (v.quantile(0.95) - v.quantile(0.05))
    ),
}


def rawsi(weights=(1 / 3, 1 / 3, 1 / 3), norm="minmax", drop_year=None):
    f = NORMALISERS[norm]
    t = dy if drop_year is None else dy[dy.year_end != drop_year]
    t = t.assign(sev=t.district.map(f(severity_raw)), dem=f(t.demand_mm), dfc=f(t.deficit_mm))
    t["score"] = weights[0] * t.sev + weights[1] * t.dem + weights[2] * t.dfc
    return t.groupby("district").score.mean().loc[DISTRICTS]


p0 = rawsi()
assert np.abs(p0.loc[primary.district].to_numpy() - primary.RAWSI.to_numpy()).max() < 1e-12  # equals stage 2

variants = [("primary", "equal weights, min-max", {})]
for label, w in [
    ("groundwater x2", (0.5, 0.25, 0.25)),
    ("demand x2", (0.25, 0.5, 0.25)),
    ("deficit x2", (0.25, 0.25, 0.5)),
]:
    variants.append(("weights", label, dict(weights=w)))
for label, w in [
    ("without groundwater", (0, 0.5, 0.5)),
    ("without demand", (0.5, 0, 0.5)),
    ("without deficit", (0.5, 0.5, 0)),
]:
    variants.append(("drop_one_component", label, dict(weights=w)))
for norm in ["zscore", "percentile_rank", "robust_minmax_p5_p95"]:
    variants.append(("normalisation", norm, dict(norm=norm)))
for yr in sorted(dy.year_end.unique()):
    variants.append(("leave_one_year_out", f"without {yr}", dict(drop_year=int(yr))))

rank0 = p0.rank(ascending=False, method="first").astype(int)
top5_0, top3_0 = set(rank0[rank0 <= 5].index), set(rank0[rank0 <= 3].index)
rank_rows, summary_rows = [], []
for family, label, kw in variants:
    s = rawsi(**kw)
    r = s.rank(ascending=False, method="first").astype(int)
    for d in DISTRICTS:
        rank_rows.append(dict(variant_family=family, variant=label, district=d, score=s[d], rank=int(r[d])))
    summary_rows.append(
        dict(
            variant_family=family,
            variant=label,
            spearman_vs_primary=float(stats.spearmanr(s, p0).statistic),
            max_abs_rank_shift=int((r - rank0).abs().max()),
            districts_shifting_3_or_more=int(((r - rank0).abs() >= 3).sum()),
            top5_overlap=len(top5_0 & set(r[r <= 5].index)),
            top3_same_set=bool(top3_0 == set(r[r <= 3].index)),
            rank1_district=r.idxmin(),
        )
    )
ranks = pd.DataFrame(rank_rows)
ranks.to_csv(O / "ext_07_rawsi_sensitivity_ranks.csv", index=False)
pd.DataFrame(summary_rows).to_csv(O / "ext_08_rawsi_sensitivity_summary.csv", index=False)
alt = ranks[ranks.variant_family != "primary"]
stability = alt.groupby("district")["rank"].agg(best_rank="min", worst_rank="max").reset_index()
stability["primary_rank"] = stability.district.map(rank0)
stability["share_of_variants_in_top5"] = stability.district.map(
    alt.assign(t=alt["rank"] <= 5).groupby("district").t.mean()
)
stability["n_variants"] = alt[["variant_family", "variant"]].drop_duplicates().shape[0]
stability = stability.sort_values("primary_rank")[
    ["primary_rank", "district", "best_rank", "worst_rank", "share_of_variants_in_top5", "n_variants"]
]
stability.to_csv(O / "ext_09_rawsi_rank_stability.csv", index=False)

# ---------------------------------------------------------------------------
# 7. Boro share of water demand (the feasibility threshold) next to the scarcity rank
# ---------------------------------------------------------------------------
thr = pd.DataFrame(
    dict(
        district=DISTRICTS,
        boro_water_share=share["fixed_NIR"].loc[DISTRICTS].to_numpy(),
        boro_water_share_three_crops=share3.loc[DISTRICTS].to_numpy(),
    )
)
thr["min_budget_ratio_full_boro"] = thr.boro_water_share  # 100% Boro floor feasible iff rho >= share
thr["min_budget_ratio_80pct_boro"] = 0.8 * thr.boro_water_share
thr["boro_share_rank"] = thr.boro_water_share.rank(ascending=False, method="first").astype(int)
thr["rawsi_rank"] = thr.district.map(rank0)
thr["rawsi"] = thr.district.map(p0)
thr["spearman_rawsi_vs_boro_share"] = float(stats.spearmanr(thr.rawsi, thr.boro_water_share).statistic)
thr = thr.sort_values("boro_share_rank")
thr.to_csv(O / "ext_10_district_boro_share.csv", index=False)
old = pd.read_csv(G / "hindcast" / "11_boro_feasibility.csv")  # stage 3 is a three-crop calculation
for rho in (0.9, 0.8, 0.7):
    stage2_set = set(old[np.isclose(old.rho, rho) & ~old.strict_Boro_100pct_feasible].district)
    assert stage2_set == set(thr[thr.boro_water_share_three_crops > rho + 1e-12].district), rho

# ---------------------------------------------------------------------------
# 8. Plans with yields from each possible source
# ---------------------------------------------------------------------------
# Main text uses the decision-time Elastic Net forecast (information up to year t-2 only).
# The other rows show whether the chosen crop mix depends on where the yields come from.
ys_rows = []
for source, ys_series in YIELD_SOURCES.items():
    ys = {d: ys_series.loc[d].loc[CROPS_A].to_numpy() for d in DISTRICTS}
    gains = {
        "food_energy": lambda d, ys=ys: 1000 * ys[d] * KCAL_A,
        "gross_margin": lambda d, ys=ys: margin_per_ha(d, POTATO["mid"], ys),
    }
    for rho in (0.8, 1.0):
        for plan in PLANS:
            for objective, g in gains.items():
                xs, us, _ = solve_region(g, rho, plan)
                ys_rows.append(
                    dict(
                        yield_source=source,
                        budget_ratio=rho,
                        plan=plan,
                        objective=objective,
                        **describe(xs, us, ys=ys),
                        _x=xs,
                    )
                )
main_x = {
    (r["budget_ratio"], r["plan"], r["objective"]): r["_x"]
    for r in ys_rows
    if r["yield_source"] == "decision_time_elastic_net"
}
for r in ys_rows:
    x = r.pop("_x")
    ref = main_x[(r["budget_ratio"], r["plan"], r["objective"])]
    r["area_reallocated_vs_main_pct"] = (
        100 * sum(np.abs(x[d] - ref[d]).sum() for d in DISTRICTS) / 2 / A_TOTAL
    )
    r["districts_with_different_plan"] = int(sum(np.abs(x[d] - ref[d]).max() > 1e-6 for d in DISTRICTS))
yield_source_table = pd.DataFrame(ys_rows)
yield_source_table.to_csv(O / "ext_11_objective_by_yield_source.csv", index=False)

# ---------------------------------------------------------------------------
# 9. Two objectives, two fallow plans, three potato prices
# ---------------------------------------------------------------------------
gm_rows = []
for rho in (0.8, 0.9, 1.0):
    for plan in PLANS:
        plans = {"food_energy": solve_region(GAIN["food_energy"], rho, plan)}
        for scen, pp in POTATO.items():
            plans[f"gross_margin_potato_{scen}"] = solve_region(
                lambda d, pp=pp: margin_per_ha(d, pp), rho, plan
            )
        for name, (xs, us, _) in plans.items():
            row = dict(budget_ratio=rho, plan=plan, objective=name)
            row.update({k: v for k, v in describe(xs, us).items() if k != "gross_margin_billion_Tk"})
            for scen, pp in POTATO.items():  # value of this plan under each potato price
                row[f"gross_margin_billion_Tk_potato_{scen}"] = describe(xs, us, pp)[
                    "gross_margin_billion_Tk"
                ]
            gm_rows.append(row)
gross_margin_table = pd.DataFrame(gm_rows)
gross_margin_table.to_csv(O / "ext_12_gross_margin_objective.csv", index=False)
per_ha = pd.DataFrame(
    dict(
        crop=CROPS_A,
        regional_mean_yield_t_ha=y_mean,
        cost_Tk_per_ha=cost_ha,
        **{
            f"margin_Tk_per_ha_potato_{s}": np.mean([margin_per_ha(d, p) for d in DISTRICTS], axis=0)
            for s, p in POTATO.items()
        },
    )
)
per_ha["water_mm"] = [NIR_A[c] for c in CROPS_A]
per_ha["margin_Tk_per_m3_potato_mid"] = per_ha.margin_Tk_per_ha_potato_mid / (10 * per_ha.water_mm)
per_ha["food_energy_kcal_per_m3"] = 1000 * y_mean * KCAL_A / (10 * per_ha.water_mm)
per_ha["food_energy_million_kcal_per_ha"] = y_mean * KCAL_A / 1e3
per_ha["hectares_per_1000_m3"] = 1000 / (10 * per_ha.water_mm)
per_ha.to_csv(O / "ext_13_margin_per_hectare.csv", index=False)
# how the gross-margin plans move with the potato price (budget 0.80)
sweep = []
for plan in PLANS:
    for pp in np.arange(6.0, 34.01, 0.5):
        xs, us, _ = solve_region(lambda d, pp=pp: margin_per_ha(d, pp), 0.8, plan)
        sweep.append(dict(plan=plan, potato_price_Tk_per_kg=pp, **describe(xs, us, pp)))
sweep = pd.DataFrame(sweep)
sweep.to_csv(O / "ext_14_potato_price_sweep.csv", index=False)

# ---------------------------------------------------------------------------
# 10. Shadow prices: what one more unit of water, land, fallow or Boro-floor relief is worth
# ---------------------------------------------------------------------------
UNITS = {"food_energy": "million kcal", "gross_margin": "thousand Tk"}
SCALE = {"food_energy": 1e-6, "gross_margin": 1e-3}
sh_rows = []
for rho in (0.8, 0.9, 1.0):
    for plan in PLANS:
        for name, g in GAIN.items():
            xs, us, sols = solve_region(g, rho, plan)
            for d in DISTRICTS:
                s, k, cap = sols[d], SCALE[name], CAPS[CAP0][d]
                ineq = -s.ineqlin.marginals  # gain per m3; gain per extra ha of fallow allowed
                w = 10 * fixed_coef(d)
                sh_rows.append(
                    dict(
                        budget_ratio=rho,
                        plan=plan,
                        objective=name,
                        unit=UNITS[name],
                        district=d,
                        water_value_per_1000_m3=1000 * ineq[0] * k,
                        land_value_per_ha=-s.eqlin.marginals[0] * k,
                        fallow_limit_value_per_ha=(ineq[1] * k) if plan == "least" else 0.0,
                        boro_floor_cost_per_ha=s.lower.marginals[0] * k,
                        wheat_cap_value_per_ha=-s.upper.marginals[1] * k,
                        potato_cap_value_per_ha=-s.upper.marginals[2] * k,
                        maize_cap_value_per_ha=-s.upper.marginals[3] * k,
                        fallow_pct=100 * us[d] / A0[d],
                        water_binding=bool(xs[d] @ w >= rho * (area0[d] @ w) * (1 - 1e-7)),
                        boro_at_floor=bool(xs[d][0] <= 0.8 * area0[d][0] + at_tol(0.8 * area0[d][0])),
                        wheat_at_cap=bool(xs[d][1] >= cap[1] - at_tol(cap[1])),
                        potato_at_cap=bool(xs[d][2] >= cap[2] - at_tol(cap[2])),
                        maize_at_cap=bool(xs[d][3] >= cap[3] - at_tol(cap[3])),
                    )
                )
shadow = pd.DataFrame(sh_rows)
shadow.to_csv(O / "ext_15_shadow_prices.csv", index=False)

# ---------------------------------------------------------------------------
# 11. Curves and descriptive statistics drawn in the figures
# ---------------------------------------------------------------------------
# (a) Response to the budget ratio, 80% Boro floor (all 16 districts feasible from 0.78 upward)
assert 0.8 * share["fixed_NIR"].max() < 0.78
resp = []
for rho in np.round(np.arange(0.78, 1.0001, 0.02), 2):
    for plan in PLANS:
        for name, g in GAIN.items():
            xs, us, _ = solve_region(g, float(rho), plan)
            resp.append(dict(budget_ratio=float(rho), plan=plan, objective=name, **describe(xs, us)))
pd.DataFrame(resp).to_csv(O / "ext_16_budget_response.csv", index=False)
# (b) Trade-off between food energy and gross margin: weighted sum of the two, each scaled by its
#     regional mean per hectare so that the weight is unit-free
S_KCAL = np.mean([GAIN["food_energy"](d).mean() for d in DISTRICTS])
S_GM = np.mean([np.abs(GAIN["gross_margin"](d)).mean() for d in DISTRICTS])
front = []
for rho in (0.8, 1.0):
    for plan in PLANS:
        for lam in np.round(np.arange(0, 1.0001, 0.05), 2):
            xs, us, _ = solve_region(
                lambda d, lam=lam: lam * GAIN["gross_margin"](d) / S_GM
                + (1 - lam) * GAIN["food_energy"](d) / S_KCAL,
                rho,
                plan,
            )
            front.append(dict(budget_ratio=rho, plan=plan, weight_on_margin=float(lam), **describe(xs, us)))
pd.DataFrame(front).to_csv(O / "ext_17_energy_margin_frontier.csv", index=False)
# (c) Regional yield and area by year, and correlation of yield with the climate predictors
desc = []
for (crop, yr), g in panel.groupby(["crop", "year_end"]):
    desc.append(
        dict(
            crop=crop,
            year_end=int(yr),
            regional_yield_t_ha=g.production_t.sum() / g.area_ha.sum(),
            district_yield_min=g.yield_t_ha.min(),
            district_yield_max=g.yield_t_ha.max(),
            regional_area_thousand_ha=g.area_ha.sum() / 1e3,
        )
    )
pd.DataFrame(desc).to_csv(O / "ext_18_regional_yield_area_by_year.csv", index=False)
corr = []
for crop in CROPS:
    g = panel[panel.crop == crop]
    for v in CL + ["year_trend"]:
        z = g[v] - g.groupby("district")[v].transform("mean")
        zy = g.yield_t_ha - g.groupby("district").yield_t_ha.transform("mean")
        corr.append(
            dict(
                crop=crop,
                variable=v,
                pearson_r=float(np.corrcoef(g[v], g.yield_t_ha)[0, 1]),
                pearson_r_within_district=float(np.corrcoef(z, zy)[0, 1]),
                n=len(g),
            )
        )
pd.DataFrame(corr).to_csv(O / "ext_19_yield_climate_correlation.csv", index=False)

# ---------------------------------------------------------------------------
# 12. Sensitivity of the gross-margin plans to the wheat cost (news-reported figure, +/-20%)
# ---------------------------------------------------------------------------
WHEAT_COST = {"minus_20pct": 0.8 * cost_kg[1], "reported": cost_kg[1], "plus_20pct": 1.2 * cost_kg[1]}
wc_rows = []
for rho in (0.8, 1.0):
    for plan in PLANS:
        for scen, wc in WHEAT_COST.items():
            ck = cost_kg.copy()
            ck[1] = wc
            c_ha = ck * y_mean * to_sold_kg
            gm = lambda d, c_ha=c_ha: margin_per_ha(d, POTATO["mid"], c_ha=c_ha)
            xs, us, _ = solve_region(gm, rho, plan)
            row = dict(
                budget_ratio=rho,
                plan=plan,
                wheat_cost_scenario=scen,
                wheat_cost_Tk_per_kg=float(wc),
                wheat_margin_Tk_per_ha=float(np.mean([gm(d)[1] for d in DISTRICTS])),
                **describe(xs, us, c_ha=c_ha),
            )
            row["wheat_area_thousand_ha"] = sum(xs[d][1] for d in DISTRICTS) / 1e3
            row["maize_area_thousand_ha"] = sum(xs[d][3] for d in DISTRICTS) / 1e3
            wc_rows.append(row)
pd.DataFrame(wc_rows).to_csv(O / "ext_20_wheat_cost_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 13. Sensitivity of the plans to the crop-area limits (observed maximum vs 1.5 x baseline)
# ---------------------------------------------------------------------------
cap_rows = []
for cap in CAPS:
    for rho in (0.8, 1.0):
        for plan in PLANS:
            for name, g in GAIN.items():
                xs, us, _ = solve_region(g, rho, plan, cap=cap)
                cap_rows.append(
                    dict(caps=cap, budget_ratio=rho, plan=plan, objective=name, **describe(xs, us))
                )
cap_table = pd.DataFrame(cap_rows)
cap_table.to_csv(O / "ext_21_crop_limit_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 14. Maize price and cost (news-reported figures): low / mid / high price, cost -20% / reported / +20%
# ---------------------------------------------------------------------------
mz_rows = []
for rho in (0.8, 1.0):
    for plan in PLANS:
        for pscen, mp in MAIZE.items():
            for cscen, cf in (("minus_20pct", 0.8), ("reported", 1.0), ("plus_20pct", 1.2)):
                ck = cost_kg.copy()
                ck[3] = cf * cost_kg[3]
                c_ha = ck * y_mean * to_sold_kg
                gm = lambda d, c_ha=c_ha, mp=mp: margin_per_ha(d, POTATO["mid"], c_ha=c_ha, maize_price=mp)
                xs, us, _ = solve_region(gm, rho, plan)
                row = dict(
                    budget_ratio=rho,
                    plan=plan,
                    maize_price_scenario=pscen,
                    maize_price_Tk_per_kg=mp,
                    maize_cost_scenario=cscen,
                    maize_cost_Tk_per_kg=float(ck[3]),
                    maize_margin_Tk_per_ha=float(np.mean([gm(d)[3] for d in DISTRICTS])),
                    **describe(xs, us, c_ha=c_ha, maize_price=mp),
                )
                row["maize_area_thousand_ha"] = sum(xs[d][3] for d in DISTRICTS) / 1e3
                mz_rows.append(row)
pd.DataFrame(mz_rows).to_csv(O / "ext_22_maize_price_cost_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 15. Boro and wheat at the government procurement price instead of the harvest-time market price
# ---------------------------------------------------------------------------
pr_rows = []
for rho in (0.8, 1.0):
    for plan in PLANS:
        for scen, bw in (("harvest_time_market", BW_FARM), ("procurement", BW_PROC)):
            gm = lambda d, bw=bw: margin_per_ha(d, POTATO["mid"], bw_price=bw)
            xs, us, _ = solve_region(gm, rho, plan)
            alloc = sum(xs.values())
            row = dict(
                budget_ratio=rho,
                plan=plan,
                boro_wheat_price=scen,
                boro_price_Tk_per_kg_paddy=bw[0],
                wheat_price_Tk_per_kg=bw[1],
                boro_margin_Tk_per_ha=float(np.mean([gm(d)[0] for d in DISTRICTS])),
                wheat_margin_Tk_per_ha=float(np.mean([gm(d)[1] for d in DISTRICTS])),
            )
            r = describe(xs, us)
            r["gross_margin_billion_Tk"] = sum(xs[d] @ gm(d) for d in DISTRICTS) / 1e9
            pr_rows.append({**row, **r})
pd.DataFrame(pr_rows).to_csv(O / "ext_24_procurement_price_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 16. What maize adds: the same plans with and without maize in the model
# ---------------------------------------------------------------------------
# three_crops: maize area fixed at 0 and its land and water left out of the pool and the budget
# four_crops: the main model.
sc_rows = []
for scope in SCOPES:
    for rho in (0.8, 0.9, 1.0):
        for plan in PLANS:
            for name, g in GAIN.items():
                xs, us, _ = solve_region(g, rho, plan, scope=scope)
                r = describe(xs, us, scope=scope)
                base_e = sum(1000 * pool(d, scope)[0] @ (y_main[d] * KCAL_A) for d in DISTRICTS) / 1e12
                base_m = sum(pool(d, scope)[0] @ margin_per_ha(d, POTATO["mid"]) for d in DISTRICTS) / 1e9
                sc_rows.append(
                    dict(
                        scope=scope,
                        budget_ratio=rho,
                        plan=plan,
                        objective=name,
                        **r,
                        baseline_food_energy_trillion_kcal=base_e,
                        baseline_gross_margin_billion_Tk=base_m,
                        food_energy_pct_of_baseline=100 * r["food_energy_trillion_kcal"] / base_e,
                        gross_margin_pct_of_baseline=100 * r["gross_margin_billion_Tk"] / base_m,
                    )
                )
scope_table = pd.DataFrame(sc_rows)
scope_table.to_csv(O / "ext_23_three_vs_four_crops.csv", index=False)


# ---------------------------------------------------------------------------
# 17. Alternate wetting and drying (AWD) on part of the Boro area
# ---------------------------------------------------------------------------
# A share phi of the Boro area adopts AWD, which cuts its irrigation water by s. The water budget stays
# at rho x the CONVENTIONAL baseline demand, so the saving frees water for other crops or less fallow.
#   Boro coefficient   n_B (1 - phi s)          (budget still uses n_B)
#   Boro yield         y_B (1 + phi dy)         dy = 0 (main) or -5.4% (Carrijo et al. 2017, all AWD)
# Savings s: 15-30% in Bangladesh field trials (Paul et al. 2013, Bangladesh Rice J. 17:33-37);
# 20.2% (Hossain et al. 2016, Bangladesh Rice J. 20:17-22). Global meta-analysis: 33.9% (Gao et al. 2024).
# The irrigation cost per hectare is left unchanged (no pumping saving is credited), so the gross-margin
# results are conservative. The saving is applied to the net irrigation coefficient of the model; this
# assumes the field-level saving carries over to the district water budget (no return-flow accounting).
AWD_S = (0.15, 0.20, 0.25, 0.30)
AWD_PHI = (0.5, 1.0)
AWD_DY = {"none": 0.0, "carrijo_2017_all_awd": -0.054}


def awd_coef(s, phi):
    k = np.r_[1.0 - phi * s, np.ones(NC - 1)]
    return lambda d: fixed_coef(d) * k


def awd_yields(phi, dy):
    k = np.r_[1.0 + phi * dy, np.ones(NC - 1)]
    return {d: y_main[d] * k for d in DISTRICTS}


awd_rows = []
for s_ in (0.0,) + AWD_S:
    for phi in (1.0,) if s_ == 0 else AWD_PHI:
        for dyn, dy in AWD_DY.items():
            if s_ == 0 and dy != 0:
                continue
            ys = awd_yields(phi, dy)
            gains = {
                "food_energy": lambda d, ys=ys: 1000 * ys[d] * KCAL_A,
                "gross_margin": lambda d, ys=ys: margin_per_ha(d, POTATO["mid"], ys=ys),
            }
            cf = awd_coef(s_, phi)
            for rho in (0.8, 0.9, 1.0):
                mf = [min_fallow(d, rho, coef=cf(d), bcoef=fixed_coef(d)) for d in DISTRICTS]
                for plan in PLANS:
                    for name, g in gains.items():
                        xs, us, _ = solve_region(g, rho, plan, coef_of=cf, bcoef_of=fixed_coef)
                        r = describe(xs, us, ys=ys)
                        # Boro yield change also moves the per-ha margin; describe() gets ys for that
                        awd_rows.append(
                            dict(
                                water_saving=s_,
                                adoption_share=phi,
                                yield_effect=dyn,
                                boro_yield_change_pct=100 * phi * dy,
                                budget_ratio=rho,
                                plan=plan,
                                objective=name,
                                min_fallow_pct=100 * sum(mf) / A_TOTAL,
                                **r,
                            )
                        )
awd_table = pd.DataFrame(awd_rows)
conv = awd_table[awd_table.water_saving == 0].set_index(["budget_ratio", "plan", "objective"])
key = ["budget_ratio", "plan", "objective"]
for col in ("food_energy_trillion_kcal", "gross_margin_billion_Tk", "fallow_pct_of_land"):
    awd_table[f"{col}_change_vs_conventional"] = (
        awd_table[col].to_numpy()
        - conv.loc[list(awd_table[key].itertuples(index=False, name=None)), col].to_numpy()
    )
# with s = 0 the AWD model is the main model: same min fallow as ext_04 and same plans as ext_16
_chk = min_fallow_table[
    (min_fallow_table.scope == "four_crops")
    & (min_fallow_table.caps == CAP0)
    & (min_fallow_table.boro_floor == 0.8)
    & (min_fallow_table.district == "REGION")
].set_index("budget_ratio")
for rho in (0.8, 0.9, 1.0):
    a_ = conv.loc[(rho, "free", "food_energy"), "min_fallow_pct"]
    assert abs(a_ - _chk.loc[rho, "min_fallow_pct"]) < 1e-6, (rho, a_)
assert (awd_table.min_fallow_pct <= conv.min_fallow_pct.max() + 1e-9).all()
awd_table.to_csv(O / "ext_25_awd_scenarios.csv", index=False)

# smallest fallow (Boro floor 0.8) and districts where full Boro (floor 1.0) has no plan, as adoption rises
awd_curve = []
for s_ in AWD_S:
    cf_full = None
    for phi in np.round(np.arange(0, 1.0001, 0.05), 2):
        cf = awd_coef(s_, float(phi))
        for rho in (0.8, 0.9, 1.0):
            mf = [min_fallow(d, rho, coef=cf(d), bcoef=fixed_coef(d)) for d in DISTRICTS]
            full = [min_fallow(d, rho, floor=1.0, coef=cf(d), bcoef=fixed_coef(d)) for d in DISTRICTS]
            awd_curve.append(
                dict(
                    water_saving=s_,
                    adoption_share=float(phi),
                    budget_ratio=rho,
                    min_fallow_pct=100 * sum(mf) / A_TOTAL,
                    districts_with_fallow=int(sum(u > 1e-6 * A0[d] for u, d in zip(mf, DISTRICTS))),
                    districts_full_boro_infeasible=int(sum(u is None for u in full)),
                )
            )
awd_curve = pd.DataFrame(awd_curve)
assert (
    awd_curve.groupby(["water_saving", "budget_ratio"]).min_fallow_pct.apply(
        lambda v: (np.diff(v.to_numpy()) <= 1e-9).all()
    )
).all()  # more adoption never needs more fallow
awd_curve.to_csv(O / "ext_26_awd_fallow_curve.csv", index=False)


# ---------------------------------------------------------------------------
# 18. Model value of water next to the irrigation charges farmers pay
# ---------------------------------------------------------------------------
# ext_15 gives the gross-margin value of one more m3 of water in the district budget (thousand Tk per
# 1000 m3 = Tk per m3). Charges come from inputs/irrigation_charge_references.csv (sources there).
# Hourly charges are turned into Tk per m3 with the discharge the same source gives. Charges per hectare
# are divided by the model's Boro coefficient (676 mm = 6,760 m3/ha, a net requirement): farmers apply
# more water than that, so these per-m3 figures are upper bounds.
chg = pd.read_csv(I / "irrigation_charge_references.csv").set_index("item")
m3h = lambda lpm: lpm * 60 / 1000
per_m3 = {
    "bmda_dtw_hourly_charge": (
        chg.loc["bmda_dtw_hourly_charge", "low"]
        / m3h(chg.loc["bmda_dtw_hourly_charge", "discharge_low_L_per_min"]),
        chg.loc["bmda_dtw_hourly_charge", "high"]
        / m3h(chg.loc["bmda_dtw_hourly_charge", "discharge_high_L_per_min"]),
    ),
    "boro_irrigation_cost_per_ha_2015_16": tuple(
        chg.loc["boro_irrigation_cost_per_ha_2015_16", ["low", "high"]] / (10 * NIR_A["Boro Rice"])
    ),
    "boro_irrigation_cost_per_ha_2016_17": tuple(
        chg.loc["boro_irrigation_cost_per_ha_2016_17", ["low", "high"]] / (10 * NIR_A["Boro Rice"])
    ),
}
per_m3 = {k: (float(min(v)), float(max(v))) for k, v in per_m3.items()}
CHARGE_HI = max(v[1] for v in per_m3.values())
wv_rows = []
for (rho, plan), g in shadow[shadow.objective == "gross_margin"].groupby(["budget_ratio", "plan"]):
    v = g.water_value_per_1000_m3  # = Tk per m3
    row = dict(
        budget_ratio=rho,
        plan=plan,
        districts_water_binding=int(g.water_binding.sum()),
        water_value_Tk_per_m3_min=v.min(),
        water_value_Tk_per_m3_median=v.median(),
        water_value_Tk_per_m3_max=v.max(),
        districts_above_highest_charge=int((v > CHARGE_HI).sum()),
    )
    for k, (lo, hi) in per_m3.items():
        row[f"{k}_Tk_per_m3_low"], row[f"{k}_Tk_per_m3_high"] = lo, hi
        row[f"median_value_over_{k}_high"] = v.median() / hi
    wv_rows.append(row)
water_value = pd.DataFrame(wv_rows)
assert abs(per_m3["bmda_dtw_hourly_charge"][0] - 115 / 201.6) < 1e-9
water_value.to_csv(O / "ext_27_water_value_vs_charges.csv", index=False)

# ---------------------------------------------------------------------------
# 19. One-factor-at-a-time sensitivities at budget 0.80 (tornado figure)
# ---------------------------------------------------------------------------
# Three outcomes: smallest fallow share (Boro floor as set), food energy of the food-energy plan and gross
# margin of the gross-margin plan (both with free fallow). Each factor is moved alone; when a factor has
# several settings the bar spans the lowest and highest result. AWD is a management option, not an
# uncertain input, and is kept as its own group.
TRHO = 0.8


def outcomes(
    coef_of=None,
    bcoef_of=None,
    cap=CAP0,
    floor=0.8,
    ys=None,
    pp=POTATO["mid"],
    mp=MAIZE["mid"],
    c_ha=None,
    bw=BW_FARM,
):
    yy = y_main if ys is None else ys
    mf = [
        min_fallow(
            d,
            TRHO,
            floor,
            None if coef_of is None else coef_of(d),
            cap,
            bcoef=None if bcoef_of is None else bcoef_of(d),
        )
        for d in DISTRICTS
    ]
    assert all(u is not None for u in mf)
    xs, us, _ = solve_region(
        lambda d: 1000 * yy[d] * KCAL_A, TRHO, "free", floor, coef_of, cap, bcoef_of=bcoef_of
    )
    fe = sum(1000 * xs[d] @ (yy[d] * KCAL_A) for d in DISTRICTS) / 1e12
    gm_of = lambda d: margin_per_ha(d, pp, ys=ys, c_ha=c_ha, maize_price=mp, bw_price=bw)
    xs, us, _ = solve_region(gm_of, TRHO, "free", floor, coef_of, cap, bcoef_of=bcoef_of)
    gm = sum(xs[d] @ gm_of(d) for d in DISTRICTS) / 1e9
    return dict(
        min_fallow_pct=100 * sum(mf) / A_TOTAL, food_energy_trillion_kcal=fe, gross_margin_billion_Tk=gm
    )


def c_ha_with(k, f):
    ck = cost_kg.copy()
    ck[k] = f * cost_kg[k]
    return ck * y_mean * to_sold_kg


ys_of = lambda src: {d: YIELD_SOURCES[src].loc[d].loc[CROPS_A].to_numpy() for d in DISTRICTS}
FACTORS = [
    (
        "input",
        "Water coefficients: FAO-56, single years 2015-2025",
        [
            dict(coef_of=lambda d, c=coef_sets[f"PM_deficit_{yr}"]: c[d])
            for yr in sorted(cache.year_end.unique())
        ],
    ),
    (
        "input",
        "Yields: persistence, climatology, observed, hindcast",
        [dict(ys=ys_of(k)) for k in YIELD_SOURCES if k != "decision_time_elastic_net"],
    ),
    ("input", "Crop limits: 1.5 x baseline area", [dict(cap="1.5x_baseline")]),
    ("input", "Boro floor: 0.75 / 0.85 of baseline", [dict(floor=0.75), dict(floor=0.85)]),
    ("input", "Potato price: low / high", [dict(pp=POTATO["low"]), dict(pp=POTATO["high"])]),
    ("input", "Maize price: low / high", [dict(mp=MAIZE["low"]), dict(mp=MAIZE["high"])]),
    ("input", "Maize cost: -20% / +20%", [dict(c_ha=c_ha_with(3, 0.8)), dict(c_ha=c_ha_with(3, 1.2))]),
    ("input", "Wheat cost: -20% / +20%", [dict(c_ha=c_ha_with(1, 0.8)), dict(c_ha=c_ha_with(1, 1.2))]),
    ("input", "Boro and wheat at procurement price", [dict(bw=BW_PROC)]),
    (
        "lever",
        "AWD on half the Boro area, 15-30% saving",
        [dict(coef_of=awd_coef(s_, 0.5), bcoef_of=fixed_coef) for s_ in AWD_S],
    ),
    (
        "lever",
        "AWD on all Boro area, 15-30% saving",
        [dict(coef_of=awd_coef(s_, 1.0), bcoef_of=fixed_coef) for s_ in AWD_S],
    ),
]
main_out = outcomes()
_r = conv.loc[(0.8, "free", "food_energy")]
assert (
    abs(main_out["min_fallow_pct"] - _r.min_fallow_pct) < 1e-9
    and abs(main_out["food_energy_trillion_kcal"] - _r.food_energy_trillion_kcal) < 1e-9
)
torn = []
for group, label, settings in FACTORS:
    res = [outcomes(**st) for st in settings]
    for k, base_v in main_out.items():
        vals = [r[k] for r in res]
        lo, hi = min(vals), max(vals)
        rel = (lambda v: v - base_v) if k == "min_fallow_pct" else (lambda v: 100 * (v - base_v) / base_v)
        torn.append(
            dict(
                group=group,
                factor=label,
                settings=len(settings),
                outcome=k,
                change_unit="percentage points" if k == "min_fallow_pct" else "% of main",
                main_value=base_v,
                low_value=lo,
                high_value=hi,
                low_change=rel(lo),
                high_change=rel(hi),
                swing=abs(rel(hi) - rel(lo)) if len(vals) > 1 else abs(rel(lo)),
            )
        )
tornado = pd.DataFrame(torn)
tornado.to_csv(O / "ext_28_sensitivity_tornado.csv", index=False)


# ---------------------------------------------------------------------------
# 20. Low-water crops: can lentil and mustard take the land that Boro releases?
# ---------------------------------------------------------------------------
# A six-crop VARIANT of the district model; the main model stays on four crops. Lentil and mustard join the
# land pool and the baseline water demand. Their area may grow to the larger of (a) the largest area observed
# in the district in 2015-2025 and (b) k x the baseline area, k = 1 (observed maximum only) to 6.
#   area, yield   BBS yearbooks 2016-2025 (inputs/lentil_mustard_16districts_2015_2025.csv)
#   NIR           pulses 164-231 mm, oilseeds 81-105 mm (Mainuddin et al. 2014, Table 4.2): midpoints 197.5, 93
#   yield         decision-time persistence (yield of year t-2, mean over targets 2023-2025); no Elastic Net
#   food energy   lentil as dal (FCT 02_0008, 317 kcal/100 g x 0.725 kg dal per kg grain);
#                 mustard as oil only (FCT 13_0008, 900 kcal/100 g x 0.35 kg fat per kg seed)
#   margin        DAM harvest prices (Yearbook 2025, Table 10.5.1) and survey costs (crop_economics_scenarios.csv)
# Land is additive as for potato: short-duration mustard can precede Boro on the same field, so the model
# is conservative about how much land mustard needs.
lm = pd.read_csv(I / "lentil_mustard_16districts_2015_2025.csv")
LW = ["Lentil", "Mustard"]
assert (
    len(lm) == 352
    and lm.groupby(["district", "crop"]).size().eq(11).all()
    and set(lm.district) == set(DISTRICTS)
)
assert (lm.area_ha > 0).all() and lm.yield_t_ha.between(0.5, 2.2).all()
NIR6 = np.r_[[NIR_A[c] for c in CROPS_A], 197.5, 93.0]
KCAL6 = np.r_[
    KCAL_A,
    3170.0 * float(econ[("Lentil", "dal_per_grain")]),
    9000.0 * float(econ[("Mustard", "oil_per_seed")]),
]
lw_a = lm[lm.year_end.isin(RECENT)].groupby(["district", "crop"]).area_ha.mean().unstack()[LW]
lw_y = (
    lm[lm.year_end.isin([t - 2 for t in RECENT])]
    .groupby(["district", "crop"])
    .yield_t_ha.mean()
    .unstack()[LW]
)
lw_max = lm.groupby(["district", "crop"]).area_ha.max().unstack()[LW]
area6 = {d: np.r_[area0[d], lw_a.loc[d].to_numpy()] for d in DISTRICTS}
y6 = {d: np.r_[y_main[d], lw_y.loc[d].to_numpy()] for d in DISTRICTS}
LAND6 = sum(a.sum() for a in area6.values())
lw_price = np.array([float(econ[(c, "price")]) for c in LW])
lw_cost_ha = np.array([float(econ[(c, "cost")]) for c in LW]) * lw_y.mean().to_numpy() * 1000
GAIN6 = {
    "food_energy": lambda d: 1000 * y6[d] * KCAL6,
    "gross_margin": lambda d: np.r_[
        margin_per_ha(d, POTATO["mid"]), lw_price * y6[d][4:] * 1000 - lw_cost_ha
    ],
}
E_U6 = np.r_[np.zeros(6), 1.0]


def lp6(d, cost, rho, k, boro_water=1.0, u_max=None):
    a = area6[d]
    cap = np.r_[CAPS[CAP0][d], np.maximum(np.maximum(lw_max.loc[d].to_numpy(), a[4:]), k * a[4:])]
    a_ub, b_ub = [np.r_[10 * NIR6 * np.r_[boro_water, np.ones(5)], 0.0]], [rho * 10 * float(a @ NIR6)]
    if u_max is not None:
        a_ub.append(E_U6)
        b_ub.append(u_max)
    s = linprog(
        cost,
        A_ub=np.array(a_ub),
        b_ub=b_ub,
        A_eq=np.ones((1, 7)),
        b_eq=[a.sum()],
        bounds=[(0.8 * a[0], cap[0])] + [(0, cap[j]) for j in range(1, 6)] + [(0, None)],
        method="highs",
    )
    assert s.status == 0, (d, rho, k)
    return s


LW_K = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LW_AWD = {"none": 1.0, "awd_half_boro_20pct_saving": 1.0 - 0.5 * 0.20}
base6_e = sum(1000 * area6[d] @ (y6[d] * KCAL6) for d in DISTRICTS) / 1e12
base6_m = sum(area6[d] @ GAIN6["gross_margin"](d) for d in DISTRICTS) / 1e9
lw_rows = []
for rho in (0.8, 0.9, 1.0):
    four = (
        float(
            min_fallow_table[
                (min_fallow_table.scope == "four_crops")
                & (min_fallow_table.caps == CAP0)
                & (min_fallow_table.boro_floor == 0.8)
                & (min_fallow_table.district == "REGION")
                & np.isclose(min_fallow_table.budget_ratio, rho)
            ].min_fallow_pct.iloc[0]
        )
        * A_TOTAL
        / 100
        / 1e3
    )
    for awd_name, bw in LW_AWD.items():
        for k in LW_K:
            um = {d: float(lp6(d, E_U6, rho, k, bw).x[6]) for d in DISTRICTS}
            for plan in PLANS:
                for name, g in GAIN6.items():
                    xs = {
                        d: lp6(
                            d,
                            -np.r_[g(d), 0.0],
                            rho,
                            k,
                            bw,
                            None if plan == "free" else um[d] * (1 + 1e-9) + 1e-6,
                        ).x
                        for d in DISTRICTS
                    }
                    tot = sum(xs.values())
                    row = dict(
                        budget_ratio=rho,
                        awd=awd_name,
                        low_water_crop_limit_x_baseline=k,
                        plan=plan,
                        objective=name,
                        min_fallow_thousand_ha=sum(um.values()) / 1e3,
                        min_fallow_pct_of_pool=100 * sum(um.values()) / LAND6,
                        four_crop_model_min_fallow_thousand_ha=four,
                        land_pool_thousand_ha=LAND6 / 1e3,
                    )
                    for j, c in enumerate(
                        ["boro", "wheat", "potato", "maize", "lentil", "mustard", "fallow"]
                    ):
                        row[f"{c}_thousand_ha"] = tot[j] / 1e3
                    row["lentil_x_baseline"] = tot[4] / lw_a.Lentil.sum()
                    row["mustard_x_baseline"] = tot[5] / lw_a.Mustard.sum()
                    row["food_energy_trillion_kcal"] = (
                        sum(1000 * xs[d][:6] @ (y6[d] * KCAL6) for d in DISTRICTS) / 1e12
                    )
                    row["gross_margin_billion_Tk"] = (
                        sum(xs[d][:6] @ GAIN6["gross_margin"](d) for d in DISTRICTS) / 1e9
                    )
                    row["food_energy_pct_of_baseline"] = 100 * row["food_energy_trillion_kcal"] / base6_e
                    row["gross_margin_pct_of_baseline"] = 100 * row["gross_margin_billion_Tk"] / base6_m
                    lw_rows.append(row)
low_water = pd.DataFrame(lw_rows)
_m = low_water.drop_duplicates(["budget_ratio", "awd", "low_water_crop_limit_x_baseline"])
assert (
    _m.groupby(["budget_ratio", "awd"])
    .min_fallow_thousand_ha.apply(lambda v: (np.diff(v.to_numpy()) <= 1e-6).all())
    .all()
)
low_water.to_csv(O / "ext_29_low_water_crops.csv", index=False)

# ---------------------------------------------------------------------------
# 21. How far the results are set by the Boro floor and the crop-area limits.
# The two are crossed rather than varied one at a time. Floors from none (0) to all of the Boro area (1.0);
# limits from the main case (largest area of 2015-2025) through the larger of that and 1.5 or 2 times the
# baseline area, to no limit at all (every crop may take the whole land pool). With no limit the plans are
# not realistic; they bound what the limits do. Nothing else changes (published NIR, decision-time Elastic
# Net yields, central prices). A run in which a district has no plan is reported as infeasible and its
# regional plan is left empty.
# ---------------------------------------------------------------------------
GRID_FLOORS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_LIMITS = {
    "observed_max_2015_2025": lambda d: CAPS[CAP0][d],
    "observed_max_or_1.5x_baseline": lambda d: np.maximum(CAPS[CAP0][d], 1.5 * area0[d]),
    "observed_max_or_2x_baseline": lambda d: np.maximum(CAPS[CAP0][d], 2.0 * area0[d]),
    "no_limit": lambda d: np.full(NC, A0[d]),
}
BASE_DESC = describe({d: area0[d] for d in DISTRICTS}, {d: 0.0 for d in DISTRICTS})


def grid_plan(d, gain, rho, floor, cap, least, coef=None, bcoef=None):
    coef = fixed_coef(d) if coef is None else coef
    um = None
    if least:
        s0 = _lp(E_U, d, coef, rho, floor, cap, None, bcoef=bcoef)
        if s0.status != 0:
            return None
        um = float(s0.x[NC]) * (1 + 1e-9) + 1e-6
    s = _lp(-np.r_[gain, 0.0], d, coef, rho, floor, cap, um, bcoef=bcoef)
    return s.x if s.status == 0 else None


grid_rows = []
for lim_name, lim in GRID_LIMITS.items():
    for floor in GRID_FLOORS:
        for rho in (0.8, 0.9):
            mins = {d: _lp(E_U, d, fixed_coef(d), rho, floor, lim(d), None) for d in DISTRICTS}
            infeasible = [d for d in DISTRICTS if mins[d].status != 0]
            row = dict(
                boro_floor=floor, crop_limits=lim_name, budget_ratio=rho, districts_infeasible=len(infeasible)
            )
            if infeasible:
                grid_rows.append(row)
                continue
            row["min_fallow_pct"] = 100 * sum(float(mins[d].x[NC]) for d in DISTRICTS) / A_TOTAL
            for tag, gname, least in (
                ("least_fallow_food_energy", "food_energy", True),
                ("free_food_energy", "food_energy", False),
                ("free_gross_margin", "gross_margin", False),
            ):
                xs = {d: grid_plan(d, GAIN[gname](d), rho, floor, lim(d), least) for d in DISTRICTS}
                r = describe({d: x[:NC] for d, x in xs.items()}, {d: float(x[NC]) for d, x in xs.items()})
                row[f"{tag}_food_energy_pct_of_baseline"] = (
                    100 * r["food_energy_trillion_kcal"] / BASE_DESC["food_energy_trillion_kcal"]
                )
                row[f"{tag}_gross_margin_pct_of_baseline"] = (
                    100 * r["gross_margin_billion_Tk"] / BASE_DESC["gross_margin_billion_Tk"]
                )
                for c in ("boro", "wheat", "potato", "maize", "fallow"):
                    row[f"{tag}_{c}_pct_of_land"] = r[f"{c}_pct_of_land"]
                if tag == "least_fallow_food_energy":
                    row["least_fallow_districts_boro_at_floor"] = int(
                        sum(xs[d][0] <= floor * area0[d][0] + at_tol(area0[d][0]) for d in DISTRICTS)
                    )
                    for j, c in ((1, "wheat"), (2, "potato"), (3, "maize")):
                        row[f"least_fallow_districts_{c}_at_limit"] = int(
                            sum(xs[d][j] >= lim(d)[j] - at_tol(lim(d)[j]) for d in DISTRICTS)
                        )
            grid_rows.append(row)
floor_limit_grid = pd.DataFrame(grid_rows)
# the main case of the grid is the main model
_g = floor_limit_grid[
    (floor_limit_grid.crop_limits == CAP0) & (floor_limit_grid.boro_floor == 0.8)
].set_index("budget_ratio")
for rho in (0.8, 0.9):
    assert abs(_g.loc[rho, "min_fallow_pct"] - _chk.loc[rho, "min_fallow_pct"]) < 1e-6, rho
# a looser limit or a lower floor never needs more fallow
_f = floor_limit_grid.dropna(subset=["min_fallow_pct"])
for rho in (0.8, 0.9):
    for lim_name in GRID_LIMITS:
        v = (
            _f[(_f.budget_ratio == rho) & (_f.crop_limits == lim_name)]
            .sort_values("boro_floor")
            .min_fallow_pct.to_numpy()
        )
        assert (np.diff(v) >= -1e-9).all(), (rho, lim_name)
floor_limit_grid.to_csv(O / "ext_30_floor_limit_grid.csv", index=False)

# ---------------------------------------------------------------------------
# 22. AWD water saving and Boro yield effect crossed, budget 0.80, Boro floor 0.8.
# The yield effect applies to the Boro area under AWD, as in ext_25: Boro yield x (1 + adoption x effect).
# Effects 0, -2.5, -5.4 (Carrijo et al. 2017, all AWD) and -10%, the last a pessimistic case beyond the
# meta-analysis mean. The smallest fallow does not depend on yields; food energy and gross margin do.
# ---------------------------------------------------------------------------
AWD_DY_GRID = (0.0, -0.025, -0.054, -0.10)
ay_rows = []
for s_ in AWD_S:
    for phi in AWD_PHI:
        cf = awd_coef(s_, phi)
        mf = sum(min_fallow(d, 0.8, coef=cf(d), bcoef=fixed_coef(d)) for d in DISTRICTS)
        for dy in AWD_DY_GRID:
            ys = awd_yields(phi, dy)
            row = dict(
                water_saving=s_,
                adoption_share=phi,
                boro_yield_effect_on_awd_area_pct=100 * dy,
                boro_yield_change_pct=100 * phi * dy,
                budget_ratio=0.8,
                min_fallow_pct=100 * mf / A_TOTAL,
            )
            for tag, gname, plan in (
                ("least_fallow_food_energy", "food_energy", "least"),
                ("free_gross_margin", "gross_margin", "free"),
            ):
                g = (
                    (lambda d, ys=ys: 1000 * ys[d] * KCAL_A)
                    if gname == "food_energy"
                    else (lambda d, ys=ys: margin_per_ha(d, POTATO["mid"], ys=ys))
                )
                xs, us, _ = solve_region(g, 0.8, plan, coef_of=cf, bcoef_of=fixed_coef)
                r = describe(xs, us, ys=ys)
                row[f"{tag}_food_energy_pct_of_baseline"] = (
                    100 * r["food_energy_trillion_kcal"] / BASE_DESC["food_energy_trillion_kcal"]
                )
                row[f"{tag}_gross_margin_pct_of_baseline"] = (
                    100 * r["gross_margin_billion_Tk"] / BASE_DESC["gross_margin_billion_Tk"]
                )
            ay_rows.append(row)
awd_yield_grid = pd.DataFrame(ay_rows)
# the two yield effects of ext_25 are reproduced
_a25 = awd_table[
    (awd_table.budget_ratio == 0.8)
    & (awd_table.plan == "least")
    & (awd_table.objective == "food_energy")
    & (awd_table.water_saving > 0)
]
for _, r in _a25.iterrows():
    dy = AWD_DY[r.yield_effect]
    m = awd_yield_grid[
        np.isclose(awd_yield_grid.water_saving, r.water_saving)
        & np.isclose(awd_yield_grid.adoption_share, r.adoption_share)
        & np.isclose(awd_yield_grid.boro_yield_effect_on_awd_area_pct, 100 * dy)
    ].iloc[0]
    assert (
        abs(
            m.least_fallow_food_energy_food_energy_pct_of_baseline
            - 100 * r.food_energy_trillion_kcal / BASE_DESC["food_energy_trillion_kcal"]
        )
        < 1e-9
    )
awd_yield_grid.to_csv(O / "ext_31_awd_yield_grid.csv", index=False)

# ---------------------------------------------------------------------------
# 23. Land accounting and simple allocation rules
# (a) Does the land accounting matter? The land pool is harvested area, so sequential crops are counted twice.
#     Drop the land balance altogether (any amount of sequential cropping allowed) and find the largest irrigated
#     harvested area the water budget, the Boro floor and the crop limits allow. Where the main model leaves fallow,
#     the land balance is slack and the two areas must be equal.
# (b) Simple allocation rules against the optimised plans: every crop cut in proportion (x = rho a), and Boro cut
#     first down to its floor, then the other crops in proportion. Both keep the water budget and never exceed baseline areas.
# ---------------------------------------------------------------------------
land_rows = []
for rho in (0.8, 0.9):
    tot_free = tot_main = 0.0
    n_bind = 0
    for d in DISTRICTS:
        w = 10 * fixed_coef(d)
        a = area0[d]
        cap = CAPS[CAP0][d]
        s_ = linprog(
            -np.ones(NC),
            A_ub=[w],
            b_ub=[rho * float(a @ w)],
            bounds=[(0.8 * a[0], cap[0])] + [(0, cap[k]) for k in range(1, NC)],
            method="highs",
        )
        assert s_.status == 0
        free_area, main_area = -s_.fun, A0[d] - min_fallow(d, rho)
        assert free_area >= main_area - 1e-6 * A0[d]
        n_bind += int(free_area > main_area + 1e-6 * A0[d])
        tot_free += free_area
        tot_main += main_area
    land_rows.append(
        dict(
            budget_ratio=rho,
            boro_floor=0.8,
            crop_limits=CAP0,
            land_pool_thousand_ha=A_TOTAL / 1e3,
            irrigated_area_with_land_balance_thousand_ha=tot_main / 1e3,
            irrigated_area_without_land_balance_thousand_ha=tot_free / 1e3,
            reduction_with_land_balance_thousand_ha=(A_TOTAL - tot_main) / 1e3,
            reduction_without_land_balance_thousand_ha=(A_TOTAL - tot_free) / 1e3,
            districts_where_land_balance_binds=n_bind,
        )
    )
land_check = pd.DataFrame(land_rows)
assert land_check.loc[land_check.budget_ratio == 0.8, "districts_where_land_balance_binds"].item() == 0
land_check.to_csv(O / "ext_32_land_accounting_check.csv", index=False)


def rule_plan(d, rho, rule):
    a = area0[d].astype(float)
    w = 10 * fixed_coef(d)
    W = rho * float(a @ w)
    if rule == "proportional_cut":
        x = rho * a
    else:  # 'boro_first': Boro down to its floor, then the others in proportion
        x = a.copy()
        x[0] = max(a[0] - (float(a @ w) - W) / w[0], 0.8 * a[0])
        rest, other = W - x[0] * w[0], float(a[1:] @ w[1:])
        x[1:] = a[1:] * min(1.0, rest / other)
    assert x @ w <= W * (1 + 1e-9) and x[0] >= 0.8 * a[0] * (1 - 1e-12) and (x <= a + 1e-9).all()
    return x


rule_rows = []
for rho in (0.8, 0.9):
    plans = {r: {d: rule_plan(d, rho, r) for d in DISTRICTS} for r in ("proportional_cut", "boro_first")}
    for plan, obj in (("least", "food_energy"), ("free", "food_energy"), ("free", "gross_margin")):
        xs_, _, _ = solve_region(GAIN[obj], rho, plan)
        plans[f"optimised_{plan}_fallow_{obj}"] = xs_
    for name, xs_ in plans.items():
        r = describe(xs_, {d: A0[d] - xs_[d].sum() for d in DISTRICTS})
        rule_rows.append(
            dict(
                budget_ratio=rho,
                plan=name,
                irrigated_area_thousand_ha=sum(x.sum() for x in xs_.values()) / 1e3,
                reduction_pct_of_pool=r["fallow_pct_of_land"],
                reduction_thousand_ha=r["fallow_thousand_ha"],
                boro_pct_of_pool=r["boro_pct_of_land"],
                wheat_pct_of_pool=r["wheat_pct_of_land"],
                potato_pct_of_pool=r["potato_pct_of_land"],
                maize_pct_of_pool=r["maize_pct_of_land"],
                food_energy_pct_of_baseline=100
                * r["food_energy_trillion_kcal"]
                / BASE_DESC["food_energy_trillion_kcal"],
                gross_margin_pct_of_baseline=100
                * r["gross_margin_billion_Tk"]
                / BASE_DESC["gross_margin_billion_Tk"],
            )
        )
rule_table = pd.DataFrame(rule_rows)
_p = rule_table[rule_table.plan == "proportional_cut"]
assert np.allclose(_p.food_energy_pct_of_baseline, 100 * _p.budget_ratio) and np.allclose(
    _p.reduction_pct_of_pool, 100 * (1 - _p.budget_ratio)
)
for rho in (0.8, 0.9):  # the optimised least-fallow plan keeps at least as much land in use as any rule
    t = rule_table[rule_table.budget_ratio == rho].set_index("plan")
    assert (
        t.loc["optimised_least_fallow_food_energy", "reduction_thousand_ha"]
        <= t.reduction_thousand_ha.min() + 1e-6
    )
rule_table.to_csv(O / "ext_33_rule_baselines.csv", index=False)

print("Climate cells:", len(cells), "| shared:", [m for m in cells if len(m) > 1])
print(spatial.round(4).to_string(index=False))
print(
    min_fallow_table[
        (min_fallow_table.district == "REGION") & min_fallow_table.budget_ratio.isin([0.8, 0.9, 1.0])
    ]
    .round(2)
    .to_string(index=False)
)
print(pm_out.iloc[[0, 1, 2, -1], :8].to_string(index=False))
print(
    "RAWSI variants:",
    len(variants) - 1,
    "| min Spearman vs primary:",
    round(min(r["spearman_vs_primary"] for r in summary_rows), 4),
)
print(per_ha.round(2).to_string(index=False))
print(gross_margin_table.round(2).to_string(index=False))
print(
    shadow.groupby(["budget_ratio", "plan", "objective"])[
        [
            "water_value_per_1000_m3",
            "land_value_per_ha",
            "fallow_limit_value_per_ha",
            "boro_floor_cost_per_ha",
            "fallow_pct",
        ]
    ]
    .median()
    .round(3)
    .to_string()
)
print(cap_table.round(2).to_string(index=False))
print(scope_table.round(2).to_string(index=False))
print(awd_curve[awd_curve.adoption_share.isin([0, 0.5, 1.0])].round(2).to_string(index=False))
print(water_value.round(2).to_string(index=False))
print(tornado[["group", "factor", "outcome", "low_change", "high_change"]].round(2).to_string(index=False))
print(
    _m[
        [
            "budget_ratio",
            "awd",
            "low_water_crop_limit_x_baseline",
            "min_fallow_thousand_ha",
            "min_fallow_pct_of_pool",
            "four_crop_model_min_fallow_thousand_ha",
        ]
    ]
    .round(2)
    .to_string(index=False)
)
print(land_check.round(2).to_string(index=False))
print(rule_table.round(2).to_string(index=False))
print("EXTENSIONS COMPLETE")
