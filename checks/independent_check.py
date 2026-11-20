#!/usr/bin/env python3
"""Independent re-computation of the main results by different routes.

  python checks/independent_check.py .      (after scripts/run_all.py; prints PASS/FAIL and ALL PASS)

No code is shared with the pipeline scripts. The linear programs are written in other formulations
(fractions of baseline areas, a land inequality, the Bertsimas-Sim dual form of the Gamma-robust
constraint, a pure minimum-fallow LP), regressions are refitted with statsmodels, FAO-56 ET0 is written
again from the equations and tested on FAO-56 Example 18, and dual values are checked by finite
differences. The check tests the algebra and the code, not the modelling choices.

Tolerances: totals in billion Tk or trillion kcal agree to 1e-5 (Tk 10,000 on tens of billions); the
remaining gaps come from the two formulations breaking ties in the least-fallow step differently.
"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr, rankdata

R = sys.argv[1]
I = R + "/inputs/"
X = R + "/generated/extensions/"
CL = R + "/generated/climate/"
p = pd.read_csv(I + "analysis_panel_full.csv")
c = pd.read_csv(CL + "pm_seasonal_fao56_16districts_3crops_2015_2025.csv")
gw = pd.read_csv(I + "hodgson_2021_groundwater_trends.csv")
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, info)


# 1 climate cells via pairwise correlation / equality of season rain for Boro
w = p[p.crop == "Boro Rice"].pivot(index="year_end", columns="district", values="season_rain_mm")
pairs = sorted((a, b) for a in w for b in w if a < b and (w[a] == w[b]).all())
cells = pd.read_csv(X + "ext_01_climate_cells.csv")
exp = sorted(
    (a, b)
    for a in cells.district
    for b in cells.district
    if a < b and cells.set_index("district").climate_cell[a] == cells.set_index("district").climate_cell[b]
)
check(
    "climate cells: identical-series pairs",
    pairs == exp,
    f"{len(pairs)} pairs; distinct={cells.climate_cell.nunique()}",
)
# 2 FE via dummy-variable OLS with cluster-robust SE in statsmodels
fe = pd.read_csv(X + "ext_02_fixed_effects.csv")
gap = 0
for crop, feats in {
    "Boro Rice": "season_rain_mm+monsoon_prev_rain_mm+terminal_tmax_c+season_tmin_c+season_rh_pct+hybrid_area_share",
    "Wheat": "season_rain_mm+monsoon_prev_rain_mm+terminal_tmax_c+season_tmin_c+season_rh_pct",
    "Potato": "season_rain_mm+monsoon_prev_rain_mm+terminal_tmax_c+season_tmin_c+season_rh_pct+hyv_area_share",
}.items():
    g = p[p.crop == crop].reset_index(drop=True)
    m = smf.ols("yield_t_ha ~ " + feats + " + C(district) + C(year_end)", g).fit(
        cov_type="cluster", cov_kwds={"groups": g.district}, use_t=True
    )
    for _, r in fe[fe.crop == crop].iterrows():
        gap = max(
            gap,
            abs(m.params[r.predictor] - r.coef),
            abs(m.bse[r.predictor] - r.cluster_se),
            abs(m.pvalues[r.predictor] - r.p_cluster_df),
        )
    # within R2 = 1 - SSR(full)/SSR(fixed effects only)
    m0 = smf.ols("yield_t_ha ~ C(district) + C(year_end)", g).fit()
    gap = max(gap, abs(1 - m.ssr / m0.ssr - fe[fe.crop == crop].within_R2.iloc[0]))
check("fixed effects: coef, clustered SE, p, within-R2 vs statsmodels LSDV", gap < 1e-8, f"max gap {gap:.2e}")
# 3 PM-deficit feasibility counts straight from the raw files (four crops: maize area from its own file)
mz = pd.read_csv(I + "maize_rabi_16districts_2015_2025.csv")
a = (
    pd.concat([p, mz])[lambda q: q.year_end.isin([2023, 2024, 2025])]
    .groupby(["district", "crop"])
    .area_ha.mean()
    .unstack()[["Boro Rice", "Wheat", "Potato", "Maize"]]
)
pm = pd.read_csv(X + "ext_05_pm_deficit_sensitivity.csv").set_index("water_coefficients")


def counts(coef):
    wd = a * coef
    s = wd["Boro Rice"] / wd.sum(axis=1)
    return [int((s > r).sum()) for r in (0.9, 0.8, 0.7)] + [int((0.8 * s > r).sum()) for r in (0.8, 0.7)]


fixed = pd.DataFrame(
    {k: v for k, v in {"Boro Rice": 676.0, "Wheat": 283.0, "Potato": 376.5, "Maize": 354.0}.items()},
    index=a.index,
)
wr = pd.read_csv(CL + "crop_water_requirement_fao56.csv")
_mz = wr[wr.crop == "Maize"].set_index(["district", "year_end"]).nir_fao56_mm


def addm(q):
    yrs = getattr(addm, "yrs", None)
    return q.assign(
        Maize=(
            _mz.unstack().loc[q.index, yrs].mean(axis=1)
            if yrs is not None
            else _mz.unstack().loc[q.index].mean(axis=1)
        )
    )


addm.yrs = [2023, 2024, 2025]
d1 = addm(
    c[c.year_end.isin([2023, 2024, 2025])]
    .groupby(["district", "crop"])
    .climate_deficit_pm_mm.mean()
    .unstack()
)
addm.yrs = None
d2 = addm(c.groupby(["district", "crop"]).climate_deficit_pm_mm.mean().unstack())
cols = [
    "infeasible_floor100_rho90",
    "infeasible_floor100_rho80",
    "infeasible_floor100_rho70",
    "infeasible_floor80_rho80",
    "infeasible_floor80_rho70",
]
for name, coef in [
    ("fixed_NIR", fixed),
    ("PM_deficit_mean_2023_2025", d1),
    ("PM_deficit_mean_2015_2025", d2),
]:
    check(
        "PM sensitivity counts " + name,
        counts(coef) == [int(v) for v in pm.loc[name, cols]],
        str(counts(coef)),
    )
yr = []
for y in range(2015, 2026):
    addm.yrs = [y]
    yr.append(
        counts(addm(c[c.year_end == y].set_index(["district", "crop"]).climate_deficit_pm_mm.unstack()))
    )
addm.yrs = None
span = [f"{min(v)}-{max(v)}" for v in zip(*yr)]
check("PM single-year ranges", span == list(pm.loc["PM_deficit_single_years_min_max", cols]), str(span))
# 4 RAWSI z-score and leave-2018-out variants by hand
j = p.merge(c, on=["district", "crop", "year_end"])
t = (
    j.assign(wd=j.area_ha * j.nir_mm, cd=j.area_ha * j.climate_deficit_pm_mm)
    .groupby(["district", "year_end"])[["area_ha", "wd", "cd"]]
    .sum()
    .reset_index()
)
t["dem"] = t.wd / t.area_ha
t["dfc"] = t.cd / t.area_ha
sev = ((gw.frac_type1 + gw.frac_falling_mindepth) / 2).set_axis(gw.district)
z = lambda v: (v - v.mean()) / v.std(ddof=0)
mm = lambda v: (v - v.min()) / (v.max() - v.min())
ranks = pd.read_csv(X + "ext_07_rawsi_sensitivity_ranks.csv")


def my(f, tt):
    s = ((tt.district.map(f(sev)) + f(tt.dem) + f(tt.dfc)) / 3).groupby(tt.district).mean()
    return s


for fam, var, s in [
    ("normalisation", "zscore", my(z, t)),
    ("leave_one_year_out", "without 2018", my(mm, t[t.year_end != 2018])),
    ("primary", "equal weights, min-max", my(mm, t)),
]:
    r = ranks[(ranks.variant_family == fam) & (ranks.variant == var)].set_index("district")
    check(
        f"RAWSI {var}",
        np.abs(s.loc[r.index] - r.score).max() < 1e-12
        and (rankdata(-s.loc[r.index], method="ordinal") == r["rank"]).all(),
    )
summ = pd.read_csv(X + "ext_08_rawsi_sensitivity_summary.csv")
s0 = my(mm, t)
s1 = ((0 * t.district.map(mm(sev)) + 0.5 * mm(t.dem) + 0.5 * mm(t.dfc))).groupby(t.district).mean()
check(
    "RAWSI without groundwater Spearman",
    abs(
        spearmanr(s0, s1.loc[s0.index]).statistic
        - summ[summ.variant == "without groundwater"].spearman_vs_primary.iloc[0]
    )
    < 1e-12,
    round(spearmanr(s0, s1.loc[s0.index]).statistic, 4),
)
# 5 Boro share of water demand and its rank correlation with RAWSI
thr = pd.read_csv(X + "ext_10_district_boro_share.csv").set_index("district")
wd = a * fixed
s = wd["Boro Rice"] / wd.sum(axis=1)
check("Boro water share", np.abs(s - thr.boro_water_share.loc[s.index]).max() < 1e-12)
check(
    "Spearman RAWSI vs Boro share",
    abs(spearmanr(s0.loc[s.index], s).statistic - thr.spearman_rawsi_vs_boro_share.iloc[0]) < 1e-12,
    round(spearmanr(s0.loc[s.index], s).statistic, 4),
)
# 5b FAO-56 reference ET written again from the equations, tested on FAO-56 Example 18, then summed per season
import math, glob as _g, re as _re


def et0_ref(tx, tn, rh, u2, rs, lat_deg, J, z):
    T = (tx + tn) / 2
    e = lambda t: 0.6108 * math.exp(17.27 * t / (t + 237.3))
    es = (e(tx) + e(tn)) / 2
    ea = rh / 100 * es
    D = 4098 * e(T) / (T + 237.3) ** 2
    P = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    g = 0.665e-3 * P
    ph = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * J)
    dl = 0.409 * math.sin(2 * math.pi / 365 * J - 1.39)
    ws = math.acos(-math.tan(ph) * math.tan(dl))
    Ra = (
        24
        * 60
        / math.pi
        * 0.0820
        * dr
        * (ws * math.sin(ph) * math.sin(dl) + math.cos(ph) * math.cos(dl) * math.sin(ws))
    )
    Rso = (0.75 + 2e-5 * z) * Ra
    Rnl = (
        4.903e-9
        * ((tx + 273.16) ** 4 + (tn + 273.16) ** 4)
        / 2
        * (0.34 - 0.14 * math.sqrt(ea))
        * (1.35 * min(rs / Rso, 1) - 0.35)
    )
    Rn = 0.77 * rs - Rnl
    return (0.408 * D * Rn + g * 900 / (T + 273) * u2 * (es - ea)) / (D + g * (1 + 0.34 * u2))


# Example 18 (Brussels, 6 July): Tmax 21.5, Tmin 12.3, ea 1.409 kPa, u2 2.078, Rs 22.07, lat 50.80, z 100 -> ET0 3.9 mm/day
_es = (
    0.6108 * math.exp(17.27 * 21.5 / (21.5 + 237.3)) + 0.6108 * math.exp(17.27 * 12.3 / (12.3 + 237.3))
) / 2
ex = et0_ref(21.5, 12.3, 100 * 1.409 / _es, 2.078, 22.07, 50.80, 187, 100)
check(
    "FAO-56 Example 18 reproduced by the check's own ET0 code",
    abs(ex - 3.9) < 0.05,
    f"{ex:.3f} mm/day (published 3.9)",
)
gmax = 0
for f in sorted(_g.glob(I + "nasa_power_daily/DAILY_*.csv")):
    dn = _re.search(r"DAILY_(.+)\.csv", f).group(1)
    L = open(f).read().splitlines()
    k = L.index("-END HEADER-")
    lat = float(_re.search(r"latitude\s+([\d.]+)", L[2] + L[3], _re.I).group(1))
    z = float(_re.search(r"region = ([\d.]+)", "\n".join(L[:k])).group(1))
    dd = pd.read_csv(f, skiprows=k + 1)
    dd["date"] = pd.to_datetime(dd.YEAR.astype(str)) + pd.to_timedelta(dd.DOY - 1, unit="D")
    dd["et0"] = [
        et0_ref(a, b, c_, d_, e_, lat, j, z)
        for a, b, c_, d_, e_, j in zip(dd.T2M_MAX, dd.T2M_MIN, dd.RH2M, dd.WS2M, dd.ALLSKY_SFC_SW_DWN, dd.DOY)
    ]
    ds = dd.set_index("date").et0
    for _, r in wr[wr.district == dn].iterrows():
        st = pd.Timestamp(r.planting)
        gmax = max(gmax, abs(ds.loc[st : st + pd.Timedelta(days=int(r.season_days) - 1)].sum() - r.ET0_mm))
check("seasonal ET0 of all 704 crop-seasons re-summed from daily data", gmax < 1e-6, f"max gap {gmax:.1e} mm")
gn = (wr.nir_fao56_mm - (np.maximum(0, wr.ETc_mm - wr.effective_rain_mm) + wr.percolation_mm)).abs().max()
check("FAO-56 NIR = max(0, ETc - Peff) + percolation", gn < 1e-9, f"max gap {gn:.1e}")
# 6 main-text tables: forecast skill with sklearn; allocation LPs re-solved by a different formulation
import os

assert os.path.exists(R + "/generated/tables/Table3_plans.csv"), "run scripts/run_all.py first"
if True:
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy.optimize import linprog
    from scipy import stats as st

    T = R + "/generated/tables/"
    DD = R + "/generated/decision/"
    fc = pd.read_csv(R + "/generated/forecasts/decision_time_forecasts.csv")
    t2 = pd.read_csv(T + "Table2_forecast_accuracy.csv")
    g2 = 0
    for (crop, model), col in {
        (cr, m): cn
        for cr in ["Boro Rice", "Wheat", "Potato", "Maize"]
        for m, cn in [
            ("Elastic Net", "EN_cutoff2"),
            ("Persistence", "Persistence_cutoff2"),
            ("Climatology", "Climatology_cutoff2"),
            ("District trend", "Trend_cutoff2"),
        ]
    }.items():
        q = fc[fc.crop == crop]
        r = t2[(t2.crop == crop) & (t2.model == model)].iloc[0]
        g2 = max(
            g2,
            abs(r2_score(q.yield_t_ha, q[col]) - r.R2),
            abs(mean_absolute_error(q.yield_t_ha, q[col]) - r.MAE_t_ha),
            abs(((q.yield_t_ha - q[col]) ** 2).mean() ** 0.5 - r.RMSE_t_ha),
        )
    check("Table 2 RMSE, MAE, R2 vs sklearn (four forecasts)", g2 < 1e-12, f"max gap {g2:.1e}")
    # district trend forecast by a closed-form least-squares slope
    pmz = pd.concat([p, mz])
    gt = 0
    for _, r in fc.iterrows():
        h = pmz[(pmz.district == r.district) & (pmz.crop == r.crop) & (pmz.year_end <= r.year_end - 2)]
        b = np.cov(h.year_end, h.yield_t_ha, bias=True)[0, 1] / np.var(h.year_end)
        f = h.yield_t_ha.mean() + b * (r.year_end - h.year_end.mean())
        gt = max(gt, abs(max(0.01, f) - r.Trend_cutoff2))
    check("district-trend forecast (closed-form slope)", gt < 1e-9, f"max gap {gt:.1e}")
    C4 = ["Boro Rice", "Wheat", "Potato", "Maize"]
    yf = (
        fc.groupby(["district", "crop"])
        .EN_cutoff2.mean()
        .unstack()[["Boro Rice", "Wheat", "Potato", "Maize"]]
    )
    n = np.array([676.0, 283.0, 376.5, 354.0])
    k = np.array([3450.0, 3440.0, 554.4, 3440.0])
    t5 = pd.read_csv(T + "Table3_plans.csv")
    HM = pd.concat([p, mz]).groupby(["district", "crop"]).area_ha.max().unstack()[C4]
    ec = pd.read_csv(I + "crop_economics_scenarios.csv").set_index(["crop", "item"]).value
    sold = np.array([1000 / ec[("Boro Rice", "paddy_to_husked_rice")], 1000.0, 1000.0, 1000.0])
    price = np.array(
        [
            ec[("Boro Rice", "price")],
            ec[("Wheat", "price")],
            ec[("Potato", "price_mid")],
            ec[("Maize", "price_mid")],
        ]
    )
    cost = np.array([ec[(cc, "cost")] for cc in C4]) * yf.mean().to_numpy() * sold
    GMh = lambda d, c=None: price * yf.loc[d].to_numpy() * sold - (cost if c is None else c)
    ENh = lambda d: yf.loc[d].to_numpy() * k * 1000

    # A different formulation from the pipeline: variables are fractions of each crop's baseline area,
    # fallow is the slack of a land inequality, and 'least fallow' = first maximise the cropped land.
    def plan(d, gain, rho, kind="free", floor=0.8, crops=4):
        ar = a.loc[d].to_numpy().copy()
        m = np.arange(4) < crops
        ar = ar[m]
        nn = n[m]
        ub = np.maximum(HM.loc[d].to_numpy()[m], ar) / ar
        bd = [(floor, ub[0])] + [(0, u) for u in ub[1:]]
        A_ub = [ar / ar.sum(), ar * nn / (ar @ nn)]
        b_ub = [1, rho]
        if kind == "least":
            s0 = linprog(-ar / ar.sum(), A_ub=A_ub, b_ub=b_ub, bounds=bd, method="highs-ds")
            assert s0.success
            A_ub = A_ub + [-ar / ar.sum()]
            b_ub = b_ub + [s0.fun + 1e-9]
        gg = gain[m]
        s = linprog(-(gg * ar) / np.abs(gg * ar).max(), A_ub=A_ub, b_ub=b_ub, bounds=bd, method="highs-ds")
        assert s.success, (d, rho, kind)
        return np.r_[s.x * ar, np.zeros(4 - crops)]

    def region(gain_of, rho, kind, crops=4):
        xs = {d: plan(d, gain_of(d), rho, kind, crops=crops) for d in a.index}
        land = a.to_numpy()[:, :crops].sum()
        return (
            xs,
            sum(xs[d] @ ENh(d) for d in a.index) / 1e12,
            sum(xs[d] @ GMh(d) for d in a.index) / 1e9,
            100 * (1 - sum(x.sum() for x in xs.values()) / land),
        )

    g5 = 0
    for _, r in t5.iterrows():
        _, en, gm_, fw = region(ENh if r.objective == "food_energy" else GMh, r.budget_ratio, r.plan)
        g5 = max(
            g5,
            (
                abs(en - r.food_energy_trillion_kcal)
                if r.objective == "food_energy"
                else abs(gm_ - r.gross_margin_billion_Tk)
            ),
        )
        if r.plan == "least":
            g5 = max(g5, abs(fw - r.fallow_pct_of_land) * 1e-2)
    check(
        "Table 3 plans re-solved (fractions, land inequality, dual simplex; 12 plans, four crops)",
        g5 < 1e-5,
        f"max gap {g5:.1e}",
    )
    e23 = pd.read_csv(X + "ext_23_three_vs_four_crops.csv")
    g23 = 0
    for _, r in e23.iterrows():
        _, en, gm_, fw = region(
            ENh if r.objective == "food_energy" else GMh,
            r.budget_ratio,
            r.plan,
            crops=3 if r.scope == "three_crops" else 4,
        )
        g23 = max(
            g23,
            (
                abs(en - r.food_energy_trillion_kcal)
                if r.objective == "food_energy"
                else abs(gm_ - r.gross_margin_billion_Tk)
            ),
        )
    check("model with and without maize re-solved (24 plans)", g23 < 1e-5, f"max gap {g23:.1e}")
    # smallest fallow share (Table S6b) = 1 - largest cropped share the budget allows
    t4b = pd.read_csv(T + "TableS6b_minimum_fallow.csv")
    g4b = 0
    for _, r in t4b.iterrows():
        for rho in (1.0, 0.9, 0.8, 0.7):
            crop = 0
            feas = True
            for d in a.index:
                ar = a.loc[d].to_numpy()
                ub = np.maximum(HM.loc[d].to_numpy(), ar) / ar
                s0 = linprog(
                    -ar,
                    A_ub=[ar, ar * n / (ar @ n)],
                    b_ub=[ar.sum(), rho],
                    bounds=[(r.boro_floor, ub[0])] + [(0, u) for u in ub[1:]],
                    method="highs-ds",
                )
                if not s0.success:
                    feas = False
                    continue
                crop += -s0.fun
            v = r[f"min_fallow_pct_at_budget_{rho:.2f}"]
            g4b = max(
                g4b, abs(100 * (1 - crop / a.to_numpy().sum()) - v) if feas else (0 if np.isnan(v) else 1)
            )
    check("Table S6b smallest fallow share re-solved (max cropped land)", g4b < 1e-6, f"max gap {g4b:.1e}")

    # AWD (ext_25, ext_26): Boro uses n_B(1 - phi s) per ha, the budget stays rho x conventional demand
    def aplan(d, gain, rho, kind, sv, phi, floor=0.8):
        ar = a.loc[d].to_numpy()
        ne = n * np.r_[1 - phi * sv, 1, 1, 1]
        ub = np.maximum(HM.loc[d].to_numpy(), ar) / ar
        bd = [(floor, ub[0])] + [(0, u) for u in ub[1:]]
        A_ub = [ar / ar.sum(), ar * ne / (ar @ n)]
        b_ub = [1, rho]
        s0 = linprog(-ar / ar.sum(), A_ub=A_ub, b_ub=b_ub, bounds=bd, method="highs-ds")
        if not s0.success:
            return None, None
        if kind == "least":
            A_ub = A_ub + [-ar / ar.sum()]
            b_ub = b_ub + [s0.fun + 1e-9]
        if gain is None:
            return None, ar.sum() * (1 + s0.fun)
        s = linprog(
            -(gain * ar) / np.abs(gain * ar).max(), A_ub=A_ub, b_ub=b_ub, bounds=bd, method="highs-ds"
        )
        assert s.success
        return s.x * ar, ar.sum() * (1 + s0.fun)

    LAND = a.to_numpy().sum()
    e26 = pd.read_csv(X + "ext_26_awd_fallow_curve.csv")
    g26 = 0
    for _, r in e26.iterrows():
        fw = sum(aplan(d, None, r.budget_ratio, "free", r.water_saving, r.adoption_share)[1] for d in a.index)
        nin = sum(
            aplan(d, None, r.budget_ratio, "free", r.water_saving, r.adoption_share, floor=1.0)[1] is None
            for d in a.index
        )
        g26 = max(g26, abs(100 * fw / LAND - r.min_fallow_pct), abs(nin - r.districts_full_boro_infeasible))
    check(
        "AWD fallow curve re-solved (smallest fallow, full-Boro infeasibility; 252 rows)",
        g26 < 1e-6,
        f"max gap {g26:.1e}",
    )
    e25 = pd.read_csv(X + "ext_25_awd_scenarios.csv")
    g25 = 0
    for _, r in e25.iterrows():
        yk = np.r_[1 + r.boro_yield_change_pct / 100, 1, 1, 1]
        en_ = lambda d: yf.loc[d].to_numpy() * yk * k * 1000
        gm_f = lambda d: price * yf.loc[d].to_numpy() * yk * sold - cost
        gf = en_ if r.objective == "food_energy" else gm_f
        xs = {
            d: aplan(d, gf(d), r.budget_ratio, r.plan, r.water_saving, r.adoption_share)[0] for d in a.index
        }
        g25 = max(
            g25,
            (
                abs(sum(xs[d] @ en_(d) for d in a.index) / 1e12 - r.food_energy_trillion_kcal)
                if r.objective == "food_energy"
                else abs(sum(xs[d] @ gm_f(d) for d in a.index) / 1e9 - r.gross_margin_billion_Tk)
            ),
        )
    check("AWD scenarios re-solved (optimised quantity, 204 plans)", g25 < 1e-5, f"max gap {g25:.1e}")
    # model value of water vs irrigation charges (ext_27) from the reference file and ext_15
    ch = pd.read_csv(I + "irrigation_charge_references.csv").set_index("item")
    e27 = pd.read_csv(X + "ext_27_water_value_vs_charges.csv")
    hr = ch.loc["bmda_dtw_hourly_charge"]
    c_lo, c_hi = sorted(
        [hr.low / (hr.discharge_low_L_per_min * 0.06), hr.high / (hr.discharge_high_L_per_min * 0.06)]
    )
    s15 = pd.read_csv(X + "ext_15_shadow_prices.csv")
    g27 = 0
    for _, r in e27.iterrows():
        v = s15[
            np.isclose(s15.budget_ratio, r.budget_ratio)
            & (s15.plan == r.plan)
            & (s15.objective == "gross_margin")
        ].water_value_per_1000_m3
        g27 = max(
            g27,
            abs(v.median() - r.water_value_Tk_per_m3_median),
            abs(c_lo - r.bmda_dtw_hourly_charge_Tk_per_m3_low),
            abs(c_hi - r.bmda_dtw_hourly_charge_Tk_per_m3_high),
            abs(
                ch.loc["boro_irrigation_cost_per_ha_2016_17", "high"] / 6760
                - r.boro_irrigation_cost_per_ha_2016_17_Tk_per_m3_high
            ),
        )
    check("water value vs irrigation charges (Tk per m3) recomputed", g27 < 1e-9, f"max gap {g27:.1e}")
    # tornado (ext_28): main case, potato price and procurement price re-solved; AWD bars agree with ext_26
    e28 = pd.read_csv(X + "ext_28_sensitivity_tornado.csv").set_index(["factor", "outcome"])
    g28 = 0
    gmain = sum(plan(d, GMh(d), 0.8) @ GMh(d) for d in a.index) / 1e9
    g28 = max(
        g28, abs(gmain - e28.loc[("Potato price: low / high", "gross_margin_billion_Tk"), "main_value"])
    )
    vals = []
    for pp in ("price_low", "price_high"):
        pr = price.copy()
        pr[2] = ec[("Potato", pp)]
        gmf = lambda d: pr * yf.loc[d].to_numpy() * sold - cost
        vals.append(sum(plan(d, gmf(d), 0.8) @ gmf(d) for d in a.index) / 1e9)
    r = e28.loc[("Potato price: low / high", "gross_margin_billion_Tk")]
    g28 = max(g28, abs(min(vals) - r.low_value), abs(max(vals) - r.high_value))
    pr = price.copy()
    pr[0] = ec[("Boro Rice", "price_procurement")]
    pr[1] = ec[("Wheat", "price_procurement")]
    gmf = lambda d: pr * yf.loc[d].to_numpy() * sold - cost
    g28 = max(
        g28,
        abs(
            sum(plan(d, gmf(d), 0.8) @ gmf(d) for d in a.index) / 1e9
            - e28.loc[("Boro and wheat at procurement price", "gross_margin_billion_Tk"), "low_value"]
        ),
    )
    c80 = e26[np.isclose(e26.budget_ratio, 0.8) & np.isclose(e26.adoption_share, 1.0)].min_fallow_pct
    r = e28.loc[("AWD on all Boro area, 15-30% saving", "min_fallow_pct")]
    g28 = max(g28, abs(c80.min() - r.low_value), abs(c80.max() - r.high_value))
    check(
        "tornado: main case, potato and procurement prices re-solved; AWD bars match the AWD curve",
        g28 < 1e-5,
        f"max gap {g28:.1e}",
    )
    # six-crop variant with lentil and mustard (ext_29): own data handling, fractions of baseline, dual simplex
    lmn = pd.read_csv(I + "lentil_mustard_16districts_2015_2025.csv")
    LWc = ["Lentil", "Mustard"]
    la = lmn[lmn.year_end.isin([2023, 2024, 2025])].pivot_table(
        index="district", columns="crop", values="area_ha", aggfunc="mean"
    )[LWc]
    ly = lmn[lmn.year_end.isin([2021, 2022, 2023])].pivot_table(
        index="district", columns="crop", values="yield_t_ha", aggfunc="mean"
    )[LWc]
    lmax = lmn.pivot_table(index="district", columns="crop", values="area_ha", aggfunc="max")[LWc]
    n6 = np.r_[n, 197.5, 93.0]
    k6 = np.r_[k, 3170 * ec[("Lentil", "dal_per_grain")], 9000 * ec[("Mustard", "oil_per_seed")]]
    lcost = np.array([ec[(c, "cost")] for c in LWc]) * ly.mean().to_numpy() * 1000
    lprice = np.array([ec[(c, "price")] for c in LWc])

    def six(d, obj, rho, kk, bw, kind):
        ar = np.r_[a.loc[d].to_numpy(), la.loc[d].to_numpy()]
        yy = np.r_[yf.loc[d].to_numpy(), ly.loc[d].to_numpy()]
        ub = (
            np.r_[
                np.maximum(HM.loc[d].to_numpy(), ar[:4]),
                np.maximum(np.maximum(lmax.loc[d].to_numpy(), ar[4:]), kk * ar[4:]),
            ]
            / ar
        )
        bd = [(0.8, ub[0])] + [(0, u) for u in ub[1:]]
        ne = n6 * np.r_[bw, 1, 1, 1, 1, 1]
        A_ub = [ar / ar.sum(), ar * ne / (ar @ n6)]
        b_ub = [1, rho]
        s0 = linprog(-ar / ar.sum(), A_ub=A_ub, b_ub=b_ub, bounds=bd, method="highs-ds")
        assert s0.success
        fal = ar.sum() * (1 + s0.fun)
        if kind == "least":
            A_ub = A_ub + [-ar / ar.sum()]
            b_ub = b_ub + [s0.fun + 1e-9]
        en = yy * k6 * 1000
        gm = np.r_[GMh(d), lprice * yy[4:] * 1000 - lcost]
        g = en if obj == "food_energy" else gm
        s = linprog(-(g * ar) / np.abs(g * ar).max(), A_ub=A_ub, b_ub=b_ub, bounds=bd, method="highs-ds")
        assert s.success
        return fal, (s.x * ar) @ g

    e29 = pd.read_csv(X + "ext_29_low_water_crops.csv")
    g29 = 0
    BW = {"none": 1.0, "awd_half_boro_20pct_saving": 0.9}
    for _, r in e29.iterrows():
        res = [
            six(d, r.objective, r.budget_ratio, r.low_water_crop_limit_x_baseline, BW[r.awd], r.plan)
            for d in a.index
        ]
        g29 = max(
            g29,
            abs(sum(x[0] for x in res) / 1e3 - r.min_fallow_thousand_ha) * 1e-2,
            (
                abs(sum(x[1] for x in res) / 1e12 - r.food_energy_trillion_kcal)
                if r.objective == "food_energy"
                else abs(sum(x[1] for x in res) / 1e9 - r.gross_margin_billion_Tk)
            ),
        )
    check(
        "six-crop variant with lentil and mustard re-solved (144 plans: smallest fallow and optimised quantity)",
        g29 < 1e-5,
        f"max gap {g29:.1e}",
    )
    # lentil and mustard input against the three values that were replaced, and the conversion constant
    okl = (
        len(lmn) == 352
        and np.allclose(lmn.area_ha, lmn.area_acre * 0.40468564224, rtol=1e-9)
        and int((lmn.note.fillna("") != "").sum()) == 3
    )
    check(
        "lentil and mustard input: 352 rows, acre-to-hectare conversion, three documented replacements", okl
    )
    # water value by finite difference (no dual values used), food-energy objective, free fallow, budget 0.80
    shd = pd.read_csv(X + "ext_15_shadow_prices.csv")
    shd = shd[
        np.isclose(shd.budget_ratio, 0.8) & (shd.objective == "food_energy") & (shd.plan == "free")
    ].set_index("district")
    gfd = 0
    for d in a.index:
        ar = a.loc[d].to_numpy()
        dr = 1e-4
        fd = (
            (plan(d, ENh(d), 0.8 + dr) @ ENh(d) - plan(d, ENh(d), 0.8) @ ENh(d))
            / (dr * (ar @ n) * 10)
            * 1000
            / 1e6
        )
        gfd = max(gfd, abs(fd - shd.water_value_per_1000_m3[d]))
    check(
        "water shadow price (food energy, budget 0.80) vs finite difference",
        gfd < 1e-6,
        f"max gap {gfd:.1e} million kcal per 1000 m3",
    )
    e16 = pd.read_csv(X + "ext_16_budget_response.csv")
    g16 = 0
    for _, r in e16.iterrows():
        _, en, gm_, _ = region(ENh if r.objective == "food_energy" else GMh, r.budget_ratio, r.plan)
        g16 = max(
            g16,
            (
                abs(en - r.food_energy_trillion_kcal)
                if r.objective == "food_energy"
                else abs(gm_ - r.gross_margin_billion_Tk)
            ),
        )
    check(
        "budget-response curves re-solved (optimised quantity, 12 budgets x 2 plans)",
        g16 < 1e-5,
        f"max gap {g16:.1e}",
    )
    e17 = pd.read_csv(X + "ext_17_energy_margin_frontier.csv")
    okf = True
    for rho in (0.8, 1.0):
        for kind in ("free", "least"):
            g = e17[np.isclose(e17.budget_ratio, rho) & (e17.plan == kind)].sort_values("weight_on_margin")
            r5 = t5[np.isclose(t5.budget_ratio, rho) & (t5.plan == kind)].set_index("objective")
            okf &= (
                abs(g.food_energy_trillion_kcal.iloc[0] - r5.food_energy_trillion_kcal["food_energy"]) < 1e-6
                and abs(g.gross_margin_billion_Tk.iloc[-1] - r5.gross_margin_billion_Tk["gross_margin"])
                < 1e-6
            )
            okf &= bool(
                (np.diff(g.gross_margin_billion_Tk) > -1e-9).all()
                and (np.diff(g.food_energy_trillion_kcal) < 1e-9).all()
            )
    check("energy-margin frontier: end points equal the two optima, monotone", okf)
    e18 = pd.read_csv(X + "ext_18_regional_yield_area_by_year.csv").set_index(["crop", "year_end"])
    s18 = p.groupby(["crop", "year_end"])[["production_t", "area_ha"]].sum()
    g18 = max(
        (s18.production_t / s18.area_ha - e18.regional_yield_t_ha).abs().max(),
        (s18.area_ha / 1e3 - e18.regional_area_thousand_ha).abs().max(),
    )
    check("regional yield and area by year from the panel", g18 < 1e-9, f"max gap {g18:.1e}")
    from scipy.stats import pearsonr

    e19 = pd.read_csv(X + "ext_19_yield_climate_correlation.csv")
    g19 = 0
    q = p.assign(year_trend=p.year_end - p.year_end.min())
    for _, r in e19.iterrows():
        g = q[q.crop == r.crop]
        g19 = max(g19, abs(pearsonr(g[r.variable], g.yield_t_ha)[0] - r.pearson_r))
        res = lambda col: smf.ols(col + " ~ C(district)", g).fit().resid
        g19 = max(g19, abs(pearsonr(res(r.variable), res("yield_t_ha"))[0] - r.pearson_r_within_district))
    check(
        "yield-predictor correlations (pooled; within district via residuals on district dummies)",
        g19 < 1e-9,
        f"max gap {g19:.1e}",
    )
    e20 = pd.read_csv(X + "ext_20_wheat_cost_sensitivity.csv")
    g20 = 0
    for _, r in e20.iterrows():
        c2 = (
            np.array(
                [
                    ec[("Boro Rice", "cost")],
                    r.wheat_cost_Tk_per_kg,
                    ec[("Potato", "cost")],
                    ec[("Maize", "cost")],
                ]
            )
            * yf.mean().to_numpy()
            * sold
        )
        gm2 = sum(plan(d, GMh(d, c2), r.budget_ratio, r.plan) @ GMh(d, c2) for d in a.index) / 1e9
        g20 = max(g20, abs(gm2 - r.gross_margin_billion_Tk))
    okc = np.allclose(
        sorted(e20.wheat_cost_Tk_per_kg.unique()),
        [0.8 * ec[("Wheat", "cost")], ec[("Wheat", "cost")], 1.2 * ec[("Wheat", "cost")]],
    )
    check("wheat-cost sensitivity (-20%, reported, +20%) re-solved", g20 < 1e-5 and okc, f"max gap {g20:.1e}")
    e22 = pd.read_csv(X + "ext_22_maize_price_cost_sensitivity.csv")
    g22 = 0
    for _, r in e22.iterrows():
        c2 = (
            np.array(
                [
                    ec[("Boro Rice", "cost")],
                    ec[("Wheat", "cost")],
                    ec[("Potato", "cost")],
                    r.maize_cost_Tk_per_kg,
                ]
            )
            * yf.mean().to_numpy()
            * sold
        )
        pr = price.copy()
        pr[3] = r.maize_price_Tk_per_kg
        gmf = lambda d: pr * yf.loc[d].to_numpy() * sold - c2
        gm2 = sum(plan(d, gmf(d), r.budget_ratio, r.plan) @ gmf(d) for d in a.index) / 1e9
        g22 = max(g22, abs(gm2 - r.gross_margin_billion_Tk) / abs(r.gross_margin_billion_Tk))
    check("maize price x cost sensitivity re-solved (36 plans)", g22 < 1e-6, f"max relative gap {g22:.1e}")
    t4 = pd.read_csv(T + "TableS6a_feasibility_grid.csv")
    sh = {"Fixed NIR": (a * fixed), "PM climate deficit, 2023-2025 mean": (a * d1)}
    okk = True
    for _, r in t4.iterrows():
        w = sh[r.water_coefficients]
        s_ = w["Boro Rice"] / w.sum(axis=1)
        okk &= [int((r.boro_floor * s_ > rho + 1e-12).sum()) for rho in (1.0, 0.9, 0.8, 0.7)] == [
            int(r[f"infeasible_at_budget_{rho:.2f}"]) for rho in (1.0, 0.9, 0.8, 0.7)
        ]
    check("Table S6a feasibility grid from raw files", okk)
    # 7 decision-time model (stage 5): Gamma-robust water by the Bertsimas-Sim dual reformulation
    # (one constraint with protection variables z, p_c) instead of the pipeline's enumeration of crop subsets
    NUP = np.array([779.0, 339.0, 444.0, 429.0])
    CV = np.array([0.072, 0.137, 0.126, 0.148])
    DEV = {"temporal_cv": n * CV, "spatial_upper": NUP - n}
    AH = pd.concat([p, mz]).pivot_table(index=["district", "year_end"], columns="crop", values="area_ha")[C4]
    YH = pd.concat([p, mz]).pivot_table(index=["district", "year_end"], columns="crop", values="yield_t_ha")[
        C4
    ]

    def dplan(d, t, y, rho, spec, gamma, kind, obj):
        ar = AH.loc[(d, t - 2)].to_numpy()
        cap = AH.loc[d][AH.loc[d].index <= t - 2].max().to_numpy() / ar
        dev = np.zeros(4) if spec == "nominal" else DEV[spec]
        W0 = ar @ n
        # variables: f (4, fraction of t-2 area), z, p (4)
        A = [np.r_[ar * n / W0, gamma, np.ones(4)]]
        b = [rho]
        for j in range(4):
            e = np.zeros(9)
            e[j] = dev[j] * ar[j] / W0
            e[4] = -1
            e[5 + j] = -1
            A.append(e)
            b.append(0)
        A.append(np.r_[ar / ar.sum(), np.zeros(5)])
        b.append(1)
        bd = [(0.8, cap[0])] + [(0, u) for u in cap[1:]] + [(0, None)] * 5
        if kind == "least":
            s0 = linprog(np.r_[-ar / ar.sum(), np.zeros(5)], A_ub=A, b_ub=b, bounds=bd, method="highs-ds")
            if not s0.success:
                return None
            A = A + [np.r_[-ar / ar.sum(), np.zeros(5)]]
            b = b + [s0.fun + 1e-9]
        cost_ha = (
            np.array([ec[(cc, "cost")] for cc in C4])
            * YH.xs(t - 2, level="year_end").mean().to_numpy()
            * sold
        )
        g = 1000 * y * k if obj == "food_energy" else price * y * sold - cost_ha
        s = linprog(
            np.r_[-(g * ar) / np.abs(g * ar).max(), np.zeros(5)], A_ub=A, b_ub=b, bounds=bd, method="highs-ds"
        )
        return (s.x[:4] * ar, g) if s.success else None

    dp = pd.read_csv(DD + "decision_plans.csv")
    gd = 0
    nfe = 0
    sub = dp[
        (dp.yield_source == "Oracle") & dp.budget_ratio.isin([0.8, 1.0]) & (dp.objective == "food_energy")
    ]
    for _, r in sub.iterrows():
        y = YH.loc[(r.district, r.year_end)].to_numpy()
        res = dplan(
            r.district, r.year_end, y, r.budget_ratio, r.uncertainty_set, r.gamma, r.plan, r.objective
        )
        if (res is None) != (not r.feasible):
            nfe += 1
            continue
        if res is not None:
            gd = max(gd, abs(res[0] @ res[1] - r.realised_value) / r.realised_value)
    check(
        "decision-time oracle plans re-solved with the dual robust formulation (food energy, budgets 0.8 and 1.0, all NIR sets)",
        gd < 1e-6 and nfe == 0,
        f"{len(sub)} plans; max relative gap {gd:.1e}; feasibility mismatches {nfe}",
    )
    # forecast-based plans scored ex post: Elastic Net plans (nominal) give the stored realised value
    sub = dp[(dp.yield_source == "EN_cutoff2") & (dp.uncertainty_set == "nominal")]
    ge = 0
    fcp = fc.set_index(["district", "year_end", "crop"])
    for _, r in sub.iterrows():
        y = np.array([fcp.loc[(r.district, r.year_end, cc)].EN_cutoff2 for cc in C4])
        x, _ = dplan(r.district, r.year_end, y, r.budget_ratio, "nominal", 0, r.plan, r.objective)
        yt = YH.loc[(r.district, r.year_end)].to_numpy()
        cost_ha = (
            np.array([ec[(cc, "cost")] for cc in C4])
            * YH.xs(r.year_end - 2, level="year_end").mean().to_numpy()
            * sold
        )
        gt_ = 1000 * yt * k if r.objective == "food_energy" else price * yt * sold - cost_ha
        ge = max(ge, abs(x @ gt_ - r.realised_value) / abs(r.realised_value))
    check(
        "Elastic Net plans scored with true yields (nominal NIR, 288 plans)",
        ge < 1e-6,
        f"max relative gap {ge:.1e}",
    )
    t7 = pd.read_csv(T + "Table4_decision_value.csv")
    gtt = 0
    for _, r in t7.iterrows():
        q = dp[
            (dp.uncertainty_set == "nominal")
            & np.isclose(dp.budget_ratio, r.budget_ratio)
            & (dp.objective == r.objective)
            & (dp.plan == r.plan)
        ]
        b = {
            "Persistence": "Persistence_cutoff2",
            "Climatology": "Climatology_cutoff2",
            "District trend": "Trend_cutoff2",
        }[r.EN_vs]
        q = q.assign(
            L=np.where(
                q.objective == "food_energy",
                100 * (q.oracle_value - q.realised_value) / q.oracle_value,
                (q.oracle_value - q.realised_value) / q.pool_ha,
            )
        )
        w = q.pivot(index=["district", "year_end"], columns="yield_source", values="L")
        dd = (w["EN_cutoff2"] - w[b]).groupby(level=0).mean()
        gtt = max(gtt, abs(dd.mean() - r.mean_difference))
        if dd.std() > 0:
            gtt = max(gtt, abs(st.ttest_1samp(dd, 0).pvalue - r.dm_p_two_sided))
    check(
        "Table 4 regret differences and t-test p-values (scipy ttest_1samp on district means)",
        gtt < 1e-6,
        f"max gap {gtt:.1e}",
    )
    # climate predictors: season rain and mean Tmin by a long-format group-by over all 16 raw files
    import glob, re, calendar

    cl = pd.read_csv(R + "/generated/climate/climate_predictors_from_raw.csv").set_index(
        ["district", "crop", "year_end"]
    )
    gcl = 0
    for f in glob.glob(I + "nasa_power/POWER_*.csv"):
        d = re.search(r"POWER_(.+)\.csv", f).group(1)
        L = open(f).read().splitlines()
        k = L.index("-END HEADER-")
        lg = pd.read_csv(f, skiprows=k + 1).melt(
            id_vars=["PARAMETER", "YEAR"],
            value_vars=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
            var_name="mon",
        )
        lg["m"] = lg.mon.map(
            {
                x: i + 1
                for i, x in enumerate(
                    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                )
            }
        )
        lg["days"] = [calendar.monthrange(y, m)[1] for y, m in zip(lg.YEAR, lg.m)]
        for crop, (a0, b0) in {"Boro Rice": (12, 5), "Wheat": (11, 3), "Potato": (11, 2)}.items():
            # season label = year of the last month; months from a0 of the previous year
            lg["season"] = np.where(lg.m >= a0, lg.YEAR + 1, lg.YEAR)
            inn = (lg.m >= a0) | (lg.m <= b0)
            rr = (
                lg[inn & (lg.PARAMETER == "PRECTOTCORR")]
                .assign(v=lambda t: t.value * t.days)
                .groupby("season")
                .v.sum()
            )
            tn = lg[inn & (lg.PARAMETER == "T2M_MIN")].groupby("season").value.mean()
            for y in range(2015, 2026):
                gcl = max(
                    gcl,
                    abs(rr[y] - cl.loc[(d, crop, y), "season_rain_mm"]),
                    abs(tn[y] - cl.loc[(d, crop, y), "season_tmin_c"]),
                )
    check(
        "climate predictors re-derived from the 16 raw NASA files (season rain, Tmin)",
        gcl < 1e-9,
        f"max gap {gcl:.1e}",
    )
# crops at their area limit in the exact least-fallow solution (pure minimum-fallow LP, no second stage), budget 0.80
from scipy.optimize import linprog as _lp_lim

_pan = pd.concat(
    [
        p[["district", "crop", "year_end", "area_ha"]],
        pd.read_csv(I + "maize_rabi_16districts_2015_2025.csv")[["district", "crop", "year_end", "area_ha"]],
    ]
)
_four = ["Boro Rice", "Wheat", "Potato", "Maize"]
_b = (
    _pan[_pan.year_end.isin([2023, 2024, 2025])].groupby(["district", "crop"]).area_ha.mean().unstack()[_four]
)
_c = _pan.groupby(["district", "crop"]).area_ha.max().unstack()[_four]
_n = np.array([676.0, 283.0, 376.5, 354.0])
_nw = _nm = _np = 0
for _d in _b.index:
    _a0 = _b.loc[_d].to_numpy()
    _cap = _c.loc[_d].to_numpy()
    _r = _lp_lim(
        np.r_[0, 0, 0, 0, 1.0],
        A_ub=[np.r_[_n, 0]],
        b_ub=[0.8 * (_n @ _a0)],
        A_eq=[np.ones(5)],
        b_eq=[_a0.sum()],
        bounds=[(0.8 * _a0[0], _cap[0])] + [(0, _cap[j]) for j in (1, 2, 3)] + [(0, None)],
        method="highs",
    )
    _nw += int(_r.x[1] >= _cap[1] - 1e-3)
    _np += int(_r.x[2] >= _cap[2] - 1e-3)
    _nm += int(_r.x[3] >= _cap[3] - 1e-3)
_s4 = pd.read_csv(R + "/generated/tables/TableS9_dual_values.csv")
_g = _s4[np.isclose(_s4.budget_ratio, 0.8) & (_s4.plan == "least")]
check(
    "districts with wheat / potato / maize at the area limit, least-fallow plan at budget 0.80 (table S4 vs pure minimum-fallow LP)",
    len(_g) == 2
    and (_g.districts_wheat_at_limit == _nw).all()
    and (_g.districts_potato_at_limit == _np).all()
    and (_g.districts_maize_at_limit == _nm).all(),
    f"{_nw} / {_np} / {_nm}",
)


# Smallest fallow of the Boro floor x crop-limit grid (ext_30) and of the AWD saving x yield grid (ext_31),
# each re-solved here as a pure minimum-fallow LP
def _minfal(a0, cap, floor, rho, coef=None, budget_coef=None):
    coef = _n if coef is None else coef
    bc = _n if budget_coef is None else budget_coef
    r = _lp_lim(
        np.r_[0, 0, 0, 0, 1.0],
        A_ub=[np.r_[coef, 0]],
        b_ub=[rho * (bc @ a0)],
        A_eq=[np.ones(5)],
        b_eq=[a0.sum()],
        bounds=[(floor * a0[0], cap[0])] + [(0, cap[j]) for j in (1, 2, 3)] + [(0, None)],
        method="highs",
    )
    return float(r.x[4]) if r.status == 0 else None


_lims = {
    "observed_max_2015_2025": lambda a0, cm: np.maximum(cm, a0),
    "observed_max_or_1.5x_baseline": lambda a0, cm: np.maximum(np.maximum(cm, a0), 1.5 * a0),
    "observed_max_or_2x_baseline": lambda a0, cm: np.maximum(np.maximum(cm, a0), 2 * a0),
    "no_limit": lambda a0, cm: np.full(4, a0.sum()),
}
_land = _b.to_numpy().sum()
_g30 = pd.read_csv(X + "ext_30_floor_limit_grid.csv")
_gap30 = 0.0
_inf_ok = True
for _, r in _g30.iterrows():
    res = [
        _minfal(
            _b.loc[d].to_numpy(),
            _lims[r.crop_limits](_b.loc[d].to_numpy(), _c.loc[d].to_numpy()),
            r.boro_floor,
            r.budget_ratio,
        )
        for d in _b.index
    ]
    n_inf = sum(v is None for v in res)
    _inf_ok &= n_inf == r.districts_infeasible
    if n_inf == 0:
        _gap30 = max(_gap30, abs(100 * sum(res) / _land - r.min_fallow_pct))
check(
    "floor x crop-limit grid: infeasible districts and smallest fallow (56 runs, minimum-fallow LP written here)",
    _inf_ok and _gap30 < 1e-6,
    f"max gap {_gap30:.1e} pp",
)
_g31 = pd.read_csv(X + "ext_31_awd_yield_grid.csv")
_gap31 = 0.0
for _, r in _g31.iterrows():
    k = np.r_[1 - r.adoption_share * r.water_saving, 1, 1, 1]
    res = [
        _minfal(
            _b.loc[d].to_numpy(),
            np.maximum(_c.loc[d].to_numpy(), _b.loc[d].to_numpy()),
            0.8,
            r.budget_ratio,
            coef=_n * k,
            budget_coef=_n,
        )
        for d in _b.index
    ]
    _gap31 = max(_gap31, abs(100 * sum(res) / _land - r.min_fallow_pct))
check(
    "AWD saving x yield grid: smallest fallow with the AWD coefficient and the budget at conventional demand (32 runs)",
    _gap31 < 1e-6,
    f"max gap {_gap31:.1e} pp",
)
# The Elastic Net contributions add up to the published forecasts (baseline + contributions = EN_cutoff2)
_ct = pd.read_csv(R + "/generated/forecasts/en_contributions.csv")
_fc = pd.read_csv(R + "/generated/forecasts/decision_time_forecasts.csv")
_m = _ct.merge(_fc, on=["district", "crop", "year_end"], validate="one_to_one")
_sum = _m.baseline_t_ha + _m[["trend", "lag2_yield", "lag2_4_mean_yield", "district_effect"]].sum(axis=1)
_gap_en = float(np.abs(np.maximum(0.01, _sum) - _m.EN_cutoff2).max())
check(
    "Elastic Net: baseline plus predictor contributions reproduce all 192 decision-time forecasts",
    len(_m) == 192 and _gap_en < 1e-9,
    f"max gap {_gap_en:.1e} t/ha",
)
# (a) Without any land balance, the largest irrigated harvested area allowed by water, Boro floor and
# crop limits equals the land pool minus the smallest fallow (ext_32). (b) The two simple rules (ext_33), recomputed here.
_g32 = pd.read_csv(X + "ext_32_land_accounting_check.csv")
_gap32 = 0.0
for _, r in _g32.iterrows():
    tot = 0.0
    for d in _b.index:
        a0 = _b.loc[d].to_numpy()
        cap = np.maximum(_c.loc[d].to_numpy(), a0)
        rr = _lp_lim(
            -np.ones(4),
            A_ub=[_n],
            b_ub=[r.budget_ratio * (_n @ a0)],
            bounds=[(0.8 * a0[0], cap[0])] + [(0, cap[j]) for j in (1, 2, 3)],
            method="highs",
        )
        tot += -rr.fun
    mf = sum(
        _minfal(
            _b.loc[d].to_numpy(), np.maximum(_c.loc[d].to_numpy(), _b.loc[d].to_numpy()), 0.8, r.budget_ratio
        )
        for d in _b.index
    )
    _gap32 = max(
        _gap32,
        abs(tot / 1e3 - r.irrigated_area_without_land_balance_thousand_ha),
        abs((_land - mf) / 1e3 - r.irrigated_area_with_land_balance_thousand_ha),
    )
check(
    "land-accounting check: largest irrigated area without a land balance, and land pool minus smallest fallow (budgets 0.8, 0.9)",
    _gap32 < 1e-3 and (_g32.districts_where_land_balance_binds == 0).all(),
    f"max gap {_gap32:.1e} thousand ha",
)
_g33 = pd.read_csv(X + "ext_33_rule_baselines.csv")
_gap33 = 0.0
for rho in (0.8, 0.9):
    for rule in ("proportional_cut", "boro_first"):
        area = 0.0
        for d in _b.index:
            a0 = _b.loc[d].to_numpy()
            W = rho * (_n @ a0)
            if rule == "proportional_cut":
                x = rho * a0
            else:
                x = a0.copy()
                x[0] = max(a0[0] - ((_n @ a0) - W) / _n[0], 0.8 * a0[0])
                x[1:] = a0[1:] * min(1.0, (W - x[0] * _n[0]) / (_n[1:] @ a0[1:]))
            area += x.sum()
        row = _g33[np.isclose(_g33.budget_ratio, rho) & (_g33.plan == rule)].iloc[0]
        _gap33 = max(_gap33, abs(area / 1e3 - row.irrigated_area_thousand_ha))
check(
    "simple allocation rules (proportional cut; Boro first to its floor): irrigated area recomputed (budgets 0.8, 0.9)",
    _gap33 < 1e-6,
    f"max gap {_gap33:.1e} thousand ha",
)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
