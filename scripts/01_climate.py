#!/usr/bin/env python3
"""Stage 1 - climate predictors and crop water requirements from NASA POWER.

Part 1 rebuilds the seasonal climate predictors of the yield panel from the monthly NASA POWER files
(inputs/nasa_power/POWER_<district>.csv: community AG, 2014-2025, parameters ALLSKY_SFC_SW_DWN,
PRECTOTCORR, RH2M, T2M, T2M_MAX, T2M_MIN, WS2M, at the district headquarters) and checks that they
agree with the columns of inputs/analysis_panel_full.csv.

Season of the crop harvested in year t: Boro Dec-May, wheat Nov-Mar, potato Nov-Feb.
  season_rain_mm        sum of monthly rain (mm/day x days in month)
  monsoon_prev_rain_mm  Jun-Oct of year t-1
  season_tmean_c        mean of monthly T2M
  season_tmax_c         mean of monthly T2M_MAX (highest daily maximum of each month)
  season_tmin_c         mean of monthly T2M_MIN (lowest daily minimum of each month)
  terminal_tmax_c       T2M_MAX of the last month of the season
  season_rh_pct         mean of monthly RH2M
  effective_rain_mm     USDA-SCS monthly formula (Smith 1992, FAO paper 46), summed over the season

Part 2 computes the FAO-56 crop water requirement of four crops from the daily NASA POWER files
(see the comment block before Part 2).

Outputs (generated/climate/):
  climate_predictors_from_raw.csv, climate_check.csv, nasa_power_locations.csv,
  crop_water_requirement_fao56.csv, crop_water_requirement_summary.csv,
  pm_seasonal_fao56_16districts_3crops_2015_2025.csv, nasa_power_daily_locations.csv
"""
import argparse, calendar, re
from pathlib import Path
import numpy as np, pandas as pd

cli = argparse.ArgumentParser()
cli.add_argument("--input-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
args = cli.parse_args()
I, O = args.input_dir, args.output_dir
O.mkdir(parents=True, exist_ok=True)
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SEASON = {"Boro Rice": (12, 5), "Wheat": (11, 3), "Potato": (11, 2)}
COLS = [
    "season_rain_mm",
    "monsoon_prev_rain_mm",
    "season_tmean_c",
    "season_tmax_c",
    "season_tmin_c",
    "terminal_tmax_c",
    "season_rh_pct",
]
# The raw files give rain in mm/day to two decimals (up to 0.16 mm per month once scaled) and temperatures
# to two decimals; the panel was built from unrounded values. The tolerances cover that rounding only.
TOL = {
    "season_rain_mm": 1.5,
    "monsoon_prev_rain_mm": 0.1,
    "season_tmean_c": 0.006,
    "season_tmax_c": 0.006,
    "season_tmin_c": 0.006,
    "terminal_tmax_c": 1e-9,
    "season_rh_pct": 0.006,
}


def load(f):
    lines = f.read_text().splitlines()
    k = lines.index("-END HEADER-")
    head = "\n".join(lines[:k])
    lat, lon = map(float, re.search(r"Latitude\s+([\d.]+)\s+Longitude\s+([\d.]+)", head).groups())
    elev = float(re.search(r"region = ([\d.\-]+) meters", head).group(1))
    raw = pd.read_csv(f, skiprows=k + 1)
    V = {(r.PARAMETER, int(r.YEAR), i + 1): float(r[m]) for _, r in raw.iterrows() for i, m in enumerate(MON)}
    assert min(V.values()) > -999, f
    return V, lat, lon, elev


def window(first, last, year):
    ms, y, m = [], (year - 1 if first > last else year), first
    while True:
        ms.append((y, m))
        if m == last and y == year:
            return ms
        m, y = (1, y + 1) if m == 12 else (m + 1, y)


usda = lambda p: p * (125 - 0.2 * p) / 125 if p <= 250 else 125 + 0.1 * p
panel = pd.read_csv(I / "analysis_panel_full.csv").set_index(["district", "crop", "year_end"])
cache = pd.read_csv(I / "nasa_pm_seasonal_cache_16districts_3crops_2015_2025.csv").set_index(
    ["district", "crop", "year_end"]
)
files = sorted((I / "nasa_power").glob("POWER_*.csv"))
assert len(files) == 16
rows, locs = [], []
for f in files:
    d = f.stem.split("_", 1)[1]
    V, lat, lon, elev = load(f)
    locs.append(dict(district=d, latitude=lat, longitude=lon, merra2_cell_elevation_m=elev, file=f.name))
    rain = lambda y, m: V[("PRECTOTCORR", y, m)] * calendar.monthrange(y, m)[1]
    for crop, (a, b) in SEASON.items():
        for yr in range(2015, 2026):
            w = window(a, b, yr)
            rows.append(
                dict(
                    district=d,
                    crop=crop,
                    year_end=yr,
                    season_rain_mm=sum(rain(*k) for k in w),
                    monsoon_prev_rain_mm=sum(rain(yr - 1, m) for m in range(6, 11)),
                    season_tmean_c=np.mean([V[("T2M", *k)] for k in w]),
                    season_tmax_c=np.mean([V[("T2M_MAX", *k)] for k in w]),
                    season_tmin_c=np.mean([V[("T2M_MIN", *k)] for k in w]),
                    terminal_tmax_c=V[("T2M_MAX", yr, b)],
                    season_rh_pct=np.mean([V[("RH2M", *k)] for k in w]),
                    effective_rain_mm=sum(usda(rain(*k)) for k in w),
                )
            )
R = pd.DataFrame(rows).set_index(["district", "crop", "year_end"]).sort_index()
assert len(R) == 528 and set(R.index) == set(panel.index)
check = []
for c in COLS:
    gap = (R[c] - panel.loc[R.index, c]).abs()
    check.append(
        dict(
            variable=c,
            compared_with="analysis_panel_full.csv",
            rows=len(gap),
            max_abs_gap=float(gap.max()),
            mean_abs_gap=float(gap.mean()),
            tolerance=TOL[c],
            within_tolerance=bool(gap.max() <= TOL[c]),
        )
    )
gap = (R.effective_rain_mm - cache.loc[R.index, "effective_rain_monthly_mm"]).abs()
check.append(
    dict(
        variable="effective_rain_mm",
        compared_with="nasa_pm_seasonal_cache",
        rows=len(gap),
        max_abs_gap=float(gap.max()),
        mean_abs_gap=float(gap.mean()),
        tolerance=0.5,
        within_tolerance=bool(gap.max() <= 0.5),
    )
)
check = pd.DataFrame(check)
assert check.within_tolerance.all(), check
R.reset_index().to_csv(O / "climate_predictors_from_raw.csv", index=False)
check.to_csv(O / "climate_check.csv", index=False)
pd.DataFrame(locs).to_csv(O / "nasa_power_locations.csv", index=False)
print(pd.DataFrame(locs).to_string(index=False))
print(check.to_string(index=False))
print("CLIMATE PREDICTORS REBUILT FROM RAW FILES: all 16 districts within rounding of the panel")

# ---------------------------------------------------------------------------
# Part 2: crop water requirement by FAO-56 from the daily NASA POWER files
# ---------------------------------------------------------------------------
# FAO-56 needs daily maxima and minima, so the daily files are used here (the monthly T2M_MAX and T2M_MIN
# are the month's extremes and would overstate ET0 by about 10%).
#
#   ET0     FAO-56 Penman-Monteith, daily (Allen et al. 1998, Eq. 6): Tmean = (Tmax+Tmin)/2; es from Tmax and
#           Tmin (Eq. 12); ea = RHmean/100 x es (Eq. 19); Rs = ALLSKY_SFC_SW_DWN; Ra, Rso (Eq. 21, 37); Rnl (Eq. 39);
#           G = 0; u2 = WS2M; pressure from the MERRA-2 cell elevation in the file header (Eq. 7).
#   Kc      Mainuddin et al. 2014 (CSIRO), Table 3.2, base-case planting date and Kc (initial, mid, end), which
#           were taken from FAO-56; stage lengths from FAO-56 Table 11 (rows named below). Kc curve as FAO-56
#           Fig. 25: constant over the initial stage, linear to mid, constant, linear to end.
#   Peff    USDA-SCS monthly formula on monthly rain, scaled by the share of the month inside the crop period.
#   Perc.   rice only: 3 mm/day (clay loam, Mainuddin et al. 2014 Table 3.1, their illustrative case) x 110 days,
#           using their result that each 1 mm/day of percolation adds about 110 mm to Boro irrigation (p.35).
#   NIR     sum(Kc x ET0) - Peff (not below 0) + percolation.
import datetime as _dt

CROP_CAL = (
    {  # planting (month, day), stage lengths (init, dev, mid, late), Kc (ini, mid, end), FAO-56 Table 11 row
        "Boro Rice": (
            (11, 22),
            (30, 30, 60, 30),
            (1.05, 1.20, 0.90),
            "Rice, Dec planting, tropics: 30/30/60/30",
        ),
        "Wheat": (
            (11, 23),
            (15, 25, 50, 30),
            (0.40, 1.15, 0.40),
            "Barley/oats/wheat, Nov, central India: 15/25/50/30",
        ),
        "Potato": (
            (10, 30),
            (30, 35, 50, 25),
            (0.60, 1.15, 0.80),
            "Potato, Dec, Calif. desert: 30/35/50/25 (140 days, the potato season of Mainuddin et al. 2014, Fig. 3.1)",
        ),
        "Maize": ((12, 1), (25, 40, 45, 30), (0.40, 1.20, 0.60), "Maize (grain), Dec/Jan, arid: 25/40/45/30"),
    }
)
PERC_MM = {"Boro Rice": 3.0 * 110, "Wheat": 0.0, "Potato": 0.0, "Maize": 0.0}


def load_daily(f):
    lines = f.read_text().splitlines()
    k = lines.index("-END HEADER-")
    head = "\n".join(lines[:k])
    lat, lon = map(float, re.search(r"latitude\s+([\d.]+)\s+longitude\s+([\d.]+)", head, re.I).groups())
    elev = float(re.search(r"region = ([\d.\-]+) meters", head).group(1))
    d = pd.read_csv(f, skiprows=k + 1)
    assert (d.drop(columns=["YEAR", "DOY"]) > -999).all().all(), f
    d["date"] = pd.to_datetime(d.YEAR.astype(str), format="%Y") + pd.to_timedelta(d.DOY - 1, unit="D")
    return d.set_index("date"), lat, lon, elev


def et0_fao56(d, lat, elev):
    tx, tn, rh, u2, rs = d.T2M_MAX, d.T2M_MIN, d.RH2M, d.WS2M, d.ALLSKY_SFC_SW_DWN
    t = (tx + tn) / 2
    e0 = lambda x: 0.6108 * np.exp(17.27 * x / (x + 237.3))
    es = (e0(tx) + e0(tn)) / 2
    ea = rh / 100 * es
    delta = 4098 * e0(t) / (t + 237.3) ** 2
    gamma = 0.000665 * 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    J = d.index.dayofyear.to_numpy()
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = (
        24
        * 60
        / np.pi
        * 0.0820
        * dr
        * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    )
    rso = (0.75 + 2e-5 * elev) * ra
    rnl = (
        4.903e-9
        * ((tx + 273.16) ** 4 + (tn + 273.16) ** 4)
        / 2
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * np.minimum(rs / rso, 1) - 0.35)
    )
    rn = 0.77 * rs - rnl
    return (0.408 * delta * rn + gamma * 900 / (t + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))


def kc_curve(n_days, stages, kc):
    i, dv, m, l = stages
    ki, km, ke = kc
    out = []
    for k in range(n_days):
        if k < i:
            out.append(ki)
        elif k < i + dv:
            out.append(ki + (km - ki) * (k - i + 1) / dv)
        elif k < i + dv + m:
            out.append(km)
        else:
            out.append(km + (ke - km) * (k - i - dv - m + 1) / l)
    return np.array(out)


wr_rows, dloc = [], []
dfiles = sorted((I / "nasa_power_daily").glob("DAILY_*.csv"))
assert len(dfiles) == 16
mon_loc = pd.DataFrame(locs).set_index("district")
for f in dfiles:
    dname = f.stem.split("_", 1)[1]
    d, lat, lon, elev = load_daily(f)
    assert (
        abs(lat - mon_loc.loc[dname, "latitude"]) < 1e-9 and abs(lon - mon_loc.loc[dname, "longitude"]) < 1e-9
    )
    dloc.append(dict(district=dname, latitude=lat, longitude=lon, elevation_m=elev, days=len(d)))
    d["ET0"] = et0_fao56(d, lat, elev)
    mrain = d.PRECTOTCORR.groupby([d.index.year, d.index.month]).sum()  # monthly rain, mm
    for crop, ((pm, pdd), stages, kc, _) in CROP_CAL.items():
        for yr in range(2015, 2026):
            n = sum(stages)
            start = pd.Timestamp(_dt.date(yr - 1 if pm >= 8 else yr, pm, pdd))
            days = pd.date_range(start, periods=n, freq="D")
            assert days[-1].year == yr, (crop, yr)
            et0 = d.loc[days, "ET0"].to_numpy()
            etc = kc_curve(n, stages, kc) * et0
            peff = 0.0
            for (y, m), g in pd.Series(1, index=days).groupby([days.year, days.month]):
                peff += usda(mrain[(y, m)]) * len(g) / calendar.monthrange(y, m)[1]
            wr_rows.append(
                dict(
                    district=dname,
                    crop=crop,
                    year_end=yr,
                    planting=str(start.date()),
                    season_days=n,
                    ET0_mm=float(et0.sum()),
                    ETc_mm=float(etc.sum()),
                    rain_mm=float(d.loc[days, "PRECTOTCORR"].sum()),
                    effective_rain_mm=peff,
                    percolation_mm=PERC_MM[crop],
                    nir_fao56_mm=max(0.0, float(etc.sum()) - peff) + PERC_MM[crop],
                )
            )
WR = pd.DataFrame(wr_rows)
assert len(WR) == 704 and WR.nir_fao56_mm.gt(0).all()
WR.to_csv(O / "crop_water_requirement_fao56.csv", index=False)
# The same requirement for the three panel crops, in the column names read by stages 2 and 6
cc = WR[WR.crop != "Maize"].rename(
    columns={
        "ETc_mm": "PET_nasa_pm_mm",
        "effective_rain_mm": "effective_rain_monthly_mm",
        "nir_fao56_mm": "climate_deficit_pm_mm",
    }
)
cc[
    ["district", "crop", "year_end", "PET_nasa_pm_mm", "effective_rain_monthly_mm", "climate_deficit_pm_mm"]
].to_csv(O / "pm_seasonal_fao56_16districts_3crops_2015_2025.csv", index=False)
# Comparison with the published NIR (Mainuddin et al. 2014) and with the earlier monthly-file estimate
# (input file nasa_pm_seasonal_cache_..., column retired_cache_PET_mm)
pub = {
    "Boro Rice": (644, 676, 779),
    "Wheat": (227, 283, 339),
    "Potato": (309, 376.5, 444),
    "Maize": (279, 354, 429),
}
comp = WR.groupby("crop").agg(
    ET0_mm=("ET0_mm", "mean"),
    ETc_mm=("ETc_mm", "mean"),
    effective_rain_mm=("effective_rain_mm", "mean"),
    nir_fao56_mean_mm=("nir_fao56_mm", "mean"),
    nir_fao56_min_mm=("nir_fao56_mm", "min"),
    nir_fao56_max_mm=("nir_fao56_mm", "max"),
    nir_fao56_cv=("nir_fao56_mm", lambda v: v.std() / v.mean()),
)
comp["published_nir_low_mm"] = [pub[c][0] for c in comp.index]
comp["published_nir_nominal_mm"] = [pub[c][1] for c in comp.index]
comp["published_nir_high_mm"] = [pub[c][2] for c in comp.index]
old = cache.reset_index().groupby("crop").PET_nasa_pm_mm.mean()
comp["retired_cache_PET_mm"] = [old.get(c, np.nan) for c in comp.index]
comp.reset_index().to_csv(O / "crop_water_requirement_summary.csv", index=False)
pd.DataFrame(dloc).to_csv(O / "nasa_power_daily_locations.csv", index=False)
print(comp.round(1).to_string())
print("CROP WATER REQUIREMENT (FAO-56, daily) WRITTEN")
