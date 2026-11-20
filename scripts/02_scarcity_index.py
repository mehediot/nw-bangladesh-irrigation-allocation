#!/usr/bin/env python3
"""Stage 2 - relative agricultural water scarcity index (RAWSI).

RAWSI is the mean of three components, each scaled 0-1 across the 16 districts:
  severity  groundwater-trend severity: mean of the share of Type 1 wells and the share of wells whose
            minimum (wet-season) depth is falling (Hodgson et al. 2021)
  demand    area-weighted crop water demand of Boro, wheat and potato with the published NIR (mm)
  deficit   area-weighted FAO-56 crop water requirement from stage 1 (mm)
Demand and deficit are scaled over all district-years and averaged over the years; the index is relative
(it ranks districts) and is not used by the allocation model.

A second, small check compares the groundwater-trend input with the BADC dry-season depletion data of
the five districts that appear in both (Spearman rank correlation with an exact permutation p-value).

Outputs (generated/scarcity_index/): rawsi.csv, badc_depletion_check.csv
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def min_max(v):
    """Scale a series to 0-1."""
    span = v.max() - v.min()
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Cannot scale a constant series")
    return (v - v.min()) / span


def scarcity_index(input_dir: Path, output_dir: Path, climate_file: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(input_dir / "analysis_panel_full.csv")
    requirement = pd.read_csv(climate_file)  # FAO-56 requirement written by stage 1
    gw = pd.read_csv(input_dir / "hodgson_2021_groundwater_trends.csv")
    badc = pd.read_csv(input_dir / "gw_depletion_badc_mis1819.csv")

    keys = ["district", "crop", "year_end"]
    assert panel.shape[0] == requirement.shape[0] == 528
    assert not panel.duplicated(keys).any() and not requirement.duplicated(keys).any()
    assert len(gw) == 16 and not gw.district.duplicated().any()
    assert panel[keys + ["area_ha", "nir_mm"]].notna().all().all()
    assert requirement[keys + ["climate_deficit_pm_mm"]].notna().all().all()

    joined = panel.merge(requirement, on=keys, validate="one_to_one", indicator=True)
    assert joined._merge.eq("both").all()

    # Component 1: groundwater-trend severity (one value per district)
    gw = gw.assign(severity_raw=(gw.frac_type1 + gw.frac_falling_mindepth) / 2)
    gw["severity"] = min_max(gw.severity_raw)
    joined = joined.merge(
        gw[["district", "severity", "frac_falling_mindepth"]], on="district", validate="many_to_one"
    )
    assert len(joined) == 528

    # Components 2 and 3: area-weighted water demand and requirement of the three crops, per district-year
    joined["wd"] = joined.area_ha * joined.nir_mm
    joined["cd"] = joined.area_ha * joined.climate_deficit_pm_mm
    agg = joined.groupby(["district", "year_end"], as_index=False).agg(
        area=("area_ha", "sum"), wd=("wd", "sum"), cd=("cd", "sum"), severity=("severity", "first")
    )
    assert len(agg) == 176 and (agg.area > 0).all()
    agg["demand_mm"] = agg.wd / agg.area
    agg["deficit_mm"] = agg.cd / agg.area
    agg["demand"] = min_max(agg.demand_mm)
    agg["deficit"] = min_max(agg.deficit_mm)
    agg["RAWSI_annual"] = (agg.severity + agg.demand + agg.deficit) / 3

    # Index = mean over the years; rank 1 = relatively most stressed
    ranks = agg.groupby("district", as_index=False).agg(
        severity=("severity", "first"),
        demand=("demand", "mean"),
        deficit=("deficit", "mean"),
        RAWSI=("RAWSI_annual", "mean"),
    )
    ranks = ranks.sort_values("RAWSI", ascending=False).reset_index(drop=True)
    ranks.insert(0, "rank", np.arange(1, 17))
    ranks.to_csv(output_dir / "rawsi.csv", index=False)

    # Check of the groundwater-trend input against BADC depletion (five districts in both sources)
    depletion = badc.groupby("district").gw_max_depletion_m_janmar.mean().reset_index()
    both = depletion.merge(gw[["district", "frac_falling_mindepth"]], on="district", validate="one_to_one")
    assert len(both) == 5
    rho = float(spearmanr(both.gw_max_depletion_m_janmar, both.frac_falling_mindepth).statistic)
    values = both.frac_falling_mindepth.values
    p_exact = float(
        np.mean(
            [
                abs(spearmanr(both.gw_max_depletion_m_janmar, perm).statistic) >= abs(rho) - 1e-12
                for perm in itertools.permutations(values)
            ]
        )
    )
    assert np.isfinite(rho) and 0 <= p_exact <= 1
    pd.DataFrame([dict(n=len(both), spearman_rho=rho, exact_two_sided_p=p_exact)]).to_csv(
        output_dir / "badc_depletion_check.csv", index=False
    )
    print("RAWSI, top five districts:", ranks.district.head(5).tolist(), flush=True)
    print(f"BADC check: n = {len(both)}, Spearman rho = {rho:.3f}, exact p = {p_exact:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--climate-file", type=Path, required=True)
    args = parser.parse_args()
    scarcity_index(args.input_dir, args.output_dir, args.climate_file)
