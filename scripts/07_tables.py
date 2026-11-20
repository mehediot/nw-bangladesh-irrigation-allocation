#!/usr/bin/env python3
"""Stage 7 - tables of the paper, assembled from the outputs of stages 1-6.

No modelling happens here. Each table is a selection or a simple count taken from files the earlier
stages wrote, so every number in the paper traces to one file. Files are named by their number in the
paper (Table 1, the model coefficients and their sources, is text): Table2_... to Table4_... for the main
text, Fig5_robust_budgets (the values plotted in Fig. 5) and TableS2_... to TableS20_... for the
Supporting Information (Table S1, the sources of prices, costs and charges, comes from the input files). extra_three_vs_four_crops.csv is not used in the paper.

Main-text evaluation design: decision-time forecasts (trained on outcomes up to year t-2), tested on
2023-2025. Allocation tables use the four-crop model; the yield-model tables stay on three crops (plus
maize for the decision-time forecasts). Column names with `gross_margin` mean the paper's net return.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

cli = argparse.ArgumentParser(description="Stage 7 - tables")
cli.add_argument("--generated-dir", type=Path, required=True)
cli.add_argument("--output-dir", type=Path, required=True)
cli.add_argument(
    "--only", nargs="+", metavar="TABLE", help="tables to build, e.g. Table2 TableS5 (default: all)"
)
args = cli.parse_args()
G, O = args.generated_dir, args.output_dir
O.mkdir(parents=True, exist_ok=True)
CROPS = ["Boro Rice", "Wheat", "Potato"]
tables = {}


def table2():
    """Table 2: forecast skill, decision-time design (16 districts x 3 test years per crop)."""
    # --- Table 2: forecast skill, decision-time design (16 districts x 3 test years per crop) ---
    fc = pd.read_csv(G / "forecasts" / "decision_time_forecasts.csv")
    acc = pd.read_csv(G / "forecasts" / "decision_time_forecast_accuracy.csv")
    MODELS = {
        "Elastic Net": "EN_cutoff2",
        "Persistence": "Persistence_cutoff2",
        "Climatology": "Climatology_cutoff2",
        "District trend": "Trend_cutoff2",
    }
    rows = []
    for crop in CROPS + ["Maize"]:  # maize: decision-time forecasts only (no hindcast model)
        g = fc[fc.crop == crop]
        assert len(g) == 48
        err = {name: g.yield_t_ha - g[col] for name, col in MODELS.items()}
        rmse = {name: float(np.sqrt(np.mean(e**2))) for name, e in err.items()}
        for name, col in MODELS.items():
            stored = acc[(acc.crop == crop) & (acc.forecast == col)].RMSE_t_ha.iloc[0]
            assert abs(rmse[name] - stored) < 1e-12
            rows.append(
                dict(
                    crop=crop,
                    model=name,
                    n_forecasts=len(g),
                    RMSE_t_ha=rmse[name],
                    MAE_t_ha=float(err[name].abs().mean()),
                    skill_vs_climatology=1 - rmse[name] / rmse["Climatology"],
                    R2=float(1 - (err[name] ** 2).sum() / ((g.yield_t_ha - g.yield_t_ha.mean()) ** 2).sum()),
                )
            )
    tables["Table2_forecast_accuracy"] = pd.DataFrame(rows)


def table3():
    """Table 3: food-energy and net-return plans, free and least fallow, decision-time Elastic Net yields."""
    # --- Table 3: food-energy and net-return plans, free and least fallow, decision-time Elastic Net yields ---
    gm = pd.read_csv(G / "extensions" / "ext_12_gross_margin_objective.csv")
    t5 = gm[gm.objective.isin(["food_energy", "gross_margin_potato_mid"])].copy()
    t5["objective"] = t5.objective.replace({"gross_margin_potato_mid": "gross_margin"})
    COLS5 = [
        "budget_ratio",
        "plan",
        "objective",
        "boro_Mt",
        "wheat_Mt",
        "potato_Mt",
        "maize_Mt",
        "food_energy_trillion_kcal",
        "gross_margin_billion_Tk_potato_mid",
        "boro_pct_of_land",
        "wheat_pct_of_land",
        "potato_pct_of_land",
        "maize_pct_of_land",
        "fallow_pct_of_land",
        "fallow_thousand_ha",
    ]
    tables["Table3_plans"] = (
        t5[COLS5]
        .rename(columns={"gross_margin_billion_Tk_potato_mid": "gross_margin_billion_Tk"})
        .reset_index(drop=True)
    )
    obj = pd.read_csv(
        G / "extensions" / "ext_11_objective_by_yield_source.csv"
    )  # same plans, solved in section 8
    chk = obj[obj.yield_source == "decision_time_elastic_net"].merge(
        t5, on=["budget_ratio", "plan", "objective"]
    )
    assert (
        len(chk) == 8
        and np.abs(chk.food_energy_trillion_kcal_x - chk.food_energy_trillion_kcal_y).max() < 1e-9
    )


def table4():
    """Table 4: decision value of the Elastic Net forecast (nominal NIR, both fallow plans, 48 cases)."""
    # --- Table 4: decision value of the Elastic Net forecast (nominal NIR, both fallow plans, 48 cases) ---
    dt = pd.read_csv(G / "decision" / "decision_value_tests.csv")
    t7 = dt[dt.uncertainty_set == "nominal"].copy()
    assert len(t7) == 3 * 2 * 2 * 3 and (t7.n_cases == 48).all() and (t7.n_districts == 16).all()
    t7["EN_vs"] = t7.EN_vs.str.replace("_cutoff2", "").str.replace("Trend", "District trend")
    COLS7 = [
        "budget_ratio",
        "objective",
        "plan",
        "EN_vs",
        "n_cases",
        "loss_unit",
        "mean_regret_EN",
        "mean_regret_benchmark",
        "mean_difference",
        "bootstrap95_low",
        "bootstrap95_high",
        "dm_t",
        "dm_p_two_sided",
        "cases_EN_better",
        "cases_benchmark_better",
        "realised_total_difference_pct",
    ]
    tables["Table4_decision_value"] = t7[COLS7].reset_index(drop=True)


def fig5_values():
    """Values plotted in Fig. 5: Gamma-robust thresholds (48 district-year cases, four crops, least fallow)."""
    # --- Fig. 5 values: Gamma-robust thresholds (48 district-year cases, four crops, land balance, least fallow) ---
    rt = pd.read_csv(G / "decision" / "robust_thresholds.csv")
    LABEL = {
        "nominal": "Nominal NIR",
        "temporal_cv": "Temporal NIR deviation (year-to-year CV)",
        "spatial_upper": "Spatial upper NIR (range across districts)",
    }
    rt.insert(
        0,
        "nir_uncertainty",
        [
            LABEL[u] + ("" if g == 0 else f", Gamma {g}") + (" (Gamma 0)" if g == 0 else "")
            for u, g in zip(rt.uncertainty_set, rt.gamma)
        ],
    )
    assert len(rt) == 9
    tables["Fig5_robust_budgets"] = rt


def table_s2():
    """Table S2: Elastic Net hyperparameters chosen inside the training years (decision-time design)."""
    # --- Table S2: Elastic Net hyperparameters chosen inside the training years (decision-time design) ---
    tables["TableS2_decision_time_tuning"] = pd.read_csv(G / "forecasts" / "decision_time_tuning.csv")


def table_s3():
    """Table S3: hindcast models (same-season climate), kept for explanation only."""
    # --- Table S3: hindcast models (same-season climate), kept for explanation only ---
    hm = pd.read_csv(G / "hindcast" / "07_hindcast_metrics.csv")
    sp = pd.read_csv(G / "extensions" / "ext_03_spatial_validation.csv")
    fe = pd.read_csv(G / "extensions" / "ext_02_fixed_effects.csv").groupby("crop").within_R2.first()
    rows = []
    for crop in CROPS:
        row = dict(crop=crop)
        for model, tag in [("climatology", "climatology"), ("ElasticNet", "elastic_net"), ("GBT", "gbt")]:
            m = hm[(hm.crop == crop) & (hm.model == model)].iloc[0]
            row[f"{tag}_RMSE_t_ha"] = m.RMSE_t_ha
            if model != "climatology":
                row[f"{tag}_skill_vs_climatology"] = m.skill_vs_climatology
                row[f"{tag}_R2"] = m.R2
        for scheme, tag in [
            ("leave_one_district_out", "gbt_R2_leave_one_district_out"),
            ("leave_one_climate_cell_out", "gbt_R2_leave_one_climate_cell_out"),
        ]:
            row[tag] = sp[(sp.scheme == scheme) & (sp.crop == crop)].GBT_spatial_R2.iloc[0]
        row["fixed_effects_within_R2"] = fe[crop]
        rows.append(row)
    tables["TableS3_hindcast_models"] = pd.DataFrame(rows)


def table_s4():
    """Table S4: RAWSI with rank stability and Boro share of water demand."""
    # --- Table S4: RAWSI with rank stability and Boro share of water demand ---
    rawsi = pd.read_csv(G / "scarcity_index" / "rawsi.csv")
    stab = pd.read_csv(G / "extensions" / "ext_09_rawsi_rank_stability.csv")
    boro = pd.read_csv(G / "extensions" / "ext_10_district_boro_share.csv")
    t3 = rawsi.merge(
        stab[["district", "best_rank", "worst_rank", "share_of_variants_in_top5"]],
        on="district",
        validate="one_to_one",
    )
    t3 = t3.merge(boro[["district", "boro_water_share"]], on="district", validate="one_to_one")
    assert len(t3) == 16
    tables["TableS4_rawsi"] = t3.rename(
        columns={
            "severity": "groundwater_component",
            "demand": "demand_component",
            "deficit": "deficit_component",
            "best_rank": "best_rank_20_variants",
            "worst_rank": "worst_rank_20_variants",
        }
    )


def table_s5():
    """Table S5: the decision-time Elastic Net."""
    # --- Table S5: the decision-time Elastic Net ---
    ec = pd.read_csv(G / "forecasts" / "en_coefficients.csv")
    ct = pd.read_csv(G / "forecasts" / "en_contributions.csv")
    grp = ["trend", "lag2_yield", "lag2_4_mean_yield", "district_effect"]
    share = ct.groupby(["crop", "year_end"])[grp].apply(lambda x: x.abs().mean())
    share = (
        (100 * share.div(share.sum(axis=1), axis=0))
        .add_prefix("share_of_mean_abs_contribution_pct_")
        .reset_index()
        .rename(columns={"year_end": "target_year"})
    )
    tables["TableS5_elastic_net"] = ec.merge(share, on=["crop", "target_year"], validate="one_to_one")
    assert len(tables["TableS5_elastic_net"]) == 12


def table_s6():
    """Table S6a: districts (of 16) where the Boro floor cannot be met."""
    # --- Table S6a: districts (of 16) where the Boro floor cannot be met ---
    share = pd.read_csv(G / "extensions" / "ext_06_pm_deficit_district.csv")
    rows = []
    for label, col in [
        ("Fixed NIR", "boro_share_fixed_NIR"),
        ("PM climate deficit, 2023-2025 mean", "boro_share_PM_2023_2025"),
    ]:
        for floor in (1.0, 0.9, 0.8):
            row = dict(water_coefficients=label, boro_floor=floor)
            for rho in (1.0, 0.9, 0.8, 0.7):
                row[f"infeasible_at_budget_{rho:.2f}"] = int((floor * share[col] > rho + 1e-12).sum())
            rows.append(row)
    t4 = pd.DataFrame(rows)
    strict = pd.read_csv(G / "hindcast" / "11_boro_feasibility.csv")
    for rho in (
        1.0,
        0.9,
        0.8,
        0.7,
    ):  # the three-crop shares must agree with the stage-2 (three-crop) screening
        n = int((~strict[np.isclose(strict.rho, rho)].strict_Boro_100pct_feasible).sum())
        assert n == int((share.boro_share_fixed_NIR_three_crops > rho + 1e-12).sum())
    tables["TableS6a_feasibility_grid"] = t4

    # --- Table S6b: smallest fallow share of the land pool needed to meet each budget (region, four crops) ---
    mf = pd.read_csv(G / "extensions" / "ext_04_minimum_fallow.csv")
    mf = mf[(mf.scope == "four_crops") & (mf.caps == "observed_max_2015_2025") & (mf.district == "REGION")]
    rows = []
    for floor in (1.0, 0.9, 0.8):
        row = dict(boro_floor=floor)
        for rho in (1.0, 0.9, 0.8, 0.7):
            r = mf[np.isclose(mf.boro_floor, floor) & np.isclose(mf.budget_ratio, rho)].iloc[0]
            row[f"min_fallow_pct_at_budget_{rho:.2f}"] = r.min_fallow_pct
            row[f"districts_infeasible_at_budget_{rho:.2f}"] = int(r.districts_infeasible)
        rows.append(row)
    t4b = pd.DataFrame(rows)
    for rho in (1.0, 0.9, 0.8, 0.7):  # infeasible counts must equal the rule floor x share > rho
        for floor in (1.0, 0.9, 0.8):
            assert (
                t4b[t4b.boro_floor == floor][f"districts_infeasible_at_budget_{rho:.2f}"].iloc[0]
                == t4[(t4.water_coefficients == "Fixed NIR") & (t4.boro_floor == floor)][
                    f"infeasible_at_budget_{rho:.2f}"
                ].iloc[0]
            )
    tables["TableS6b_minimum_fallow"] = t4b


def table_s7():
    """Table S7: Boro floor x crop-area limits, crossed."""
    # --- Table S7: Boro floor x crop-area limits, crossed ---
    fl = pd.read_csv(G / "extensions" / "ext_30_floor_limit_grid.csv")
    tables["TableS7_floor_limit_grid"] = fl[
        [
            "crop_limits",
            "boro_floor",
            "budget_ratio",
            "districts_infeasible",
            "min_fallow_pct",
            "least_fallow_food_energy_food_energy_pct_of_baseline",
            "least_fallow_food_energy_boro_pct_of_land",
            "free_food_energy_food_energy_pct_of_baseline",
            "free_gross_margin_gross_margin_pct_of_baseline",
            "free_gross_margin_fallow_pct_of_land",
            "least_fallow_districts_boro_at_floor",
            "least_fallow_districts_wheat_at_limit",
            "least_fallow_districts_potato_at_limit",
            "least_fallow_districts_maize_at_limit",
        ]
    ].rename(
        columns={
            "least_fallow_food_energy_food_energy_pct_of_baseline": "least_fallow_plan_food_energy_pct_of_baseline",
            "least_fallow_food_energy_boro_pct_of_land": "least_fallow_plan_boro_pct_of_land",
            "free_food_energy_food_energy_pct_of_baseline": "food_energy_plan_free_fallow_food_energy_pct_of_baseline",
            "free_gross_margin_gross_margin_pct_of_baseline": "gross_margin_plan_free_fallow_gross_margin_pct_of_baseline",
            "free_gross_margin_fallow_pct_of_land": "gross_margin_plan_free_fallow_fallow_pct_of_land",
        }
    )
    assert len(tables["TableS7_floor_limit_grid"]) == 56


def table_s8():
    """Table S8: net-return plans under three potato prices."""
    gm = pd.read_csv(G / "extensions" / "ext_12_gross_margin_objective.csv")
    # --- Table S8: net-return plans under three potato prices ---
    s3 = gm[gm.objective.str.startswith("gross_margin")].copy()
    s3["potato_price_scenario"] = s3.objective.str.replace("gross_margin_potato_", "")
    tables["TableS8_potato_price_plans"] = s3[
        [
            "budget_ratio",
            "plan",
            "potato_price_scenario",
            "boro_pct_of_land",
            "wheat_pct_of_land",
            "potato_pct_of_land",
            "maize_pct_of_land",
            "fallow_pct_of_land",
            "food_energy_trillion_kcal",
            "gross_margin_billion_Tk_potato_low",
            "gross_margin_billion_Tk_potato_mid",
            "gross_margin_billion_Tk_potato_high",
        ]
    ].reset_index(drop=True)


def table_s9():
    """Table S9: dual values (shadow prices) across the 16 districts."""
    # --- Table S9: dual values (shadow prices) across the 16 districts ---
    sh = pd.read_csv(G / "extensions" / "ext_15_shadow_prices.csv")
    rows = []
    for (rho, plan, objective, unit), g in sh.groupby(
        ["budget_ratio", "plan", "objective", "unit"], sort=False
    ):
        assert len(g) == 16
        rows.append(
            dict(
                budget_ratio=rho,
                plan=plan,
                objective=objective,
                unit=unit,
                water_value_per_1000_m3_median=g.water_value_per_1000_m3.median(),
                water_value_per_1000_m3_min=g.water_value_per_1000_m3.min(),
                water_value_per_1000_m3_max=g.water_value_per_1000_m3.max(),
                boro_floor_cost_per_ha_median=g.boro_floor_cost_per_ha.median(),
                fallow_limit_value_per_ha_median=g.fallow_limit_value_per_ha.median(),
                land_value_per_ha_median=g.land_value_per_ha.median(),
                fallow_pct_median=g.fallow_pct.median(),
                districts_water_binding=int(g.water_binding.sum()),
                districts_boro_at_floor=int(g.boro_at_floor.sum()),
                districts_wheat_at_limit=int(g.wheat_at_cap.sum()),
                districts_potato_at_limit=int(g.potato_at_cap.sum()),
                districts_maize_at_limit=int(g.maize_at_cap.sum()),
            )
        )
    tables["TableS9_dual_values"] = pd.DataFrame(rows)


def table_s10():
    """Table S10: the same comparison for every NIR set, Gamma and fallow plan."""
    dt = pd.read_csv(G / "decision" / "decision_value_tests.csv")
    # --- Table S10: the same comparison for every NIR set, Gamma and fallow plan ---
    tables["TableS10_decision_value_all"] = dt.reset_index(drop=True)


def table_s11():
    """Table S11: break-even forecast error (nominal NIR, least fallow)."""
    # --- Table S11: break-even forecast error (nominal NIR, least fallow) ---
    tables["TableS11_break_even"] = pd.read_csv(G / "decision" / "break_even_summary.csv")


def table_s12():
    """Table S12: AWD on part of the Boro area (budget 0.80, least fallow)."""
    # --- Table S12: AWD on part of the Boro area (budget 0.80, least fallow) ---
    aw = pd.read_csv(G / "extensions" / "ext_25_awd_scenarios.csv")
    tables["TableS12_awd_scenarios"] = aw[np.isclose(aw.budget_ratio, 0.8) & (aw.plan == "least")][
        [
            "water_saving",
            "adoption_share",
            "yield_effect",
            "boro_yield_change_pct",
            "objective",
            "min_fallow_pct",
            "food_energy_trillion_kcal",
            "food_energy_trillion_kcal_change_vs_conventional",
            "gross_margin_billion_Tk",
            "gross_margin_billion_Tk_change_vs_conventional",
            "boro_pct_of_land",
            "wheat_pct_of_land",
            "potato_pct_of_land",
            "maize_pct_of_land",
        ]
    ].reset_index(drop=True)
    assert len(tables["TableS12_awd_scenarios"]) == 34


def table_s13():
    """Table S13: AWD water saving x Boro yield effect, budget 0.80."""
    # --- Table S13: AWD water saving x Boro yield effect, budget 0.80 ---
    tables["TableS13_awd_yield_grid"] = pd.read_csv(G / "extensions" / "ext_31_awd_yield_grid.csv")
    assert len(tables["TableS13_awd_yield_grid"]) == 32


def table_s14():
    """Table S14: model value of water next to irrigation charges."""
    # --- Table S14: model value of water next to irrigation charges ---
    tables["TableS14_water_value_vs_charges"] = pd.read_csv(
        G / "extensions" / "ext_27_water_value_vs_charges.csv"
    )


def table_s15():
    """Table S15: six-crop variant with lentil and mustard (budgets 0.80 and 0.90, least fallow)."""
    # --- Table S15: six-crop variant with lentil and mustard (budgets 0.80 and 0.90, least fallow) ---
    lw = pd.read_csv(G / "extensions" / "ext_29_low_water_crops.csv")
    tables["TableS15_low_water_crops"] = lw[(lw.budget_ratio < 0.95) & (lw.plan == "least")][
        [
            "budget_ratio",
            "awd",
            "low_water_crop_limit_x_baseline",
            "objective",
            "min_fallow_thousand_ha",
            "min_fallow_pct_of_pool",
            "four_crop_model_min_fallow_thousand_ha",
            "lentil_x_baseline",
            "mustard_x_baseline",
            "boro_thousand_ha",
            "wheat_thousand_ha",
            "potato_thousand_ha",
            "maize_thousand_ha",
            "lentil_thousand_ha",
            "mustard_thousand_ha",
            "food_energy_pct_of_baseline",
            "gross_margin_pct_of_baseline",
        ]
    ].reset_index(drop=True)
    assert len(tables["TableS15_low_water_crops"]) == 48


def table_s16():
    """Table S16: one-factor-at-a-time sensitivities at budget 0.80."""
    # --- Table S16: one-factor-at-a-time sensitivities at budget 0.80 ---
    tables["TableS16_sensitivity"] = pd.read_csv(G / "extensions" / "ext_28_sensitivity_tornado.csv")


def table_s17():
    """Table S17: Boro and wheat at the procurement price."""
    # --- Table S17: Boro and wheat at the procurement price ---
    tables["TableS17_procurement_price"] = pd.read_csv(
        G / "extensions" / "ext_24_procurement_price_sensitivity.csv"
    )[
        [
            "budget_ratio",
            "plan",
            "boro_wheat_price",
            "boro_price_Tk_per_kg_paddy",
            "wheat_price_Tk_per_kg",
            "boro_margin_Tk_per_ha",
            "wheat_margin_Tk_per_ha",
            "gross_margin_billion_Tk",
            "boro_pct_of_land",
            "wheat_pct_of_land",
            "potato_pct_of_land",
            "maize_pct_of_land",
            "fallow_pct_of_land",
        ]
    ]


def table_s18():
    """Table S18: maize price and cost."""
    # --- Table S18: maize price and cost ---
    mz = pd.read_csv(G / "extensions" / "ext_22_maize_price_cost_sensitivity.csv")
    tables["TableS18_maize_price_cost"] = mz[
        [
            "budget_ratio",
            "plan",
            "maize_price_scenario",
            "maize_price_Tk_per_kg",
            "maize_cost_scenario",
            "maize_cost_Tk_per_kg",
            "maize_margin_Tk_per_ha",
            "maize_area_thousand_ha",
            "gross_margin_billion_Tk",
            "food_energy_trillion_kcal",
            "fallow_pct_of_land",
        ]
    ].reset_index(drop=True)


def table_s19_s20():
    """Tables S19 and S20: land-accounting check; simple rules against the optimised plans."""
    # --- Tables S19 and S20: land-accounting check; simple rules against the optimised plans ---
    tables["TableS19_land_accounting_check"] = pd.read_csv(
        G / "extensions" / "ext_32_land_accounting_check.csv"
    )
    assert len(tables["TableS19_land_accounting_check"]) == 2
    tables["TableS20_rule_baselines"] = (
        pd.read_csv(G / "extensions" / "ext_33_rule_baselines.csv")
        .rename(columns={"gross_margin_pct_of_baseline": "net_return_pct_of_baseline"})
        .assign(plan=lambda t: t.plan.str.replace("gross_margin", "net_return"))
    )
    assert len(tables["TableS20_rule_baselines"]) == 10


def extra():
    """Extra table (not in the paper): the model with and without maize."""
    # --- Extra table (not in the paper): the model with and without maize ---
    sc = pd.read_csv(G / "extensions" / "ext_23_three_vs_four_crops.csv")
    tables["extra_three_vs_four_crops"] = sc[
        [
            "scope",
            "budget_ratio",
            "plan",
            "objective",
            "land_pool_thousand_ha",
            "food_energy_trillion_kcal",
            "gross_margin_billion_Tk",
            "food_energy_pct_of_baseline",
            "gross_margin_pct_of_baseline",
            "boro_pct_of_land",
            "wheat_pct_of_land",
            "potato_pct_of_land",
            "maize_pct_of_land",
            "fallow_pct_of_land",
        ]
    ].reset_index(drop=True)
    # three-crop rows reproduce the plans of the model before maize: same plan, nothing in maize
    assert (sc[sc.scope == "three_crops"].maize_Mt == 0).all()


TABLES = {
    "Table2": table2,
    "Table3": table3,
    "Table4": table4,
    "Fig5": fig5_values,  # the values plotted in Fig. 5 (Table 5 of earlier drafts)
    "TableS2": table_s2,
    "TableS3": table_s3,
    "TableS4": table_s4,
    "TableS5": table_s5,
    "TableS6": table_s6,  # S6a and S6b together
    "TableS6a": table_s6,
    "TableS6b": table_s6,
    "TableS7": table_s7,
    "TableS8": table_s8,
    "TableS9": table_s9,
    "TableS10": table_s10,
    "TableS11": table_s11,
    "TableS12": table_s12,
    "TableS13": table_s13,
    "TableS14": table_s14,
    "TableS15": table_s15,
    "TableS16": table_s16,
    "TableS17": table_s17,
    "TableS18": table_s18,
    "TableS19": table_s19_s20,
    "TableS20": table_s19_s20,
    "extra": extra,
}


def markdown(df):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return (
        head
        + "\n"
        + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))
    )


def paper_order(name):
    """Main tables first, then Tables S2-S20, then extra tables."""
    m = re.match(r"Table(S?)(\d+)([ab]?)_", name)
    return (2, 0, "") if m is None else (int(bool(m.group(1))), int(m.group(2)), m.group(3))


if __name__ == "__main__":
    wanted = args.only or list(TABLES)
    unknown = set(wanted) - set(TABLES)
    if unknown:
        raise SystemExit(f"Unknown table(s): {sorted(unknown)}; choose from {list(TABLES)}")
    for fn in dict.fromkeys(TABLES[k] for k in wanted):  # each function once, in registry order
        fn()
    for name, df in tables.items():
        df.to_csv(O / f"{name}.csv", index=False)
        print("Table written:", name, flush=True)
    if not args.only:  # one readable file with every table, in paper order
        parts = [
            "# Tables of the paper built by the pipeline (values rounded to 3 decimals here; full precision in the CSVs)"
        ]
        for name, df in sorted(tables.items(), key=lambda kv: paper_order(kv[0])):
            parts.append(f"## {name}\n\n{markdown(df)}")
        (O / "TABLES.md").write_text("\n\n".join(parts) + "\n")
