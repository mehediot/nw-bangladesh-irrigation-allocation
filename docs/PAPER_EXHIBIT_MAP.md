# Exhibits of the paper and the files that hold them

Tables and figures are named by their number in the paper. After a run, tables are in `generated/tables/` (and in `reference/tables/`) and figures in `generated/figures/` (PNG and TIFF at 600 dpi, and PDF).

## Main text

| Paper | File | Built from |
|---|---|---|
| Table 1 | Coefficients and their sources: `inputs/crop_economics_scenarios.csv`; NIR values and their CVs in `scripts/05_decision_value.py` and `scripts/06_allocation_scenarios.py` (Mainuddin et al. 2014, Tables 4.1 and 4.2) | — |
| Table 2 | `Table2_forecast_accuracy.csv` | `forecasts/` |
| Table 3 | `Table3_plans.csv` | `extensions/ext_12` |
| Table 4 | `Table4_decision_value.csv` | `decision/decision_value_tests.csv` |
| Fig. 5 (values) | `Fig5_robust_budgets.csv` | `decision/robust_thresholds.csv` |
| Figs. 1–8 | `Fig1_study_area` … `Fig8_sensitivity` | see `scripts/08_figures.py` |

## Supporting Information

| Paper | File |
|---|---|
| Table S1 | Sources in `inputs/crop_economics_scenarios.csv` and `inputs/irrigation_charge_references.csv` |
| Tables S2–S20 | `TableS2_decision_time_tuning.csv` … `TableS20_rule_baselines.csv` (Table S6 has two files, `TableS6a_feasibility_grid.csv` and `TableS6b_minimum_fallow.csv`; in them "Fixed NIR" means the published NIR and "PM climate deficit" the FAO-56 requirement) |
| Figs. S1–S6 | `FigS1_crop_data` … `FigS6_decision_value_dual_values` |

`extra_three_vs_four_crops.csv` (the model with three crops beside the model with four) is produced but not used in the paper.

In file and column names, `gross_margin` means price minus full production cost, which the paper calls net return.
