# Irrigation-constrained crop allocation in northwest Bangladesh: code and data

Code, input data and reference outputs for the paper (in preparation):

> Khan, M.M.Z., Ahmed, M.T., Ashiqussalehin, M. *An Explainable Machine Learning and Optimization Framework for Irrigation-Constrained Multi-Crop Decision Support in Northwest Bangladesh* (provisional title).

The analysis asks what a smaller irrigation budget costs the 16 districts of Rajshahi and Rangpur divisions in cropped land, food energy and net return; whether better yield forecasts lead to better plans; how uncertainty in crop water requirements changes the budget a cut needs; and how much alternate wetting and drying (AWD) and low-water crops relieve the budget. Every number, table and figure in the paper is produced by this pipeline.

## Run it

Python 3.13 and the package versions in `requirements.txt` are required. With other versions the last digits of some outputs can change, and the exact comparison with `reference/` then fails.

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py          # about 7 minutes
python checks/independent_check.py .
```

`run_all.py` checks every input file and script against `SOURCE_INPUT_MANIFEST.json`, runs the eight stages below into a new `generated/` folder, and ends with `SUCCESS 91 exact CSV matches` when every output table is byte-identical to its copy in `reference/`. Reference files are used only for this comparison, never as inputs.

`checks/independent_check.py` recomputes the main results by different routes (other formulations of the linear programs, statsmodels instead of the pipeline's own regression code, a separate FAO-56 implementation tested on FAO-56 Example 18, finite differences instead of dual values) and ends with `ALL PASS` (44 checks). It tests the algebra and the code, not the modelling choices.

**Step by step:** `run_all.ipynb` runs the same eight stages one at a time and shows, after each stage, the tables and figures of the paper that it produces. Open it from the repository root and run all cells, or `jupyter execute run_all.ipynb`. It ends with the same reference comparison and the independent check.

## Stages

| Stage | Script | What it does | Output folder |
|---|---|---|---|
| 1 | `scripts/01_climate.py` | Season climate predictors from the NASA POWER files; FAO-56 crop water requirements (daily Penman–Monteith ET0, crop coefficients, effective rain, Boro percolation) | `climate/` |
| 2 | `scripts/02_scarcity_index.py` | Relative agricultural water scarcity index (RAWSI) | `scarcity_index/` |
| 3 | `scripts/03_hindcast_models.py` | Hindcast yield models with same-season climate (Elastic Net, gradient-boosted trees), SHAP values, fixed effects; descriptive only | `hindcast/` |
| 4 | `scripts/04_forecasts.py` | Decision-time yield forecasts with information up to two years before the season (Elastic Net, persistence, climatology, district trend) and the exact contribution of each predictor to every Elastic Net forecast | `forecasts/` |
| 5 | `scripts/05_decision_value.py` | Decision-time planning for 48 district-years: decision value of the forecasts, break-even forecast errors, Γ-robust water constraint | `decision/` |
| 6 | `scripts/06_allocation_scenarios.py` | District allocation model and its scenarios: feasibility, smallest fallow, two objectives, dual values, prices, AWD, six-crop variant, value of water, sensitivity, Boro floor crossed with crop limits, land accounting, simple rules; robustness checks of stages 2–3 | `extensions/` |
| 7 | `scripts/07_tables.py` | Tables of the paper, named by their number (`Table2_…` to `Table4_…`, `Fig5_robust_budgets` with the values of Fig. 5, `TableS2_…` to `TableS20_…`) | `tables/` |
| 8 | `scripts/08_figures.py` | Figures of the paper, named by their number (`Fig1_…` to `Fig8_…`, `FigS1_…` to `FigS6_…`), as PNG and TIFF (600 dpi) and PDF | `figures/` |

Stages 7 and 8 take `--only` to build selected exhibits, e.g. `python scripts/08_figures.py --input-dir inputs --generated-dir generated --output-dir generated/figures --only Fig3 FigS4`.

The allocation model is a linear program per district: crop areas and fallow on a fixed land pool, a water budget equal to a ratio ρ of the baseline demand, a floor on the Boro rice area, and upper limits equal to the largest area of each crop observed in 2015–2025. It is solved with HiGHS through SciPy.

In file and column names, `gross_margin` means price minus full production cost, which the paper calls *net return*.

## Folders

| Folder | Content |
|---|---|
| `inputs/` | All model inputs (described below) |
| `scripts/` | The eight stages and `run_all.py` |
| `run_all.ipynb` | The pipeline step by step, with every table and figure |
| `reference/` | Expected outputs, for the exact comparison after a run |
| `checks/` | Independent re-computation of the main results |
| `evidence/` | Data audit against the BBS yearbooks (with the parsing scripts), and source notes for maize, lentil, mustard and the budget range |
| `docs/` | Map from the paper's exhibits to the output files, and the change log |

## Inputs and their sources

| File | Source |
|---|---|
| `analysis_panel_full.csv` | District area, production and yield of Boro rice, wheat and potato, 2014–15 to 2024–25, from the BBS *Yearbook of Agricultural Statistics* (editions 2015 to 2025); every row was checked against a yearbook table (`evidence/data_audit/`) |
| `maize_rabi_16districts_2015_2025.csv`, `lentil_mustard_16districts_2015_2025.csv` | The same yearbooks; source table and edition in each row |
| `nasa_power_daily/`, `nasa_power/` | NASA POWER daily and monthly agroclimatology data at each district headquarters (accessed November 2026). The 16 districts share 11 distinct series |
| `crop_economics_scenarios.csv` | Prices, costs, energy contents and conversion factors; source and note in each row |
| `irrigation_charge_references.csv` | Irrigation charges and costs used for comparison with the model value of water |
| `hodgson_2021_groundwater_trends.csv` | Well counts by trend class from Hodgson et al. (2021), CSIRO |
| `gw_depletion_badc_mis1819.csv` | Groundwater depth values read from the BADC Minor Irrigation Survey Report 2018–19, used to check the groundwater-trend input |
| `bgd_districts_geoboundaries_adm2_simplified.geojson` | District boundaries from geoBoundaries (source BBS / OCHA ROAP) |
| `nasa_pm_seasonal_cache_16districts_3crops_2015_2025.csv` | Seasonal effective rain and an earlier monthly-file estimate of crop evapotranspiration, used only for comparison in stage 1 |

Net irrigation requirements are the published values of Mainuddin et al. (2014, 2015). The source documents themselves (yearbook PDFs, CSIRO reports, government letters, the Food Composition Table) are not included; they are cited in the input files and in the paper.

## How to read the results

The water budget is a planning scenario in units of net irrigation requirement, not a measured sustainable yield or pumped volume. Fallow in the model means harvested area leaving irrigated dry-season cropping. The scarcity index is a relative ranking. Plans, net returns and dual values are model values under fixed national prices, not forecasts of what farmers will grow or earn.

## Licence

Code: MIT (`LICENSE`). Data files created for this work and all outputs: CC BY 4.0 (`LICENSE-DATA`). Third-party data keep their own terms: BBS statistics, NASA POWER (please acknowledge the NASA POWER project), geoBoundaries (CC BY 3.0 IGO for this layer), and values taken from the cited reports.

## Citation

See `CITATION.cff`. A DOI will be added when the archived release is created.
