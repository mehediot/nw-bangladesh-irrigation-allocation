# Change log

## 1.1.1 (November 2026)

- Fig. 8 redrawn: inputs grouped by kind (water, land and policy; yields; prices and costs; management option), one range per input and outcome, single alternatives as dots and "no change" as grey dots, value labels on each range. Same data (ext_28_sensitivity_tornado.csv); no table changes.

## 1.1.0 (November 2026)

- FAO-56 crop water requirement of potato: the season is now 140 days (FAO-56 Table 11 stage lengths 30/35/50/25), the potato season of Mainuddin et al. (2014, Fig. 3.1) whose planting date and crop coefficients the calculation already used; it was 115 days (25/30/30/30). Mean potato requirement 309 mm (was 214 mm), now at the lower edge of the published range 309-444 mm.
- Outputs that use the FAO-56 requirement change accordingly (14 of 91 reference tables): the FAO-56 sensitivity case (smallest fallow at budget 0.80 18.2%, was 17.4%; districts that cannot keep all Boro 4/10/14 at 0.90/0.80/0.70, were 5/12/14), the relative scarcity index (Gaibandha and Sirajganj, Lalmonirhat and Chapainawabganj swap ranks; Spearman with the Boro share 0.60, was 0.62) and the FAO-56 bar of the sensitivity tornado. The allocation model with the published NIR, the forecasts and every other result are unchanged.

## 1.0.1 (November 2026)

Presentation changes only; all 91 output tables are unchanged in content.

- Table 5 of earlier drafts is no longer a table of the paper: its values are plotted in Fig. 5, so the file is renamed `Table5_robust_budgets.csv` -> `Fig5_robust_budgets.csv` (same content).
- Fig. 2: the water constraint is written as in Eq. (4) of the paper, `10 Σ n_c x_c ≤ ρ W_0`.

## 1.0.0 (November 2026)

First public release, accompanying the submitted paper.

- Eight-stage pipeline from the raw inputs to every table and figure of the paper; 91 output tables reproduced byte-for-byte against `reference/`; 44 independent checks.
- Step-by-step notebook (`run_all.ipynb`) that shows each table and figure next to the stage that produces it.
- Tables and figures named by their number in the paper; figures as PNG and TIFF at 600 dpi and as vector PDF.

Data corrections made while assembling the inputs are documented in `evidence/data_audit/` (the only correction to the published statistics used is the wheat production of FY2022-23, see `data_corrections_13nov2026.csv`).
