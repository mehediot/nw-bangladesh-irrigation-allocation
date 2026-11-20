# Change log

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
