# Lentil and mustard: the six-crop variant (13 Nov 2026)

The main allocation model has four crops. `ext_29_low_water_crops.csv` (Table S13, Figure 14) is a variant
that adds lentil and mustard, the two low-water dry-season crops with district data, to ask one question:
how far would they have to expand to take the land that a smaller water budget releases from Boro?

| Item | Value | Source |
|---|---|---|
| Area, production | 16 districts, 2014-15 to 2024-25 (352 rows) | BBS Yearbook of Agricultural Statistics 2016-2025: Table 3.2.3 (lentil), Table 3.3.2 (rape and mustard, local + HYV). Latest edition for each year. Parser: `evidence/data_audit/parsers/parse_minor.py`; all parsed editions in `lentil_mustard_all_editions_parsed.csv` |
| Edition agreement | 315 district-years are printed in more than one edition; all agree | The mustard table of the 2018 edition is broken in the PDF text (6 districts parsed) and is not used |
| Replaced values (3) | Mustard, Dinajpur 2015-16 (printed production gives 7.9 t/ha) and Natore 2015-16 (0.15 t/ha): production = mean yield of the two neighbouring years x printed area. Mustard, Gaibandha 2014-15 (printed "-" / 0): area and yield = mean of 2013-14 and 2015-16 | Columns `production_t_published` and `note` in the input file |
| Net irrigation requirement | lentil 197.5 mm (pulses 164-231), mustard 93 mm (oilseeds 81-105) | Mainuddin et al. 2014 (CSIRO), Table 4.2, p.35; midpoints, the same rule as wheat, potato and maize |
| Yield in the plan | yield of year t-2, mean over targets 2023-2025 (decision-time persistence) | no Elastic Net model for these crops |
| Food energy | lentil 3,170 kcal/kg dal x 0.725 kg dal per kg grain; mustard 9,000 kcal/kg oil x 0.35 kg fat per kg seed | FCT for Bangladesh 2013: 02_0008 lentil, dried, raw 317 kcal/100 g; 13_0008 mustard oil 900 kcal/100 g; 06_0012 mustard seed fat 35.0 g/100 g. Milling: Hajong et al. 2020 |
| Price | lentil Tk 82.66/kg, mustard Tk 77.68/kg | Yearbook 2025, Table 10.5.1 (DAM harvest-time price), mean FY2022-23 to FY2024-25 |
| Cost | lentil Tk 40.67/kg (Tk 66,374/ha at 1.632 t/ha); mustard Tk 39.55/kg (Tk 71,393/ha at 1.805 t/ha) | Hajong et al. 2020, Int J Sustain Agric Res 7(4):234-243; Uddin, Kundu & Tushar 2024, Bangladesh J Nucl Agric 38(1):65-80 (Sirajganj, 2023-24) |

Limits to state in the paper:
- Lentil and mustard are often grown on residual soil moisture with little irrigation; the NIR used is the
  modelled requirement of a fully irrigated crop.
- Land is additive in the model. Short-duration mustard can precede Boro on the same field, so mustard needs
  less extra land than the model assumes.
- Mustard energy counts all seed fat as oil (an upper bound on what an expeller recovers).
- The crop limit k x baseline is a scenario, not a forecast of adoption. Observed 2015-2025 maxima are
  1.49 x (lentil) and 1.10 x (mustard) the 2023-2025 baseline for the region.
