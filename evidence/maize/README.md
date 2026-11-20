# Maize as a fourth crop in the allocation models (13 Nov 2026)

Maize enters the district allocation model (scripts/06_allocation_scenarios.py) and the decision-time model
(scripts/05_decision_value.py), and gets the four decision-time forecasts (scripts/04_forecasts.py). The hindcast
models, SHAP and RAWSI stay on the three original crops (no maize season climate windows).

| Item | Value | Source |
|---|---|---|
| Area, production | 16 districts, 2014-15 to 2024-25 (11 years, as the other crops) | BBS Yearbook of Agricultural Statistics 2017-2025, rabi maize by district (Table 3.1.2.6 in the 2017-18 editions, 3.1.2.5 in 2019-22, 3.1.2.4 in 2023-25). Latest edition used for each year; every value printed in more than one edition is identical (96 + 64 checks). Parsers: evidence/data_audit/parsers/parse_maize.py, parse_maize_old.py |
| Net irrigation requirement | 354 mm nominal (midpoint of 279-429 mm); spatial deviation 75 mm; temporal CV 14.8% | Mainuddin et al. 2014 (CSIRO), Table 4.2, p.35, maize sown 1 December. Same rule as the other non-rice crops |
| Food energy | 3,440 kcal/kg | Food Composition Table for Bangladesh 2013 (Shaheen et al.), code 01_0005, maize/corn, yellow, dried, raw: 344 kcal/100 g, edible portion 1.00 (Table 1, printed p.3; checked in the PDF and the FCDB xlsx) |
| Production cost | Tk 12/kg (Tk 11-13 reported) | crop_economics_scenarios.csv; farmer-reported, cross-checked with Biswas et al. 2022 and Ahmad et al. 2026 |
| Price | Tk 18.65 / 23.0 / 28.5 per kg (low / mid / high) | crop_economics_scenarios.csv; mid is the base case. Low = BBS Yearbook 2025 Table 10.5.2 (farmers' average monthly price, FY2021-22; corrected from an earlier wrong label) |

Decision-time maize yield: the same four forecasts as the other crops (Elastic Net, persistence,
climatology, district trend). The hindcast source has no maize model; there maize takes its
decision-time Elastic Net forecast.

Land: maize joins the dry-season land pool. The pool of a district is the baseline area of Boro,
wheat, potato and maize (mean 2023-2025). As before, the pool is the sum of crop areas, not a
measured physical area (the dry-season crops together cover 104-158% of the net cropped area
because potato and mustard are followed by Boro on the same land; see evidence/data_audit).
