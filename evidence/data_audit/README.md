# Data audit against the BBS Yearbooks of Agricultural Statistics (2020-2025), 13 Nov 2026

Source PDFs (supplied by the first author; not included here because of size): BBS Yearbook of Agricultural Statistics
2020, 2021, 2022, 2023, 2024 and 2025. Text was extracted with `pdftotext -layout` and parsed with the scripts in `parsers/`.

| File | Content |
| --- | --- |
| `panel_vs_yearbook_audit_2019_2025.csv` | Every panel row 2019-2025 compared with the yearbook district tables ("Estimates of Total Boro", "Estimates of Wheat", "Estimates of Total Potato"). Boro 110 and potato 105 parsed rows match exactly; wheat areas match; wheat FY2022-23 production did not. |
| `wheat_2022_23_check.csv` | Wheat FY2022-23 area and production: panel vs yearbooks 2023 and 2024 (the two editions agree). |
| `data_corrections_13nov2026.csv` | The 14 corrected rows (wheat production FY2022-23; yield recomputed as production / area). |
| `analysis_panel_full_before_13nov2026.csv` | The panel before the correction. |
| `landuse_16districts_2017_2023.csv` | Table 5.1, land utilisation by district ('000 acres): current fallow, single / double / triple cropped, net and gross cropped area, FY2017-18 to FY2022-23. |
| `dry_season_land_use_2022_23.csv` | FY2022-23 area ('000 ha) of the three model crops and of all other dry-season (rabi) crops by district, vs net cropped area. All 16 districts exceed 100% (region 119%): dry-season crops follow one another on the same land (BBS crop calendar, Yearbook 2025 Table 1.8: potato harvested mid-Jan to March, mustard late Jan to mid-Feb, HYV/hybrid Boro transplanted Dec to mid-Feb). |

Known source issues: a few Table 5.1 rows do not satisfy net = single + double + triple + quadruple exactly (differences of 1-2 thousand acres in the source).

## Added 13 Nov 2026 (evening): years 2015-2019 checked against the 2015-2019 yearbooks

The first author supplied the BBS Yearbooks of Agricultural Statistics 2012, 2013 and 2015-2019. Total Boro,
wheat and total potato by district were parsed from the 2015-2019 editions (`parsers/parse_old.py`).
All 240 panel rows for 2015-2019 (3 crops x 16 districts x 5 years) match the yearbooks: area and
production agree to within 0.05% in every edition that prints the year
(`panel_vs_yearbook_audit_2015_2019.csv`). With the earlier check of 2019-2025, all 528 panel rows are
now traced to a yearbook; the only correction ever needed is wheat FY2022-23 (above).

Rabi maize for 2014-15 to 2016-17 was parsed from the 2016-2019 editions (`parsers/parse_maize_old.py`),
which extends the maize input to 2015-2025. The 2012 and 2013 editions cover years before the panel and
were not used.
