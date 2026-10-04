# Data

The raw datasets required to reproduce this analysis are not included in the repository due to standard ONS redistribution restrictions. All source files should be placed in `data/raw/` with the exact filenames below.

## Required files

| Filename | Source | Notes |
|---|---|---|
| `Employee_counts_IS8_LADs.parquet` | [NOMIS](https://www.nomisweb.co.uk/) | Business Register and Employment Survey (BRES) employee counts, 2015 to 2024, by Local Authority District and IS8 sector |
| `Business_counts_IS8_LADs.parquet` | [NOMIS](https://www.nomisweb.co.uk/) | UK Business Counts, 2016 to 2025, by Local Authority District and IS8 sector |
| `merged_cleaned.csv` | ONS, HESA, deprivation indices, plus derived variables | Harmonised local indicators dataset (~55 variables across ~350 LADs); see note below |
| `uk_lad_boundaries_2023.geojson` | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) | 2023 Local Authority District boundaries |
| `IS-8_SIC_Lookup.csv` | [gov.uk](https://www.gov.uk/government/publications/industrial-strategy/industrial-strategy-sector-definitions-list) | Mapping of SIC codes to the eight IS8 sectors |

## Notes on `merged_cleaned.csv`

This file combines ONS Local Indicators, English Indices of Deprivation, Scottish Index of Multiple Deprivation, HESA higher education data, median house prices, and derived labour market variables. It was built by the group during the original PP422 project and is treated as an input to the scripts in this repository rather than being rebuilt here.

If you need to reconstruct it from scratch, the primary sources are:

- ONS Local Indicators (2022)
- English Indices of Deprivation (2019)
- Scottish Index of Multiple Deprivation (2020)
- HESA higher education research income and student counts (2024)
- NOMIS UK Census (2021) labour market statistics

The exact harmonisation and Scottish backfill decisions are not fully documented in this repository, so reproducing `merged_cleaned.csv` from scratch would require re-deriving them.

## Expected folder structure

    data/
    ├── README.md                           # This file
    └── raw/
        ├── Employee_counts_IS8_LADs.parquet
        ├── Business_counts_IS8_LADs.parquet
        ├── merged_cleaned.csv
        ├── uk_lad_boundaries_2023.geojson
        └── IS-8_SIC_Lookup.csv

## Validation

The scripts in `scripts/` do not perform explicit file-existence checks. If a required file is missing from `data/raw/`, the script will raise a `FileNotFoundError` on the relevant read call.