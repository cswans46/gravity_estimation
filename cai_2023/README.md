# ICIO gravity estimation

This project estimates industry-specific bilateral distance elasticities using the OECD
Inter-Country Input-Output (ICIO) framework, following the empirical strategy in
Mattia Cai (2023), *A calibrated gravity model of interregional trade*.

The objective is to estimate elasticities from international and domestic country-level
flows, independently of the Canadian provincial input-output tables to which the
elasticities may later be applied.

## Current implementation status

The pipeline is designed for the 1995–2022 panel and 50 ICIO industries. It constructs
bilateral sales by supplying industry from intermediate and final demand, merges CEPII
geography, reads the local RTA/tariff Stata file, and estimates sector-specific OLS and
PPML specifications. **The first local run is a required data-validation step:** the
OECD RData object layouts and the Stata variable schema cannot be verified from this
repository alone. The pipeline writes schema reports and stops with actionable errors
rather than silently guessing variable mappings.

## Setup

From the repository root:

```bash
cd cai_2023
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
python -m pip install -e ".[dev]"
copy config.example.yaml config.yaml
```

Edit `config.yaml` to point to your local ICIO, CEPII, and Stata files. Raw data are
not committed to Git. The default paths reflect the current project setup but can be
changed without editing Python source.

## First run: inspect data

```bash
python inspect_trade_agreements.py --config config.yaml
python run_estimation.py --config config.yaml --inspect-only
```

The Stata inspection command reads metadata first and writes a variable/schema report.
It does not load the full file into memory. Configure the origin, destination, year,
RTA, and tariff columns in `config.yaml` using the report. The loader then reads only
the selected columns in chunks and aggregates to country-pair-year.

The ICIO inspect-only run records object types, dimensions, labels, and candidate
country/industry parsing. Confirm these against the OECD release documentation before
running estimation.

## Estimate

```bash
python run_estimation.py --config config.yaml
```

Outputs are written to the configured output directory:

- `bilateral_industry_flows.parquet`: bilateral intermediate + final sales by source industry.
- `estimation_panel.parquet`: estimation panel with CEPII and agreement/tariff controls.
- `estimates_ols.csv`, `estimates_ppml.csv`: industry-specific coefficients and diagnostics.
- `sample_diagnostics.csv`: missingness, zero-flow, and merge coverage diagnostics.
- `run_metadata.json`: configuration, package versions, sample, and run timestamp.
- `logs/estimation.log`: processing and estimation log.

Parquet output requires `pyarrow` (included in the project dependencies).

## Important methodological notes

1. For origin country (i), destination country (j), supplying industry (k), year (t),
   bilateral sales are the sum of all intermediate uses and final uses in destination (j).
2. Destination absorption is total purchases of supplying-industry output (k) by
   destination (j), summed over origins.
3. The Cai normalized outcome is (z_{ijkt}=F_{ijkt}/(Y_{ikt}E_{jkt})).
4. OLS estimates (log z) on log distance, border, language, RTA, and tariff controls,
   with origin-industry-year and destination-industry-year fixed effects. Zero flows
   cannot enter log OLS.
5. PPML is an extension that estimates (F) in levels with (log(YE)) as an offset.
   It retains zero bilateral flows and uses the same fixed-effect structure.
6. Distance is time-invariant. Country-pair fixed effects are therefore not included in
   the main specification because they would absorb distance.
7. The agreement file's actual schema and coverage must be checked before the RTA/tariff
   regressors can be considered validated. No variable mapping is presumed.

See [docs/methodology.md](docs/methodology.md) for equations, choices, and limitations.
