# Gravity and spatial-econometric estimation of interregional I–O flows

This project extends Cai (2023) to the OECD ICIO (85 regions, 28 years, 50 industries). It
applies the resulting distance elasticities to split Canadian interprovincial flows by
demanding sector. As a separate routine, it includes Torój's Bayesian spatial Durbin IRIO
model with gamma distance decay.

Full methodology, formulas and open questions: **[docs/METHODS.md](docs/METHODS.md)**.

## Install
```bash
pip install -e .[dev]      # numpy, pandas, scipy, pyreadr, xarray, xlrd, joblib
pytest                     # 15 tests on synthetic data with known parameters
```

## Run
```bash
# 1. Distance elasticities from the ICIO (delta_k and delta_{k,m})
python estimation.py --data-dir "D:/Models/OECD Research" --cepii-dir . --n-jobs 4

# 2. Canadian Z[i, j, b, u]: gravity-RAS vs import proportionality
python canada_disaggregation.py --canada-dir data/canada \
       --elasticities results/elasticities_pair_filled.csv

# 3. Bayesian spatial Durbin IRIO (separate routine)
python spatial_irio.py --sdm-dir data/canada_sdm --prior gravity --chains 4 --iter 4000

# Steps 2 and 3 also run on synthetic data:
python canada_disaggregation.py --demo
python spatial_irio.py --demo
```
Input file layouts are described in the docstrings of `gravity_irio/canada_io.py` and
`spatial_irio.py`.

## Layout
```
gravity_irio/
  ppml.py          PPML with high-dimensional FE (exact two-way solve, ppmlhdfe-style IRLS)
  icio.py          ICIO preparation (Cai normalisation; supplier x user flows)
  cepii.py         CEPII distances/controls -> ICIO codes (CN1/CN2, MX1/MX2, ROW handled)
  elasticities.py  panel delta_k, delta_{k,m}, filling/shrinkage
  ras.py           RAS / doubly constrained gravity
  disaggregate.py  Canadian 4-D split: gravity-RAS and IPA
  distances.py     CES population-weighted distances (distwces analogue)
  sdm.py           Torój spatial IRIO: gamma-decay W, posterior, MCMC, flows
  synthetic.py     known-parameter data generators
  canada_io.py     tidy-CSV loaders
```
