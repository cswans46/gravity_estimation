# 4-D interregional–intersectoral flows for Canada: Cai (2023) gravity and Torój spatial Durbin IRIO

* Exact Cai (2023) gravity, eq. (9), on the OECD ICIO 2025 (50 industries, panel 1995–2022),
  by sector and by supplier × user pair.
* Cai's specification inside Canada, using StatCan interprovincial flows and ODBus /
  Business Register activity-weighted distances (NAICS × province elasticities).
* Gravity-RAS split of interprovincial flows by demanding sector, Z[i, j, b, u], compared with
  import proportionality, Fréchet bounds and the Schulte et al. (2021) envelope.
* A separate routine: Torój (2024) Bayesian spatial Durbin IRIO (1 − gamma CDF decay,
  inverse-Wishart covariance, Torój prior), with the 2016 supply exponent and sensitivities.

**Full methodology, per-paper 4-D constructions, Canadian recipes and open points:
[docs/METHODS.md](docs/METHODS.md).**

## Install and test
```bash
pip install -e .[dev]
pytest            # 33 tests on synthetic data with known parameters
```

## Run
```bash
python estimation.py --data-dir "D:/Models/OECD Research" --cepii-dir .        # 1a ICIO (Cai)
python estimation.py ... --sample eu27 --year 2009                               # Cai's own design
python canada_elasticities.py --canada-dir data/canada --slope-by origin         # 1b IPT + ODBus
python make_concordances.py --canada-dir data/canada                             # SUT -> ICIO
python canada_disaggregation.py --canada-dir data/canada \
       --delta-source ipt --distances user-specific --schulte-draws 200         # 2  Z[i,j,b,u]
python canada_disaggregation.py ... --trade-source provincial-tables             # 2  without IPT
python spatial_irio.py --sdm-dir data/canada_sdm --prior toroj --sensitivity     # 3  SDM
# demos on synthetic data:
python canada_disaggregation.py --demo --schulte-draws 50
python spatial_irio.py --demo --sensitivity
```

## Layout
```
gravity_irio/
  cai.py               Cai (2023) eq. (9) OLS-FE, calibration, STPE/DTPE, EU27 sample
  ppml.py              PPML-HDFE (exact two-way FE projection)
  icio.py              ICIO preparation; CN1/CN2, MX1/MX2 aggregation
  cepii.py             CEPII GeoDist alignment
  elasticities.py      PPML panel elasticities (robustness)
  concordance.py       ICIO 2025 industries, NAICS->ICIO bridge, IOIC/SUPC mapping
  data/naics2017_to_icio2025.csv
  business_locations.py ODBus/BR -> weights -> grid -> sector-pair CES distances
  canada_gravity.py    Cai spec on StatCan interprovincial flows (NAICS x province)
  disaggregate.py      gravity-RAS 4-D split, IPA, Cai provincial calibration, Fréchet, Schulte
  sdm.py               Torój spatial IRIO (prior, sampler, flows, 3-margin RAS)
  sdm_sensitivity.py   sensitivity scenarios
  final_demand.py      regional final demand from provincial SUTs / category keys
  distances.py, ras.py, canada_io.py, synthetic.py
```
