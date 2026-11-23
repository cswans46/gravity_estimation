"""Gravity-based and spatial-econometric estimation of interregional input-output flows.

Modules
-------
ppml          PPML with high-dimensional fixed effects (IRLS + alternating projections)
icio          OECD ICIO preparation (Cai 2023 normalisation; supplier x user pairs)
cepii         CEPII GeoDist distances/controls aligned to ICIO codes
elasticities  Panel distance elasticities delta_k and delta_{k,m}
ras           Biproportional balancing (doubly constrained gravity)
disaggregate  Canadian Z[i, j, b, u]: gravity-RAS vs import proportionality
distances     CES population-weighted distances (CEPII distwces analogue)
sdm           Bayesian multi-equation spatial Durbin model (Torój) with gamma decay
synthetic     Known-parameter data generators for tests
"""
