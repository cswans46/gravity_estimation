# Methods: gravity-based and spatial-econometric estimation of Canadian interprovincial I–O flows

This document records every modelling step, formula, default and design choice in
`cai_2023/`, and lists the open questions (Section 8). Code references are to
`gravity_irio/<module>.py`.

---

## 1. Overview

| Step | Script | What it does |
|---|---|---|
| 1 | `estimation.py` | Panel PPML gravity on the OECD ICIO (85 regions × 28 years × 50 industries): distance elasticities by supplying industry, δ_k (Cai 2023), and by supplying × using industry, δ_{k,m} (extension). |
| 2 | `canada_disaggregation.py` | Splits Statistics Canada interprovincial flows T[i,j,b] by demanding sector u in the destination, giving a 4-D Z[i,j,b,u]. Gravity-RAS with δ_{k,m}, compared against import proportionality (IPA). |
| 3 | `spatial_irio.py` | Separate routine: Torój's multi-equation spatial Durbin model (SDM) with gamma distance decay, estimated by Bayesian MCMC, giving a posterior for Z[r,s,p,v]. |

Notation: i, b = origin and destination region; j, k, s = supplying commodity or
industry; u, m, v = using sector (industry or final-demand category); t = year.

---

## 2. Data: OECD ICIO and CEPII

### 2.1 ICIO (`icio.py`)
* Inputs are `ICIO2025econZ/X/FD.Rdata`, read with `pyreadr` (same as the original code).
* Z is reshaped to Z5[t,i,k,j,m]. The code checks that labels are country-major (`CCC_ind`).
* FD must be aggregated to one column per destination country, as in the original code.
* **Sector-level flows (Cai):** F_{ijk,t} = Σ_m Z_{ik,jm,t} + FD_{ik,j,t}.
  The original code left FD out of F but included it in E. The default now includes it,
  so the numerator and denominator cover the same thing. `--no-final-demand` restores the
  old behaviour (see §8, item 4).
* Normalised flow (Cai): z_{ijk,t} = F_{ijk,t} / (Y_{ik,t} E_{jk,t}), where Y is gross
  output and E_{jk,t} = Σ_i Σ_m Z + Σ_i FD is destination expenditure on k.
* **Supplier × user flows:** F_{ij,km,t} = Z_{ik,jm,t} for m = 1..50, plus m = "FD"
  (F = FD_{ik,j,t}). Normalised by Y_{ik,t} · E_{jm,t}, where E_{jm,t} = Σ_i F_{ij,km,t}.
* Domestic flows (i = j) stay in throughout, as in the original code.

### 2.2 CEPII GeoDist (`cepii.py`)
* `dist_cepii.xls` gives distances and bilateral dummies (contig, comlang_off,
  comlang_ethno, colony, comcol, smctry). It includes iso_o = iso_d rows with internal
  distances. Reading `.xls` needs `xlrd`, which is now in `pyproject.toml`.
  **Note:** your code loaded the "language etc." controls from `geo_cepii.xlsx`. In GeoDist,
  the *bilateral* dummies are in `dist_cepii`; `geo_cepii` holds country-level variables
  (area, landlocked, coordinates). The new code takes bilateral controls from `dist_cepii`.
* Distance variable: `distwces` by default (population-weighted, CES with θ = −1).
  Where it is missing, `dist` fills in and the count is logged.
* Legacy CEPII codes are remapped (ROM→ROU, ZAR→COD, TMP→TLS, YUG→SRB, PAL→PSE).
* **ROW** is dropped because it has no distance.
* **CN1/CN2 and MX1/MX2** (the extended ICIO split of China and Mexico) map to CHN and MEX.
  A pair between the two halves of one country gets that country's internal distance and
  `home = 1`. The mapping is configurable (`parent_map`). See §8, item 5.

---

## 3. Distance elasticities (`ppml.py`, `elasticities.py`)

### 3.1 Estimator
Structural gravity estimated by PPML (Santos Silva & Tenreyro 2006):

  E[y_{ij,t}] = exp( α_{i,t} + γ_{j,t} − δ ln d_ij + β_home·home_ij + Σ_c β_c·control_{c,ij} )

* y is z_{ij,t} (default, matching your Cai-style set-up) or F_{ij,t} (`--dep F`).
  The origin-year and destination-year FE absorb Y_{i,t} and E_{j,t}, so both choices
  estimate the same δ consistently. They differ only in implicit weights: PPML on z gives
  less weight to large countries. With noiseless data the two give identical δ (tested).
* Default controls: `home, contig, comlang_off, colony`. The home dummy matters most: it
  stops the domestic-versus-international border effect from loading onto δ.
* FE: origin×year and destination×year, so the 28 years are pooled with δ held constant.
  `by_year=True` gives one cross-section per year, to check stability over time.
* Algorithm: IRLS as in ppmlhdfe (Correia, Guimarães & Zylkin 2020). With exactly two FE
  sets, the weighted FE projection is solved *exactly* through the Schur complement (a
  sparse LU, block-diagonal by year). It matches alternating projections to 5e-12 and is
  about 17× faster: 0.8 s rather than 14.6 s for one 202k-observation regression. With other FE structures
  the code falls back to alternating projections.
* Groups whose outcome is all zero are dropped iteratively (separation; the FE would go to −∞).
* Negative outcomes (possible in FD through inventory changes) are set to 0 and counted
  (`n_neg_clipped`).
* SEs are clustered by country pair (i,j) with a G/(G−1) correction.

### 3.2 Why δ_k is not enough for the Canadian split (key result)
In the Canadian problem, T[i,j,b] (origin × commodity × destination) is observed. For each
(j, b), the unknowns form a matrix over origins i × users u. Its row sums are T[·,j,b] and
its column sums are the users' domestic purchases Udom[j,b,·]. A gravity seed built from
supplier-only elasticities is

  S[i,u] = d_ib^(−δ_j) · f(i) · g(u),

which is **rank one** in (i, u), because d_ib is fixed once b is fixed. RAS on a rank-one
seed returns exactly r_i c_u / Σc, which is the IPA solution. So **the 50 supplier-level
elasticities cannot change the demanding-sector split.** The split moves away from IPA only
if the elasticity varies by *user*, through δ_{j,u}. This is why step 1 also estimates
δ_{k,m} for 50 suppliers × (50 industries + FD) = 2,550 regressions. The nesting is a unit
test (`test_gravity_split_nests_ipa_when_delta_constant_over_users`). For the same reason,
any regressor that depends only on the (i,b) pair, such as contiguity or language, also
drops out unless its coefficient varies by user.

### 3.3 Pair-level estimation details
* Pairs with fewer than `min_positive` (default 500) positive flows in the panel are skipped.
* `fill_pair_elasticities` fills skipped or non-converged pairs with δ_k. By default it also
  applies empirical-Bayes shrinkage toward δ_k within each supplier:
  δ̃_km = w·δ_km + (1−w)·δ_k, where w = τ²_k / (τ²_k + se²_km) and
  τ²_k = max(0, Var_m δ_km − mean se²) (method of moments). **This is my design choice, not
  something from Cai.** It keeps noisy cells from driving the Canadian split. Turn it off
  with `shrink=False`.
* Runtime is about 0.8 s per regression, so roughly 35 minutes for all pairs on one core.
  `--n-jobs` uses threads. Memory: Z5 is 28×85×50×85×50 float64 ≈ 4 GB, the same as your
  current code.

---

## 4. Canadian 4-D disaggregation (`disaggregate.py`)

Target: Z[i,j,b,u] = purchases by user u in province b of commodity j produced in province i.

**Constraints** (for every estimator):
  Σ_u Z[i,j,b,u] = T[i,j,b]   and   Σ_i Z[i,j,b,u] = Udom[j,b,u].

**Domestic use** (`domestic_use`): Udom = U · (1 − M_jb / Σ_u U_jbu). This means
international imports are allocated to users by import proportionality. You can pass
user-specific domestic shares instead, for example taken from the ICIO CAN block. Column
totals are rescaled to match Σ_i T where the two disagree (because of inventories,
re-exports and the like). The rescale factors are reported.

**IPA (benchmark):** Z = T[i,j,b] · Udom[j,b,u] / Σ_u' Udom[j,b,u'].

**Gravity-RAS:** for each (j,b), RAS on the seed
  S[i,u] = d_ib^(−δ_{j,u}) · exp(κ · h_{j,u} · 1[i=b]).
δ_{j,u} comes from the ICIO δ_{k,m} through concordances (`map_pair_elasticities`):
δ_{ju} = Σ_k Σ_m C^s_{jk} δ_km C^u_{um}, with weights in each row summing to 1.
h_{j,u} is the ICIO home coefficient, and κ ∈ [0,1] scales it (default 0, off). The
international home coefficient measures the effect of a national border, which is probably
too strong for a provincial one. Distance units cancel (they are absorbed by the column
scaling). Internal distances on the diagonal must be positive.

**Interpretation:** a user with a larger δ_{j,u} sources more of j from its own and nearby
provinces than the average user, and RAS keeps the observed totals exact.

**Distances** (`distances.py`): for δ to transfer from the ICIO, Canadian distances should
be built like `distwces`, as CES (θ = −1) population-weighted averages of great-circle
distances between populated places (for example census subdivisions or population centres),
with internal distance 0.67·√(area/π) for each place.

**Outputs:** `results/canada_Z4_gravity.{csv,npy}` and `canada_Z4_ipa.*`, plus the
own-province sourcing share by user under each method.

---

## 5. Canadian data (inputs to prepare)

The scripts read tidy CSVs (`canada_io.py` docstring) rather than parsing StatCan's raw
layouts. Suggested sources:
* Table 12-10-0101-01 (detail level) or 12-10-0088-01 (summary level): interprovincial and
  international trade flows at basic prices → T[i,j,b] (own-province supply = domestic
  supply − interprovincial exports), and international imports M[j,b].
* Table 36-10-0478-01: provincial and territorial supply and use tables, detail level →
  U[j,b,u] (use by industries and final-demand categories).
* Concordances: SUPC commodities → ICIO industries (ISIC Rev. 4) and NAICS-based SUT
  industries → ICIO industries. **You need to supply these**; no official one-step mapping
  exists.
* For the SDM: provincial value added by industry (from the SUT), national Z and X (the
  Canada table, domestic flows), and provincial final demand by commodity.

---

## 6. Spatial IRIO: Bayesian spatial Durbin model (`sdm.py`, `spatial_irio.py`)

### 6.1 Model (Torój 2016 eqs. 7–11, 19–23; Torój 2024, 2025)
For each sector s, with value added by region va^s (R×1) and regional final demand y^s:

  va^s_t = W_{s,t} ( Σ_v β^s_{v,t} va^v_t + β^s_{0,t} y^s_t ) + ε^s_t,  ε^s_t ~ N(0, σ²_s I)

The β are **known** national ratios, β^s_v = (VA^s/X^s)·Z^{sv}/VA^v and
β^s_0 = VA^s/X^s, so the equation summed over regions is an exact national identity. The
code checks this (`national_identity_gap`). It is an SDM: va^s appears on the left and,
through β^s_s W_s va^s, as a spatial lag on the right. The other sectors enter as spatially
lagged regressors (WX).

### 6.2 Spatial weights with gamma distance decay
  W_s(r,p) ∝ g(d_rp; shape_s, scale_s) · (m^s_r)^{γ_s}, normalised so each **column** sums to 1.
* g = 1 − F_Γ (the gamma survival function, Torój 2016 eq. 19; default `kind="sf"`).
  `kind="pdf"` uses the gamma density instead (the 2021 working paper refers to
  "gamma-PDF parameters"), which needs positive internal distances. See §8, item 9.
* Column p gives the shares of region p's demand for s supplied by each region r.
* m^s_r is the supply mass: the region's share of sector-s value added by default (eq. 20),
  or any exogenous `supply_mass`.
* γ_s: fixed at 1 by default, because the 2025 review describes the Bayesian version as
  having only shape and scale. Options are `fixed0`, a fixed float, or `estimate`
  (γ ∈ (0,1) with a Beta prior).

### 6.3 Likelihood and priors
Stacking all sectors gives A_t va_t = b_t + ε_t, with A_t = I − [β^s_v W_s]. Integrating
out σ²_s under the Jeffreys prior p(σ²) ∝ 1/σ²:

  log p(φ | data) = Σ_t log|A_t| − (n_eff/2) Σ_s log SSR_s + log p(φ),

where φ_s = (log shape, log scale [, logit γ]). n_eff = T(R−1) by default, because the
residuals sum to zero across regions every year (§8, item 11). σ² can be drawn afterwards
from Inv-Gamma(n_eff/2, SSR/2). Torój (2016) prints the log-likelihood without the −½
factors; I use the standard Gaussian form.

Priors (log-normal on shape and scale):
* `weak` (default): log shape ~ N(0, 1); log scale ~ N(log median distance, 1.5).
* `gravity`: centred on gamma decays fitted to the ICIO δ_s. For each sector, (shape,
  scale) is chosen so that log g(d) best matches −δ_s ln d (up to a constant) over the
  Canadian distances (`fit_decay_to_power_law`), with sd 0.5 on each log parameter. This
  plays the role of Torój's (2024) Finnish-table prior: outside evidence on spatial decay.
  **It is my construction, not his.**
* `fit_decay_to_shares` fits (shape, scale) to observed origin shares, such as the StatCan
  interprovincial shares. Use it for an informative prior or to validate the SDM's W against
  observed trade.

### 6.4 Sampler
* Metropolis-within-Gibbs with one 2- or 3-parameter block per sector and an adaptive
  Gaussian random walk (Haario-type: 2.38²/P × empirical covariance, adapted only during
  burn-in, so the kept draws are a valid MCMC sample).
* One sector's update changes R rows of A_t. log|A_t| is updated with the matrix
  determinant lemma, det(A + EΔ) = det A · det(I + Δ A⁻¹ E), and A⁻¹ with Woodbury.
  Each proposal therefore costs O(R²·RS) rather than O((RS)³). A full refresh runs every 50
  sweeps. The update is exact to 1e-14 (tested).
* Starting point: posterior mode from block-coordinate Nelder–Mead (`fit_mode`); with a flat
  prior this is the ML estimator of Torój 2016/2021. Several dispersed chains; split-R̂ is
  reported.
* Runtime: about 0.11 s per sweep at 50 sectors × 13 provinces × 3 years, so 4 chains ×
  4,000 iterations takes about 30 minutes.

### 6.5 Flows
Z[r,s,p,v] = colZ[s,p,v] · W_s(r,p) (Torój 2016 eqs. 4, 6). By default
colZ = Z^{sv}_national · va^v_p / VA^v; you can also supply column totals, such as
provincial domestic use from the SUT. The posterior mean, sd and 5%/95% quantiles are
computed over draws (`posterior_flows`). `balance_flows` RAS-balances the result to regional
row and column targets (Torój 2025, Sect. 4–5). **Note:** W is indexed by the supplying
sector only, so within each (s,p) the SDM also splits across demanding sectors v in
proportion to colZ. This is the same structural limit described in §3.2.

---

## 7. Validation

`pytest` (15 tests) on known-parameter synthetic data (`synthetic.py`) checks:
* PPML recovers δ and the home coefficient; dropping separated groups works.
* Pair-level δ_{k,m} estimates fall within 4 cluster-robust SEs of the truth; z and F give
  identical δ on noiseless data.
* CEPII alignment works for CN1/CN2 and ROW.
* RAS reproduces the margins; gravity-RAS equals IPA when δ is constant across users;
  gravity-RAS recovers the true Z when the data follow the gravity DGP (and IPA does not).
* SDM: the incremental determinant equals the full computation; the mode recovers W
  (max error < 0.05); posterior flows add back to the national table; the γ-estimation path
  runs; prior builders work; balancing works.
* In a separate SDM run, all true shape and scale values fell inside the 90% posterior
  intervals, with R̂ ≈ 1.00–1.02.

Demos: `python canada_disaggregation.py --demo` and `python spatial_irio.py --demo`.

---

## 8. Uncertainties and decisions for you to confirm

1. **Cai (2023) full text was not accessible** (paywalled). The abstract confirms gravity-RAS /
   doubly constrained gravity with an exogenous distance elasticity estimated from
   international data. The exact estimating equation (F vs z, FE structure, controls, which
   distance measure, whether domestic flows are included) is **not verified**. The defaults
   follow your existing code (z, domestic flows kept, distwces) plus standard practice (PPML
   with exporter-year and importer-year FE, a home dummy).
2. **Torój (2024, ESR) full text was not accessible.** The SDM follows Torój (2016, open
   access) for the equations, plus what the 2025 CEJEME review and the 2021/22 working paper
   abstracts say about the Bayesian version (gamma shape and scale per sector, Finnish-table
   prior, MCMC, convergence problems when shape and scale are weakly identified). The
   paper's exact prior families and hyperparameters, its MCMC settings, and whether the
   supply exponent survives are **not verified**.
3. **The 50 supplier-level elasticities alone cannot change the demanding-sector split**
   (§3.2). I read your goal as needing δ by supplier × user and implemented that. If you
   meant something else, such as gravity-estimating T itself, tell me.
4. Sector-level F now **includes final demand** (it was intermediate only in your code, while
   E included FD). Confirm which you intended.
5. **85 regions:** please confirm the region list. I assumed the extended ICIO's CN1/CN2 and
   MX1/MX2 (a split by processing-trade status, not by geography) plus ROW. Those halves are
   mapped to the parent country (internal distance, home = 1). An alternative is to drop
   within-country pairs between the halves.
6. **Transferability:** the elasticities come from international trade (controlling for the
   border with the home dummy). Applying them to provinces assumes that within-country
   distance decay matches international decay. Cai's paper rests on this assumption, but it
   can be checked: estimating δ_j directly from StatCan's T[i,j,b] (a 13×13 PPML per
   commodity) would be a useful comparison.
7. **International imports** are allocated to users by IPA (the default). User-specific
   import shares from the ICIO CAN block would be better and are supported (`dom_share`).
8. **Concordances** from SUPC/NAICS to ICIO are needed. The code averages elasticities rather
   than decay functions, which is a first-order approximation.
9. **Gamma form:** survival function (Torój 2016) by default; the pdf is available. The 2021
   working paper says "gamma-PDF", the 2025 review says the CDF. Which one the 2024 paper
   uses is not verified.
10. **Supply exponent γ:** fixed at 1 by default. The 2016 paper estimates it; the 2025 review
    lists only shape and scale.
11. **The SDM likelihood is approximate:** residuals sum to exactly zero across regions
    (because of the national identity), so the Gaussian density is singular in one direction
    per sector-year. n_eff = T(R−1) is a pragmatic correction. The default supply mass is
    va itself, which makes W mildly endogenous (as in Torój 2016); pass an exogenous mass
    (employment, lagged va) to avoid this.
12. Panel δ is constant over 1995–2022. Run `by_year=True` to check drift.
13. Regional final demand y for the SDM: Torój allocates it by population, which is a
    strong assumption for investment and exports. The provincial SUT gives this directly
    for Canada.

---

## 9. References
* Cai, M. (2023). A calibrated gravity model of interregional trade. *Spatial Economic
  Analysis* 18(1), 89–107. doi:10.1080/17421772.2022.2081715
* Correia, S., Guimarães, P., Zylkin, T. (2020). Fast Poisson estimation with
  high-dimensional fixed effects. *Stata Journal* 20(1), 95–115.
* Head, K., Mayer, T. (2010). Illusory border effects. In *The Gravity Model in
  International Trade*, CUP.
* Mayer, T., Zignago, S. (2011). Notes on CEPII's distances measures: the GeoDist database.
  CEPII WP 2011-25.
* Santos Silva, J., Tenreyro, S. (2006). The log of gravity. *REStat* 88(4), 641–658.
* Torój, A. (2016). Regional economic impact assessment with missing input-output data:
  a spatial econometrics approach for Poland. *CEJEME* 8, 61–91.
* Torój, A. (2021). Construction of multiregion–multisector input–output tables: a spatial
  econometric approach for Poland. *Spatial Economic Analysis* 16(4), 550–569.
* Torój, A. (2024). Estimating high-resolution interregional input–output tables: a
  Bayesian spatial approach. *Economic Systems Research* 36(3), 353–377.
  doi:10.1080/09535314.2024.2358357
* Torój, A. (2025). (Inter)regional input–output table estimation: from surveys to spatial
  econometrics. *CEJEME* 16, 285–322.
