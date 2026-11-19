# Methodology and implementation notes

## 1. Research objective

Estimate a separate bilateral distance elasticity for each of the 50 ICIO supplying
industries. Estimates use OECD country-level ICIO transactions rather than the Canadian
provincial input-output table that may subsequently be balanced/calibrated. This avoids
using the target Canadian interregional matrix both to estimate and apply the elasticities.

The conceptual reference is Cai (2023), DOI
[10.1080/17421772.2022.2081715](https://doi.org/10.1080/17421772.2022.2081715).

## 2. Bilateral industry flows

Let \(Z_{ik,jm,t}\) be intermediate sales from supplying industry \(k\) in origin \(i\)
to purchasing industry \(m\) in destination \(j\). Let \(FD_{ik,jf,t}\) denote sales
from the same origin-industry to final-demand category \(f\) in destination \(j\).
Construct:

\[
F_{ijk,t}=\sum_m Z_{ik,jm,t}+\sum_f FD_{ik,jf,t}.
\]

Domestic flows (\(i=j\)) are retained. Rest-of-World and aggregate regions are excluded
according to configuration. Intermediate-only flows are retained as a possible
robustness extension; the main flow measure includes final demand because it measures
all sales of an industry's output to the destination economy.

Define origin output and destination absorption:

\[
Y_{ik,t}=X_{ik,t},\qquad
E_{jk,t}=\sum_i F_{ijk,t}.
\]

The normalized bilateral share is:

\[
z_{ijk,t}=\frac{F_{ijk,t}}{Y_{ik,t}E_{jk,t}}.
\]

The normalization follows Cai's derivation: moving origin output and destination
expenditure to the left-hand side of the structural gravity equation leaves origin-
industry and destination-industry multilateral-resistance terms to be represented by
fixed effects.

**Accounting validation:** for each origin-industry-year, row sums of \(F\) should
reconcile to \(X\) within the coverage of included destinations; for each
destination-industry-year, \(E\) must equal the sum of \(F\) over included origins.
Differences caused by excluded ROW/aggregate economies are reported rather than hidden.

## 3. Main econometric equation

For each supplying industry \(k\), the log-linear benchmark is:

\[
\log z_{ijk,t}
=
\theta_k\log d_{ij}
+\phi_k Border_{ij}
+\psi_k Language_{ij}
+\mu_{ik,t}
+\nu_{jk,t}
+\epsilon_{ijk,t}.
\]

Here \(\theta_k\) is the industry-specific distance elasticity. \(d_{ij}\) is the CEPII
population-weighted distance measure (normally \`distwces\`). Border is one for domestic pairs (origin equals destination) and zero for international
pairs. Common official language is a bilateral control. Domestic and international flows
are both included; domestic observations provide border variation and the distance
measure must be positive for domestic pairs.

The proposed extension adds agreement and tariff variables:

\[
\begin{aligned}
\log z_{ijk,t}={}&
\theta_k\log d_{ij}
+\phi_k Border_{ij}
+\psi_k Language_{ij}\\
&+\beta_k RTA_{ij,t}
+\tau_k Tariff_{ij,t}
+\mu_{ik,t}
+\nu_{jk,t}
+\epsilon_{ijk,t}.
\end{aligned}
\]

The fixed effects are origin-industry-by-year and destination-industry-by-year. They
absorb origin output-side and destination expenditure-side multilateral resistance,
as well as country-industry-year shocks. The RTA and tariff variables must be merged
at the appropriate directed country-pair-year level. Direction is important: tariffs
are generally importer-specific and must not be symmetrized unless the source defines
them symmetrically.

Distance and the domestic-border indicator are time invariant. Country-pair fixed
effects are not used in the main model because they would absorb distance and border.
Standard errors are clustered by directed country pair in the default implementation.
Alternative clustering should be considered if the chosen inference design requires
two-way clustering by origin and destination.

## 4. OLS and PPML

### OLS benchmark

Estimate \(\log z\) on positive-flow observations only. This reproduces the log-linear
Cai equation. Report the number and share of zero flows omitted for every industry.
OLS results can be sensitive to selection into positive trade.

### PPML extension

Estimate the conditional mean of bilateral flows:

\[
\mathbb{E}[F_{ijk,t}\mid\cdot]
=
Y_{ik,t}E_{jk,t}
\exp\left(
\theta_k\log d_{ij}
+\phi_k Border_{ij}
+\psi_k Language_{ij}
+\beta_k RTA_{ij,t}
+\tau_k Tariff_{ij,t}
+\mu_{ik,t}
+\nu_{jk,t}
\right).
\]

Equivalently, estimate PPML on \(F\) with \(\log(YE)\) as an offset. This retains zero
flows and corresponds to a multiplicative conditional-mean model for normalized
bilateral shares. It is an extension/robustness estimator, not a claim that Cai's paper
used PPML. Negative bilateral flows, if present in ICIO, are excluded from PPML and
reported separately because the Poisson conditional mean requires a nonnegative outcome.

The implementation estimates industries separately. For each industry-year sample,
the fixed-effect identifiers are origin-year and destination-year. Industries with
insufficient observations, no distance variation, or separation/convergence problems
are reported as failed rather than silently omitted.

## 5. CEPII variables

The default distance variable is \`distwces\`, population-weighted bilateral distance
from CEPII GeoDist. The domestic distance must be positive and should use the CEPII
internal-distance definition. The geography file supplies common-language and, where
available, other bilateral geography controls. The configured origin and destination
ISO fields must match the ICIO country codes. Country code mismatches are listed in
merge diagnostics.

## 6. Trade agreements and tariffs

The Stata file is deliberately read locally. Its schema is inspected before data are
loaded. The report should be used to identify:
- exporter/origin ISO code;
- importer/destination ISO code;
- year;
- agreement indicator or agreement type;
- bilateral tariff measure and its unit (fraction, percent, or ad valorem rate).

The raw file may contain multiple agreements, products, or tariff lines per pair-year.
Aggregation must reflect the meaning of the data. The configured default is maximum
for a binary RTA indicator and mean for tariffs, but these are not universally correct.
The analyst must confirm whether tariffs are weighted, product-specific, preferential,
MFN, or applied rates. Do not interpret a missing tariff as zero without justification.
The data pipeline reports unmatched pairs, missing years, duplicates, and coverage.

## 7. Scope and limitations

- The model is designed for 1995–2022 and 50 industries; actual ICIO coverage is checked.
- ICIO values are monetary flows; currency units cancel in the normalized outcome if
  numerator and denominators are consistently measured.
- Excluding ROW changes output and absorption margins. Diagnostics quantify the
  resulting reconciliation gap.
- The same distance elasticity is ultimately extrapolated to domestic provincial trade.
  This is the maintained Cai assumption, not something identified directly from
  international country-pair variation.
- RTA and tariff controls can improve the interpretation of distance, but incomplete
  coverage or measurement error can change the estimation sample. Report estimates
  both with and without these controls.
- PPML and OLS estimates are not directly interchangeable when zero flows are material.

## 8. Reference

Cai, M. (2023). A calibrated gravity model of interregional trade. *Spatial Economic
Analysis*, 18(1), 89–107. DOI: 10.1080/17421772.2022.2081715.
