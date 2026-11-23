# Methods: 4-D interregional–intersectoral flows for Canada from gravity and spatial-econometric models

This document records every modelling step in `cai_2023/`. It explains how each source paper
constructs (or does not construct) the 4-dimensional flow matrix Z[origin region, supplying
sector, destination region, using sector], how this project reproduces and extends them, and
how to apply each method to Statistics Canada's interprovincial tables or to provincial
tables alone. Section 10 lists what is still open.

Notation: i, b (or r, s) are origin and destination regions; j, k are the supplying commodity or
sector; u, m are the user in the destination (an industry or a final-demand category); t is the year.
"4-D" means Z[i, j, b, u].

---

## 0. Where each of your points is answered

| Your point | Section |
|---|---|
| 3. How Cai, Torój and Schulte build the 4-D matrix, and recipes for StatCan IPIOs or provincial tables alone | 2, 5 |
| 4. Exactly Cai's gravity model (applied to 50 sectors) | 3 |
| 5. CN1/CN2, MX1/MX2 | 3.3 |
| 6. IPT elasticities from ODBus at NAICS × region level → 4-D | 4, 5.1 |
| 7. What "IPA for international imports" referred to | 8 |
| 8. Concordances | 7 |
| 9. Gamma form | 6.2 |
| 10. Supply exponent (2016 approach) and its data | 6.3 |
| 11. Sensitivities | 6.7 |
| 13. Final demand from provincial SUTs; other regions (method chosen and reported) | 6.5 |
| ODBus columns (`derived_NAICS`) and finer NAICS | 4.1 |
| Where r_home ≈ 530 km came from; exact computation | 6.4 |

---

## 1. Pipeline

| Step | Script | Output |
|---|---|---|
| 1a | `estimation.py` | Cai (2023) eq. (9) on the OECD ICIO 2025: θ_k (50 sectors) and θ_{k,m} (50 × 51 supplier × user pairs) |
| 1b | `canada_elasticities.py` | ODBus / Business Register → activity-weighted sector-pair distances; Cai eq. (9) on StatCan interprovincial flows → θ_k or θ_{k,i} (sector × origin province) |
| — | `make_concordances.py` | SUT commodities/industries → ICIO 2025 |
| 2 | `canada_disaggregation.py` | Z[i,j,b,u] by gravity-RAS, IPA benchmark, Fréchet bounds, Schulte envelope |
| 3 | `spatial_irio.py` | Torój (2024) Bayesian spatial Durbin IRIO → posterior of Z[r,i,s,j] (+ sensitivities) |

Sign convention: Cai reports θ_k < 0 (e.g. −1.69). The code stores both `theta` (Cai's sign)
and `delta = −theta` (positive decay, used in seeds d^−δ).

---

## 2. How each paper builds the 4-D matrix

### 2.1 Cai (2023): a 3-D trade matrix, then IPA at the user level
Cai estimates, product by product, the **origin × destination** trade matrix X_k (n × n regions).
The structural AvW gravity (eq. 1–3) gives x_ijk = (y_ik e_jk / y_k)(t_ijk/(Π_ik P_jk))^{1−σ_k}
with t_ijk = g_k d_ij^{δ_k}. So X_k = ⟨a_k⟩ F_k ⟨b_k⟩ with F_k = [d_ij^{θ_k}] (eq. 4). Given θ_k,
the vectors a_k and b_k follow from the known row totals y_k (regional output) and column totals
e_k (regional expenditure) (eq. 5–6), which is exactly RAS with seed F_k. The border and
common-language terms are constant within a country, so they are absorbed into a and b
(Sect. 3.4). Intra-regional flows get no special treatment.

Cai's output is 3-D (i, k, j). He notes that trade coefficients for a multiregional I–O model
follow "immediately" (Miller & Blair, i.e. Chenery–Moses): Z[i,k,j,u] = (x_ijk / Σ_i x_ijk) · z^j_{k,u}.
**That is import proportionality across users**: every user u in j sources k from origin i in
the same proportion. Cai's method therefore delivers the 4-D matrix under IPA. The new
information is the origin × destination split (T), not the split across users.

### 2.2 Gravity-RAS / IRIOLQ (Jahn 2017, as presented in Torój 2025, Sect. 4)
Intra-regional blocks come first from FLQ. Then the FLQ residual of each national cell,
z_ij − Σ_r z^{rr}_ij (eq. 13), is spread over region pairs r ≠ p with gravity predictions
h^{rp}_ij = ẑ^{rp} (eq. 14), where ẑ^{rp} = c·x_{ir}^{β1} x_{jp}^{β2} d_rp^{β3} adj^{β4}
(Torój 2024, eq. 13). Because the destination mass x_{jp} is the **user's** output, this route
does give user-specific shares across region pairs. It does not impose an observed T, though.
When T is observed (as in Canada), the destination-mass term is a column factor within (j, b)
and RAS removes it. Not implemented here (you asked for Cai's and Torój's methods); it is listed
for completeness.

### 2.3 Torój (2016, 2021, 2024): spatial Durbin IRIO
Column sums of each (i, j) block are allocated in proportion to value added,
Σ_r ẑ^{rs}_ij = (v^s_j / Σ v_j) z_ij (2024 eq. 1), and split across supplying regions with
w_i^{rs}: Z[r,i,s,j] = w_i^{rs} · (v^s_j/Σv_j) z_ij (eq. 1–2). The weights W_i are estimated
from regional value added through the SDM (Sect. 6). Torój says explicitly that **w_i^{s} depends
only on the supplying sector i, not on the demand-side sector j** (2024, Sect. 2). So before
balancing, the SDM's 4-D matrix is also proportional across users within each (i, s). The final
extended RAS (row sums = regional supply potential, column sums, R×R block sums = national z_ij)
breaks the proportionality only through the row adjustment.

### 2.4 Schulte, Jakobs & Pauliuk (2021): quantifying the proportionality assumption
For each product i and importing region s there is an "import matrix" T^{si} (origin regions ×
target sectors) whose row sums (imports by origin) and column sums (use by target sector) are
known (eqs. 1–2). IPA is one point in that set. Schulte et al. draw "extreme" members by
block-wise allocation (origins in order, target sectors in random order, north-west-corner
filling) to obtain an upper-bound estimate of footprint uncertainty. **This is the Canadian
(j, b) problem exactly**, with origins = provinces and targets = users. It is an uncertainty
tool, not an estimator.

### 2.5 What this project adds to obtain a non-proportional 4-D matrix
For a fixed (j, b), the unknown is an origin × user matrix with known rows T[·,j,b] and columns
Udom[j,b,·]. A gravity seed S[i,u] = d(·)^{−δ(·)} moves away from IPA **only if it is not rank one
in (i, u)**. That is the case if
1. the elasticity varies by user, δ_{j,u}: ICIO supplier × user estimates (`cai_pair.csv`); or
2. **the distance varies by user**, d^{(j,u)}_{ib}: the activity-weighted distance from where j's
   producers are in province i to where u's establishments are in province b, built from ODBus /
   Business Register locations (Sect. 4.1). With this, even Cai's supplier-only θ_j (from the ICIO
   or from the IPT) produces a user-specific split; or
3. the elasticity varies by origin and the distance by user (θ_{j,i} with item 2).

Province-pair distances with supplier-only (or origin-only) elasticities reproduce IPA exactly
(tested). As you noted, every method stays inside the margins. For that reason the code also
reports:
* **Fréchet bounds** (`frechet_own_share_bounds`): the exact minimum and maximum share of each
  user's purchases that can come from its own province, given the margins. Every method lies
  inside them (tested).
* **The Schulte envelope** (`schulte_envelope`): the distribution of that share over block-random
  allocations. Because it samples only extreme allocations, it does not necessarily contain IPA
  or the gravity solution; the Fréchet bounds do.

---

## 3. Cai (2023) on the OECD ICIO 2025: exact specification (`cai.py`, `estimation.py`)

### 3.1 What Cai does (confirmed from the paper)
| Element | Cai (2023) | This implementation |
|---|---|---|
| Equation | (9): log z_ijk = ξ_k + θ_k log d_ij + φ_k b_ij + c_k l_ij + μ_ik + ν_jk + ε | identical |
| Dependent variable | z = x/(y e), y = row totals, e = column totals of the product-k trade matrix | identical (row/col totals of the estimation sample) |
| Flows | WIOD world IO table, "add up over users" (intermediate + final) | ICIO 2025: Σ_m Z + FD |
| Domestic flows | included (WIOD has intra-national flows; b_ij varies) | included |
| Distance | CEPII GeoDist `distwces` (internal distance on the diagonal) | identical |
| Controls | border dummy b_ij = 1 if i ≠ j; common official language l_ij | identical (`comlang_off`, set to 1 for i = j) |
| Estimator | "fixed effect estimator", product by product: OLS with exporter and importer FE | OLS with absorbed FE (exact two-way projection) |
| Zeros | dropped | dropped (counted in `n_zero_dropped`) |
| SEs | robust | HC1 (option: clustered by pair) |
| Sample | 27 EU countries (pre-Brexit EU, excluding Croatia), year 2009 | default: all ICIO countries with a CEPII match, all 28 years; `--sample eu27 --year 2009` reproduces Cai's design |
| Products | 11 aggregates | 50 ICIO industries; plus 50 × 51 supplier × user pairs (extension) |

### 3.2 Panel extension
μ_ik and ν_jk become origin×year and destination×year effects; θ_k is common across 1995–2022.
Run single years (`--year`) to check stability. PPML (`--estimator ppml`) is available as a
robustness check; Cai uses OLS.

### 3.3 CN1/CN2 and MX1/MX2
According to the OECD's ICIO 2025 documentation, these splits are **functional, not
geographic**: CN1 = China excluding processing-export activities, CN2 = China processing
exports, MX1 = Mexico excluding global manufacturing, MX2 = Mexico global manufacturing. The
85 entries are 80 economies, ROW and these four splits. By default `estimation.py` aggregates
them to CHN and MEX (`icio.aggregate_regions`), which matches Cai's country-level design and
avoids giving two co-located "regions" a distance. The code refuses to aggregate if CHN or MEX
also carry their own flows, since that would double count. `--keep-splits` keeps them, with the
parent's internal distance and home = 1 between the halves. ROW is dropped (no distance).

### 3.4 Calibration and accuracy (Cai Sect. 2.2, 3.5)
`cai_calibrate(y, e, d, θ)` = RAS(d^θ; y, e). `stpe`, `dtpe` reproduce Cai's STPE and DTPE.
A test confirms Cai's diagnosis: too steep a decay overstates intra-regional trade (DTPE > 0).

---

## 4. Canadian elasticities from interprovincial flows and business locations (`business_locations.py`, `canada_gravity.py`)

### 4.1 From business locations to sector-pair distances
* **ODBus v1.0** (StatCan, about 450,000 records from municipal and provincial open data,
  collected May–Dec 2022). Fields: NAICS code, number of employees, latitude/longitude, CSD,
  province. It is not the whole business population, and coverage follows open-data
  availability, so cities with open business licences are over-represented.
* **Business Register "Businesses by Census Subdivision"** (AAFC from StatCan's BR): counts by
  CSD × NAICS × employment size class, full coverage, no coordinates (join CSD centroids from
  the Census geography files).

Steps:
1. **Read ODBus** (`load_odbus`) with its actual column names: `derived_NAICS` (the harmonised
   code, the default sector, 2-digit in most records), `source_NAICS_primary` /
   `source_NAICS_secondary` (provider codes, often finer), `business_sector` /
   `business_subsector` (provider text), `total_no_employees`, `latitude`, `longitude`,
   `prov_terr`, `CSDUID`, `status`. `..` is read as missing. Records flagged closed or inactive
   in `status` are dropped (`keep_closed=True` keeps them).
2. **NAICS code per record**, in this order (column `naics_source`):
   * `source`: a provider code with ≥ 3 digits that agrees with `derived_NAICS` at the 2-digit
     sector level;
   * `derived`: `derived_NAICS`;
   * `text`: still missing, or coarser than requested → the modal code of records with the same
     `business_sector` + `business_subsector` text that *do* have a code at that level. Applied
     only if the text group has ≥ 5 coded records, the modal code's share is ≥ 60%, and the code
     agrees with any 2-digit code the record already carries. Example from your rows: "Limited
     Groceries" has no `derived_NAICS`, but if other providers' "Limited Groceries" records are
     coded 445120, these records get 445. "General Groceries" with no coded peers stays
     unassigned. The loader reports counts by source (`attrs["naics_source"]`).
   * NAICS 2-digit sectors use the official combinations 31-33, 44-45 and 48-49. `derived_NAICS`
     gives the coarse sectors; `--sector-level naics3`/`naics4` uses the text-learned and source
     codes for finer sectors. Records known only at 2 digits are then spread over the finer
     sectors of the same parent, in proportion to the province's weights of those sectors
     (`expand_coarse`). This keeps location and totals.
2. Employment weights: observed employees, otherwise the median in the province × sector
   (`impute_employees`; `count` makes every establishment weigh 1). BR records get count ×
   size-class midpoint.
3. Benchmark (`benchmark_weights`): rescale so province × sector totals equal an external
   employment total, LFS Table 14-10-0023-01 or SEPH Table 14-10-0202-01. This corrects
   coverage differences *between* provinces and sectors. Within a province only the BR CSD
   counts can correct coverage, so combine the two sources where possible.
4. Aggregate to ~25 km cells with activity-weighted centroids; cell self-distance
   0.67√(area/π) (CEPII convention).
5. CES distances (θ = −1, as in `distwces`):
   d^{(j,u)}_{ib} = (Σ_{k∈i} Σ_{l∈b} w^j_k w^u_l d_kl^{−1})^{−1}.
   Output: `canada_sector_pair_distances.npy` with shape (R, K, R, K).

### 4.2 Estimation (Cai's eq. 9 within Canada)
log z_{ib,k,t} = θ_k log d^{(k)}_{ib} + φ_k b_ib + c_k l_ib + μ_{ik,t} + ν_{bk,t} + ε, where:
* T[i, commodity, b, t] comes from Table 12-10-0101 (detail) or 12-10-0088 (summary). Flows go
  to producing sectors through the supply table (commodity → industry shares, then the
  industry concordance).
* b_ib = 1 for interprovincial flows: the provincial border effect, Cai's border dummy one level down.
* l_ib is common official language: Quebec French, New Brunswick bilingual, all others English.
* The distance is the sector-k origin distance to all users in b. By default users are equally
  weighted (`aggregate_user_distances` takes SUT use shares instead).
* OLS (Cai) or PPML; zero flows dropped for OLS.
* **NAICS × region** (`--slope-by origin`): θ_{k,i}, one elasticity per sector and origin
  province, identified from the other provinces × years. Nunavut is dropped by default
  (sparse flows).
* **Sector level** (`--sector-level`): `naics2` (default) uses the ODBus `derived_NAICS` sectors;
  `naics3`/`naics4` finer; `icio` the 50 ICIO industries. IPT commodity flows are mapped to these
  sectors with the supply table (commodity → producing IOIC industry), then IOIC → NAICS sector
  (`naics_level_concordance`: the NAICS digits embedded in the IOIC code). IOIC aggregates coarser
  than the level (e.g. `BS21E000` at 3 digits) are split by the national activity weights in the
  location data, and the script reports them.

---

## 5. Recipes for Canadian data

### 5.1 Situation A: interprovincial flows (IPT) + provincial SUTs → Z[i, j, b, u]
1. **Margins.** T[i,j,b] comes from 12-10-0101 (own-province supply = domestic supply less
   interprovincial exports). U[j,b,u] (use by industries and final-demand categories) comes from
   the provincial SUT 36-10-0478-01. M[j,b] (international imports by province) comes from the
   supply table.
2. **Domestic use.** Udom = U·(1 − M/ΣU) (international imports by IPA within a province; see
   Sect. 8) or user-specific domestic shares (`dom_share`).
3. **Elasticities.** Pick one: (a) ICIO θ_{k,m} (`--delta-source icio-pair`); (b) ICIO θ_k,
   Cai exactly (`icio-sector`); (c) IPT θ_k or θ_{k,i} (`ipt`, `--ipt-slope origin`).
4. **Distances.** Province pairs (`--distances province`), or ODBus/BR user-specific
   d^{(j,u)}_{ib} (`user-specific`, mapped to commodities × users with `map_distances`).
5. **Gravity-RAS** for every (j, b): S[i,u] = d^{−δ}, RAS to T[·,j,b] and Udom[j,b,·].
6. **Report** IPA, the Fréchet bounds and the Schulte envelope beside the estimate.

### 5.2 Situation B: provincial SUT/IO tables only (no IPT) → Z[i, j, b, u] (Cai's original use case)
1. Domestic supply of j from i to Canada: output − international exports − change in
   inventories (`supply_domestic.csv`). Domestic absorption in b: Σ_u Udom[j,b,u].
2. Cai calibration: T̂[·,j,·] = RAS(d^{−δ_j}; supply, absorption)
   (`--trade-source provincial-tables`).
3. Continue with 5.1 steps 4–6 using T̂.
Provincial SUTs include interprovincial imports and exports in total. They can serve as extra
checks (Σ_{b≠i} T̂ against interprovincial exports), but they are not imposed by plain RAS. Use
a constrained balancing step if you want them imposed.

### 5.3 Spatial Durbin route → Z[r, i, s, j]
1. Sector × province value added (36-10-0402-01 or the SUT), the national symmetric IO table
   (Z, X), and provincial final demand from the SUT (Sect. 6.5).
2. Distances between the most-populated cities (Torój) or CES distances; optional driving times.
3. Prior: Torój elicitation calibrated to home-province supply shares from the IPT (Sect. 6.4).
4. MCMC, then Z[r,i,s,j] = w_i^{rs} · colZ[i,s,j], with colZ = provincial domestic use of i by j
   in s (`column_totals.csv`; better than the VA proportion when provincial SUTs exist).
5. Extended RAS (`balance_three_way`): rows = regional supply potential (`row_totals.csv`),
   columns, blocks = national z_ij.

### 5.4 Statistics Canada sources
| Item | Table |
|---|---|
| Interprovincial and international trade flows, basic prices | 12-10-0101-01 (detail), 12-10-0088-01 (summary) |
| Provincial supply and use tables | 36-10-0478-01 |
| Provincial symmetric IO tables | 15-211-X |
| Provincial GDP by industry (supply mass, SDM v) | 36-10-0402-01 |
| Employment by industry (benchmarks, exogenous mass) | LFS 14-10-0023-01; SEPH 14-10-0202-01 |
| Business locations | ODBus 21-26-0003; BR counts by CSD (open.canada.ca) |

---

## 6. Spatial Durbin IRIO (`sdm.py`, `spatial_irio.py`)

### 6.1 Model (Torój 2024 eq. 3–4)
v^i_t = W_{i,t}(Σ_j β_ij v^j_t + β_if f^i_t) [+ β_if f^{i,dir}_t] + ε^i_t, with
β_ij = (v_i/x_i) a_ij (x_j/v_j) and β_if = v_i/x_i known from the national table. Stacked:
A_t v_t = B_t f_t + ε_t with A_t = I − [β_ij W_i].
f^{dir} (international exports, supplied by the origin province itself rather than through W) is
**my extension**, kept as the default at your request (Canada: `exports_direct.csv` from the
supply side of the provincial SUT / IPT by province of origin). Without it, exports must be inside
f, which W then spreads over supplying regions.

### 6.2 Gamma form (your point 9)
Torój (2024) eq. 5 is the current form:
w_i^{rs} ∝ 1 − P_Γ(d_rs^{1−γ_i} · d̃_rs^{γ_i}; shape_i, scale_i), with columns normalised. P_Γ is
the **gamma CDF**, and d̃ is a second distance measure (driving time, rescaled to the mean of d)
with γ_i ~ U(0,1). The code uses 1 − CDF by default (`kind="cdf"`) and turns on the γ mixture
when `driving_time.csv` is provided. The gamma pdf remains only as a sensitivity option.

### 6.3 Supply exponent (your point 10: 2016 approach)
The 2016 paper multiplies the distance term by (m^i_r)^{ψ_i}, where m^i_r = region r's share of
sector-i value added and ψ_i ∈ [0,1] is estimated jointly (Torój 2016 eq. 20–21; there, by ML).
In the Bayesian framework ψ_i gets a U(0,1) prior, like γ_i, and is sampled
(`--supply-exponent estimate`, the default). The 2024 paper drops the term (equivalent to ψ = 0,
`fixed0`). **Data:** provincial GDP at basic prices by industry, Table 36-10-0402-01 (or the
value-added rows of the provincial SUT). Because this is the model's own dependent variable,
there is a sensitivity with exogenous mass: LFS 14-10-0023-01 or SEPH 14-10-0202-01 employment
by industry, or lagged value added (`supply_mass.csv`).

### 6.4 Likelihood, covariance and prior (Torój 2024 eq. 6–10)
* ε ~ N(0, Σ ⊗ I_R) with a full cross-sector Σ. Inverse-Wishart prior with df = S,
  H_ii = 40% of sector i's value-added variance across regions, prior correlations 0.3 (eq. 9).
  The code integrates Σ out analytically:
  log p(θ|data) = Σ_t log|A_t| − ½(df + n) log|H + E′E| + log p(θ). Torój instead samples Σ by
  random-walk MH; the marginal posterior of the decay parameters is the same.
* Prior on shape and scale (eq. 7–8, `TorojPrior`): shape ~ U(a_min, a_max);
  scale | shape ~ TN(μ_i(shape), σ_i(shape), T_min(shape), T_max(shape)). μ_i(shape) is the
  scale at which the share of the decay profile (on a 0…d_max grid) falling within r_home
  equals the observed mean home-region share of sector i's supply; σ_i comes from the scale
  matching mean + sd. T_min and T_max are the scales beyond which the profile is numerically
  L-shaped or flat (Torój's "indifference" regions).
  * **Where Torój's numbers come from.** The home *shares* (μ̂_i, σ̂_i) come from the Finnish
    regional tables. The radius **51.2 km and d_max = 800 km belong to the geography he estimates
    on, Poland's 73 NUTS-3 regions**: "an average of all the distances from the regional capitals
    to the region's border, measured on the straight lines to the capitals of the neighbouring
    regions". Finnish shares are transferred to Polish regions because both are NUTS-3-size
    regions. (An earlier version of this document wrongly said "51.2 km for Finland".)
  * **Canada.** The home shares come from StatCan's IPT (`home_share_stats`: own-province share of
    each sector's shipments, mean and sd across provinces), so no cross-country transfer is
    needed. r_home must follow the same definition on Canadian geography:
    - **exact** (`--boundaries`, `geography.toroj_r_home`): StatCan's provincial boundary
      shapefile (EPSG:3347). For each province and each neighbour, the straight line from the
      capital toward the neighbour's capital is intersected with the province's own boundary,
      and all capital-to-border distances are averaged (pooled). Pure-numpy shapefile reader and
      EPSG:3347 projection included. Per-pair distances are written to `r_home_pairs.csv`.
    - **approximation** (no boundary file): the border is assumed half-way between the two
      capitals. This is how the earlier figure of ~530 km arose: all 13 regions, with long
      territorial pairs (MB–NU 1,145 km, NT–NU 1,131 km, BC–NT 828 km). On the 10 provinces it
      is ~343 km (~401 km without Prince Edward Island's sea links to NS and NB). The
      approximation overstates the radius for capitals deep inside their province and
      understates it for capitals near a border. **Use the exact version for real runs.**
    - Neighbour lists are hard-coded (`CANADA_NEIGHBOURS`, land borders plus PE's sea links,
      which you may want to drop). `geography.neighbours_from_polygons` derives land adjacency
      from the shapefile instead.
    - Because Canadian provinces are far larger than NUTS-3 regions, r_home is hundreds of km,
      not tens. The prior is a mapping from "share of the decay profile within r_home" to the
      gamma scale, so it is only as good as the radius. Hence the `r_home_x0.5` / `r_home_x2.0`
      sensitivity runs (6.7). On synthetic data they re-allocate about 6% of flows.
    - d_max = the maximum distance in the model's distance matrix (Torój: 800 km for Poland).
  * a_min = 0.01 and a_max = 10 are my choices. Torój computes the bounds numerically and does
    not report them; his posterior shapes are 0.012–0.035.
* Alternative priors: `gravity` (log-normal around gamma profiles fitted to IPT/ICIO elasticities)
  and `weak`.

### 6.5 Final demand (your point 13)
**Canada.** Torój allocates national final demand by population because Polish regional final
demand is not available. For Canada, `final_demand.from_provincial_sut` takes provincial final use
by commodity and category from the SUT (36-10-0478-01) and removes international imports. It puts
international exports by origin province in f^{dir} (6.1), then rescales so that Σ_r f equals
national final output x − Σ_j z. Eq. (3) then aggregates to the national identity, checked by
`national_identity_gap`.

**Regions without StatCan-type data ("other regions").** Chosen method: *learn the allocation
keys on Canada and transfer them* (`fit_fd_keys_on_canada` → `FDKeyModel.apply`).
1. Use only indicators that exist for the target regions: population, GDP, household income,
   employment, and sector output or value added (the last can be commodity-specific, (T,S,R)).
2. For each final-demand category (households, NPISH, government, GFCF, inventories, exports),
   fit, on Canadian provinces and years,
   FU[t,i,r] ~ exp(FE_{t,i} + Σ_k b_k ln indicator_k[t,i,r]) by PPML. The (t,i) fixed effects
   absorb national totals, so the b_k are elasticities of a region's share with respect to each
   indicator. Every subset of up to two indicators is fitted, along with fixed proportional keys
   (b = 1 on one indicator; this includes Torój's population key).
3. Score each candidate by **leave-one-province-out** prediction: fit without province r, predict
   r's final demand from its indicators and the national total, and compute the weighted absolute
   percentage error (WAPE) pooled over provinces.
4. Choose the simplest candidate within 2% of the best WAPE (fewest indicators, fixed keys before
   estimated). The full table (`model.report`) shows every candidate's coefficients and LOO error
   with the chosen one flagged, i.e. the evidence for each category's key.
5. Apply the chosen key and elasticities to the target regions' indicators, allocate each
   category's national total, put exports in f^{dir}, and rescale to national f (as for Canada).

Why this method: it is "the same way as Canada" in an evidence-based sense. The category
structure, the export-by-origin treatment and the key for each category are what the Canadian
data support, and the LOO error tells you how much accuracy to expect for a province whose data
you didn't use. On synthetic data it recovers the true keys (income with elasticity 0.9 for
households, population for government, sector output for exports).
Limits: (i) the elasticities are Canadian; transferring them assumes the target regions' final
demand responds to the indicators the same way; (ii) 13 provinces is a small sample, which is why
at most two indicators are used and the parsimony rule is applied.

**Other inputs for data-poor regions** (same "best available" logic):
* trade margins without IPT: Cai's calibration (5.2) with Canadian IPT or ICIO elasticities;
* column totals without regional use tables: Torój eq. 1 (national coefficients × regional VA);
* home shares for the Torój prior: from Canada's IPT (transfer, as Torój transfers from Finland to
  Poland), with r_home computed on the target geography (6.4).

### 6.6 Sampler
Metropolis-within-Gibbs with one block per sector over (log shape, log scale, logit γ, logit ψ);
adaptive Gaussian random walk during burn-in only, or Torój's fixed proposal variances
(`--proposal toroj`: 5e-5, 25e-5, 10e-5). The rank-R determinant and Woodbury updates are exact
(tested). Diagnostics: split-R̂ and the correlation of log shape with log scale per sector. Torój
(2025) notes that a correlation near −1 means the decay profile is identified even when R̂ for
shape and scale individually is not. The script warns only when R̂ > 1.05 *and* that correlation
is weaker than −0.9. Runtime is about 0.17 s per sweep at 50 sectors × 13 provinces × 3 years (4 chains × 6,000 iterations ≈ 70 minutes).

### 6.7 Sensitivities (your point 11, `sdm_sensitivity.py`, `--sensitivity`)
| Scenario | Change |
|---|---|
| `n_R-1` | n = T(R−1) instead of TR. The national identity makes residuals sum to zero over regions, so the Gaussian density is singular in one direction per sector-year |
| `mass_exogenous` | supply mass from employment or lagged VA instead of current VA (endogeneity) |
| `psi_0` | no supply term (Torój 2024) |
| `psi_1` | ψ = 1 |
| `diag_cov` | diagonal Σ with Jeffreys priors instead of the inverse-Wishart |
| `r_home_x0.5`, `r_home_x2.0` | Torój prior rebuilt with half / double the home radius |
Reported: the own-region supply share by sector, and the share of Z re-allocated relative to the
baseline. On synthetic data, ψ = 0 changes Z the most (about 28–55% of flows re-allocated) and
n = T(R−1) the least (under 0.5%).

---

## 7. Concordances (your point 8, `concordance.py`)

### 7.1 ICIO 2025 side
The 50 ICIO 2025 industries and the ISIC Rev.4 divisions and groups they cover are coded in
`ICIO2025`, following the OECD documentation annex. These include the splits C24A/C24B (ISIC
241 + 2431 vs 242 + 2432) and C301/C302T309. `canonical_icio` reads both label styles (`A01`,
`C17_18`, `D` and `01`, `17T18`, `35`, `97T98`).

### 7.2 NAICS → ICIO
* `data/naics2017_to_icio2025.csv` is a **hand-built bridge** at the NAICS 2- to 5-digit level
  (longest-prefix match; every ICIO industry is covered). It handles the known cross-cutting
  cases: pharmaceuticals 3254 → C21; foundries 33151/33152 → C24A/C24B; computer systems design
  5415 → J62_63; libraries 51912 → R; automotive repair 8111 → G; machinery repair 8113 → C31T33;
  aquaculture 1125 → A03; water/sewage 2213 and waste 562 → E; and others.
* The official NAICS 2017 V3.0 → ISIC Rev.4 concordance is an HTML table on StatCan's site.
  Network access from my workspace is blocked, so I could not download it. Save it as CSV and
  run `validate_bridge(load_official_naics_isic(path))` to list every NAICS class where the
  bridge disagrees. `official_naics_to_icio_weights` gives the official mapping directly (equal
  split over linked ISIC classes, because the table has no value weights).

### 7.3 SUT industries and commodities
* IOIC industry codes embed a NAICS prefix (`BS111A00` → 111, `GS611100` → 6111, `NP813000` → 813).
  `industry_concordance` maps them. Letter aggregates that span several ICIO industries
  (e.g. `BS21E000`, mining except oil and gas) are reported. They are split by ICIO Canada gross
  output if `icio_canada_output.csv` is supplied, otherwise equally; `ioic_overrides.csv` resolves
  them exactly.
* Commodities (SUPC) have no NAICS code. `commodity_to_industry` uses the national supply table:
  the share of each commodity produced by each industry. Composed with the industry concordance,
  this gives commodity → ICIO weights (`make_concordances.py` writes both files).
* Final-demand categories map to the ICIO user `FD`.

---

## 8. Your point 7: what "IPA for international imports" meant

The user split needs each user's **domestic-origin** purchases, Udom[j,b,u], as column totals.
The provincial use table gives each user's *total* use of j (domestic plus imported). The supply
table gives province b's international imports of j only in total, not by user. To get
Udom[j,b,u] the code assumes every user in b has the same international import share of j:
Udom = U · (1 − M_jb / Σ_u U_jbu). This is the same proportionality assumption applied to
*international* imports, one level up from the interprovincial split we are estimating. It means
the interprovincial 4-D estimate inherits the international IPA in its column totals. The
alternative supported by the code (`dom_share`) is user-specific import shares, for example from
the ICIO CAN block, where the OECD differentiates imports by end-use. These would be applied as
Canada-wide user shares and rescaled to each province's import total.

---

## 9. Validation (`pytest`: 39 tests, synthetic data with known parameters, plus your ODBus rows)
* PPML and OLS-HDFE recover parameters; OLS-HDFE equals the dummy-variable regression.
* Cai eq. (9) recovers θ, border and language coefficients in cross-section and panel;
  calibration reproduces the true matrix (STPE = 0); a too-steep θ gives DTPE > 0.
* CN/MX aggregation preserves totals and refuses to double count.
* Concordance: the 50 ICIO codes; label styles; NAICS spot checks; ambiguous aggregates are split.
* Business-location distances: internal < interprovincial; distances vary by user sector;
  benchmarking hits the targets.
* IPT elasticities: δ and the border effect are recovered, OLS and PPML, by origin.
* 4-D: user-specific distances with supplier-only δ recover the true non-IPA Z; Cai calibration
  from provincial tables recovers T; IPA, the truth and gravity all lie within the Fréchet bounds.
* ODBus: the ten rows you supplied plus illustrative coded rows; exact columns, `..` handling,
  text-based refinement, expansion of coarse codes. Shapefile reader (byte-level test file),
  exact Torój radius on known geometry, concave boundaries. Final-demand keys learned on Canada
  recover the true structure.
* SDM: exact incremental determinant (with the IW and diagonal covariances, γ and ψ); recovery of
  W; the Torój prior is finite inside its support and −∞ outside; the 3-margin RAS converges;
  final demand matches national f; the sensitivity runner works.

---

## 10. Open points and what I need from you
1. **CN1/CN2, MX1/MX2.** The OECD documentation describes them as processing / global-manufacturing
   splits (Sect. 3.3), not sub-national regions. If you have a geographic version, send the
   member provinces or states and I'll add sub-national distances (the CES code is ready).
2. **ODBus.** Wired to the actual columns (4.1). Please run `load_odbus` on the full file and look
   at `attrs["naics_source"]` and `n_no_sector`. They show how many records are coded from
   providers, from `derived_NAICS`, from text, or not at all. That decides how fine a sector level
   is credible (`naics2` is safe; `naics3`/`naics4` only where text learning covers most records).
3. **Official NAICS → ISIC file.** Please download it (Sect. 7.2) so the bridge can be validated.
4. **Provincial boundary shapefile** (StatCan boundary file, provinces/territories) for the exact
   Torój radius (6.4). Also decide whether PE's sea links count as neighbours and whether the
   territories are in the SDM.
5. **Indicators for data-poor regions.** Tell me which indicators you will have for them, so the
   Canadian key-learning (6.5) only considers those.
6. **Shape bounds** (a_min = 0.01, a_max = 10) for the Torój prior are my choice; he computes them
   numerically and does not report them.
7. **ODBus coverage.** Distances use where businesses *appear* in open data. Benchmarking fixes
   totals by province × sector but not the within-province geography. Prefer the BR CSD counts
   for that, with ODBus for detail.

---

## References
* Anderson & van Wincoop (2003, 2004); Head & Mayer (2010); Mayer & Zignago (2011) CEPII GeoDist.
* Cai, M. (2023). A calibrated gravity model of interregional trade. *Spatial Economic Analysis* 18(1), 89–107.
* Correia, Guimarães & Zylkin (2020). ppmlhdfe. *Stata Journal* 20(1).
* Jahn, M. (2017). Extending the FLQ formula: a location quotient-based interregional input–output framework. *Regional Studies*.
* OECD (2026). Development of the OECD Inter-Country Input-Output Database 2025 edition (annex).
* Schulte, S., Jakobs, A. & Pauliuk, S. (2021). Relaxing the import proportionality assumption in multi-regional input–output modelling. *Journal of Economic Structures* 10:20.
* Torój, A. (2016). *CEJEME* 8, 61–91; (2021). *Spatial Economic Analysis* 16(4); (2024). *Economic Systems Research* 36(3), 353–377; (2025). *CEJEME* 16, 285–322.
* Statistics Canada tables as listed in 5.4.
