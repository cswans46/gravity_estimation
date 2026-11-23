"""Regional final demand f^i_r for the spatial IRIO model.

Torój (2016, 2024) has no regional final demand for Poland and allocates national final
demand by population. For Canada the provincial supply and use tables (Table 36-10-0478-01)
give final use by commodity, category and province, so it is taken directly:

  y_dom[t,i,r]    = sum_c FU[t,i,r,c] * (1 - m[t,i,r,c])    domestic-origin final use located in r
  y_direct[t,i,r] = EX[t,i,r]                               international exports from r (supplied
                                                            by r itself, not via W)

  m = international-import share of the use (IPA within province, or category-specific
  shares if available). Both are then rescaled jointly so that sum_r (y_dom + y_direct)
  = national f_i = x_i - sum_j z_ij, which makes eq. (3) aggregate to the national identity.

"Other regions" (no SUT): ``share_out_like_canada`` reproduces the same structure -- each
final-demand category is allocated with its own regional indicator, exports by origin --
instead of a single population key.
"""
from __future__ import annotations

import numpy as np

# Default key per final-demand category when no regional SUT is available and the keys have
# not been learned on Canada (``fit_fd_keys_on_canada``): the first indicator in each list that
# the target regions have. Order = closeness to the national-accounts concept of the category:
#   households : regional household consumption (e.g. BEA PCE by state) > household disposable
#                income (OECD TL2, Eurostat nama_10r_2hhinc) > population (Torój's key)
#   government : regional government consumption > public-sector employment > population
#   gfcf       : regional GFCF (e.g. Eurostat nama_10r_2gfcf) > GDP/GVA
#   inventories, exports : GVA by sector (by origin) > employment by sector
DEFAULT_INDICATORS = {
    "household": ["hh_consumption", "hh_disposable_income", "population"],
    "npish": ["population"],
    "government": ["gov_consumption", "public_employment", "population"],
    "gfcf": ["gfcf", "va"],
    "inventories": ["va_by_sector", "employment_by_sector", "va"],
    "exports": ["va_by_sector", "employment_by_sector"],
}


def default_keys(available: dict, categories=None) -> dict:
    """{category: indicator array} picking, for each category, the first indicator of
    ``DEFAULT_INDICATORS`` present in ``available`` ({name: (T,R) or (T,S,R)})."""
    out, chosen = {}, {}
    for cat in categories or DEFAULT_INDICATORS:
        for name in DEFAULT_INDICATORS[cat]:
            if name in available:
                out[cat], chosen[cat] = available[name], name
                break
        else:
            raise KeyError(f"no indicator for {cat}: need one of {DEFAULT_INDICATORS[cat]}")
    default_keys.chosen = chosen
    return out


def national_final_output(Xnat, Znat):
    """f_i = x_i - sum_j z_ij (final output of domestic production, incl. exports)."""
    return np.asarray(Xnat, float) - np.asarray(Znat, float).sum(axis=2)


def _rescale(y_dom, y_dir, f_nat):
    tot = y_dom.sum(axis=2) + y_dir.sum(axis=2)
    k = np.divide(f_nat, tot, out=np.ones_like(tot), where=tot != 0)
    return y_dom * k[:, :, None], y_dir * k[:, :, None], k


def from_provincial_sut(final_use, import_share, exports, f_nat, exports_direct: bool = True):
    """final_use (T,S,R,C), import_share (T,S,R) or (T,S,R,C), exports (T,S,R), f_nat (T,S).
    Returns (y_dom, y_direct, scale) with scale (T,S) the consistency factors applied."""
    FU = np.asarray(final_use, float)
    m = np.asarray(import_share, float)
    if m.ndim == 3:
        m = m[..., None]
    y_dom = (FU * (1 - np.clip(m, 0, 1))).sum(axis=3)
    EX = np.asarray(exports, float)
    if exports_direct:
        y_dir = EX
    else:
        y_dom, y_dir = y_dom + EX, np.zeros_like(EX)
    return _rescale(y_dom, y_dir, np.asarray(f_nat, float))


def share_out_like_canada(nat_by_category: dict, indicators: dict, f_nat, exports_key="exports"):
    """Allocate national final demand by category with category-specific regional keys.

    nat_by_category : {category: (T,S) national final demand of domestic output}
    indicators      : {category: (T,R) or (T,S,R) regional key}; exports' key should be
                      sector-specific production (e.g. value added by sector and region)
    Returns (y_dom, y_direct, scale).
    """
    y_dom = y_dir = None
    for cat, nat in nat_by_category.items():
        key = np.asarray(indicators[cat], float)
        if key.ndim == 2:
            key = key[:, None, :]
        sh = key / np.maximum(key.sum(axis=2, keepdims=True), 1e-300)
        part = np.asarray(nat, float)[:, :, None] * sh
        if cat == exports_key:
            y_dir = part if y_dir is None else y_dir + part
        else:
            y_dom = part if y_dom is None else y_dom + part
    if y_dir is None:
        y_dir = np.zeros_like(y_dom)
    return _rescale(y_dom, y_dir, np.asarray(f_nat, float))


# ============================================================================
# Regions without StatCan-type regional tables: keys learned on Canada
# ============================================================================
from dataclasses import dataclass, field  # noqa: E402
from itertools import combinations  # noqa: E402


def _ind_array(ind, T, S, R):
    a = np.asarray(ind, float)
    if a.shape == (T, R):
        a = np.repeat(a[:, None, :], S, axis=1)
    if a.shape != (T, S, R):
        raise ValueError(f"indicator shape {a.shape} must be (T,R) or (T,S,R)")
    return a


def _shares(b, X):
    """X: (T,S,R,K) log indicators -> regional shares (T,S,R) of exp(X b)."""
    eta = X @ b
    eta = eta - eta.max(axis=2, keepdims=True)
    w = np.exp(eta)
    return w / w.sum(axis=2, keepdims=True)


def _fit_key(FU, X, rows=None):
    """PPML of FU[t,i,r] on log indicators with (t,i) fixed effects; rows = provinces used."""
    from .ppml import ppml_hdfe
    T, S, R, K = X.shape
    use = np.arange(R) if rows is None else np.asarray(rows)
    t, i, r = (v.ravel() for v in np.meshgrid(np.arange(T), np.arange(S), use, indexing="ij"))
    res = ppml_hdfe(FU[t, i, r], X[t, i, r], fe=[t * S + i])
    return res.coef


def _wape(pred, act):
    return float(np.abs(pred - act).sum() / max(act.sum(), 1e-300))


@dataclass
class FDKeyModel:
    """Chosen key per final-demand category: indicator names, elasticities, LOO error."""
    keys: dict = field(default_factory=dict)        # category -> (names, coef)
    report: object = None                           # pandas DataFrame of all candidates

    def shares(self, category, indicators, T, S, R):
        names, b = self.keys[category]
        X = np.stack([np.log(np.maximum(_ind_array(indicators[n], T, S, R), 1e-300)) for n in names], axis=-1)
        return _shares(np.asarray(b), X)

    def apply(self, nat_by_category: dict, indicators: dict, f_nat, exports_key="exports"):
        """Allocate national final demand of a data-poor country/region set with the learned keys.
        nat_by_category: {category: (T,S)}; indicators: {name: (T,R) or (T,S,R)} for the target regions.
        Returns (y_dom, y_direct, scale) like ``share_out_like_canada``."""
        first = np.asarray(next(iter(nat_by_category.values())))
        T, S = first.shape
        R = np.asarray(next(iter(indicators.values()))).shape[-1]
        keys = {c: self.shares(c, indicators, T, S, R) for c in nat_by_category}
        return share_out_like_canada(nat_by_category, keys, f_nat, exports_key=exports_key)


def fit_fd_keys_on_canada(final_use: dict, indicators: dict, max_indicators: int = 2,
                          parsimony: float = 0.02) -> FDKeyModel:
    """Learn, from Canada's provincial SUTs, how each final-demand category is distributed
    across regions, so data-poor regions can be "shared out in the same way as Canada".

    final_use  : {category: (T,S,R) provincial final use of domestic output, by commodity}
                 (include 'exports' with international exports by province of origin)
    indicators : {name: (T,R) or (T,S,R)} regional indicators that are also available for the
                 target regions, e.g. population, GDP, household income, employment, sector
                 output/VA (T,S,R) -- only indicators you can obtain elsewhere should be passed.
    For each category, every subset of up to ``max_indicators`` indicators is fitted by PPML
    (FU[t,i,r] ~ exp(FE_{t,i} + sum_k b_k ln ind_k[t,i,r])) and evaluated by leave-one-province-out
    prediction of that province's final demand (weighted absolute percentage error, WAPE,
    given national totals). Fixed proportional keys (b = 1 on one indicator, e.g. Torój's
    population key) are evaluated too. Selection: the simplest candidate (fewest indicators,
    fixed before estimated) whose LOO WAPE is within ``parsimony`` (relative) of the best.
    """
    import pandas as pd
    names = list(indicators)
    rows, model = [], FDKeyModel()
    for cat, FU in final_use.items():
        FU = np.asarray(FU, float)
        T, S, R = FU.shape
        logs = {n: np.log(np.maximum(_ind_array(indicators[n], T, S, R), 1e-300)) for n in names}
        nat = FU.sum(axis=2, keepdims=True)
        cands = [("fixed", (n,), np.array([1.0])) for n in names]
        for k in range(1, max_indicators + 1):
            cands += [("ppml", c, None) for c in combinations(names, k)]
        scored = []
        for kind, combo, b_fixed in cands:
            X = np.stack([logs[n] for n in combo], axis=-1)
            err_num = err_den = 0.0
            for r in range(R):
                b = b_fixed if kind == "fixed" else _fit_key(FU, X, [q for q in range(R) if q != r])
                pred = _shares(b, X)[:, :, r] * nat[:, :, 0]
                err_num += np.abs(pred - FU[:, :, r]).sum()
                err_den += FU[:, :, r].sum()
            wape = err_num / max(err_den, 1e-300)
            b_all = b_fixed if kind == "fixed" else _fit_key(FU, X)
            rows.append(dict(category=cat, key=kind, indicators="+".join(combo),
                             coef=np.round(b_all, 4).tolist(), loo_wape=wape))
            scored.append((wape, len(combo), kind != "fixed", combo, b_all))
        wmin = min(x[0] for x in scored)
        ok = [x for x in scored if x[0] <= wmin * (1 + parsimony)]
        pick = min(ok, key=lambda x: (x[1], x[2], x[0]))
        model.keys[cat] = (list(pick[3]), np.asarray(pick[4]))
        for rrow in rows:
            if rrow["category"] == cat:
                rrow["chosen"] = (rrow["indicators"] == "+".join(pick[3])
                                  and rrow["key"] == ("fixed" if not pick[2] else "ppml"))
    model.report = pd.DataFrame(rows).sort_values(["category", "loo_wape"]).reset_index(drop=True)
    return model
