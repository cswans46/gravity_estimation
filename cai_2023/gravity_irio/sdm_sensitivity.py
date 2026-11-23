"""Sensitivity analysis for the spatial IRIO model (METHODS.md Sect. 6.7).

Each scenario re-estimates the posterior mode (fast) and, optionally, the posterior by MCMC,
and reports (i) the implied own-region supply share by sector and (ii) the share of the
intermediate flow matrix that is re-allocated relative to the baseline.

Default scenarios
  baseline        : Torój 2024 IW covariance, n = T R, supply exponent estimated (2016), mass = va
  n_R-1           : n = T (R-1)   (residuals sum to zero across regions each year)
  mass_exogenous  : supply mass = exogenous indicator (employment / lagged va) instead of va
  psi_0           : no supply term (Torój 2024 specification)
  psi_1           : supply proportional to mass (gamma = 1 in Torój 2016)
  diag_cov        : diagonal Sigma with Jeffreys priors instead of the inverse-Wishart
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .sdm import SpatialIRIO

DEFAULT_SCENARIOS = {
    "baseline": {},
    "n_R-1": {"n_eff": "R-1"},
    "mass_exogenous": {"_mass": "exogenous"},
    "psi_0": {"supply_exponent": "fixed0"},
    "psi_1": {"supply_exponent": "fixed1"},
    "diag_cov": {"covariance": "diag"},
}


def run_sensitivity(data, prior=None, exogenous_mass=None, scenarios=None, t=-1, col_totals=None,
                    mcmc: dict | None = None, base_kw: dict | None = None):
    scenarios = dict(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    if exogenous_mass is None:
        scenarios.pop("mass_exogenous", None)
    base_kw = dict(base_kw or {})
    out_rows, Zs = [], {}
    for name, kw in scenarios.items():
        kw = dict(kw)
        d = data
        if kw.pop("_mass", None) == "exogenous":
            d = dataclasses.replace(data, mass=np.asarray(exogenous_mass, float))
        m = SpatialIRIO(d, prior=prior, **{**base_kw, **kw})
        phi = m.fit_mode()
        if mcmc:
            res = m.sample(init=phi, **mcmc)
            Z = m.posterior_flows(res, t=t, col_totals=col_totals, quantiles=None)["mean"]
        else:
            Z = m.flows(phi, t=t, col_totals=col_totals)
        Zs[name] = Z
        tt = t % d.T
        for s in range(d.S):
            W = m.W(s, phi[s], tt)
            out_rows.append(dict(scenario=name, sector=m.sectors[s], own_share=float(np.diag(W).mean()),
                                 shape=float(np.exp(phi[s, 0])), scale=float(np.exp(phi[s, 1]))))
    df = pd.DataFrame(out_rows)
    base = Zs[next(iter(Zs))]
    realloc = {k: float(0.5 * np.abs(v - base).sum() / base.sum()) for k, v in Zs.items()}
    df["share_reallocated_vs_baseline"] = df["scenario"].map(realloc)
    return df, Zs
