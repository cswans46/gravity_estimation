"""Industry-by-industry log-OLS and PPML gravity estimators."""
import logging

import numpy as np
import pandas as pd
import pyfixest as pf

LOGGER = logging.getLogger(__name__)


def _model_specifications(frame, settings):
    base = ["log_distance"]
    if settings.get("include_border", True):
        base.append("border")
    if settings.get("include_language", True) and frame["language"].notna().any():
        base.append("language")

    policy = list(base)
    if settings.get("include_rta", True) and "rta" in frame and frame["rta"].notna().any():
        policy.append("rta")
    if settings.get("include_tariff", True) and "tariff" in frame and frame["tariff"].notna().any():
        policy.append("tariff")

    specs = {"baseline": base}
    if policy != base:
        specs["with_policy_controls"] = policy
    return specs


def _append_tidy_results(results, fit, industry, estimator, specification, nobs, **extra):
    tidy = fit.tidy()
    for _, row in tidy.iterrows():
        results.append({
            "industry": industry,
            "estimator": estimator,
            "specification": specification,
            "term": str(row.name),
            "estimate": row.get("Estimate", row.get("estimate")),
            "std_error": row.get("Std. Error", row.get("std_error")),
            "nobs": nobs,
            "status": "ok",
            **extra,
        })


def estimate_by_industry(panel, settings):
    results = []
    min_n = int(settings.get("min_observations", 30))
    fixed_effects = "origin_year + destination_year"

    for industry, sample in panel.groupby("industry", sort=True):
        specifications = _model_specifications(sample, settings)

        for specification, terms in specifications.items():
            if settings.get("run_ols", True):
                ols = sample.dropna(
                    subset=["z", "pair", "origin_year", "destination_year"] + terms
                ).copy()
                ols = ols.loc[ols["z"] > 0]
                if len(ols) >= min_n:
                    try:
                        ols["log_z"] = np.log(ols["z"])
                        fit = pf.feols(
                            "log_z ~ " + " + ".join(terms) + " | " + fixed_effects,
                            data=ols,
                            vcov={"CRV1": "pair"},
                        )
                        _append_tidy_results(
                            results, fit, industry, "OLS", specification, len(ols),
                            zero_flows_excluded=int((sample["F"] == 0).sum()),
                        )
                    except Exception as exc:
                        results.append({
                            "industry": industry, "estimator": "OLS",
                            "specification": specification, "term": None,
                            "status": f"failed: {exc}",
                        })
                else:
                    results.append({
                        "industry": industry, "estimator": "OLS",
                        "specification": specification, "status": f"insufficient observations: {len(ols)}",
                    })

            if settings.get("run_ppml", True):
                ppml = sample.dropna(
                    subset=["F", "output", "absorption", "pair", "origin_year", "destination_year"] + terms
                ).copy()
                negative_flows_excluded = int((ppml["F"] < 0).sum())
                ppml = ppml.loc[
                    (ppml["F"] >= 0) & (ppml["output"] > 0) & (ppml["absorption"] > 0)
                ]
                if len(ppml) >= min_n:
                    ppml["log_exposure"] = np.log(ppml["output"] * ppml["absorption"])
                    try:
                        fit = pf.fepois(
                            "F ~ " + " + ".join(terms) + " | " + fixed_effects,
                            data=ppml,
                            offset="log_exposure",
                            vcov={"CRV1": "pair"},
                        )
                        _append_tidy_results(
                            results, fit, industry, "PPML", specification, len(ppml),
                            zero_flows_included=int((ppml["F"] == 0).sum()),
                            negative_flows_excluded=negative_flows_excluded,
                        )
                    except Exception as exc:
                        results.append({
                            "industry": industry, "estimator": "PPML",
                            "specification": specification, "term": None,
                            "status": f"failed: {exc}",
                        })
                else:
                    results.append({
                        "industry": industry, "estimator": "PPML",
                        "specification": specification, "status": f"insufficient observations: {len(ppml)}",
                    })

    return pd.DataFrame(results)
