"""Industry-by-industry log-OLS and PPML gravity estimators."""
import logging
import numpy as np
import pandas as pd
import pyfixest as pf

LOGGER = logging.getLogger(__name__)


def _model_terms(frame, settings):
    terms = ["log_distance"]
    if settings.get("include_border", True):
        terms.append("border")
    if settings.get("include_language", True) and frame["language"].notna().any():
        terms.append("language")
    if settings.get("include_rta", True) and "rta" in frame and frame["rta"].notna().any():
        terms.append("rta")
    if settings.get("include_tariff", True) and "tariff" in frame and frame["tariff"].notna().any():
        terms.append("tariff")
    return terms


def estimate_by_industry(panel, settings):
    results = []
    for industry, sample in panel.groupby("industry", sort=True):
        terms = _model_terms(sample, settings)
        fe = "origin_year + destination_year"
        min_n = int(settings.get("min_observations", 30))
        base_cols = ["F", "z", "output", "absorption", "log_distance", "origin_year", "destination_year", "pair"]
        base_cols += [x for x in ("border", "language", "rta", "tariff") if x in sample]
        if settings.get("run_ols", True):
            ols = sample.dropna(subset=["z"] + terms + ["origin_year", "destination_year"]).copy()
            ols = ols.loc[ols["z"] > 0]
            if len(ols) >= min_n:
                try:
                    fit = pf.feols("log_z ~ " + " + ".join(terms) + " | " + fe,
                                   data=ols.assign(log_z=np.log(ols["z"])),
                                   vcov={"CRV1": "pair"})
                    tidy = fit.tidy()
                    for _, row in tidy.iterrows():
                        results.append({"industry": industry, "estimator":"OLS", "term":str(row.name),
                                        "estimate":row.get("Estimate", row.get("estimate")),
                                        "std_error":row.get("Std. Error", row.get("std_error")),
                                        "nobs":len(ols), "zero_flows_excluded":int((sample["F"]==0).sum()), "status":"ok"})
                except Exception as exc:
                    results.append({"industry":industry,"estimator":"OLS","term":None,"status":f"failed: {exc}"})
            else:
                results.append({"industry":industry,"estimator":"OLS","status":f"insufficient observations: {len(ols)}"})
        if settings.get("run_ppml", True):
            ppml = sample.dropna(subset=["F", "output", "absorption"] + terms + ["origin_year", "destination_year"]).copy()
            negative_flows_excluded = int((ppml["F"] < 0).sum())\n            ppml = ppml.loc[(ppml["F"] >= 0) & (ppml["output"] > 0) & (ppml["absorption"] > 0)]
            if len(ppml) >= min_n:
                ppml["log_exposure"] = np.log(ppml["output"] * ppml["absorption"])
                try:
                    fit = pf.fepois("F ~ " + " + ".join(terms) + " | " + fe, data=ppml,
                                    offset="log_exposure", vcov={"CRV1":"pair"})
                    tidy = fit.tidy()
                    for _, row in tidy.iterrows():
                        results.append({"industry":industry,"estimator":"PPML",
                                        "term":str(row.name),
                                        "estimate":row.get("Estimate", row.get("estimate")),
                                        "std_error":row.get("Std. Error", row.get("std_error")),
                                        "nobs":len(ppml),"zero_flows_included":int((ppml["F"]==0).sum()),\n                                        "negative_flows_excluded":negative_flows_excluded,"status":"ok"})
                except Exception as exc:
                    results.append({"industry":industry,"estimator":"PPML","term":None,"status":f"failed: {exc}"})
            else:
                results.append({"industry":industry,"estimator":"PPML","status":f"insufficient observations: {len(ppml)}"})
    return pd.DataFrame(results)
