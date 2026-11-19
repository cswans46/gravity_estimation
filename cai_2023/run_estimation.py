"""Run the OECD ICIO industry gravity pipeline."""
import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest
import yaml
from gravity_icio.config import load_config, resolve_path
from gravity_icio.icio import load_icio, build_bilateral_flows
from gravity_icio.agreements import load_agreements
from gravity_icio.merge import merge_controls
from gravity_icio.estimate import estimate_by_industry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = resolve_path(cfg, cfg["paths"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    logdir = resolve_path(cfg, cfg["paths"].get("log_dir", "logs")); logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, cfg.get("logging", {}).get("level", "INFO")),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(logdir / "estimation.log", encoding="utf-8"), logging.StreamHandler()])
    log = logging.getLogger("run_estimation")
    log.info("Loading ICIO tables")
    tables = load_icio(cfg)
    if args.inspect_only:
        report = {}
        for key, obj in tables.items():
            report[key] = {"type":type(obj).__name__, "shape":list(obj.shape),
                           "index_sample":[str(x) for x in list(obj.index[:5])] if hasattr(obj,"index") else [],
                           "columns_sample":[str(x) for x in list(obj.columns[:8])] if hasattr(obj,"columns") else [],
                           "dims":list(getattr(obj,"dims",[]))}
        (output / "icio_schema.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)); return
    flow, output_df, absorption_df = build_bilateral_flows(tables, cfg)
    panel = flow.merge(output_df, on=["year","origin","industry"], how="left")
    panel = panel.merge(absorption_df, on=["year","industry","destination"], how="left")
    paths = {"distance":resolve_path(cfg,cfg["paths"]["cepii_dir"]) / cfg["cepii"]["distance_file"],
             "geography":resolve_path(cfg,cfg["paths"]["cepii_dir"]) / cfg["cepii"]["geography_file"]}
    agr_path = resolve_path(cfg,cfg["paths"]["rta_dta"])
    agr = load_agreements(agr_path,cfg["agreements"],int(cfg["agreements"].get("chunk_size",500000)))
    panel = merge_controls(panel,cfg,paths,agr)
    panel["origin_year"] = panel["origin"].astype(str) + "_" + panel["year"].astype(str)
    panel["destination_year"] = panel["destination"].astype(str) + "_" + panel["year"].astype(str)
    panel["pair"] = panel["origin"].astype(str) + "_" + panel["destination"].astype(str)
    panel.to_parquet(output / "estimation_panel.parquet", index=False)
    flow.to_parquet(output / "bilateral_industry_flows.parquet", index=False)
    aggregation = {
        "observations": ("F", "size"),
        "zero_flows": ("F", lambda x: int((x == 0).sum())),\n        "negative_flows": ("F", lambda x: int((x < 0).sum())),
        "positive_flows": ("F", lambda x: int((x > 0).sum())),
        "distance_missing": ("distance", lambda x: int(x.isna().sum())),
        "cepii_unmatched": ("_cepii_merge", lambda x: int((x != "both").sum())),
        "agreement_unmatched": ("_agreement_merge", lambda x: int((x != "both").sum())),
    }
    for variable in ("rta", "tariff"):
        if variable in panel:
            aggregation[f"{variable}_missing"] = (variable, lambda x: int(x.isna().sum()))
    diagnostics = panel.groupby("industry").agg(**aggregation).reset_index()
    diagnostics.to_csv(output / "sample_diagnostics.csv", index=False)
    estimates = estimate_by_industry(panel, cfg["estimation"])
    estimates.to_csv(output / "estimates.csv", index=False)
    for estimator in ("OLS", "PPML"):
        estimates.loc[estimates["estimator"] == estimator].to_csv(
            output / f"estimates_{estimator.lower()}.csv", index=False
        )
    metadata = {"run_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
                "numpy":np.__version__,"pandas":pd.__version__,"pyfixest":getattr(pyfixest,"__version__","unknown"),
                "config":{k:v for k,v in cfg.items() if not k.startswith("_")},
                "n_rows":len(panel),"industries":sorted(panel.industry.astype(str).unique().tolist()),
                "years":sorted(pd.to_numeric(panel.year).dropna().astype(int).unique().tolist())}
    (output / "run_metadata.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    log.info("Completed. Outputs in %s",output)


if __name__ == "__main__":
    main()
