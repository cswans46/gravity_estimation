"""Inspect a local Stata file's schema without loading the full dataset."""
import argparse
import json
from pathlib import Path
import pyreadstat
from gravity_icio.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = resolve_path(cfg, cfg["paths"]["rta_dta"])
    if not path.exists():
        raise FileNotFoundError(f"Stata file not found: {path}")
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    report = {
        "file": str(path), "row_count": meta.number_rows,
        "column_count": meta.number_columns,
        "columns": [{"name": name, "label": meta.column_names_to_labels.get(name, ""),
                     "type": str(meta.readstat_variable_types.get(name, ""))}
                    for name in meta.column_names],
        "file_label": meta.file_label,
    }
    out = resolve_path(cfg, cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    target = out / "trade_agreement_schema.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSchema report written to: {target}")


if __name__ == "__main__":
    main()
