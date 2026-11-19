from pathlib import Path
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_dir"] = str(path.parent)
    return config


def resolve_path(config, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path
