#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from edge_anpr.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA Jetson Edge ANPR")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = Path(args.config)
    if not path.exists():
        raise SystemExit("Missing config.yaml. Copy config.example.yaml to config.yaml first.")
    with path.open(encoding="utf-8") as handle:
        run(yaml.safe_load(handle))


if __name__ == "__main__":
    main()

