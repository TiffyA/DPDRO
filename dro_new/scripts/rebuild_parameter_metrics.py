#!/usr/bin/env python3
"""Rebuild the parameter sweep CSV files from finished run directories."""

import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_BASE = PROJECT_ROOT / "runs" / "parameter_sweep"

HEADER = [
    "eta",
    "rho",
    "depth",
    "width_multiplier",
    "extra_conv",
    "extra_conv_factor",
    "batch_size",
    "loss_scale",
    "warmup_steps",
    "step",
    "val_acc",
    "train_acc",
    "val_loss",
    "train_loss",
    "output_dir",
    "log_file",
]


def token_to_float_str(token: str) -> str:
    """Restore the original decimal point representation."""

    return token.replace("p", ".")


def parse_run_id(run_name: str) -> dict:
    parts = run_name.split("_")
    values = {}
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "eta":
            values["eta"] = token_to_float_str(parts[i + 1])
            i += 2
        elif part == "rho":
            values["rho"] = token_to_float_str(parts[i + 1])
            i += 2
        elif part == "depth":
            values["depth"] = parts[i + 1]
            i += 2
        elif part == "width":
            values["width_multiplier"] = token_to_float_str(parts[i + 1])
            i += 2
        elif part.startswith("ec"):
            values["extra_conv"] = part[2:]
            i += 1
        elif part.startswith("f"):
            values["extra_conv_factor"] = token_to_float_str(part[1:])
            i += 1
        elif part == "bs":
            values["batch_size"] = parts[i + 1]
            i += 2
        elif part == "ls":
            values["loss_scale"] = token_to_float_str(parts[i + 1])
            i += 2
        elif part == "warm":
            values["warmup_steps"] = parts[i + 1]
            i += 2
        else:
            i += 1
    return values


def collect_rows():
    rows = []
    for run_dir in sorted(OUT_BASE.glob("eta_*")):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics_ascdro.json"
        if not metrics_path.exists():
            continue
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        history = data.get("history") or []
        if not history:
            continue
        final = history[-1]
        params = parse_run_id(run_dir.name)
        if len(params) < 8:
            # Unexpected naming scheme, skip to avoid writing bad rows.
            continue
        row = {
            **params,
            "step": final.get("step"),
            "val_acc": final.get("val_acc"),
            "train_acc": final.get("train_acc"),
            "val_loss": final.get("val_loss"),
            "train_loss": final.get("train_loss"),
            "output_dir": str(run_dir),
            "log_file": str(run_dir / "training.log"),
        }
        required = {
            "eta",
            "rho",
            "depth",
            "width_multiplier",
            "extra_conv",
            "extra_conv_factor",
            "batch_size",
            "loss_scale",
            "warmup_steps",
        }
        if not required.issubset(row):
            continue
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    rows = collect_rows()
    if not rows:
        print("[WARN] No completed runs with metrics found.")
        return 1
    write_csv(OUT_BASE / "para_acc.csv", rows)
    write_csv(OUT_BASE / "results_resnet20.csv", rows)
    print(f"[INFO] Wrote {len(rows)} rows to results_resnet20.csv and para_acc.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
