#!/usr/bin/env python
"""Aggregate ImageNet ASCDRO results and plot test accuracy vs epsilon."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(paths: List[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in paths:
        metrics_file = path / "metrics_ascdro.json"
        if not metrics_file.exists():
            continue
        with metrics_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        history = payload.get("history", [])
        if not history:
            continue
        final_entry = history[-1]
        epsilon = float(payload["epsilon"])
        val_acc = float(final_entry.get("val_acc", final_entry.get("val_acc1", math.nan)))
        if math.isnan(val_acc):
            continue
        record = {
            "epsilon": epsilon,
            "val_acc": val_acc,
            "path": str(path.resolve()),
            "args": payload.get("args", {}),
            "dp_params": payload.get("dp_params", {}),
        }
        records.append(record)
    return records


def discover_run_dirs(base_dirs: List[Path], pattern: Optional[str]) -> List[Path]:
    run_dirs: List[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        if pattern:
            run_dirs.extend(sorted(p for p in base.glob(pattern) if p.is_dir()))
        else:
            run_dirs.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    return run_dirs


def group_records(records: List[Dict[str, object]], group_key: Optional[str]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for rec in records:
        args = rec.get("args", {})
        label = "default"
        if group_key:
            value = args.get(group_key)
            if value is None:
                value = rec.get(group_key)
            label = str(value)
        grouped.setdefault(label, []).append(rec)
    return grouped


def plot_records(
    grouped: Dict[str, List[Dict[str, object]]],
    *,
    output: Path,
    title: str,
    show_legend: bool,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, group in sorted(grouped.items(), key=lambda item: item[0]):
        group_sorted = sorted(group, key=lambda rec: rec["epsilon"])
        epsilons = [rec["epsilon"] for rec in group_sorted]
        accuracies = [rec["val_acc"] for rec in group_sorted]
        plt.plot(epsilons, accuracies, marker="o", label=label)
    plt.xlabel("epsilon")
    plt.ylabel("test accuracy (%)")
    plt.title(title)
    if show_legend and len(grouped) > 1:
        plt.legend(title="group")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ImageNet ASCDRO accuracy vs epsilon")
    parser.add_argument(
        "--base-dir",
        action="append",
        type=Path,
        default=None,
        help="Run directory to scan (can be provided multiple times). Default: dro_new/runs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/metrics_ascdro.json",
        help="Glob pattern relative to each base dir. Default scans recursively for metrics files.",
    )
    parser.add_argument(
        "--group-key",
        type=str,
        default="eta",
        help="Key in args to use for grouping curves (default: eta).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dro_new/plots/imagenet_accuracy.png"),
        help="Output path for the generated plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ImageNet ASCDRO",
        help="Plot title",
    )
    parser.add_argument("--no-legend", action="store_true", help="Disable legend even if multiple groups")
    args = parser.parse_args()

    if args.base_dir is None:
        base_dirs = [Path("dro_new/runs").resolve()]
    else:
        base_dirs = [path.resolve() for path in args.base_dir]

    run_dirs: List[Path] = []
    if args.pattern == "**/metrics_ascdro.json":
        for base in base_dirs:
            run_dirs.extend([metrics.parent for metrics in base.glob(args.pattern) if metrics.is_file()])
    else:
        run_dirs = discover_run_dirs(base_dirs, args.pattern)

    records = load_metrics(run_dirs)
    if not records:
        raise SystemExit("No metrics found to plot.")

    grouped = group_records(records, args.group_key)
    plot_records(grouped, output=args.output.resolve(), title=args.title, show_legend=not args.no_legend)

    summary_path = args.output.with_suffix(".json")
    summary = {}
    for label, group in grouped.items():
        group_sorted = sorted(group, key=lambda rec: rec["epsilon"])
        summary[label] = [{"epsilon": rec["epsilon"], "val_acc": rec["val_acc"], "path": rec["path"]} for rec in group_sorted]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved plot to {args.output} and summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
