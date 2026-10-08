from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluation import circular_block_sharpe_interval, purged_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit leakage and return uncertainty for point-in-time equity signals.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=2)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"signal_time", "label_end", "net_return"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            parser.error(f"CSV must contain {', '.join(sorted(required))}")
        rows = list(reader)
    if len({r["signal_time"] for r in rows}) != len(rows):
        parser.error("signal_time must be unique")
    folds = purged_walk_forward([r["signal_time"] for r in rows], [r["label_end"] for r in rows], args.splits, args.embargo_days)
    # Score only rows held out by at least one chronological fold.
    oos_indices = sorted({int(i) for fold in folds for i in fold["test"]}, key=lambda i: rows[i]["signal_time"])
    returns = [float(rows[i]["net_return"]) for i in oos_indices]
    report = {
        "input_rows": len(rows),
        "out_of_sample_rows": len(oos_indices),
        "folds": [{"test_start": f["test_start"], "test_end": f["test_end"], "train_rows": len(f["train"]), "test_rows": len(f["test"])} for f in folds],
        "bootstrap": circular_block_sharpe_interval(returns, args.block_length, args.bootstrap),
        "warning": "Fold masks do not refit a model. Supply predictions generated without access to their test labels; net_return must already reflect documented costs.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "fold_indices.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "role", "row_index"])
        for fold_no, fold in enumerate(folds, start=1):
            writer.writerows([[fold_no, "train", int(i)] for i in fold["train"]])
            writer.writerows([[fold_no, "test", int(i)] for i in fold["test"]])
    print(f"Wrote fold and uncertainty audit to {args.out}")
