"""Check historical CSV consistency and source preservation using only Python stdlib."""
import ast
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path.relative_to(ROOT)))
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf-8"))
    copied = [item for item in manifest if item["treatment"] == "byte-identical copy"]
    for item in copied:
        actual = hashlib.sha256((ROOT / item["packaged"]).read_bytes()).hexdigest()
        assert actual == item["sha256"], f"Source preservation mismatch: {item['packaged']}"

    expected = set(itertools.product(("ring", "grid", "star", "complete"), ("sync", "async"), (1, 2, 4)))
    short = rows(ROOT / "results/historical_40step/summary.csv")
    long = rows(ROOT / "results/historical_200step/summary_from_transcript.csv")
    for label, table in (("40-step", short), ("200-step", long)):
        actual = {(row["topology"], row["mode"], int(row["nu"])) for row in table}
        assert len(table) == 24 and actual == expected, f"Incomplete {label} configuration set"
        for row in table:
            assert 0 <= float(row["final_acc"]) <= 1
            assert math.isfinite(float(row["final_E"])) and float(row["final_E"]) >= 0
    for summary in short:
        name = f"{summary['topology']}_{summary['mode']}_nu{summary['nu']}.csv"
        run = rows(ROOT / "results/historical_40step/runs" / name)
        assert [int(row["step"]) for row in run] == [0, 10, 20, 30, 39], name
        for row in run:
            assert int(row["samples_processed"]) == (int(row["step"]) + 1) * int(summary["world"]) * 64, name
        end = run[-1]
        assert float(end["train_acc_eval"]) == float(summary["final_acc"]), name
        assert float(end["disagreement_energy"]) == float(summary["final_E"]), name
        assert int(end["samples_processed"]) == int(summary["final_samples"]), name
    print(f"PASS: Python syntax; {len(copied)} preserved files; 24 short-run trajectories / summaries; 24 separate long-run records")
    print("This verifies archived artifacts, not distributed execution or fresh MNIST results.")


if __name__ == "__main__":
    main()
