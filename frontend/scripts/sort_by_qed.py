#!/usr/bin/env python3
"""
Sort a CSV by QED in descending order.

Defaults to reading kaggle_zinc_filtered.csv in the repo root and writes
kaggle_zinc_filtered_sorted.csv alongside it. The script keeps the header
and row count unchanged; rows lacking a QED value are pushed to the end.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort a CSV by the QED column in descending order."
    )
    parser.add_argument(
        "--input",
        default="kaggle_zinc_filtered.csv",
        help="Path to the input CSV (default: kaggle_zinc_filtered.csv).",
    )
    parser.add_argument(
        "--output",
        default="kaggle_zinc_filtered_sorted.csv",
        help="Path to write the sorted CSV (default: kaggle_zinc_filtered_sorted.csv).",
    )
    parser.add_argument(
        "--qed-field",
        default="qed",
        help="Column name for QED (case-insensitive, default: qed).",
    )
    return parser.parse_args()


def resolve_qed_field(fieldnames: Sequence[str], desired: str) -> str:
    mapping = {name.lower(): name for name in fieldnames}
    key = desired.lower()
    if key not in mapping:
        available = ", ".join(fieldnames)
        raise SystemExit(f"QED column '{desired}' not found. Available columns: {available}")
    return mapping[key]


def parse_qed_value(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def sort_rows(rows: List[Dict[str, str]], qed_key: str) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: parse_qed_value(row.get(qed_key, "")),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV is missing a header row.")
        qed_field = resolve_qed_field(reader.fieldnames, args.qed_field)
        rows = list(reader)

    sorted_rows = sort_rows(rows, qed_field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(
        f"Wrote {len(sorted_rows)} rows sorted by '{qed_field}' (desc) to {output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
