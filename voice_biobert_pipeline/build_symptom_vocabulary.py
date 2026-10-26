#!/usr/bin/env python
"""Build the deployable symptom vocabulary from the model's source dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "resources" / "symptom_vocabulary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_json",
        type=Path,
        help=(
            "Path to symptomsDisease246k.json from "
            "fhai50032/SymptomsDisease246k."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source_json.expanduser().resolve()
    rows = json.loads(source_path.read_text(encoding="utf-8"))

    counts: Counter[str] = Counter()
    for row in rows:
        query = str(row["query"])
        counts.update(
            phrase.strip().lower()
            for phrase in query.split(",")
            if phrase.strip()
        )

    output = {
        "schema_version": 1,
        "generated_on": date.today().isoformat(),
        "source_dataset": "fhai50032/SymptomsDisease246k",
        "source_file": "symptomsDisease246k.json",
        "source_rows": len(rows),
        "notes": (
            "Comma-delimited symptom tokens from the underlying dataset used to "
            "create sajjadhadi/disease-diagnosis-dataset."
        ),
        "symptoms": [
            {"canonical": symptom, "source_frequency": counts[symptom]}
            for symptom in sorted(counts)
        ],
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(counts)} symptoms to {output_path}")


if __name__ == "__main__":
    main()
