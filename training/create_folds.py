"""Create a deterministic, image-level stratified five-fold APTOS manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random

from preprocessing import load_aptos_records


def assign_stratified_folds(
    records: list[object],
    *,
    number_of_folds: int,
    seed: int,
) -> dict[str, int]:
    by_class: dict[int, list[object]] = defaultdict(list)
    for record in records:
        if record.diagnosis is None:
            raise ValueError("Fold creation requires labelled training records")
        by_class[record.diagnosis].append(record)

    assignments: dict[str, int] = {}
    for diagnosis in sorted(by_class):
        class_records = list(by_class[diagnosis])
        random.Random(seed + diagnosis).shuffle(class_records)
        for position, record in enumerate(class_records):
            assignments[record.image_id] = position % number_of_folds
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default="/workspace/aptos2019-blindness-detection.zip",
    )
    parser.add_argument(
        "--output",
        default="/workspace/project/training/splits/aptos_five_folds.csv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_aptos_records(args.zip, labelled=True)
    assignments = assign_stratified_folds(
        records,
        number_of_folds=args.folds,
        seed=args.seed,
    )
    if len(assignments) != len(records):
        raise ValueError("Image IDs are not unique")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_id", "diagnosis", "fold"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "diagnosis": record.diagnosis,
                    "fold": assignments[record.image_id],
                }
            )

    fold_counts = Counter(assignments.values())
    print(f"records={len(records)}")
    print(f"fold_counts={dict(sorted(fold_counts.items()))}")
    for fold in range(args.folds):
        per_class = Counter(
            record.diagnosis
            for record in records
            if assignments[record.image_id] == fold
        )
        print(f"fold_{fold}_classes={dict(sorted(per_class.items()))}")
    print(f"output={output}")
    print("split_level=image")
    print("patient_wise_split_possible=False")


if __name__ == "__main__":
    main()
