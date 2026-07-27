from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

DEFAULT_MANIFEST_PATH = Path("data/metadata/caltech_mvp_subset.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/splits")
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42
DEFAULT_NUM_RESTARTS = 200

SPLITS = ("train", "val", "test")
ALLOWED_CATEGORIES = ("empty", "deer", "coyote", "bobcat", "bird", "opossum")
REQUIRED_COLUMNS = ("image_id", "file_name", "category_name", "location", "seq_id")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationStats:
    location: str
    row_count: int
    category_counts: np.ndarray

    @property
    def category_presence_count(self) -> int:
        return int(np.count_nonzero(self.category_counts))


@dataclass(frozen=True)
class SplitCandidate:
    assignments: dict[str, str]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe grouped Caltech MVP train/val/test splits."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--train-ratio", default=DEFAULT_TRAIN_RATIO, type=float)
    parser.add_argument("--val-ratio", default=DEFAULT_VAL_RATIO, type=float)
    parser.add_argument("--test-ratio", default=DEFAULT_TEST_RATIO, type=float)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--num-restarts", default=DEFAULT_NUM_RESTARTS, type=int)
    return parser.parse_args()


def validate_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> dict[str, float]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError(f"All split ratios must be greater than zero: {ratios}")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")
    return ratios


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input manifest path is not a file: {path}")
    return pd.read_csv(path)


def validate_manifest(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    null_columns = [column for column in REQUIRED_COLUMNS if df[column].isnull().any()]
    if null_columns:
        raise ValueError(
            f"Manifest has null values in required columns: {null_columns}"
        )

    if not df["image_id"].is_unique:
        duplicate_count = int(df["image_id"].duplicated().sum())
        raise ValueError(
            "Manifest image_id values must be unique; "
            f"duplicates: {duplicate_count}"
        )

    if not df["file_name"].is_unique:
        duplicate_count = int(df["file_name"].duplicated().sum())
        raise ValueError(
            "Manifest file_name values must be unique; "
            f"duplicates: {duplicate_count}"
        )

    categories = sorted(str(category) for category in df["category_name"].unique())
    expected = sorted(ALLOWED_CATEGORIES)
    if categories != expected:
        raise ValueError(
            f"Manifest categories must be exactly {expected}; found {categories}"
        )

    unique_location_count = int(df["location"].nunique())
    if unique_location_count < len(SPLITS):
        raise ValueError(
            "Too few unique locations to construct three non-empty grouped splits: "
            f"{unique_location_count}"
        )


def build_location_stats(df: pd.DataFrame) -> list[LocationStats]:
    category_index = {
        category: index for index, category in enumerate(ALLOWED_CATEGORIES)
    }
    stats: list[LocationStats] = []
    for location, location_df in df.groupby("location", sort=True):
        counts = np.zeros(len(ALLOWED_CATEGORIES), dtype=np.float64)
        category_counts = location_df["category_name"].value_counts()
        for category, count in category_counts.items():
            counts[category_index[str(category)]] = float(count)
        stats.append(
            LocationStats(
                location=str(location),
                row_count=int(len(location_df)),
                category_counts=counts,
            )
        )
    return stats


def category_coverage_is_feasible(stats: list[LocationStats]) -> bool:
    coverage = np.zeros(len(ALLOWED_CATEGORIES), dtype=np.int64)
    for item in stats:
        coverage += item.category_counts > 0
    return bool(np.all(coverage >= len(SPLITS)))


def candidate_score(
    split_rows: dict[str, float],
    split_category_counts: dict[str, np.ndarray],
    target_rows: dict[str, float],
    target_category_counts: dict[str, np.ndarray],
    coverage_required: bool,
) -> float:
    total_rows = sum(target_rows.values())
    total_category = np.sum(next(iter(target_category_counts.values()))) / len(SPLITS)
    score = 0.0
    for split in SPLITS:
        row_delta = (split_rows[split] - target_rows[split]) / max(total_rows, 1.0)
        score += row_delta * row_delta

        category_delta = (
            split_category_counts[split] - target_category_counts[split]
        ) / np.maximum(target_category_counts[split], 1.0)
        score += 2.5 * float(np.sum(category_delta * category_delta))

        overflow = max(0.0, split_rows[split] - target_rows[split])
        score += 4.0 * (overflow / max(total_rows, 1.0)) ** 2

        if coverage_required:
            missing_categories = int(np.sum(split_category_counts[split] == 0))
            score += 250.0 * missing_categories

    return float(score + 0.0 * total_category)


def order_locations(
    stats: list[LocationStats], seed: int, restart_index: int
) -> list[LocationStats]:
    rng = np.random.default_rng(seed + restart_index * 9973)
    tie_breakers = {item.location: float(rng.random()) for item in stats}
    return sorted(
        stats,
        key=lambda item: (
            -item.category_presence_count,
            -item.row_count,
            tie_breakers[item.location],
            item.location,
        ),
    )


def greedy_candidate(
    stats: list[LocationStats],
    ratios: dict[str, float],
    seed: int,
    restart_index: int,
    coverage_required: bool,
) -> SplitCandidate:
    total_rows = float(sum(item.row_count for item in stats))
    overall_category_counts = np.sum([item.category_counts for item in stats], axis=0)
    target_rows = {split: total_rows * ratios[split] for split in SPLITS}
    target_category_counts = {
        split: overall_category_counts * ratios[split] for split in SPLITS
    }
    split_rows = {split: 0.0 for split in SPLITS}
    split_category_counts = {
        split: np.zeros(len(ALLOWED_CATEGORIES), dtype=np.float64) for split in SPLITS
    }
    assignments: dict[str, str] = {}

    ordered_stats = order_locations(stats, seed=seed, restart_index=restart_index)
    for index, item in enumerate(ordered_stats):
        best_split: str | None = None
        best_score = float("inf")
        for split in SPLITS:
            candidate_rows = dict(split_rows)
            candidate_category_counts = {
                name: counts.copy() for name, counts in split_category_counts.items()
            }
            candidate_rows[split] += float(item.row_count)
            candidate_category_counts[split] += item.category_counts

            remaining_locations = len(ordered_stats) - index - 1
            empty_splits = sum(1 for name in SPLITS if candidate_rows[name] == 0)
            if remaining_locations < empty_splits:
                continue

            score = candidate_score(
                split_rows=candidate_rows,
                split_category_counts=candidate_category_counts,
                target_rows=target_rows,
                target_category_counts=target_category_counts,
                coverage_required=coverage_required,
            )
            if (score, split) < (best_score, best_split or "zzzz"):
                best_score = score
                best_split = split

        if best_split is None:
            raise ValueError("Unable to keep all three grouped splits non-empty.")
        assignments[item.location] = best_split
        split_rows[best_split] += float(item.row_count)
        split_category_counts[best_split] += item.category_counts

    final_score = candidate_score(
        split_rows=split_rows,
        split_category_counts=split_category_counts,
        target_rows=target_rows,
        target_category_counts=target_category_counts,
        coverage_required=coverage_required,
    )
    return SplitCandidate(assignments=assignments, score=final_score)


def improve_candidate(
    candidate: SplitCandidate,
    stats: list[LocationStats],
    ratios: dict[str, float],
    coverage_required: bool,
) -> SplitCandidate:
    stats_by_location = {item.location: item for item in stats}
    total_rows = float(sum(item.row_count for item in stats))
    overall_category_counts = np.sum([item.category_counts for item in stats], axis=0)
    target_rows = {split: total_rows * ratios[split] for split in SPLITS}
    target_category_counts = {
        split: overall_category_counts * ratios[split] for split in SPLITS
    }
    assignments = dict(candidate.assignments)

    def score_for(assignments_to_score: dict[str, str]) -> float:
        split_rows = {split: 0.0 for split in SPLITS}
        split_category_counts = {
            split: np.zeros(len(ALLOWED_CATEGORIES), dtype=np.float64)
            for split in SPLITS
        }
        for location, split in assignments_to_score.items():
            item = stats_by_location[location]
            split_rows[split] += float(item.row_count)
            split_category_counts[split] += item.category_counts
        return candidate_score(
            split_rows=split_rows,
            split_category_counts=split_category_counts,
            target_rows=target_rows,
            target_category_counts=target_category_counts,
            coverage_required=coverage_required,
        )

    best_score = score_for(assignments)
    for _pass in range(2):
        changed = False
        for location in sorted(assignments):
            original_split = assignments[location]
            if sum(1 for split in assignments.values() if split == original_split) <= 1:
                continue
            for new_split in SPLITS:
                if new_split == original_split:
                    continue
                trial = dict(assignments)
                trial[location] = new_split
                trial_score = score_for(trial)
                if trial_score + 1e-12 < best_score:
                    assignments = trial
                    best_score = trial_score
                    changed = True
                    break
        if not changed:
            break
    return SplitCandidate(assignments=assignments, score=best_score)


def choose_best_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    num_restarts: int,
) -> SplitCandidate:
    if num_restarts < 1:
        raise ValueError(f"num-restarts must be at least 1: {num_restarts}")

    stats = build_location_stats(df)
    coverage_required = category_coverage_is_feasible(stats)
    if not coverage_required:
        category_location_counts = {
            category: int(df.loc[df["category_name"] == category, "location"].nunique())
            for category in ALLOWED_CATEGORIES
        }
        raise ValueError(
            "Full category coverage is not feasible for three grouped splits; "
            "each category must appear in at least three unique locations. "
            f"Category location counts: {category_location_counts}"
        )
    best_candidate: SplitCandidate | None = None

    for restart_index in range(num_restarts):
        candidate = greedy_candidate(
            stats=stats,
            ratios=ratios,
            seed=seed,
            restart_index=restart_index,
            coverage_required=coverage_required,
        )
        candidate = improve_candidate(
            candidate=candidate,
            stats=stats,
            ratios=ratios,
            coverage_required=coverage_required,
        )
        if best_candidate is None or (
            candidate.score,
            sorted(candidate.assignments.items()),
        ) < (
            best_candidate.score,
            sorted(best_candidate.assignments.items()),
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError("Could not construct a valid grouped split candidate.")
    return best_candidate


def apply_split(df: pd.DataFrame, candidate: SplitCandidate) -> dict[str, pd.DataFrame]:
    split_df = df.copy()
    split_df["split"] = split_df["location"].astype(str).map(candidate.assignments)
    if split_df["split"].isnull().any():
        raise ValueError(
            "Internal error: at least one location was not assigned a split."
        )
    result: dict[str, pd.DataFrame] = {}
    sort_columns = ["location", "category_name", "image_id"]
    for split in SPLITS:
        output = split_df[split_df["split"] == split].copy()
        result[split] = output.sort_values(sort_columns, kind="mergesort").reset_index(
            drop=True
        )
    return result


def location_hash(locations: list[str]) -> str:
    payload = json.dumps(locations, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_category_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["category_name"].value_counts().to_dict()
    return {category: int(counts.get(category, 0)) for category in ALLOWED_CATEGORIES}


def validate_outputs(
    input_df: pd.DataFrame, outputs: dict[str, pd.DataFrame]
) -> dict[str, int]:
    combined = pd.concat([outputs[split] for split in SPLITS], ignore_index=True)
    train_locations = set(outputs["train"]["location"])
    val_locations = set(outputs["val"]["location"])
    test_locations = set(outputs["test"]["location"])
    input_ids = set(input_df["image_id"])
    output_ids = list(combined["image_id"])
    output_id_set = set(output_ids)

    checks = {
        "train_val_location_overlap_count": len(train_locations & val_locations),
        "train_test_location_overlap_count": len(train_locations & test_locations),
        "val_test_location_overlap_count": len(val_locations & test_locations),
        "duplicate_image_id_count": len(output_ids) - len(output_id_set),
        "missing_image_id_count": len(input_ids - output_id_set),
        "unexpected_image_id_count": len(output_id_set - input_ids),
    }

    errors: list[str] = []
    empty_splits = [split for split in SPLITS if outputs[split].empty]
    if empty_splits:
        errors.append(f"Output splits must be non-empty: {empty_splits}")
    if len(combined) != len(input_df):
        errors.append(f"Output row count mismatch: {len(combined)} != {len(input_df)}")
    for key, value in checks.items():
        if value:
            errors.append(f"{key} must be zero; found {value}")

    location_split_counts = combined.groupby("location")["split"].nunique()
    split_location_violations = location_split_counts[location_split_counts > 1]
    if not split_location_violations.empty:
        errors.append(
            "Locations assigned to multiple splits: "
            f"{sorted(str(location) for location in split_location_violations.index)}"
        )

    stats = build_location_stats(input_df)
    if category_coverage_is_feasible(stats):
        missing_by_split = {
            split: [
                category
                for category, count in split_category_counts(split_df).items()
                if count == 0
            ]
            for split, split_df in outputs.items()
        }
        missing_by_split = {
            split: categories
            for split, categories in missing_by_split.items()
            if categories
        }
        if missing_by_split:
            errors.append(
                "Could not construct grouped splits with full category coverage: "
                f"{missing_by_split}"
            )

    if errors:
        raise ValueError("; ".join(errors))
    return checks


def build_summary(
    manifest_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    ratios: dict[str, float],
    seed: int,
    num_restarts: int,
    score: float,
    overlap_checks: dict[str, int],
) -> dict[str, Any]:
    total_rows = len(df)
    overall_counts = split_category_counts(df)
    split_summaries: dict[str, Any] = {}
    for split, split_df in outputs.items():
        locations = sorted(str(location) for location in split_df["location"].unique())
        category_counts = split_category_counts(split_df)
        split_summaries[split] = {
            "row_count": int(len(split_df)),
            "actual_row_ratio": float(len(split_df) / total_rows),
            "unique_location_count": int(split_df["location"].nunique()),
            "unique_sequence_count": int(split_df["seq_id"].nunique()),
            "category_counts": category_counts,
            "category_ratios_relative_to_full_dataset": {
                category: (
                    float(category_counts[category] / overall_counts[category])
                    if overall_counts[category]
                    else 0.0
                )
                for category in ALLOWED_CATEGORIES
            },
            "assigned_locations": locations,
            "assigned_locations_sha256": location_hash(locations),
        }

    return {
        "input_manifest_path": str(manifest_path),
        "output_directory": str(output_dir),
        "seed": seed,
        "num_restarts": num_restarts,
        "requested_ratios": ratios,
        "total_input_rows": int(total_rows),
        "total_unique_locations": int(df["location"].nunique()),
        "total_unique_sequences": int(df["seq_id"].nunique()),
        "overall_category_counts": overall_counts,
        "selected_candidate_score": float(score),
        "splits": split_summaries,
        **overlap_checks,
        "validation_passed": True,
    }


def write_outputs(
    outputs: dict[str, pd.DataFrame], summary: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
        "summary": output_dir / "split_summary.json",
    }
    for split in SPLITS:
        outputs[split].to_csv(paths[split], index=False)
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def create_splits(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    num_restarts: int = DEFAULT_NUM_RESTARTS,
) -> dict[str, Any]:
    ratios = validate_ratios(train_ratio, val_ratio, test_ratio)
    df = read_manifest(manifest_path)
    validate_manifest(df)
    LOGGER.info("Read %s input rows from %s", len(df), manifest_path)
    LOGGER.info("Found %s unique locations", df["location"].nunique())
    LOGGER.info("Requested ratios: %s", ratios)
    LOGGER.info("Candidate restarts: %s", num_restarts)

    candidate = choose_best_split(
        df=df,
        ratios=ratios,
        seed=seed,
        num_restarts=num_restarts,
    )
    outputs = apply_split(df, candidate)
    overlap_checks = validate_outputs(df, outputs)
    summary = build_summary(
        manifest_path=manifest_path,
        output_dir=output_dir,
        df=df,
        outputs=outputs,
        ratios=ratios,
        seed=seed,
        num_restarts=num_restarts,
        score=candidate.score,
        overlap_checks=overlap_checks,
    )
    paths = write_outputs(outputs, summary, output_dir)

    LOGGER.info("Selected score: %.6f", candidate.score)
    for split in SPLITS:
        LOGGER.info(
            "%s: %s rows, %s locations, category counts %s",
            split,
            len(outputs[split]),
            outputs[split]["location"].nunique(),
            split_category_counts(outputs[split]),
        )
    LOGGER.info("Location overlap checks: %s", overlap_checks)
    LOGGER.info("Output paths: %s", {key: str(value) for key, value in paths.items()})
    LOGGER.info("Validation passed: true")
    return {"summary": summary, "paths": paths, "outputs": outputs}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        create_splits(
            manifest_path=args.manifest_path,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            num_restarts=args.num_restarts,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
