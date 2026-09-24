from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results/run_3b_gdn"
    / "06c_ruler_cross_term_context_conditioning.json"
)
ANSWER_SOURCE = ROOT / "results/run_3b_gdn/06_ahn_on_ruler.json"
OUTPUT = ROOT / "results/validation/p1_3_c2_statistics.json"

SEED = 20260924
N_BOOTSTRAP = 20_000
N_PERMUTATION = 100_000
TOLERANCE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    raise KeyError(key)


def parse_digits(value: Any, length: int) -> list[int] | None:
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(rf"\d{{{length}}}", text):
            return [int(char) for char in text]

    if isinstance(value, (list, tuple)) and len(value) == length:
        digits = []
        for item in value:
            if isinstance(item, bool):
                return None
            if isinstance(item, (int, np.integer)) and 0 <= int(item) <= 9:
                digits.append(int(item))
            else:
                return None
        return digits

    return None


def candidate_digits(row: dict[str, Any], length: int) -> list[int]:
    preferred = [
        "candidate_digits",
        "answer_digits",
        "digits",
        "candidate_answer",
        "answer",
        "candidate_text",
    ]

    for key in preferred:
        if key in row:
            parsed = parse_digits(row[key], length)
            if parsed is not None:
                return parsed

    for key, value in row.items():
        lowered = key.lower()
        if "candidate" in lowered or "answer" in lowered or "digit" in lowered:
            parsed = parse_digits(value, length)
            if parsed is not None:
                return parsed

    for value in row.values():
        parsed = parse_digits(value, length)
        if parsed is not None:
            return parsed

    raise RuntimeError(
        "Could not recover candidate digits from raw-cross row. "
        f"Available keys: {list(row)}"
    )


def nested_answer_digits(value: Any) -> list[int] | None:
    parsed = parse_digits(value, 7)
    if parsed is not None:
        return parsed

    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = nested_answer_digits(item)
            if parsed is not None:
                return parsed

    return None


def load_saved_answer_map(
    frozen: list[int],
) -> dict[int, list[int]]:
    artifact = json.loads(ANSWER_SOURCE.read_text())
    frozen_set = set(frozen)
    found: dict[int, list[int]] = {}

    index_keys = [
        "idx",
        "index",
        "example",
        "example_idx",
        "example_index",
    ]
    answer_keys = [
        "gold",
        "answer",
        "gold_answer",
        "target",
        "answers",
    ]

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            example_index = None

            for key in index_keys:
                if key not in value:
                    continue

                try:
                    candidate_index = int(value[key])
                except (TypeError, ValueError):
                    continue

                if candidate_index in frozen_set:
                    example_index = candidate_index
                    break

            if example_index is not None:
                for key in answer_keys:
                    if key not in value:
                        continue

                    digits = nested_answer_digits(value[key])

                    if digits is None:
                        continue

                    previous = found.get(example_index)

                    if previous is not None and previous != digits:
                        raise RuntimeError(
                            "Conflicting saved answers for example "
                            f"{example_index}: {previous} versus {digits}"
                        )

                    found[example_index] = digits
                    break

            for child in value.values():
                visit(child)

        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(artifact)

    missing = sorted(frozen_set - set(found))

    if missing:
        raise RuntimeError(
            "Could not recover saved gold answers for examples: "
            f"{missing}"
        )

    return found


def percentile_ci(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    data = json.loads(SOURCE.read_text())

    frozen = [int(value) for value in data["design"]["frozen_examples"]]
    raw = data["raw_cross_tables"]
    saved_folds = np.asarray(data["pair_log_folds"], dtype=float)

    n = len(frozen)
    expected_pairs = n * (n - 1) // 2

    if n != 32:
        raise RuntimeError(f"Expected 32 examples; found {n}.")

    if len(saved_folds) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pair effects; "
            f"found {len(saved_folds)}."
        )

    score_matrix = np.empty((n, n), dtype=float)
    answer_by_candidate = load_saved_answer_map(frozen)

    for column, candidate in enumerate(frozen):
        digits = answer_by_candidate[candidate]

        for row_number, prompt in enumerate(frozen):
            prompt_rows = keyed(raw, prompt)
            row = keyed(prompt_rows, candidate)

            table = np.asarray(row["digit_logprobs"], dtype=float)

            if table.ndim != 2 or table.shape != (len(digits), 10):
                raise RuntimeError(
                    f"Invalid digit table for ({prompt}, {candidate}): "
                    f"{table.shape}"
                )

            score_matrix[row_number, column] = sum(
                table[position, digit]
                for position, digit in enumerate(digits)
            )

    reconstructed_folds = []

    for left, right in itertools.combinations(range(n), 2):
        reconstructed_folds.append(
            score_matrix[left, left]
            - score_matrix[left, right]
            - score_matrix[right, left]
            + score_matrix[right, right]
        )

    reconstructed_folds = np.asarray(reconstructed_folds, dtype=float)
    max_pair_error = float(
        np.max(np.abs(reconstructed_folds - saved_folds))
    )

    observed_log_effect = 0.5 * float(reconstructed_folds.mean())
    observed_effect = math.exp(observed_log_effect)

    recorded_effect = float(
        data["results"]["candidate_conditioned_cross_term_effect"]
    )
    recorded_log_effect = math.log(recorded_effect)

    if max_pair_error > TOLERANCE:
        raise RuntimeError(
            f"Pair reconstruction failed: max error={max_pair_error}"
        )

    if abs(observed_log_effect - recorded_log_effect) > TOLERANCE:
        raise RuntimeError("Recorded corrected effect was not reproduced.")

    # Example-level bootstrap. Resample the 32 examples, not the 496
    # overlapping pairs. H[a,b] is the C2 pair kernel; H[a,a] = 0.
    pair_kernel = np.zeros((n, n), dtype=float)
    cursor = 0

    for left, right in itertools.combinations(range(n), 2):
        value = reconstructed_folds[cursor]
        pair_kernel[left, right] = value
        pair_kernel[right, left] = value
        cursor += 1

    upper = np.triu_indices(n, 1)
    rng = np.random.default_rng(SEED)
    bootstrap_log = np.empty(N_BOOTSTRAP, dtype=float)

    for replicate in range(N_BOOTSTRAP):
        sample = rng.integers(0, n, size=n)
        resampled = pair_kernel[np.ix_(sample, sample)]
        bootstrap_log[replicate] = 0.5 * float(
            resampled[upper].mean()
        )

    bootstrap_effect = np.exp(bootstrap_log)

    # Leave-one-example-out jackknife.
    jackknife_log = np.empty(n, dtype=float)

    for removed in range(n):
        retained = np.delete(np.arange(n), removed)
        retained_kernel = pair_kernel[np.ix_(retained, retained)]
        retained_upper = np.triu_indices(n - 1, 1)
        jackknife_log[removed] = 0.5 * float(
            retained_kernel[retained_upper].mean()
        )

    pseudo_values = (
        n * observed_log_effect
        - (n - 1) * jackknife_log
    )
    jackknife_se = float(
        pseudo_values.std(ddof=1) / math.sqrt(n)
    )

    # Example-identity permutation test using the complete 32×32 score
    # matrix. Candidate identities are permuted as one block.
    row_indices = np.arange(n)
    total_score = float(score_matrix.sum())
    permutation_null = np.empty(N_PERMUTATION, dtype=float)

    for replicate in range(N_PERMUTATION):
        permutation = rng.permutation(n)
        matched_sum = float(
            score_matrix[row_indices, permutation].sum()
        )
        matched_mean = matched_sum / n
        mismatched_mean = (
            total_score - matched_sum
        ) / (n * (n - 1))
        permutation_null[replicate] = (
            matched_mean - mismatched_mean
        )

    matrix_observed_log = float(
        np.diag(score_matrix).mean()
        - (
            score_matrix.sum() - np.trace(score_matrix)
        ) / (n * (n - 1))
    )

    if abs(matrix_observed_log - observed_log_effect) > TOLERANCE:
        raise RuntimeError(
            "Score-matrix statistic does not reproduce C2."
        )

    one_sided_hits = int(
        np.count_nonzero(permutation_null >= observed_log_effect)
    )
    two_sided_hits = int(
        np.count_nonzero(
            np.abs(permutation_null) >= abs(observed_log_effect)
        )
    )

    result = {
        "experiment": "P1-3 C2 statistical-dependence validation",
        "status": "completed",
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256_file(SOURCE),
            "answer_source_path": str(
                ANSWER_SOURCE.relative_to(ROOT)
            ),
            "answer_source_sha256": sha256_file(ANSWER_SOURCE),
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
        },
        "design": {
            "independent_unit": "frozen RULER example",
            "n_independent_examples": n,
            "n_overlapping_unordered_pairs": expected_pairs,
            "pairs_per_example": n - 1,
            "pair_rows_treated_as_independent": False,
            "bootstrap_unit": "example identity",
            "bootstrap_replicates": N_BOOTSTRAP,
            "permutation_unit": "candidate identity",
            "permutation_replicates": N_PERMUTATION,
            "seed": SEED,
        },
        "validation": {
            "pair_count_correct": (
                len(saved_folds) == expected_pairs
            ),
            "max_reconstructed_pair_error": max_pair_error,
            "recorded_effect_reproduced": (
                abs(observed_log_effect - recorded_log_effect)
                <= TOLERANCE
            ),
            "score_matrix_effect_reproduced": (
                abs(matrix_observed_log - observed_log_effect)
                <= TOLERANCE
            ),
            "pairs_dropped": data["results"]["pairs_dropped"],
        },
        "results": {
            "corrected_log_effect": observed_log_effect,
            "corrected_effect": observed_effect,
            "example_bootstrap_log_ci95":
                percentile_ci(bootstrap_log),
            "example_bootstrap_effect_ci95":
                percentile_ci(bootstrap_effect),
            "jackknife_log_standard_error": jackknife_se,
            "permutation_one_sided_p":
                (one_sided_hits + 1) / (N_PERMUTATION + 1),
            "permutation_two_sided_p":
                (two_sided_hits + 1) / (N_PERMUTATION + 1),
            "permutation_one_sided_hits": one_sided_hits,
            "permutation_two_sided_hits": two_sided_hits,
        },
        "interpretation": {
            "valid_primary_statement": (
                "The corrected candidate-conditioned C2 effect is "
                "estimated across 32 frozen examples. Its uncertainty "
                "is calculated by resampling example identities, not by "
                "treating the 496 overlapping pairs as independent."
            ),
            "pseudo_replication_avoided": True,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    print("P1-3 COMPLETE")
    print("RESULT:", OUTPUT)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
