from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

NO_AHN_PATH = ROOT / "results/run_3b_floor/06_no_ahn_floor.json"
GLOBAL_PATH = ROOT / "results/validation/p0_4_global_behavior.json"
AHN_ON_PATH = ROOT / "results/run_3b_gdn/06_ahn_on_ruler.json"
OUTPUT_PATH = ROOT / "results/validation/p1_5_noahn_vs_nowrite.json"

SEED = 20260924
N_BOOTSTRAP = 20_000

METRICS = [
    "substring_match",
    "exact_match",
    "f1",
    "f1_first_line",
]

COHORT_FIELDS = [
    "idx",
    "task",
    "n_tokens",
    "needle_pos",
    "compression_boundary",
    "eviction_distance",
    "needle_is_evicted",
    "gold",
]


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


def placement(row: dict[str, Any]) -> str:
    return "evicted" if row["needle_is_evicted"] else "in_window"


def indexed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["idx"]): row for row in rows}

    if len(result) != len(rows):
        raise RuntimeError("Duplicate example indices detected.")

    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    groups = {
        "all": rows,
        "evicted": [
            row for row in rows
            if placement(row) == "evicted"
        ],
        "in_window": [
            row for row in rows
            if placement(row) == "in_window"
        ],
    }

    for group_name, group_rows in groups.items():
        result[group_name] = {
            "n": len(group_rows),
        }

        for metric in METRICS:
            result[group_name][metric] = float(np.mean([
                float(row[metric])
                for row in group_rows
            ]))

    return result


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> list[float]:
    n = len(left)
    differences = np.empty(N_BOOTSTRAP, dtype=float)

    for replicate in range(N_BOOTSTRAP):
        sample = rng.integers(0, n, size=n)
        differences[replicate] = float(
            np.mean(left[sample] - right[sample])
        )

    low, high = np.quantile(differences, [0.025, 0.975])
    return [float(low), float(high)]


def exact_mcnemar_p(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    left_better = int(np.sum((left == 1) & (right == 0)))
    right_better = int(np.sum((left == 0) & (right == 1)))
    discordant = left_better + right_better

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(left_better, right_better)
        tail = sum(
            math.comb(discordant, value)
            for value in range(smaller + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)

    return {
        "left_better": left_better,
        "right_better": right_better,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def main() -> None:
    no_ahn = json.loads(NO_AHN_PATH.read_text())
    global_result = json.loads(GLOBAL_PATH.read_text())
    ahn_on = json.loads(AHN_ON_PATH.read_text())

    no_rows = no_ahn["ruler_niah"]["rows"]
    global_rows = global_result["global_nowrite_run"]["rows"]
    on_rows = ahn_on["ruler_niah"]["rows"]

    no_by_idx = indexed(no_rows)
    global_by_idx = indexed(global_rows)
    on_by_idx = indexed(on_rows)

    ids = sorted(no_by_idx)

    if ids != sorted(global_by_idx) or ids != sorted(on_by_idx):
        raise RuntimeError("The three artifacts use different indices.")

    cohort_mismatches = []

    for idx in ids:
        for field in COHORT_FIELDS:
            values = {
                "no_ahn": no_by_idx[idx].get(field),
                "global_nowrite": global_by_idx[idx].get(field),
                "ahn_on": on_by_idx[idx].get(field),
            }

            if len({
                json.dumps(value, sort_keys=True)
                for value in values.values()
            }) != 1:
                cohort_mismatches.append({
                    "idx": idx,
                    "field": field,
                    "values": values,
                })

    if cohort_mismatches:
        raise RuntimeError(
            "Cohort mismatch detected: "
            + json.dumps(cohort_mismatches[:5], indent=2)
        )

    if not global_result[
        "intervention_verification"
    ]["all_global_outputs_zero"]:
        raise RuntimeError(
            "P0-4 did not verify global AHN suppression."
        )

    summaries = {
        "no_ahn": summarize(no_rows),
        "global_nowrite": summarize(global_rows),
        "ahn_on": summarize(on_rows),
    }

    rng = np.random.default_rng(SEED)
    comparisons: dict[str, Any] = {}

    for group_name in ["all", "evicted", "in_window"]:
        group_ids = [
            idx for idx in ids
            if placement(no_by_idx[idx]) == group_name
        ] if group_name != "all" else ids

        comparisons[group_name] = {
            "n": len(group_ids),
            "metrics": {},
        }

        for metric in METRICS:
            global_values = np.asarray([
                float(global_by_idx[idx][metric])
                for idx in group_ids
            ])
            no_values = np.asarray([
                float(no_by_idx[idx][metric])
                for idx in group_ids
            ])

            metric_result = {
                "global_nowrite_mean":
                    float(global_values.mean()),
                "no_ahn_mean":
                    float(no_values.mean()),
                "global_minus_no_ahn":
                    float((global_values - no_values).mean()),
                "paired_bootstrap_95_ci":
                    paired_bootstrap(
                        global_values,
                        no_values,
                        rng,
                    ),
            }

            if metric in {"substring_match", "exact_match"}:
                metric_result["mcnemar"] = exact_mcnemar_p(
                    global_values,
                    no_values,
                )

            comparisons[group_name]["metrics"][metric] = (
                metric_result
            )

    no_config = no_ahn["config"]
    ahn_config = ahn_on["config"]

    result = {
        "experiment":
            "P1-5 no-AHN versus GLOBAL NOWRITE validation",
        "status": "completed",
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "provenance": {
            "git_branch":
                git_value("branch", "--show-current"),
            "git_commit":
                git_value("rev-parse", "HEAD"),
            "no_ahn_path":
                str(NO_AHN_PATH.relative_to(ROOT)),
            "no_ahn_sha256":
                sha256_file(NO_AHN_PATH),
            "global_nowrite_path":
                str(GLOBAL_PATH.relative_to(ROOT)),
            "global_nowrite_sha256":
                sha256_file(GLOBAL_PATH),
            "ahn_on_path":
                str(AHN_ON_PATH.relative_to(ROOT)),
            "ahn_on_sha256":
                sha256_file(AHN_ON_PATH),
            "seed": SEED,
            "bootstrap_replicates": N_BOOTSTRAP,
        },
        "cohort_validation": {
            "exact_row_alignment": True,
            "n_examples": len(ids),
            "n_evicted": sum(
                placement(no_by_idx[idx]) == "evicted"
                for idx in ids
            ),
            "n_in_window": sum(
                placement(no_by_idx[idx]) == "in_window"
                for idx in ids
            ),
            "compared_fields": COHORT_FIELDS,
            "mismatches": [],
        },
        "configuration_comparison": {
            "same_seed":
                no_ahn["seed"] == ahn_on["seed"],
            "same_sliding_window": (
                no_config["sliding_window"]
                == ahn_config["sliding_window"]
            ),
            "same_checkpoint": (
                no_config["model_path"]
                == ahn_config["model_path"]
            ),
            "same_model_cell": (
                no_config["cell"] == ahn_config["cell"]
            ),
            "same_attention_sink_count": (
                no_config["num_attn_sinks"]
                == ahn_config["num_attn_sinks"]
            ),
            "no_ahn": {
                "model_path": no_config["model_path"],
                "cell": no_config["cell"],
                "num_attn_sinks":
                    no_config["num_attn_sinks"],
                "sliding_window":
                    no_config["sliding_window"],
            },
            "global_nowrite": {
                "model_path": ahn_config["model_path"],
                "cell": ahn_config["cell"],
                "num_attn_sinks":
                    ahn_config["num_attn_sinks"],
                "sliding_window":
                    ahn_config["sliding_window"],
                "ahn_outputs_zeroed": 36,
            },
        },
        "condition_summaries": summaries,
        "paired_global_nowrite_vs_no_ahn": comparisons,
        "interpretation": {
            "controls_equivalent": False,
            "global_nowrite_estimand": (
                "Acute intervention within the trained AHN-GDN "
                "checkpoint: retain the checkpoint and attention "
                "configuration while zeroing all 36 AHN outputs."
            ),
            "no_ahn_estimand": (
                "Between-model reference using stock Qwen with no "
                "AHN modules and no attention sinks."
            ),
            "evicted_retrieval_floor_agrees": (
                summaries["no_ahn"]["evicted"][
                    "substring_match"
                ] == 0.0
                and summaries["global_nowrite"]["evicted"][
                    "substring_match"
                ] == 0.0
            ),
            "primary_conclusion": (
                "No-AHN and GLOBAL NOWRITE agree that evicted "
                "answer retrieval is at floor, but they differ "
                "substantially on in-window behavior. They answer "
                "different causal questions and must not be used "
                "interchangeably."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print("P1-5 COMPLETE")
    print("RESULT:", OUTPUT_PATH)
    print(json.dumps({
        "configuration":
            result["configuration_comparison"],
        "evicted_substring":
            comparisons["evicted"]["metrics"][
                "substring_match"
            ],
        "in_window_substring":
            comparisons["in_window"]["metrics"][
                "substring_match"
            ],
        "conclusion":
            result["interpretation"]["primary_conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
