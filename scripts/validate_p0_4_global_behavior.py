#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import ahn_interp as ai
from no_ahn_floor import first_line, generate, run_ruler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/run_3b_gdn.json"
AHN_ON_PATH = ROOT / "results/run_3b_gdn/06_ahn_on_ruler.json"
P0_1_PATH = ROOT / "results/validation/p0_1_nowrite_scope.json"
OUTPUT_PATH = ROOT / "results/validation/p0_4_global_behavior.json"
GLOBAL_CHECKPOINT_PATH = (
    ROOT / "results/validation/p0_4_global_behavior_global_checkpoint.json"
)

N_EXAMPLES = 60
RULER_CONFIG = "16384"
BOOTSTRAP_SAMPLES = 10_000
METRICS = ["substring_match", "exact_match", "f1", "f1_first_line"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def output_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported AHN output type: {type(output)}")


def zero_hook(module, inputs, output):
    if torch.is_tensor(output):
        return torch.zeros_like(output)

    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]), *output[1:])

    if isinstance(output, list):
        return [torch.zeros_like(output[0]), *output[1:]]

    raise TypeError(f"Unsupported AHN output type: {type(output)}")


def observer_hook(layer: int, state: dict):
    def hook(module, inputs, output):
        tensor = output_tensor(output)
        record = state.setdefault(
            str(layer),
            {"calls": 0, "first_max_abs": None},
        )
        record["calls"] += 1

        if record["first_max_abs"] is None:
            record["first_max_abs"] = float(
                tensor.detach().float().abs().max().item()
            )

    return hook


def register_zero_hooks(bundle):
    return [
        bundle.model.model.layers[layer].ahn.register_forward_hook(zero_hook)
        for layer in bundle.ahn_layers
    ]


def register_observers(bundle, state):
    return [
        bundle.model.model.layers[layer].ahn.register_forward_hook(
            observer_hook(layer, state)
        )
        for layer in bundle.ahn_layers
    ]


def remove_hooks(handles):
    for handle in reversed(handles):
        handle.remove()


def placement_metadata(tokenizer, example, sliding_window, num_attn_sinks):
    prompt = example["prompt"]
    gold = example["answer"]
    hit = prompt.find(gold)

    if hit < 0:
        raise RuntimeError("Gold answer is missing from the prompt.")

    n_tokens = len(
        tokenizer(prompt, add_special_tokens=True)["input_ids"]
    )
    needle_pos = len(
        tokenizer(prompt[:hit], add_special_tokens=True)["input_ids"]
    )
    boundary = n_tokens - sliding_window
    in_sink = needle_pos < num_attn_sinks
    evicted = (not in_sink) and needle_pos < boundary

    placement = (
        "in_sink_region"
        if in_sink
        else ("evicted" if evicted else "in_window")
    )

    return {
        "n_tokens": n_tokens,
        "needle_pos": needle_pos,
        "compression_boundary": boundary,
        "eviction_distance": boundary - needle_pos,
        "needle_is_evicted": bool(evicted),
        "placement": placement,
    }


def score_one(model, tokenizer, example, sliding_window, num_attn_sinks, idx):
    prediction, n_tokens = generate(
        model,
        tokenizer,
        example["prompt"],
        example.get("max_new_tokens", 32),
    )
    metadata = placement_metadata(
        tokenizer,
        example,
        sliding_window,
        num_attn_sinks,
    )

    if n_tokens != metadata["n_tokens"]:
        raise RuntimeError(
            f"Token-count mismatch: generation={n_tokens}, "
            f"metadata={metadata['n_tokens']}"
        )

    gold = example["answer"]
    gold_n = ai.normalize_answer(gold)
    prediction_n = ai.normalize_answer(prediction)

    return {
        "idx": idx,
        **metadata,
        "gold": gold,
        "prediction": prediction,
        "substring_match": (
            float(gold_n in prediction_n) if gold_n else 0.0
        ),
        "exact_match": ai.exact_match(prediction, gold),
        "f1": ai.qa_f1_score(prediction, gold),
        "f1_first_line": ai.qa_f1_score(first_line(prediction), gold),
    }


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return None

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)

    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def summarize_pairs(rows, seed):
    summary = {}

    groups = {
        "all": rows,
        "evicted": [
            row for row in rows if row["placement"] == "evicted"
        ],
        "in_window": [
            row for row in rows if row["placement"] == "in_window"
        ],
    }

    for group_number, (group, subset) in enumerate(groups.items()):
        record = {
            "n": len(subset),
            "prediction_changed": sum(
                row["on_prediction"] != row["global_nowrite_prediction"]
                for row in subset
            ),
        }
        record["prediction_change_rate"] = (
            record["prediction_changed"] / len(subset)
            if subset else None
        )

        for metric_number, metric in enumerate(METRICS):
            on = np.asarray(
                [row[f"on_{metric}"] for row in subset],
                dtype=np.float64,
            )
            off = np.asarray(
                [row[f"global_nowrite_{metric}"] for row in subset],
                dtype=np.float64,
            )
            delta = on - off

            record[metric] = {
                "ahn_on_mean": float(on.mean()) if len(on) else None,
                "global_nowrite_mean": (
                    float(off.mean()) if len(off) else None
                ),
                "paired_on_minus_global": (
                    float(delta.mean()) if len(delta) else None
                ),
                "paired_bootstrap_95_ci": bootstrap_ci(
                    delta,
                    seed + 100 * group_number + metric_number,
                ),
                "on_better": int(np.sum(delta > 0)),
                "global_better": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
            }

        summary[group] = record

    return summary


def verify_saved_cohort(tokenizer, cohort, saved_rows, sliding_window, sinks):
    if len(cohort) != len(saved_rows):
        raise RuntimeError(
            f"Cohort size mismatch: fresh={len(cohort)}, "
            f"saved={len(saved_rows)}"
        )

    by_idx = {int(row["idx"]): row for row in saved_rows}

    for idx, example in enumerate(cohort):
        saved = by_idx[idx]
        observed = placement_metadata(
            tokenizer,
            example,
            sliding_window,
            sinks,
        )

        checks = {
            "gold": example["answer"],
            **observed,
        }

        for key, value in checks.items():
            if saved[key] != value:
                raise RuntimeError(
                    f"Cohort mismatch at example {idx}, {key}: "
                    f"saved={saved[key]!r}, fresh={value!r}"
                )

    canonical = [
        {
            "idx": idx,
            "task": example.get("task", "niah"),
            "prompt": example["prompt"],
            "answer": example["answer"],
            "max_new_tokens": example.get("max_new_tokens", 32),
        }
        for idx, example in enumerate(cohort)
    ]

    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    return digest


def build_paired_rows(on_rows, global_rows):
    on_by_idx = {int(row["idx"]): row for row in on_rows}
    global_by_idx = {int(row["idx"]): row for row in global_rows}

    if set(on_by_idx) != set(global_by_idx):
        raise RuntimeError("AHN-ON and GLOBAL NOWRITE example IDs differ.")

    paired = []

    for idx in sorted(on_by_idx):
        on = on_by_idx[idx]
        off = global_by_idx[idx]

        for key in (
            "gold",
            "n_tokens",
            "needle_pos",
            "compression_boundary",
            "eviction_distance",
            "needle_is_evicted",
            "placement",
        ):
            if on[key] != off[key]:
                raise RuntimeError(
                    f"Pair mismatch at example {idx}, {key}: "
                    f"ON={on[key]!r}, GLOBAL={off[key]!r}"
                )

        row = {
            "idx": idx,
            "task": on["task"],
            "n_tokens": on["n_tokens"],
            "needle_pos": on["needle_pos"],
            "compression_boundary": on["compression_boundary"],
            "eviction_distance": on["eviction_distance"],
            "needle_is_evicted": on["needle_is_evicted"],
            "placement": on["placement"],
            "gold": on["gold"],
            "on_prediction": on["prediction"],
            "global_nowrite_prediction": off["prediction"],
        }

        for metric in METRICS:
            row[f"on_{metric}"] = on[metric]
            row[f"global_nowrite_{metric}"] = off[metric]
            row[f"on_minus_global_{metric}"] = (
                on[metric] - off[metric]
            )

        paired.append(row)

    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"{OUTPUT_PATH} already exists. Use --allow-overwrite "
            "only for a deliberate rerun."
        )

    config = ai.load_run_config(str(CONFIG_PATH))
    seed = int(config.get("seed", ai.SEED))
    ai.set_seed(seed)

    saved_on = json.loads(AHN_ON_PATH.read_text())
    on_rows = saved_on["ruler_niah"]["rows"]

    if len(on_rows) != N_EXAMPLES:
        raise RuntimeError(
            f"Expected {N_EXAMPLES} saved AHN-ON rows; "
            f"found {len(on_rows)}."
        )

    sliding_window = int(config["sliding_window"])
    sinks = int(config["num_attn_sinks"])

    print("=== P0-4: ON VS GLOBAL NOWRITE BEHAVIOR ===")
    print("Loading model:", config["model_path"])

    bundle = ai.load_ahn_model(
        config["model_path"],
        attn_implementation=config.get(
            "attn_impl",
            "flash_attention_2",
        ),
        device_map="auto",
        sliding_window=sliding_window,
        num_attn_sinks=sinks,
    )

    if len(bundle.ahn_layers) != 36:
        raise RuntimeError(
            f"Expected 36 AHN layers; found {len(bundle.ahn_layers)}: "
            f"{bundle.ahn_layers}"
        )

    cohort = ai.load_ruler(
        config=RULER_CONFIG,
        split="test",
        n=N_EXAMPLES,
        seed=seed,
    )

    cohort_hash = verify_saved_cohort(
        bundle.tokenizer,
        cohort,
        on_rows,
        sliding_window,
        sinks,
    )

    print("COHORT PREFLIGHT: PASS")
    print("Examples:", len(cohort))
    print("Cohort SHA256:", cohort_hash)
    print("AHN layers:", bundle.ahn_layers)

    global_audit = {}
    zero_handles = register_zero_hooks(bundle)
    observer_handles = register_observers(bundle, global_audit)

    try:
        print("\nRunning GLOBAL NOWRITE on the frozen 60 examples...")
        global_result = run_ruler(
            bundle.model,
            bundle.tokenizer,
            sliding_window,
            N_EXAMPLES,
            RULER_CONFIG,
            seed,
            num_attn_sinks=sinks,
        )
    finally:
        remove_hooks(observer_handles)
        remove_hooks(zero_handles)

    GLOBAL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "global_result": global_result,
                "global_audit": global_audit,
            },
            indent=2,
            default=str,
        ) + "\n"
    )
    print("GLOBAL CHECKPOINT:", GLOBAL_CHECKPOINT_PATH)

    if set(map(int, global_audit)) != set(bundle.ahn_layers):
        raise RuntimeError(
            "Global intervention audit did not observe all 36 AHN layers."
        )

    nonzero_global = {
        layer: record["first_max_abs"]
        for layer, record in global_audit.items()
        if record["first_max_abs"] != 0.0
    }

    if nonzero_global:
        raise RuntimeError(
            f"GLOBAL NOWRITE produced nonzero AHN outputs: "
            f"{nonzero_global}"
        )

    global_rows = global_result["rows"]
    paired_rows = build_paired_rows(on_rows, global_rows)
    summary = summarize_pairs(paired_rows, seed)

    # Independent greedy rerun: two evicted and two in-window examples.
    selected = (
        [row["idx"] for row in paired_rows if row["placement"] == "evicted"][:2]
        + [
            row["idx"]
            for row in paired_rows
            if row["placement"] == "in_window"
        ][:2]
    )

    on_by_idx = {int(row["idx"]): row for row in on_rows}
    global_by_idx = {int(row["idx"]): row for row in global_rows}

    independent = []
    on_audit = {}
    observer_handles = register_observers(bundle, on_audit)

    try:
        for idx in selected:
            fresh = score_one(
                bundle.model,
                bundle.tokenizer,
                cohort[idx],
                sliding_window,
                sinks,
                idx,
            )
            saved = on_by_idx[idx]
            primary_match = (
                fresh["exact_match"] == saved["exact_match"]
                and fresh["substring_match"] == saved["substring_match"]
            )
            independent.append({
                "idx": idx,
                "placement": fresh["placement"],
                "condition": "ahn_on",
                "prediction": fresh["prediction"],
                "saved_prediction": saved["prediction"],
                "raw_prediction_match": (
                    fresh["prediction"] == saved["prediction"]
                ),
                "exact_match_reproduced": (
                    fresh["exact_match"] == saved["exact_match"]
                ),
                "substring_match_reproduced": (
                    fresh["substring_match"] == saved["substring_match"]
                ),
                "match": primary_match,
            })
            ai.free_cuda()
    finally:
        remove_hooks(observer_handles)

    zero_handles = register_zero_hooks(bundle)
    try:
        for idx in selected:
            fresh = score_one(
                bundle.model,
                bundle.tokenizer,
                cohort[idx],
                sliding_window,
                sinks,
                idx,
            )
            saved = global_by_idx[idx]
            primary_match = (
                fresh["exact_match"] == saved["exact_match"]
                and fresh["substring_match"] == saved["substring_match"]
            )
            independent.append({
                "idx": idx,
                "placement": fresh["placement"],
                "condition": "global_nowrite",
                "prediction": fresh["prediction"],
                "saved_prediction": saved["prediction"],
                "raw_prediction_match": (
                    fresh["prediction"] == saved["prediction"]
                ),
                "exact_match_reproduced": (
                    fresh["exact_match"] == saved["exact_match"]
                ),
                "substring_match_reproduced": (
                    fresh["substring_match"] == saved["substring_match"]
                ),
                "match": primary_match,
            })
            ai.free_cuda()
    finally:
        remove_hooks(zero_handles)

    if set(map(int, on_audit)) != set(bundle.ahn_layers):
        raise RuntimeError("AHN-ON audit did not observe all 36 AHN layers.")

    inactive_on = {
        layer: record["first_max_abs"]
        for layer, record in on_audit.items()
        if not record["first_max_abs"]
    }

    if inactive_on:
        raise RuntimeError(
            f"AHN-ON produced zero output at layers: {inactive_on}"
        )

    reproduction_pass = all(row["match"] for row in independent)
    raw_prediction_reproduction_pass = all(
        row["raw_prediction_match"] for row in independent
    )

    reproduction_warning = (
        None
        if reproduction_pass
        else (
            "The independent four-example rerun did not reproduce every "
            "saved binary score. The full 60-example result is retained, "
            "but generated-text differences are not interpreted causally."
        )
    )

    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    model_hashes = {
        path.name: sha256_file(path)
        for path in sorted(model_path.glob("*.safetensors"))
    }

    evicted = summary["evicted"]
    evicted_exact = evicted["exact_match"]
    evicted_changed = evicted["prediction_changed"]

    if evicted_exact["paired_on_minus_global"] != 0:
        if reproduction_pass:
            conclusion = (
                "GLOBAL NOWRITE changed exact-answer performance on the "
                "frozen evicted cohort."
            )
        else:
            conclusion = (
                "An exact-answer difference was observed, but the "
                "independent reproduction check did not pass; the "
                "difference is not interpreted causally."
            )
    else:
        conclusion = (
            "GLOBAL NOWRITE did not change answer-retrieval performance "
            "on the frozen evicted cohort. Raw generated strings may "
            "differ, but wording differences are not attributed causally "
            "unless they reproduce independently."
        )

    payload = {
        "experiment": "P0-4 clean output-level behavioral diagnostic",
        "status": (
                "completed"
                if reproduction_pass
                else "completed_with_reproduction_warning"
            ),
        "generated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "conclusion": conclusion,
        "design": {
            "primary_metric": "exact_match",
            "secondary_metrics": [
                "substring_match",
                "f1",
                "f1_first_line",
                "prediction_string_change",
            ],
            "conditions": [
                "AHN ON",
                "GLOBAL NOWRITE: all 36 AHN outputs zeroed",
            ],
            "cohort": "RULER niah_single_1, config 16384",
            "n_examples": N_EXAMPLES,
            "paired": True,
            "j_lens_used": False,
        },
        "provenance": {
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short"),
            "seed": seed,
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": sha256_file(CONFIG_PATH),
            "source_ahn_on_path": str(AHN_ON_PATH.relative_to(ROOT)),
            "source_ahn_on_sha256": sha256_file(AHN_ON_PATH),
            "p0_1_path": str(P0_1_PATH.relative_to(ROOT)),
            "p0_1_sha256": sha256_file(P0_1_PATH),
            "model_path": str(model_path),
            "model_safetensor_sha256": model_hashes,
            "cohort_sha256": cohort_hash,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "intervention_verification": {
            "ahn_layer_count": len(bundle.ahn_layers),
            "ahn_layers": bundle.ahn_layers,
            "ahn_on_first_output_max_abs": on_audit,
            "global_nowrite_first_output_max_abs": global_audit,
            "all_on_outputs_nonzero": True,
            "all_global_outputs_zero": True,
        },
        "cohort_validation": {
            "matches_saved_ahn_on_artifact": True,
            "n_evicted": summary["evicted"]["n"],
            "n_in_window": summary["in_window"]["n"],
        },
        "summary": summary,
        "independent_reproduction": {
            "selected_examples": selected,
            "rows": independent,
            "passed": reproduction_pass,
            "raw_prediction_passed": raw_prediction_reproduction_pass,
            "warning": reproduction_warning,
        },
        "global_nowrite_run": global_result,
        "paired_rows": paired_rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )

    print("\n===== P0-4 COMPLETE =====")
    print("RESULT:", OUTPUT_PATH)
    print("CONCLUSION:", conclusion)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
