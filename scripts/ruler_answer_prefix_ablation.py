#!/usr/bin/env python3
"""Reproduce experiment 06b: RULER answer-prefix punctuation-control ablation."""

# %% Imports

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import ahn_interp as ai


# %% Experiment configuration

MODEL_PATH = Path("merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN")
LENS_PATH = Path("results/run_3b_gdn/jlens_qwen25_3b_corpusA.pt")
SOURCE_ROWS_PATH = Path("results/run_3b_gdn/04i_ruler_controls_rows.json")

OUTPUT_PATH = Path(
    "results/run_3b_gdn/06b_ruler_answer_prefix_ablation_rerun.json"
)

EXPERIMENT_SOURCE_COMMIT = "032aa5efef69ac3183d10c4ff159dce8439599ef"

EXPECTED_MODEL_HASHES = {
    "model-00001-of-00002.safetensors":
        "639de2c369ed0b383d1b8f4f61db26a371f31be06ebc545d80a2af22e2abba57",
    "model-00002-of-00002.safetensors":
        "0406ceb594144db43fc6826dc285cdaab973c76c00487f32c99eb208dd090dd2",
}

EXPECTED_LENS_HASH = (
    "17af009149b23c0b6d288ea993e555c6bb33a57fae10e4fefe6b07ea31166b2a"
)

EXPECTED_SOURCE_HASH = (
    "356c84e7f31039c65cfcba683e1985dc2b0d268dc0d0e711050f08e42c300aa2"
)

FROZEN_EXAMPLES = [
    2, 7, 12, 14, 16, 21, 23, 24,
    25, 26, 27, 28, 30, 32, 33, 34,
    35, 36, 37, 40, 41, 42, 43, 44,
    45, 49, 50, 51, 52, 55, 56, 59,
]

RULER_CONFIG = "16384"
RULER_SPLIT = "test"
RULER_N = 60

SEED = 20260820
LAYERS = [9, 18, 27]
ANALYSIS_LAYER = 27

SLIDING_WINDOW = 8064
NUM_ATTN_SINKS = 128

NEUTRAL_TOKEN_TEXT = " ."

N_BOOTSTRAP = 2000
N_RANDOMIZATION = 5000


# %% Provenance utilities

def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def current_git_commit() -> str:
    """Return the current repository commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout.strip() if result.returncode == 0 else "unknown"


def model_hashes() -> dict[str, str]:
    """Hash all model safetensor shards."""
    return {
        path.name: sha256_file(path)
        for path in sorted(MODEL_PATH.glob("*.safetensors"))
    }


def verify_provenance() -> dict[str, Any]:
    """Verify the frozen model, lens, and historical RULER artifact."""
    observed_model_hashes = model_hashes()
    observed_lens_hash = sha256_file(LENS_PATH)
    observed_source_hash = sha256_file(SOURCE_ROWS_PATH)

    if observed_model_hashes != EXPECTED_MODEL_HASHES:
        raise RuntimeError(
            f"Model checkpoint hash mismatch:\n{observed_model_hashes}"
        )

    if observed_lens_hash != EXPECTED_LENS_HASH:
        raise RuntimeError("Corpus-A J-Lens hash mismatch.")

    if observed_source_hash != EXPECTED_SOURCE_HASH:
        raise RuntimeError("Historical 04i RULER artifact hash mismatch.")

    return {
        "experiment_source_commit": EXPERIMENT_SOURCE_COMMIT,
        "runtime_git_commit": current_git_commit(),
        "model_safetensor_sha256": observed_model_hashes,
        "corpusA_lens_sha256": observed_lens_hash,
        "source_04i_sha256": observed_source_hash,
    }


# %% RULER cohort utilities

def load_source_artifact() -> dict[str, Any]:
    """Load the historical Run026 RULER rows."""
    with SOURCE_ROWS_PATH.open() as handle:
        return json.load(handle)


def historical_evicted_examples(source: dict[str, Any]) -> list[int]:
    """Recover the frozen L27 ordered/evicted cohort."""
    return sorted(
        {
            int(row["example"])
            for row in source["rows"]
            if int(row["layer"]) == ANALYSIS_LAYER
            and row["condition"] == "ordered"
            and row["placement"] == "evicted"
        }
    )


def parse_answer_digits(examples: Sequence[dict[str, Any]]) -> list[list[int] | None]:
    """Convert numeric RULER answers into digit sequences."""
    sequences: list[list[int] | None] = []

    for example in examples:
        answer = example["answer"].strip()
        sequences.append(
            [int(char) for char in answer]
            if answer.isdigit()
            else None
        )

    return sequences


# %% Prefix-control construction

def find_prefix_token_span(
    tokenizer: Any,
    example: dict[str, Any],
) -> tuple[list[int], list[int], int]:
    """Return prompt token IDs, answer-prefix suffix span, and prompt length."""
    base = example["prompt"]
    prefix_char_start = len(example["context"] + example["question"])

    encoded = tokenizer(
        base,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    prompt_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    prompt_length = len(prompt_ids)

    span = [
        index
        for index, (_, end) in enumerate(offsets)
        if end > prefix_char_start
    ]

    if not span:
        raise RuntimeError("Could not identify answer-prefix token suffix.")

    expected_suffix = list(range(span[0], prompt_length))

    if span != expected_suffix:
        raise RuntimeError("Answer-prefix token span is not a prompt suffix.")

    return prompt_ids, span, prompt_length


def build_paired_inputs(
    tokenizer: Any,
    model_device: torch.device,
    example: dict[str, Any],
    digits: Sequence[int],
    neutral_token_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:
    """Build original and same-length punctuation-control token sequences."""
    base = example["prompt"]

    prompt_ids, prefix_span, prompt_length = find_prefix_token_span(
        tokenizer,
        example,
    )

    full_text = base + " " + "".join(str(digit) for digit in digits)

    original = tokenizer(
        full_text,
        return_tensors="pt",
    ).to(model_device)

    expected_prompt = torch.tensor(prompt_ids)

    if not torch.equal(
        original["input_ids"][0, :prompt_length].cpu(),
        expected_prompt,
    ):
        raise RuntimeError("Prompt is not a clean prefix of prompt+answer.")

    control = {
        name: tensor.clone()
        for name, tensor in original.items()
    }

    control["input_ids"][0, prefix_span] = neutral_token_id

    # Critical invariants.
    assert original["input_ids"].shape == control["input_ids"].shape

    assert torch.equal(
        original["input_ids"][0, :prefix_span[0]],
        control["input_ids"][0, :prefix_span[0]],
    )

    assert torch.equal(
        original["input_ids"][0, prompt_length:],
        control["input_ids"][0, prompt_length:],
    )

    return original, control, prompt_length, len(prefix_span)


def verify_evicted_placement(
    tokenizer: Any,
    example: dict[str, Any],
    n_tokens: int,
) -> None:
    """Confirm that the historical member remains evicted."""
    base = example["prompt"]
    answer = example["answer"]

    hit = base.find(answer)

    if hit < 0:
        raise RuntimeError("Needle answer not found in RULER prompt.")

    needle_position = len(
        tokenizer(
            base[:hit],
            add_special_tokens=True,
        )["input_ids"]
    )

    window_start = n_tokens - SLIDING_WINDOW

    if not NUM_ATTN_SINKS <= needle_position < window_start:
        raise RuntimeError(
            f"Expected evicted placement, got needle_pos={needle_position}, "
            f"window_start={window_start}."
        )


# %% Inference

@torch.no_grad()
def score_variant(
    inputs: dict[str, torch.Tensor],
    prompt_length: int,
    digits: Sequence[int],
    digit_token_ids: Sequence[int],
    bundle: Any,
    probe: Any,
    lens: Any,
) -> dict[str, Any]:
    """Run AHN-ON/NOWRITE and return the L27 7x10 digit table."""
    on = probe.run(
        inputs,
        nowrite=False,
        layers=LAYERS,
        capture_residual=True,
    )

    off = probe.run(
        inputs,
        nowrite=True,
        layers=LAYERS,
        capture_residual=True,
    )

    digit_logprobs: list[list[float]] = []
    digit_ranks: list[int] = []

    for position, digit in enumerate(digits):
        readout_position = prompt_length + position

        residual_delta = (
            on.residual(
                ANALYSIS_LAYER,
                pos=readout_position,
            ).float()
            - off.residual(
                ANALYSIS_LAYER,
                pos=readout_position,
            ).float()
        )

        logits = ai.readout_logits(
            residual_delta,
            bundle,
            lens=lens,
            layer=ANALYSIS_LAYER,
        )

        log_probs = torch.log_softmax(
            logits.float(),
            dim=-1,
        )

        digit_logprobs.append(
            [
                float(log_probs[token_id])
                for token_id in digit_token_ids
            ]
        )

        digit_ranks.append(
            ai.token_rank(
                logits,
                digit_token_ids[digit],
            )
        )

    del on, off
    ai.free_cuda()

    return {
        "digit_logprobs": digit_logprobs,
        "digit_ranks": digit_ranks,
    }


def run_paired_inference(
    examples: Sequence[dict[str, Any]],
    answer_sequences: Sequence[list[int] | None],
    bundle: Any,
    probe: Any,
    lens: Any,
) -> tuple[dict[int, Any], dict[int, Any], list[int]]:
    """Run original and punctuation-control conditions on the frozen cohort."""
    tokenizer = bundle.tokenizer

    digit_token_ids = [
        tokenizer.encode(
            str(digit),
            add_special_tokens=False,
        )[0]
        for digit in range(10)
    ]

    if len(set(digit_token_ids)) != 10:
        raise RuntimeError("Digits are not ten distinct single-token targets.")

    neutral_token_ids = tokenizer.encode(
        NEUTRAL_TOKEN_TEXT,
        add_special_tokens=False,
    )

    if len(neutral_token_ids) != 1:
        raise RuntimeError("Neutral punctuation must tokenize to one token.")

    neutral_token_id = neutral_token_ids[0]

    original_rows: dict[int, Any] = {}
    control_rows: dict[int, Any] = {}
    prefix_lengths: list[int] = []

    start_time = time.time()

    for run_index, example_index in enumerate(FROZEN_EXAMPLES, start=1):
        example = examples[example_index]
        digits = answer_sequences[example_index]

        if digits is None:
            raise RuntimeError(
                f"Non-numeric answer for example {example_index}."
            )

        original, control, prompt_length, prefix_length = build_paired_inputs(
            tokenizer=tokenizer,
            model_device=bundle.model.device,
            example=example,
            digits=digits,
            neutral_token_id=neutral_token_id,
        )

        verify_evicted_placement(
            tokenizer=tokenizer,
            example=example,
            n_tokens=int(original["input_ids"].shape[1]),
        )

        original_rows[example_index] = score_variant(
            inputs=original,
            prompt_length=prompt_length,
            digits=digits,
            digit_token_ids=digit_token_ids,
            bundle=bundle,
            probe=probe,
            lens=lens,
        )

        control_rows[example_index] = score_variant(
            inputs=control,
            prompt_length=prompt_length,
            digits=digits,
            digit_token_ids=digit_token_ids,
            bundle=bundle,
            probe=probe,
            lens=lens,
        )

        prefix_lengths.append(prefix_length)

        del original, control
        ai.free_cuda()

        elapsed_minutes = (time.time() - start_time) / 60

        print(
            f"[{run_index:02d}/{len(FROZEN_EXAMPLES)}] "
            f"example={example_index:02d} "
            f"prefix_tokens={prefix_length:02d} "
            f"elapsed={elapsed_minutes:.1f} min",
            flush=True,
        )

    return original_rows, control_rows, prefix_lengths


# %% C2 statistic

def answer_logprob(
    table: Sequence[Sequence[float]],
    digits: Sequence[int],
) -> float:
    """Return log-probability of a candidate digit sequence."""
    n_positions = min(len(table), len(digits))

    return sum(
        table[position][digits[position]]
        for position in range(n_positions)
    )


def c2_log_effect(
    rows: dict[int, Any],
    answer_sequences: Sequence[list[int] | None],
    examples: Sequence[int],
) -> tuple[float, int, int]:
    """Return RULER C2 per-example log effect."""
    log_folds: list[float] = []
    dropped = 0

    for left in range(len(examples)):
        for right in range(left + 1, len(examples)):
            i = examples[left]
            j = examples[right]

            digits_i = answer_sequences[i]
            digits_j = answer_sequences[j]

            if (
                digits_i is None
                or digits_j is None
                or digits_i == digits_j
            ):
                dropped += 1
                continue

            table_i = rows[i]["digit_logprobs"]
            table_j = rows[j]["digit_logprobs"]

            log_fold = (
                answer_logprob(table_i, digits_i)
                - answer_logprob(table_i, digits_j)
                - answer_logprob(table_j, digits_i)
                + answer_logprob(table_j, digits_j)
            )

            log_folds.append(log_fold)

    if not log_folds:
        raise RuntimeError("No valid C2 pairs.")

    # Historical RULER fold is the square of the per-example effect.
    log_effect = 0.5 * float(np.mean(log_folds))

    return log_effect, len(log_folds), dropped


def digit_effect(
    rows: dict[int, Any],
    answer_sequences: Sequence[list[int] | None],
    position: int,
) -> float:
    """Compute the C2 per-example effect for one digit position."""
    log_folds: list[float] = []

    for left in range(len(FROZEN_EXAMPLES)):
        for right in range(left + 1, len(FROZEN_EXAMPLES)):
            i = FROZEN_EXAMPLES[left]
            j = FROZEN_EXAMPLES[right]

            digits_i = answer_sequences[i]
            digits_j = answer_sequences[j]

            if digits_i is None or digits_j is None:
                continue

            target_i = digits_i[position]
            target_j = digits_j[position]

            if target_i == target_j:
                continue

            table_i = rows[i]["digit_logprobs"][position]
            table_j = rows[j]["digit_logprobs"][position]

            log_folds.append(
                table_i[target_i]
                - table_i[target_j]
                - table_j[target_i]
                + table_j[target_j]
            )

    return math.exp(
        0.5 * float(np.mean(log_folds))
    )


# %% Example-level bootstrap

def bootstrap_log_effect(
    rows: dict[int, Any],
    answer_sequences: Sequence[list[int] | None],
    sample: Sequence[int],
) -> float:
    """Compute C2 log effect for one bootstrap resample."""
    log_folds: list[float] = []

    for left in range(len(sample)):
        for right in range(left + 1, len(sample)):
            i = sample[left]
            j = sample[right]

            digits_i = answer_sequences[i]
            digits_j = answer_sequences[j]

            if (
                digits_i is None
                or digits_j is None
                or digits_i == digits_j
            ):
                continue

            table_i = rows[i]["digit_logprobs"]
            table_j = rows[j]["digit_logprobs"]

            log_folds.append(
                answer_logprob(table_i, digits_i)
                - answer_logprob(table_i, digits_j)
                - answer_logprob(table_j, digits_i)
                + answer_logprob(table_j, digits_j)
            )

    if not log_folds:
        return float("nan")

    return 0.5 * float(np.mean(log_folds))


def paired_bootstrap(
    original_rows: dict[int, Any],
    control_rows: dict[int, Any],
    answer_sequences: Sequence[list[int] | None],
) -> tuple[list[float], list[float], list[float]]:
    """Return example-level bootstrap distributions."""
    rng = np.random.default_rng(SEED)

    original_effects: list[float] = []
    control_effects: list[float] = []
    ratios: list[float] = []

    for _ in range(N_BOOTSTRAP):
        sample = list(
            rng.choice(
                FROZEN_EXAMPLES,
                size=len(FROZEN_EXAMPLES),
                replace=True,
            )
        )

        original_log = bootstrap_log_effect(
            original_rows,
            answer_sequences,
            sample,
        )

        control_log = bootstrap_log_effect(
            control_rows,
            answer_sequences,
            sample,
        )

        if math.isnan(original_log) or math.isnan(control_log):
            continue

        original_effects.append(math.exp(original_log))
        control_effects.append(math.exp(control_log))
        ratios.append(math.exp(control_log - original_log))

    return original_effects, control_effects, ratios


# %% Paired randomization test

def paired_randomization_p(
    original_rows: dict[int, Any],
    control_rows: dict[int, Any],
    answer_sequences: Sequence[list[int] | None],
    observed_difference: float,
) -> float:
    """Swap condition labels within examples and recompute C2."""
    rng = random.Random(SEED + 1)
    hits = 0

    for _ in range(N_RANDOMIZATION):
        permuted_original: dict[int, Any] = {}
        permuted_control: dict[int, Any] = {}

        for example_index in FROZEN_EXAMPLES:
            if rng.random() < 0.5:
                permuted_original[example_index] = original_rows[example_index]
                permuted_control[example_index] = control_rows[example_index]
            else:
                permuted_original[example_index] = control_rows[example_index]
                permuted_control[example_index] = original_rows[example_index]

        original_log, _, _ = c2_log_effect(
            permuted_original,
            answer_sequences,
            FROZEN_EXAMPLES,
        )

        control_log, _, _ = c2_log_effect(
            permuted_control,
            answer_sequences,
            FROZEN_EXAMPLES,
        )

        if abs(control_log - original_log) >= abs(observed_difference):
            hits += 1

    return (hits + 1) / (N_RANDOMIZATION + 1)


# %% Result assembly

def confidence_interval(values: Sequence[float]) -> list[float]:
    """Return percentile 95% confidence interval."""
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def build_result(
    provenance: dict[str, Any],
    answer_sequences: Sequence[list[int] | None],
    original_rows: dict[int, Any],
    control_rows: dict[int, Any],
    prefix_lengths: Sequence[int],
) -> dict[str, Any]:
    """Compute and assemble the complete reproducibility artifact."""
    original_log, n_pairs, dropped = c2_log_effect(
        original_rows,
        answer_sequences,
        FROZEN_EXAMPLES,
    )

    control_log, _, _ = c2_log_effect(
        control_rows,
        answer_sequences,
        FROZEN_EXAMPLES,
    )

    original_effect = math.exp(original_log)
    control_effect = math.exp(control_log)
    effect_ratio = control_effect / original_effect

    original_boot, control_boot, ratio_boot = paired_bootstrap(
        original_rows,
        control_rows,
        answer_sequences,
    )

    observed_difference = control_log - original_log

    randomization_p = paired_randomization_p(
        original_rows,
        control_rows,
        answer_sequences,
        observed_difference,
    )

    per_digit = []

    for position in range(7):
        original_digit_effect = digit_effect(
            original_rows,
            answer_sequences,
            position,
        )

        control_digit_effect = digit_effect(
            control_rows,
            answer_sequences,
            position,
        )

        per_digit.append(
            {
                "position": position + 1,
                "original": original_digit_effect,
                "control": control_digit_effect,
                "control_over_original":
                    control_digit_effect / original_digit_effect,
            }
        )

    return {
        "experiment_id": "06b_ruler_answer_prefix_ablation",
        "status": "completed",
        "provenance": provenance,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "design": {
            "ruler": {
                "config": RULER_CONFIG,
                "split": RULER_SPLIT,
                "n": RULER_N,
                "seed": SEED,
            },
            "frozen_evicted_examples": FROZEN_EXAMPLES,
            "analysis_layer": ANALYSIS_LAYER,
            "probe_layers": LAYERS,
            "sliding_window": SLIDING_WINDOW,
            "num_attn_sinks": NUM_ATTN_SINKS,
            "neutral_token_text": NEUTRAL_TOKEN_TEXT,
            "prefix_token_counts":
                sorted(set(prefix_lengths)),
            "bootstrap_replicates": N_BOOTSTRAP,
            "randomization_replicates": N_RANDOMIZATION,
        },
        "results": {
            "n_examples": len(FROZEN_EXAMPLES),
            "n_unordered_c2_pairs": n_pairs,
            "pairs_dropped": dropped,
            "original_prefix": {
                "effect": original_effect,
                "ci95": confidence_interval(original_boot),
            },
            "punctuation_control": {
                "effect": control_effect,
                "ci95": confidence_interval(control_boot),
            },
            "controlled_change": {
                "control_over_original": effect_ratio,
                "ci95": confidence_interval(ratio_boot),
                "paired_randomization_p": randomization_p,
            },
            "per_digit_c2_effect": per_digit,
        },
        "raw_rows": {
            "original_prefix": original_rows,
            "punctuation_control": control_rows,
        },
        "interpretation": {
            "primary":
                "Same-length punctuation replacement did not eliminate "
                "the L27 C2 effect.",
            "historical_caveat":
                "This is a new internally controlled replication/ablation, "
                "not an exact Run026 remeasurement.",
        },
    }


# %% Save and print

def save_result(result: dict[str, Any]) -> None:
    """Write the rerun artifact without silently overwriting an existing run."""
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Rename or remove it only for a deliberate rerun."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2) + "\n"
    )


def print_summary(result: dict[str, Any]) -> None:
    """Print the headline controlled result."""
    results = result["results"]

    original = results["original_prefix"]
    control = results["punctuation_control"]
    change = results["controlled_change"]

    print("\n========== 06b RESULT ==========")

    print(
        f"Original: {original['effect']:.6f}x "
        f"CI [{original['ci95'][0]:.6f}, "
        f"{original['ci95'][1]:.6f}]"
    )

    print(
        f"Control:  {control['effect']:.6f}x "
        f"CI [{control['ci95'][0]:.6f}, "
        f"{control['ci95'][1]:.6f}]"
    )

    print(
        f"Ratio:    {change['control_over_original']:.6f}x "
        f"CI [{change['ci95'][0]:.6f}, "
        f"{change['ci95'][1]:.6f}]"
    )

    print(
        "Paired randomization p:",
        f"{change['paired_randomization_p']:.6f}",
    )

    print("Saved:", OUTPUT_PATH)


# %% Main orchestration

def main() -> None:
    provenance = verify_provenance()

    source = load_source_artifact()

    frozen_from_source = historical_evicted_examples(
        source
    )

    if frozen_from_source != FROZEN_EXAMPLES:
        raise RuntimeError(
            "Historical frozen cohort does not match "
            f"the preregistered list:\n{frozen_from_source}"
        )

    bundle = ai.load_ahn_model(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        sliding_window=SLIDING_WINDOW,
        num_attn_sinks=NUM_ATTN_SINKS,
    )

    probe = ai.AHNProbe(bundle)

    lens = ai.JacobianLens.load(
        str(LENS_PATH),
        map_location=str(bundle.model.device),
    )

    examples = ai.load_ruler(
        config=RULER_CONFIG,
        split=RULER_SPLIT,
        n=RULER_N,
        seed=SEED,
    )

    answer_sequences = parse_answer_digits(
        examples
    )

    original_rows, control_rows, prefix_lengths = run_paired_inference(
        examples=examples,
        answer_sequences=answer_sequences,
        bundle=bundle,
        probe=probe,
        lens=lens,
    )

    result = build_result(
        provenance=provenance,
        answer_sequences=answer_sequences,
        original_rows=original_rows,
        control_rows=control_rows,
        prefix_lengths=prefix_lengths,
    )

    save_result(result)
    print_summary(result)

    sys.stdout.flush()
    os._exit(0)


# %% Entrypoint

if __name__ == "__main__":
    main()
