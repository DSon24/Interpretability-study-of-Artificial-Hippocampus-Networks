from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset

import ahn_interp as ai


LAYERS = [9, 18, 27]
WINDOW = 8064
SINKS = 128
N = 60

VERSION = "row87-long-ruler-optimized-v1"

LENS_PATH = Path(
    "results/run_3b_gdn/jlens_qwen25_3b_1000ctx.pt"
)

VALIDATION_PATH = Path(
    "results/run_3b_gdn/02_table3_jlens_validation_1000ctx.json"
)


def save_atomic(path: Path, blob: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2)
    os.replace(tmp, path)


def capture_selected_residuals(
    bundle,
    inputs,
    positions,
    *,
    nowrite: bool,
):
    """
    Optimized equivalent of the residual part of AHNProbe.run().

    IMPORTANT:
    - same post-attention / pre-MLP residual location as AHNProbe
    - same partial NOWRITE scope: layers [9,18,27]
    - captures ONLY the answer-scoring positions
    - use_cache=True because the custom AHN inference path requires a KV cache object
    """
    model = bundle.model
    captured = {}
    handles = []

    def zero_ahn(module, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)

    def capture_residual(layer_idx):
        def hook(module, inp, out):
            # Exact same location used by AHNProbe:
            # input to post_attention_layernorm =
            # post-attention, pre-MLP residual stream.
            x = inp[0]

            captured[layer_idx] = (
                x[0, positions, :]
                .detach()
                .clone()
            )

        return hook

    for L in LAYERS:
        layer = model.model.layers[L]

        if not hasattr(layer, "ahn"):
            raise RuntimeError(
                f"layer {L} has no AHN module"
            )

        # Same order as AHNProbe.run():
        # zero AHN first for NOWRITE.
        if nowrite:
            handles.append(
                layer.ahn.register_forward_hook(zero_ahn)
            )

        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                capture_residual(L)
            )
        )

    try:
        with torch.inference_mode():
            model(
                **inputs,
                use_cache=True,
                num_logits_to_keep=1,
            )
    finally:
        for h in handles:
            h.remove()

    missing = sorted(set(LAYERS) - set(captured))
    if missing:
        raise RuntimeError(
            f"failed to capture residuals at layers {missing}"
        )

    return captured


def score_delta(
    delta,
    layer,
    digits,
    digit_ids,
    bundle,
    lens,
):
    """
    Vectorized Run-026 digit scoring.

    delta: [answer_len, hidden]

    One J-Lens projection handles all 7 positions together.
    """
    with torch.inference_mode():

        logits = ai.readout_logits(
            delta.float(),
            bundle,
            lens=lens,
            layer=layer,
        )

        # shape: [7, vocab]
        logprobs = torch.log_softmax(
            logits.float(),
            dim=-1,
        )

        digit_cols = torch.tensor(
            digit_ids,
            device=logprobs.device,
            dtype=torch.long,
        )

        # Full 7 x 10 table needed for C2 cross-answer analysis.
        table = logprobs.index_select(
            1,
            digit_cols,
        )

        target_ids = torch.tensor(
            [digit_ids[d] for d in digits],
            device=logits.device,
            dtype=torch.long,
        )

        target_logits = logits.gather(
            1,
            target_ids[:, None],
        ).squeeze(1)

        # Exact rank definition except pathological exact-logit ties:
        # number of vocabulary logits strictly larger than target.
        ranks = (
            logits > target_logits[:, None]
        ).sum(dim=1)

        target_logprobs = logprobs.gather(
            1,
            target_ids[:, None],
        ).squeeze(1)

    return {
        "digit_logprobs": table.cpu().tolist(),
        "digit_ranks": [
            int(x) for x in ranks.cpu().tolist()
        ],
        "answer_logprob": float(
            target_logprobs.sum().item()
        ),
        "mean_digit_rank": float(
            ranks.float().mean().item()
        ),
    }


def prepare_example(ex, tok, bundle, digit_ids):
    base = ex["input"]
    answer = ex["outputs"][0].strip()

    if not answer.isdigit():
        raise RuntimeError(
            f"non-digit RULER answer: {answer!r}"
        )

    digits = [int(x) for x in answer]

    prompt_ids = tok(
        base,
        return_tensors="pt",
    )["input_ids"]

    full = base + " " + answer

    inputs = tok(
        full,
        return_tensors="pt",
    ).to(bundle.model.device)

    P = int(prompt_ids.shape[1])
    n_tokens = int(inputs["input_ids"].shape[1])

    # Same prefix guard as corrected Run 026.
    if not torch.equal(
        inputs["input_ids"][0, :P].cpu(),
        prompt_ids[0],
    ):
        raise RuntimeError(
            "prompt is not a clean token prefix"
        )

    positions = [
        P + t
        for t in range(len(digits))
    ]

    # Position P+t predicts token P+t+1.
    expected_target_ids = [
        digit_ids[d]
        for d in digits
    ]

    actual_target_ids = (
        inputs["input_ids"][
            0,
            [p + 1 for p in positions]
        ]
        .detach()
        .cpu()
        .tolist()
    )

    if actual_target_ids != expected_target_ids:
        raise RuntimeError(
            "answer digit tokenization mismatch:\n"
            f"expected={expected_target_ids}\n"
            f"actual={actual_target_ids}"
        )

    occurrences = base.count(answer)

    if occurrences != 1:
        raise RuntimeError(
            f"expected exactly one needle occurrence; "
            f"found {occurrences}"
        )

    hit = base.find(answer)

    needle_pos = len(
        tok(
            base[:hit],
            add_special_tokens=True,
        )["input_ids"]
    )

    boundary = n_tokens - WINDOW
    eviction_distance = boundary - needle_pos

    needle_is_evicted = (
        SINKS <= needle_pos < boundary
    )

    if needle_pos < SINKS:
        placement = "in_sink_region"
    elif needle_is_evicted:
        placement = "evicted"
    else:
        placement = "in_window"

    return {
        "base": base,
        "answer": answer,
        "digits": digits,
        "inputs": inputs,
        "positions": positions,
        "n_tokens": n_tokens,
        "needle_pos": needle_pos,
        "boundary": boundary,
        "eviction_distance": eviction_distance,
        "needle_is_evicted": needle_is_evicted,
        "placement": placement,
    }


def parity_test(
    examples,
    tok,
    bundle,
    lens,
    digit_ids,
):
    """
    Scientific guard:

    Compare the optimized capture against the ORIGINAL
    AHNProbe Run-026 implementation on one evicted 32k example.
    """
    print("\n=== OPTIMIZATION PARITY TEST ===")

    chosen = None

    for idx, ex in enumerate(examples):
        p = prepare_example(
            ex,
            tok,
            bundle,
            digit_ids,
        )

        if p["needle_is_evicted"]:
            chosen = (idx, p)
            break

    assert chosen is not None

    idx, p = chosen

    print("example:", idx)
    print("tokens:", p["n_tokens"])
    print(
        "eviction distance:",
        p["eviction_distance"],
    )

    probe = ai.AHNProbe(bundle)

    # -------------------------------------------------
    # Original Run-026 implementation
    # -------------------------------------------------
    with torch.inference_mode():
        legacy_on = probe.run(
            p["inputs"],
            nowrite=False,
            layers=LAYERS,
            capture_residual=True,
        )

        legacy_off = probe.run(
            p["inputs"],
            nowrite=True,
            layers=LAYERS,
            capture_residual=True,
        )

    legacy = {}

    for L in LAYERS:
        on_sel = torch.stack([
            legacy_on.residual(L, pos=pos)
            for pos in p["positions"]
        ])

        off_sel = torch.stack([
            legacy_off.residual(L, pos=pos)
            for pos in p["positions"]
        ])

        legacy[L] = (
            on_sel.float() - off_sel.float()
        )

    del legacy_on, legacy_off
    ai.free_cuda()

    # -------------------------------------------------
    # Optimized implementation
    # -------------------------------------------------
    opt_on = capture_selected_residuals(
        bundle,
        p["inputs"],
        p["positions"],
        nowrite=False,
    )

    opt_off = capture_selected_residuals(
        bundle,
        p["inputs"],
        p["positions"],
        nowrite=True,
    )

    for L in LAYERS:

        opt_delta = (
            opt_on[L].float()
            - opt_off[L].float()
        )

        ref_delta = legacy[L]

        max_abs = float(
            (ref_delta - opt_delta)
            .abs()
            .max()
            .item()
        )

        ref_score = score_delta(
            ref_delta,
            L,
            p["digits"],
            digit_ids,
            bundle,
            lens,
        )

        opt_score = score_delta(
            opt_delta,
            L,
            p["digits"],
            digit_ids,
            bundle,
            lens,
        )

        same_ranks = (
            ref_score["digit_ranks"]
            == opt_score["digit_ranks"]
        )

        lp_diff = abs(
            ref_score["answer_logprob"]
            - opt_score["answer_logprob"]
        )

        print(
            f"L{L}: "
            f"residual max_abs={max_abs:.8f} "
            f"ranks_equal={same_ranks} "
            f"logprob_diff={lp_diff:.8f}"
        )

        if not same_ranks:
            raise RuntimeError(
                f"L{L}: optimized digit ranks differ "
                "from original Run-026 implementation"
            )

        if lp_diff > 1e-3:
            raise RuntimeError(
                f"L{L}: optimized answer logprob differs "
                f"by {lp_diff}"
            )

    del opt_on, opt_off, p
    ai.free_cuda()

    print("\nPARITY PASS")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        type=int,
        choices=[32768, 65536],
        required=True,
    )

    ap.add_argument(
        "--parity-only",
        action="store_true",
    )

    args = ap.parse_args()

    dataset_name = (
        f"lighteval/"
        f"RULER-{args.config}-Qwen2.5-Instruct"
    )

    out_path = Path(
        f"results/validation/"
        f"row87_ruler_{args.config}_rows.json"
    )

    # ---------------------------------------------------------
    # Config/model
    # ---------------------------------------------------------
    cfg = ai.load_run_config("run_3b_gdn")
    cfg["model_path"] = ai.resolve_ckpt(
        cfg["ckpt_name"]
    )

    with open(VALIDATION_PATH) as f:
        validation = json.load(f)

    bundle = ai.load_ahn_model(
        cfg["model_path"],
        dtype=torch.bfloat16,
        attn_implementation=cfg["attn_impl"],
        sliding_window=WINDOW,
        num_attn_sinks=SINKS,
    )

    assert bundle.ahn_impl == "GatedDeltaNet"

    tok = bundle.tokenizer

    # ---------------------------------------------------------
    # J-Lens
    # ---------------------------------------------------------
    lens = ai.JacobianLens.load(
        str(LENS_PATH),
        map_location=str(bundle.model.device),
    )

    assert lens.meta.get("n_prompts") == 1000
    assert sorted(lens.jacobians) == LAYERS

    digit_ids = [
        tok.encode(
            str(d),
            add_special_tokens=False,
        )[0]
        for d in range(10)
    ]

    assert len(set(digit_ids)) == 10

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    ds = load_dataset(
        dataset_name,
        split="niah_single_1",
        streaming=True,
    )

    examples = list(ds.take(N))

    assert len(examples) == N

    answer_digit_seqs = [
        [int(x) for x in ex["outputs"][0].strip()]
        for ex in examples
    ]

    print("model:", cfg["ckpt_name"])
    print("dataset:", dataset_name)
    print("examples:", len(examples))
    print("window:", WINDOW)
    print("sinks:", SINKS)
    print("lens prompts:", lens.meta["n_prompts"])

    # ---------------------------------------------------------
    # PARITY MODE
    # ---------------------------------------------------------
    if args.parity_only:

        if args.config != 32768:
            raise RuntimeError(
                "parity test should be run on 32k only"
            )

        parity_test(
            examples,
            tok,
            bundle,
            lens,
            digit_ids,
        )

        return

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------
    rows = []

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

        old_version = (
            existing
            .get("meta", {})
            .get("runner_version")
        )

        if old_version != VERSION:
            raise RuntimeError(
                f"{out_path} exists but was created by "
                f"{old_version!r}, not {VERSION!r}. "
                "Do not mix runner versions."
            )

        rows = existing["rows"]

        print(
            f"resuming: {len(rows)} saved rows"
        )

    completed = {}

    for r in rows:
        completed.setdefault(
            int(r["example"]),
            set(),
        ).add(int(r["layer"]))

    meta = {
        "runner_version": VERSION,
        "task": (
            "tracker row 87 — long-context "
            "RULER retention extension"
        ),
        "dataset": dataset_name,
        "split": "niah_single_1",
        "n_examples": N,
        "model": cfg["ckpt_name"],
        "cell": "GatedDeltaNet",
        "sliding_window": WINDOW,
        "num_attn_sinks": SINKS,
        "layers": LAYERS,
        "readout": "jlens",
        "lens_path": str(LENS_PATH),
        "lens_meta": lens.meta,
        "lens_validated": bool(
            validation.get(
                "TABLE_3_PASSED",
                False,
            )
        ),
        "scoring": (
            "corrected Run-026 gold-digit sequence "
            "scoring from AHN-ON minus partial-NOWRITE "
            "post-attention residual"
        ),
        "nowrite_scope": LAYERS,
        "optimization": {
            "capture_only_answer_positions": True,
            "use_cache": True,
            "vectorized_digit_readout": True,
            "model_forward_passes_per_example": 2,
        },
        "provenance": {
            "original_16k":
                "simonjegou/ruler config=16384 split=test",
            "long_context_family":
                "lighteval/RULER-*-Qwen2.5-Instruct",
            "verification":
                "16k first 60 niah_single_1 examples",
            "prompt_exact_matches": 60,
            "answer_exact_matches": 60,
            "verified_before_run": True,
        },
        "jlens_fit_corpus": (
            "Salesforce/wikitext "
            "wikitext-103-raw-v1 train"
        ),
        "homemade_llm_dataset_used": False,
    }

    start = time.time()

    for idx, ex in enumerate(examples):

        if completed.get(idx) == set(LAYERS):
            print(
                f"[{idx+1:02d}/60] already complete"
            )
            continue

        p = prepare_example(
            ex,
            tok,
            bundle,
            digit_ids,
        )

        t0 = time.time()

        # ---------------------------------------------
        # Exactly two model passes.
        # ---------------------------------------------
        on = capture_selected_residuals(
            bundle,
            p["inputs"],
            p["positions"],
            nowrite=False,
        )

        off = capture_selected_residuals(
            bundle,
            p["inputs"],
            p["positions"],
            nowrite=True,
        )

        # Remove any partial save for this example.
        rows = [
            r for r in rows
            if int(r["example"]) != idx
        ]

        for L in LAYERS:

            delta = (
                on[L].float()
                - off[L].float()
            )

            score = score_delta(
                delta,
                L,
                p["digits"],
                digit_ids,
                bundle,
                lens,
            )

            rows.append({
                "example": idx,
                "layer": L,
                "condition": "ordered",
                "readout": "jlens",
                "lens_validated":
                    meta["lens_validated"],
                "cohort": "ruler_niah",
                "ruler_config":
                    str(args.config),
                "dataset": dataset_name,

                "n_tokens":
                    p["n_tokens"],

                "needle_pos":
                    p["needle_pos"],

                "compression_boundary":
                    p["boundary"],

                "eviction_distance":
                    p["eviction_distance"],

                "needle_is_evicted":
                    p["needle_is_evicted"],

                "placement":
                    p["placement"],

                "answer_digits":
                    p["digits"],

                **score,
            })

        blob = {
            "meta": meta,

            # Compatibility with existing RULER
            # CPU analysis code.
            "cfg": {
                "run_name": "run_3b_gdn",
                "cell": "GatedDeltaNet",
                "scale": "3B",
            },

            "ruler_cfg": {
                "source": dataset_name,
                "config": str(args.config),
                "split": "niah_single_1",
                "n": N,
            },

            "digit_ids": digit_ids,
            "answer_len": 7,
            "answer_digit_seqs":
                answer_digit_seqs,

            "rows": sorted(
                rows,
                key=lambda r: (
                    int(r["example"]),
                    int(r["layer"]),
                ),
            ),
        }

        save_atomic(
            out_path,
            blob,
        )

        elapsed = (
            time.time() - t0
        ) / 60

        total = (
            time.time() - start
        ) / 60

        print(
            f"[{idx+1:02d}/60] "
            f"{p['placement']:9s} "
            f"d={p['eviction_distance']:6d} "
            f"tok={p['n_tokens']:5d} "
            f"{elapsed:.2f}m "
            f"total={total:.1f}m"
        )

        del on, off, p

        # Do not empty CUDA cache after every example.
        # Periodic cleanup avoids fragmentation without
        # throwing away the allocator cache constantly.
        if (idx + 1) % 5 == 0:
            ai.free_cuda()

    ai.free_cuda()

    print("\n=== COMPLETE ===")
    print("saved:", out_path)
    print("rows:", len(rows))

    evicted_examples = {
        int(r["example"])
        for r in rows
        if r["needle_is_evicted"]
    }

    print(
        "evicted:",
        len(evicted_examples),
        "/ 60",
    )


if __name__ == "__main__":
    main()
