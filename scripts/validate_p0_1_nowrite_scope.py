import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import flash_attn
import ahn_interp as ai

OUT = Path("results/validation/p0_1_nowrite_scope.json")
CAPTURE_LAYERS = [9, 18, 27]
N_TOKENS = 9000


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def zero_hook(module, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


print("=== P0-1 NOWRITE SCOPE VALIDATION ===")

cfg = ai.load_run_config("run_3b_gdn")

bundle = ai.load_ahn_model(
    cfg["model_path"],
    sliding_window=cfg["sliding_window"],
    num_attn_sinks=cfg["num_attn_sinks"],
    attn_implementation=cfg["attn_impl"],
)

probe = ai.AHNProbe(bundle)

print("AHN layers:", bundle.ahn_layers)
print("n AHN layers:", len(bundle.ahn_layers))
print("capture layers:", CAPTURE_LAYERS)

assert len(bundle.ahn_layers) == 36

# Long enough to activate AHN.
text = " validation" * 12000
enc = bundle.tokenizer(text, return_tensors="pt", truncation=False)

inputs = {
    "input_ids": enc["input_ids"][:, :N_TOKENS].to(bundle.model.device),
    "attention_mask": enc["attention_mask"][:, :N_TOKENS].to(bundle.model.device),
}

n_tokens = int(inputs["input_ids"].shape[1])
threshold = bundle.sliding_window + bundle.num_attn_sinks

print("n_tokens:", n_tokens)
print("AHN activation threshold:", threshold)

assert n_tokens > threshold


# ----------------------------------------------------------
# 1. AHN ON
# ----------------------------------------------------------
print("\n[1/3] AHN ON")

on = probe.run(
    inputs,
    nowrite=False,
    layers=CAPTURE_LAYERS,
    capture_residual=True,
    keep_logits=True,
)


# ----------------------------------------------------------
# 2. LOCAL NOWRITE: only layers 9,18,27
# ----------------------------------------------------------
print("[2/3] LOCAL NOWRITE [9,18,27]")

partial = probe.run(
    inputs,
    nowrite=True,
    layers=CAPTURE_LAYERS,
    capture_residual=True,
    keep_logits=True,
)


# ----------------------------------------------------------
# 3. GLOBAL NOWRITE: all 36 AHN layers
# ----------------------------------------------------------
print("[3/3] GLOBAL NOWRITE all AHN layers")

handles = []

try:
    for L in bundle.ahn_layers:
        handles.append(
            bundle.model.model.layers[L].ahn.register_forward_hook(zero_hook)
        )

    global_off = probe.run(
        inputs,
        nowrite=False,
        layers=CAPTURE_LAYERS,
        capture_residual=True,
        keep_logits=True,
    )

finally:
    for h in handles:
        h.remove()


def raw_norms(cap):
    return {
        str(L): float(cap.raw(L, -1).float().norm().item())
        for L in CAPTURE_LAYERS
    }


def residual_l2(a, b):
    return {
        str(L): float(
            (
                a.residual(L, -1).float()
                - b.residual(L, -1).float()
            ).norm().item()
        )
        for L in CAPTURE_LAYERS
    }


def logits_l2(a, b):
    return float(
        (a.logits.float() - b.logits.float()).norm().item()
    )


record = {
    "experiment": "P0-1 NOWRITE scope validation",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),

    "git": {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
    },

    "environment": {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "flash_attention": flash_attn.__version__,
    },

    "config": {
        "run": "run_3b_gdn",
        "model_path": bundle.model_path,
        "sliding_window": bundle.sliding_window,
        "num_attn_sinks": bundle.num_attn_sinks,
        "activation_threshold": threshold,
        "n_tokens": n_tokens,
        "n_ahn_layers": len(bundle.ahn_layers),
        "ahn_layers": bundle.ahn_layers,
        "capture_layers": CAPTURE_LAYERS,
    },

    "conditions": {
        "on": {
            "zeroed_layers": [],
            "raw_norms": raw_norms(on),
        },
        "partial_nowrite": {
            "zeroed_layers": CAPTURE_LAYERS,
            "raw_norms": raw_norms(partial),
        },
        "global_nowrite": {
            "zeroed_layers": bundle.ahn_layers,
            "raw_norms": raw_norms(global_off),
        },
    },

    "comparisons": {
        "final_logits_l2_on_vs_partial":
            logits_l2(on, partial),

        "final_logits_l2_on_vs_global":
            logits_l2(on, global_off),

        "final_logits_l2_partial_vs_global":
            logits_l2(partial, global_off),

        "residual_l2_on_vs_partial":
            residual_l2(on, partial),

        "residual_l2_on_vs_global":
            residual_l2(on, global_off),

        "residual_l2_partial_vs_global":
            residual_l2(partial, global_off),
    },
}

OUT.write_text(json.dumps(record, indent=2))

print("\n=== COMPARISONS ===")
print(json.dumps(record["comparisons"], indent=2))

print("\n=== RAW AHN NORMS ===")
print("ON     :", record["conditions"]["on"]["raw_norms"])
print("PARTIAL:", record["conditions"]["partial_nowrite"]["raw_norms"])
print("GLOBAL :", record["conditions"]["global_nowrite"]["raw_norms"])

print("\nSaved:", OUT)
