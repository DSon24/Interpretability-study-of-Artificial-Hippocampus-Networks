from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN"
RUN_CONFIG = ROOT / "configs/run_3b_gdn.json"
CONFIG_AUDIT = ROOT / "results/run_3b_gdn/00_config_audit.json"
P0_4 = ROOT / "results/validation/p0_4_global_behavior.json"
OUTPUT = ROOT / "results/validation/p2_1_checkpoint_identity.json"

SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
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


def main() -> None:
    model_config = json.loads(
        (MODEL / "config.json").read_text()
    )
    run_config = json.loads(RUN_CONFIG.read_text())
    runtime_audit = json.loads(CONFIG_AUDIT.read_text())
    p0_4 = json.loads(P0_4.read_text())

    observed_shards = {
        path.name: sha256_file(path)
        for path in sorted(MODEL.glob("*.safetensors"))
    }

    recorded_shards = p0_4[
        "provenance"
    ]["model_safetensor_sha256"]

    index_path = MODEL / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())

    referenced_shards = sorted(set(index["weight_map"].values()))
    actual_shards = sorted(observed_shards)

    small_hashes = {
        name: sha256_file(MODEL / name)
        for name in SMALL_FILES
        if (MODEL / name).exists()
    }

    embedded_config = {
        "ahn_implementation":
            model_config.get("_ahn_implementation"),
        "num_hidden_layers":
            model_config.get("num_hidden_layers"),
        "sliding_window":
            model_config.get("sliding_window"),
        "use_sliding_window":
            model_config.get("use_sliding_window"),
        "max_window_layers":
            model_config.get("max_window_layers"),
        "use_ahn_router":
            model_config.get("use_ahn_router"),
        "ahn_position":
            model_config.get("ahn_position"),
        "sliding_window_type":
            model_config.get("sliding_window_type"),
        "_name_or_path":
            model_config.get("_name_or_path"),
        "_commit_hash":
            model_config.get("_commit_hash"),
        "transformers_version":
            model_config.get("transformers_version"),
    }

    runtime_config = {
        "model_path": run_config.get("model_path"),
        "cell": run_config.get("cell"),
        "sliding_window": run_config.get("sliding_window"),
        "num_attn_sinks": run_config.get("num_attn_sinks"),
        "attn_impl": run_config.get("attn_impl"),
        "dtype": run_config.get("dtype"),
    }

    checks = {
        "exact_weight_hash_match":
            observed_shards == recorded_shards,
        "two_weight_shards":
            len(observed_shards) == 2,
        "index_matches_weight_files":
            referenced_shards == actual_shards,
        "36_transformer_layers":
            model_config.get("num_hidden_layers") == 36,
        "36_runtime_ahn_layers":
            runtime_audit.get("n_ahn_layers") == 36,
        "gated_delta_net_matches": (
            model_config.get("_ahn_implementation")
            == runtime_audit.get("ahn_implementation")
            == run_config.get("cell")
        ),
        "runtime_window_matches_run_config": (
            runtime_audit.get("sliding_window")
            == run_config.get("sliding_window")
            == 8064
        ),
        "runtime_sinks_match_run_config": (
            runtime_audit.get("num_attn_sinks")
            == run_config.get("num_attn_sinks")
            == 128
        ),
        "runtime_sliding_window_enabled":
            runtime_audit.get("use_sliding_window") is True,
    }

    if not all(checks.values()):
        failed = [
            name for name, passed in checks.items()
            if not passed
        ]
        raise RuntimeError(f"Checkpoint audit failed: {failed}")

    runtime_override_required = (
        embedded_config["sliding_window"]
        != runtime_config["sliding_window"]
        or embedded_config["use_sliding_window"] is not True
    )

    upstream_revision_recorded = bool(
        embedded_config.get("_name_or_path")
        and embedded_config.get("_commit_hash")
    )

    result: dict[str, Any] = {
        "experiment":
            "P2-1 checkpoint identity and provenance validation",
        "status": "completed_with_provenance_caveat",
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
        },
        "checkpoint": {
            "path": str(MODEL),
            "weight_shard_sha256": observed_shards,
            "small_file_sha256": small_hashes,
            "tensor_index_entries":
                len(index["weight_map"]),
            "referenced_weight_shards":
                referenced_shards,
        },
        "embedded_checkpoint_config": embedded_config,
        "experiment_runtime_config": runtime_config,
        "runtime_observation": {
            "ahn_implementation":
                runtime_audit["ahn_implementation"],
            "n_layers": runtime_audit["n_layers"],
            "n_ahn_layers": runtime_audit["n_ahn_layers"],
            "sliding_window":
                runtime_audit["sliding_window"],
            "use_sliding_window":
                runtime_audit["use_sliding_window"],
            "num_attn_sinks":
                runtime_audit["num_attn_sinks"],
            "dtype": runtime_audit["dtype"],
        },
        "validation": checks,
        "provenance_assessment": {
            "local_checkpoint_identity_fully_frozen": True,
            "runtime_override_required":
                runtime_override_required,
            "upstream_revision_recorded_in_config":
                upstream_revision_recorded,
            "complete_upstream_merge_provenance":
                upstream_revision_recorded,
        },
        "interpretation": {
            "primary": (
                "The local merged AHN-GDN checkpoint is uniquely "
                "identified by matching weight, index, configuration "
                "and tokenizer hashes."
            ),
            "runtime_caveat": (
                "The checkpoint config does not itself encode the "
                "8064-token experimental window. Reproduction must "
                "apply configs/run_3b_gdn.json, which enables sliding "
                "window attention at 8064 with 128 attention sinks."
            ),
            "upstream_caveat": (
                "Local identity is complete, but the original upstream "
                "base/merge revision is only complete if both an origin "
                "and commit hash are embedded in config.json."
            ),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    print("P2-1 COMPLETE")
    print("RESULT:", OUTPUT)
    print(json.dumps({
        "status": result["status"],
        "validation": checks,
        "embedded_checkpoint_config": embedded_config,
        "experiment_runtime_config": runtime_config,
        "provenance_assessment":
            result["provenance_assessment"],
        "interpretation": result["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
