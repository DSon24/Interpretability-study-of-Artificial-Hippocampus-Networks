"""CPU-only: emit the Table 1 (ablation grid) and Table 2 (evaluation cohorts) artefacts.

Expected_Tables_and_Figures lists Table 1 and Table 2 as "fill in before running -- no
artefact yet" (PROPOSAL.md, deviation 5). Both are specification tables, not computed
from data, but the *status* column of each row (has this condition been run? does this
cohort's artefact exist?) is derived from the filesystem here so the tables regenerate
instead of going stale by hand.

    python build_tables_1_2.py

Outputs:
    results/05_table1_ablation_grid.json
    results/05_table2_eval_cohorts.json
"""
from __future__ import annotations

import json
import os
import time

import ahn_interp as ai

RESULTS = "results"
SEED = 20260820


def _exists(*rel_paths: str) -> bool:
    return any(os.path.exists(os.path.join(RESULTS, p)) if not p.startswith(RESULTS)
              else os.path.exists(p) for p in rel_paths)


def _status(*rel_paths: str) -> str:
    return "run" if _exists(*rel_paths) else "not run"


# --------------------------------------------------------------------------------------
# Table 1 -- ablation grid: which conditions exist, which are primary
# --------------------------------------------------------------------------------------
def table1() -> dict:
    conditions = [
        {
            "condition": "no-AHN floor",
            "description": "stock Qwen2.5-3B-Instruct, sliding-window attention forced on "
                           "at 8064 for every layer, no AHN merge, no attention sinks",
            "role": "Primary -- floor",
            "backbone_family": "Qwen2.5",
            "scale": "3B",
            "cell": "n/a (no memory module)",
            "purpose": "the baseline every retention claim must exceed; replaces NOWRITE "
                       "as the true module-removal control (rows 35/36)",
            "artefact": "run_3b_floor/06_no_ahn_floor.json",
            "status": _status("run_3b_floor/06_no_ahn_floor.json"),
        },
        {
            "condition": "NOWRITE (zero-state)",
            "description": "AHN checkpoint with the memory write hooked to zero",
            "role": "Control C1",
            "backbone_family": "Qwen2.5",
            "scale": "3B",
            "cell": "GatedDeltaNet / DeltaNet / Mamba2",
            "purpose": "C1 zero-state control and the RQ1 comparison arm; NOWRITE != no-AHN "
                       "(see the no-AHN floor)",
            "artefact": "run_3b_gdn/03_nowrite_reproduction.json, run_3b_gdn/04i_ruler_controls_stats.json",
            "status": _status("run_3b_gdn/03_nowrite_reproduction.json"),
        },
        {
            "condition": "AHN active -- GatedDeltaNet",
            "description": "merged Qwen2.5-3B + AHN-GDN, sliding_window=8064, num_attn_sinks=128",
            "role": "Primary",
            "backbone_family": "Qwen2.5",
            "scale": "3B",
            "cell": "GatedDeltaNet",
            "purpose": "settings of record; the reference cell for RQ2",
            "artefact": "run_3b_gdn/04i_ruler_controls_stats.json",
            "status": _status("run_3b_gdn/04i_ruler_controls_stats.json"),
        },
        {
            "condition": "AHN active -- DeltaNet",
            "description": "merged Qwen2.5-3B + AHN-DN, same window/sinks",
            "role": "Primary (RQ2 cell-family axis)",
            "backbone_family": "Qwen2.5",
            "scale": "3B",
            "cell": "DeltaNet",
            "purpose": "RQ2: does retention behaviour differ by recurrent cell?",
            "artefact": "run_3b_dn/04i_ruler_controls_stats.json",
            "status": _status("run_3b_dn/04i_ruler_controls_stats.json"),
        },
        {
            "condition": "AHN active -- Mamba2",
            "description": "merged Qwen2.5-3B + AHN-Mamba2, same window/sinks; run on Modal "
                           "with real CUDA kernels",
            "role": "Primary (RQ2 cell-family axis)",
            "backbone_family": "Qwen2.5",
            "scale": "3B",
            "cell": "Mamba2",
            "purpose": "RQ2: third architecturally distinct cell",
            "artefact": "run_3b_m2/04i_ruler_controls_stats.json",
            "status": _status("run_3b_m2/04i_ruler_controls_stats.json"),
        },
        {
            "condition": "AHN active -- 7B tier",
            "description": "Qwen2.5-7B + AHN, all three cells",
            "role": "Secondary (cut-first under §8 cut order)",
            "backbone_family": "Qwen2.5",
            "scale": "7B",
            "cell": "GatedDeltaNet / DeltaNet / Mamba2",
            "purpose": "scale generality; gated on the 3B result and the map-cost budget "
                       "(Table 10)",
            "artefact": "run_7b_*/",
            "status": _status("run_7b_gdn", "run_7b_dn", "run_7b_m2"),
        },
        {
            "condition": "Multi-family -- Llama / Mistral",
            "description": "{Llama, Mistral} x {DeltaNet, GatedDeltaNet, Mamba2} at 7B",
            "role": "Secondary (generality check on RQ2/RQ3, not a fourth primary cell)",
            "backbone_family": "Llama, Mistral",
            "scale": "7B",
            "cell": "GatedDeltaNet / DeltaNet / Mamba2",
            "purpose": "show the readout describes AHN compressive memory, not Qwen2.5+AHN; "
                       "reintroduces the backbone confound so reported as a generality check",
            "artefact": "run_7b_llama_*/, run_7b_mistral_*/",
            "status": "planned, not started",
        },
    ]
    return {
        "table": "Table 1 -- ablation grid",
        "source": "Expected_Tables_and_Figures / PROPOSAL.md; status derived from results/ tree",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axes": {
            "condition": "no-AHN floor | NOWRITE | AHN active",
            "cell": "GatedDeltaNet | DeltaNet | Mamba2",
            "scale": "3B (Primary) | 7B (Secondary)",
            "backbone_family": "Qwen2.5 (Primary) | Llama | Mistral (Secondary)",
        },
        "primary_row": "AHN active -- GatedDeltaNet, Qwen2.5, 3B (settings of record)",
        "never_cut": "n/a -- Table 1 is a spec table; Tables 3, 4 and Figure 2 are the "
                     "never-cut validation set",
        "conditions": conditions,
    }


# --------------------------------------------------------------------------------------
# Table 2 -- evaluation cohorts and sizes
# --------------------------------------------------------------------------------------
def table2() -> dict:
    cohorts = [
        {
            "cohort": "RULER NIAH",
            "dataset": "simonjegou/ruler, config 16384",
            "n": 60,
            "composition": "niah_single_1, 7-digit answers; ~32 evicted / ~28 in-window "
                           "at the 8064 window (needle depth is random)",
            "role": "Primary RQ2 cohort (Run 026)",
            "metric": "mean digit rank of the answer sequence in the J-lens readout; "
                      "substring accuracy for the behavioural floor",
            "seed": SEED,
            "primary": True,
            "artefact": "run_3b_gdn/04g_ruler_niah_rows.json, run_3b_gdn/04i_ruler_controls_rows.json",
            "status": _status("run_3b_gdn/04i_ruler_controls_rows.json"),
        },
        {
            "cohort": "Homemade NIAH sweep",
            "dataset": "build_niah_prompt (synthetic)",
            "n": "10 needles x 7 eviction distances x 3 filler variants",
            "composition": "single-token needles, controlled eviction distance 64..8192",
            "role": "Supporting / limitations analysis (under-reads L27 vs RULER)",
            "metric": "target-token rank (o_t and D-resid bases)",
            "seed": SEED,
            "primary": False,
            "artefact": "run_3b_gdn/04_retention_rows.json",
            "status": _status("run_3b_gdn/04_retention_rows.json"),
        },
        {
            "cohort": "LongBench-E HotpotQA",
            "dataset": "THUDM/LongBench, hotpotqa_e (v1)",
            "n": 60,
            "composition": "length-stratified: 20 short / 20 mid / 20 long; eligible "
                           "prompts only (> window + sinks, <= 32000 tokens)",
            "role": "RQ1 behavioural effect (AHN vs NOWRITE), first-line scoring",
            "metric": "QA F1 per example; answer-change rate; per-example boundary JS",
            "seed": SEED,
            "primary": True,
            "artefact": "run_3b_gdn/03_nowrite_reproduction.json",
            "status": _status("run_3b_gdn/03_nowrite_reproduction.json"),
        },
        {
            "cohort": "RQ3 join",
            "dataset": "LongBench-E HotpotQA (same 60 as RQ1)",
            "n": 60,
            "composition": "gold-answer first-token rank at layers 9/18/27 joined to delta_f1",
            "role": "RQ3 headline: rho(retention half-life, delta_f1) vs rho(JS, F1) ~ 0",
            "metric": "Spearman rho, Holm-corrected across layers, bootstrap CI",
            "seed": SEED,
            "primary": True,
            "artefact": "run_3b_gdn/04b_joined_retention_task.json",
            "status": _status("run_3b_gdn/04b_joined_retention_task.json"),
        },
        {
            "cohort": "J-lens averaging corpus",
            "dataset": "Salesforce/wikitext, wikitext-103-raw-v1 (train)",
            "n": "2 x 1000 contexts (disjoint halves; skip 0 vs skip 20000)",
            "composition": "generic English, strictly disjoint from RULER / LongBench / LV-Eval",
            "role": "Table 3 map fit + map-stability check (r17, 1000-context refit)",
            "metric": "top-10 token overlap between the two disjoint-corpus maps",
            "seed": SEED,
            "primary": True,
            "artefact": "run_3b_gdn/02_table3_jlens_validation_1000ctx.json",
            "status": _status("run_3b_gdn/02_table3_jlens_validation_1000ctx.json"),
        },
        {
            "cohort": "Known-fact prompts",
            "dataset": "hand-written (\"The capital of France is\", ...)",
            "n": 8,
            "composition": "single-token answers with a leading space",
            "role": "Table 3 check 3 (known-fact recall) + lens diagnostics",
            "metric": "rank of the answer token from ~layer 20",
            "seed": None,
            "primary": True,
            "artefact": "run_3b_gdn/02_table3_jlens_validation_1000ctx.json",
            "status": _status("run_3b_gdn/02_table3_jlens_validation_1000ctx.json"),
        },
        {
            "cohort": "Needle-category test",
            "dataset": "hand-picked single-token needles",
            "n": "24 (12 place names / 12 common nouns)",
            "composition": "matched to the CONTROL_CANDIDATES format",
            "role": "Limitations: needle-identity / category-dependent retention at L18/L27",
            "metric": "needle-level mean rank, two-sample permutation test, Holm-corrected",
            "seed": SEED,
            "primary": False,
            "artefact": "run_3b_gdn/04e_needle_category_extended.json, run_3b_gdn/04f_needle_category_stats.json",
            "status": _status("run_3b_gdn/04f_needle_category_stats.json"),
        },
    ]
    return {
        "table": "Table 2 -- evaluation cohorts and sizes",
        "source": "PROPOSAL.md, DATASET_REGISTER_2026-09-08.md; status derived from results/ tree",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "standing_rule": "RULER NIAH n=60 is the primary RQ2 cohort. Any added dataset "
                         "(BABILong / NoLiMa / SCROLLS) is exploratory unless a pre-run "
                         "amendment defines its role, metric, size and decision rule.",
        "cohorts": cohorts,
    }


def main() -> None:
    ai.set_results_dir(RESULTS)
    for fn, name in ((table1, "05_table1_ablation_grid.json"),
                     (table2, "05_table2_eval_cohorts.json")):
        obj = fn()
        ai.save_json(obj, name)
        rows_key = "conditions" if "conditions" in obj else "cohorts"
        print(f"\n{obj['table']}  ->  {os.path.join(RESULTS, name)}")
        for r in obj[rows_key]:
            label = r.get("condition") or r.get("cohort")
            print(f"  [{r['status']:>16}]  {label}")


if __name__ == "__main__":
    main()
