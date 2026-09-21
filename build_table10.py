"""CPU-only: emit the Table 10 (compute accounting) artefact.

Expected_Tables_and_Figures Table 10: "Compute accounting per artefact. Filled in
continuously, not at the end." Columns: Artefact | GPU | GPU-hours | Peak VRAM | Where it ran.
Numbers that a saved result JSON records (`wall_min`, `map_cost_gpu_hours`, ...) are READ from
that JSON so the table regenerates; numbers that only exist in a run log / notebook stdout are
carried here with a `status` and a `source` so nobody mistakes an estimate for a measurement.

    python build_table10.py

Output: results/05_table10_compute.json

status: "measured"  wall-clock recorded in a saved artefact or stdout that we hold
        "estimated" reasoned from a sibling run; stdout not saved
        "todo"      exists but the number has not been pulled yet (say where to get it)
        "not_run"   artefact does not exist yet
Peak VRAM was not instrumented in nb03/nb04 (no torch.cuda.max_memory_allocated); the values
given are slice-headroom bounds, not measurements.
"""
from __future__ import annotations

import json
import os
import time

RESULTS = "results"
H = 60.0  # min per hour


def _load(rel):
    p = os.path.join(RESULTS, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def _wall(rel, *path):
    d = _load(rel)
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def row(artefact, gpu, where, gpu_h, status, source, group, vram=None, note=None):
    return {"group": group, "artefact": artefact, "gpu": gpu, "gpu_hours": gpu_h,
            "status": status, "peak_vram_gb": vram, "where_it_ran": where,
            "source": source, "note": note}


def main():
    box = "H100 MIG 1g.20gb slice, shared box jupyter-dphs-8080"
    modal = "Modal A100-40GB"
    rows = []

    # --- J-lens maps -----------------------------------------------------------------
    t3 = _load("run_3b_gdn/02_table3_jlens_validation.json") or {}
    t3k = _load("run_3b_gdn/02_table3_jlens_validation_1000ctx.json") or {}
    rows += [
        row("J-lens map, Qwen2.5-3B, corpus A (500 ctx, layers 9/18/27)", "not recorded",
            "not recorded (Table 3 JSON has no device field)", t3.get("map_cost_gpu_hours"),
            "measured", "results/run_3b_gdn/02_table3_jlens_validation.json:map_cost_gpu_hours",
            "jlens", note="Run 006. Proposal budget 15-20 GPU-h, abort >40."),
        row("J-lens map, Qwen2.5-3B, corpus B (500 ctx, stability check)", "not recorded",
            "not recorded", (t3.get("check4_map_stability") or t3.get("map_stability") or {}).get("fit_b_gpu_hours")
            or _find(t3, "fit_b_gpu_hours"), "measured",
            "results/run_3b_gdn/02_table3_jlens_validation.json:fit_b_gpu_hours", "jlens",
            note="Run 007."),
        row("J-lens map, Qwen2.5-3B, corpus B (1000 ctx, fresh)", box.split(",")[0], box,
            _find(t3k, "fit_b_gpu_hours"), "measured",
            "results/run_3b_gdn/02_table3_jlens_validation_1000ctx.json:fit_b_gpu_hours", "jlens",
            note="4.7x the 500-ctx cost for 2x the contexts; likely MIG contention on a shared "
                 "box. Re-measure uncontended before it feeds the 7B decision."),
        row("J-lens map, Qwen2.5-3B, corpus A (1000 ctx, RESUMED)", box.split(",")[0], box,
            t3k.get("map_cost_gpu_hours"), "measured",
            "results/run_3b_gdn/02_table3_jlens_validation_1000ctx.json:map_cost_gpu_hours", "jlens",
            note="Resumed from a partial checkpoint, so this UNDERSTATES a fresh fit. Do not sum "
                 "it with the fresh corpus-B figure as a two-corpus cost."),
        row("J-lens map, Qwen2.5-7B", None, None, None, "not_run", "gated on Table 7", "jlens"),
        row("Table 3 check 1 (final-layer identity) -- fit_layer35.py; jlens refused J35", "A100-40GB",
            modal, None, "todo",
            "Modal dashboard: apps ahn-layer35 (2 stopped runs, 2026-09-20). Both aborted at the fit "
            "step (no jlens pkg, then source<target); only the reference-KL pass ran.", "jlens",
            note="Small (model load + 8 forward passes x2 attempts). Pull billed seconds."),
    ]

    # --- control conditions (Table 4, nb04 RULER battery) ----------------------------------
    rows += [
        row("Control battery C1-C4, GDN (nb04 04i, config 16384)", "H100 MIG", box, 0.2, "estimated",
            "tracker row 78: stdout not saved; same 60x2 cond x 3 layers as DN", "controls", vram="11-14 (slice headroom)"),
        row("Control battery C1-C4, DeltaNet (nb04 04i incl. C4 layer-perm)", "H100 MIG", box,
            round(11.6 / H, 3), "measured", "run log: 11.6 min (tracker row 78)", "controls",
            vram="11-14 (slice headroom)"),
        row("Control battery C1-C4, Mamba2 (nb04 RULER path)", "A100-40GB", modal, None, "todo",
            "Modal dashboard: ahn-m2 app, `run --job nb04` run", "controls",
            note="Real mamba kernels, so NOT comparable to the box MIG rows."),
    ]

    # --- behavioural evaluation ---------------------------------------------------------------
    rows += [
        row("RQ1 NOWRITE reproduction (nb03), GDN", "H100 MIG", box, None, "todo",
            "not logged in the tracker; scrollback / results/run_3b_gdn/03_*.json timestamps", "eval"),
        row("RQ1 NOWRITE reproduction (nb03), DeltaNet (LongBench-E, 60x2, <=32 tok)", "H100 MIG", box,
            "0.3-0.5", "estimated", "tracker row 78", "eval", vram="8-11 (slice headroom)"),
        row("RQ1 NOWRITE reproduction (nb03), Mamba2", "A100-40GB", modal, round(10.5 / H, 3),
            "measured", "run 031 stdout: 10.5 min wall (tracker row 78)", "eval"),
    ]

    fl = _load("run_3b_floor/06_no_ahn_floor.json")
    fr = _wall("run_3b_floor/06_no_ahn_floor.json", "ruler_niah", "summary", "wall_min") \
        if fl else None
    fh = _wall("run_3b_floor/06_no_ahn_floor.json", "hotpot_qa", "summary", "wall_min") if fl else None
    if fr is None and fl:
        fr = (fl.get("ruler_niah") or {}).get("wall_min")
    fnw = (_load("run_3b_floor/06_no_ahn_floor_nowindow.json") or {}).get("ruler_niah", {})
    fnw = fnw.get("wall_min") or (fnw.get("summary") or {}).get("wall_min")
    rows += [
        row("No-AHN floor, RULER n=60 + HotpotQA n=60 (run 032, all three cells share it)", "H100 MIG",
            box, round(((fr or 0) + (fh or 0)) / H, 3), "measured",
            f"results/run_3b_floor/06_no_ahn_floor.json:wall_min (RULER {fr:.2f} + HotpotQA {fh:.2f} min)"
            if fr and fh else "results/run_3b_floor/06_no_ahn_floor.json", "eval"),
        row("No-AHN full-context control (RULER only, --no-window)", "H100 MIG", box,
            round((fnw or 0) / H, 3), "measured",
            "results/run_3b_floor/06_no_ahn_floor_nowindow.json:wall_min", "eval"),
    ]

    # --- row 70: AHN-on behavioural RULER ----------------------------------------------------------
    r70 = {}
    for cell in ("gdn", "dn", "m2"):
        r70[cell] = _wall(f"run_3b_{cell}/06_ahn_on_ruler.json", "ruler_niah", "summary", "wall_min")
        rows.append(row(
            f"AHN-on behavioural RULER NIAH n=60 (row 70), {cell.upper()}", "A100-40GB", modal,
            round(r70[cell] / H, 3) if r70[cell] else None, "measured" if r70[cell] else "todo",
            f"results/run_3b_{cell}/06_ahn_on_ruler.json:wall_min (generation only)", "eval",
            note="Excludes container start, checkpoint load and the one-off CPU merge; billed "
                 "Modal time is higher. Pull billed seconds from the dashboard to close."))

    # --- BABILong pilot (Son, 17 Sep): the only run so far with measured peak VRAM ------------
    bb = _load("babilong/07c_babilong_c1c2_full.json")
    if bb and bb.get("run_log"):
        secs = sum(x["seconds"] for x in bb["run_log"])
        rows.append(row(
            "BABILong qa1 64k C1/C2 pilot, GDN, 32 evicted examples", "not recorded", "not recorded",
            round(secs / 3600, 3), "measured",
            "results/babilong/07c_babilong_c1c2_full.json:run_log (sum of per-example seconds)", "eval",
            vram=round(max(x["peak_memory_gb"] for x in bb["run_log"]), 1),
            note="Per-example forward pass only; excludes model/lens load and the earlier "
                 "16k/32k/64k inspection passes."))

    # --- totals: never sum estimates into measured -------------------------------------------------
    def tot(pred):
        return round(sum(r["gpu_hours"] for r in rows if isinstance(r["gpu_hours"], (int, float)) and pred(r)), 3)

    payload = {
        "table": "Table 10 - compute accounting",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "columns": ["artefact", "gpu", "gpu_hours", "peak_vram_gb", "where_it_ran", "status", "source"],
        "rows": rows,
        "totals_gpu_hours": {
            "measured_total": tot(lambda r: r["status"] == "measured"),
            "measured_by_group": {g: tot(lambda r, g=g: r["status"] == "measured" and r["group"] == g)
                                  for g in ("jlens", "controls", "eval")},
            "note": "Measured only; includes the resumed 1000-ctx corpus-A fit (compute really spent, but not a fresh-fit cost). Estimated rows (GDN 04i ~0.2, DN nb03 0.3-0.5) and todo rows are "
                    "NOT included. Hardware is mixed (H100 MIG slices vs Modal A100-40GB, contended vs "
                    "not) so GPU-hours across rows are not interchangeable.",
        },
        "open": [r["artefact"] for r in rows if r["status"] == "todo"],
        "peak_vram": "Not instrumented in nb03/nb04. Add torch.cuda.max_memory_allocated() prints before "
                     "any further run so future rows are measured.",
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "05_table10_compute.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    for r in rows:
        print(f"{r['status']:9s} {str(r['gpu_hours']):>8s}  {r['artefact'][:78]}")
    print("\ntotals:", json.dumps(payload["totals_gpu_hours"]["measured_by_group"]),
          "measured total", payload["totals_gpu_hours"]["measured_total"])
    print("open:", len(payload["open"]), "-> saved", out)


def _find(d, key):
    """First value for `key` anywhere in a nested dict/list."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, key)
            if r is not None:
                return r
    return None


if __name__ == "__main__":
    main()
