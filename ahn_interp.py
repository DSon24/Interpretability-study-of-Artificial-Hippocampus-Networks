"""
ahn_interp.py — shared instrumentation for
"What Does the Artificial Hippocampus Store?"  (Kim & Nguyen, Algoverse LLM track)

Single flat module on purpose: the GPU box is reached through a Jupyter web UI with
no SSH, so a run means uploading `ahn_interp.py` plus one notebook. No package install,
no relative imports, no path surgery.

WHAT THIS FIXES relative to the 18 Aug pilot
--------------------------------------------
1. Basis.  A forward hook on `layer.ahn` returns `ahn_attn_output`, which lives in the
   concatenated-head space BEFORE `self_attn.o_proj`.  In qwen2_ahn.py the combination is

       attn_output[:, -L:, :] = attn_output[:, -L:, :] + ahn_attn_output
       hidden_states = self.self_attn.o_proj(attn_output)
       hidden_states = residual + hidden_states

   so the residual-stream contribution of the memory is `o_proj(ahn_attn_output)`, not
   `ahn_attn_output`.  On Qwen2.5-3B both are 2048-dim (16 heads x 128), so decoding the
   raw hook output through the unembedding runs without error and returns noise.
   `AHNProbe.o_t(...)` applies o_proj.  Always use it.

2. Final norm.  A logit-lens readout on Qwen must pass through `model.model.norm`
   before `lm_head`.  `readout_logits(...)` does this.

3. NOWRITE.  Zeroing the AHN module output makes the captured AHN vector identically
   zero, so "Delta = o_t(AHN) - o_t(NOWRITE)" collapses to o_t and control C1 becomes
   vacuous.  C1 has to be run on the RESIDUAL STREAM at the same position:
   `resid(AHN) - resid(NOWRITE)`.  `AHNProbe.residual(...)` captures that.

4. Attention sinks.  `in_ahn_seq_len = cache_size - sliding_window - num_attn_sinks`,
   and memory K/V are taken from index `num_attn_sinks` onward.  Tokens inside the sink
   region are never compressed AND stay losslessly visible to attention.  The upstream
   eval scripts use NUM_ATTENTION_SINK=128.  A needle placed at token ~5 is therefore
   very likely never evicted.  `audit_config()` reports this; `build_niah_prompt()`
   places the needle after the sink region by construction.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "SEED", "set_seed", "ModelBundle", "load_ahn_model", "audit_config",
    "AHNProbe", "readout_logits", "token_rank", "token_prob", "readout_entropy",
    "JacobianLens", "fit_jacobian_lens",
    "build_niah_prompt", "single_token_needles", "load_ruler",
    "qa_f1_score", "normalize_answer", "exact_match",
    "bootstrap_ci", "spearman", "fit_exponential_halflife", "variance_decomposition",
    "save_json", "load_json", "RESULTS_DIR", "set_results_dir", "free_cuda",
]

SEED = 20260820


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# results dir
# --------------------------------------------------------------------------------------

RESULTS_DIR = "results"


def set_results_dir(path: str) -> str:
    global RESULTS_DIR
    RESULTS_DIR = path
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def save_json(obj: Any, name: str, results_dir: Optional[str] = None) -> str:
    d = results_dir or RESULTS_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_json(name: str, results_dir: Optional[str] = None) -> Any:
    d = results_dir or RESULTS_DIR
    with open(os.path.join(d, name)) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().float().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    n_layers: int
    hidden: int
    vocab: int
    num_heads: int
    head_dim: int
    sliding_window: Optional[int]
    num_attn_sinks: int
    use_ahn_router: bool
    use_q_proj: bool
    use_normalized_l2: bool
    ahn_layers: List[int]
    ahn_impl: str
    model_path: str

    def summary(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("model", "tokenizer")}
        d["n_ahn_layers"] = len(self.ahn_layers)
        d["ahn_layers"] = self.ahn_layers
        return d


def load_ahn_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    attn_implementation: str = "flash_attention_2",
    sliding_window: Optional[int] = None,
    num_attn_sinks: Optional[int] = None,
) -> ModelBundle:
    """Load a merged Qwen2.5 + AHN checkpoint and register the custom architecture.

    `sliding_window` / `num_attn_sinks` override the checkpoint config. Pass them
    explicitly for every experiment so the setting is recorded rather than inherited.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from ahn.transformer.qwen2_ahn import register_customized_qwen2

    register_customized_qwen2()

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()

    cfg = model.config
    if sliding_window is not None:
        cfg.sliding_window = int(sliding_window)
        cfg.use_sliding_window = True
    if num_attn_sinks is not None:
        cfg.num_attn_sinks = int(num_attn_sinks)

    ahn_layers = [i for i, l in enumerate(model.model.layers) if hasattr(l, "ahn")]
    impl = "unknown"
    if ahn_layers:
        fn = getattr(model.model.layers[ahn_layers[0]].ahn, "fn", None)
        if fn is not None:
            impl = type(fn).__name__

    n_heads = getattr(cfg, "num_attention_heads")
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)

    return ModelBundle(
        model=model,
        tokenizer=tok,
        n_layers=cfg.num_hidden_layers,
        hidden=cfg.hidden_size,
        vocab=model.lm_head.weight.shape[0],
        num_heads=n_heads,
        head_dim=head_dim,
        sliding_window=getattr(cfg, "sliding_window", None),
        num_attn_sinks=int(getattr(cfg, "num_attn_sinks", 0)),
        use_ahn_router=bool(getattr(cfg, "use_ahn_router", False)),
        use_q_proj=bool(getattr(cfg, "use_q_proj", False)),
        use_normalized_l2=bool(getattr(cfg, "use_normalized_l2", True)),
        ahn_layers=ahn_layers,
        ahn_impl=impl,
        model_path=model_path,
    )


def audit_config(b: ModelBundle, verbose: bool = True) -> Dict[str, Any]:
    """Report the config values that silently change what a retention measurement means.

    Every one of these has bitten this project at least once. Run it before every
    experiment and save the output next to the results.
    """
    m = b.model
    o_proj = m.model.layers[b.ahn_layers[0]].self_attn.o_proj if b.ahn_layers else None

    out = {
        "model_path": b.model_path,
        "ahn_implementation": b.ahn_impl,
        "n_layers": b.n_layers,
        "n_ahn_layers": len(b.ahn_layers),
        "hidden": b.hidden,
        "vocab": b.vocab,
        "num_heads": b.num_heads,
        "head_dim": b.head_dim,
        "heads_x_head_dim": b.num_heads * b.head_dim,
        "sliding_window": b.sliding_window,
        "use_sliding_window": bool(getattr(m.config, "use_sliding_window", False)),
        "num_attn_sinks": b.num_attn_sinks,
        "use_ahn_router": b.use_ahn_router,
        "use_q_proj": b.use_q_proj,
        "use_normalized_l2": b.use_normalized_l2,
        "o_proj_has_bias": bool(o_proj is not None and o_proj.bias is not None),
        "lm_head_tied": bool(getattr(m.config, "tie_word_embeddings", False)),
        "dtype": str(next(m.parameters()).dtype),
    }

    warnings: List[str] = []
    if b.num_attn_sinks > 0:
        warnings.append(
            f"num_attn_sinks={b.num_attn_sinks}: the first {b.num_attn_sinks} tokens are "
            "never compressed and stay losslessly visible to attention. Any needle placed "
            "inside that prefix is NOT evicted, whatever the sliding window is. Place "
            "needles after it (build_niah_prompt does this)."
        )
    if b.use_ahn_router:
        warnings.append(
            "use_ahn_router=True: the memory is combined through a learned sigmoid gate, "
            "not a plain sum. o_proj(ahn_out) is then an UPPER BOUND on the contribution, "
            "not the contribution itself. Multiply by the router gate before reading out."
        )
    if out["o_proj_has_bias"]:
        warnings.append(
            "o_proj has a bias: o_proj(a) - o_proj(b) != o_proj(a-b) is still fine "
            "(bias cancels in the difference) but a single-condition readout is offset."
        )
    if b.num_heads * b.head_dim != b.hidden:
        warnings.append(
            "heads*head_dim != hidden: the pre-o_proj space and the residual stream have "
            "different dimensions. Good news, a basis mistake will now raise instead of "
            "silently returning noise."
        )
    if b.sliding_window is None or not out["use_sliding_window"]:
        warnings.append("sliding_window is unset/disabled: AHN will never activate.")

    out["warnings"] = warnings

    if verbose:
        print("=== CONFIG AUDIT ===")
        for k, v in out.items():
            if k != "warnings":
                print(f"  {k:24s} {v}")
        if warnings:
            print("\n--- WARNINGS ---")
            for w in warnings:
                print(f"  ! {w}")
        else:
            print("\n  no warnings")
    return out


# --------------------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------------------

class AHNProbe:
    """One forward pass, three views of every AHN layer.

    ahn_raw[L]   pre-o_proj AHN output, shape (B, L_ahn, heads*head_dim)
    o_t[L]       residual-stream contribution = o_proj(ahn_raw), shape (B, L_ahn, hidden)
    residual[L]  full decoder-layer output hidden state, shape (B, T, hidden)

    Usage
    -----
        probe = AHNProbe(bundle)
        on  = probe.run(inputs)                 # AHN active
        off = probe.run(inputs, nowrite=True)   # writes suppressed

        # the memory's own contribution, correct basis:
        v = on.o_t(18, pos=-1)

        # control C1 lives here, NOT on o_t (which is 0 by construction under NOWRITE):
        d = on.residual(18, pos=-1) - off.residual(18, pos=-1)

    Hooks are installed and removed per call; nothing is left attached to the model.
    """

    def __init__(self, bundle: ModelBundle):
        self.b = bundle
        self.model = bundle.model

    # -- capture container ------------------------------------------------------------
    class Capture:
        def __init__(self, bundle: ModelBundle, nowrite: bool):
            self.b = bundle
            self.nowrite = nowrite
            self.ahn_raw: Dict[int, torch.Tensor] = {}
            self._resid: Dict[int, torch.Tensor] = {}
            self.n_tokens: int = 0
            self.ahn_active: bool = False
            self.logits: Optional[torch.Tensor] = None

        def _pick(self, t: torch.Tensor, pos: Optional[int]) -> torch.Tensor:
            t = t[0] if t.dim() == 3 else t
            return t if pos is None else t[pos]

        def raw(self, layer: int, pos: Optional[int] = -1) -> torch.Tensor:
            if layer not in self.ahn_raw:
                raise KeyError(
                    f"layer {layer} not captured. AHN active only when "
                    f"n_tokens > sliding_window + num_attn_sinks "
                    f"({self.b.sliding_window} + {self.b.num_attn_sinks}); "
                    f"this prompt was {self.n_tokens} tokens."
                )
            return self._pick(self.ahn_raw[layer], pos)

        def o_t(self, layer: int, pos: Optional[int] = -1) -> torch.Tensor:
            """Residual-stream contribution of compressed memory: o_proj(ahn_raw)."""
            v = self.raw(layer, pos=None)
            o_proj = self.b.model.model.layers[layer].self_attn.o_proj
            with torch.no_grad():
                out = o_proj(v.to(o_proj.weight.dtype))
            return self._pick(out, pos)

        def residual(self, layer: int, pos: Optional[int] = -1) -> torch.Tensor:
            if layer not in self._resid:
                raise KeyError(f"residual for layer {layer} not captured")
            return self._pick(self._resid[layer], pos)

        @property
        def captured_layers(self) -> List[int]:
            return sorted(self.ahn_raw.keys())

    # -- the run ----------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        nowrite: bool = False,
        layers: Optional[Sequence[int]] = None,
        capture_residual: bool = True,
        keep_logits: bool = False,
    ) -> "AHNProbe.Capture":
        b = self.b
        target = list(layers) if layers is not None else list(b.ahn_layers)
        cap = AHNProbe.Capture(b, nowrite)
        cap.n_tokens = int(inputs["input_ids"].shape[1])
        handles = []

        def make_zero(store_idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + out[1:]
                return torch.zeros_like(out)
            return hook

        def make_capture(idx):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cap.ahn_raw[idx] = t.detach().clone()
            return hook

        def make_resid(idx):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cap._resid[idx] = t.detach().clone()
            return hook

        for i in target:
            layer = self.model.model.layers[i]
            if not hasattr(layer, "ahn"):
                continue
            # order matters: NOWRITE must fire first so the capture sees the zeroed value
            # AND so downstream layers actually run without memory.
            if nowrite:
                handles.append(layer.ahn.register_forward_hook(make_zero(i)))
            handles.append(layer.ahn.register_forward_hook(make_capture(i)))
            if capture_residual:
                handles.append(layer.register_forward_hook(make_resid(i)))

        try:
            out = self.model(**inputs, use_cache=True)
            if keep_logits:
                cap.logits = out.logits.detach().float()
        finally:
            for h in handles:
                h.remove()

        cap.ahn_active = len(cap.ahn_raw) > 0 and any(
            v.numel() > 0 for v in cap.ahn_raw.values()
        )
        return cap

    # -- the Stage-1 gate -------------------------------------------------------------
    @torch.no_grad()
    def verify_isolation(
        self,
        inputs: Dict[str, torch.Tensor],
        layer: int,
        atol: float = 5e-2,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Prove the hook captures the memory's actual residual-stream contribution.

        The claim being tested, straight from qwen2_ahn.py:

            resid_AHN(L) - resid_NOWRITE(L)  ==  o_proj(ahn_raw(L))

        for the FIRST AHN layer only (at deeper layers the two runs have already
        diverged upstream, so the identity no longer holds and a mismatch there is
        expected rather than a bug).

        If this fails on the first AHN layer, nothing downstream is interpretable.
        Tolerance is loose because the forward pass is bf16.
        """
        on = self.run(inputs, nowrite=False, layers=[layer])
        off = self.run(inputs, nowrite=True, layers=[layer])

        lhs = (on.residual(layer, pos=-1).float() - off.residual(layer, pos=-1).float())
        rhs = on.o_t(layer, pos=-1).float()

        num = (lhs - rhs).norm().item()
        den = max(rhs.norm().item(), 1e-9)
        rel = num / den
        cos = torch.nn.functional.cosine_similarity(lhs, rhs, dim=0).item()

        res = {
            "layer": layer,
            "is_first_ahn_layer": bool(self.b.ahn_layers and layer == self.b.ahn_layers[0]),
            "n_tokens": on.n_tokens,
            "ahn_active": on.ahn_active,
            "nowrite_o_t_is_zero": bool(off.raw(layer, pos=-1).abs().sum().item() == 0.0),
            "ahn_raw_norm": float(on.raw(layer, pos=-1).float().norm().item()),
            "o_t_norm": float(rhs.norm().item()),
            "residual_norm": float(on.residual(layer, pos=-1).float().norm().item()),
            "relative_error": float(rel),
            "cosine": float(cos),
            "passed": bool(rel < atol and cos > 0.99),
        }
        if verbose:
            note = "FIRST AHN layer" if res["is_first_ahn_layer"] else "deeper layer, divergence expected"
            print(f"--- isolation check, layer {layer} ({note}) ---")
            for k, v in res.items():
                print(f"  {k:24s} {v}")
            print(f"  => {'PASS' if res['passed'] else 'FAIL'}")
        return res


# --------------------------------------------------------------------------------------
# readout
# --------------------------------------------------------------------------------------

@torch.no_grad()
def readout_logits(
    vec: torch.Tensor,
    bundle: ModelBundle,
    lens: Optional["JacobianLens"] = None,
    layer: Optional[int] = None,
    apply_final_norm: bool = True,
) -> torch.Tensor:
    """Decode a residual-stream vector to vocabulary logits.

    apply_final_norm: Qwen applies model.model.norm before lm_head. Skipping it -- as
    the pilot did -- changes both the scale and the ranking. Keep it on.

    lens: if given, the vector is first transported to the final-layer basis by the
    Jacobian map for `layer`. Without a lens this is a plain logit lens, which reads
    next-token disposition only; the proposal's RQ2 needs the J-lens.
    """
    m = bundle.model
    v = vec.float()
    if lens is not None:
        if layer is None:
            raise ValueError("layer is required when a lens is given")
        v = lens.transport(v, layer)
    if apply_final_norm:
        v = m.model.norm(v.to(m.model.norm.weight.dtype)).float()
    w = m.lm_head.weight.float()
    return v @ w.T


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    """0-indexed rank of token_id. 0 = top-1."""
    return int((logits.argsort(descending=True) == token_id).nonzero()[0].item())


def token_prob(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.softmax(logits.float(), dim=-1)[token_id].item())


def readout_entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float(), dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())


# --------------------------------------------------------------------------------------
# Jacobian lens
# --------------------------------------------------------------------------------------

class JacobianLens:
    """Thin wrapper over saved per-layer Jacobians (d_model x d_model)."""

    def __init__(self, jacobians: Dict[int, torch.Tensor], meta: Optional[Dict] = None):
        self.jacobians = {int(k): v.float() for k, v in jacobians.items()}
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str, map_location: str = "cuda") -> "JacobianLens":
        blob = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(blob, dict) and "jacobians" in blob:
            return cls(blob["jacobians"], blob.get("meta"))
        if isinstance(blob, dict):
            return cls(blob)
        raise ValueError(
            "expected a dict of {layer: J} or {'jacobians':..., 'meta':...}; got a bare "
            "tensor. Re-save with save() so the layer index and fit metadata travel with it."
        )

    def save(self, path: str) -> str:
        torch.save(
            {"jacobians": {k: v.cpu() for k, v in self.jacobians.items()}, "meta": self.meta},
            path,
        )
        return path

    def transport(self, vec: torch.Tensor, layer: int) -> torch.Tensor:
        if layer not in self.jacobians:
            raise KeyError(f"no Jacobian for layer {layer}; have {sorted(self.jacobians)}")
        J = self.jacobians[layer].to(vec.device, dtype=torch.float32)
        return vec.float() @ J.T

    def shuffled(self, seed: int = SEED) -> "JacobianLens":
        """Control C3': row-permuted map. Structure that survives this is an artefact."""
        g = torch.Generator().manual_seed(seed)
        out = {}
        for k, J in self.jacobians.items():
            perm = torch.randperm(J.shape[0], generator=g)
            out[k] = J[perm].clone()
        return JacobianLens(out, {**self.meta, "shuffled_seed": seed})

    def permuted_layers(self) -> "JacobianLens":
        """Control C4: map layer L's Jacobian onto a different layer's index."""
        ks = sorted(self.jacobians)
        rolled = ks[1:] + ks[:1]
        return JacobianLens(
            {k: self.jacobians[r] for k, r in zip(ks, rolled)},
            {**self.meta, "layer_permuted": True},
        )


def fit_jacobian_lens(
    bundle: ModelBundle,
    prompts: Sequence[str],
    source_layers: Sequence[int],
    max_seq_len: int = 256,
    skip_first: int = 4,
    dim_batch: int = 8,
    **kwargs,
) -> JacobianLens:
    """Fit via Anthropic's `jlens` package, with the averaging corpus made explicit.

    The pilot fitted layer 18 from ONE 106-character prompt at max_seq_len=64. The
    averaging step is the entire reason to prefer a J-lens over a logit lens, so a
    one-prompt map is a single-context Jacobian and the corresponding row of Table 3
    (map stability across disjoint corpora) cannot pass. Pass a real corpus here:
    a few hundred contexts, strictly disjoint from every evaluation set.
    """
    import jlens

    if len(prompts) < 50:
        print(
            f"WARNING: fitting on {len(prompts)} prompts. Table 3's map-stability check "
            "needs two disjoint corpora of ~500 contexts each. Treat anything smaller "
            "as a smoke test, not a lens."
        )

    jl_model = jlens.from_hf(bundle.model, bundle.tokenizer)
    fitted = jlens.fit(
        jl_model,
        prompts=list(prompts),
        source_layers=list(source_layers),
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        **kwargs,
    )
    meta = {
        "n_prompts": len(prompts),
        "source_layers": list(source_layers),
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "dim_batch": dim_batch,
        "model_path": bundle.model_path,
    }
    return JacobianLens({int(k): v for k, v in fitted.jacobians.items()}, meta)


# --------------------------------------------------------------------------------------
# needle-in-a-haystack
# --------------------------------------------------------------------------------------

_FILLERS = [
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
    "The quick brown fox jumps over the lazy dog while the kettle boils on the stove. ",
    "In a hole in the ground there lived a hobbit, and it was neither dirty nor wet. ",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor. ",
]


def single_token_needles(tokenizer, candidates: Sequence[str]) -> Dict[str, int]:
    """Keep only needles that encode to exactly one token with a leading space.

    The pilot silently dropped "42" (3 tokens) mid-loop, leaving n=2. Filter up front
    and report, so the cohort size is a decision rather than an accident.
    """
    keep, drop = {}, {}
    for c in candidates:
        ids = tokenizer.encode(f" {c}", add_special_tokens=False)
        (keep if len(ids) == 1 else drop)[c] = ids[0] if len(ids) == 1 else ids
    if drop:
        print(f"dropped multi-token needles: { {k: len(v) for k, v in drop.items()} }")
    print(f"kept {len(keep)} single-token needles: {sorted(keep)}")
    return keep


def _pad_to_length(tokenizer, filler: str, target_tokens: int, filler_toks: int) -> str:
    """Build `filler` repeated so the FULL tokenized string lands near `target_tokens`.

    A single `ceil(target/filler_toks)` repeat count is not reliable: `filler * k`
    is tokenized as ONE string, and BPE merges across every one of the k-1 internal
    repeat boundaries (the period/space/capital-letter junction between consecutive
    copies). That makes long, heavily-repeated padding compress well below what
    `k * filler_toks` predicts — at a few hundred repeats the shortfall can be large
    enough to erase most of a requested eviction distance. Iterate instead of trusting
    the one-shot estimate.
    """
    if target_tokens <= 0:
        return ""
    k = max(1, math.ceil(target_tokens / max(filler_toks, 1)))
    text = filler * k
    n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    tol = max(4, target_tokens // 200)
    guard = 0
    while abs(n - target_tokens) > tol and guard < 25:
        ratio = target_tokens / max(n, 1)
        new_k = max(1, math.ceil(k * ratio))
        if new_k == k:
            new_k = k + (1 if n < target_tokens else -1)
        k = max(1, new_k)
        text = filler * k
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        guard += 1
    return text


def build_niah_prompt(
    tokenizer,
    needle: str,
    bundle: ModelBundle,
    eviction_distance: int,
    in_window: bool = False,
    filler_idx: int = 0,
    question: str = "What was the special word?",
) -> Dict[str, Any]:
    """Build a NIAH prompt with the needle at a CONTROLLED eviction distance.

    eviction_distance = how many tokens past the compression boundary the needle sits.
    The boundary is at num_attn_sinks + sliding_window counted back from the end, and
    the needle must also start after the sink prefix or it is never compressed at all.

    in_window=True places the needle inside the local window instead — the pre-eviction
    baseline (control C4 in the proposal, the ceiling the decay curve is calibrated
    against). In the pilot this baseline came out WORSE than the evicted condition
    (Paris rank 110712 vs ~95000), which is a sign the placement was wrong, not that
    eviction helps.

    Padding is length-corrected (see `_pad_to_length`): naively repeating a filler
    string `ceil(n / filler_toks)` times can undershoot the target by a large fraction
    once BPE merges across the repeat boundaries at scale. `actual_eviction_distance`
    is still measured post-hoc from the real tokenization either way — trust that
    field, not `eviction_distance`, but it should now sit close to what you asked for.
    """
    sinks = bundle.num_attn_sinks
    window = bundle.sliding_window or 0
    filler = _FILLERS[filler_idx % len(_FILLERS)]
    filler_toks = len(tokenizer.encode(filler, add_special_tokens=False))

    def pad(n_tokens: int) -> str:
        return _pad_to_length(tokenizer, filler, n_tokens, filler_toks)

    # prefix long enough to push the needle clear of the attention-sink region
    prefix = pad(sinks + 16)
    needle_sentence = f"The special word is {needle}. "
    suffix_target = window + eviction_distance if not in_window else max(eviction_distance, 0)

    if in_window:
        # needle sits inside the local window: put it near the end
        body = pad(sinks + window + 256)
        prompt = body + needle_sentence + pad(max(suffix_target, 8)) + question
    else:
        prompt = prefix + needle_sentence + pad(suffix_target) + question

    ids = tokenizer(prompt, return_tensors="pt")
    n = int(ids["input_ids"].shape[1])

    # where the needle actually landed
    head_ids = tokenizer(prefix if not in_window else prompt.split(needle_sentence)[0],
                         add_special_tokens=True)["input_ids"]
    needle_pos = len(head_ids)

    # `window_start` is the index where the lossless local window begins -- tokens at
    # index < window_start are candidates for compression (once past the sink prefix).
    # An earlier version of this function subtracted `sinks` a second time here
    # (`n - window - sinks`), which is actually the COUNT of compressed tokens, not an
    # index -- using it as the index threshold silently shrank every reported eviction
    # distance by exactly `num_attn_sinks` tokens. Fixed: no second subtraction.
    window_start = n - window
    return {
        "prompt": prompt,
        "needle": needle,
        "n_tokens": n,
        "needle_pos": needle_pos,
        "compression_boundary": window_start,
        "actual_eviction_distance": window_start - needle_pos,
        "needle_is_evicted": bool(sinks <= needle_pos < window_start),
        "needle_in_sink_region": bool(needle_pos < sinks),
        "ahn_will_activate": bool(n > window + sinks),
        "in_window": in_window,
        "filler_idx": filler_idx,
    }


def load_ruler(config: str = "8192", split: str = "test", n: int = 60, seed: int = SEED):
    """Stream RULER NIAH examples. Returns list of {prompt, answer, task, n_chars}."""
    from datasets import load_dataset
    ds = load_dataset("simonjegou/ruler", config, split=split, streaming=True)
    out = []
    for ex in ds:
        out.append({
            "prompt": ex["context"] + ex["question"] + ex["answer_prefix"],
            "answer": ex["answer"][0] if isinstance(ex["answer"], list) else ex["answer"],
            "task": ex.get("task", "niah"),
            "max_new_tokens": ex.get("max_new_tokens", 32),
        })
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------------------
# task metrics (LongBench-compatible)
# --------------------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """Return (point, lo, hi). Percentile bootstrap."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.array([stat(v[i]) for i in idx])
    return float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def spearman(x: Sequence[float], y: Sequence[float], n_boot: int = 10000, seed: int = SEED):
    """Spearman rho with a paired bootstrap CI. Returns dict."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"rho": float("nan"), "p": float("nan"), "ci": (float("nan"),) * 2, "n": int(x.size)}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    boots = boots[np.isfinite(boots)]
    return {
        "rho": float(rho), "p": float(p), "n": int(x.size),
        "ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        "excludes_zero": bool(np.quantile(boots, 0.025) > 0 or np.quantile(boots, 0.975) < 0),
    }


def fit_exponential_halflife(
    distances: Sequence[float],
    values: Sequence[float],
    n_boot: int = 2000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fit P(t) = P0 * exp(-lambda t) in log space; return half-life, CI and R^2.

    The R^2 is not decoration. Expected_Tables_and_Figures is explicit: if decay is
    stepped, plateaued or non-monotonic, "half-life" is a meaningless summary and
    Table 6 has to be replaced by a non-parametric statistic. Inspect the figure and
    this R^2 before filling Table 6 in.
    """
    t = np.asarray(distances, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v) & (v > 0)
    t, v = t[ok], v[ok]
    if t.size < 3:
        return {"half_life": float("nan"), "r2": float("nan"), "n": int(t.size),
                "ci": (float("nan"),) * 2, "fit_adequate": False}

    def _fit(tt, vv):
        lam = -np.polyfit(tt, np.log(vv), 1)[0]
        return lam

    lam = _fit(t, v)
    pred = np.exp(np.polyval(np.polyfit(t, np.log(v), 1), t))
    ss_res = float(((v - pred) ** 2).sum())
    ss_tot = float(((v - v.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    hl_boots = []
    for _ in range(n_boot):
        i = rng.integers(0, t.size, t.size)
        if np.unique(t[i]).size < 2:
            continue
        try:
            l = _fit(t[i], v[i])
            if l > 0:
                hl_boots.append(math.log(2) / l)
        except Exception:
            continue
    hl = math.log(2) / lam if lam > 0 else float("inf")
    ci = (
        (float(np.quantile(hl_boots, 0.025)), float(np.quantile(hl_boots, 0.975)))
        if hl_boots else (float("nan"), float("nan"))
    )
    # non-parametric fallback, always reported
    return {
        "half_life": float(hl), "lambda": float(lam), "r2": float(r2),
        "ci": ci, "n": int(t.size),
        "fit_adequate": bool(np.isfinite(r2) and r2 > 0.8 and lam > 0),
        "nonparametric_half_distance": _half_crossing(t, v),
    }


def _half_crossing(t: np.ndarray, v: np.ndarray) -> float:
    """Distance at which v first falls below half its peak. No functional form assumed."""
    order = np.argsort(t)
    t, v = t[order], v[order]
    target = v.max() / 2.0
    below = np.where(v <= target)[0]
    return float(t[below[0]]) if below.size else float("inf")


def variance_decomposition(rows: Sequence[Dict], value_key: str, factors: Sequence[str]) -> Dict:
    """Crude one-way variance-explained per factor (eta^2). Table 9.

    Not a proper mixed model. Adequate for the "is this a cell effect or an example
    effect?" question, which is all Table 9 claims to answer. Say so in the caption.
    """
    y = np.array([r[value_key] for r in rows if np.isfinite(r.get(value_key, np.nan))], dtype=float)
    if y.size < 2:
        return {}
    grand = y.mean()
    ss_tot = float(((y - grand) ** 2).sum())
    out = {}
    for f in factors:
        ss_b = 0.0
        groups: Dict[Any, List[float]] = {}
        for r in rows:
            val = r.get(value_key)
            if val is None or not np.isfinite(val):
                continue
            groups.setdefault(r.get(f), []).append(val)
        for _, g in groups.items():
            g = np.asarray(g)
            ss_b += g.size * (g.mean() - grand) ** 2
        out[f] = float(ss_b / ss_tot) if ss_tot > 0 else float("nan")
    out["residual"] = float(max(0.0, 1.0 - sum(v for k, v in out.items() if k != "residual")))
    return out
