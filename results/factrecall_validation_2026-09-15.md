## Findings from the 15 Sep LV-Eval FactRecall validation

1. The earlier FactRecall `F1 = 0.0` reproduction used a mismatched evaluation setup: `recent_size=8064`, `max_length=15500`, and the first 30 dataset examples. The official AHN LV-Eval configuration is `model_max_len=256000`, `start_size=128`, and `recent_size=32640`.

2. The corrected pipeline was checked against upstream before scaling. The local AHN implementation matches ByteDance upstream apart from Python cache directories, and the AHN tensors in the merged Qwen2.5-3B + AHN-GDN checkpoint exactly match the released ByteDance AHN-GDN checkpoint.

3. A 50-example evenly spaced validation across `factrecall_en_128k` produced:
   - nonzero F1: 11/50
   - mean raw F1: 0.16076
   - reported F1 × 100: 16.08

   This is a validation subset, not the official full 200-example benchmark score.

4. A strong position dependence appears in this subset. The planted fact is outside the 32,640-token recent window for the first 36 sampled cases, and all 36 score zero. Once the fact enters the recent window, 11/14 sampled cases receive nonzero F1, including several exact `Ludwig Beethoven` answers with F1=1.0.

5. The position effect is recorded as an observation only. No mechanistic claim is made yet about AHN compressed-memory failure; further interpretation is deferred until the team decides whether this should take priority over the current execution tasks.

FactRecall initially reproduced `F1=0.0` under the local setup using `recent_size=8064`, `max_length=15500`, and the first 30 examples. We verified that the scorer itself was working, then compared the evaluation code against a fresh ByteDance AHN checkout and found that the main issue was a configuration mismatch: `8064` is the LongBench sliding-window setting, while the official LV-Eval setup uses `model_max_len=256000`, `start_size=128`, and `recent_size=32640`. Long-context generation also required `num_logits_to_keep=1` to avoid unnecessary full-sequence logits allocation and CUDA OOM. We then verified that `config.py`, `eval.py`, the AHN source tree, and the merged AHN-GDN weights match upstream, ruling out a local implementation or checkpoint-merge error. The original `data[:30]` sanity run was also found to be position-biased because the planted `Ludwig Beethoven` fact moves progressively through the context as dataset index increases. Using the corrected official LV-Eval configuration on 50 evenly spaced examples produced 11/50 nonzero scores, mean raw F1 `0.16076`, and `F1×100 = 16.08`. A strong position effect remains: all 36 sampled facts outside the 32,640-token recent window scored zero, while 11/14 facts inside the recent window scored nonzero, including several exact `Ludwig Beethoven` predictions with `F1=1.0`. This is recorded as an observation only; deeper interpretation of the position-dependent behavior is deferred until the team decides whether to prioritize it over the remaining execution tasks.
