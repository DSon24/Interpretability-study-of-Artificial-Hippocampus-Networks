#!/usr/bin/env python3
"""
P0-5.11 mixed-distribution J-Lens transfer pilot.

Official benchmark content only:
- BABILong: RMT-team/babilong, QA1, configs 0k and 1k
- NoLiMa: official needle_set.json + rand_book_1..5
- NoLiMa configs: context_length 250 and 1000

Custom experimental design:
- 8 fit + 8 held-out prompts from each of four distributions
- layers 9, 18, 27
- max_seq_len 2048
- held-out metric: rank of Qwen final-layer top-1 token

No benchmark facts/questions/haystacks are LLM-generated.
"""

import argparse
import hashlib
import json
from copy import copy
from pathlib import Path

import numpy as np
import torch
import jlens
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from book_haystack import BookHaystack


MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYERS = [9, 18, 27]
MAX_SEQ_LEN = 2048

LENS_PATH = Path(
    "results/validation/jlens_mixed_babi_nolima_multilen.pt"
)
CKPT_PATH = Path(
    "results/validation/jlens_mixed_babi_nolima_multilen.ckpt"
)

# Verified byte-for-byte against official NoLiMa HF files.
OFFICIAL_NOLIMA_SHA256 = {
    "needle_set.json":
        "b197f5633668be11aa5855144716b3009d6246c5fb05a508c37db0d08303aa99",
    "rand_book_1.txt":
        "1f7bdc697141b888565c851e6c34dbd649aaf580280eae92199561c4f0e93dab",
    "rand_book_2.txt":
        "e77a1c8c52b31b0cb02b4b78af2e4f662ef7baaf0cfca52d84b33c710e353357",
    "rand_book_3.txt":
        "3a45fdf98871e0a0e4924037661646b6ec20ecea490e2195fbd7103582782010",
    "rand_book_4.txt":
        "9cf0ee84b8969c7a6a3b607d241af6b4f4dd4d6ab13ae8464d8cd4787ad5f618",
    "rand_book_5.txt":
        "0424d75836f692d04c07c14dad362828e89ff69c802c7021774a32ec289a62b9",
}


def sha_text(x):
    return hashlib.sha256(x.encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ============================================================
# BABILONG OFFICIAL QA1 PROMPT
# ============================================================

INSTRUCTION = (
    "I will give you context with the facts about positions of different persons "
    "hidden in some random text and a question. You need to answer the question "
    "based only on the information from the facts. If a person was in different "
    "locations, use the latest location to answer the question."
)

EXAMPLES = (
    "<example>\n"
    "Charlie went to the hallway. Judith come back to the kitchen. "
    "Charlie travelled to balcony. Where is Charlie?\n"
    "Answer: The most recent location of Charlie is balcony.\n"
    "</example>\n\n"
    "<example>\n"
    "Alan moved to the garage. Charlie went to the beach. Alan went to the shop. "
    "Rouse travelled to balcony. Where is Alan?\n"
    "Answer: The most recent location of Alan is shop.\n"
    "</example>"
)

POST = (
    "Always return your answer in the following format: "
    "The most recent location of ’person’ is ’location’. "
    "Do not write anything else after that."
)


def babi_prefix(tok, row):
    user = (
        f"{INSTRUCTION}\n\n"
        f"{EXAMPLES}\n\n"
        f"{POST}\n\n"
        f"<context>\n{row['input'].strip()}\n</context>\n\n"
        f"Question: {row['question']}"
    )

    person = (
        row["question"]
        .replace("Where is ", "")
        .replace("?", "")
        .strip()
    )

    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )

    return prefix + f"The most recent location of {person} is"


# ============================================================
# NOLIMA OFFICIAL TEMPLATE
# ============================================================

DEFAULT_TEMPLATE = (
    "You will answer a question based on the following book snippet:\n\n"
    "{haystack}\n\n"
    "Use the information provided in the book snippet to answer the question. "
    "Your answer should be short and based on either explicitly stated facts "
    "or strong, logical inferences.\n\n"
    "Question: {question}\n\n"
    " Return only the final answer with no additional explanation or reasoning."
)


def verify_nolima_files(root):
    root = Path(root)

    for name, expected in OFFICIAL_NOLIMA_SHA256.items():
        path = root / name
        got = sha_file(path)

        if got != expected:
            raise RuntimeError(
                f"NoLiMa provenance failure: {name}\n"
                f"expected {expected}\n"
                f"got      {got}"
            )

    print("NoLiMa official-file SHA256 check: PASS")


def load_nolima_tests(root):
    with open(Path(root) / "needle_set.json") as f:
        needle_set = json.load(f)

    tests = []

    for exp in needle_set:
        for qtype, qraw in exp["questions"].items():
            for test_id, test in exp["tests"].items():

                needle = copy(exp["needle"])
                question = copy(qraw)

                for i, arg in enumerate(test["input_args"]):
                    ph = "{" + str(i + 1) + "}"
                    needle = needle.replace(ph, str(arg))
                    question = question.replace(ph, str(arg))

                tests.append({
                    "id": f"{exp['id']}_{test_id}_{qtype}",
                    "exp_id": exp["id"],
                    "needle": needle,
                    "question": question,
                    "character_set": exp.get("character_set", []),
                    "system": exp.get("system_prompt", ""),
                    "template": exp.get(
                        "task_template",
                        DEFAULT_TEMPLATE,
                    ),
                })

    return tests


def make_nolima_prompt(
    tok,
    test,
    context_length,
    book_no,
    depth_idx,
    root,
):
    depth = float(
        np.linspace(0, 100, 26)[depth_idx] / 100.0
    )

    book = BookHaystack(
        str(Path(root) / f"rand_book_{book_no + 1}.txt")
    )

    def encode(text):
        return tok.encode(
            text,
            add_special_tokens=False,
        )

    def decode(ids):
        return tok.decode(
            ids,
            skip_special_tokens=False,
        )

    def count(text):
        return len(encode(text))

    rng = np.random.RandomState(
        42 + int(test["exp_id"][:4]) + book_no
    )

    selected_char = None

    # Reproduce official sequential RNG through this depth.
    for _ in range(depth_idx + 1):
        if "{CHAR}" in test["needle"]:
            selected_char = str(
                rng.choice(test["character_set"])
            )

    needle = test["needle"]
    question = test["question"]

    if selected_char is not None:
        needle = needle.replace(
            "{CHAR}",
            selected_char,
        )
        question = question.replace(
            "{CHAR}",
            selected_char,
        )

    placement = book.generate_w_needle_placement(
        needle=needle,
        token_count_func=count,
        encoding_func=encode,
        decoding_func=decode,
        context_length=context_length,
        depth=depth,
        shift=0,
        static_depth=-1,
    )

    user = test["template"].format(
        haystack=placement["text"],
        question=question,
    )

    messages = []

    if test["system"]:
        messages.append({
            "role": "system",
            "content": test["system"],
        })

    messages.append({
        "role": "user",
        "content": user,
    })

    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# EXACT DETERMINISTIC SPLIT USED IN P0-5.11
# ============================================================

def build_split(tok, root):
    train = []
    heldout = []

    for config in ["0k", "1k"]:

        ds = load_dataset(
            "RMT-team/babilong",
            config,
            split="qa1",
        )

        ordered = sorted(
            range(len(ds)),
            key=lambda i: sha_text(
                f"BABI|{config}|{i}"
            ),
        )

        for i in ordered[:8]:
            train.append({
                "source": f"BABILong-{config}",
                "id": str(i),
                "prefix": babi_prefix(tok, ds[i]),
            })

        for i in ordered[8:16]:
            heldout.append({
                "source": f"BABILong-{config}",
                "id": str(i),
                "prefix": babi_prefix(tok, ds[i]),
            })

    tests = load_nolima_tests(root)

    for context_length in [250, 1000]:

        candidates = []

        for test_idx, test in enumerate(tests):

            book_no = test_idx % 5
            depth_idx = (test_idx * 7) % 26

            prefix = make_nolima_prompt(
                tok,
                test,
                context_length,
                book_no,
                depth_idx,
                root,
            )

            candidates.append({
                "source": f"NoLiMa-{context_length}",
                "id": (
                    f"{test['id']}|"
                    f"book{book_no + 1}|"
                    f"depth{depth_idx}"
                ),
                "prefix": prefix,
            })

        candidates = sorted(
            candidates,
            key=lambda x: sha_text(
                f"NOLIMA|{context_length}|{x['id']}"
            ),
        )

        train.extend(candidates[:8])
        heldout.extend(candidates[8:16])

    assert len(train) == 32
    assert len(heldout) == 32

    assert {
        sha_text(x["prefix"]) for x in train
    }.isdisjoint({
        sha_text(x["prefix"]) for x in heldout
    })

    return train, heldout


def token_count(tok, text):
    return len(
        tok(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


# ============================================================
# FIT
# ============================================================

def fit_lens(tok, train, heldout):

    if LENS_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite {LENS_PATH}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    lm = jlens.from_hf(model, tok)

    fitted = jlens.fit(
        lm,
        prompts=[x["prefix"] for x in train],
        source_layers=LAYERS,
        dim_batch=8,
        max_seq_len=MAX_SEQ_LEN,
        skip_first=4,
        checkpoint_path=str(CKPT_PATH),
        checkpoint_every=4,
        resume=True,
    )

    blob = {
        "jacobians": {
            int(k): v.detach().cpu().float()
            for k, v in fitted.jacobians.items()
        },
        "meta": {
            "model": MODEL,
            "source_layers": LAYERS,
            "max_seq_len": MAX_SEQ_LEN,
            "skip_first": 4,
            "dim_batch": 8,
            "n_train": len(train),
            "n_heldout": len(heldout),
            "train": [
                {
                    "source": x["source"],
                    "id": x["id"],
                    "tokens": token_count(
                        tok,
                        x["prefix"],
                    ),
                    "prompt_hash":
                        sha_text(x["prefix"]),
                }
                for x in train
            ],
            "heldout": [
                {
                    "source": x["source"],
                    "id": x["id"],
                    "tokens": token_count(
                        tok,
                        x["prefix"],
                    ),
                    "prompt_hash":
                        sha_text(x["prefix"]),
                }
                for x in heldout
            ],
        },
    }

    torch.save(blob, LENS_PATH)

    print("lens:", LENS_PATH)
    print("SHA256:", sha_file(LENS_PATH))


# ============================================================
# EVAL
# ============================================================

def rank_of(logits, token_id):
    target = logits[token_id]
    return int(
        (logits > target).sum().item()
    )


def describe(values):
    a = np.asarray(values)

    return {
        "median_rank": float(np.median(a)),
        "top1": int(np.sum(a == 0)),
        "top10": int(np.sum(a < 10)),
        "top100": int(np.sum(a < 100)),
        "n": len(a),
    }


def evaluate(tok, heldout):

    blob = torch.load(
        LENS_PATH,
        map_location="cpu",
        weights_only=True,
    )

    lens = jlens.JacobianLens(
        jacobians=blob["jacobians"],
        n_prompts=blob["meta"]["n_train"],
        d_model=2048,
    )

    saved = blob["meta"]["heldout"]

    for got, expected in zip(heldout, saved):
        assert got["source"] == expected["source"]
        assert got["id"] == expected["id"]
        assert (
            sha_text(got["prefix"])
            == expected["prompt_hash"]
        )

    print("HELDOUT PROVENANCE CHECK: PASS")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    lm = jlens.from_hf(model, tok)

    sources = [
        "BABILong-0k",
        "BABILong-1k",
        "NoLiMa-250",
        "NoLiMa-1000",
    ]

    results = {
        s: {
            L: {
                "jlens": [],
                "plain": [],
            }
            for L in LAYERS
        }
        for s in sources
    }

    final_layer = lm.n_layers - 1
    record_at = sorted(
        set(LAYERS) | {final_layer}
    )

    with torch.no_grad():

        for row in heldout:

            input_ids = lm.encode(
                row["prefix"],
                max_length=MAX_SEQ_LEN,
            )

            with jlens.ActivationRecorder(
                lm.layers,
                at=record_at,
            ) as recorder:

                lm.forward(input_ids)

                acts = {
                    L: recorder.activations[L].detach()
                    for L in record_at
                }

            h_final = (
                acts[final_layer][0, -1]
                .float()
            )

            model_logits = (
                lm.unembed(h_final)
                .float()
                .cpu()
            )

            model_top1 = int(
                torch.argmax(
                    model_logits
                ).item()
            )

            for L in LAYERS:

                h = (
                    acts[L][0, -1]
                    .float()
                )

                plain_logits = (
                    lm.unembed(h)
                    .float()
                    .cpu()
                )

                transported = lens.transport(
                    h,
                    L,
                )

                j_logits = (
                    lm.unembed(transported)
                    .float()
                    .cpu()
                )

                results[row["source"]][L]["plain"].append(
                    rank_of(
                        plain_logits,
                        model_top1,
                    )
                )

                results[row["source"]][L]["jlens"].append(
                    rank_of(
                        j_logits,
                        model_top1,
                    )
                )

    print("\nGLOBAL 32-PROMPT SUMMARY")

    for L in LAYERS:

        j_all = []
        p_all = []

        for source in sources:
            j_all.extend(
                results[source][L]["jlens"]
            )
            p_all.extend(
                results[source][L]["plain"]
            )

        print(
            f"L{L:02d} J-LENS:",
            describe(j_all),
        )

        print(
            f"L{L:02d} PLAIN :",
            describe(p_all),
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["fit", "eval"],
        required=True,
    )

    parser.add_argument(
        "--nolima-root",
        default="/tmp/nolima_pilot",
    )

    args = parser.parse_args()

    verify_nolima_files(
        args.nolima_root
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL
    )

    train, heldout = build_split(
        tok,
        args.nolima_root,
    )

    max_tokens = max(
        token_count(
            tok,
            x["prefix"],
        )
        for x in train + heldout
    )

    print("train:", len(train))
    print("heldout:", len(heldout))
    print("max tokens:", max_tokens)

    assert max_tokens <= MAX_SEQ_LEN

    if args.mode == "fit":
        fit_lens(
            tok,
            train,
            heldout,
        )

    elif args.mode == "eval":
        evaluate(
            tok,
            heldout,
        )


if __name__ == "__main__":
    main()
