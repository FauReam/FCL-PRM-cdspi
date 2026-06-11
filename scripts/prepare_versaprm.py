"""Convert UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled to VersaPRMLoader format.

The HF dataset has columns:
    question, answer, category, src, id, chain_of_thoughts, labels,
    cot_id, parsed_answer, parsed_answer_correctness

`chain_of_thoughts` and `labels` are Python-repr strings of lists.

VersaPRMLoader expects JSONL with:
    {"domain": str, "question": str, "steps": list[str], "labels": list[int]}

We additionally map 14 MMLU-Pro categories onto a 4-domain federated split
(math / code / medical / general) so configs/m*.yaml work out of the box;
the original `category` is preserved alongside `domain` for finer splits.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from datasets import load_dataset

CATEGORY_TO_DOMAIN = {
    # Direct hits
    "math": "math",
    # Code-ish
    "computer science": "code",
    "engineering": "code",
    # Medical-ish
    "health": "medical",
    "biology": "medical",
    # Everything else falls into "general"
    "psychology": "general",
    "other": "general",
    "law": "general",
    "economics": "general",
    "business": "general",
    "physics": "general",
    "chemistry": "general",
    "philosophy": "general",
    "history": "general",
}

OUT_PATH = Path("data/versaprm/versa_prm.jsonl")


def _parse_list(field: str) -> list:
    if field is None:
        return []
    if isinstance(field, list):
        return field
    return ast.literal_eval(field)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[versa] loading HF split...")
    ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")
    print(f"[versa] {len(ds)} rows")

    written = 0
    skipped = 0
    domain_counts: dict[str, int] = {}
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for row in ds:
            try:
                steps = _parse_list(row["chain_of_thoughts"])
                labels = _parse_list(row["labels"])
            except (SyntaxError, ValueError):
                skipped += 1
                continue
            if not steps or len(steps) != len(labels):
                skipped += 1
                continue
            cat = (row["category"] or "").lower()
            domain = CATEGORY_TO_DOMAIN.get(cat, "general")
            sample = {
                "domain": domain,
                "category": cat,
                "question": row["question"],
                "steps": steps,
                "labels": labels,
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"[versa] wrote {written} rows ({skipped} skipped) -> {OUT_PATH}")
    for d in ["math", "code", "medical", "general"]:
        print(f"  {d:8s} {domain_counts.get(d, 0)}")


if __name__ == "__main__":
    main()
