#!/usr/bin/env python3
"""Pre-filter VersaPRM: drop steps whose tokenized length > max_length.

Avoids training-time truncation which introduces overfitting to incomplete steps.
Produces a cleaned .jsonl with only valid-length steps, keeping the original intact.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def main() -> int:
    src = Path("data/versaprm/versa_prm.jsonl")
    dst = Path("data/versaprm/versa_prm_clean_384.jsonl")
    max_length = 384

    if dst.exists():
        print(f"[SKIP] {dst} already exists")
        return 0

    tok = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-1.4b", local_files_only=True
    )

    lines = src.read_text(encoding="utf-8").strip().split("\n")
    kept_samples = 0
    total_steps = 0
    kept_steps = 0

    with open(dst, "w", encoding="utf-8") as out:
        for line in tqdm(lines, desc="Filtering", unit=" CoT"):
            sample = json.loads(line)
            question = sample.get("question", "")
            steps = sample.get("steps", [])
            labels = sample.get("labels", [])

            clean_steps = []
            clean_labels = []
            for step_text, label in zip(steps, labels):
                total_steps += 1
                text = f"{question}\n{step_text}"
                if len(tok.encode(text)) <= max_length:
                    clean_steps.append(step_text)
                    clean_labels.append(label)
                    kept_steps += 1

            if clean_steps:
                sample["steps"] = clean_steps
                sample["labels"] = clean_labels
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept_samples += 1

    dropped_steps = total_steps - kept_steps
    print(f"  CoT samples: {len(lines)} → {kept_samples} kept "
          f"({100*dropped_steps/total_steps:.1f}% steps dropped)")
    print(f"  Steps:      {total_steps} → {kept_steps} kept "
          f"({100*dropped_steps/total_steps:.1f}% dropped)")
    print(f"  Output:      {dst} ({dst.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
