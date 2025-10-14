#!/usr/bin/env python3
"""Extract a trimmed GPT-2 vocab/merge fixture for Zeptochat tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import GPT2Tokenizer


def build_fixture(output_dir: Path, vocab_limit: int, merge_limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # encoder is an OrderedDict sorted by rank already
    encoder_dict = tokenizer.encoder
    encoder_items = list(encoder_dict.items())
    trimmed_encoder = encoder_items[:vocab_limit]

    required_tokens = [
        "Ġthe",
        "Ġand",
        "ĠHello",
        "Ġworld",
        "the",
        "and",
        "hello",
        "world",
    ]

    kept_tokens = {token for token, _ in trimmed_encoder}
    for token in required_tokens:
        if token in encoder_dict and token not in kept_tokens:
            trimmed_encoder.append((token, encoder_dict[token]))
            kept_tokens.add(token)

    trimmed_encoder.sort(key=lambda item: item[1])

    # Maintain a stable order by rank
    vocab_path = output_dir / "vocab.txt"
    with vocab_path.open("w", encoding="utf-8") as vf:
        for token, idx in trimmed_encoder:
            vf.write(f"{idx} {token}\n")

    # Merge ranks is dict: pair -> rank
    kept_set = {token for token, _ in trimmed_encoder}
    kept_set.update(required_tokens)

    merge_items = sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])
    kept_merges: list[tuple[str, str]] = []
    for (left, right), rank in merge_items:
        if left in kept_set and right in kept_set:
            kept_merges.append((left, right))
        if merge_limit > 0 and len(kept_merges) >= merge_limit:
            break

    merges_path = output_dir / "merges.txt"
    with merges_path.open("w", encoding="utf-8") as mf:
        mf.write("#version: 0.2\n")
        for left, right in kept_merges:
            mf.write(f"{left} {right}\n")

    metadata = {
        "source": "gpt2",
        "vocab_size": len(trimmed_encoder),
        "merge_count": len(kept_merges),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/gpt2_mini"),
        help="Directory to write fixtures into.",
    )
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=512,
        help="Number of tokens to keep from GPT-2 vocab.",
    )
    parser.add_argument(
        "--merge-limit",
        type=int,
        default=-1,
        help="Number of merges to keep (ordered by rank). Use -1 for all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_fixture(args.output, args.vocab_limit, args.merge_limit)


if __name__ == "__main__":
    main()
