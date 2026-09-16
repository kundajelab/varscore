"""Command-line runtime for fold-sharded Cherimoya variant scoring."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Sequence
from typing import Any

import pandas as pd

from varscore.core.io import load_variants
from varscore.scoring.cherimoya.core import (
    SCORE_COLUMNS,
    allele_windows,
    reduce_variant_scores,
)


SCORE_FIELDS = ("variant_id", *SCORE_COLUMNS)


def read_variants(path: str) -> list[dict[str, Any]]:
    """Read varscore's canonical headerless ``chr,pos,ref,alt[,variant_id]`` TSV."""
    try:
        frame = load_variants(path)
    except pd.errors.EmptyDataError:
        return []
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in frame.to_dict(orient="records"):
        chrom = str(record["chr"])
        pos = int(record["pos"])
        ref = str(record["ref"])
        alt = str(record["alt"])
        raw_id = record.get("variant_id")
        variant_id = f"{chrom}:{pos}:{ref}:{alt}" if pd.isna(raw_id) else str(raw_id)
        if variant_id in seen_ids:
            raise ValueError(f"{path} contains duplicate variant_id {variant_id!r}")
        seen_ids.add(variant_id)
        rows.append(
            {
                "chr": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "variant_id": variant_id,
            }
        )
    return rows


def read_fold_scores(path: str) -> tuple[list[str], dict[str, dict[str, float | None]]]:
    """Read one headered fold-score TSV while retaining its variant order."""
    order: list[str] = []
    scores: dict[str, dict[str, float | None]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SCORE_FIELDS:
            raise ValueError(
                f"{path} has score fields {reader.fieldnames!r}; expected {list(SCORE_FIELDS)!r}"
            )
        for row in reader:
            variant_id = row["variant_id"]
            if variant_id in scores:
                raise ValueError(f"{path} contains duplicate variant_id {variant_id!r}")
            order.append(variant_id)
            scores[variant_id] = {
                column: None if row[column] == "" else float(row[column])
                for column in SCORE_COLUMNS
            }
    return order, scores


def write_scores(path: str, rows: Sequence[dict[str, object]]) -> None:
    """Write a headered score TSV, creating its parent directory when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCORE_FIELDS), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SCORE_FIELDS})


def score(args: argparse.Namespace) -> None:
    """Score every input variant using one Cherimoya fold."""
    variants = read_variants(args.variants)
    if not variants:
        write_scores(args.output, [])
        return

    import torch
    from cherimoya import Cherimoya
    from huggingface_hub import hf_hub_download
    from pyfaidx import Fasta
    from tangermeme.predict import predict
    from tangermeme.utils import one_hot_encode

    weights_path = hf_hub_download(
        repo_id=args.hf_repo,
        filename=args.hf_filename,
        revision=args.hf_revision,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Cherimoya.load(weights_path, device=device).eval()
    genome = Fasta(args.genome)

    ref_onehots = []
    alt_onehots = []
    for variant in variants:
        ref_sequence, alt_sequence = allele_windows(
            genome,
            variant["chr"],
            variant["pos"],
            variant["ref"],
            variant["alt"],
            in_window=args.in_window,
        )
        ref_onehots.append(one_hot_encode(ref_sequence))
        alt_onehots.append(one_hot_encode(alt_sequence))

    ref_batch = torch.stack(ref_onehots).float()
    alt_batch = torch.stack(alt_onehots).float()
    profile_ref, counts_ref = predict(
        model, ref_batch, batch_size=args.batch_size, device=device
    )
    profile_alt, counts_alt = predict(
        model, alt_batch, batch_size=args.batch_size, device=device
    )

    log_two = math.log(2)
    write_scores(
        args.output,
        [
            {
                "variant_id": variant["variant_id"],
                "counts_log2fc": float(
                    (counts_alt[index].sum() - counts_ref[index].sum()) / log_two
                ),
                "profile_l1": float(
                    (profile_alt[index] - profile_ref[index]).abs().sum()
                ),
            }
            for index, variant in enumerate(variants)
        ],
    )


def summarize(args: argparse.Namespace) -> None:
    """Validate and average a complete set of per-fold score TSVs."""
    folds = [read_fold_scores(path) for path in args.folds]
    order = folds[0][0]
    expected_ids = set(order)
    for path, (fold_order, fold_scores) in zip(args.folds[1:], folds[1:]):
        actual_ids = set(fold_scores)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"{path} has inconsistent variants; missing={missing}, extra={extra}"
            )
        if len(fold_order) != len(order):
            raise ValueError(f"{path} has an inconsistent number of variants")

    rows = [
        reduce_variant_scores(
            variant_id, [fold_scores[variant_id] for _, fold_scores in folds]
        )
        for variant_id in order
    ]
    write_scores(args.output, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cherimoya-score", description="Cherimoya variant scoring"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    score_parser = commands.add_parser("score", help="score variants with one fold")
    score_parser.add_argument("--hf-repo", required=True)
    score_parser.add_argument("--hf-filename", required=True)
    score_parser.add_argument("--hf-revision", default="main")
    score_parser.add_argument("--in-window", type=int, required=True)
    score_parser.add_argument("--batch-size", type=int, default=64)
    score_parser.add_argument("-g", "--genome", required=True)
    score_parser.add_argument("-v", "--variants", required=True)
    score_parser.add_argument("-o", "--output", required=True)
    score_parser.set_defaults(run=score)

    summarize_parser = commands.add_parser("summarize", help="average per-fold scores")
    summarize_parser.add_argument(
        "-f", "--fold", dest="folds", action="append", required=True, metavar="PATH"
    )
    summarize_parser.add_argument("-o", "--output", required=True)
    summarize_parser.set_defaults(run=summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run ``cherimoya-score``; numbered ``-fN`` flags are accepted for Lava compatibility."""
    if argv is None:
        import sys

        argv = sys.argv[1:]
    normalized = [
        "-f" if value.startswith("-f") and value[2:].isdigit() else value
        for value in argv
    ]
    args = build_parser().parse_args(normalized)
    args.run(args)


if __name__ == "__main__":
    main()
