"""Pure scientific helpers shared by the Cherimoya CLI and its tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCORE_COLUMNS = ("counts_log2fc", "profile_l1")
_DNA_ALPHABET = frozenset("ACGTN")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def reduce_variant_scores(
    variant_id: str,
    fold_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average Cherimoya's scalar effects across fold records for one variant."""
    return {
        "variant_id": variant_id,
        **{
            column: _mean([_number(record.get(column)) for record in fold_records])
            for column in SCORE_COLUMNS
        },
    }


def allele_windows(
    genome: Any,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    in_window: int,
) -> tuple[str, str]:
    """Build fixed-length reference and alternate sequence windows around a variant.

    ``pos`` is one-based and points at the first base of ``ref``. Sequence missing beyond either contig
    boundary is padded with ``N``. The reference allele is checked against the genome before substitution,
    so an assembly or coordinate mismatch fails rather than producing a plausible but incorrect score.
    """
    if pos < 1:
        raise ValueError(f"variant position must be one-based and positive, got {pos}")
    if not ref or not alt:
        raise ValueError("ref and alt must be non-empty canonical alleles")
    if in_window < 1:
        raise ValueError(f"in_window must be positive, got {in_window}")

    ref = ref.upper()
    alt = alt.upper()
    if not set(ref) <= _DNA_ALPHABET or not set(alt) <= _DNA_ALPHABET:
        raise ValueError(
            f"ref and alt must contain only A, C, G, T, or N; got ref={ref!r}, alt={alt!r}"
        )
    variant_start = pos - 1
    contig = genome[chrom]
    observed_ref = str(contig[variant_start : variant_start + len(ref)]).upper()
    if observed_ref != ref:
        raise ValueError(
            f"reference mismatch at {chrom}:{pos}: input ref={ref!r}, genome={observed_ref!r}"
        )

    start = variant_start - in_window // 2

    def sequence_slice(left: int, right: int) -> str:
        clamped_left = max(left, 0)
        return "N" * (clamped_left - left) + str(contig[clamped_left:right]).upper()

    ref_sequence = sequence_slice(start, start + in_window)[:in_window].ljust(
        in_window, "N"
    )
    prefix = sequence_slice(start, variant_start)
    suffix = sequence_slice(
        variant_start + len(ref),
        variant_start + len(ref) + in_window,
    )
    alt_sequence = (prefix + alt + suffix)[:in_window].ljust(in_window, "N")
    return ref_sequence, alt_sequence
