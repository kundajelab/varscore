from __future__ import annotations

import csv
import math

import pytest

from varscore.scoring.cherimoya import allele_windows, reduce_variant_scores
from varscore.scoring.cherimoya.cli import main, read_variants


def test_allele_windows_centers_snv_and_keeps_fixed_length():
    sequence = "ACGT" * 25
    ref, alt = allele_windows({"chr1": sequence}, "chr1", 50, "C", "A", in_window=10)
    assert len(ref) == len(alt) == 10
    assert ref[5] == "C"
    assert alt[5] == "A"
    assert ref[:5] == alt[:5]
    assert ref[6:] == alt[6:]


def test_allele_windows_handles_indel_and_contig_edge():
    sequence = "ACGT" * 25
    ref, alt = allele_windows({"chr1": sequence}, "chr1", 50, "C", "CGGG", in_window=12)
    assert len(ref) == len(alt) == 12
    assert alt[6:10] == "CGGG"

    edge_ref, _ = allele_windows(
        {"chr1": "ACGTACGTAC"}, "chr1", 1, "A", "C", in_window=8
    )
    assert edge_ref.startswith("NNNN")
    assert len(edge_ref) == 8


def test_allele_windows_rejects_reference_mismatch():
    with pytest.raises(ValueError, match="reference mismatch"):
        allele_windows({"chr1": "ACGT"}, "chr1", 2, "A", "T", in_window=4)


def test_allele_windows_rejects_symbolic_alleles():
    with pytest.raises(ValueError, match="only A, C, G, T, or N"):
        allele_windows({"chr1": "ACGT"}, "chr1", 1, "A", "<DEL>", in_window=4)


def test_reduce_variant_scores_ignores_missing_values():
    row = reduce_variant_scores(
        "v1",
        [
            {"counts_log2fc": 1.0, "profile_l1": 100.0},
            {"counts_log2fc": 2.0, "profile_l1": None},
            {"counts_log2fc": math.nan, "profile_l1": 300.0},
        ],
    )
    assert row["counts_log2fc"] == pytest.approx(1.5)
    assert row["profile_l1"] == pytest.approx(200.0)


def test_read_variants_uses_canonical_headerless_schema(tmp_path):
    variants = tmp_path / "variants.tsv"
    variants.write_text("chr1\t10\tA\tT\tcustom-id\nchr2\t20\tC\tG\n")
    assert read_variants(str(variants)) == [
        {"chr": "chr1", "pos": 10, "ref": "A", "alt": "T", "variant_id": "custom-id"},
        {"chr": "chr2", "pos": 20, "ref": "C", "alt": "G", "variant_id": "chr2:20:C:G"},
    ]


def test_read_variants_accepts_empty_batch(tmp_path):
    variants = tmp_path / "variants.tsv"
    variants.write_text("")
    assert read_variants(str(variants)) == []


def test_read_variants_rejects_duplicate_ids(tmp_path):
    variants = tmp_path / "variants.tsv"
    variants.write_text("chr1\t10\tA\tT\tv1\nchr1\t20\tC\tG\tv1\n")
    with pytest.raises(ValueError, match="duplicate variant_id 'v1'"):
        read_variants(str(variants))


def test_score_cli_writes_header_for_empty_batch_without_model_dependencies(tmp_path):
    variants = tmp_path / "variants.tsv"
    output = tmp_path / "scores.tsv"
    variants.write_text("")

    main(
        [
            "score",
            "--hf-repo",
            "unused",
            "--hf-filename",
            "unused",
            "--in-window",
            "2114",
            "-g",
            str(tmp_path / "unused.fa"),
            "-v",
            str(variants),
            "-o",
            str(output),
        ]
    )

    assert output.read_text() == "variant_id\tcounts_log2fc\tprofile_l1\n"


def _write_fold(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant_id", "counts_log2fc", "profile_l1"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_cli_accepts_numbered_fold_flags(tmp_path):
    fold_zero = tmp_path / "fold-0.tsv"
    fold_one = tmp_path / "fold-1.tsv"
    output = tmp_path / "nested" / "scores.tsv"
    _write_fold(
        fold_zero, [{"variant_id": "v1", "counts_log2fc": 1.0, "profile_l1": 10.0}]
    )
    _write_fold(
        fold_one, [{"variant_id": "v1", "counts_log2fc": 3.0, "profile_l1": 30.0}]
    )

    main(["summarize", "-f0", str(fold_zero), "-f1", str(fold_one), "-o", str(output)])

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows == [{"variant_id": "v1", "counts_log2fc": "2.0", "profile_l1": "20.0"}]


def test_summarize_rejects_inconsistent_fold_membership(tmp_path):
    fold_zero = tmp_path / "fold-0.tsv"
    fold_one = tmp_path / "fold-1.tsv"
    _write_fold(
        fold_zero, [{"variant_id": "v1", "counts_log2fc": 1.0, "profile_l1": 10.0}]
    )
    _write_fold(
        fold_one, [{"variant_id": "v2", "counts_log2fc": 3.0, "profile_l1": 30.0}]
    )

    with pytest.raises(ValueError, match="inconsistent variants"):
        main(
            [
                "summarize",
                "-f0",
                str(fold_zero),
                "-f1",
                str(fold_one),
                "-o",
                str(tmp_path / "scores.tsv"),
            ]
        )
