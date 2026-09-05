"""Contracts for bounded, occurrence-preserving VCF and TSV ingestion."""

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from varscore.preprocessing.streaming import (
    CANONICALIZER_VERSION,
    GVCF_UNSUPPORTED_ERROR_CODE,
    OCCURRENCE_COLUMNS,
    VariantIngestError,
    inspect_vcf_header,
    iter_input_occurrence_batches,
    stream_validate_variants,
)


VCF_HEADER = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=200>
##ALT=<ID=DEL,Description="Deletion">
##INFO=<ID=END,Number=1,Type=Integer,Description="End">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""


@pytest.fixture
def genome(tmp_path):
    path = tmp_path / "genome.fa"
    path.write_text(">chr1\n" + "ACGT" * 50 + "\n")
    return path


@pytest.fixture
def occurrence_vcf(tmp_path):
    path = tmp_path / "input.vcf"
    path.write_text(
        VCF_HEADER
        + "chr1\t20\trsMulti\tT\tC,G\t50\tPASS\t.\n"
        + "chr1\t20\tduplicate\tT\tC,G\t40\tPASS\t.\n"
        + "chr1\t24\t.\tT\t<DEL>,*\t.\tPASS\tEND=30\n"
    )
    return path


def test_occurrence_batches_keep_alt_order_duplicates_and_unsupported_alleles(
    occurrence_vcf,
):
    batches = list(
        iter_input_occurrence_batches(str(occurrence_vcf), "vcf", batch_rows=2)
    )
    rows = pd.concat([batch.rows for batch in batches], ignore_index=True)

    assert all(len(batch.rows) <= 2 for batch in batches)
    assert list(rows.columns) == OCCURRENCE_COLUMNS
    assert rows[["record_ordinal", "alt_index"]].values.tolist() == [
        [0, 1],
        [0, 2],
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
    ]
    assert rows["source_variant_id"].tolist()[:4] == [
        "rsMulti",
        "rsMulti",
        "duplicate",
        "duplicate",
    ]
    assert rows["error_code"].tolist()[-2:] == ["SYMBOLIC_ALT", "SPANNING_DELETION"]
    assert batches[-1].records_seen == 3


@pytest.mark.parametrize(
    "header_line",
    [
        "##GVCFBlock0-5=minGQ=0(maxGQ=5)",
        '##ALT=<ID=NON_REF,Description="Represents any possible alternative allele">',
    ],
)
def test_declared_gvcf_is_rejected_from_header(tmp_path, header_line):
    path = tmp_path / "declared.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        + header_line
        + "\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )

    with pytest.raises(VariantIngestError) as error:
        inspect_vcf_header(str(path))

    assert error.value.error_code == GVCF_UNSUPPORTED_ERROR_CODE


def test_undeclared_gvcf_record_fails_fast(tmp_path):
    path = tmp_path / "undeclared.vcf"
    path.write_text(VCF_HEADER + "chr1\t20\t.\tT\t<NON_REF>\t.\t.\tEND=30\n")

    with pytest.raises(VariantIngestError) as error:
        list(iter_input_occurrence_batches(str(path), "vcf", batch_rows=10))

    assert error.value.error_code == GVCF_UNSUPPORTED_ERROR_CODE


def test_unsorted_vcf_is_rejected(tmp_path):
    path = tmp_path / "unsorted.vcf"
    path.write_text(
        VCF_HEADER + "chr1\t24\t.\tT\tC\t.\t.\t.\n" + "chr1\t20\t.\tT\tC\t.\t.\t.\n"
    )

    with pytest.raises(VariantIngestError) as error:
        list(iter_input_occurrence_batches(str(path), "vcf", batch_rows=10))

    assert error.value.error_code == "UNSORTED_VCF"


def test_streaming_validation_writes_fixed_schema_shards_and_manifest(
    occurrence_vcf,
    genome,
    tmp_path,
):
    output = tmp_path / "out"
    result = stream_validate_variants(
        str(occurrence_vcf),
        str(genome),
        str(output / "valid.tsv"),
        str(output / "invalid.tsv"),
        str(output / "occurrences"),
        str(output / "canonical" / "valid"),
        str(output / "canonical" / "invalid"),
        str(output / "manifest.json"),
        str(output / "header.vcf"),
        width=20,
        batch_size=2,
        fmt="vcf",
    )

    assert result.record_count == 3
    assert result.alt_occurrence_count == 6
    assert result.valid_occurrence_count == 4
    assert result.invalid_occurrence_count == 0
    assert result.unsupported_occurrence_count == 2
    assert result.unique_canonical_variants == 2
    assert result.duplicate_canonical_occurrences == 2
    assert len(result.occurrence_shards) == 3
    assert all(shard.row_count <= 2 for shard in result.occurrence_shards)

    first_occurrence = pq.read_table(
        output / "occurrences" / result.occurrence_shards[0].uri
    )
    assert first_occurrence.schema.field("source_variant_id").type == "string"
    assert first_occurrence.schema.field("record_ordinal").type == "int64"
    assert all(
        (output / "occurrences" / shard.uri)
        .with_suffix(".parquet.success.json")
        .exists()
        for shard in result.occurrence_shards
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["canonicalizer_version"] == CANONICALIZER_VERSION
    assert manifest["reference_build"] == "GRCh38"
    assert manifest["input_size_bytes"] == occurrence_vcf.stat().st_size
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["header_sha256"]) == 64

    stale = output / "occurrences" / "part-99999999.parquet"
    stale.write_bytes(b"stale")
    original_mtimes = {
        shard.uri: (output / "occurrences" / shard.uri).stat().st_mtime_ns
        for shard in result.occurrence_shards
    }
    retried = stream_validate_variants(
        str(occurrence_vcf),
        str(genome),
        str(output / "valid.tsv"),
        str(output / "invalid.tsv"),
        str(output / "occurrences"),
        str(output / "canonical" / "valid"),
        str(output / "canonical" / "invalid"),
        str(output / "manifest.json"),
        str(output / "header.vcf"),
        width=20,
        batch_size=2,
        fmt="vcf",
    )
    assert not stale.exists()
    assert {
        shard.uri: (output / "occurrences" / shard.uri).stat().st_mtime_ns
        for shard in retried.occurrence_shards
    } == original_mtimes


def test_tsv_uses_same_occurrence_contract(tmp_path):
    path = tmp_path / "variants.tsv"
    path.write_text("1\t20\tT\tC\tuser-1\n1\t24\tT\t*\n")

    batches = list(iter_input_occurrence_batches(str(path), "tsv", batch_rows=1))
    rows = pd.concat([batch.rows for batch in batches], ignore_index=True)

    assert rows[["record_ordinal", "alt_index"]].values.tolist() == [[0, 1], [1, 1]]
    assert rows["source_variant_id"].tolist() == ["user-1", None]
    assert rows["error_code"].tolist() == [None, "SPANNING_DELETION"]
