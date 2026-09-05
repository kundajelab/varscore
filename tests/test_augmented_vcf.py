import json
from pathlib import Path

import pandas as pd
import pysam

from varscore.export.augmented_vcf import write_augmented_vcf


def _write_fixture(path: Path) -> None:
    path.write_text(
        """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##INFO=<ID=KEEP,Number=1,Type=String,Description="existing">
##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t10\trs1\tA\tC,G\t42\tPASS\tKEEP=yes\tGT\t1|2
chr1\t20\t.\tT\t*\t.\t.\t.\tGT\t0/1
"""
    )


def test_writer_preserves_records_and_aligns_per_alt_results(tmp_path: Path) -> None:
    source = tmp_path / "source.vcf"
    _write_fixture(source)
    buckets = tmp_path / "buckets" / "bucket=000000"
    buckets.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "record_ordinal": 0,
                "alt_index": 2,
                "status": "PENDING",
                "error_code": None,
                "result_variant_id": "chr1:10:A:G",
                "prioritized": False,
                "most_active_celltype": "model beta",
                "region_type": "intronic",
                "nearest_genes": '[{"gene_name":"GENE2"}]',
                "model_beta_logfc": -0.7,
            },
            {
                "record_ordinal": 0,
                "alt_index": 1,
                "status": "PENDING",
                "error_code": None,
                "result_variant_id": "chr1:10:A:C",
                "prioritized": True,
                "most_active_celltype": "model alpha",
                "region_type": "coding",
                "nearest_genes": '[{"gene_name":"GENE1"}]',
                "model_alpha_logfc": 1.25,
            },
            {
                "record_ordinal": 1,
                "alt_index": 1,
                "status": "UNSUPPORTED",
                "error_code": "SPANNING_DELETION",
                "result_variant_id": None,
            },
        ]
    ).to_parquet(buckets / "part-00000000.parquet", index=False)
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "uuid-a",
                        "name": "model alpha",
                        "column_prefix": "model_alpha",
                    },
                    {
                        "model_id": "uuid-b",
                        "name": "model beta",
                        "column_prefix": "model_beta",
                    },
                ]
            }
        )
    )
    ingest = tmp_path / "ingest.json"
    ingest.write_text(
        json.dumps(
            {"input_sha256": "a" * 64, "record_count": 2, "alt_occurrence_count": 3}
        )
    )
    output = tmp_path / "augmented.vcf.gz"
    output_manifest = tmp_path / "manifest.json"

    result = write_augmented_vcf(
        str(source),
        str(buckets.parent),
        str(models),
        str(ingest),
        str(output),
        str(output_manifest),
    )

    assert result["record_count"] == 2
    assert output.exists()
    assert Path(f"{output}.csi").exists()
    with pysam.VariantFile(str(output)) as augmented:
        records = list(augmented)
        assert records[0].id == "rs1"
        assert records[0].qual == 42
        assert records[0].info["KEEP"] == "yes"
        assert records[0].samples["SAMPLE"]["GT"] == (1, 2)
        assert records[0].samples["SAMPLE"].phased
        assert records[0].info["LAVA1_STATUS"] == ("SCORED", "SCORED")
        assert records[0].info["LAVA1_PRIORITIZED"] == (1, 0)
        assert records[0].info["LAVA1_TOP_MODEL"] == ("M0001", "M0002")
        assert records[0].info["LAVA1_NEAREST_GENE"] == ("GENE1", "GENE2")
        assert records[1].info["LAVA1_STATUS"] == ("UNSUPPORTED",)
        assert records[1].info["LAVA1_ERROR"] == ("SPANNING_DELETION",)

    with pysam.VariantFile(str(output)) as indexed:
        assert [record.id for record in indexed.fetch("chr1", 9, 11)] == ["rs1"]

    second_output = tmp_path / "augmented-again.vcf.gz"
    write_augmented_vcf(
        str(output),
        str(buckets.parent),
        str(models),
        str(ingest),
        str(second_output),
        str(tmp_path / "second-manifest.json"),
    )
    with pysam.VariantFile(str(second_output)) as augmented_again:
        model_headers = [
            record
            for record in augmented_again.header.records
            if record.key == "LAVA_MODEL"
        ]
        assert len(model_headers) == 2
