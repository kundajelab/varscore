import gzip
import json

import pysam

from varscore.annotation.family import (
    classify_inheritance,
    export_family_evidence,
    load_family_context,
)


def _write_family(tmp_path):
    packet = tmp_path / "phenopacket.json"
    packet.write_text(
        json.dumps(
            {
                "id": "KING165",
                "proband": {"id": "176", "subject": {"id": "176"}},
                "pedigree": {
                    "persons": [
                        {
                            "individualId": "176",
                            "maternalId": "382",
                            "paternalId": "474",
                            "affectedStatus": "AFFECTED",
                        },
                        {"individualId": "382", "affectedStatus": "UNAFFECTED"},
                        {
                            "individualId": "384",
                            "maternalId": "382",
                            "paternalId": "474",
                            "affectedStatus": "UNAFFECTED",
                        },
                        {"individualId": "474", "affectedStatus": "UNAFFECTED"},
                    ]
                },
                "files": [
                    {
                        "uri": "family.vcf",
                        "individualToFileIdentifiers": {
                            "176": "176",
                            "382": "382",
                            "384": "384",
                            "474": "474",
                        },
                    }
                ],
            }
        )
    )
    vcf = tmp_path / "family.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=200>\n"
        "##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count\">\n"
        "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele number\">\n"
        "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths\">\n"
        "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
        "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t176\t382\t384\t474\n"
        "chr1\t20\t.\tT\tC\t50\tPASS\tAC=2;AN=8;AF=.25\tGT:AD:DP:GQ\t0/1:10,10:20:99\t0/1:9,8:17:80\t0/0:18,0:18:60\t0/0:20,0:20:70\n"
        "chr1\t24\t.\tT\tG\t50\tPASS\tAC=1;AN=8;AF=.125\tGT:AD:DP:GQ\t0/0:20,0:20:60\t0/0:20,0:20:60\t0/1:8,7:15:50\t0/0:20,0:20:60\n"
    )
    return packet, vcf


def test_context_and_export_only_proband_carried_variants(tmp_path):
    packet, vcf = _write_family(tmp_path)
    context = load_family_context(str(packet), str(vcf))
    assert context.proband.sample_id == "176"
    assert context.mother.individual_id == "382"
    assert context.father.individual_id == "474"
    assert [member.individual_id for member in context.members if member.relationship == "SIBLING"] == ["384"]

    output = tmp_path / "family.tsv.gz"
    assert export_family_evidence(str(vcf), context, str(output)) == 1
    with gzip.open(output, "rt") as handle:
        header = handle.readline().rstrip().split("\t")
        row = dict(zip(header, handle.readline().rstrip().split("\t")))
        assert not handle.readline()
    assert row["canonical_variant_id"] == "chr1:20:T:C"
    assert row["inheritance"] == "MATERNAL"
    assert row["unaffected_carriers"] == "382"
    assert row["shared_siblings"] == ""


def test_parent_reference_requires_confident_calls(tmp_path):
    _packet, vcf = _write_family(tmp_path)
    with pysam.VariantFile(str(vcf)) as source:
        records = list(source)
        reference = records[1].samples["382"]
        low_depth_reference = {"GT": (0, 0), "DP": 1, "GQ": 60}

        assert classify_inheritance(reference, reference, 1) == "PARENTS_REFERENCE"
        assert classify_inheritance(reference, low_depth_reference, 1) == "UNRESOLVED"
