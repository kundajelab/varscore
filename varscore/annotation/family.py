"""Bounded-memory family evidence export for a multi-sample VCF.

The Phenopacket is the source of truth for pedigree relationships and for the
individual-to-VCF-sample mapping.  The export deliberately reports
``PARENTS_REFERENCE`` rather than ``DE_NOVO``: genotype evidence alone is not a
validated de-novo call.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import pysam

from varscore.core.logging import get_logger


logger = get_logger(__name__)

MIN_CONFIDENT_REFERENCE_DP = 10
MIN_CONFIDENT_REFERENCE_GQ = 20

FAMILY_COLUMNS = [
    "canonical_variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "cohort_ac",
    "cohort_an",
    "cohort_af",
    "proband_id",
    "proband_sample",
    "proband_gt",
    "proband_ad",
    "proband_dp",
    "proband_gq",
    "mother_id",
    "mother_sample",
    "mother_gt",
    "mother_ad",
    "mother_dp",
    "mother_gq",
    "father_id",
    "father_sample",
    "father_gt",
    "father_ad",
    "father_dp",
    "father_gq",
    "inheritance",
    "affected_carriers",
    "unaffected_carriers",
    "shared_siblings",
    "quality_flags",
    "member_genotypes_json",
]


class FamilyConfigurationError(ValueError):
    """A user-actionable Phenopacket/VCF family binding error."""


@dataclass(frozen=True)
class FamilyMember:
    individual_id: str
    sample_id: str
    affected_status: str
    relationship: str


@dataclass(frozen=True)
class FamilyContext:
    family_id: str
    proband: FamilyMember
    members: Tuple[FamilyMember, ...]
    mother: Optional[FamilyMember]
    father: Optional[FamilyMember]


def load_family_context(phenopacket_path: str, vcf_path: str) -> FamilyContext:
    """Load a GA4GH Family document and resolve its samples for ``vcf_path``."""
    with Path(phenopacket_path).open() as handle:
        document = json.load(handle)

    try:
        family_id = str(document["id"])
        proband_id = str(document["proband"]["subject"]["id"])
    except (KeyError, TypeError) as exc:
        raise FamilyConfigurationError(
            "Family Phenopacket must include id and proband.subject.id."
        ) from exc

    files = document.get("files") or []
    if not files:
        raise FamilyConfigurationError(
            "Family Phenopacket must include a file with individualToFileIdentifiers."
        )
    input_name = Path(vcf_path).name
    matching = [item for item in files if Path(str(item.get("uri", ""))).name == input_name]
    if len(matching) == 1:
        file_entry = matching[0]
    elif len(files) == 1:
        file_entry = files[0]
    else:
        raise FamilyConfigurationError(
            f"Could not select one Phenopacket file entry for {input_name!r}."
        )
    sample_by_individual = {
        str(key): str(value)
        for key, value in (file_entry.get("individualToFileIdentifiers") or {}).items()
    }
    if proband_id not in sample_by_individual:
        raise FamilyConfigurationError(
            f"No VCF sample mapping is defined for proband {proband_id!r}."
        )

    persons = {
        str(person["individualId"]): person
        for person in (document.get("pedigree") or {}).get("persons", [])
        if person.get("individualId") is not None
    }
    proband_person = persons.get(proband_id, {})
    mother_id = _parent_id(proband_person.get("maternalId"))
    father_id = _parent_id(proband_person.get("paternalId"))

    ordered_ids = [proband_id] + [key for key in sample_by_individual if key != proband_id]
    members = []
    for individual_id in ordered_ids:
        person = persons.get(individual_id, {})
        if individual_id == proband_id:
            relationship = "PROBAND"
        elif individual_id == mother_id:
            relationship = "MOTHER"
        elif individual_id == father_id:
            relationship = "FATHER"
        elif _shares_parents(person, mother_id, father_id):
            relationship = "SIBLING"
        else:
            relationship = "RELATIVE"
        members.append(
            FamilyMember(
                individual_id=individual_id,
                sample_id=sample_by_individual[individual_id],
                affected_status=str(person.get("affectedStatus", "MISSING")),
                relationship=relationship,
            )
        )

    by_id = {member.individual_id: member for member in members}
    context = FamilyContext(
        family_id=family_id,
        proband=by_id[proband_id],
        members=tuple(members),
        mother=by_id.get(mother_id),
        father=by_id.get(father_id),
    )
    _validate_vcf_samples(context, vcf_path)
    return context


def export_family_evidence(
    vcf_path: str,
    context: FamilyContext,
    output_path: str,
) -> int:
    """Write one gzip-compressed TSV row per ALT carried by the proband."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    row_count = 0
    with pysam.VariantFile(vcf_path) as source, gzip.open(
        temporary, "wt", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=FAMILY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for record in source:
            for alt_index, alt in enumerate(record.alts or (), start=1):
                proband_call = record.samples[context.proband.sample_id]
                if not carries_alt(proband_call, alt_index):
                    continue
                writer.writerow(_family_row(record, str(alt), alt_index, context))
                row_count += 1
    temporary.replace(destination)
    logger.info(
        "Exported %d proband-carried family ALT occurrences to %s.",
        row_count,
        output_path,
    )
    return row_count


def carries_alt(call: Mapping[str, Any], alt_index: int) -> bool:
    """Return whether a sample GT contains this 1-based ALT index."""
    genotype = call.get("GT")
    return genotype is not None and alt_index in genotype


def confident_reference(call: Optional[Mapping[str, Any]]) -> bool:
    if call is None:
        return False
    genotype = call.get("GT")
    return bool(
        genotype
        and all(allele == 0 for allele in genotype)
        and _as_int(call.get("DP")) >= MIN_CONFIDENT_REFERENCE_DP
        and _as_int(call.get("GQ")) >= MIN_CONFIDENT_REFERENCE_GQ
    )


def classify_inheritance(
    mother_call: Optional[Mapping[str, Any]],
    father_call: Optional[Mapping[str, Any]],
    alt_index: int,
) -> str:
    """Classify direct parent evidence without asserting de-novo status."""
    mother_carries = mother_call is not None and carries_alt(mother_call, alt_index)
    father_carries = father_call is not None and carries_alt(father_call, alt_index)
    if mother_carries and father_carries:
        return "BIPARENTAL"
    if mother_carries:
        return "MATERNAL"
    if father_carries:
        return "PATERNAL"
    if confident_reference(mother_call) and confident_reference(father_call):
        return "PARENTS_REFERENCE"
    return "UNRESOLVED"


def _family_row(
    record: pysam.VariantRecord,
    alt: str,
    alt_index: int,
    context: FamilyContext,
) -> Dict[str, Any]:
    calls = {
        member.individual_id: record.samples[member.sample_id]
        for member in context.members
    }
    proband_call = calls[context.proband.individual_id]
    mother_call = calls.get(context.mother.individual_id) if context.mother else None
    father_call = calls.get(context.father.individual_id) if context.father else None
    member_evidence = [
        _member_evidence(member, calls[member.individual_id], alt_index)
        for member in context.members
    ]
    carriers = [item for item in member_evidence if item["carries_alt"]]
    siblings = [
        item["individual_id"]
        for item in carriers
        if item["relationship"] == "SIBLING"
    ]
    quality_flags = []
    for item in member_evidence:
        if not item["carries_alt"]:
            continue
        if item["dp"] is None or item["dp"] < MIN_CONFIDENT_REFERENCE_DP:
            quality_flags.append(f"{item['individual_id']}:LOW_DP")
        if item["gq"] is None or item["gq"] < MIN_CONFIDENT_REFERENCE_GQ:
            quality_flags.append(f"{item['individual_id']}:LOW_GQ")

    row: Dict[str, Any] = {
        "canonical_variant_id": f"{record.contig}:{record.pos}:{record.ref}:{alt}",
        "chrom": record.contig,
        "pos": record.pos,
        "ref": record.ref,
        "alt": alt,
        "cohort_ac": _info_alt_value(record.info.get("AC"), alt_index),
        "cohort_an": _scalar(record.info.get("AN")),
        "cohort_af": _info_alt_value(record.info.get("AF"), alt_index),
        "proband_id": context.proband.individual_id,
        "proband_sample": context.proband.sample_id,
        "inheritance": classify_inheritance(mother_call, father_call, alt_index),
        "affected_carriers": ",".join(
            item["individual_id"] for item in carriers if item["affected_status"] == "AFFECTED"
        ),
        "unaffected_carriers": ",".join(
            item["individual_id"] for item in carriers if item["affected_status"] == "UNAFFECTED"
        ),
        "shared_siblings": ",".join(siblings),
        "quality_flags": ",".join(quality_flags),
        "member_genotypes_json": json.dumps(member_evidence, separators=(",", ":")),
    }
    _add_call_columns(row, "proband", context.proband, proband_call)
    _add_call_columns(row, "mother", context.mother, mother_call)
    _add_call_columns(row, "father", context.father, father_call)
    return row


def _member_evidence(
    member: FamilyMember, call: Mapping[str, Any], alt_index: int
) -> Dict[str, Any]:
    return {
        "individual_id": member.individual_id,
        "sample_id": member.sample_id,
        "relationship": member.relationship,
        "affected_status": member.affected_status,
        "gt": _format_gt(call),
        "ad": _format_value(call.get("AD")),
        "dp": _nullable_int(call.get("DP")),
        "gq": _nullable_int(call.get("GQ")),
        "carries_alt": carries_alt(call, alt_index),
        "confident_reference": confident_reference(call),
    }


def _add_call_columns(
    row: Dict[str, Any],
    prefix: str,
    member: Optional[FamilyMember],
    call: Optional[Mapping[str, Any]],
) -> None:
    row[f"{prefix}_id"] = member.individual_id if member else ""
    row[f"{prefix}_sample"] = member.sample_id if member else ""
    row[f"{prefix}_gt"] = _format_gt(call) if call is not None else ""
    row[f"{prefix}_ad"] = _format_value(call.get("AD")) if call is not None else ""
    row[f"{prefix}_dp"] = _nullable_int(call.get("DP")) if call is not None else ""
    row[f"{prefix}_gq"] = _nullable_int(call.get("GQ")) if call is not None else ""


def _validate_vcf_samples(context: FamilyContext, vcf_path: str) -> None:
    with pysam.VariantFile(vcf_path) as source:
        samples = set(source.header.samples)
    missing = sorted(member.sample_id for member in context.members if member.sample_id not in samples)
    if missing:
        raise FamilyConfigurationError(
            "Phenopacket sample identifiers are absent from the VCF header: "
            + ", ".join(missing)
        )


def _shares_parents(person: Mapping[str, Any], mother_id: Optional[str], father_id: Optional[str]) -> bool:
    return bool(
        mother_id
        and father_id
        and _parent_id(person.get("maternalId")) == mother_id
        and _parent_id(person.get("paternalId")) == father_id
    )


def _parent_id(value: Any) -> Optional[str]:
    if value in (None, "", "0", 0):
        return None
    return str(value)


def _format_gt(call: Mapping[str, Any]) -> str:
    genotype = call.get("GT")
    if genotype is None:
        return "./."
    separator = "|" if getattr(call, "phased", False) else "/"
    return separator.join("." if allele is None else str(allele) for allele in genotype)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ",".join("." if item is None else str(item) for item in value)
    return str(value)


def _nullable_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _as_int(value: Any) -> int:
    return -1 if value is None else int(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return value[0] if value else ""
    return "" if value is None else value


def _info_alt_value(value: Any, alt_index: int) -> Any:
    if isinstance(value, (tuple, list)):
        return value[alt_index - 1] if len(value) >= alt_index else ""
    return "" if value is None else value
