"""Trio inheritance annotation.

A *separate* pipeline from the per-variant annotation chain in
:mod:`varscore.annotation.annotate`: it needs cross-sample context (a child plus
its mother and father), which the single-variant chain can't provide, so it has
its own entrypoint.

Two modes, auto-selected from the inputs:

- **Genotype/zygosity-aware** (a single jointly-genotyped multi-sample VCF, with
  ``--child-sample/--mother-sample/--father-sample``): reads ``GT`` directly and
  applies real Mendelian logic. This is the only input that supports *true*
  de-novo calling, because a hom-ref parent has a confident reference call at the
  child's site even with no ALT record.

- **Presence/absence** (three separate child/mother/father files, TSV or VCF):
  classifies a child variant by canonical set membership in the parent call sets.
  ``De_Novo`` here means "not observed in either parent file", which over-calls
  de novo when a parent simply has a no-call at that site — documented in
  ``docs/inheritance.md``.

Both modes canonicalize variants (chromosome naming, indel left-alignment against
the genome, and -- by default -- MNP atomization) so equivalent representations
compare equal. See ``docs/inheritance.md``.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyfaidx

import varscore.core.io as io_utils
from varscore.core.logging import get_logger, log_timing, setup_logging

if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)

VARIANT_SCHEMA = io_utils.VARIANT_SCHEMA

# De-novo confidence thresholds for a parent's reference call (genotype mode).
DEFAULT_MIN_GQ = 20
DEFAULT_MIN_DP = 10

_CARRIER = {"het", "hom_alt"}


########################
# CANONICAL NORMALIZE  #
########################


def _norm_chr(chro: str) -> str:
    """Add a ``chr`` prefix if absent (matches ``validate_variant``)."""
    chro = str(chro)
    return chro if chro.startswith("chr") else "chr" + chro


def _trim(pos: int, ref: str, alt: str) -> Tuple[int, str, str]:
    """Genome-free parsimony: trim shared suffix then shared prefix (keeping >=1 base)."""
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


def left_align(
    chro: str, pos: int, ref: str, alt: str, genome: pyfaidx.Fasta
) -> Tuple[int, str, str]:
    """Left-align and trim an indel against the reference FASTA (vt/bcftools algorithm).

    Shifts an indel to its leftmost equivalent position so the same variant in a
    repeat region matches regardless of how it was originally written. ``pos`` is
    1-based. SNVs/MNPs are returned unchanged (they cannot shift). Falls back to
    :func:`_trim` if the chromosome is missing from ``genome``.
    """
    if chro not in genome.keys():
        logger.warning("chromosome %s not in genome; left-align falls back to trim", chro)
        return _trim(pos, ref, alt)
    contig = genome[chro]
    while True:
        if len(ref) > 0 and len(alt) > 0 and ref[-1] == alt[-1]:
            ref, alt = ref[:-1], alt[:-1]
        elif len(ref) == 0 or len(alt) == 0:
            pos -= 1
            base = str(contig[pos - 1]).upper()  # 1-based pos -> 0-based slice
            ref, alt = base + ref, base + alt
        else:
            break
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


def normalized_atoms(
    chro: str,
    pos: int,
    ref: str,
    alt: str,
    genome: Optional[pyfaidx.Fasta] = None,
    atomize: bool = True,
) -> List[Tuple[str, int, str, str]]:
    """Canonicalize a variant to a list of atomic ``(chr, pos, ref, alt)`` keys.

    Normalization: ``chr`` prefix, then indel left-alignment (if a ``genome`` is
    given) or genome-free trimming. When ``atomize`` is true an MNP is split into
    one SNV per differing position (phase-free canonical form, like
    ``bcftools norm --atomize``); SNVs and indels are single atoms either way.
    """
    chro = _norm_chr(chro)
    pos = int(pos)
    ref, alt = str(ref).upper(), str(alt).upper()

    if genome is not None and len(ref) != len(alt):
        pos, ref, alt = left_align(chro, pos, ref, alt, genome)
    else:
        pos, ref, alt = _trim(pos, ref, alt)

    if atomize and len(ref) == len(alt) and len(ref) > 1:
        return [
            (chro, pos + i, r, a)
            for i, (r, a) in enumerate(zip(ref, alt))
            if r != a
        ]
    return [(chro, pos, ref, alt)]


#######################
# PRESENCE / ABSENCE  #
#######################


def build_parent_lookup(
    df: pd.DataFrame,
    genome: Optional[pyfaidx.Fasta] = None,
    atomize: bool = True,
) -> Set[Tuple[str, int, str, str]]:
    """Build a set of canonical atom keys from a parent variant DataFrame."""
    lookup: Set[Tuple[str, int, str, str]] = set()
    if df is None or df.empty:
        return lookup
    for row in df.itertuples(index=False):
        lookup.update(
            normalized_atoms(row.chr, row.pos, row.ref, row.alt, genome, atomize)
        )
    return lookup


def _membership(
    n_atoms: int, n_in: int
) -> bool:
    """A variant is inherited from a parent iff *all* of its atoms are present."""
    return n_atoms > 0 and n_in == n_atoms


def classify_presence(
    child: pd.DataFrame,
    mother: Optional[pd.DataFrame],
    father: Optional[pd.DataFrame],
    genome: Optional[pyfaidx.Fasta] = None,
    atomize: bool = True,
) -> pd.DataFrame:
    """Classify each child variant by canonical set membership in the parents.

    Returns ``child`` (original columns / representation preserved) with
    ``inheritance``, ``inheritance_mode`` and ``inheritance_flag`` appended.
    """
    mother_set = build_parent_lookup(mother, genome, atomize)
    father_set = build_parent_lookup(father, genome, atomize)

    labels: List[str] = []
    flags: List[str] = []
    for row in child.itertuples(index=False):
        atoms = normalized_atoms(row.chr, row.pos, row.ref, row.alt, genome, atomize)
        n = len(atoms)
        m_in = sum(1 for a in atoms if a in mother_set)
        f_in = sum(1 for a in atoms if a in father_set)

        in_mother = _membership(n, m_in)
        in_father = _membership(n, f_in)

        if in_mother and in_father:
            labels.append("Both")
        elif in_mother:
            labels.append("M")
        elif in_father:
            labels.append("F")
        else:
            labels.append("De_Novo")

        # Suspicious partial-MNP overlap: some atoms inherited but not all.
        partial = (0 < m_in < n) or (0 < f_in < n)
        flags.append("partial_mnp_overlap" if (n > 1 and partial and not (in_mother or in_father)) else "")

    out = child.copy()
    out["inheritance"] = labels
    out["inheritance_mode"] = "presence"
    out["inheritance_flag"] = flags
    return out


#######################
# GENOTYPE / ZYGOSITY #
#######################


def classify_genotyped(
    df: pd.DataFrame,
    child: str,
    mother: str,
    father: str,
    min_gq: int = DEFAULT_MIN_GQ,
    min_dp: int = DEFAULT_MIN_DP,
) -> pd.DataFrame:
    """Classify trio inheritance from a genotyped multi-sample VCF DataFrame.

    ``df`` is the output of :func:`io_utils.load_variants_vcf_genotyped` and must
    carry ``ZYG__/DP__/GQ__`` columns for the three named samples. Only rows where
    the child carries the ALT are returned, with ``inheritance``,
    ``inheritance_mode``, ``child_zygosity`` and ``inheritance_flag`` appended.
    """
    cz, mz, fz = df[f"ZYG__{child}"], df[f"ZYG__{mother}"], df[f"ZYG__{father}"]
    carrier = cz.isin(list(_CARRIER))
    sub = df[carrier].copy()
    if sub.empty:
        out = sub[VARIANT_SCHEMA].copy()
        for c in ("inheritance", "inheritance_mode", "child_zygosity", "inheritance_flag"):
            out[c] = []
        return out

    cz, mz, fz = sub[f"ZYG__{child}"], sub[f"ZYG__{mother}"], sub[f"ZYG__{father}"]
    m_has, f_has = mz.isin(list(_CARRIER)), fz.isin(list(_CARRIER))

    def conf_ref(zyg, dp_col, gq_col):
        dp = pd.to_numeric(sub[dp_col], errors="coerce")
        gq = pd.to_numeric(sub[gq_col], errors="coerce")
        # NaN (missing DP/GQ) compares False, so an unscored ref call is not "confident".
        return (zyg == "hom_ref") & (dp >= min_dp) & (gq >= min_gq)

    m_ref = conf_ref(mz, f"DP__{mother}", f"GQ__{mother}")
    f_ref = conf_ref(fz, f"DP__{father}", f"GQ__{father}")

    inheritance = np.select(
        [m_has & f_has, m_has & ~f_has, f_has & ~m_has, m_ref & f_ref],
        ["Both", "M", "F", "De_Novo"],
        default="Uncertain",
    )

    # A hom-alt child must inherit an ALT from BOTH parents; otherwise Mendelian error.
    violation = (cz == "hom_alt") & ~(m_has & f_has)

    out = sub[VARIANT_SCHEMA].copy()
    out["inheritance"] = inheritance
    out["inheritance_mode"] = "genotype"
    out["child_zygosity"] = cz.values
    out["inheritance_flag"] = np.where(violation, "mendelian_violation", "")
    return out


#################
# ORCHESTRATION #
#################


def annotate_inheritance(
    child_loc: str,
    out_path: str,
    mother_loc: Optional[str] = None,
    father_loc: Optional[str] = None,
    genome_loc: Optional[str] = None,
    fmt: str = "auto",
    atomize: bool = True,
    child_sample: Optional[str] = None,
    mother_sample: Optional[str] = None,
    father_sample: Optional[str] = None,
    min_gq: int = DEFAULT_MIN_GQ,
    min_dp: int = DEFAULT_MIN_DP,
) -> None:
    """Annotate child variant inheritance and write a TSV.

    Genotype mode is used when all three ``*_sample`` names are given (``child_loc``
    is then the multi-sample VCF); otherwise presence/absence mode runs over the
    three separate ``child_loc``/``mother_loc``/``father_loc`` files.
    """
    start = datetime.now()
    genotype_mode = bool(child_sample and mother_sample and father_sample)

    if genotype_mode:
        logger.info("Genotype/zygosity-aware mode (multi-sample VCF: %s)", child_loc)
        df = io_utils.load_variants_vcf_genotyped(
            child_loc, samples=[child_sample, mother_sample, father_sample]
        )
        result = classify_genotyped(
            df, child_sample, mother_sample, father_sample, min_gq, min_dp
        )
    else:
        logger.info("Presence/absence mode (child=%s mother=%s father=%s)",
                    child_loc, mother_loc, father_loc)
        genome = None
        if genome_loc:
            genome = pyfaidx.Fasta(genome_loc)
        else:
            logger.warning(
                "No --genome given: indels are trimmed but not left-aligned, which "
                "can miss repeat-shifted indels (occasional false De_Novo)."
            )
        child = io_utils.read_variants(child_loc, fmt)
        mother = io_utils.read_variants(mother_loc, fmt) if mother_loc else None
        father = io_utils.read_variants(father_loc, fmt) if father_loc else None
        result = classify_presence(child, mother, father, genome, atomize)

    counts = result["inheritance"].value_counts().to_dict()
    logger.info("Classified %d child variants: %s", len(result), counts)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    logger.info("Results saved to %s", out_path)
    log_timing(logger, "Inheritance annotation", start)


########
# MAIN #
########


def main():
    args = _parse_args()
    annotate_inheritance(
        child_loc=args.child,
        out_path=args.out_path,
        mother_loc=args.mother,
        father_loc=args.father,
        genome_loc=args.genome,
        fmt=args.format,
        atomize=not args.no_atomize,
        child_sample=args.child_sample,
        mother_sample=args.mother_sample,
        father_sample=args.father_sample,
        min_gq=args.min_gq,
        min_dp=args.min_dp,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify trio inheritance of child variants (M / F / Both / De_Novo, "
            "plus Uncertain in genotype mode)."
        )
    )
    parser.add_argument(
        "-v", "--child", required=True,
        help="Child variants file (TSV/VCF), or the multi-sample VCF in genotype mode.",
    )
    parser.add_argument("-o", "--out_path", required=True, help="Output TSV path.")
    parser.add_argument("-m", "--mother", help="Mother variants file (presence/absence mode).")
    parser.add_argument("-p", "--father", help="Father variants file (presence/absence mode).")
    parser.add_argument(
        "-g", "--genome",
        help="Reference FASTA for indel left-alignment (presence/absence mode).",
    )
    parser.add_argument(
        "-f", "--format", default="auto", choices=["auto", "tsv", "vcf"],
        help="Input format for the separate child/parent files.",
    )
    parser.add_argument(
        "--no-atomize", action="store_true",
        help="Disable MNP atomization; match whole normalized variants instead.",
    )
    parser.add_argument("--child-sample", help="Child sample name (enables genotype mode).")
    parser.add_argument("--mother-sample", help="Mother sample name (enables genotype mode).")
    parser.add_argument("--father-sample", help="Father sample name (enables genotype mode).")
    parser.add_argument(
        "--min-gq", type=int, default=DEFAULT_MIN_GQ,
        help="Min parent GQ for a confident reference call (genotype mode de novo).",
    )
    parser.add_argument(
        "--min-dp", type=int, default=DEFAULT_MIN_DP,
        help="Min parent DP for a confident reference call (genotype mode de novo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()


"""
# presence/absence (separate files)
python -m varscore.annotation.inheritance -v child.tsv -m mother.tsv -p father.tsv \
    -g genome.fa -o inheritance.tsv

# genotype/zygosity-aware (one jointly-genotyped VCF)
python -m varscore.annotation.inheritance -v trio.vcf.gz -o inheritance.tsv \
    --child-sample CHILD --mother-sample MOM --father-sample DAD
"""
