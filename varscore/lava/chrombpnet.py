"""varscore's ChromBPNet model plugin for a ``lava-core`` platform.

A platform built on ``lava-core`` runs a model architecture by asking its
``ModelPlugin`` two kinds of question: how to *run* the model (which container
commands, in what order, reading and writing which paths) and how to *read* the
scores it produced (which columns, and what makes a variant worth prioritizing).

Both answers are facts about varscore. The commands are varscore's own CLI; the
prioritization thresholds are varscore's science. Keeping the plugin here means
they are stated once, next to the code they describe, and the contract tests can
check the commands against the real parsers -- see ``varscore.lava.commands``.

This plugin builds plans, it does not execute them. The returned tasks name a
container image and an argv and carry no filesystem paths of their own: each path
comes from a base-relative layout combined with the execution backend's path
resolver, so one plan runs unchanged wherever the platform schedules it.

Installing ``lava-core`` alongside varscore (see ``requirements-lava.txt``)
registers this class automatically through the ``lava.model_plugins`` entry
point.
"""

from __future__ import annotations

import enum
import math
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from lava_core.plugins.execution import ContainerTaskSpec, GpuPool, ResourceRequest, Transfer
from lava_core.plugins.model.plugin import (
    ContainerScoringPlan,
    InterpretationPlan,
    InterpretationRequest,
    ModelPlugin,
    PreprocessingPlan,
    PreprocessingRequest,
    ScoreColumn,
    ScoringRequest,
)
from lava_core.plugins.model.preprocessing_paths import ext_for_peaks
from lava_core.plugins.model.scoring_paths import ext_for_fold
from lava_core.plugins.predicate import Abs, And, Col, Eq, Ge, Gt, Lit, Or, Predicate

from varscore.lava import commands


class ShardKind(enum.Enum):
    """The ``kind`` label each container task carries.

    The platform groups finished tasks by this string to decide when a fan-in
    stage may run, so a task's kind must equal the one the platform groups on or
    its fan-in never fires. These values are therefore a wire contract, not free
    labels: changing one silently strands the stage that waits on it.

    Kept as a plain ``Enum`` of strings rather than ``StrEnum`` so this module
    imports on every Python version varscore supports; ``.value`` is used at each
    use site.
    """

    MODEL_PREPROCESSING = "model_preprocessing"
    SCORING_FOLDS = "variant_scoring_folds"
    SCORING_SUMMARIZATION = "variant_scoring_summarization"
    INTERPRETATION_FOLDS = "interpretation_folds"
    INTERPRETATION_AVERAGE = "interpretation_average"
    INTERPRETATION_MOTIF = "interpretation_motif"
    INTERPRETATION_PLOT = "interpretation_plot"


#: The image every varscore command runs in. Its entrypoint is ``python -m``, so
#: a task's argv starts with a module path rather than an executable name.
VARSCORE_IMAGE = "kundajelab/varscore:dev"
#: Motif hit-calling runs finemo, which is a separate tool in its own image. It is
#: the one stage of the interpretation pipeline that is not a varscore command.
FINEMO_IMAGE = "kundajelab/finemo_gpu:latest"

#: finemo hit-calling parameters: the ``-b`` batch size and ``-l`` lambda flags.
_FINEMO_BATCH_SIZE = "100"
_FINEMO_LAMBDA = "0.8"

# Compute profile per shard. `gpu` names a pool the backend maps onto real
# hardware, so the plugin states the baseline need and leaves the choice of
# specific accelerator to the backend's capacity planning.
_SCORE_RESOURCES = ResourceRequest(gpu=GpuPool.NORMAL_GPU, count=1, memory_gb=64)
_PREPROCESS_RESOURCES = ResourceRequest(gpu=GpuPool.NORMAL_GPU, count=1, memory_gb=64)
_INTERPRET_RESOURCES = ResourceRequest(gpu=GpuPool.NORMAL_GPU, count=1, memory_gb=64)
# Summarization only averages TSVs, so it needs no GPU.
_SUMMARIZE_RESOURCES = ResourceRequest(memory_gb=8)


class ChromBPNetArtifact(BaseModel):
    """The registration facts this plugin reads out of a request's ``artifact``.

    A request carries ``artifact`` as an open mapping so each architecture can
    pass its own facts without changing the shared request signature. This model
    is ChromBPNet's schema for that mapping; validating through it up front means
    a missing or wrong-typed field raises an error naming the field, rather than a
    ``KeyError`` from somewhere deep in path construction. Unrelated keys are
    ignored, since the mapping is shared with other consumers.

    - ``model_accession_pattern``: names the per-fold model files' extensions.
      Required by every plan.
    - ``peaks_path``: the raw peaks file. Required for preprocessing only, and
      absent for scoring and interpretation.
    """

    model_config = ConfigDict(extra="ignore")

    model_accession_pattern: str
    peaks_path: str | None = None


def _num(value: Any) -> float | None:
    """Return ``float(value)``, or ``None`` when the value is missing or NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _flag(value: Any) -> bool | None:
    """Return ``bool(value)``, or ``None`` when the value is missing or NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return bool(value)


class ChromBPNetPlugin(ModelPlugin):
    """Runs and interprets ChromBPNet models on a ``lava-core`` platform."""

    model_type = "CHROMBPNET"

    @property
    def num_folds(self) -> int:
        """The number of folds one scoring run fans into.

        Fixed at five by varscore's CLI, which declares five required fold flags
        on the commands that consume folds -- see ``commands.NUM_FOLDS``.
        """
        return commands.NUM_FOLDS

    # --- preprocessing ------------------------------------------------------

    def build_preprocessing_plan(self, request: PreprocessingRequest) -> PreprocessingPlan:
        """Build the one-task plan that turns a model's raw files into derived artifacts.

        ``ingest`` reads the per-fold model files and the raw peaks and writes the
        peak distributions and the peak interval tree that scoring later needs.
        Both input extensions are derived from the registration fields by string
        manipulation alone, so the task needs no directory listing at runtime to
        find its own inputs.
        """
        r, layout = request.resolver, request.layout
        model_id, genome = request.model_id, request.genome_label
        artifact = ChromBPNetArtifact.model_validate(dict(request.artifact))
        if artifact.peaks_path is None:
            msg = "ChromBPNet preprocessing requires artifact['peaks_path']"
            raise ValueError(msg)
        pattern = artifact.model_accession_pattern

        command = commands.ingest_argv(
            genome=r.resolve(layout.genome_fasta(genome)),
            fold_models=[
                r.resolve(layout.fold_model_file(model_id, fold, ext_for_fold(pattern, fold)))
                for fold in range(self.num_folds)
            ],
            peaks=r.resolve(layout.peaks_file(model_id, ext_for_peaks(artifact.peaks_path))),
            out_dir=r.resolve(layout.model_dir(model_id)),
        )

        job = ContainerTaskSpec(
            kind=ShardKind.MODEL_PREPROCESSING.value,
            image=VARSCORE_IMAGE,
            command=command,
            labels={"model_id": model_id, "kind": ShardKind.MODEL_PREPROCESSING.value},
            resources=_PREPROCESS_RESOURCES,
        )
        return PreprocessingPlan(job=job)

    # --- scoring ------------------------------------------------------------

    def build_scoring_plan(self, request: ScoringRequest) -> ContainerScoringPlan:
        """Build the scoring plan: one ``score`` task per fold, then a ``predictions`` fan-in.

        Each fold task declares the files it reads and writes, so a backend that
        stages inputs per task can run it unchanged, while a backend that shares a
        volume across pods stages them itself and ignores the declarations.
        ``ready_when`` is the fold count, so the fan-in waits on exactly the tasks
        this plan produced rather than a hardcoded number.
        """
        r, layout = request.resolver, request.layout
        job_id, model_id, genome = request.job_id, request.model_id, request.genome_label
        pattern = ChromBPNetArtifact.model_validate(dict(request.artifact)).model_accession_pattern

        genome_logical = layout.genome_fasta(genome)
        variants_logical = layout.model_variants_file(job_id, model_id)

        shards: list[ContainerTaskSpec] = []
        for fold in range(self.num_folds):
            model_logical = layout.fold_model_file(model_id, fold, ext_for_fold(pattern, fold))
            peaks_dist_logical = layout.peak_distribution_file(model_id, fold)
            output_logical = layout.fold_score_file(job_id, model_id, fold)
            command = commands.score_argv(
                model=r.resolve(model_logical),
                peak_distribution=r.resolve(peaks_dist_logical),
                genome=r.resolve(genome_logical),
                variants=r.resolve(variants_logical),
                out=r.resolve(output_logical),
            )
            shards.append(
                ContainerTaskSpec(
                    kind=ShardKind.SCORING_FOLDS.value,
                    image=VARSCORE_IMAGE,
                    command=command,
                    # `num_folds` rides along on the task so the fan-in reads this
                    # run's fold width from the task rather than assuming a value.
                    labels={
                        "job_id": job_id,
                        "model_id": model_id,
                        "fold": str(fold),
                        "num_folds": str(self.num_folds),
                        "kind": ShardKind.SCORING_FOLDS.value,
                    },
                    resources=_SCORE_RESOURCES,
                    inputs=[
                        Transfer(uri=model_logical, logical_path=model_logical, locality_key=model_id),
                        Transfer(uri=peaks_dist_logical, logical_path=peaks_dist_logical, locality_key=model_id),
                        Transfer(uri=genome_logical, logical_path=genome_logical),
                        Transfer(uri=variants_logical, logical_path=variants_logical, locality_key=model_id),
                    ],
                    outputs=[Transfer(uri=output_logical, logical_path=output_logical, locality_key=model_id)],
                )
            )

        dna_tree_logical = layout.dna_tree_file(model_id)
        summarize_out_logical = layout.scoring_result_file(job_id, model_id)
        fold_score_logicals = [layout.fold_score_file(job_id, model_id, f) for f in range(self.num_folds)]
        predictions_command = commands.predictions_argv(
            fold_scores=[r.resolve(logical) for logical in fold_score_logicals],
            peaks_dnatree=r.resolve(dna_tree_logical),
            out=r.resolve(summarize_out_logical),
        )
        summarize = ContainerTaskSpec(
            kind=ShardKind.SCORING_SUMMARIZATION.value,
            image=VARSCORE_IMAGE,
            command=predictions_command,
            labels={"job_id": job_id, "model_id": model_id, "kind": ShardKind.SCORING_SUMMARIZATION.value},
            resources=_SUMMARIZE_RESOURCES,
            inputs=[
                *(Transfer(uri=lg, logical_path=lg, locality_key=model_id) for lg in fold_score_logicals),
                Transfer(uri=dna_tree_logical, logical_path=dna_tree_logical, locality_key=model_id),
            ],
            outputs=[Transfer(uri=summarize_out_logical, logical_path=summarize_out_logical, locality_key=model_id)],
        )

        return ContainerScoringPlan(shards=shards, summarize=summarize, ready_when=self.num_folds)

    # --- interpretation -----------------------------------------------------

    def build_interpretation_plan(self, request: InterpretationRequest) -> InterpretationPlan:
        """Build the interpretation plan: per-fold contributions, fold averaging, motif hit-calling, plots.

        The motif stage runs finemo rather than varscore, which is why it names a
        different image; it is one task per allele, so hits are called separately
        for the reference and alternate sequences. ``ready_when`` is the fold
        count, holding the averaging stage until every fold has finished.
        """
        r, layout = request.resolver, request.layout
        run_id, model_id, genome = request.run_id, request.model_id, request.genome_label
        pattern = ChromBPNetArtifact.model_validate(dict(request.artifact)).model_accession_pattern

        genome_arg = r.resolve(layout.genome_fasta(genome))
        variants_arg = r.resolve(layout.model_variants_file(run_id, model_id))

        folds: list[ContainerTaskSpec] = []
        for fold in range(self.num_folds):
            command = commands.interpretation_argv(
                model=r.resolve(layout.fold_model_file(model_id, fold, ext_for_fold(pattern, fold))),
                genome=genome_arg,
                variants=variants_arg,
                out_dir=r.resolve(layout.fold_results_dir(run_id, model_id, fold)),
            )
            folds.append(
                ContainerTaskSpec(
                    kind=ShardKind.INTERPRETATION_FOLDS.value,
                    image=VARSCORE_IMAGE,
                    command=command,
                    labels={
                        "run_id": run_id,
                        "model_id": model_id,
                        "fold": str(fold),
                        "num_folds": str(self.num_folds),
                        "kind": ShardKind.INTERPRETATION_FOLDS.value,
                    },
                    resources=_INTERPRET_RESOURCES,
                )
            )

        average_command = commands.average_interpretations_argv(
            fold_dirs=[r.resolve(layout.fold_results_dir(run_id, model_id, f)) for f in range(self.num_folds)],
            out_dir=r.resolve(layout.model_dir(run_id, model_id)),
        )
        average = ContainerTaskSpec(
            kind=ShardKind.INTERPRETATION_AVERAGE.value,
            image=VARSCORE_IMAGE,
            command=average_command,
            labels={"run_id": run_id, "model_id": model_id, "kind": ShardKind.INTERPRETATION_AVERAGE.value},
            resources=_INTERPRET_RESOURCES,
        )

        motif: dict[str, ContainerTaskSpec] = {}
        for allele in ("ref", "alt"):
            motif_command = [
                "finemo",
                "call-hits",
                "-r",
                r.resolve(layout.hitcaller_region_file(run_id, model_id, allele)),
                "-m",
                r.resolve(layout.motif_file()),
                "-o",
                r.resolve(layout.hits_dir(run_id, model_id, allele)),
                "-b",
                _FINEMO_BATCH_SIZE,
                "-l",
                _FINEMO_LAMBDA,
            ]
            motif[allele] = ContainerTaskSpec(
                kind=ShardKind.INTERPRETATION_MOTIF.value,
                image=FINEMO_IMAGE,
                command=motif_command,
                labels={
                    "run_id": run_id,
                    "model_id": model_id,
                    "allele": allele,
                    "kind": ShardKind.INTERPRETATION_MOTIF.value,
                },
                resources=_INTERPRET_RESOURCES,
            )

        plot_command = commands.prepare_variant_plotting_argv(
            variants=variants_arg,
            plotting_data_dir=r.resolve(layout.model_dir(run_id, model_id)),
            out=r.resolve(layout.variants_with_plots_file(run_id, model_id)),
        )
        plot = ContainerTaskSpec(
            kind=ShardKind.INTERPRETATION_PLOT.value,
            image=VARSCORE_IMAGE,
            command=plot_command,
            labels={"run_id": run_id, "model_id": model_id, "kind": ShardKind.INTERPRETATION_PLOT.value},
            resources=_INTERPRET_RESOURCES,
        )

        return InterpretationPlan(folds=folds, average=average, motif=motif, plot=plot, ready_when=self.num_folds)

    # --- results ------------------------------------------------------------

    def score_columns(self) -> list[ScoreColumn]:
        """The per-variant columns a ChromBPNet scoring run produces."""
        return [
            ScoreColumn(name="logfc", dtype="float", label="LogFC"),
            ScoreColumn(name="jsd", dtype="float", label="JSD"),
            ScoreColumn(name="active_allele_quantile", dtype="float", label="Active Allele Quantile"),
            ScoreColumn(name="in_peak", dtype="bool", label="In Peak"),
        ]

    def prioritize_predicate(self) -> Predicate:
        """The rule that flags a variant as worth attention.

        A variant qualifies when its predicted effect is both large and in a
        region the model is confident about: the absolute log fold-change clears
        0.25, the active allele sits above the 5th percentile of the model's peak
        distribution, and the variant is somewhere the effect is credible -- in a
        promoter, inside a called peak, or outside a peak but *gaining*
        accessibility (which a peak call would not yet capture).

        The predicate is returned as data, not evaluated here, so the platform can
        render it to SQL and filter server-side or apply it row-wise in Python and
        get the same answer either way.

        The promoter test reads the ``in_promoter`` membership flag, matching
        ``varscore.prioritization`` -- the in-process implementation of this same
        rule. It must not read ``region_type``: that column is a single
        severity-collapsed label, so a variant that is both exonic and in a
        promoter reports ``exonic`` and its promoter membership disappears. A
        variant carries a *set* of region labels and membership is the only sound
        way to test one, which is why the two implementations of this rule are
        written against the same flag.
        """
        return And(
            Ge(Abs(Col("logfc")), Lit(0.25)),
            Ge(Col("active_allele_quantile"), Lit(0.05)),
            Or(
                Eq(Col("in_promoter"), Lit(True)),
                Eq(Col("in_peak"), Lit(True)),
                And(Eq(Col("in_peak"), Lit(False)), Gt(Col("logfc"), Lit(0))),
            ),
        )

    def to_model_score(self, *, model_id: str, model_name: str, score: Mapping[str, Any]) -> dict[str, Any]:
        """Build one per-model entry from a scored variant's raw values."""
        return {
            "model_id": model_id,
            "model_name": model_name,
            "logfc": _num(score.get("logfc")),
            "jsd": _num(score.get("jsd")),
            "active_allele_quantile": _num(score.get("active_allele_quantile")),
            "in_peak": _flag(score.get("in_peak")),
            "prioritized": _flag(score.get("prioritized")) or False,
        }
