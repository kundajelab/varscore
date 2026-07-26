"""Tests for the orchestration plugin in ``varscore.lava``.

The point of these tests is the argv contract. An orchestration platform runs
varscore by constructing ``python -m <module> <flags>`` and handing it to a
container; nothing type-checks that argv, so a renamed flag used to surface as an
argparse usage error inside a GPU container, minutes into a job. Here the plugin
that builds the argv and the parsers that consume it are in the same repository,
so the two can be checked against each other directly -- ``TestArgvContract`` runs
every command the plugin emits through the real parser.

Everything here needs ``lava-core`` (varscore's ``[lava]`` extra, Python 3.12
only), so the whole module skips when it is not installed. A base install must
still collect cleanly, which is why the imports are guarded rather than
top-level.
"""

from __future__ import annotations

import pytest


pytest.importorskip("lava_core", reason="requires varscore's [lava] extra (Python 3.12)")

from lava_core.plugins.execution import PathResolver  # noqa: E402
from lava_core.plugins.model import (  # noqa: E402
    InterpretationLayout,
    InterpretationRequest,
    PreprocessingLayout,
    PreprocessingRequest,
    ScoringLayout,
    ScoringRequest,
)

from varscore.lava import commands  # noqa: E402
from varscore.lava.chrombpnet import VARSCORE_IMAGE, ChromBPNetPlugin, ShardKind  # noqa: E402


JOB_ID = "job-1"
RUN_ID = "run-1"
MODEL_ID = "model-1"
GENOME = "hg38"
PATTERN = "model.fold_{fold}.h5"
MOUNT = "/mnt/volume"


@pytest.fixture
def plugin() -> ChromBPNetPlugin:
    return ChromBPNetPlugin()


@pytest.fixture
def resolver() -> PathResolver:
    return PathResolver(MOUNT)


def _scoring_request(resolver: PathResolver, **artifact: object) -> ScoringRequest:
    return ScoringRequest(
        job_id=JOB_ID,
        model_id=MODEL_ID,
        genome_label=GENOME,
        artifact={"model_accession_pattern": PATTERN, **artifact},
        resolver=resolver,
        layout=ScoringLayout(),
    )


def _preprocessing_request(resolver: PathResolver) -> PreprocessingRequest:
    return PreprocessingRequest(
        model_id=MODEL_ID,
        genome_label=GENOME,
        artifact={"model_accession_pattern": PATTERN, "peaks_path": "s3://x/peaks.bed.gz"},
        resolver=resolver,
        layout=PreprocessingLayout(),
    )


def _interpretation_request(resolver: PathResolver) -> InterpretationRequest:
    return InterpretationRequest(
        run_id=RUN_ID,
        model_id=MODEL_ID,
        genome_label=GENOME,
        artifact={"model_accession_pattern": PATTERN},
        resolver=resolver,
        layout=InterpretationLayout(),
    )


def _all_task_argvs(plugin: ChromBPNetPlugin, resolver: PathResolver) -> list[list[str]]:
    """Every container argv the plugin can emit, across all three plan types."""
    scoring = plugin.build_scoring_plan(_scoring_request(resolver))
    interpretation = plugin.build_interpretation_plan(_interpretation_request(resolver))
    preprocessing = plugin.build_preprocessing_plan(_preprocessing_request(resolver))
    tasks = [
        *scoring.shards,
        scoring.summarize,
        preprocessing.job,
        *interpretation.folds,
        interpretation.average,
        *interpretation.motif.values(),
        interpretation.plot,
    ]
    return [list(task.command) for task in tasks if task is not None]


class TestArgvContract:
    """Every command the plugin emits must be one varscore actually accepts."""

    def test_every_varscore_argv_parses(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        argvs = [a for a in _all_task_argvs(plugin, resolver) if commands.is_varscore_command(a)]
        # Guard against the assertion passing vacuously if the filter over-matches.
        assert len(argvs) == 1 + commands.NUM_FOLDS * 2 + 3, "expected folds x2 + summarize + ingest + average + plot"
        for argv in argvs:
            commands.validate_argv(argv)

    def test_plan_covers_every_documented_command(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        """``COMMANDS`` and the plans must not drift apart in either direction."""
        emitted = {a[0] for a in _all_task_argvs(plugin, resolver) if commands.is_varscore_command(a)}
        assert emitted == set(commands.COMMANDS)

    def test_motif_task_is_not_a_varscore_command(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        """Motif hit-calling runs finemo in its own image, so it is out of scope here."""
        plan = plugin.build_interpretation_plan(_interpretation_request(resolver))
        for task in plan.motif.values():
            assert not commands.is_varscore_command(list(task.command))
            assert task.image != VARSCORE_IMAGE

    def test_score_flags_carry_the_intended_paths(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        """Parsing is necessary but not sufficient -- two swapped paths still parse."""
        layout = ScoringLayout()
        shard = plugin.build_scoring_plan(_scoring_request(resolver)).shards[3]
        args = commands.validate_argv(list(shard.command))
        assert args.model_loc == f"{MOUNT}/{layout.fold_model_file(MODEL_ID, 3, '.h5')}"
        assert args.peaks_dist_loc == f"{MOUNT}/{layout.peak_distribution_file(MODEL_ID, 3)}"
        assert args.genome_loc == f"{MOUNT}/{layout.genome_fasta(GENOME)}"
        assert args.variants_loc == f"{MOUNT}/{layout.model_variants_file(JOB_ID, MODEL_ID)}"
        assert args.out_path == f"{MOUNT}/{layout.fold_score_file(JOB_ID, MODEL_ID, 3)}"

    def test_unknown_command_is_rejected(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="not a varscore command"):
            commands.validate_argv(["varscore.scoring.chrombpnet.nope", "-m", "x"])

    def test_bad_flag_is_rejected(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="rejected its arguments"):
            commands.validate_argv([commands.SCORE, "--not-a-real-flag", "x"])

    def test_missing_required_flag_is_rejected(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="rejected its arguments"):
            commands.validate_argv([commands.SCORE, "-m", "x"])


class TestFoldCount:
    """``num_folds`` is pinned by the CLI, not a free parameter."""

    def test_plugin_agrees_with_the_cli_constant(self, plugin: ChromBPNetPlugin) -> None:
        assert plugin.num_folds == commands.NUM_FOLDS

    @pytest.mark.parametrize(
        ("module", "suffix"),
        [
            (commands.INGEST, "_loc"),
            (commands.PREDICTIONS, "_scores_loc"),
            (commands.AVERAGE_INTERPRETATIONS, "_dir"),
        ],
    )
    def test_fold_consuming_commands_require_exactly_num_folds(self, module: str, suffix: str) -> None:
        """These commands declare one required flag per fold, so the count is structural.

        A model scored with a different number of folds could not be passed to
        them at all -- there is no flag to put the sixth fold on, and omitting one
        of the five fails as a missing required argument.
        """
        parser = commands.parser_for(module)
        fold_flags = {a.dest for a in parser._actions if a.dest.startswith("fold_") and a.dest.endswith(suffix)}
        assert fold_flags == {f"fold_{i}{suffix}" for i in range(commands.NUM_FOLDS)}
        assert all(a.required for a in parser._actions if a.dest in fold_flags)

    def test_plan_shard_count_matches(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        plan = plugin.build_scoring_plan(_scoring_request(resolver))
        assert len(plan.shards) == commands.NUM_FOLDS
        assert plan.ready_when == commands.NUM_FOLDS


class TestDropInParity:
    """varscore's plugin must behave identically to the one currently shipped in ``lava-core``.

    Both are registered for ``CHROMBPNET`` while the plugin is moving from one to
    the other, and which of the two an environment resolves depends on entry-point
    discovery order. These tests keep that ambiguity harmless by requiring the two
    to be indistinguishable, and they are what makes it safe to delete the
    ``lava-core`` copy. Delete this class along with it.
    """

    @pytest.fixture
    def reference(self):
        from lava_core.plugins.model.chrombpnet import ChromBPNetPlugin as ReferencePlugin

        return ReferencePlugin()

    def test_same_model_type_and_fold_count(self, plugin: ChromBPNetPlugin, reference) -> None:
        assert plugin.model_type == reference.model_type
        assert plugin.num_folds == reference.num_folds

    def test_same_score_columns(self, plugin: ChromBPNetPlugin, reference) -> None:
        assert plugin.score_columns() == reference.score_columns()

    def test_same_prioritization_rule(self, plugin: ChromBPNetPlugin, reference) -> None:
        assert plugin.prioritize_predicate().to_sql() == reference.prioritize_predicate().to_sql()

    def test_same_model_score_entry(self, plugin: ChromBPNetPlugin, reference) -> None:
        score = {"logfc": 0.5, "jsd": 0.1, "active_allele_quantile": 0.9, "in_peak": True, "prioritized": True}
        kwargs = {"model_id": MODEL_ID, "model_name": "m", "score": score}
        assert plugin.to_model_score(**kwargs) == reference.to_model_score(**kwargs)

    def test_same_shard_kinds(self) -> None:
        from lava_core.plugins.model import chrombpnet as reference_module

        assert {k.value for k in ShardKind} == {k.value for k in reference_module.ShardKind}

    def test_same_images(self, plugin: ChromBPNetPlugin, reference) -> None:
        from lava_core.plugins.model import chrombpnet as reference_module

        assert VARSCORE_IMAGE == reference_module.VARSCORE_IMAGE

    @pytest.mark.parametrize("plan_kind", ["scoring", "preprocessing", "interpretation"])
    def test_same_emitted_argv(
        self, plugin: ChromBPNetPlugin, reference, resolver: PathResolver, plan_kind: str
    ) -> None:
        """Byte-for-byte argv parity, which is what a drop-in replacement means here."""
        if plan_kind == "scoring":
            mine = plugin.build_scoring_plan(_scoring_request(resolver))
            theirs = reference.build_scoring_plan(_scoring_request(resolver))
            assert [s.command for s in mine.shards] == [s.command for s in theirs.shards]
            assert mine.summarize.command == theirs.summarize.command
        elif plan_kind == "preprocessing":
            assert (
                plugin.build_preprocessing_plan(_preprocessing_request(resolver)).job.command
                == reference.build_preprocessing_plan(_preprocessing_request(resolver)).job.command
            )
        else:
            mine = plugin.build_interpretation_plan(_interpretation_request(resolver))
            theirs = reference.build_interpretation_plan(_interpretation_request(resolver))
            assert [s.command for s in mine.folds] == [s.command for s in theirs.folds]
            assert mine.average.command == theirs.average.command
            assert mine.plot.command == theirs.plot.command
            assert {k: v.command for k, v in mine.motif.items()} == {k: v.command for k, v in theirs.motif.items()}


class TestKnownDivergence:
    """Pins a real disagreement between the two implementations of the same rule.

    varscore states the ChromBPNet prioritization rule twice: once as a pandas
    expression in ``varscore.prioritization``, for scoring in-process, and once as
    a predicate here, for a platform to push into SQL. They should agree. They do
    not: the pandas version tests the ``in_promoter`` membership flag, while the
    predicate tests ``region_type == 'promoter'``.

    ``region_type`` is a severity-collapsed single label, so a variant that is both
    exonic and in a promoter reports ``region_type == 'exonic'`` and fails the
    predicate's promoter test, while passing the pandas one. varscore's own
    conventions call for testing membership for exactly this reason.

    The predicate is deliberately left matching ``lava-core`` for now, so that
    moving the plugin changes nothing observable (see ``TestDropInParity``).
    Fixing it is a separate, behaviour-changing decision. This test fails the
    moment someone changes one side, which is the point: it makes the divergence
    impossible to lose track of, and it should be deleted -- not updated -- once
    both sides agree.
    """

    def test_predicate_still_reads_region_type(self, plugin: ChromBPNetPlugin) -> None:
        sql = plugin.prioritize_predicate().to_sql()
        assert "region_type" in sql
        assert "in_promoter" not in sql

    def test_pandas_rule_still_reads_in_promoter(self) -> None:
        import inspect

        from varscore import prioritization

        source = inspect.getsource(prioritization.prioritize_variants)
        assert "in_promoter" in source
        assert "region_type" not in source
