"""Tests for the orchestration plugin in ``varscore.lava.chrombpnet``.

Covers what needs the framework installed: whole plans, the prioritization
predicate, and parity with the copy still shipped in ``lava-core``. The argv
contract itself is framework-free and tested in ``test_lava_commands.py``, which
runs in every install -- keep new argv checks there, not here, so they stay
enforced by ordinary CI.

This module needs ``lava-core`` (``requirements-lava.txt``, Python 3.12 only) and
skips without it, so the imports are guarded rather than top-level.
"""

from __future__ import annotations

import pytest


pytest.importorskip("lava_core", reason="needs lava-core; see requirements-lava.txt (Python 3.12)")

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


def _scoring_request(resolver: PathResolver) -> ScoringRequest:
    return ScoringRequest(
        job_id=JOB_ID,
        model_id=MODEL_ID,
        genome_label=GENOME,
        artifact={"model_accession_pattern": PATTERN},
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


def _all_tasks(plugin: ChromBPNetPlugin, resolver: PathResolver) -> list:
    """Every container task the plugin can emit, across all three plan types."""
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
    return [t for t in tasks if t is not None]


class TestPlanArgv:
    """Plan-level checks. Per-command flag detail lives in ``test_lava_commands.py``."""

    def test_every_varscore_argv_in_every_plan_parses(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        commands.validate_all(list(t.command) for t in _all_tasks(plugin, resolver))

    def test_plans_cover_every_documented_command(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        """``COMMANDS`` and the plans must not drift apart in either direction."""
        emitted = {t.command[0] for t in _all_tasks(plugin, resolver) if commands.is_varscore_command(t.command)}
        assert emitted == set(commands.COMMANDS)

    def test_motif_task_belongs_to_another_tool(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        """Motif hit-calling runs finemo in its own image, so it is out of scope here."""
        plan = plugin.build_interpretation_plan(_interpretation_request(resolver))
        for task in plan.motif.values():
            assert not commands.is_varscore_command(list(task.command))
            assert task.image != VARSCORE_IMAGE

    def test_fold_shard_count_matches_the_cli_constant(
        self, plugin: ChromBPNetPlugin, resolver: PathResolver
    ) -> None:
        plan = plugin.build_scoring_plan(_scoring_request(resolver))
        assert plugin.num_folds == commands.NUM_FOLDS
        assert len(plan.shards) == commands.NUM_FOLDS
        assert plan.ready_when == commands.NUM_FOLDS


class TestPrioritizationRule:
    """The promoter test must read membership, never the collapsed label.

    ``region_type`` is a single severity-collapsed label, so a variant that is
    both exonic and in a promoter reports ``exonic``. Testing
    ``region_type == 'promoter'`` therefore drops its promoter membership and
    fails to prioritize it -- a bug this repo has hit before, and the reason
    ``varscore.prioritization`` tests the ``in_promoter`` flag. Both statements of
    the rule must agree.
    """

    def _evaluate(self, plugin: ChromBPNetPlugin, **row: object) -> bool:
        return plugin.prioritize_predicate().to_python()(row)

    def test_exonic_variant_in_a_promoter_is_prioritized(self, plugin: ChromBPNetPlugin) -> None:
        """The regression. Under the old ``region_type`` form this returned False."""
        assert self._evaluate(
            plugin,
            logfc=-0.5,
            active_allele_quantile=0.9,
            in_peak=False,
            in_promoter=True,
            region_type="exonic",
        )

    def test_promoter_route_is_not_reachable_via_region_type(self, plugin: ChromBPNetPlugin) -> None:
        """A promoter ``region_type`` with the flag unset must not prioritize.

        Guards against reintroducing the collapsed-label test as an extra
        alternative, which would make the rule pass for the wrong reason.
        """
        assert not self._evaluate(
            plugin,
            logfc=-0.5,
            active_allele_quantile=0.9,
            in_peak=False,
            in_promoter=False,
            region_type="promoter",
        )

    def test_predicate_sql_reads_in_promoter(self, plugin: ChromBPNetPlugin) -> None:
        sql = plugin.prioritize_predicate().to_sql()
        assert "in_promoter" in sql
        assert "region_type" not in sql

    def test_agrees_with_the_in_process_rule(self, plugin: ChromBPNetPlugin) -> None:
        """Cross-check against ``varscore.prioritization`` on the disputed case.

        The two implementations of this rule are written for different engines --
        a pandas expression and a pushed-down predicate -- so this compares
        outcomes on the row that used to distinguish them, rather than comparing
        source text.
        """
        import pandas as pd

        from varscore import prioritization

        db = pd.DataFrame(
            [
                {
                    "variant_id": "v1",
                    "chr": "chr1",
                    "pos": 100,
                    "ref": "A",
                    "alt": "T",
                    "region_type": "exonic",
                    "in_promoter": True,
                    "model_id": MODEL_ID,
                    "logfc": -0.5,
                    "active_allele_quantile": 0.9,
                    "in_peak": False,
                }
            ]
        )
        pandas_result = bool(prioritization.prioritize_variants(db)["prioritized"].iloc[0])
        predicate_result = self._evaluate(
            plugin,
            logfc=-0.5,
            active_allele_quantile=0.9,
            in_peak=False,
            in_promoter=True,
            region_type="exonic",
        )
        assert pandas_result == predicate_result is True

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            # Effect too small.
            ({"logfc": 0.1, "active_allele_quantile": 0.9, "in_peak": True, "in_promoter": False}, False),
            # Allele not active enough.
            ({"logfc": 1.0, "active_allele_quantile": 0.01, "in_peak": True, "in_promoter": False}, False),
            # In a called peak.
            ({"logfc": 1.0, "active_allele_quantile": 0.9, "in_peak": True, "in_promoter": False}, True),
            # Outside a peak but gaining accessibility.
            ({"logfc": 1.0, "active_allele_quantile": 0.9, "in_peak": False, "in_promoter": False}, True),
            # Outside a peak and losing accessibility, not a promoter.
            ({"logfc": -1.0, "active_allele_quantile": 0.9, "in_peak": False, "in_promoter": False}, False),
        ],
    )
    def test_thresholds(self, plugin: ChromBPNetPlugin, row: dict, expected: bool) -> None:
        assert self._evaluate(plugin, region_type="intergenic", **row) is expected


class TestDropInParity:
    """varscore's plugin must be indistinguishable from the copy in ``lava-core``.

    Both are registered for ``CHROMBPNET`` while the plugin moves from one to the
    other, and which an environment resolves depends on entry-point discovery
    order. These tests keep that ambiguity harmless, and they are what makes it
    safe to delete the ``lava-core`` copy. Delete this class along with it.

    Comparison is over whole plan objects, not just argv: labels, resources, and
    declared transfers are observable orchestration behaviour too -- a shard that
    requests the wrong GPU pool or omits an input transfer is a different shard,
    however identical its command line.
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

    def test_same_images(self) -> None:
        from lava_core.plugins.model import chrombpnet as reference_module

        assert VARSCORE_IMAGE == reference_module.VARSCORE_IMAGE

    def test_whole_scoring_plan_is_identical(
        self, plugin: ChromBPNetPlugin, reference, resolver: PathResolver
    ) -> None:
        mine = plugin.build_scoring_plan(_scoring_request(resolver))
        theirs = reference.build_scoring_plan(_scoring_request(resolver))
        assert mine == theirs

    def test_whole_preprocessing_plan_is_identical(
        self, plugin: ChromBPNetPlugin, reference, resolver: PathResolver
    ) -> None:
        mine = plugin.build_preprocessing_plan(_preprocessing_request(resolver))
        theirs = reference.build_preprocessing_plan(_preprocessing_request(resolver))
        assert mine == theirs

    def test_whole_interpretation_plan_is_identical(
        self, plugin: ChromBPNetPlugin, reference, resolver: PathResolver
    ) -> None:
        mine = plugin.build_interpretation_plan(_interpretation_request(resolver))
        theirs = reference.build_interpretation_plan(_interpretation_request(resolver))
        assert mine == theirs

    def test_plan_equality_is_meaningful(self, plugin: ChromBPNetPlugin, resolver: PathResolver) -> None:
        """Guard the three tests above against passing because ``==`` is identity-ish.

        Frozen dataclasses holding lists compare structurally, but if any plan
        member fell back to object identity the comparisons would be vacuously
        true for two separately-built plans. Two independent builds of the *same*
        plan must be equal, and a plan built from a different request must not be.
        """
        a = plugin.build_scoring_plan(_scoring_request(resolver))
        b = plugin.build_scoring_plan(_scoring_request(resolver))
        assert a == b

        other = plugin.build_scoring_plan(
            ScoringRequest(
                job_id="job-2",
                model_id=MODEL_ID,
                genome_label=GENOME,
                artifact={"model_accession_pattern": PATTERN},
                resolver=resolver,
                layout=ScoringLayout(),
            )
        )
        assert a != other

    def test_task_resources_and_labels_match(
        self, plugin: ChromBPNetPlugin, reference, resolver: PathResolver
    ) -> None:
        """Explicit per-field check, so a failure names what diverged."""
        mine = _all_tasks(plugin, resolver)
        theirs = _all_tasks(reference, resolver)
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert a.kind == b.kind
            assert a.image == b.image
            assert a.labels == b.labels
            assert a.resources == b.resources
            assert a.inputs == b.inputs
            assert a.outputs == b.outputs
