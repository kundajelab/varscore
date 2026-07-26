"""Tests for the argv contract in ``varscore.lava.commands``.

These run in **every** varscore install, on every supported Python version, with
no orchestration framework and no credentials -- ``commands.py`` imports neither.
That is the whole point of them: the failure this contract exists to prevent is a
flag renamed on the varscore side while a caller elsewhere keeps emitting the old
one, and this file catches that in ordinary CI rather than in a GPU container
minutes into a job.

The round trip is what does the work: every builder's output is parsed by the
command's own ``build_parser()``. Renaming a flag in the CLI without updating the
builder fails here immediately.

Plan-level and orchestration-framework behaviour is covered separately in
``test_lava_plugin.py``, which needs ``lava-core``.
"""

from __future__ import annotations

import pytest

from varscore.lava import commands


FOLDS = [f"/f{i}" for i in range(commands.NUM_FOLDS)]

# One representative invocation per command, as the builder would be called.
BUILDERS = {
    commands.SCORE: lambda: commands.score_argv(
        model="/m.h5", peak_distribution="/pd.npy", genome="/g.fa", variants="/v.tsv", out="/o.tsv"
    ),
    commands.PREDICTIONS: lambda: commands.predictions_argv(
        fold_scores=FOLDS, peaks_dnatree="/peaks.dnatree", out="/o.tsv"
    ),
    commands.INGEST: lambda: commands.ingest_argv(
        genome="/g.fa", fold_models=FOLDS, peaks="/peaks.bed.gz", out_dir="/models/m"
    ),
    commands.INTERPRETATION: lambda: commands.interpretation_argv(
        model="/m.h5", genome="/g.fa", variants="/v.tsv", out_dir="/out"
    ),
    commands.AVERAGE_INTERPRETATIONS: lambda: commands.average_interpretations_argv(
        fold_dirs=FOLDS, out_dir="/out"
    ),
    commands.PREPARE_VARIANT_PLOTTING: lambda: commands.prepare_variant_plotting_argv(
        variants="/v.tsv", plotting_data_dir="/plots", out="/o.tsv"
    ),
}


class TestRoundTrip:
    """Every builder must emit argv the real parser accepts."""

    def test_a_builder_exists_for_every_command(self) -> None:
        """Neither side may grow an entry the other lacks."""
        assert set(BUILDERS) == set(commands.COMMANDS)

    @pytest.mark.parametrize("module", list(BUILDERS))
    def test_builder_output_parses(self, module: str) -> None:
        args = commands.validate_argv(BUILDERS[module]())
        assert args is not None

    def test_every_command_module_exposes_a_parser(self) -> None:
        """``build_parser()`` must exist and be importable without the ``[model]`` extra."""
        for module in commands.COMMANDS:
            assert commands.parser_for(module) is not None

    def test_score_paths_land_on_the_intended_flags(self) -> None:
        """Parsing is necessary but not sufficient -- two swapped paths still parse."""
        args = commands.validate_argv(
            commands.score_argv(
                model="/m.h5", peak_distribution="/pd.npy", genome="/g.fa", variants="/v.tsv", out="/o.tsv"
            )
        )
        assert args.model_loc == "/m.h5"
        assert args.peaks_dist_loc == "/pd.npy"
        assert args.genome_loc == "/g.fa"
        assert args.variants_loc == "/v.tsv"
        assert args.out_path == "/o.tsv"

    def test_ingest_paths_land_on_the_intended_flags(self) -> None:
        args = commands.validate_argv(
            commands.ingest_argv(genome="/g.fa", fold_models=FOLDS, peaks="/peaks.bed.gz", out_dir="/models/m")
        )
        assert args.genome_loc == "/g.fa"
        assert args.peaks_loc == "/peaks.bed.gz"
        assert args.output_dir == "/models/m"
        assert [getattr(args, f"fold_{i}_loc") for i in range(commands.NUM_FOLDS)] == FOLDS


class TestFoldCount:
    """``NUM_FOLDS`` is pinned by the CLI, not a free parameter."""

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
        them at all -- there is no flag to put a sixth fold on, and omitting one
        of the five fails as a missing required argument.
        """
        parser = commands.parser_for(module)
        fold_flags = {a.dest for a in parser._actions if a.dest.startswith("fold_") and a.dest.endswith(suffix)}
        assert fold_flags == {f"fold_{i}{suffix}" for i in range(commands.NUM_FOLDS)}
        assert all(a.required for a in parser._actions if a.dest in fold_flags)

    @pytest.mark.parametrize("count", [0, commands.NUM_FOLDS - 1, commands.NUM_FOLDS + 1])
    def test_builders_reject_a_wrong_sized_fold_list(self, count: int) -> None:
        """Rejected at build time rather than emitted for the command to refuse."""
        folds = [f"/f{i}" for i in range(count)]
        with pytest.raises(commands.ArgvContractError, match="takes exactly"):
            commands.predictions_argv(fold_scores=folds, peaks_dnatree="/p", out="/o")
        with pytest.raises(commands.ArgvContractError, match="takes exactly"):
            commands.ingest_argv(genome="/g", fold_models=folds, peaks="/p", out_dir="/o")
        with pytest.raises(commands.ArgvContractError, match="takes exactly"):
            commands.average_interpretations_argv(fold_dirs=folds, out_dir="/o")


class TestValidationErrors:
    def test_unknown_varscore_module_is_rejected(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="not a varscore command"):
            commands.validate_argv(["varscore.scoring.chrombpnet.nope", "-m", "x"])

    def test_bad_flag_is_rejected(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="rejected its arguments"):
            commands.validate_argv([commands.SCORE, "--not-a-real-flag", "x"])

    def test_missing_required_flag_is_rejected(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="rejected its arguments"):
            commands.validate_argv([commands.SCORE, "-m", "x"])


class TestCommandOwnership:
    """``validate_all`` must not mistake a varscore typo for another tool's command."""

    def test_another_tools_command_is_skipped(self) -> None:
        commands.validate_all([["finemo", "call-hits", "-r", "/r"]])

    def test_misspelled_varscore_module_is_caught_not_skipped(self) -> None:
        """A membership test against ``COMMANDS`` would silently pass this over.

        This is the regression that matters: a typo'd module is exactly the
        mistake the contract exists to catch, and skipping it as "somebody else's
        command" would make the check worse than useless.
        """
        with pytest.raises(commands.ArgvContractError, match="not a varscore command"):
            commands.validate_all([["varscore.scoring.chrombpnet.nope", "-m", "x"]])

    def test_a_varscore_module_outside_chrombpnet_is_still_claimed(self) -> None:
        with pytest.raises(commands.ArgvContractError, match="not a varscore command"):
            commands.validate_all([["varscore.preprocessing.validate", "-i", "/x"]])

    def test_ownership_is_by_prefix(self) -> None:
        assert commands.is_varscore_command([commands.SCORE])
        assert commands.is_varscore_command(["varscore.anything.at.all"])
        assert not commands.is_varscore_command(["finemo", "call-hits"])
        assert not commands.is_varscore_command([])

    def test_a_valid_mixed_plan_passes(self) -> None:
        commands.validate_all(
            [
                BUILDERS[commands.SCORE](),
                ["finemo", "call-hits", "-r", "/r"],
                BUILDERS[commands.PREDICTIONS](),
            ]
        )


def test_importing_the_package_does_not_require_lava_core() -> None:
    """``varscore.lava`` must stay importable in a base install.

    ``ChromBPNetPlugin`` is resolved through a module ``__getattr__`` for this
    reason; an eager import would take this whole file out of ordinary CI.
    """
    import varscore.lava

    assert varscore.lava.validate_argv is commands.validate_argv
