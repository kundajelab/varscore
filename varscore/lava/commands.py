"""The command line varscore exposes to an orchestration layer, in one place.

A platform that runs varscore in a container never imports it. It builds an argv
list and runs it -- the scoring image's entrypoint is ``python -m``, so a task is
``python -m varscore.scoring.chrombpnet.score -m ... -p ... -g ...``. That argv
*is* the interface between varscore and its caller.

Historically that interface was written down only on the caller's side, as string
literals, with nothing checking the two halves agreed. Renaming a flag here broke
the caller silently and the failure surfaced inside a GPU container, minutes into
a job, as an argparse usage error.

This module owns that interface. Each command has a builder (``score_argv``,
``ingest_argv``, ...) that names the flags, so no caller writes them out, and
``validate_argv`` checks a built list against the command's real parser.

Nothing here imports the orchestration framework. That is deliberate: it means
the whole contract can be exercised by varscore's own test suite, on every Python
version it supports, with no credentials and no extra install. A renamed flag
fails those tests immediately. ``varscore.lava.chrombpnet`` builds on this module
and *is* framework-specific; the split keeps the checkable part checkable.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
from typing import Iterable, Sequence


# --- command module paths ---------------------------------------------------
# Each is a module runnable with `python -m`, which is what the scoring image's
# entrypoint prepends.

SCORE = "varscore.scoring.chrombpnet.score"
PREDICTIONS = "varscore.scoring.chrombpnet.predictions"
INGEST = "varscore.scoring.chrombpnet.ingest"
INTERPRETATION = "varscore.scoring.chrombpnet.interpret.interpretation"
AVERAGE_INTERPRETATIONS = "varscore.scoring.chrombpnet.interpret.average_interpretations"
PREPARE_VARIANT_PLOTTING = "varscore.scoring.chrombpnet.interpret.prepare_variant_plotting"

#: Every varscore command an orchestrator invokes. Each module exposes
#: ``build_parser()`` and is importable without the ``[model]`` extra, so the
#: contract can be checked in a base install with no GPU and no TensorFlow.
COMMANDS: tuple[str, ...] = (
    SCORE,
    PREDICTIONS,
    INGEST,
    INTERPRETATION,
    AVERAGE_INTERPRETATIONS,
    PREPARE_VARIANT_PLOTTING,
)

#: Module prefix identifying a command as varscore's. Used to tell "a command
#: this module is responsible for" apart from "a command belonging to some other
#: tool", which matters because a plan legitimately mixes both.
MODULE_PREFIX = "varscore."

#: The number of cross-validation folds a ChromBPNet model is scored with.
#:
#: This is a hard structural constraint, not a default. ``ingest``,
#: ``predictions``, and ``average_interpretations`` each declare exactly five
#: ``required=True`` fold flags (``-f0`` through ``-f4``), so a model scored with
#: any other fold count cannot be passed to them: there is no flag to put a sixth
#: fold on, and omitting one of the five fails as a missing required argument.
#: The builders below reject a wrong-sized fold list rather than emitting argv
#: that the command would refuse.
NUM_FOLDS = 5


class ArgvContractError(ValueError):
    """An argv list does not match the varscore command it names.

    Raised for a module that is not a known command, and for argv that the
    command's own parser rejects (an unrecognized flag, a missing required one, a
    bad value). The message carries argparse's own diagnostic.
    """


# --- argv builders ----------------------------------------------------------
# One per command. These are the only place flag names are written down, so a
# caller cannot drift from the parser independently -- and the round-trip test
# (build here, parse with the real parser) catches a rename with no orchestration
# framework installed.


def _fold_flags(paths: Sequence[str], *, command: str) -> list[str]:
    """Expand per-fold paths into ``-f0 <path> -f1 <path> ...``.

    Raises ``ArgvContractError`` unless exactly ``NUM_FOLDS`` paths are given,
    because the receiving commands declare exactly that many required fold flags.
    """
    if len(paths) != NUM_FOLDS:
        msg = f"{command} takes exactly {NUM_FOLDS} folds (-f0..-f{NUM_FOLDS - 1}), got {len(paths)}"
        raise ArgvContractError(msg)
    argv: list[str] = []
    for fold, path in enumerate(paths):
        argv += [f"-f{fold}", path]
    return argv


def score_argv(*, model: str, peak_distribution: str, genome: str, variants: str, out: str) -> list[str]:
    """Score variants through one fold of a model."""
    return [SCORE, "-m", model, "-p", peak_distribution, "-g", genome, "-v", variants, "-o", out]


def predictions_argv(*, fold_scores: Sequence[str], peaks_dnatree: str, out: str) -> list[str]:
    """Average per-fold scores into model-level scores and flag peak overlap."""
    return [PREDICTIONS, *_fold_flags(fold_scores, command=PREDICTIONS), "-p", peaks_dnatree, "-o", out]


def ingest_argv(*, genome: str, fold_models: Sequence[str], peaks: str, out_dir: str) -> list[str]:
    """Derive a model's peak distributions and peak interval tree from its raw files."""
    return [INGEST, "-g", genome, *_fold_flags(fold_models, command=INGEST), "-p", peaks, "-o", out_dir]


def interpretation_argv(*, model: str, genome: str, variants: str, out_dir: str) -> list[str]:
    """Compute contribution scores for one fold."""
    return [INTERPRETATION, "-m", model, "-g", genome, "-v", variants, "-o", out_dir]


def average_interpretations_argv(*, fold_dirs: Sequence[str], out_dir: str) -> list[str]:
    """Average per-fold contribution scores into one model-level result."""
    return [
        AVERAGE_INTERPRETATIONS,
        *_fold_flags(fold_dirs, command=AVERAGE_INTERPRETATIONS),
        "-o",
        out_dir,
    ]


def prepare_variant_plotting_argv(*, variants: str, plotting_data_dir: str, out: str) -> list[str]:
    """Render per-variant interpretation plots."""
    return [PREPARE_VARIANT_PLOTTING, "-v", variants, "-p", plotting_data_dir, "-o", out]


# --- validation -------------------------------------------------------------


def parser_for(module_path: str) -> argparse.ArgumentParser:
    """Return the real ``argparse`` parser for one varscore command.

    Imports the command module and calls its ``build_parser()``. Raises
    ``ArgvContractError`` if ``module_path`` is not a known command, so a typo in
    a caller's module string fails here rather than as a container
    ``ModuleNotFoundError``.
    """
    if module_path not in COMMANDS:
        known = "\n  ".join(COMMANDS)
        msg = f"{module_path!r} is not a varscore command. Known commands:\n  {known}"
        raise ArgvContractError(msg)
    module = importlib.import_module(module_path)
    return module.build_parser()  # type: ignore[no-any-return]


def is_varscore_command(argv: Sequence[str]) -> bool:
    """Whether ``argv`` claims to invoke varscore.

    Matches on the module prefix, not on membership of ``COMMANDS``. A
    ``COMMANDS`` membership test would answer "no" for a *misspelled* varscore
    module and cause ``validate_all`` to skip it as somebody else's command --
    passing over exactly the mistake this contract exists to catch. Anything under
    ``varscore.`` is claimed here and must survive ``validate_argv``; a command
    belonging to another tool (motif hit-calling, for example) is not.
    """
    return bool(argv) and argv[0].startswith(MODULE_PREFIX)


def validate_argv(argv: Sequence[str]) -> argparse.Namespace:
    """Check one ``python -m`` argv list against its command's real parser.

    ``argv[0]`` is the command module and the rest are its flags -- the exact list
    handed to the container, minus the ``python -m`` the entrypoint supplies.
    Returns the parsed namespace so a caller can additionally assert *which* paths
    landed on which flags, not merely that the argv parsed.

    Raises ``ArgvContractError`` if the command is unknown or the parser rejects
    the arguments. argparse writes its diagnostic to stderr and raises
    ``SystemExit``; both are captured so this behaves like a normal function
    instead of terminating the caller.
    """
    parser = parser_for(argv[0])
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            return parser.parse_args(list(argv[1:]))
    except SystemExit as exc:
        detail = stderr.getvalue().strip() or "argparse rejected the arguments"
        msg = f"{argv[0]} rejected its arguments:\n{detail}"
        raise ArgvContractError(msg) from exc


def validate_all(argvs: Iterable[Sequence[str]]) -> None:
    """Validate every varscore argv in ``argvs``, skipping other tools' commands.

    The convenience form for checking a whole orchestration plan at once: pass
    each task's command list and any task running a different tool is ignored.
    Anything under the ``varscore.`` prefix is validated, including a module that
    does not exist -- see ``is_varscore_command``. Raises ``ArgvContractError`` on
    the first argv that does not match.
    """
    for argv in argvs:
        if is_varscore_command(argv):
            validate_argv(argv)
