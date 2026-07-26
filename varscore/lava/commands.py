"""The command line varscore exposes to an orchestration layer, in one place.

A platform that runs varscore in a container never imports it. It builds an argv
list and runs it -- the scoring image's entrypoint is ``python -m``, so a task is
``python -m varscore.scoring.chrombpnet.score -m ... -p ... -g ...``. That argv
*is* the interface between varscore and its caller.

Historically that interface was written down only on the caller's side, as string
literals, with nothing checking the two halves agreed. Renaming a flag here broke
the caller silently and the failure surfaced inside a GPU container, minutes into
a job, as an argparse usage error.

This module moves the contract to the side that implements it. ``COMMANDS`` names
every command an orchestrator may invoke, and ``validate_argv`` checks one argv
list against that command's real parser. A caller can then assert in its own test
suite that every command it emits is one varscore actually accepts, so a flag
rename fails a test here instead of a job in production.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
from typing import Iterable, Sequence


# --- command module paths ---------------------------------------------------
# Each is a module runnable with `python -m`, which is what the scoring image's
# entrypoint prepends. Import these constants rather than retyping the strings.

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

#: The number of cross-validation folds a ChromBPNet model is scored with.
#:
#: This is a hard structural constraint, not a default. ``ingest``,
#: ``predictions``, and ``average_interpretations`` each declare exactly five
#: ``required=True`` fold flags (``-f0`` through ``-f4``), so a model scored with
#: any other fold count cannot be passed to them at all. A caller that varies its
#: own fold count independently of this value emits argv those commands reject;
#: ``validate_argv`` catches that.
NUM_FOLDS = 5


class ArgvContractError(ValueError):
    """An argv list does not match the varscore command it names.

    Raised for an unknown command module, and for argv that the command's own
    parser rejects (an unrecognized flag, a missing required one, a bad value).
    The message carries argparse's own diagnostic.
    """


def parser_for(module_path: str) -> argparse.ArgumentParser:
    """Return the real ``argparse`` parser for one varscore command.

    Imports the command module and calls its ``build_parser()``. Raises
    ``ArgvContractError`` if ``module_path`` is not a known command, so a typo in
    a caller's module string fails here rather than as a container ``ModuleNotFoundError``.
    """
    if module_path not in COMMANDS:
        known = "\n  ".join(COMMANDS)
        msg = f"{module_path!r} is not a varscore command. Known commands:\n  {known}"
        raise ArgvContractError(msg)
    module = importlib.import_module(module_path)
    return module.build_parser()  # type: ignore[no-any-return]


def is_varscore_command(argv: Sequence[str]) -> bool:
    """Whether ``argv`` invokes a varscore command.

    An orchestration plan may mix varscore tasks with tasks that run other images
    entirely (motif hit-calling, for example). A caller validating a whole plan
    uses this to select the tasks this module can speak for.
    """
    return bool(argv) and argv[0] in COMMANDS


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
    """Validate every varscore argv in ``argvs``, skipping non-varscore commands.

    The convenience form for checking a whole orchestration plan at once: pass
    each task's command list and any task that runs a different image is ignored.
    Raises ``ArgvContractError`` on the first argv that does not match.
    """
    for argv in argvs:
        if is_varscore_command(argv):
            validate_argv(argv)
