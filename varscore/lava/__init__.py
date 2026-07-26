"""Integration with a ``lava-core`` orchestration platform.

varscore is a library and a set of container commands: it scores variants, it
does not schedule work. A platform built on ``lava-core`` does the scheduling,
and to do that it needs a ``ModelPlugin`` describing how to invoke varscore and
how to read what it produced. This subpackage is that description.

It lives here, rather than on the platform side, because everything in it is a
fact about varscore -- the command modules and their flags, the fold count the
CLI enforces, the prioritization thresholds. Stated here, those facts sit next to
the code that implements them and can be tested against it: see
``varscore.lava.commands`` for the argv contract and ``tests/test_lava_plugin.py``
for the checks that hold the two in agreement.

**This subpackage is optional and isolated.** It is the only part of varscore
that imports ``lava-core``, and nothing in the scoring path imports it. Install
it with the ``[lava]`` extra, which needs read access to a private repository and
Python 3.12; a plain ``pip install varscore`` neither needs nor gets it, and the
scoring container does not install it.

Installing the extra is normally all that is required -- ``ChromBPNetPlugin`` is
advertised through the ``lava.model_plugins`` entry point and the platform finds
it. Call ``register()`` only to override a plugin already registered for the same
architecture; see its docstring.
"""

from __future__ import annotations

from varscore.lava.chrombpnet import ChromBPNetPlugin
from varscore.lava.commands import ArgvContractError, validate_argv


__all__ = ["ChromBPNetPlugin", "ArgvContractError", "register", "validate_argv"]


def register() -> None:
    """Register varscore's plugins directly, taking precedence over entry points.

    Not needed in the ordinary case: installing the ``[lava]`` extra advertises
    ``ChromBPNetPlugin`` through the ``lava.model_plugins`` entry point and the
    platform discovers it on its own.

    It matters when another installed distribution advertises a plugin for the
    same architecture. The registry merges every entry point for a group into one
    mapping keyed by architecture name, so two providers of ``CHROMBPNET`` leave
    the winner down to discovery order -- not something to rely on. Programmatic
    registration is applied after entry points and overrides them, so calling this
    at startup makes varscore's plugin win deterministically.
    """
    from lava_core.plugins.model.plugin import get_model_plugin_registry

    get_model_plugin_registry().register(ChromBPNetPlugin.model_type, ChromBPNetPlugin)
