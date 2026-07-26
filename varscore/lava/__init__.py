"""Integration with a ``lava-core`` orchestration platform.

varscore is a library and a set of container commands: it scores variants, it
does not schedule work. A platform built on ``lava-core`` does the scheduling,
and to do that it needs a ``ModelPlugin`` describing how to invoke varscore and
how to read what it produced. This subpackage is that description.

It lives here, rather than on the platform side, because everything in it is a
fact about varscore -- the command modules and their flags, the fold count the
CLI enforces, the prioritization thresholds. Stated here, those facts sit next to
the code that implements them and can be tested against it.

The subpackage is in two halves, split by what they depend on:

- ``varscore.lava.commands`` -- the argv contract. Imports **no** orchestration
  framework, so it and its tests run in any varscore install, on every Python
  version varscore supports. This is what keeps a renamed flag caught by ordinary
  CI rather than by a container failing mid-job.
- ``varscore.lava.chrombpnet`` -- the plugin itself, which does import
  ``lava-core``. Install it separately from ``requirements-lava.txt``; it needs
  read access to a private repository and Python 3.12. A plain
  ``pip install varscore`` neither needs nor gets it, and the scoring container
  does not install it.

Importing this package does not pull in ``lava-core``: ``ChromBPNetPlugin`` is
resolved lazily on attribute access, so ``from varscore.lava import commands``
works in a base install.

With ``lava-core`` installed, no wiring is required -- ``ChromBPNetPlugin`` is
advertised through the ``lava.model_plugins`` entry point and the platform finds
it. Call ``register()`` only to override a plugin already registered for the same
architecture; see its docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from varscore.lava.commands import ArgvContractError, validate_all, validate_argv


if TYPE_CHECKING:
    from varscore.lava.chrombpnet import ChromBPNetPlugin


__all__ = ["ArgvContractError", "ChromBPNetPlugin", "register", "validate_all", "validate_argv"]


def __getattr__(name: str) -> Any:
    """Resolve ``ChromBPNetPlugin`` on demand so importing this package stays light.

    The plugin module imports ``lava-core``, which most varscore installs do not
    have. Importing it eagerly here would make ``from varscore.lava import
    commands`` -- the framework-free half -- fail wherever ``lava-core`` is
    absent, and take the argv contract's tests out of ordinary CI with it.
    """
    if name == "ChromBPNetPlugin":
        from varscore.lava.chrombpnet import ChromBPNetPlugin

        return ChromBPNetPlugin
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def register() -> None:
    """Register varscore's plugins directly, taking precedence over entry points.

    Not needed in the ordinary case: with ``lava-core`` installed,
    ``ChromBPNetPlugin`` is advertised through the ``lava.model_plugins`` entry
    point and the platform discovers it on its own.

    It matters when another installed distribution advertises a plugin for the
    same architecture. The registry merges every entry point for a group into one
    mapping keyed by architecture name, so two providers of ``CHROMBPNET`` leave
    the winner down to discovery order -- not something to rely on. Programmatic
    registration is applied after entry points and overrides them, so calling this
    at startup makes varscore's plugin win deterministically.
    """
    from lava_core.plugins.model.plugin import get_model_plugin_registry

    from varscore.lava.chrombpnet import ChromBPNetPlugin

    get_model_plugin_registry().register(ChromBPNetPlugin.model_type, ChromBPNetPlugin)
