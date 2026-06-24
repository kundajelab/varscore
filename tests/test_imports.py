"""Smoke tests for the package's public surface and the lightweight install path.

These guard two productionization invariants:
  1. Every core (TensorFlow-free) module imports cleanly.
  2. Importing the core API does not drag in the heavy `[model]` (TensorFlow)
     stack.

The companion `package` CI job additionally builds a wheel and runs equivalent
checks against a *clean install*, which is what catches missing-subpackage
packaging regressions that a source-tree run cannot.
"""

import importlib
import sys

import pytest

CORE_MODULES = [
    "varscore",
    "varscore.annotation.annotate",
    "varscore.prioritization",
    "varscore.preprocessing.validate",
    "varscore.preprocessing.pipeline",
    "varscore.preprocessing.region_filter",
    "varscore.scoring.alphamissense.score",
    "varscore.scoring.chrombpnet.score",
    "varscore.annotation.regions",
    "varscore.core.io",
    "varscore.annotation.maf",
    "varscore.scoring.alphamissense.lookup",
]

PUBLIC_API = [
    "__version__",
    "annotate_variant",
    "annotate_variants",
    "AnnotatedVariant",
    "VariantAnnotationInput",
    "prioritize_variants",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
    importlib.import_module(module)


def test_public_api_is_exported():
    import varscore

    for name in PUBLIC_API:
        assert hasattr(varscore, name), f"varscore.{name} is missing from the public API"


def test_core_import_is_tensorflow_free():
    # Importing the whole core surface must not pull in TensorFlow.
    for module in CORE_MODULES:
        importlib.import_module(module)
    assert "tensorflow" not in sys.modules, "TensorFlow leaked into the core import path"
