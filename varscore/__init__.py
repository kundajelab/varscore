"""varscore: common functions for variant effect prediction & prioritization.

The names re-exported here are the TensorFlow-free core API (annotation,
region classification, lookups, prioritization). Model scoring / SHAP live
under ``varscore.scoring.chrombpnet`` (``.score``, ``.predictions``,
``.interpret``) and require the ``[model]`` extra; import those modules directly.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("varscore")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+dev"

from varscore.annotation.annotate import (
    AnnotatedVariant,
    VariantAnnotationInput,
    annotate_variant,
    annotate_variants,
)
from varscore.prioritization import prioritize_variants

__all__ = [
    "__version__",
    "AnnotatedVariant",
    "VariantAnnotationInput",
    "annotate_variant",
    "annotate_variants",
    "prioritize_variants",
]
