"""This module handles variant preprocessing functional flows.

It takes in a variants TSV file and returns coding and non-coding
variant DataFrames.
"""

from varscore.variant_region_filter import filter_variants_by_region
from varscore.validate_variants import validate_variants

import pandas as pd
from typing import Tuple

from varscore.utils.logging_config import get_logger

logger = get_logger(__name__)


def preprocess_variants(variant_tsv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess a variants DataFrame through validation and region filtering.

    Args:
        variant_tsv: DataFrame of variants to preprocess.

    Returns:
        Tuple of (valid_variants_df, invalid_variants_df, coding_df, noncoding_df).
        coding_df and noncoding_df are empty DataFrames when there are no valid variants.

    Raises:
        ValueError: If variant_tsv is not a DataFrame.
        RuntimeError: If validation or region filtering fails unexpectedly.
    """
    if not isinstance(variant_tsv, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame, got {type(variant_tsv).__name__}.")

    logger.info("Starting variant preprocessing on %d rows.", len(variant_tsv))

    try:
        valid_df, invalid_df = validate_variants(variant_tsv)
    except Exception as e:
        raise RuntimeError(f"Variant validation failed: {e}") from e

    logger.info(
        "Validation complete: %d valid, %d invalid variants.",
        len(valid_df),
        len(invalid_df),
    )

    coding_df = pd.DataFrame()
    noncoding_df = pd.DataFrame()

    if not valid_df.empty:
        logger.info("Filtering %d valid variants by region.", len(valid_df))
        try:
            coding_df, noncoding_df = filter_variants_by_region(valid_df)
        except Exception as e:
            raise RuntimeError(f"Region filtering failed: {e}") from e

        logger.info(
            "Region filtering complete: %d coding, %d non-coding variants.",
            len(coding_df),
            len(noncoding_df),
        )
    else:
        logger.warning("No valid variants to filter by region.")

    return valid_df, invalid_df, coding_df, noncoding_df
