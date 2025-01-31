import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from varscore.scoring import _compute_active_allele_quantile, _compute_jsd, _score_variant_df



def test_score_variant_df():
    variant_df = _score_variant_df(
        pd.DataFrame({
            "chr": ["chr1"] * 5,
            "pos": [1, 2, 3, 4, 5],
            "ref": ["A"] * 5,
            "alt": ["T"] * 5,
            "variant_id": ["v1", "v2", "v3", "v4", "v5"],
        }),
        np.zeros((5, 1000)),
        np.zeros((5, 1000)),
        np.zeros((5, 1)),
        np.zeros((5, 1)),
        np.array(range(1000)),
    )

    # assert has columns
    assert "lfc" in variant_df.columns
    assert "lfc_pval" in variant_df.columns
    assert "jsd" in variant_df.columns
    assert "active_allele_quantile" in variant_df.columns
    assert "ips" in variant_df.columns
    assert len(variant_df.columns) == 10
    assert len(variant_df) == 5


def test_compute_active_allele_quantile():
    ref_pred_logcts = np.array([1, 2, 400, 4, 5])
    alt_pred_logcts = np.array([200, 350, 4, 5, 900])
    peaks_dist = np.array(range(1000))

    active_allele_quantile = _compute_active_allele_quantile(
        ref_pred_logcts, alt_pred_logcts, peaks_dist
    )

    assert_array_equal(active_allele_quantile, np.array([0.2, 0.35, 0.4, 0.005, 0.9]))


@pytest.mark.parametrize("ref_logits, alt_logits, expected_jsd", [
    # Identical logits → JSD should be 0
    (np.zeros((5, 1000)), np.zeros((5, 1000)), np.zeros(5)),
    
    # One-hot logits → high JSD
    (np.array([[1000] + [-1000] * 999]), np.array([[-1000] + [1000] + [-1000] * 998]), np.array([1.0])),
])
def test_compute_jsd_identical_distrs(ref_logits, alt_logits, expected_jsd):
    jsd = _compute_jsd(ref_logits, alt_logits)

    assert_array_equal(jsd, expected_jsd)