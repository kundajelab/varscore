import numpy as np
from numpy.testing import assert_array_equal

from varscore.scoring import _compute_active_allele_quantile



def test_compute_active_allele_quantile():
    ref_pred_logcts = np.array([1, 2, 400, 4, 5])
    alt_pred_logcts = np.array([200, 350, 4, 5, 900])
    peaks_dist = np.array(range(1000))

    active_allele_quantile = _compute_active_allele_quantile(
        ref_pred_logcts, alt_pred_logcts, peaks_dist
    )

    assert_array_equal(active_allele_quantile, np.array([0.2, 0.35, 0.4, 0.005, 0.9]))
