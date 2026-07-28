import numpy as np

from spectrum_semcom.quality_diagnostics import audited_quality_scores, binary_auc, controlled_faults, expected_calibration_error


def test_controlled_faults_reduce_quality_score() -> None:
    quality = np.asarray([[6, .8, .2, 0, 0, .1, 10, .01, .95, .05]], dtype=float)
    healthy = audited_quality_scores(quality)
    for faulty in controlled_faults(quality).values():
        assert audited_quality_scores(faulty)[0] < healthy[0]


def test_ece_and_auc() -> None:
    ece, table = expected_calibration_error(np.asarray([.1, .2, .8, .9]), np.asarray([0, 0, 1, 1]), 2)
    assert 0 <= ece <= 1
    assert len(table) == 2
    assert binary_auc(np.asarray([.8, .9]), np.asarray([.1, .2])) == 1.0
