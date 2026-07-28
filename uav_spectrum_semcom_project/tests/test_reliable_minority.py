import numpy as np

from spectrum_semcom.multigranular_semantics import SemanticQuality
from spectrum_semcom.reliable_minority import AuditedOccupancyReport, audited_quality_score, confirmed_conflict_retransmission, reliable_minority_fusion


def quality(snr=6,confidence=.8,clip=0,age=.05,reliability=.95,calibration=0):
    return SemanticQuality(snr,confidence,.2,clip,0,0,10,age,reliability,calibration)


def test_reliable_occupied_minority_is_protected_against_degraded_majority() -> None:
    reports=[AuditedOccupancyReport(i,np.asarray([.05,.05,.05]),quality(snr=-6,clip=.4,calibration=.4)) for i in range(3)]
    reports.append(AuditedOccupancyReport(3,np.asarray([.9,.05,.8]),quality(snr=12,confidence=.95)))
    result=reliable_minority_fusion(reports,disagreement_threshold=.1,quality_ratio=1.2)
    assert result.protected_node==3 and result.protected_channels
    assert result.occupancy[0]>.3


def test_untrusted_singleton_only_triggers_retransmission() -> None:
    reports=[AuditedOccupancyReport(i,np.asarray([.05,.05]),quality()) for i in range(3)]
    reports.append(AuditedOccupancyReport(3,np.asarray([.9,.05]),quality(snr=-12,confidence=.2,clip=.5)))
    result=reliable_minority_fusion(reports,disagreement_threshold=.1)
    assert result.protected_node==-1 and result.retransmission_triggered
    assert audited_quality_score(reports[-1])<audited_quality_score(reports[0])


def test_confirmed_G3_corrects_sparse_conflict_toward_best_audited_node() -> None:
    reports=[AuditedOccupancyReport(i,np.asarray([.05,.05,.05]),quality(snr=-6,clip=.4,calibration=.4)) for i in range(3)]
    reports.append(AuditedOccupancyReport(3,np.asarray([.9,.05,.8]),quality(snr=12,confidence=.95)))
    result=confirmed_conflict_retransmission(reports,trigger_threshold=.2)
    assert result.retransmission_triggered and result.protected_node==3
    assert set(result.protected_channels)=={0,2}
