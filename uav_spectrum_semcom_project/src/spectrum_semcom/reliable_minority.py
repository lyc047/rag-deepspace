"""Audited reliable-minority protection for spectrum occupancy fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multigranular_semantics import SemanticQuality


@dataclass(frozen=True)
class AuditedOccupancyReport:
    node_id: int
    occupancy: np.ndarray
    quality: SemanticQuality


@dataclass(frozen=True)
class MinorityFusionResult:
    occupancy: np.ndarray
    protected_node: int
    protected_channels: tuple[int, ...]
    retransmission_triggered: bool
    maximum_disagreement: float


def confirmed_conflict_retransmission(reports: list[AuditedOccupancyReport], *, trigger_threshold: float=.2, maximum_corrected_channels: int=2, blend_strength: float=.5, freshness_tau_s: float=.5, noise_scale_db: float=4.0) -> MinorityFusionResult:
    """Trigger G3 on sparse channel conflict, then blend toward confirmed node."""
    if len(reports)<2: raise ValueError("confirmed conflict handling requires at least two reports")
    values=np.stack([np.asarray(report.occupancy,dtype=np.float64) for report in reports]);spread=np.max(values,axis=0)-np.min(values,axis=0);maximum=float(np.max(spread));base=audited_quality_mean(reports,freshness_tau_s,noise_scale_db)
    if maximum<trigger_threshold:return MinorityFusionResult(base,-1,(),False,maximum)
    scores=np.asarray([audited_quality_score(report,freshness_tau_s,noise_scale_db) for report in reports]);candidate=int(np.argmax(scores));difference=np.abs(values[candidate]-base);eligible=np.flatnonzero(spread>=trigger_threshold);selected=tuple(int(index) for index in eligible[np.argsort(difference[eligible])[::-1][:maximum_corrected_channels]]) if eligible.size else ();fused=base.copy()
    if selected:fused[list(selected)]+=blend_strength*(values[candidate,list(selected)]-base[list(selected)])
    return MinorityFusionResult(np.clip(fused,0,1),reports[candidate].node_id,selected,True,maximum)


def audited_quality_score(report: AuditedOccupancyReport, freshness_tau_s: float = .5, noise_scale_db: float = 4.0) -> float:
    q=report.quality
    sensing=1/(1+np.exp(-float(q.sensing_snr_db)/4)); frontend=(1-q.clipping_ratio)*(1-q.out_of_band_leakage_ratio)*np.exp(-q.noise_floor_stability_db/noise_scale_db); freshness=np.exp(-q.age_s/freshness_tau_s)
    return float(sensing*q.prediction_confidence*q.report_success_probability*freshness*frontend*(1-q.calibration_error))


def audited_quality_mean(reports: list[AuditedOccupancyReport], freshness_tau_s: float = .5, noise_scale_db: float = 4.0) -> np.ndarray:
    if not reports: raise ValueError("at least one report is required")
    values=np.stack([np.asarray(report.occupancy,dtype=np.float64) for report in reports]); weights=np.asarray([audited_quality_score(report,freshness_tau_s,noise_scale_db) for report in reports]);
    if float(weights.sum())<=1e-12: return values.mean(axis=0)
    positive=weights[weights>1e-12]
    if positive.size: weights=np.minimum(weights,2*float(np.median(positive)))
    return np.average(values,axis=0,weights=weights)


def reliable_minority_fusion(reports: list[AuditedOccupancyReport], *, freshness_tau_s: float=.5, noise_scale_db: float=4.0, disagreement_threshold: float=.15, channel_excess_threshold: float=.2, quality_ratio: float=1.25, minimum_quality: float=.05, protection_strength: float=.5, maximum_protected_channels: int=2) -> MinorityFusionResult:
    if len(reports)<2: raise ValueError("reliable minority fusion requires at least two reports")
    values=np.stack([np.asarray(report.occupancy,dtype=np.float64) for report in reports]); base=audited_quality_mean(reports,freshness_tau_s,noise_scale_db); consensus=np.median(values,axis=0); disagreement=np.mean(np.abs(values-consensus[None]),axis=1); scores=np.asarray([audited_quality_score(report,freshness_tau_s,noise_scale_db) for report in reports]); candidate=int(np.argmax(scores)); others=np.delete(scores,candidate); majority=float(np.median(others)) if others.size else 0.0; trustworthy=disagreement[candidate]>=disagreement_threshold and scores[candidate]>=minimum_quality and scores[candidate]>=quality_ratio*max(majority,1e-12)
    excess=values[candidate]-base; eligible=np.flatnonzero(excess>=channel_excess_threshold) if trustworthy else np.empty(0,dtype=int); selected=tuple(int(index) for index in eligible[np.argsort(excess[eligible])[::-1][:maximum_protected_channels]]) if eligible.size else (); fused=base.copy()
    if selected: fused[list(selected)]+=protection_strength*excess[list(selected)]
    maximum=float(np.max(disagreement)); ambiguous=maximum>=disagreement_threshold and not bool(selected)
    return MinorityFusionResult(np.clip(fused,0,1),reports[candidate].node_id if selected else -1,selected,ambiguous,maximum)
