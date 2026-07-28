"""Protocol-width utilities for Stage-5 scale and query-set boundary tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from spectrum_semcom.stage5_query_bundle import query_bundle_payload_width


@dataclass(frozen=True)
class RepresentationPlan:
    """Application packets needed for one semantic refresh operation."""

    name: str
    application_packet_bits: tuple[int, ...]
    aggregation: str


def demand_set_from_fractions(
    n_channels: int,
    fractions: tuple[str, ...] | list[str],
) -> tuple[int, ...]:
    """Map exact rational bandwidth fractions to a unique ordered demand set."""
    if n_channels < 2 or not fractions:
        raise ValueError("invalid channel count or empty query fractions")
    demands = []
    for value in fractions:
        fraction = Fraction(str(value))
        if not 0 < fraction < 1:
            raise ValueError("query fractions must lie strictly between zero and one")
        scaled = fraction * int(n_channels)
        if scaled.denominator != 1:
            raise ValueError("query fraction does not map to an integer demand")
        demands.append(int(scaled))
    if len(set(demands)) != len(demands):
        raise ValueError("query fractions produce duplicate demands")
    return tuple(demands)


def index_width(n_channels: int, demand_channels: int) -> int:
    candidates = int(n_channels) - int(demand_channels) + 1
    if n_channels < 1 or candidates < 1:
        raise ValueError("invalid channel count or demand")
    return int(math.ceil(math.log2(candidates)))


def representation_plans(
    *,
    n_channels: int,
    demand_channels: tuple[int, ...] | list[int],
    application_header_bits: int,
    soft_power_bits_per_channel: int,
) -> tuple[RepresentationPlan, ...]:
    """Build fair packet plans for one current query or all-query refreshes."""
    demands = tuple(int(value) for value in demand_channels)
    if application_header_bits < 40 or soft_power_bits_per_channel < 1:
        raise ValueError("invalid application header or quantizer width")
    widths = tuple(index_width(n_channels, demand) for demand in demands)
    bundle_width = query_bundle_payload_width(n_channels, demands)
    if bundle_width != sum(widths):
        raise AssertionError("bundle payload accounting is inconsistent")
    return (
        RepresentationPlan(
            "single_index_mean",
            tuple(application_header_bits + width for width in widths),
            "mean_over_query_alternatives",
        ),
        RepresentationPlan(
            "bundle_all_queries",
            (application_header_bits + bundle_width,),
            "joint_refresh",
        ),
        RepresentationPlan(
            "separate_indices_all_queries",
            tuple(application_header_bits + width for width in widths),
            "joint_refresh",
        ),
        RepresentationPlan(
            "occupancy_vector_all_queries",
            (application_header_bits + int(n_channels),),
            "joint_refresh",
        ),
        RepresentationPlan(
            "soft_power_all_queries",
            (
                application_header_bits
                + int(n_channels) * int(soft_power_bits_per_channel),
            ),
            "joint_refresh",
        ),
    )

