"""
Reference plant primitives used by gas routing and recovery.

The Streamlit app lets users move the accounting boundary, but emission factors
for electricity and process gas remain anchored to a deterministic reference
plant. This module provides the lightweight dataclasses used to represent those
metrics so they can be cached or persisted independent of scenario choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReferencePlantConfig:
    """Declarative description of the fixed reference chain."""

    route_preset: str = "BF-BOF"
    stage_ref: str = "IP3"
    demand_qty: float = 1000.0


@dataclass(frozen=True)
class ReferencePlantMetrics:
    """
    Anchors for emission factor blending.

    Attributes:
        inside_elec_ref: Total in-mill electricity (MJ) for the reference chain.
        total_gas_consumption: Total process-gas consumption (MJ) for the reference chain.
        util_efficiency: Utility plant efficiency (MJ electricity produced per MJ gas).
    """

    inside_elec_ref: float
    total_gas_consumption: float
    util_efficiency: float


DEFAULT_REFERENCE_CONFIG = ReferencePlantConfig()


def make_reference_metrics(
    inside_elec_ref: float,
    total_gas_consumption: float,
    util_efficiency: Optional[float] = None,
) -> ReferencePlantMetrics:
    """
    Helper used by callers that pre-compute the reference metrics elsewhere
    (e.g., during a preprocessing step or cached on disk).
    """
    util_eff = 0.0 if util_efficiency is None else float(util_efficiency)
    return ReferencePlantMetrics(
        inside_elec_ref=float(inside_elec_ref),
        total_gas_consumption=float(total_gas_consumption),
        util_efficiency=util_eff,
    )
