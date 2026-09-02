"""Energy and material cost analysis utilities.

These functions are duplicated from the monolith to provide a clean import path
and simplify testing in isolation. Signatures and behavior are preserved.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def analyze_energy_costs(bal_data, en_price: Dict[str, float]) -> float:
    """Calculate total energy cost from an energy balance DataFrame.

    Expects a 'TOTAL' row with MJ per carrier. Multiplies by carrier prices
    provided in `en_price` and returns the sum.
    """
    total_cost = 0.0

    total_row = bal_data.loc['TOTAL']
    for carrier, energy_mj in total_row.items():
        if carrier in en_price:
            cost = float(energy_mj) * float(en_price[carrier])
            total_cost += cost
            logger.debug("%s: %0.1f MJ x $%0.2f = $%0.2f", carrier, energy_mj, en_price[carrier], cost)

    return total_cost


def analyze_material_costs(matrix_data, mat_price: Dict[str, float], external_rows: Optional[Iterable[str]] = None) -> float:
    """Calculate total material cost from external purchase rows in the balance matrix.

    Scans specific rows representing external purchases; sums positive material
    quantities and multiplies by provided prices.
    """
    material_cost = 0.0

    default_rows = [
        'External Inputs',
        'Scrap Purchase',
        'Limestone from Market',
        'Burnt Lime from market',
        'Dolomite from market',
        'Nitrogen from market',
        'Oxygen from market',
    ]
    external_purchase_rows = list(external_rows) if external_rows else default_rows

    logger.debug("Analyzing material costs from external purchase rows")

    total_external: Dict[str, float] = defaultdict(float)
    for row_name in external_purchase_rows:
        if row_name in matrix_data.index:
            row_data = matrix_data.loc[row_name]
            logger.debug("External purchase row: %s", row_name)
            for material, quantity in row_data.items():
                q = float(quantity)
                if abs(q) > 1e-9 and q > 0:
                    total_external[material] += q
                    logger.debug("  %s: %0.4f units", material, q)

    logger.info("TOTAL EXTERNAL MATERIAL PURCHASES:")
    for material, quantity in sorted(total_external.items()):
        if material in mat_price:
            cost = quantity * float(mat_price[material])
            material_cost += cost
            logger.info("%s %0.4f units x $%0.2f = $%0.2f", material, quantity, mat_price[material], cost)
        else:
            logger.info("%s %0.4f units - NO PRICE AVAILABLE", material, quantity)

    logger.info("TOTAL MATERIAL COST: $%0.2f", material_cost)
    return material_cost


def _annualize_capex(capex_value: float, interest_rate: float, lifespan_years: float) -> float:
    """Compute an annualized payment using a standard capital recovery factor."""
    try:
        principal = float(capex_value)
        years = float(lifespan_years)
    except Exception:
        return 0.0
    if principal <= 0.0 or years <= 0.0:
        return 0.0
    try:
        rate = float(interest_rate)
    except Exception:
        rate = 0.0
    if rate <= 0.0:
        return principal / years
    factor = rate * (1.0 + rate) ** years / (((1.0 + rate) ** years) - 1.0)
    return principal * factor


def _infer_capex_route(route_preset: str, production_routes: Dict[str, int]) -> str:
    """Infer a capex route key from selected production routes."""
    if production_routes.get("Blast Furnace", 0) > 0:
        return "BF-BOF"
    if production_routes.get("Direct Reduction Iron", 0) > 0:
        return "DRI-EAF"
    if production_routes.get("Electric Arc Furnace", 0) > 0:
        return "EAF-scrap"
    key = str(route_preset or "").strip()
    mapping = {
        "BF-BOF": "BF-BOF",
        "DRI-EAF": "DRI-EAF",
        "EAF-Scrap": "EAF-scrap",
        "EAF-scrap": "EAF-scrap",
    }
    return mapping.get(key, key)


def compute_capex_costs(
    capex_cfg: Dict[str, object],
    route_preset: str,
    production_routes: Dict[str, int],
) -> Dict[str, Optional[float]]:
    """Compute capex-related payments and oem cost for a scenario."""
    if not isinstance(capex_cfg, dict):
        return {
            "capex_total": None,
            "capex_payment": None,
            "relining_payment": None,
            "oem_cost": None,
        }

    capex_table = capex_cfg.get("capex", {}) if isinstance(capex_cfg, dict) else {}
    lifespan_table = capex_cfg.get("lifespan", {}) if isinstance(capex_cfg, dict) else {}
    relining_table = capex_cfg.get("relining", {}) if isinstance(capex_cfg, dict) else {}
    oem_table = capex_cfg.get("oem_rate", {}) if isinstance(capex_cfg, dict) else {}
    interest_rate = capex_cfg.get("interest_rate", 0.0) if isinstance(capex_cfg, dict) else 0.0

    route_key = _infer_capex_route(route_preset, production_routes)
    route_capex = float(capex_table.get(route_key, 0.0) or 0.0)
    conformation_capex = float(capex_table.get("Conformation", 0.0) or 0.0)
    if route_key == "DRI-EAF":
        route_capex += float(capex_table.get("EAF-scrap", 0.0) or 0.0)
    capex_total = route_capex + conformation_capex

    lifespan_years = float(lifespan_table.get(route_key, 0.0) or 0.0)
    capex_payment = _annualize_capex(capex_total, float(interest_rate or 0.0), lifespan_years)

    relining_payment = None
    if route_key == "BF-BOF":
        relining_years = float(relining_table.get("BF-BOF", 20.0) or 20.0)
        relining_capex = 0.3 * float(capex_table.get("BF-BOF", 0.0) or 0.0)
        relining_payment = _annualize_capex(relining_capex, float(interest_rate or 0.0), relining_years)

    oem_rate = float(oem_table.get(route_key, 0.0) or 0.0)
    oem_cost = capex_total * oem_rate

    return {
        "capex_total": capex_total,
        "capex_payment": capex_payment,
        "relining_payment": relining_payment,
        "oem_cost": oem_cost,
    }


__all__ = [
    "analyze_energy_costs",
    "analyze_material_costs",
    "compute_capex_costs",
]

