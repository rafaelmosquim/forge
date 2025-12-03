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


__all__ = [
    "analyze_energy_costs",
    "analyze_material_costs",
]

