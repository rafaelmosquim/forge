"""Descriptor facade.

Provides a stable import path for descriptor-related helpers, while keeping
legacy modules in place. Downstream code can use:

    from forge.descriptor import load_sector_descriptor
    from forge.descriptor import build_stage_material_map

and so on.
"""

from .sector_descriptor import *  # re-export
from .scenario_resolver import *  # re-export

__all__ = [
    # sector descriptor
    'load_sector_descriptor', 'StageMenuItem', 'SectorDescriptor', 'RoutePreset',
    # scenario resolver helpers
    'build_stage_material_map', 'resolve_stage_material', 'reference_stage_for_gas',
    'match_route', 'match_route_in_name', 'build_route_mask_for_descriptor', 'resolve_feed_mode',
]

