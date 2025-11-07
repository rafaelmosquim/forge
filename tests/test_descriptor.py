import os
from pathlib import Path

from forge.descriptor import (
    load_sector_descriptor,
    build_stage_material_map,
    match_route,
    resolve_feed_mode,
    build_route_mask_for_descriptor,
)


def test_descriptor_stage_map_and_route(repo_root):
    data_dir = repo_root / 'datasets' / 'steel' / 'likely'
    assert data_dir.exists(), 'dataset missing'

    d = load_sector_descriptor(str(data_dir))
    stage_map = build_stage_material_map(d)
    # Should at least include Finished stage via dataset or fallback
    assert 'Finished' in stage_map
    # Route alias should resolve
    assert match_route(d, 'BF-BOF') == 'BF-BOF'
    # Default feed mode for EAF-Scrap is 'scrap'
    assert resolve_feed_mode(d, 'EAF-Scrap') == 'scrap'

    # build_route_mask_for_descriptor should disable EAF on BF-BOF
    class R:  # minimal Process-like
        def __init__(self, name):
            self.name = name
    recs = [R('Blast Furnace'), R('Direct Reduction Iron'), R('Electric Arc Furnace')]
    mask = build_route_mask_for_descriptor(d, 'BF-BOF', recs)
    assert mask['Electric Arc Furnace'] == 0
    assert mask['Direct Reduction Iron'] == 0
    assert mask['Blast Furnace'] == 1
