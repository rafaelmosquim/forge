import os
from pathlib import Path

from forge.descriptor import (
    load_sector_descriptor,
    build_stage_material_map,
    match_route,
    resolve_feed_mode,
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

