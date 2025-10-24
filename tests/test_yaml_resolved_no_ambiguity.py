import glob, pathlib, re, pytest, yaml

GLOB = "datasets/steel/*/scenarios/*_resolved.yml"
BAD = re.compile(r"\b(or|one of|choice|alt)\b", flags=re.IGNORECASE)

def _normalized_text(p: pathlib.Path) -> str:
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return yaml.dump(data, sort_keys=True)

@pytest.mark.skipif(len(glob.glob(GLOB)) == 0, reason="no *_resolved.yml scenarios to check")
def test_resolved_configs_have_no_ambiguity_tokens():
    for path in glob.glob(GLOB):
        txt = _normalized_text(pathlib.Path(path))
        assert not BAD.search(txt), f"Ambiguity markers in {path}"
