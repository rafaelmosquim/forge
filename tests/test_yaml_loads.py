import glob, yaml, pathlib

YAML_GLOBS = ["data/**/*.yml", "data/**/*.yaml", "configs/**/*.yml", "configs/**/*.yaml"]

def _all_yaml_paths():
    paths = []
    for pat in YAML_GLOBS:
        paths.extend(glob.glob(pat, recursive=True))
    # keep only files that exist
    return [p for p in paths if pathlib.Path(p).is_file()]

def test_all_yaml_files_parse_to_mappings():
    for p in _all_yaml_paths():
        with open(p, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        assert isinstance(doc, (dict, list)), f"{p} did not parse to dict/list"
