import glob, pathlib

# “Resolved” configs must not contain choice markers or pipes.
AMBIGUITY_MARKERS = [" or ", "||", "??", "choice:", " alt:", "| "]

def test_resolved_configs_have_no_ambiguity_tokens():
    for p in glob.glob("configs/*_resolved.yml"):
        text = pathlib.Path(p).read_text(encoding="utf-8")
        assert not any(tok in text for tok in AMBIGUITY_MARKERS), f"Ambiguity in {p}"
