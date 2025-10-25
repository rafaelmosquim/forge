# Contributing to FORGE

Thanks for your interest in improving FORGE! We welcome bug reports, small fixes, and suggestions.

## How to contribute
- **Report bugs** via GitHub Issues (include OS, Python version, steps to reproduce, and screenshots/logs if possible).
- **Suggest enhancements** via GitHub Discussions or an issue tagged `enhancement`.
- **Submit pull requests** for focused bug fixes or documentation improvements.

## Code of conduct
By participating, you agree to abide by the project’s Code of Conduct (see `CODE_OF_CONDUCT.md`).

## Development setup
1. **Fork** the repository and create a feature branch  
   `git checkout -b fix/short-description`
2. **Create a virtual environment** (Python ≥ 3.10)  
   - Unix/macOS: `python -m venv .venv && source .venv/bin/activate`  
   - Windows: `.venv\Scripts\activate`
3. **Install dependencies**  
   `pip install -r requirements.txt`
4. **Run the app locally** (typical workflow)  
   `streamlit run streamlit_app.py`

### Verifying changes
- Run the automated checks locally: `pytest`.
- For UI or routing updates, reproduce the validation preset in Streamlit  
  *(Validation → “Validation (as cast)” → Dataset = Likely → Grid = BRA)* and confirm values match Table 1 / README.
- Optional: use `python steel_batch_cli.py run --scenario datasets/steel/likely/scenarios/BF_BOF_coal.yml --route BF-BOF --stage-key Cast --stage-role validation --country-code BRA` to capture deterministic JSON/CSV outputs outside the UI.

## Style & conventions
- Keep PRs **small and focused** (one change set per PR).
- Write clear commit messages (imperative mood, e.g., “Fix BF gas share rounding”).
- Prefer explicit names and docstrings over comments explaining “why”.
- If you touch YAML data files, include a brief note in the PR about the source/rationale.

## Pull request checklist
Before opening a PR, please ensure:
- [ ] `pytest` passes locally.
- [ ] The app runs locally without errors (`streamlit_app.py`).
- [ ] Validation preset still produces expected magnitudes (no regressions).
- [ ] Docs/README updated if behavior or options changed.
- [ ] Changelog entry added (if user-visible change).

## AI-assisted contributions
AI tools may be used for scaffolding code, but **you must**:
- Review and understand all generated code.
- Verify functionality via the Validation preset.
- Declare AI assistance briefly in the PR description (e.g., “Used LLM to draft parsing function; reviewed and tested manually”).

## Licensing
By contributing, you agree your contributions are released under the project’s MIT license.
