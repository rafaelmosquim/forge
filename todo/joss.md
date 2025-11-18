# JOSS documentation follow-up

Reviewer: “Needs documentation of some sort.”  
Interpretation: JOSS requires a specific checklist of documentation artifacts that must be visible from the repo root. The existing `README.md` already covers the statement of need and install/quick-start sections; what’s missing is a dedicated user guide/tutorial and an API reference pointer.

## Checklist to satisfy
1. **Statement of need** – already present in `README.md` intro (problem the software solves, who it’s for).
2. **Installation instructions** – covered in `README.md` → Install/Quick start/Makefile sections.
3. **Usage examples/tutorial** – add a short “Getting started” doc or notebook that walks through (a) Streamlit workflow, (b) CLI batch run, (c) interpreting outputs. Link it from the README.
4. **API documentation** – document the public Python surface (e.g., `forge.steel_core_api_v2.RouteConfig`, `ScenarioInputs`, `run_scenario`). At minimum, add an “API reference” section in the README pointing to docstrings or `docs/api.md`.

## Action items
- [ ] Create `docs/getting_started.md` (or similar) with step-by-step UI + CLI example runs and screenshots/output snippets.
- [ ] Add a short section in `README.md` that links to the new guide (“Need a walkthrough? See docs/getting_started.md”) so reviewers notice it immediately.
- [ ] Draft `docs/api.md` (or README section) outlining the key public classes/functions and their parameters. Include references to inline docstrings for deeper detail.
- [ ] Optional but helpful: include `docs/reviewer_expected_summary.json` + “Reviewer quick start” snippet so the one-command repro path is discoverable.

Once these are in place, respond to the reviewer noting the locations (README lines + docs paths) that satisfy the JOSS checklist.
