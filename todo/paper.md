# Paper polish todo

Reviewer feedback indicates `paper.md` felt rushed; below is a checklist aligned with the JOSS guidelines (Summary, Statement of need, State of the field, Quality, Reproducibility).

## Summary (`paper.md:22-32`)
- Expand beyond two sentences. Describe FORGE’s novel capabilities, scope (steel + aluminum), and how the Python port advances reproducibility versus the Excel baseline.
- Mention key outputs (energy/GHG balances, sensitivity sweeps, Monte Carlo, UI/CLI) and provide a sentence on anticipated user communities.

## Statement of need (`paper.md:34-56`)
- Introduce the gap in currently published/open tools (cite at least 1–2 comparative works instead of generic categories).
- Clarify the scientific questions FORGE enables (e.g., decarbonization scenario analysis, impact of route locking, electricity crediting).
- Tighten bullet list: emphasize differentiators (YAML transparency, recipe graph, deterministic validations) and remove marketing-style language.

## State of the field / Related work
- Add a dedicated paragraph referencing existing steel LCA tools or datasets (e.g., Worldsteel, Argonne GREET, commercial LCAs) and explain how FORGE differs.
- Cite literature on BF-BOF/DRI modeling or hybrid approaches to show awareness of prior art.

## Implementation & Architecture (`paper.md:58-79`)
- Currently just bullet points; add a narrative describing the data layout (`datasets/<sector>/<variant>`), the refactored engine, recipe graph, and how Streamlit interacts with the API.
- Describe how scenario locking, energy matrix, and Monte Carlo are implemented technically (modules/functions, determinism controls).
- Mention dependencies (numpy, pandas, etc.) and how calculations are validated (unit tests + regression data).

## Validation (`paper.md:81-129`)
- Complement Table 1 with uncertainty ranges or sensitivity discussion; explain why FORGE deviates from Worldsteel for DRI/EAF beyond “Brazil grid”.
- Include at least one additional validation (e.g., finished-stage totals, energy balance check, comparison with excel baseline) to demonstrate robustness.
- Provide explicit commands / configs that reproduce the validation (link to `configs/*` or `Makefile` targets).

## Use cases / Impact
- Add a section highlighting real analyses run with FORGE (paper scenarios, policy comparisons, aluminum extension) and what insights were obtained.

## Reproducibility & Installation (`paper.md:137-146`)
- Reference the reviewer workflow (`make reviewer` or CLI commands), tests (`pytest`), and deterministic seeds used in Monte Carlo.
- Move the raw installation snippet to Supplementary or expand it to mention Python version, optional Docker, etc.

## AI assistance disclosure (`paper.md:132-136`)
- Strengthen the wording to clarify which portions were AI-assisted vs. human authored, matching the README disclosure.

## Metadata
- Update `date:` to the intended submission date, ensure author affiliations follow JOSS template, and confirm ORCIDs.

## General edits
- Proofread for grammar (“analysies” typo) and consistent capitalization (BF-BOF vs BF–BOF).
- Ensure references cited in new sections exist in `paper.bib`.
