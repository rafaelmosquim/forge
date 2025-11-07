# Simple make targets to streamline long runs

PY ?= python3

.PHONY: help list finished paper run parallel mc-as-cast mc-finished

help:
	@echo "Targets:"
	@echo "  list            - list available run profiles"
	@echo "  finished        - run finished steel portfolio"
	@echo "  paper           - run paper portfolio"
	@echo "  run PROFILE=... - run a named profile from configs/run_profiles.yml"
	@echo "  parallel        - run 'finished' and 'paper' in parallel"
	@echo "  mc-as-cast      - example Monte Carlo (as-cast, ALL countries)"
	@echo "  mc-finished     - example Monte Carlo (finished portfolio, ALL countries)"

list:
	$(PY) scripts/run_profiles.py --list

finished:
	$(PY) scripts/run_profiles.py finished

paper:
	$(PY) scripts/run_profiles.py paper

run:
	@test -n "$(PROFILE)" || (echo "Set PROFILE=<name>" && exit 2)
	$(PY) scripts/run_profiles.py $(PROFILE)

parallel:
	$(PY) scripts/run_profiles.py finished paper --parallel

# --- Monte Carlo examples (edit or copy as needed) ---
mc-as-cast:
	$(PY) -m forge.scenarios.monte_carlo_tri \
		--min datasets/steel/optimistic_low \
		--mode datasets/steel/likely \
		--max datasets/steel/pessimistic_high \
		--base datasets/steel/likely \
		--route BF-BOF \
		--portfolio configs/as_cast_portfolio.yml \
		--countries ALL \
		--n 500 \
		--out results/mc_as_cast

mc-finished:
	$(PY) -m forge.scenarios.monte_carlo_tri \
		--min datasets/steel/optimistic_low \
		--mode datasets/steel/likely \
		--max datasets/steel/pessimistic_high \
		--base datasets/steel/likely \
		--route BF-BOF \
		--portfolio configs/finished_steel_portfolio.yml \
		--countries ALL \
		--n 500 \
		--out results/mc_finished

