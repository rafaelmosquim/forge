# Simple make targets to streamline long runs

PY ?= python3

.PHONY: help list finished paper run parallel fgv mc-as-cast mc-finished

help:
	@echo "Targets:"
	@echo "  list            - list available run profiles"
	@echo "  finished        - run finished steel portfolio"
	@echo "  paper           - run paper portfolio"
	@echo "  run PROFILE=... - run a named profile from configs/run_profiles.yml"
	@echo "  parallel        - run 'finished' and 'paper' in parallel"
	@echo "  fgv             - run FGV regular portfolio (3 BR grid mixes, parallel)"
	@echo "  mc-as-cast      - example Monte Carlo (as-cast, ALL countries)"
	@echo "  mc-finished     - example Monte Carlo (finished portfolio, ALL countries)"
	@echo "  docker-build    - build Docker image (tag: forge:paper)"
	@echo "  docker-finished - run 'finished' profile inside Docker"
	@echo "  docker-paper    - run 'paper' profile inside Docker"
	@echo "  engine-smoke    - quick engine CLI run (BF-BOF, Finished, 1000 kg)"

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

fgv:
	$(PY) scripts/run_profiles.py \
		fgv_regular_br fgv_regular_br_low fgv_regular_br_high \
		fgv_high_br fgv_high_br_low fgv_high_br_high \
		--parallel

engine-smoke:
	$(PY) -m forge.cli.engine_cli --data datasets/steel/likely --route BF-BOF --stage Finished --country BRA --demand 1000 --lci --out results/engine_demo

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

# --- Docker helpers ---
docker-build:
	docker build -t forge:paper .

docker-finished: docker-build
	docker run --rm \
	  -v "$(PWD)/results:/app/results" \
	  -v "$(PWD)/configs:/app/configs" \
	  -e FORGE_PAPER_PRODUCT_CONFIG=portfolio \
	  -e FORGE_PAPER_PORTFOLIO_SPEC=configs/finished_steel_portfolio.yml \
	  -e FORGE_PAPER_PORTFOLIO_BLEND=finished_portfolio \
	  -e FORGE_PAPER_CACHE_DIR=results/cache_finished \
	  -e FORGE_OUTPUT_LABEL=finished \
	  forge:paper python3 scripts/run_profiles.py finished

docker-paper: docker-build
	docker run --rm \
	  -v "$(PWD)/results:/app/results" \
	  -v "$(PWD)/configs:/app/configs" \
	  -e FORGE_PAPER_PRODUCT_CONFIG=portfolio \
	  -e FORGE_PAPER_PORTFOLIO_SPEC=configs/paper_portfolio.yml \
	  -e FORGE_PAPER_PORTFOLIO_BLEND=paper_portfolio \
	  -e FORGE_PAPER_CACHE_DIR=results/cache_paper \
	  -e FORGE_OUTPUT_LABEL=paper \
	  forge:paper python3 scripts/run_profiles.py paper
