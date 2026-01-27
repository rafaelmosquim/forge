# Simple make targets to streamline long runs

PY ?= python3
ifneq ("$(wildcard .venv/bin/python)","")
PY := .venv/bin/python
endif

.PHONY: help list finished paper aluminum aluminum_fgv mc-as-cast mc-finished run parallel validation-table reference-aluminum reference-aluminum-primary reference-aluminum-finished reference-aluminum-finished-all

help:
	@echo "Targets:"
	@echo "  list            - list available run profiles"
	@echo "  finished        - run finished steel portfolio"
	@echo "  paper           - run paper portfolio"
	@echo "  aluminum        - run aluminum baseline scenarios via CLI"
	@echo "  aluminum_fgv    - run aluminum FGV portfolio (rolled/extruded/casted blend)"
	@echo "  reference-aluminum - generate aluminum primary finished/BRA reference tables"
	@echo "  reference-aluminum-primary  - generate aluminum primary/BRA reference tables"
	@echo "  reference-aluminum-finished - generate aluminum finished/BRA reference tables"
	@echo "  mc-as-cast      - Monte Carlo as-cast portfolio (via run_profiles)"
	@echo "  mc-finished     - Monte Carlo finished portfolio (via run_profiles)"
	@echo "  run PROFILE=... - run a named profile from configs/run_profiles.yml"
	@echo "  parallel        - run 'finished' and 'paper' in parallel"
	@echo "  docker-build    - build Docker image (tag: forge:paper)"
	@echo "  docker-finished - run 'finished' profile inside Docker"
	@echo "  docker-paper    - run 'paper' profile inside Docker"
	@echo "  engine-smoke    - quick engine CLI run (BF-BOF, Finished, 1000 kg)"
	@echo "  reproduce-validation - Likely/BRA Validation (as-cast) reproducible run"
	@echo "  validation-table - generate LaTeX table for Validation/BRA across all steel routes"

list:
	$(PY) scripts/run_profiles.py --list

finished:
	$(PY) scripts/run_profiles.py finished

paper:
	$(PY) scripts/run_profiles.py paper

aluminum:
	$(PY) scripts/run_profiles.py aluminum

aluminum_fgv:
	$(PY) scripts/run_profiles.py aluminum_fgv

run:
	@test -n "$(PROFILE)" || (echo "Set PROFILE=<name>" && exit 2)
	$(PY) scripts/run_profiles.py $(PROFILE)

parallel:
	$(PY) scripts/run_profiles.py finished paper --parallel

engine-smoke:
	$(PY) -m forge.cli.engine_cli --data datasets/steel/likely --route BF-BOF --stage Finished --country BRA --demand 1000 --out results/engine_demo

# Reproducible Validation (Likely dataset, Brazil, Validation as-cast)
reproduce-validation:
	PYTHONPATH=src $(PY) -m forge.cli.engine_cli \
	  --data datasets/steel/likely \
	  --route BF-BOF \
	  --stage Cast \
	  --country BRA \
	  --demand 1000 \
	  --out results/reproduce_validation

mc-as-cast:
	$(PY) scripts/run_profiles.py mc-as-cast

mc-finished:
	$(PY) scripts/run_profiles.py mc-finished

validation-table:
	PYTHONPATH=src $(PY) scripts/generate_validation_table.py --routes BF-BOF,DRI-EAF,EAF-Scrap --stage Cast

# Aluminum reference tables (BRA, 1000 kg)
reference-aluminum:
	PYTHONPATH=src $(PY) scripts/generate_reference_tables.py \
	  --data datasets/aluminum/baseline \
	  --route Primary \
	  --stage finished \
	  --scenario primary-al.yml \
	  --country BRA \
	  --name aluminum_primary_finished_bra

reference-aluminum-primary:
	PYTHONPATH=src $(PY) scripts/generate_reference_tables.py \
	  --data datasets/aluminum/baseline \
	  --route Primary \
	  --stage primary \
	  --scenario primary-al.yml \
	  --country BRA \
	  --name aluminum_primary_bra

reference-aluminum-finished:
	PYTHONPATH=src $(PY) scripts/generate_reference_tables.py \
	  --data datasets/aluminum/baseline \
	  --route Primary \
	  --stage finished \
	  --scenario primary-al.yml \
	  --country BRA \
	  --name aluminum_finished_bra

reference-aluminum-finished-all: reference-aluminum-finished

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
