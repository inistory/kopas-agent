PYTHONPATH := src
PYTHON ?= python3

MAIN_SEEDS ?= datasets/intent_seeds_reviewed.jsonl
MAIN_DATASET ?= datasets/kopas_pragmatic_main.jsonl
DATASET ?= $(MAIN_DATASET)

DUMMY_MODEL_CONFIG ?= configs/dummy_model_configs.yaml
OPENROUTER_MODEL_CONFIG ?= configs/main_openrouter_qwen3_all.yaml
OPENROUTER_RUN_OUTPUT ?= runs/main_openrouter_results.jsonl
OPENROUTER_METRICS_OUTPUT ?= reports/main_openrouter_metrics.csv
OPENROUTER_READABLE_PREFIX ?= reports/main_openrouter

RUN_ARGS ?=
METRICS_ARGS ?=
CLEAN_ARCHIVE ?= kopas-agent-clean.tar.gz

export PYTHONPATH
export XDG_CACHE_HOME := $(CURDIR)/.cache

.PHONY: install install-api install-local build-reviewed-seeds generate-main-data validate validate-main test lint check-openrouter run-dummy run-openrouter run-openrouter-main analyze-dummy analyze-openrouter format-openrouter clean-results archive-clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-api:
	$(PYTHON) -m pip install -e ".[dev,llm]"

install-local:
	$(PYTHON) -m pip install -e ".[dev,local]"

build-reviewed-seeds:
	$(PYTHON) -m kopas_agent.dataset_tools.build_reviewed_intent_seeds

generate-main-data:
	$(PYTHON) -m kopas_agent.dataset_tools.generate_pragmatic_dataset \
		--seeds $(MAIN_SEEDS) \
		--templates datasets/pragmatic_templates.yaml \
		--output $(MAIN_DATASET) \
		--main-only

validate:
	$(PYTHON) -m kopas_agent.dataset_tools.validate_dataset $(DATASET)

validate-main:
	$(PYTHON) -m kopas_agent.dataset_tools.validate_dataset $(MAIN_DATASET) \
		--require-human-reviewed \
		--require-main-forms \
		--max-duplicate-neutral-prompt 1

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests analysis

check-openrouter:
	$(PYTHON) -m kopas_agent.eval_runner.check_real_model_setup \
		--model-config $(OPENROUTER_MODEL_CONFIG)

run-dummy:
	$(PYTHON) -m kopas_agent.eval_runner.run_eval \
		--dataset $(DATASET) \
		--model-config $(DUMMY_MODEL_CONFIG) \
		--output runs/pragmatic_dummy_results.jsonl \
		$(RUN_ARGS)

run-openrouter:
	$(MAKE) run-openrouter-main

run-openrouter-main:
	$(PYTHON) -m kopas_agent.eval_runner.run_eval \
		--dataset $(MAIN_DATASET) \
		--model-config $(OPENROUTER_MODEL_CONFIG) \
		--output $(OPENROUTER_RUN_OUTPUT) \
		$(RUN_ARGS)

analyze-dummy:
	$(PYTHON) analysis/compute_metrics.py \
		--input runs/pragmatic_dummy_results.jsonl \
		--output reports/pragmatic_dummy_metrics.csv \
		--model-config $(DUMMY_MODEL_CONFIG) \
		$(METRICS_ARGS)

analyze-openrouter:
	$(PYTHON) analysis/compute_metrics.py \
		--input $(OPENROUTER_RUN_OUTPUT) \
		--output $(OPENROUTER_METRICS_OUTPUT) \
		--model-config $(OPENROUTER_MODEL_CONFIG) \
		$(METRICS_ARGS)

format-openrouter:
	$(PYTHON) analysis/format_results.py \
		--input $(OPENROUTER_RUN_OUTPUT) \
		--dataset $(MAIN_DATASET) \
		--csv-output $(OPENROUTER_READABLE_PREFIX)_readable.csv \
		--markdown-output $(OPENROUTER_READABLE_PREFIX)_readable.md

clean-results:
	rm -rf runs reports

archive-clean:
	tar --exclude='.env' \
		--exclude='.env.local' \
		--exclude='.env.*.local' \
		--exclude='.venv' \
		--exclude='.git' \
		--exclude='.pytest_cache' \
		--exclude='.ruff_cache' \
		--exclude='.mypy_cache' \
		--exclude='.cache' \
		--exclude='._*' \
		--exclude='runs' \
		--exclude='reports' \
		--exclude='datasets/source' \
		-czf $(CLEAN_ARCHIVE) .
