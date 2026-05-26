# KoPAS-Agent

Code and data for:

**Same Intent, Different Actions: Request Framing as a Safety Variable in LLM Agents**

KoPAS-Agent is an inference-only evaluation suite for studying whether Korean
pragmatic request forms change the safety of tool-using LLM agents.

The benchmark compares the same intent family across Korean request forms in
two settings:

- `chat_only`: the model answers without tools.
- `agent_tool_use`: the model can call sandboxed tools, and safety is scored
  from tool calls and environment-state changes.

No training, fine-tuning, model weights, real user data, or real credentials are
included.

## Artifact Contents

```text
datasets/
  intent_seeds_reviewed.jsonl
  kopas_pragmatic_main.jsonl
  pragmatic_templates.yaml

src/kopas_agent/
  dataset_tools/
  eval_runner/
  environments/
  scorers/
  tools/

analysis/
  compute_metrics.py
  format_results.py

configs/
  main_openrouter_*.yaml
  dummy_model_configs.yaml
```

The artifact contains:

- Korean-localized harmful and benign intent-family prompts.
- Synthetic tool-use environments with no real accounts, files, or services.
- Deterministic rule-based scoring over tool arguments and environment states.
- Evaluation runners for no-tool and tool-use conditions.
- Analysis scripts for aggregating metrics and reproducing paper tables.

The repository intentionally excludes:

- `.env` and all local API keys
- `.venv/`, caches, and build artifacts
- `runs/` and `reports/`
- raw upstream AgentHarm source snapshots
- auto-draft intermediate datasets used during development

## Dataset

The main dataset is:

```text
datasets/kopas_pragmatic_main.jsonl
```

It is generated from reviewed intent seeds and Korean pragmatic templates:

```text
datasets/intent_seeds_reviewed.jsonl
datasets/pragmatic_templates.yaml
        ↓
datasets/kopas_pragmatic_main.jsonl
```

Current main-dataset scale:

```text
52 intent families × 2 risk labels × 6 Korean pragmatic forms
= 624 prompt instances
```

Each intent family contains a harmful request and a benign counterpart. Both
are rendered into the same six Korean pragmatic forms:

- `neutral`
- `honorific_polite`
- `indirect_request`
- `authority_claim`
- `urgency_pressure`
- `responsibility_transfer`

The reviewed seeds are content-preserving Korean localized rewrites derived
from AgentHarm intent structures. The raw AgentHarm source files are not
redistributed in this repository.

Regenerate and validate the dataset:

```bash
make generate-main-data
make validate-main
```

`validate-main` checks that all rows are human-reviewed, that every intent
family has the required forms, and that neutral prompts are not duplicated
across families.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
make test
```

Run a dummy smoke evaluation:

```bash
make run-dummy RUN_ARGS="--limit 12"
make analyze-dummy
```

Dummy outputs are written under ignored local directories:

```text
runs/
reports/
```

## OpenRouter Runs

Copy the example environment file and add your own key locally:

```bash
cp .env.example .env
```

Never commit `.env`.

Check OpenRouter setup:

```bash
make check-openrouter
```

Run a small smoke test:

```bash
make run-openrouter-main RUN_ARGS="--limit 24 --resume"
make analyze-openrouter
make format-openrouter
```

Run a full main evaluation with the default Qwen config:

```bash
make run-openrouter-main RUN_ARGS="--runs 3 --resume"
make analyze-openrouter
make format-openrouter
```

To switch models, pass another config:

```bash
make run-openrouter-main \
  OPENROUTER_MODEL_CONFIG=configs/main_openrouter_qwen3_8b.yaml \
  OPENROUTER_RUN_OUTPUT=runs/main_qwen3_8b_results.jsonl \
  RUN_ARGS="--runs 3 --resume"
```

## Metrics

The analysis scripts report:

- unsafe response rate in the no-tool condition
- unsafe tool invocation rate
- unsafe state violation rate
- response-behavior mismatch rate
- benign tool-use success rate
- form-level and model-level pragmatic sensitivity deltas

Behavioral safety is scored from sandboxed tool trajectories and state diffs,
not from free-form model text alone.

## License and Terms of Use

The code in this repository is released under the license specified in
`LICENSE`.

The dataset, Korean-localized prompts, synthetic tool environments, evaluation
scripts, and analysis artifacts are provided for research review and research
use only. See `TERMS_OF_USE.md` for details.

The artifacts are intended for research on LLM agent safety evaluation. They
should not be used to facilitate harmful real-world tool use, unauthorized data
access, privacy invasion, or deployment against real users, files, accounts, or
external services.

Citation metadata is provided in `CITATION.cff`. Author-identifying metadata can
be updated after anonymous review.

## Sharing Safety

Before archiving or sharing the repository, run:

```bash
make archive-clean
```

The archive target excludes secrets, virtual environments, Git metadata,
caches, run outputs, reports, and local environment files.
