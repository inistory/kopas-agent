# Datasets

The public data is synthetic and safe to redistribute. It contains no real
personal data, real secrets, real institutions, or actionable security details.

- `pragmatic_templates.yaml`: controlled Korean pragmatic-form templates.
- `intent_seeds_reviewed.jsonl`: human-reviewed AgentHarm-derived Korean
  localized intent-family seeds.
- `kopas_pragmatic_main.jsonl`: generated paper-main dataset. This file is
  paper-ready only after `make validate-main` passes.

Main evaluation scope:

- Included: 52 human-reviewed AgentHarm-derived intent families with harmful
  and benign counterparts.
- Final paper-main target in this public snapshot:
  `52 intent families × 2 risk labels × 6 Korean pragmatic forms = 624 prompt
  instances`.
- Raw AgentHarm source files and auto-draft intermediate datasets are not
  included in the public repository. Recreate them from the upstream AgentHarm
  release if provenance reconstruction is needed.

Each task has synthetic tool state, intent
family identifiers, pragmatic-form labels, safe behavior labels, and declared
unsafe tool conditions. Regenerate the files with:

```bash
make generate-main-data
make validate-main
```
