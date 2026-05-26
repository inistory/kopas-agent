# Terms of Use

These terms apply to the dataset, Korean-localized prompts, synthetic tool
environments, evaluation scripts, scoring code, configuration files, and
analysis artifacts in this repository.

## Intended Use

The artifacts are provided for research review and research use only. They are
intended to support controlled research on LLM agent safety evaluation,
including response-level safety, tool-use behavior, request framing, and
environment-state based scoring.

## Safety Scope

The dataset contains harmful requests only for controlled safety evaluation.
These requests are included so that researchers can measure whether language
models and tool-using agents refuse, redirect, or avoid unsafe behavior under
simulated conditions.

All tool environments are simulated. No real accounts, files, users,
institutions, credentials, external services, or production systems are
accessed by the included tools.

## Prohibited Use

The artifacts must not be used to:

- facilitate harmful real-world tool use;
- perform or assist unauthorized data access;
- invade privacy, expose personal data, or target real people;
- generate, deploy, or operationalize harmful requests against real services;
- bypass approval, audit, authorization, human-review, or safety controls in
  external systems;
- train or tune systems for harmful compliance, evasion, or abuse.

## Redistribution

Redistributions of the artifacts should preserve this terms file, the license
file, and the safety notices in the README. If modified versions are shared,
they should clearly indicate what was changed.

## Upstream Data

The public repository contains Korean localized intent-family data derived from
AgentHarm-style task structures. Raw upstream AgentHarm source snapshots are
not redistributed here. Users who need upstream provenance should obtain the
upstream dataset from its original release and follow its applicable license
and terms.

## No Warranty

The artifacts are provided as-is for research purposes, without any warranty.
Users are responsible for ensuring that their use complies with applicable
laws, institutional review requirements, platform policies, and safety
standards.
