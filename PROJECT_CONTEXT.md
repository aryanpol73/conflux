# CONFLUX — Project Context

## What is CONFLUX?

CONFLUX is a fraud-detection system designed to detect coordinated cross-merchant card-testing campaigns.

The problem is NOT simply:

"Is this individual transaction fraudulent?"

The problem is:

"Are many individually ambiguous transactions actually part of the same coordinated attack campaign?"

An attacker may distribute many small payment attempts across different merchants. Individual transactions may not look sufficiently suspicious in isolation. CONFLUX combines behavioral, temporal, entity and graph evidence to identify the coordinated pattern.

## Core Idea

Individual transaction
→ weak behavioral signals
→ temporal/entity relationships
→ graph structure
→ candidate campaign
→ campaign risk
→ explanation

The key differentiator is coordinated campaign detection rather than ordinary transaction-level fraud classification.

## Frozen Dataset

The current dataset is:

`data/raw/dataset_v4_final.csv`

Validation metadata:

`data/raw/dataset_v4_validation.json`

The V4 dataset is FROZEN.

It must not be regenerated, modified, rebalanced, or altered unless explicitly approved.

Known characteristics:

- ~31,873 transactions
- ~6.4% attack prevalence
- 45 synthetic attack campaigns
- 400 merchants
- 24-hour simulated period

Actual transaction fields:

- transaction_id
- timestamp
- merchant_id
- card_fingerprint
- bin
- amount
- device_fingerprint
- ip_signature
- auth_outcome
- label
- campaign_id

## Ground Truth

`label` and `campaign_id` are ground truth.

They are allowed for:

- evaluation
- validation
- auditing
- measuring campaign recovery

They are NEVER allowed as feature/model inputs.

The debug dataset may contain `_source_type`.

`_source_type` is audit/debug information only and must never enter production features or models.

## Core Architecture

RAW TRANSACTIONS
→ BEHAVIORAL FEATURES
→ BEHAVIORAL ML/RISK
→ ENTITY + TEMPORAL GRAPH
→ CAMPAIGN DETECTION
→ CAMPAIGN RISK SCORING
→ EXPLANATION
→ API/DASHBOARD

Each layer has a separate responsibility.

## Six Behavioral Signal Groups

The behavioral layer contains exactly:

1. Amount
2. Device
3. BIN
4. Merchant Pattern
5. Velocity
6. Decline Ratio

These signals are complementary.

A single signal such as:

- shared device
- high velocity
- low amount
- high decline rate
- BIN reuse

must not automatically mean fraud.

The system is intended to combine multiple weak signals with temporal/entity relationships.

## Important Design Principle

CONFLUX should not become:

"amount < X → fraud"

or:

"device reused → fraud"

or:

"decline rate > X → fraud"

The objective is coordinated behavioral detection.

## Current Development Philosophy

Correctness > complexity.

Causality > artificially high metrics.

Interpretability > unnecessary sophistication.

Actual validation > AI claims.

No AI is allowed to invent dataset facts, test results, architecture decisions, or performance numbers.