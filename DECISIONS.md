
### 4. `DECISIONS.md`

This is your project's "don't change this unless we explicitly decide otherwise" file.

```markdown
# CONFLUX — Locked Decisions

## Dataset

V4 dataset is frozen.

Do not regenerate or modify:

`data/raw/dataset_v4_final.csv`

Ground truth:

- label
- campaign_id

Forbidden feature inputs:

- label
- campaign_id
- _source_type

## Project Objective

Detect coordinated cross-merchant card-testing campaigns.

Not ordinary single-transaction fraud classification.

## Behavioral Signals

Six groups are locked:

1. Amount
2. Device
3. BIN
4. Merchant Pattern
5. Velocity
6. Decline Ratio

## Temporal Design

Features must be causal.

Future information must never influence current features.

## BIN

BIN is categorical issuer context.

Do not treat it as a continuous numerical quantity.

Do not derive BIN from card_fingerprint.

## Graph

Graph is a separate layer from behavioral features.

Do not implement graph construction inside feature modules.

## Campaign Ground Truth

campaign_id is evaluation ground truth.

It must not be used to discover campaigns.

## ML

Current behavioral baseline:

Logistic Regression.

Do not replace it with a more complex model without an explicit decision.

## Evaluation

Accuracy alone is not an appropriate primary metric for the imbalanced dataset.

Use appropriate metrics such as:

- precision
- recall
- F1
- false-positive rate
- PR-oriented evaluation where appropriate

ROC-AUC can be used as an audit metric.

## Feature Design

Do not optimize the feature layer purely for AUC.

The goal is complementary and interpretable signals.

## Validation

An AI claim that "tests passed" is not evidence.

The test must actually be executed.

If not executed:

NOT EXECUTED

If not verified:

NOT VERIFIED

## Change Policy

Locked decisions may only be changed deliberately.

Any AI agent proposing a change to a locked decision must explain:

1. why the current decision is insufficient
2. what would change
3. what downstream components are affected
4. why the change is necessary