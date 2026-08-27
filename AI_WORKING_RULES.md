# CONFLUX — AI Working Rules

## 1. Source of Truth

Before doing ANY work:

1. Read PROJECT_CONTEXT.md
2. Read ARCHITECTURE.md
3. Read FEATURE_SPEC.md
4. Read DECISIONS.md
5. Read this file
6. Inspect the actual repository/code relevant to the task
7. Inspect the actual dataset when the task involves data

Do not rely on previous AI conversations.

## 2. No Hallucination

Never invent:

- dataset characteristics
- feature distributions
- campaign behaviour
- model performance
- test results
- graph results
- deployment behaviour
- hackathon requirements

If unknown:

`UNKNOWN`

If not verified:

`NOT VERIFIED`

If not executed:

`NOT EXECUTED`

## 3. Inspect Before Editing

Never immediately rewrite code.

First inspect:

- existing implementation
- configuration
- interfaces
- tests
- downstream dependencies

Understand how the current code works before modifying it.

## 4. Frozen Dataset

Never modify or regenerate:

`data/raw/dataset_v4_final.csv`

Never modify the validation file unless the task explicitly concerns validation metadata.

## 5. Ground Truth Leakage

Never use:

- label
- campaign_id
- _source_type

as feature/model inputs.

Ground truth may be used for evaluation only.

## 6. Temporal Leakage

Never use future information.

Every causal feature must be computed using information available at the relevant transaction time.

Never use naive full-day aggregations that expose future transactions.

## 7. Architecture Boundaries

Do not:

- put graph logic in feature modules
- put campaign discovery in ML
- put campaign logic in the API
- put fraud logic in the dashboard
- calculate campaign features before candidate campaigns exist

Respect layer boundaries.

## 8. Feature Discipline

Do not add features merely because:

- they sound sophisticated
- they improve AUC
- another AI suggested them
- the feature count looks too small

Every feature requires a clear purpose.

## 9. Configuration

Thresholds and window sizes belong in configuration.

Do not hide important thresholds inside feature functions.

## 10. Validation

Never claim a test passed unless it was actually executed.

After meaningful changes, run the relevant tests.

Check:

- row count
- transaction_id alignment
- forbidden columns
- NaN
- Inf
- causality
- feature contract
- downstream compatibility

## 11. Existing Code Is Not Automatically Correct

Do not assume existing code is correct simply because it runs.

Do not assume existing code is wrong simply because it looks different from generic examples.

Compare it against:

- project specification
- locked decisions
- actual dataset
- tests

## 12. Do Not Over-Engineer

Do not introduce:

- unnecessary neural networks
- GNNs
- LLM-based fraud scoring
- random algorithms
- unnecessary dependencies

unless explicitly requested and justified.

CONFLUX's value is the detection architecture, not maximum algorithmic complexity.

## 13. Change Reporting

After making changes, report:

- files changed
- what changed
- why it changed
- tests executed
- test results
- remaining concerns

Do not hide modifications.

## 14. Scope Control

Only modify files relevant to the requested task.

Do not silently redesign unrelated modules.

If another component must change for integration reasons, explain why first.

## 15. AI Agents Are Not the Authority

AI output is a proposal.

The actual repository, tests, validation results, and locked project decisions determine correctness.

Do not treat another AI's previous answer as authoritative.

## 16. Final Principle

CORRECTNESS
>
CAUSALITY
>
VALIDATION
>
INTERPRETABILITY
>
COMPLEXITY

The goal is a technically defensible CONFLUX system, not an impressive-looking collection of AI-generated code.