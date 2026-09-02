<div align="center">

# **CONFLUX**
### **Coordinated Structure Intelligence for Card-Fraud Campaigns**

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-bundler-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tests](https://img.shields.io/badge/tests-42%20passed-brightgreen)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Fraud rings don't look suspicious one transaction at a time. They look suspicious as a shape.**

</div>

---

## **Table of Contents**

- [**What CONFLUX Does**](#-what-conflux-does)
- [**Screenshots**](#-screenshots)
- [**How It Works**](#-how-it-works)
- [**The Deterministic Scorer**](#-the-deterministic-scorer)
- [**Explainability Layer**](#-explainability-layer)
- [**Build Phases**](#-build-phases)
- [**Architecture**](#-architecture)
- [**Folder Structure**](#-folder-structure)
- [**Quickstart**](#-quickstart)
- [**API Reference**](#-api-reference)
- [**Testing**](#-testing)
- [**Deployment**](#-deployment)
- [**Honest Limitations**](#-honest-limitations)
- [**Roadmap**](#-roadmap)
- [**Topics**](#-topics)

---

## **What CONFLUX Does**

Conventional fraud scoring asks *"is this transaction bad?"* — one row at a time. CONFLUX asks a different question: **"is this transaction part of a coordinated campaign?"**

It builds a **temporal entity graph** over the transaction stream, linking rows that share payment instruments, devices, IPs, or merchants. Connected sub-graphs become **candidates**. Each candidate is scored by a transparent, six-feature **percentile scorer**, assigned a risk tier, and shipped to the dashboard with a plain-English explanation of *why* it scored the way it did.

The core research question driving the whole build:

> Of **4,372** multi-transaction candidates discovered **without ever touching `campaign_id`**, how many actually correspond to genuinely coordinated campaigns?

Answer: **81 positives across 45 distinct campaigns** — a **1.85%** prevalence. That low base rate is the reason every design decision below leans toward campaign-grouped validation and away from anything that can silently memorise a ring.

---

## **Screenshots**

**Detection graph** — entity clusters, live transaction stream, and risk tiers

![Detection graph](docs/images/detection-graph.png)

**Investigation panel** — campaign structure, connected entities, contribution bars

![Investigation panel](docs/images/investigation-panel.png)

**Explainability panel** — decoded percentiles and natural-language verdict

![Explainability panel](docs/images/explainability-panel.png)

---

## **How It Works**

**1. Ingestion** — `POST /transactions` validates the payload against a strict schema. Ground-truth fields (`label`, `campaign_id`) are **rejected with HTTP 422**; the API is structurally incapable of accepting the answer key at inference time.

**2. Temporal entity graph** — transactions become nodes; shared cards, devices, IPs, and merchants become time-aware edges. Edges respect observation windows so the graph cannot see into the future.

**3. Candidate formation** — connected components of size ≥ 2 are extracted as candidates. No labels are consulted.

**4. Campaign scoring** — six decorrelated features are computed per candidate, mapped to percentiles against a frozen reference distribution, and averaged into a single score.

**5. Risk threshold + action** — the score maps to a tier and a recommended action: **flag for review**, **stop group**, or **block**.

**6. Evidence + explanation** — the response carries the linking entities, the contribution breakdown, and a human-readable explanation block.

---

## **The Deterministic Scorer**

Six features, **equal weight (1/6 each)**, deliberately chosen to be low-correlation with one another:

| Feature | What it captures |
|---|---|
| `burst_rate_per_minute` | Temporal compression — how tightly packed the transactions are |
| `link_density` | How densely interconnected the candidate's entity graph is |
| `max_transactions_per_shared_card` | Reuse intensity on the single hottest card |
| `multi_entity_link_fraction` | Share of links backed by more than one entity type |
| `distinct_merchants_per_transaction` | Merchant spread across the group |
| `distinct_bins_per_transaction` | BIN diversity across the group |

Each raw value is converted to a **percentile against a frozen reference distribution** (`models/artifacts/scorer_reference_v1.json`), then averaged. Because the weights are uniform, each feature's contribution to the final score is `percentile × 1/6` — which makes the score fully reconstructible after the fact.

**Why deterministic first?** A transparent scorer that an analyst can audit line-by-line is the baseline any ML challenger has to beat on *grouped, out-of-fold* data. Not on the training set. Not on a random split.

---

## **Explainability Layer**

`src/conflux/scoring/explain.py` inverts the weighting to recover the exact percentile behind every contribution, then turns it into a sentence.

For example, a contribution of `0.1613` on `burst_rate_per_minute` inverts to the **97th percentile** — and renders as:

> *"These 7 transactions landed in an unusually tight time window — tighter than 97% of comparable groups."*

Design rules the layer follows:

- **No hard-coded arithmetic.** It consumes the scorer's own `percentile_info`, so if the weighting ever changes the explanation follows automatically.
- **Percentile thresholds.** `STRONG_PERCENTILE = 0.90` marks a signal as genuinely unusual; `ORDINARY_PERCENTILE = 0.60` marks it as unremarkable. Anything below reads as "normal for this population."
- **Honest verdicts.** If nothing crosses the strong threshold, the verdict says so rather than manufacturing alarm.
- **Rank, not probability.** The panel states explicitly that "15th of 15" is a *ranking* position, not a 15/15 likelihood of fraud.
- **All six signals, always.** `top_n_signals` is set to the full feature count so the panel never silently truncates the story.

---

## **Build Phases**

Every phase below was gated on its own checklist before the next one started.

### **Phase 3 — Graph & Candidate Foundations** ✅

| Sub-phase | Scope | Status |
|---|---|---|
| **3A** | Temporal entity graph | ✅ 7/7 checks |
| **3B** | Candidate formation | ✅ 13/13 checks |
| **3C** | Candidate / campaign evaluation | ✅ |
| **3D** | Campaign scoring + ML integration | ✅ |

### **Phase 4A — Deterministic Campaign Scorer** ✅

- ✅ Small decorrelated feature set
- ✅ Transparent scoring
- ✅ Campaign-aware validation
- ✅ Temporal validation
- ✅ Threshold / recall trade-off
- ❌ **BIN / no-BIN ablation** — *outstanding*

### **Phase 4B — Robustness & Adversarial Testing** ✅

| Scenario | Perturbation applied |
|---|---|
| ✅ Unseen campaigns | Held-out campaign groups |
| ✅ Changed attack cadence | Scaled group cadence, jittered timestamps |
| ✅ More legitimate traffic | Benign volume injection |
| ✅ Weaker entity reuse | Reduced shared-entity overlap |
| ✅ Temporal boundary cases | Observation-window truncation, right-censoring |
| ✅ False-positive stress test | Benign bursts engineered to mimic campaigns |

### **Phase 4C — ML Comparison** ⬜ *not yet executed*

The pre-registered plan, written **before** any model was fit:

- Small **interpretable** model only — no deep model is warranted at n=45 campaigns
- Compared against the deterministic scorer under **repeated, campaign-grouped CV** (5 repeats × 5 folds)
- **Decision rule:** adopt the challenger only if mean PR-AUC gain **≥ 0.05** with a 95% CI excluding zero, *and* it beats a cheap-feature baseline
- Keep whichever arm **genuinely generalises** — the deterministic scorer stays unless the challenger clears the bar

Planned arms: `D` (deterministic), `X_core` (monotone-constrained XGBoost on the six core features), `X_free`, `X_cheap`, `X_noBIN`, `X_ext`, and `LR_agg` as a diagnostic-only arm.

### **Phase 5 — Final Detection Pipeline** ✅

- ✅ Candidate generation
- ✅ Campaign scoring (scorer reference loaded, 5/5 rules)
- ✅ Risk threshold
- ✅ Explanation / evidence
- ✅ Action: flag review / stop group / block

### **Phase 6 — Backend & API Integration** ✅

- ✅ Inference endpoint
- ✅ Transaction ingestion — `POST /transactions` → validate → store in memory
- ✅ Campaign state
- ✅ Scoring response — once the buffer fills, convert stored transactions, run the detection pipeline, return JSON
- ✅ Explanation payload

### **Phase 7 — Frontend & Demo Dashboard** ✅

- ✅ Transaction / campaign view
- ✅ Graph visualisation
- ✅ Risk score
- ✅ Connected entities
- ✅ Evidence
- ❌ **Attack timeline** — *outstanding*

### **Phase 8 — Final Evaluation & Demo Hardening** ⬜ *in progress*

- ⬜ Full end-to-end run
- ⬜ Latency measurement
- ⬜ Failure handling
- ⬜ Leakage audit
- ⬜ Reproducibility check
- ⬜ Demo dataset / scenarios

---

## **Architecture**

```
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND  (TypeScript + Vite, deployed on Vercel)           │
│                                                              │
│   main.ts ──> ui-controller.ts ──> graph.ts                  │
│      │             │                                         │
│      │             └──> explain-panel.ts  ("What this means")│
│      │                                                       │
│   data-loader.ts <── types.ts        api/rest-client.ts      │
│                                      api/websocket-client.ts │
└────────────────────────┬─────────────────────────────────────┘
                         │  REST  +  WebSocket
┌────────────────────────▼─────────────────────────────────────┐
│  BACKEND  (FastAPI, deployed on Render)                      │
│                                                              │
│   api/main.py       routes, validation, exception handlers   │
│   api/schemas.py    strict input contract (rejects labels)   │
│   api/state.py      in-memory store  ─>  run_detection()     │
│   api/websocket.py  live push to dashboard                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  DETECTION CORE                                              │
│                                                              │
│   pipeline.py                                                │
│      ├─> graph/temporal_graph.py     entity graph            │
│      ├─> graph/build_candidates.py   connected components    │
│      ├─> scoring/candidate_features.py   six features        │
│      ├─> scoring/deterministic_scorer.py percentile scoring  │
│      ├─> scoring/campaign_risk.py    tier + action           │
│      └─> scoring/explain.py          natural-language layer  │
│                                                              │
│   models/artifacts/scorer_reference_v1.json  frozen ref dist │
└──────────────────────────────────────────────────────────────┘
```

---

## **Folder Structure**

```
conflux/
│
├── data/
│   ├── raw/
│   │   ├── dataset_v4_final.csv
│   │   └── dataset_v4_validation.json
│   └── processed/
│       ├── evaluation/
│       │   └── phase3c_diagnostics/
│       │       ├── boolean_comparison.csv
│       │       ├── candidate_diagnostic_features.csv
│       │       ├── numeric_comparison.csv
│       │       ├── phase3c_diagnostic_report.json
│       │       ├── sensitivity_campaign_majority.csv
│       │       └── top_feature_redundancy_summary.csv
│       ├── graph/
│       │   ├── campaign_candidate_assignments.csv
│       │   ├── campaign_candidate_links.csv
│       │   ├── campaign_candidate_report.json
│       │   └── campaign_candidates.csv
│       ├── robustness/
│       │   └── baseline_rebuild.txt
│       ├── scoring/
│       │   ├── artifacts/
│       │   ├── candidate_scoring_features.csv
│       │   ├── core_feature_spearman.csv
│       │   ├── out_of_fold_scores.csv
│       │   ├── phase4a_scoring_report.json
│       │   └── recall_exchange_table.csv
│       ├── ablation_report.json
│       ├── baseline_metrics_report.json
│       ├── feature_dictionary.csv
│       ├── features_v4.csv
│       ├── univariate_auc.csv
│       └── validation_report.json
│
├── frontend/
│   ├── public/
│   ├── dist/                          # build output (Vercel)
│   ├── src/
│   │   ├── api/
│   │   │   ├── rest-client.ts
│   │   │   └── websocket-client.ts
│   │   ├── assets/
│   │   ├── data/
│   │   │   ├── data-loader.ts         # normalises API -> view models
│   │   │   ├── replay-source.ts
│   │   │   └── types.ts               # Campaign, CampaignExplanation, CandidateView
│   │   ├── graph/
│   │   │   ├── graph-builder.ts
│   │   │   ├── graph-interactions.ts
│   │   │   └── graph.ts
│   │   ├── story/
│   │   │   └── story-controller.ts
│   │   ├── styles/
│   │   │   ├── animations.css
│   │   │   ├── explain.css            # cfx- prefixed, additive only
│   │   │   ├── global.css
│   │   │   └── graph.css
│   │   ├── ui/
│   │   │   ├── explain-panel.ts       # renders the explanation block
│   │   │   └── ui-controller.ts
│   │   └── main.ts
│   ├── index.html
│   └── package.json
│
├── src/conflux/
│   ├── api/
│   │   ├── main.py                    # FastAPI app + routes
│   │   ├── schemas.py                 # input contract
│   │   ├── state.py                   # in-memory state + run_detection
│   │   └── websocket.py
│   ├── evaluation/
│   │   ├── ablation.py
│   │   ├── campaign_evaluation.py
│   │   ├── candidate_diagnostics.py
│   │   ├── leakage_audit.py
│   │   ├── metrics.py
│   │   ├── run_campaign_evaluation.py
│   │   └── run_candidate_diagnostics.py
│   ├── features/
│   │   ├── amount_features.py
│   │   ├── bin_features.py
│   │   ├── build_feature_table.py
│   │   ├── decline_features.py
│   │   ├── device_features.py
│   │   ├── merchant_features.py
│   │   └── velocity_features.py
│   ├── graph/
│   │   ├── build_candidates.py
│   │   ├── build_graph.py
│   │   ├── campaign_detection.py
│   │   ├── config.py
│   │   ├── graph_metrics.py
│   │   └── temporal_graph.py
│   ├── ingestion/
│   │   └── load_transactions.py
│   ├── models/
│   │   ├── artifacts/
│   │   │   ├── baseline_logreg_v4_report.json
│   │   │   ├── baseline_model.pkl
│   │   │   ├── scorer_reference_v1.json
│   │   │   └── scorer_reference_v1_meta.json
│   │   ├── predict.py
│   │   └── train_baseline.py
│   ├── robustness/
│   │   ├── perturbations.py
│   │   ├── rebuild.py
│   │   ├── scenario_cadence.py
│   │   ├── scenario_entity_reuse.py
│   │   ├── scenario_false_positive_stress.py
│   │   ├── scenario_legit_volume.py
│   │   ├── scenario_right_censoring.py
│   │   └── world.py
│   ├── scoring/
│   │   ├── campaign_risk.py
│   │   ├── candidate_features.py
│   │   ├── config.py
│   │   ├── deterministic_scorer.py    # six-feature percentile scorer
│   │   ├── evaluation.py
│   │   ├── explain.py                 # explainability layer
│   │   ├── run_scoring_evaluation.py
│   │   ├── scorer_reference_io.py
│   │   └── splits.py                  # campaign-grouped / temporal splits
│   ├── config.py
│   └── pipeline.py                    # Phase 5 end-to-end pipeline
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_campaign_candidates.py
│   ├── test_campaign_evaluation.py
│   ├── test_deterministic_scorer.py
│   ├── test_phase3b_integrity.py
│   ├── test_phase4b_perturbations.py
│   ├── test_phase4b_scenario_cadence.py
│   ├── test_phase4b_scenario_entity_reuse.py
│   ├── test_phase4b_scenario_false_positive.py
│   ├── test_phase4b_scenario_legit_volume.py
│   ├── test_phase4b_scenario_right_censoring.py
│   ├── test_phase4b_world.py
│   ├── test_phase5_pipeline.py
│   ├── test_scorer_reference.py
│   ├── test_scoring_features.py
│   ├── test_scoring_leakage.py
│   ├── test_scoring_splits.py
│   └── test_scoring_thresholds.py
│
├── tools/
│   ├── build_scorer_reference.py
│   ├── phase4a_dev_feature_selection.py
│   ├── phase4a_diagnostic.py
│   ├── phase4a_diagnostic_v2.py
│   └── phase4a_feature_cv.py
│
├── AI_WORKING_RULES.md
├── ARCHITECTURE.md
├── DECISIONS.md
├── FEATURE_SPEC.md
├── PROJECT_CONTEXT.md
├── README.md
├── render.yaml
├── requirements.txt
└── verification_output.txt
```

---

## **Quickstart**

**Prerequisites:** Python 3.14+, Node 18+

### **Backend**

```bash
git clone https://github.com/aryanpol73/conflux.git
cd conflux

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The package is imported from `src/`, so set the path before running:

```powershell
# Windows PowerShell
$env:PYTHONPATH = "$PWD\src"
```

```bash
# macOS / Linux
export PYTHONPATH="$PWD/src"
```

Then start the API:

```bash
uvicorn conflux.api.main:app --reload --port 8000
curl http://localhost:8000/health
```

Confirm the response shows `"scorer_loaded": true` — without the frozen reference distribution the scorer cannot produce percentiles.

### **Frontend**

```bash
cd frontend
npm install
npm run dev
```

---

## **API Reference**

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, `scorer_loaded`, `transactions_in_memory` |
| `GET` | `/campaigns` | Current campaign population with scores and explanations |
| `POST` | `/transactions` | Ingest one transaction, return the **population** result |
| `POST` | `/reset` | Clear in-memory state (demo control) |
| `WS` | `/ws` | Live push of detection updates to the dashboard |

**Important contract detail:** `POST /transactions` returns the *whole population* — `{"status", "summary", "campaigns"}` — not a single row score. A campaign only exists relative to everything else in the buffer, so a per-row response would be meaningless.

**Sample campaign payload:**

```json
{
  "candidate_id": "CAND-000216",
  "size": 7,
  "score": 0.4127,
  "tier": "elevated",
  "action": "flag_review",
  "evidence": {
    "shared_cards": 2,
    "shared_devices": 1,
    "shared_ips": 0
  },
  "explanation": {
    "headline": "7 transactions linked by shared payment entities",
    "verdict": "One signal stands out clearly.",
    "summary": "Grouped by 2 shared cards and 1 shared device.",
    "all_signals": [
      {
        "feature": "burst_rate_per_minute",
        "percentile": 0.97,
        "is_strong": true,
        "text": "Landed in an unusually tight time window — tighter than 97% of comparable groups."
      },
      {
        "feature": "max_transactions_per_shared_card",
        "percentile": 0.36,
        "is_strong": false,
        "text": "Card reuse is unremarkable for a group this size."
      },
      {
        "feature": "distinct_bins_per_transaction",
        "percentile": 0.02,
        "is_strong": false,
        "text": "BIN diversity is very low compared with similar groups."
      }
    ],
    "truncated": false,
    "rank_text": "Ranked 15th of 15 current candidates.",
    "method": "Six equal-weight percentile features.",
    "caveat": "This is a ranking position, not a probability of fraud."
  }
}
```

---

## **Testing**

```bash
# API contract, route registration, label rejection
pytest tests/test_api.py -q

# Scorer, reference distribution, thresholds
pytest tests/test_deterministic_scorer.py tests/test_scorer_reference.py tests/test_scoring_thresholds.py -q

# Leakage and split integrity
pytest tests/test_scoring_leakage.py tests/test_scoring_splits.py tests/test_phase3b_integrity.py -q

# Phase 4B robustness suite
pytest tests/test_phase4b_*.py -q

# Everything
pytest -q
```

Current status on all the `tests/`: **426 passed, 3 skipped**.

Two tests worth calling out because they encode design commitments rather than behaviour:

- `test_ground_truth_rejected_over_rest` — posting `label` or `campaign_id` must return **422**. This is the anti-leakage guardrail at the API boundary.
- `test_rest_ingest_returns_population_result_not_a_row_score` — pins the response shape to exactly `{status, summary, campaigns}`.

---

## **Deployment**

**Backend — Render** (`render.yaml`)

```bash
uvicorn conflux.api.main:app --host 0.0.0.0 --port $PORT
```

**Frontend — Vercel**

```bash
npm run build   # output: frontend/dist
```

**Deploy order matters.** The dashboard reads the `explanation` block from the API. Ship the backend first, wait for Render to go live, then push the frontend. Verify before promoting:

```bash
curl -s https://YOUR-RENDER-URL/campaigns \
  | python -c "import sys,json; d=json.load(sys.stdin); c=d.get('campaigns') or []; print('campaigns:', len(c)); print(json.dumps(c[0].get('explanation'), indent=2) if c else 'none')"
```

The frontend degrades gracefully if `explanation` is absent — the panel simply doesn't render — so a stale backend won't break the graph.

---

## **Honest Limitations**

These are stated up front because the alternative is someone discovering them in a demo.

- **The score is a rank, not a probability.** A 0.41 does not mean 41% likely fraud. It means this candidate sits at a particular position in the reference distribution.
- **Effective sample size is 45 campaigns, not 4,372 rows.** Positives cluster inside campaigns, so campaign-grouped CV is the only honest evaluation. Any row-level split would leak.
- **Full re-scoring on every ingest.** The population is recomputed from scratch per transaction. Correct, but O(n) per call — fine for a demo, not for production throughput.
- **State is process-local and in-memory.** A restart clears everything; there is no persistence layer yet.
- **Phase 4C has not been run.** The XGBoost comparison is pre-registered but unexecuted. No claim is made that ML beats the deterministic scorer, because that has not been tested.
- **BIN / no-BIN ablation is outstanding**, so the contribution of BIN-derived features to the score is not yet isolated.
- **Zero-weight features are omitted from explanations** by design — the layer only narrates features that actually moved the score.
- **Attack timeline view is not built**, so temporal sequencing inside a campaign is currently only visible as a burst-rate percentile.

---

## **Roadmap**

- Execute **Phase 4C** and publish the full leaderboard, learning curves, and CI intervals — including the negative result if the deterministic scorer wins
- Complete the **BIN / no-BIN ablation** to close out Phase 4A
- Build the **attack timeline** panel to close out Phase 7
- Finish **Phase 8**: end-to-end run, latency measurement, failure handling, leakage audit, reproducibility check, demo scenarios
- Persistence layer so state survives restarts
- Analyst feedback capture — record which flagged campaigns were confirmed, and feed that back into threshold tuning
- Optional TreeSHAP comparison against the deterministic explanation, as a sanity check on the percentile inversion

---

## **Topics**

`#fraud-detection` `#graph-analytics` `#entity-resolution` `#anomaly-detection`
`#explainable-ai` `#xai` `#interpretable-machine-learning` `#fastapi` `#python`
`#typescript` `#vite` `#data-visualization` `#websockets` `#rest-api`
`#payment-fraud` `#card-fraud` `#fraud-analytics` `#machine-learning`
`#xgboost` `#percentile-scoring` `#temporal-graph` `#campaign-detection`
`#adversarial-testing` `#data-leakage` `#cross-validation` `#fintech`
`#risk-scoring` `#real-time-analytics` `#dashboard` `#vercel` `#render`

<div align="center">

**Built with a bias toward transparency over accuracy claims.**

</div>
