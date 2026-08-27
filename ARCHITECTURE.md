# CONFLUX — Architecture

## Pipeline

```text
Raw Transactions
       ↓
Ingestion
       ↓
Behavioral Feature Engineering
       ↓
Behavioral ML / Risk
       ↓
Entity + Temporal Graph
       ↓
Graph Metrics
       ↓
Campaign Detection
       ↓
Campaign Risk Scoring
       ↓
Explanation
       ↓
API
       ↓
Dashboard

1. Ingestion

Location:

src/conflux/ingestion/

Responsibility:

load the frozen transaction dataset
validate schema
parse timestamps
preserve transaction identity
provide clean input to downstream layers

It must not modify the frozen raw dataset.

2. Feature Engineering

Location:

src/conflux/features/

Responsibility:

Convert raw transactions into causal behavioral signals.

Modules:

amount_features.py
device_features.py
bin_features.py
merchant_features.py
velocity_features.py
decline_features.py
build_feature_table.py

The feature layer does NOT detect campaigns.

3. Behavioral ML

Location:

src/conflux/models/

Current intended baseline:

Logistic Regression.

Purpose:

Estimate transaction-level behavioral risk from engineered features.

The ML model is not the entire CONFLUX solution.

4. Graph

Location:

src/conflux/graph/

The graph represents relationships among relevant entities such as:

transactions
cards
devices
IP/network identifiers
BINs
merchants

The graph captures relationships that transaction-level features cannot fully represent.

The exact graph algorithm must be determined from the current implementation/specification and validated. Do not invent a GNN or graph algorithm merely to make the project sound more advanced.

5. Campaign Detection

Location:

src/conflux/graph/campaign_detection.py

Purpose:

Identify connected/related suspicious transaction/entity structures that may represent coordinated campaigns.

campaign_id from the dataset is ground truth for evaluation only. It must not be used to discover campaigns.

6. Campaign Risk

Location:

src/conflux/scoring/

Purpose:

Combine behavioral and graph evidence into campaign-level risk.

The score should remain explainable.

7. Explanation

Location:

src/conflux/scoring/explain.py

Provide human-readable reasons for a campaign risk decision.

Explanations must use actual computed values and must not invent numbers.

8. API

Location:

src/conflux/api/

Purpose:

Expose the detection pipeline to the frontend/live-feed simulation.

The API should orchestrate existing components rather than contain fraud-detection logic itself.

9. Dashboard

The dashboard visualizes:

transactions
behavioral risk
suspicious entities
graph/campaign relationships
campaign risk
explanations

The dashboard should consume backend results rather than independently implementing detection logic.

Layer Boundaries

Feature layer:
behavioral evidence.

ML:
behavioral transaction-level risk.

Graph:
entity and temporal relationships.

Campaign detection:
candidate coordinated campaigns.

Scoring:
campaign-level risk.

Explanation:
human-readable reasoning.

API:
serving/orchestration.

Dashboard:
visualization.

Do not silently move responsibilities between layers.