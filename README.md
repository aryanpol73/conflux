CONFLUX
Cross-Merchant Campaign Intelligence

CONFLUX is an explainable fraud detection and investigation system designed to detect coordinated card-testing campaigns across multiple merchants.

Built for the Razorpay AI Buildathon — Track 02: AI Risk Manager.

A single fraudulent transaction can look normal.
A coordinated pattern across cards, merchants, devices, IPs, and time usually does not.

The Story

Card-testing fraud rarely appears as one obviously bad transaction.

Instead, attackers may rapidly test stolen or generated card credentials using small transactions distributed across multiple merchants. Individual merchants may only see a few seemingly independent attempts.

Viewed in isolation, many of those transactions can appear legitimate.

The actual signal emerges only when the transactions are connected.

The same card may appear across different merchants. Multiple cards may reuse a device. Different merchants may receive requests associated with the same IP signature. Activity may suddenly accelerate within a short time window.

CONFLUX is designed to detect that coordinated pattern rather than simply classify one transaction as fraudulent.

The system combines:

Transaction-level behavioral signals
Temporal and cross-entity relationships
Campaign candidate discovery
Deterministic campaign risk scoring
Explainable evidence
A live investigation interface

The core principle is simple:

A transaction is not proof. A coordinated pattern across entities and time is evidence.

1. The Problem

Distributed card-testing attacks are difficult to detect because the attack is often fragmented across multiple merchants.

An attacker may:

Test many cards
Distribute attempts across multiple merchants
Reuse infrastructure such as devices or IPs
Operate within short bursts
Use small transaction amounts
Avoid making any individual merchant's traffic look obviously malicious

This creates a visibility problem.

A single merchant might observe:

Card A ──► Merchant 1
Card B ──► Merchant 1
Card C ──► Merchant 1

Another merchant might independently observe:

Card A ──► Merchant 2
Card D ──► Merchant 2
Card E ──► Merchant 2

Neither merchant necessarily sees the full campaign.

CONFLUX connects the observations:

                  Card A
                 /      \
                /        \
         Merchant 1    Merchant 2
              |            |
              └────┐  ┌────┘
                   │  │
                Device X
                   │
                 IP Y

The important question is therefore not simply:

"Is this transaction risky?"

It is:

"Are these apparently independent transactions actually part of the same coordinated campaign?"

2. What CONFLUX Does

CONFLUX progressively builds an investigation picture from incoming transactions.

Incoming Transactions
        │
        ▼
Transaction Processing
        │
        ▼
Entity & Temporal Relationships
        │
        ▼
Campaign Candidate Discovery
        │
        ▼
Campaign Feature Extraction
        │
        ▼
Campaign Risk Scoring
        │
        ├──────────────► Risk Score
        ├──────────────► Risk Tier
        ├──────────────► Recommended Action
        │
        ▼
Explainable Evidence
        │
        ▼
Live Investigation Console

The system:

Receives transactions
Maintains transaction and entity relationships
Identifies connected campaign candidates
Evaluates those candidates using campaign-level signals
Assigns a risk score and tier
Produces explainable evidence
Streams results to an investigation interface

CONFLUX is therefore not just a transaction fraud classifier.

It is a:

Coordinated Cross-Merchant Campaign Detection System

3. What Makes a Campaign Cross-Merchant?

Cross-merchant intelligence is central to CONFLUX.

During backend validation, transactions belonging to the same detected candidate were verified across different merchants.

Transaction A ──► Merchant M0137
Transaction B ──► Merchant M0148

Shared Card Fingerprint:
918040a152fbf88a

Result:

UNIQUE MERCHANTS: 2

CROSS-MERCHANT = YES

The transactions occurred at different merchants but shared an entity relationship that allowed the backend to connect them into a campaign candidate.

CONFLUX can connect transactions through entities such as:

Card fingerprint
BIN
Device fingerprint
IP signature
Merchant
Temporal proximity

A campaign becomes suspicious when multiple forms of evidence reinforce one another.

4. Dataset

CONFLUX uses a purpose-built synthetic transaction dataset.

The dataset was designed specifically for this project rather than relying on a trivially separable public dataset.

The design goal was to ensure that the detection problem required multi-signal reasoning.

Frozen Dataset
dataset_v4_final.csv
Dataset Characteristics
Property	Value
Total transactions	31,873
Attack rate	6.40%
Campaigns	45
Attacker archetypes	6
Legitimate hard negatives	Included
Identifier overlap between populations	Included
Independent IP field	Included

The dataset intentionally includes difficult cases where legitimate traffic can resemble suspicious activity.

Examples include:

Benign transaction bursts
Entity reuse
Overlapping identifiers
Shared BINs
Legitimate traffic patterns that should not automatically be treated as attacks

This prevents the system from relying on simplistic shortcuts such as:

"High BIN risk = campaign."

That is explicitly not the CONFLUX design.

5. Detection Architecture

CONFLUX operates in layers.

┌─────────────────────────────┐
│     Incoming Transactions   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Transaction-Level Processing│
│  Behavioral Feature Signals │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Temporal Cross-Entity Graph │
│                             │
│ Card · BIN · Device · IP    │
│ Merchant · Transaction      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Campaign Candidate Discovery│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Campaign Feature Extraction │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Deterministic Campaign      │
│ Risk Scorer                 │
└──────────────┬──────────────┘
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
    Score     Tier    Action
       │
       ▼
┌─────────────────────────────┐
│ Explainable Campaign Evidence│
└──────────────┬──────────────┘
               │
               ▼
        Investigation Console
6. Transaction-Level Behavioral Modeling

CONFLUX includes transaction-level behavioral feature engineering.

Feature groups include:

Amount behavior
Device behavior
BIN behavior
Merchant pattern behavior
Velocity
Decline behavior

Feature construction follows causal and temporal constraints so that future transactions are not used to explain earlier transactions.

A logistic regression baseline was developed as part of the modeling pipeline.

However, transaction-level ML output is not treated as sufficient evidence of a coordinated campaign.

Campaign detection requires relationship-level evidence.

7. Cross-Entity Temporal Graph

CONFLUX models transactions as part of a heterogeneous entity network.

The conceptual entity types include:

Transaction
Card
BIN
Merchant
Device
IP

Each transaction creates relationships between the entities involved.

                    Card
                     │
                     │
BIN ─────────── Transaction ───────── Merchant
                     │
              ┌──────┴──────┐
              │             │
            Device          IP

As additional transactions arrive, these relationships can reveal connected structures.

For example:

Card A ── Transaction 1 ── Merchant X
   │
   └────── Transaction 2 ── Merchant Y


Device D ── Transaction 1
   │
   └────── Transaction 3 ── Merchant Z

Individually, these transactions may appear unrelated.

Together, they can reveal:

Cross-merchant activity
Shared infrastructure
Entity reuse
Temporal coordination
8. Campaign Candidate Discovery

The graph layer identifies groups of transactions that may represent coordinated activity.

Candidate generation intentionally casts a reasonably broad net.

A candidate is not automatically a confirmed attack.

The pipeline separates discovery from risk assessment:

Connected Transactions
        │
        ▼
Campaign Candidate
        │
        ▼
Campaign Feature Extraction
        │
        ▼
Campaign Risk Scoring
        │
        ▼
Risk Tier + Evidence + Action

This separation is important.

Candidate Discovery answers:

What transactions might belong together?

Campaign Scoring answers:

How suspicious is that connected group?

9. Campaign Risk Scoring

CONFLUX uses a deterministic and explainable campaign-level scorer.

The scorer evaluates evidence across a connected transaction group.

Campaign-level signals can include structural and relationship characteristics such as:

Shared entity reuse
Transactions connected through shared cards
Link density
Merchant diversity
BIN diversity
Multi-entity link fraction
Transaction concentration
Other campaign-level structural signals

A campaign response can expose evidence such as:

{
  "candidate_id": "CAND-000004",
  "score": 0.5492528209820067,
  "tier": "MEDIUM",
  "action": "review",
  "evidence": {
    "top_signals": [
      {
        "feature": "max_transactions_per_shared_card",
        "contribution": 0.14089661482159196
      },
      {
        "feature": "link_density",
        "contribution": 0.10639676730710583
      },
      {
        "feature": "distinct_bins_per_transaction",
        "contribution": 0.08609713327233913
      },
      {
        "feature": "distinct_merchants_per_transaction",
        "contribution": 0.08529658432448917
      }
    ]
  }
}

This means the system does not merely say:

Risk Score: 0.54

It can answer:

Why did this campaign receive that score?

10. Risk Tiers and Recommended Actions

Campaigns are assigned risk levels and recommended actions.

Risk Tier	Typical Response
LOW	Monitor / no immediate intervention
MEDIUM	REVIEW
HIGH	Escalated intervention

The exact action returned to the frontend originates from the backend scoring pipeline.

CONFLUX is designed to support investigation rather than hide its reasoning behind a black-box prediction.

11. Explainability

Explainability is a core part of the CONFLUX design.

For every scored campaign, the backend can expose evidence including:

Campaign ID
Connected transaction IDs
Risk score
Risk tier
Recommended action
Top contributing signals

The reasoning can therefore be represented as:

Suspicious Campaign
        │
        ▼
       Why?
        │
        ├── Shared card reuse
        ├── Cross-merchant connections
        ├── Dense entity relationships
        ├── Multiple shared entities
        └── Temporal concentration

Rather than:

"AI says fraud."

12. Backend Architecture

The backend is implemented in Python using FastAPI.

Its primary responsibilities include:

Receiving transactions
Maintaining in-memory transaction state
Maintaining entity relationships
Discovering campaign candidates
Scoring campaigns
Exposing campaign results
Providing health information
Supporting live frontend communication
API Architecture
                  ┌──────────────────────┐
                  │   Investigation UI   │
                  │      Frontend        │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
        REST API                       WebSocket
              │                             │
              └──────────────┬──────────────┘
                             ▼
                  ┌──────────────────────┐
                  │   CONFLUX Backend    │
                  │      FastAPI         │
                  └──────────────────────┘

REST endpoints provide state and campaign retrieval.

The WebSocket channel supports live transaction and investigation updates.

13. Backend Health

The deployed backend exposes a health endpoint.

Example response:

{
  "status": "ok",
  "scorer_loaded": true,
  "transactions_in_memory": 0,
  "active_websocket_clients": 0
}

The health check confirms that the campaign scoring system successfully loads.

Live backend:

CONFLUX API

14. Deployment

The CONFLUX backend is deployed as a live FastAPI service.

Deployment Stack
Python
FastAPI
Uvicorn
Render

The service starts using:

PYTHONPATH=src uvicorn conflux.api.main:app --host 0.0.0.0 --port $PORT

The deployment platform provides the runtime port through $PORT.

Health Check

CONFLUX API Health Check

A successful deployment confirms that:

✓ API is running
✓ FastAPI application started
✓ Campaign scorer loaded
✓ Model/scoring artifacts accessible
✓ Backend ready for frontend communication
15. Frontend

CONFLUX includes a separate frontend application built with:

Vite
TypeScript
HTTP API communication
WebSocket-based live updates

The frontend and backend are intentionally decoupled.

┌──────────────────────┐
│      Frontend        │
│ Investigation Console│
└──────────┬───────────┘
           │
     HTTP / WebSocket
           │
           ▼
┌──────────────────────┐
│   CONFLUX Backend    │
│      FastAPI         │
└──────────────────────┘

The frontend is designed as an investigation console, not simply a generic fraud dashboard.

Its purpose is to make the progression of coordinated activity visible.

Apparently Independent Transactions
                │
                ▼
       Entity Relationships Emerge
                │
                ▼
        Campaign Candidate Forms
                │
                ▼
         Campaign Risk Increases
                │
                ▼
          Evidence Becomes Visible
                │
                ▼
        Reviewer Receives Action
16. Investigation Console

The investigation experience focuses on making campaign reasoning understandable.

The interface includes concepts such as:

System status
Transaction activity
Campaign summaries
Risk tiers
Campaign alerts
Campaign details
Top contributing signals
Connected entities
Graph-based investigation

The goal is not simply to display:

Risk = 87%

Instead, the system aims to show the relationships behind the decision:

Transaction A
       │
       ├── Shared Card
       │
Transaction B ───── Merchant 1
       │
       └── Shared Device / IP
       │
Transaction C ───── Merchant 2

This makes the cross-merchant campaign structure understandable to a human reviewer.

17. Live Transaction Flow

CONFLUX can process transactions progressively.

During backend testing, transaction ingestion produced behavior such as:

Sending 8 transactions

transactions=1  candidates=1  scored=0
transactions=2  candidates=2  scored=0
transactions=3  candidates=3  scored=0
transactions=4  candidates=4  scored=0
transactions=5  candidates=4  scored=1
transactions=6  candidates=5  scored=1
transactions=7  candidates=6  scored=1
transactions=8  candidates=7  scored=1

This demonstrates an important property of the system:

A campaign does not necessarily become scoreable after the first transaction.

Evidence builds as additional transactions and entity relationships appear.

18. Example Campaign Output

After transactions are processed, campaign results can be summarized as:

Transactions:       8
Candidates:         7
Scored Campaigns:   1

High Risk:          0
Medium Risk:        1
Low Risk:           0
Example Detected Campaign
Candidate:
CAND-000004

Transactions:
9598253c...
f66cd021...

Score:
0.549

Tier:
MEDIUM

Recommended Action:
REVIEW

The connected transactions were independently verified across different merchants:

Merchant M0137
Merchant M0148

while sharing the same card fingerprint.

Therefore:

CROSS-MERCHANT = YES

19. Repository Structure

The repository is organized around the CONFLUX Python package and a separate frontend.

conflux/
│
├── data/
│   └── raw/
│       └── dataset_v4_final.csv
│
├── frontend/
│   └── Vite + TypeScript application
│
├── src/
│   └── conflux/
│       │
│       ├── api/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   ├── state.py
│       │   └── websocket.py
│       │
│       ├── evaluation/
│       │
│       ├── features/
│       │
│       ├── graph/
│       │
│       ├── models/
│       │   └── artifacts/
│       │
│       └── scoring/
│
├── tests/
├── tools/
│
├── ARCHITECTURE.md
├── DECISIONS.md
├── FEATURE_SPEC.md
├── PROJECT_CONTEXT.md
├── requirements.txt
└── render.yml

The project structure separates:

API and service logic
Feature engineering
Graph processing
Modeling artifacts
Campaign scoring
Evaluation
Testing
Frontend implementation
20. Model and Scoring Artifacts

The backend loads production scoring artifacts from:

src/conflux/models/artifacts/

Tracked artifacts include scoring references and metadata required by the campaign scorer.

The deployed health check confirms:

"scorer_loaded": true

This is important because the frontend is not intended to display invented campaign results.

Campaign candidates, scores, evidence, and actions originate from the actual backend pipeline.

21. Evaluation and Validation

The CONFLUX development process involved more than training a model and checking a single accuracy metric.

Validation work included:

Feature specification
Leakage auditing
Temporal and causal reasoning checks
Candidate diagnostics
Campaign scoring validation
Robustness testing
Adversarial testing
Benign burst testing
False-positive stress testing
Changed attack cadence
Weaker entity reuse
Increased legitimate traffic
Unseen campaign behavior
Temporal boundary cases

The production campaign scorer was deliberately designed to be deterministic and explainable.

This was an intentional design decision.

For an AI Risk Manager use case, a reviewer needs to understand:

What triggered the campaign?
        │
        ▼
What evidence connected the transactions?
        │
        ▼
Why was this risk score assigned?
        │
        ▼
Why is this action recommended?
22. Key Design Decisions
1. Campaigns, Not Just Transactions

A suspicious transaction alone is weak evidence.

The stronger signal is coordinated behavior across:

Entities + Relationships + Time

2. Cross-Merchant Evidence Matters

The system is specifically designed to identify relationships that individual merchants may not be able to see independently.

3. BIN Alone Is Not Campaign Evidence

A risky or frequently associated BIN is not sufficient proof of a coordinated campaign.

Campaign detection requires combined evidence.

4. Explainability Over Unnecessary Black-Box Complexity

A deterministic campaign scorer was selected for the production path because its decision process can be inspected and explained.

5. No Fake Evidence for the Demo

The investigation interface is intended to be driven by real backend output.

Scores, campaigns, entities, evidence, and actions should originate from the actual detection and scoring pipeline.

6. Temporal Causality Matters

Earlier decisions should not depend on future transaction information.

The system is designed around this principle throughout feature construction and temporal analysis.

23. Technology Stack
Backend
Python
FastAPI
Uvicorn
NumPy
Pandas
Scikit-learn
Joblib
Pydantic
Frontend
Vite
TypeScript
REST API communication
WebSocket-based live updates
Deployment
GitHub
Render
24. Running the Backend Locally
Install Dependencies
pip install -r requirements.txt
Start the API
PYTHONPATH=src uvicorn conflux.api.main:app --reload

On Windows PowerShell, environment variable syntax may differ depending on shell configuration.

The backend exposes a health endpoint once running.

25. Current Project Status
Component	Status
Synthetic Dataset v4	✅ Complete
Feature Engineering	✅ Complete
Transaction-Level ML Baseline	✅ Complete
Leakage / Feature Validation	✅ Complete
Cross-Entity Temporal Graph	✅ Complete
Campaign Candidate Discovery	✅ Complete
Campaign-Level Scoring	✅ Complete
Explainable Evidence Output	✅ Complete
Robustness / Adversarial Testing	✅ Complete
FastAPI Backend	✅ Complete
WebSocket Live Channel	✅ Complete
Backend Deployment	🟢 Live
Frontend Investigation Console	🟢 Functional / polishing
Final Demo Polish	🔧 In Progress
26. Project Philosophy

CONFLUX is built around a few principles:

A transaction is not proof. A pattern is.

A risky BIN is not a campaign.

A graph connection is not automatically malicious.

Suspicion increases when independent signals reinforce one another.

A human reviewer should be able to understand why the system raised an alert.

And most importantly:

Fraud campaigns are often distributed. Detection should be able to connect what individual merchants see separately.

27. Why the Name CONFLUX?

A conflux is a coming together or merging of things.

That is exactly what happens inside the system.

Separate transaction streams converge through shared relationships:

Cards
   +
Merchants
   +
Devices
   +
IPs
   +
Time
   │
   ▼
CONFLUX
   │
   ▼
Coordinated Campaign Intelligence
