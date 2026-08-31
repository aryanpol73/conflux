CONFLUX
Cross-Merchant Campaign Intelligence

CONFLUX is an explainable fraud detection system built to detect coordinated card-testing campaigns across multiple merchants.

Built for the Razorpay AI Buildathon — Track 02: AI Risk Manager.

A single fraudulent transaction can look normal.
A coordinated pattern across cards, merchants, devices, IPs, and time usually does not.

The Story

Card-testing fraud rarely appears as one obviously bad transaction.

Instead, attackers may rapidly test stolen or generated card credentials using small transactions distributed across multiple merchants. Individual merchants may only see a few seemingly independent attempts. Viewed in isolation, many of those transactions can appear legitimate.

The actual signal emerges only when transactions are connected.

The same card may appear across different merchants. Multiple cards may reuse a device. Different merchants may receive requests from the same IP signature. Activity may suddenly accelerate within a short time window.

CONFLUX is designed to detect that coordinated pattern rather than simply classify one transaction as fraudulent.

The system combines:

transaction-level behavioral signals,
temporal and cross-entity relationships,
campaign candidate discovery,
deterministic campaign risk scoring,
explainable evidence,
and a live investigation interface.

The core principle is simple:

A transaction is not proof. A coordinated pattern across entities and time is evidence.

1. The Problem

Distributed card-testing attacks are difficult to detect because the attack is often fragmented.

An attacker may:

test many cards,
distribute attempts across multiple merchants,
reuse infrastructure such as devices or IPs,
operate within short bursts,
use small transaction amounts,
and avoid making any single merchant's traffic look obviously malicious.

This creates a visibility problem.

A single merchant may see:

Card A → Merchant 1
Card B → Merchant 1
Card C → Merchant 1

Another merchant may independently see:

Card A → Merchant 2
Card D → Merchant 2
Card E → Merchant 2

Neither merchant necessarily sees the full campaign.

CONFLUX connects the observations:

                  Card A
                 /      \
          Merchant 1   Merchant 2
              |             |
           Device X ------ IP Y

The suspicious behavior is therefore not just:

"Is this transaction risky?"

It is:

"Are these apparently independent transactions actually part of the same coordinated campaign?"

2. What CONFLUX Does

CONFLUX processes transactions and progressively builds an investigation picture.

The system:

Receives transactions.
Maintains transaction and entity relationships.
Identifies connected campaign candidates.
Evaluates those candidates using campaign-level signals.
Assigns a risk score and tier.
Produces an explainable recommended action.
Streams the results to an investigation console.

The important distinction is that CONFLUX is not just a transaction fraud classifier.

It is a:

Coordinated cross-merchant campaign detection system.

3. What Makes a Campaign "Cross-Merchant"?

Cross-merchant behavior is a central part of the project.

For example, during backend testing, two transactions belonging to the same detected candidate were verified against the dataset:

Transaction A → Merchant M0137
Transaction B → Merchant M0148

Shared card fingerprint:
918040a152fbf88a

Result:

UNIQUE MERCHANTS: 2

RESULT: CROSS-MERCHANT = YES

This demonstrates the core CONFLUX idea:

The transactions occurred at different merchants, but shared an entity relationship that allowed the backend to connect them into a campaign candidate.

The graph and campaign logic can connect transactions through entities such as:

Card fingerprint
BIN
Device fingerprint
IP signature
Merchant
Temporal proximity

A campaign therefore becomes suspicious when multiple forms of evidence reinforce one another.

4. Dataset

CONFLUX uses a purpose-built synthetic transaction dataset.

The dataset was designed specifically for this project rather than relying on a trivially separable public dataset.

The design goal was to ensure that the detection problem required multi-signal reasoning.

Frozen dataset
dataset_v4_final.csv

Dataset characteristics include:

Property	Value
Total transactions	31,873
Attack rate	6.40%
Campaigns	45
Attacker archetypes	6
Legitimate hard negatives	Included
Identifier overlap between populations	Included
Independent IP field	Included

The dataset intentionally includes difficult cases where legitimate traffic can resemble suspicious behavior.

Examples include:

benign transaction bursts,
entity reuse,
overlapping identifiers,
shared BINs,
and traffic patterns that should not automatically be treated as attacks.

This prevents the solution from relying on a simplistic shortcut such as:

"High BIN risk = campaign."

That is explicitly not the CONFLUX design.

5. Detection Architecture

The system works in layers.

Incoming Transactions
        │
        ▼
Transaction Processing
        │
        ├──────────────► Transaction-level behavioral features
        │
        ▼
Temporal Entity Relationships
        │
        ▼
Campaign Candidate Discovery
        │
        ▼
Campaign Feature Extraction
        │
        ▼
Deterministic Campaign Risk Scorer
        │
        ├────────────► Risk Score
        ├────────────► Risk Tier
        ├────────────► Recommended Action
        └────────────► Explainable Evidence
                              │
                              ▼
                    Live Investigation Console
6. Transaction-Level Behavioral Modeling

CONFLUX includes transaction-level behavioral feature engineering.

Feature groups include:

Amount behavior
Device behavior
BIN behavior
Merchant pattern behavior
Velocity
Decline behavior

Feature construction follows causal and temporal constraints so that future transactions are not used to explain earlier ones.

A logistic regression baseline was developed as part of the modeling pipeline.

The ML layer provides useful behavioral signal, but CONFLUX deliberately does not treat transaction-level ML output as sufficient evidence of a coordinated campaign.

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

A transaction creates relationships between the entities involved.

Conceptually:

Card ─────┐
BIN ──────┤
Device ───┼── Transaction ─── Merchant
IP ───────┘

As more transactions arrive, these relationships can reveal connected structures.

For example:

Card A ─ Transaction 1 ─ Merchant X
   │
   └──── Transaction 2 ─ Merchant Y

Device D ─ Transaction 1
Device D ─ Transaction 3 ─ Merchant Z

Individually, these transactions may appear unrelated.

Together, they may reveal:

cross-merchant activity,
shared infrastructure,
entity reuse,
and temporal coordination.
8. Campaign Candidate Discovery

The graph layer identifies groups of transactions that may represent coordinated activity.

Candidate generation intentionally casts a reasonably broad net.

A candidate is not automatically a confirmed attack.

Instead:

Candidate
    ↓
Feature extraction
    ↓
Campaign scoring
    ↓
Risk tier + evidence

This separation is important.

Candidate discovery answers:

"What transactions might belong together?"

Campaign scoring answers:

"How suspicious is that connected group?"

9. Campaign Risk Scoring

CONFLUX uses a deterministic and explainable campaign-level scorer.

The scorer evaluates evidence across the connected transaction group.

Signals can include relationship and topology characteristics such as:

shared entity reuse,
number of transactions connected through shared cards,
link density,
merchant diversity,
BIN diversity,
multi-entity link fraction,
transaction concentration,
and other campaign-level structural signals.

A real backend response, for example, can return evidence like:

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

This is important for the project philosophy.

The UI does not need to say merely:

"Risk score: 0.54"

It can show:

Why did the system assign that score?

10. Risk Tiers and Actions

Campaigns are assigned risk levels and recommended actions.

Typical outputs include:

Risk Tier	Action
LOW	Monitor / no immediate intervention
MEDIUM	REVIEW
HIGH	Escalated intervention

The exact action returned to the frontend comes from the backend scoring pipeline.

The system is designed to support investigation rather than hide its reasoning behind a black-box prediction.

11. Explainability

Explainability is a core requirement of CONFLUX.

For every scored campaign, the backend can expose evidence including:

campaign ID,
connected transaction IDs,
risk score,
risk tier,
recommended action,
and the top contributing signals.

This allows a reviewer to understand the reasoning behind the alert.

The goal is:

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

AI says fraud.
12. Backend

The backend is implemented in Python using FastAPI.

The primary backend responsibilities include:

receiving transactions,
maintaining in-memory transaction state,
discovering candidates,
scoring campaigns,
exposing campaign results,
exposing health information,
and supporting live frontend communication.
API architecture

The backend provides REST endpoints for querying system state and uses WebSocket communication as the primary live channel.

Conceptually:

Frontend
   │
   ├── REST ─────────────► Backend state / campaigns
   │
   └── WebSocket ────────► Live transaction updates

The WebSocket channel is intended to drive the live investigation experience, while REST provides fallback and state retrieval.

13. Backend Health

The deployed backend exposes a health endpoint.

Example response:

{
  "status": "ok",
  "scorer_loaded": true,
  "transactions_in_memory": 0,
  "active_websocket_clients": 0
}

The health endpoint also confirms that the campaign scoring artifact was successfully loaded.

14. Deployment

The CONFLUX backend is deployed as a live FastAPI service.

Backend API:

CONFLUX API

Health endpoint:

CONFLUX API Health Check

The deployment uses:

Python
FastAPI
Uvicorn
Render

The Render startup configuration uses:

PYTHONPATH=src uvicorn conflux.api.main:app --host 0.0.0.0 --port $PORT

Render supplies the deployment port through $PORT.

15. Frontend

CONFLUX includes a separate frontend application built as a Vite-based vanilla TypeScript application.

The frontend and backend are intentionally decoupled.

Frontend
    │
    │ HTTP / WebSocket
    ▼
CONFLUX Backend

The frontend acts as an Investigation Console, not just a generic fraud dashboard.

Its purpose is to make the progression of the attack visible.

The intended experience is:

Apparently independent transactions
                ↓
       Entity relationships emerge
                ↓
      Candidate campaign forms
                ↓
       Campaign risk increases
                ↓
       Evidence becomes visible
                ↓
      Reviewer receives action
16. Investigation Console

The frontend focuses on making campaign reasoning visible.

The interface includes concepts such as:

system status,
transaction activity,
campaign summaries,
risk tiers,
campaign alerts,
campaign details,
top contributing signals,
connected entities,
and graph-based investigation.

The centerpiece is the ability to visualize how transactions are connected through entities.

Instead of presenting fraud detection as only:

Risk = 87%

CONFLUX aims to show:

Transaction A
       │
       ├── Shared Card
       │
Transaction B ─── Merchant 1
       │
       └── Shared Device / IP
       │
Transaction C ─── Merchant 2

This makes the cross-merchant campaign structure understandable to a human reviewer.

17. Live Transaction Flow

The system can process transactions progressively.

During backend testing, transaction ingestion produced results such as:

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

After transactions are processed, the campaigns endpoint can return a summary like:

Transactions: 8
Candidates: 7
Scored campaigns: 1

High risk: 0
Medium risk: 1
Low risk: 0

Example detected campaign:

Candidate: CAND-000004

Transactions:
- 9598253c...
- f66cd021...

Score:
0.549

Tier:
MEDIUM

Recommended Action:
REVIEW

The two transactions were independently verified as occurring at different merchants:

M0137
M0148

while sharing the same card fingerprint.

Therefore:

Cross-Merchant = YES

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
│       ├── api/
│       ├── evaluation/
│       ├── features/
│       ├── graph/
│       ├── models/
│       │   └── artifacts/
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

Some directories and internal modules may evolve as the project is polished.

20. Model and Scoring Artifacts

The backend loads trained/scoring artifacts from:

src/conflux/models/artifacts/

Tracked artifacts include scoring references and metadata required by the production campaign scorer.

The deployed health check confirms that the scoring system successfully loads:

"scorer_loaded": true

This is critical because the frontend is not displaying invented campaign data—the backend is responsible for generating actual candidate and scoring outputs.

21. Evaluation and Validation

The project development process included more than simply training a model and checking accuracy.

Validation work included:

feature specification,
leakage auditing,
temporal/causal reasoning checks,
candidate diagnostics,
campaign scoring validation,
robustness testing,
adversarial testing,
benign burst testing,
false-positive stress testing,
changed attack cadence,
weaker entity reuse,
increased legitimate traffic,
unseen campaign behavior,
and temporal boundary cases.

The production campaign scorer was deliberately chosen to be deterministic and explainable.

This was a design decision.

For an AI Risk Manager use case, a reviewer needs to understand:

what triggered the campaign,
what evidence connected the transactions,
and why the recommended action was produced.
22. Key Design Decisions
1. Campaigns, not just transactions

A suspicious transaction alone is weak evidence.

The stronger signal is coordinated behavior across:

Entities + Relationships + Time
2. Cross-merchant evidence matters

The system is designed specifically to identify relationships that individual merchants may not see.

3. BIN alone is not campaign evidence

A risky or frequently associated BIN is not sufficient proof of a coordinated campaign.

Campaign detection requires stronger combined evidence.

4. Explainability over unnecessary black-box complexity

A deterministic campaign scorer was selected for the production path because it makes the decision process inspectable.

5. No fake evidence for the demo

The investigation interface should be driven by actual backend output.

Scores, campaigns, entities, evidence, and actions should originate from the backend rather than being manually invented for visual effect.

6. Temporal causality matters

The system is designed around the principle that earlier decisions should not use future transaction information.

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
Vanilla TypeScript
WebSocket-based live communication
Deployment
GitHub
Render
24. Running the Backend Locally

Install dependencies:

pip install -r requirements.txt

Start the API:

PYTHONPATH=src uvicorn conflux.api.main:app --reload

On Windows PowerShell, the environment variable syntax may differ depending on the shell configuration.

The API can then be accessed locally at:

http://127.0.0.1:8000

Health check:

http://127.0.0.1:8000/health
25. Current Project Status
Component	Status
Synthetic Dataset v4	✅ Complete
Feature Engineering	✅ Complete
Transaction-level ML Baseline	✅ Complete
Leakage / Feature Validation	✅ Complete
Cross-Entity Temporal Graph	✅ Complete
Campaign Candidate Discovery	✅ Complete
Campaign-Level Scoring	✅ Complete
Explainable Evidence Output	✅ Complete
Robustness / Adversarial Testing	✅ Complete
FastAPI Backend	✅ Complete
WebSocket Live Channel	✅ Complete
Backend Deployment	✅ Live
Frontend Investigation Console	✅ Functional / actively polishing
Final Demo Polish	🔧 In progress
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
        ↓
      CONFLUX
        ↓
Coordinated Campaign Intelligence
