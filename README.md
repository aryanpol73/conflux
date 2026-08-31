CONFLUX
Cross-Merchant Campaign Intelligence

A coordinated structure intelligence system for detecting distributed card-testing and fraud campaigns across merchants, entities, and time.

<br/>

<img width="1916" height="1138" alt="image" src="https://github.com/user-attachments/assets/42da6649-a7ae-4e35-9edf-429eee8897ba" />


1. What CONFLUX Does

Card-testing fraud rarely shows up as one obviously bad transaction.

Instead, it appears as a pattern distributed across many merchants, cards, devices, IPs, and transactions over a short period of time. A single transaction may look completely normal when viewed in isolation.

CONFLUX is built to detect the pattern, not just the transaction.

It:

Scores individual transactions using behavioral ML features.
Builds a cross-entity graph connecting Cards, BINs, Devices, IP signatures, Merchants, and Transactions.
Detects related groups of activity as candidate coordinated structures.
Scores those groups as campaigns, rather than treating every transaction independently.
Provides a transparent and explainable risk score with supporting evidence.
Surfaces a recommended action for investigation:

REVIEW · STEP-UP · BLOCK

Core Design Principle
A transaction is not proof. A coordinated pattern is evidence.

No single signal is ever treated as proof of a campaign.

For example:

A suspicious BIN alone is not enough.
A shared device alone is not enough.
A burst of transactions alone is not enough.

A campaign requires combined evidence:

Cross-Entity Overlap
+ Temporal Structure
+ Behavioral Signals
+ Campaign-Level Evidence

2. The Problem — Razorpay Track 02: AI Risk Manager

CONFLUX focuses on detecting distributed card-testing attacks.

In a typical card-testing campaign, attackers rapidly test stolen or generated card details through small transactions. The goal is to identify cards that successfully authorize before they are used for larger fraudulent activity.

These attacks are difficult to detect because they are often:

Distributed Across Merchants

No individual merchant necessarily sees the complete attack.

Fast

The campaign can emerge as a burst of activity within a short observation window.

Designed to Blend In

Individual transactions may appear legitimate when examined independently.

The real signal emerges only when the activity is analyzed as a coordinated structure across entities and time.

3. CONFLUX in Action
Coordinated Structure Detection

CONFLUX visualizes relationships between transactions and the entities connected to them.

The system identifies candidate structures based on shared cards, devices, infrastructure, merchants, BIN relationships, and temporal activity.

The graph is not a decorative visualization. It represents the actual structural relationships used by the detection pipeline.

<img width="1021" height="662" alt="image" src="https://github.com/user-attachments/assets/7d9953e2-d56f-43da-97e5-dfb13b276fd4" />


Candidate Ranking and Investigation

Detected candidates are ranked using the backend campaign scorer.

Each investigation exposes:

Campaign Risk Score
Risk Tier
Recommended Action
Rank Among Candidates
Top Contributing Signals
Member Transactions
Connected Evidence
<img width="1917" height="1141" alt="image" src="https://github.com/user-attachments/assets/f0908777-81a3-42bc-b08e-42fa563d70d0" />


Interactive Entity Investigation

The graph allows suspicious structures to be explored visually.

Investigators can inspect how transactions connect through shared entities and identify relationships that would be difficult to see in a traditional transaction table.

4. Dataset

A synthetic transaction dataset was purpose-built for this project.

It was not scraped or pulled from a real payment dataset. The goal was to create a safe dataset that still required genuine multi-signal reasoning to solve.

The dataset was deliberately designed so that no single raw feature could trivially separate attacks from legitimate traffic.

Frozen Version

dataset_v4_final.csv

Property	Value
Total Transactions	31,873
Attack Rate	6.40%
Campaigns	45 across 6 attacker archetypes
Legitimate Traffic	Includes deliberate hard-negative clusters
Identifier Overlap	Deliberate overlap between legitimate and attack populations
IP Signature	Independent field, not derived from other leak-prone features
Dataset Design Philosophy

The dataset design framework was locked before implementation.

It covered:

Campaign definition
Available signals
Legitimate vs attack differentiation
Attacker variation across archetypes
Feature layers
Graph structure
How ML and graph evidence combine
Campaign scoring and explanation
Time-aware evaluation
What judges will see in the demo
V3 → V4 Improvements

The final dataset version addressed important design issues:

Fixed non-seeded uuid4 generation to improve reproducibility.
Fixed a card-reuse-count feature that was close to solving the task independently.
Replaced a thresholded "stump" leakage test with an AUC-based leakage test.

The purpose was not simply to make the dataset harder. It was to prevent artificial shortcuts and force the system to rely on meaningful combined evidence.

5. Architecture
High-Level Detection Pipeline
                         ┌─────────────────────┐
                         │  Transaction Stream │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │  Behavioral Signals │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Transaction-Level  │
                         │     ML Baseline     │
                         └──────────┬──────────┘
                                    │
                                    ▼
             ┌────────────────────────────────────────┐
             │     Cross-Entity Temporal Graph        │
             │                                        │
             │ Card · BIN · Device · IP · Merchant   │
             │                · Transaction           │
             └───────────────────┬────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Candidate Discovery │
                      │ Coordinated Groups  │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Campaign Scoring    │
                      │ Explainable Risk    │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ FastAPI + WebSocket │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ CONFLUX Frontend    │
                      │ Detection + Review  │
                      └─────────────────────┘
6. Transaction-Level Intelligence — Phase 1–2

CONFLUX first establishes transaction-level behavioral intelligence.

Six feature groups were engineered against a locked feature specification, with strict causal-window rules to prevent future information from leaking into historical scoring.

Feature Groups
Amount Behavior
Device Behavior
BIN Behavior
Merchant Pattern Behavior
Velocity
Decline Ratio

The feature specification includes exact behavioral formulations such as:

Coefficient of variation
Normalized-entropy merchant dispersion
Burst density
Other time-aware behavioral signals
ML Baseline

A Logistic Regression baseline was trained and evaluated using the project's transaction-level feature pipeline.

Relevant components include:

train_baseline.py
predict.py

Evaluation included:

Standard classification metrics
Feature analysis
Ablation work
Leakage auditing
Leakage Audit

The ML pipeline passed all 7 planned leakage checks:

Forbidden inputs
Feature-target alignment
Feature integrity
Temporal integrity
Preprocessing leakage
Causality
Suspicious / too-good-to-be-true signal strength

The objective was not just model performance. The objective was ensuring that the model was learning from information realistically available at scoring time.

Known Finding: BIN Dominance

BIN-group features were found to be particularly strong.

A static per-BIN historical fraud-rate signal across approximately 250 BINs reached around 0.90 AUC alone.

This was explicitly identified as a potential generalization risk.

It was not hidden or treated as an unquestioned success.

The finding remains documented as an important limitation and consideration for future generalization beyond the synthetic environment.

7. Cross-Entity Graph and Campaign Discovery — Phase 3

The central idea behind CONFLUX is that suspicious activity should be understood as a structure of relationships.

The graph is a heterogeneous temporal graph.

Node Types
Transaction
Card
BIN
Device
IP Signature
Merchant

A transaction connects to the entities involved in it.

This allows the system to discover structures such as:

         Card A ─────┐
                     │
Merchant 1 ── Transaction ── Device X
                     │
                 IP Signature
                     │
               Merchant 2
                     │
         Other Transactions

A coordinated campaign may therefore become visible through entity reuse and structural overlap, even when individual transactions are not obviously suspicious.

Causality Constraint

The graph is evaluated using information available only up to the relevant transaction timestamp.

No future transactions are allowed to provide evidence for historical detection.

This is essential because fraud detection systems must operate under realistic observation constraints.

Why an Explainable Graph Instead of a GNN?

For the first version of CONFLUX, an explainable graph-based approach was chosen over a Graph Neural Network.

The reasoning was practical:

Interpretability is critical
A reviewer needs to understand why a campaign was flagged
Judges should be able to inspect the evidence
The graph structure itself provides meaningful visual intelligence

The system therefore prioritizes:

Explainable structural evidence over black-box complexity.

Campaign Evidence Rule

A critical design rule was enforced:

BIN concentration alone is never campaign evidence.

A meaningful campaign requires cross-entity overlap combined with temporal structure.

The system looks for combinations such as:

Multiple Cards
+ Multiple Merchants
+ Shared Device or IP Infrastructure
+ Temporal Burst

8. Candidate Generation — Phase 3B and 3C

Candidate generation produced:

Metric	Result
Multi-Transaction Candidates Generated	4,372
Attack-Containing Candidates	81
Non-Campaign / Noise Candidates	4,291
Campaign Transaction Recall	99.27%
Campaign Transactions Recovered	2,026 / 2,041
Mixed / Contaminated Campaigns	0
Pure Campaign Candidates	46
Campaign + Normal Traffic Candidates	35

The candidate-generation process was verified as leakage-clean.

Important Design Decision

Candidates were not hand-cleaned to artificially improve downstream results.

Instead, diagnostic properties such as:

Inter-arrival time
BIN diversity
Burst rate
Link topology

were used to inform the campaign-scoring design.

The goal was to make the scorer robust to imperfect candidate generation rather than pretending the graph always produces perfectly isolated campaigns.

9. Campaign Scoring — Phase 3D and Phase 4

CONFLUX moves beyond transaction-level classification by scoring the candidate campaign itself.

Phase 3D — ML Signal Integration

Campaign-level scoring was integrated with transaction-level ML evidence.

This allows campaign intelligence to incorporate both:

Behavioral transaction signals
Structural campaign signals
Phase 4A — Deterministic Campaign Scorer

A deterministic and explainable campaign scorer was built and validated.

The scorer produces campaign-level outputs used by the backend and frontend, including:

Risk score
Risk tier
Recommended action
Signal contributions
Campaign evidence

The frontend therefore does not need to invent explanations.

Every visible investigation signal is intended to trace back to actual backend/scorer output.
Phase 4B — Robustness and Adversarial Testing

The campaign scorer was tested against multiple difficult conditions.

These included:

Unseen campaigns
Changed attack cadence
Changed scale-up cadence
Increased legitimate traffic
Weaker entity reuse
Temporal boundary cases
Observation-window vs horizon edge cases
Benign traffic bursts
Dedicated false-positive stress tests
Temporal train/test splitting

This phase was intended to answer a more important question than:

"Does the scorer work on the original data?"

Instead:

"Does the detection logic remain meaningful when attacker behavior and legitimate traffic conditions change?"

Phase 4C

The planned ML vs deterministic-scorer comparison was deliberately skipped to protect the project ship date.

The decision was:

Use the validated and adversarially tested deterministic campaign scorer as the production scorer.

Phase 4C remains a possible future extension rather than a blocker for the working system.

10. Backend

The backend is implemented as a Python FastAPI service.

It is decoupled from the frontend.

The frontend communicates with the backend through:

HTTP APIs
WebSocket connections

The backend is responsible for:

Loading scoring artifacts
Processing transaction activity
Generating candidate structures
Producing campaign scoring output
Providing evidence for investigations
Supporting real-time frontend updates
Deployment

The backend is deployed as a live web service.

Production backend:

CONFLUX API

The deployment was verified through the health endpoint.

The health response confirmed:

status: ok
scorer_loaded: true
Scoring artifacts successfully loaded
FastAPI application running successfully

A 404 response on / is expected because the application does not define a root route. The deployed health endpoint is the correct verification route.

11. Frontend and Live Investigation Experience

The CONFLUX frontend is designed to make the detection system's reasoning visible.

The focus is not simply displaying a risk number.

The interface exposes:

Risk Overview

High-level system state, candidate counts, and campaign activity.

Cross-Merchant Graph

The central visualization shows relationships between:

Merchants
Cards
Devices
IP-related entities
Transactions

Suspicious structures can be visually investigated.

Live Transaction Stream

Transactions are streamed into the interface during the replay experience.

Candidate Structures

Detected candidate campaigns are ranked and presented for investigation.

Investigation Panel

Investigators can inspect:

Risk score
Risk tier
Recommended action
Campaign members
Signal contributions
Transaction evidence
Graph Interaction

Selecting entities and campaign structures helps isolate relevant relationships while reducing unrelated visual noise.

12. Recommended Actions

Campaigns are surfaced with an operational recommendation.

REVIEW

The structure requires human investigation.

STEP-UP

Additional verification or friction should be introduced.

BLOCK

The coordinated evidence is sufficiently strong to justify stronger intervention.

The recommended action is based on campaign-level evidence rather than treating a single transaction as definitive proof.

13. Frontend Integrity Rule

A strict demo constraint was established:

The frontend must not invent campaign evidence.

Every meaningful signal displayed to the reviewer should originate from:

Backend output
Scorer output
Actual transaction/candidate data

The UI is therefore intended to function as an investigation interface, not a static dashboard containing fabricated explanations.

14. Technology Stack
Layer	Technology
Backend	Python
API Framework	FastAPI
Real-Time Communication	WebSocket
ML	Scikit-learn
Data Processing	Pandas / NumPy
Frontend	React
Frontend Tooling	Vite
Language	TypeScript
Styling	Tailwind CSS
Graph Visualization	Cytoscape.js
Backend Deployment	Render
15. Why Cytoscape.js?

The centerpiece of CONFLUX is a genuine relationship network.

The visual model contains entities and connections rather than a sequential workflow or DAG.

For this reason, a graph-focused visualization library was chosen.

Cytoscape.js fits the project because the core visual object is:

A network of entities and relationships.

This is fundamentally different from a standard flowchart.

16. Repository Structure

The repository is organized around the CONFLUX Python package, supporting project documentation, evaluation, testing, data, deployment configuration, and a separate frontend.

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
│
├── requirements.txt
└── render.yml
17. Model and Scoring Artifacts

The backend loads trained and scoring artifacts from:

src/conflux/models/artifacts/

Tracked artifacts include the scoring references and metadata required by the production campaign scorer.

The deployed health check confirms:

"scorer_loaded": true

This is important because the frontend is not simply displaying static campaign examples.

The backend is responsible for generating actual scoring output used by the investigation experience.

18. Evaluation and Validation

The project development process included more than simply training a model and checking accuracy.

Validation work included:

Feature leakage auditing
Forbidden-input checks
Temporal integrity checks
Causal feature construction
Candidate-generation diagnostics
Campaign transaction recall
Contamination checks
Robustness testing
Adversarial testing
False-positive stress testing
Temporal evaluation

The overall philosophy was:

Speed of implementation should not replace validation.

AI-assisted development was used to accelerate implementation, but major project decisions were structured around:

Locked specifications
Explicit validation
Leakage awareness
Robustness testing
Explainability
19. Tooling Used During Development
Purpose	Tooling Used
Implementation / Feature Development	AI-assisted development workflows
Evaluation and Analysis	AI-assisted review and analysis
Review / Validation / Testing	Iterative AI-assisted validation
Backend Development	Python ecosystem
Frontend Development	React + TypeScript ecosystem

The development approach was intentionally fast and AI-assisted.

However:

AI assistance was used to accelerate implementation — not to eliminate specification, validation, leakage auditing, or robustness testing.

20. Development Timeline
Original deadline: August 28, 2026
Buffer / stabilization period: August 30 – September 3, 2026
Final submission target: September 4, 2026

The project timeline prioritized:

Correctness
Backend completion
Deployment verification
Frontend integration
Demo quality and polish
21. Status Summary
Component	Status
Dataset v4	✅ Frozen
Transaction-Level ML Baseline	✅ Complete
Leakage Audit	✅ Completed
Cross-Entity Graph	✅ Complete
Candidate Generation	✅ Complete
Deterministic Campaign Scorer	✅ Built and Validated
Robustness / Adversarial Testing	✅ Completed
Backend API	✅ Complete
WebSocket Integration	✅ Working
Backend Deployment	✅ Live and Verified
Frontend Investigation Experience	✅ Implemented
Demo Polish / Final Submission Preparation	🔧 Final Stage

22. Key Design Principles
1. A Transaction Is Not Proof

A pattern across entities and time is stronger evidence than an isolated transaction.

2. No Single Signal Is Sufficient

A risky BIN, shared device, or transaction burst should not independently define a coordinated campaign.

3. Structure Matters

Fraud campaigns can be detected through the relationships between:

Cards · Devices · IP Infrastructure · BINs · Merchants · Transactions · Time

4. Causality Matters

The system should not rely on information that would only become available in the future.

5. Explainability Matters

A risk manager must understand:

Why a candidate was flagged
Which signals contributed
Which transactions belong to the structure
What action is recommended
6. The UI Must Reflect Real Evidence

No invented campaign explanations. No fake risk signals for visual effect.

The investigation interface should expose real backend and scorer output.

23. The Core Idea
A suspicious transaction
        │
        ▼
may look harmless alone
        │
        ▼
but becomes meaningful when connected to
        │
        ├── other cards
        ├── other merchants
        ├── shared devices
        ├── shared infrastructure
        └── a coordinated time pattern
        │
        ▼
COORDINATED STRUCTURE
        │
        ▼
CAMPAIGN INTELLIGENCE
CONFLUX
Cross-Merchant Campaign Intelligence

Detect the structure. Understand the evidence. Act on the campaign.
