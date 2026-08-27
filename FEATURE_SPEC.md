
### 3. `FEATURE_SPEC.md`

This one is important because it prevents AI from randomly adding features.

```markdown
# CONFLUX — Behavioral Feature Specification

## Purpose

The feature layer captures weak behavioral evidence that will later be combined with graph/entity evidence.

The objective is NOT to create a feature that perfectly separates fraud from normal traffic.

## Locked Signal Groups

Exactly six signal groups:

1. Amount
2. Device
3. BIN
4. Merchant Pattern
5. Velocity
6. Decline Ratio

---

## 1. Amount

Purpose:

Capture unusual transaction amount behavior.

Required concepts:

- historical amount deviation
- amount similarity/clustering
- low-value ratio
- causal window statistics
- historical card amount behavior
- historical device amount behavior

Historical baselines must be causal.

The current transaction must not contaminate its own historical baseline.

Amount must remain a supporting signal.

---

## 2. Device

Purpose:

Capture device reuse and cross-entity behavior.

Required concepts:

- device transaction count
- distinct cards per device
- distinct merchants per device
- device velocity
- prior device reuse

Shared device activity is not automatically fraudulent.

---

## 3. BIN

The dataset's BIN field is the source of truth.

BIN is issuer-prefix context.

It must be treated as categorical/contextual information, not a continuous numerical quantity.

Required relationship concepts include:

- cards per BIN
- merchants per BIN
- BIN activity
- device → distinct BINs
- IP → distinct BINs
- valid BIN ↔ entity relationships

Do not derive BIN from card_fingerprint.

---

## 4. Merchant Pattern

Purpose:

Capture cross-merchant behavior.

Required concepts:

- merchants touched by a card
- merchants touched by a device
- merchants touched by an IP
- cross-merchant spread
- merchant dispersion
- related entities appearing across merchants

Example:

```text
Device A
→ Merchant 1
→ Merchant 2
→ Merchant 3
→ Merchant 4

is more interesting than repeated activity at one merchant.

Merchant features are behavioral precursors.

Actual graph construction belongs to the graph layer.

5. Velocity

Velocity captures temporal concentration.

Keep separate signals for:

card
device
IP

Do not collapse them into a single velocity feature.

The implementation supports multiple configurable windows appropriate to the simulated campaign durations.

Current starting windows:

30 seconds
120 seconds
300 seconds
1200 seconds

Windows belong in configuration.

6. Decline Ratio

Purpose:

Capture abnormal authorization behavior.

Required concepts:

card decline rate
device decline rate
IP decline rate
merchant decline rate

Conceptually:

declines / attempts

Historical/window calculations must be causal.

Campaign-level decline rate does not belong in this layer because campaigns have not yet been discovered.

Causality

For transaction t, features must use only information available according to their defined causal window.

Historical baselines use prior observations.

Future transactions must never affect the current transaction's features.

Timestamp precision must be preserved.

Do not truncate timestamps in a way that creates artificial ordering.

Duplicate timestamps require deterministic ordering.

Feature Rules

Every feature must have:

clear definition
entity
window
causal semantics
purpose
missing-value behavior
configuration source

Do not add features simply because they improve AUC.

A high AUC feature must be investigated for:

legitimate signal
synthetic artifact
missingness artifact
temporal leakage
actual leakage

AUC alone is not a reason to keep or remove a feature.