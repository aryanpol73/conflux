/**
 * Frontend mirror of the CONFLUX backend contract, plus shared view models.
 *
 * Verified against src/conflux/api/schemas.py (Phase 6). Every wire-level name
 * here exists on the wire. Index signatures preserve pipeline fields declared
 * extra="allow" server-side without the frontend inventing any.
 *
 * This is the only place types are declared. No other module redeclares them.
 */

/* ================================================================== */
/* Transaction                                                         */
/* ================================================================== */

/** Exact production columns, in dataset order. extra="forbid" server-side. */
export const TRANSACTION_FIELDS = [
  'transaction_id',
  'timestamp',
  'merchant_id',
  'card_fingerprint',
  'bin',
  'amount',
  'device_fingerprint',
  'ip_signature',
  'auth_outcome',
] as const;

export type TransactionField = (typeof TRANSACTION_FIELDS)[number];

/** Evaluation-only ground truth. Rejected loudly by the API, never used here. */
export const FORBIDDEN_FIELDS = ['label', 'campaign_id'] as const;

/** Identifier fields the server rejects when blank. */
export const IDENTIFIER_FIELDS = [
  'transaction_id',
  'merchant_id',
  'card_fingerprint',
  'bin',
  'device_fingerprint',
  'ip_signature',
  'auth_outcome',
] as const;

export interface TransactionRecord {
  transaction_id: string;
  timestamp: string;
  merchant_id: string;
  card_fingerprint: string;
  /** String, always. Sending a string keeps the backend grouping key stable. */
  bin: string;
  amount: number;
  device_fingerprint: string;
  ip_signature: string;
  auth_outcome: string;
}

/* ================================================================== */
/* Detection result (mirrors DetectionResult / Campaign / ...)         */
/* ================================================================== */

export interface TopSignal {
  feature: string;
  /** Server-side: `contribution: float | None`. */
  contribution?: number | null;
  [extra: string]: unknown;
}

export interface CampaignEvidence {
  top_signals?: TopSignal[];
  [extra: string]: unknown;
}

export interface Campaign {
  /** `str | int` on the wire. Coerce before use. */
  candidate_id: string | number;
  transaction_ids?: Array<string | number>;
  score?: number | null;
  tier?: string | null;
  action?: string | null;
  evidence?: CampaignEvidence;
  [extra: string]: unknown;
}

export interface DetectionSummary {
  n_transactions?: number;
  n_candidates?: number;
  n_scored?: number;
  n_high_risk?: number;
  n_medium_risk?: number;
  n_low_risk?: number;
  [extra: string]: unknown;
}

export interface DetectionResult {
  status?: string;
  summary?: DetectionSummary;
  campaigns?: Campaign[];
  [extra: string]: unknown;
}

export interface HealthResponse {
  status: string;
  scorer_loaded: boolean;
  transactions_in_memory: number;
  active_websocket_clients?: number;
  scorer_artifact_path?: string | null;
  transaction_columns?: string[];
  load_error?: string | null;
  [extra: string]: unknown;
}

/* ================================================================== */
/* WebSocket protocol                                                  */
/* ================================================================== */

export type ClientMessageType = 'transaction' | 'ping' | 'snapshot';

export const CLIENT_MESSAGE_TYPES: readonly ClientMessageType[] = [
  'transaction',
  'ping',
  'snapshot',
] as const;

export type ClientMessage =
  | { type: 'transaction'; data: TransactionRecord }
  | { type: 'ping' }
  | { type: 'snapshot' };

/** Exhaustive as of Phase 6 websocket.py / main.py. */
export type ServerErrorCode =
  | 'invalid_json'
  | 'invalid_message'
  | 'unknown_message_type'
  | 'invalid_transaction'
  | 'scorer_unavailable'
  | 'detection_failed'
  | 'error';

export interface ConnectionAckPayload {
  transactions_in_memory: number;
  scorer_loaded: boolean;
  transaction_columns?: string[];
  accepted_message_types?: string[];
  [extra: string]: unknown;
}

export type ServerMessage =
  | { type: 'connection_ack'; data: ConnectionAckPayload }
  | { type: 'detection_update'; data: DetectionResult }
  | { type: 'pong' }
  | { type: 'error'; code: ServerErrorCode; message: string; detail?: unknown };

/** Any frame, before narrowing. `type` is the only guaranteed key. */
export interface ServerEnvelope {
  type: string;
  [extra: string]: unknown;
}

/** Structured field errors from format_validation_error(). */
export interface ValidationDetail {
  field: string;
  message: string;
  type: string;
}

/* ================================================================== */
/* View models                                                         */
/* ================================================================== */

export type ConnectionState =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'failed';

export type RiskBand = 'high' | 'medium' | 'low' | 'unranked';

export type GraphMode = 'detection' | 'investigation';

/**
 * A candidate prepared for rendering.
 *
 * `key` derives from the sorted transaction-id set, not from candidate_id,
 * because candidate_id is regenerated per pipeline run and is not guaranteed
 * stable. Selection and animation key on `key`.
 */
export interface CandidateView {
  key: string;
  candidateId: string;
  transactionIds: string[];
  score: number | null;
  tier: string;
  band: RiskBand;
  action: string;
  topSignals: TopSignal[];
  /** 1-based rank by score within THIS run. Not a population percentile. */
  rank: number;
  /** Scored candidates in this run — the denominator for `rank`. */
  totalScoredInRun: number;
  raw: Campaign;
}

export interface DetectionView {
  status: string;
  summary: DetectionSummary;
  candidates: CandidateView[];
  byKey: Map<string, CandidateView>;
  receivedAt: number;
  /** REST applies DetectionResult defaults; WS delivers the raw pipeline dict. */
  source: 'websocket' | 'rest';
}

export interface ServerErrorView {
  code: ServerErrorCode;
  message: string;
  details: ValidationDetail[];
  raw: unknown;
  /** scorer_unavailable is operator-actionable: show a blocking banner. */
  blocking: boolean;
}

/* ================================================================== */
/* Story                                                               */
/* ================================================================== */

/** Acts I–V of the narrative, plus pre-roll states. */
export type StoryBeat =
  | 'idle'
  | 'warmup'
  | 'observe'
  | 'link'
  | 'converge'
  | 'investigate'
  | 'score';

export type ReplayState = 'idle' | 'loading' | 'ready' | 'running' | 'paused' | 'done';

/* ================================================================== */
/* Graph                                                               */
/* ================================================================== */

export type GraphNodeKind = 'transaction' | 'candidate';

/**
 * `membership` edges come from campaigns[].transaction_ids and ARE backend
 * output. Entity edges are computed locally from transactions this frontend
 * sent, because the backend returns no edges. The distinction is rendered.
 */
export type GraphLinkKind = 'membership' | 'card' | 'device' | 'ip';

export interface GraphNode {
  id: string;
  kind: GraphNodeKind;
  label: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  /** Pinned position while dragging. */
  fx: number | null;
  fy: number | null;
  band: RiskBand;
  /** Present on candidate nodes and on member transaction nodes. */
  candidateKey: string | null;
  transaction: TransactionRecord | null;
  memberCount: number;
  createdAt: number;
}

export interface GraphLink {
  id: string;
  source: string;
  target: string;
  kind: GraphLinkKind;
  /** Masked shared-entity value, for the tooltip. Empty for membership. */
  detail: string;
  /** True only for membership edges. Drives solid vs dashed rendering. */
  backendDerived: boolean;
}

export interface GraphModel {
  nodes: GraphNode[];
  links: GraphLink[];
}

/* ================================================================== */
/* UI                                                                  */
/* ================================================================== */

export interface ActivityItem {
  transactionId: string;
  merchant: string;
  amount: number;
  outcome: string;
  at: number;
}

export interface AppStats {
  sent: number;
  remaining: number;
  candidates: number;
  scored: number;
}

/* ================================================================== */
/* Guards and coercion                                                 */
/* ================================================================== */

export function isRecordObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function asString(v: unknown): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number' && Number.isFinite(v)) return String(v);
  if (typeof v === 'bigint') return v.toString();
  return '';
}

export function asFiniteNumber(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export function toRiskBand(tier: unknown, score: unknown): RiskBand {
  const t = asString(tier).toLowerCase();
  if (t.includes('high')) return 'high';
  if (t.includes('med')) return 'medium';
  if (t.includes('low')) return 'low';
  // No tier string: report presence of a score only, never invent thresholds.
  return asFiniteNumber(score) === null ? 'unranked' : 'low';
}

/** Stable identity across pipeline runs: sorted transaction-id set, joined. */
export function candidateKey(transactionIds: Array<string | number>): string {
  const ids = transactionIds.map(asString).filter(Boolean).sort();
  return ids.length ? ids.join('|') : '';
}

export function isServerEnvelope(v: unknown): v is ServerEnvelope {
  return isRecordObject(v) && typeof v.type === 'string';
}

/** Short, readable rendering of an opaque fingerprint. */
export function maskIdentifier(value: string, keep = 8): string {
  const text = asString(value);
  return text.length <= keep ? text : `${text.slice(0, keep)}…`;
}
