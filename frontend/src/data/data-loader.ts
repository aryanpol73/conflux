/**
 * Transport layer for the CONFLUX Phase 6 API.
 *
 * Responsibilities: one WebSocket to /ws, strict single-flight request
 * discipline with a FIFO queue, transaction-id deduplication, reconnection,
 * and normalisation of pipeline output into view models. No rendering.
 *
 * Contains no detection logic. It never scores, never derives tiers, and never
 * fabricates fields the backend did not send.
 *
 * Single-flight rationale: the backend re-runs the entire pipeline over the
 * whole in-memory population on every ingest. Sending faster than it can score
 * only grows a server-side backlog. One frame at a time, always.
 *
 * FIFO reply matching is sound because ConnectionManager.broadcast() excludes
 * the sender: a socket receives exactly one message per frame it sends. A
 * second connected client can inject unsolicited detection_updates, so we check
 * /health.active_websocket_clients on connect and warn.
 */

import {
  TRANSACTION_FIELDS,
  FORBIDDEN_FIELDS,
  IDENTIFIER_FIELDS,
  asFiniteNumber,
  asString,
  candidateKey,
  isRecordObject,
  isServerEnvelope,
  toRiskBand,
  type Campaign,
  type CandidateView,
  type ClientMessage,
  type ConnectionAckPayload,
  type ConnectionState,
  type DetectionResult,
  type DetectionSummary,
  type DetectionView,
  type HealthResponse,
  type ServerEnvelope,
  type ServerErrorCode,
  type ServerErrorView,
  type TopSignal,
  type TransactionRecord,
  type ValidationDetail,
} from './types';

/* ------------------------------------------------------------------ */
/* Endpoint resolution                                                 */
/* ------------------------------------------------------------------ */

export function resolveApiBase(): string {
  const override = new URLSearchParams(window.location.search).get('api');
  if (override) return override.replace(/\/+$/, '');
  const env = (import.meta as { env?: Record<string, string> }).env;
  if (env?.VITE_CONFLUX_API) return env.VITE_CONFLUX_API.replace(/\/+$/, '');
  return 'http://localhost:8000';
}

export function toWebSocketUrl(base: string): string {
  const url = new URL(base, window.location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = `${url.pathname.replace(/\/+$/, '')}/ws`;
  url.search = '';
  return url.toString();
}

/* ------------------------------------------------------------------ */
/* Sanitisation                                                        */
/* ------------------------------------------------------------------ */

export class TransactionRejected extends Error {
  field: string;

  constructor(field: string, message: string) {
    super(message);
    this.name = 'TransactionRejected';
    this.field = field;
  }
}

/**
 * Produce exactly the nine accepted fields, or throw.
 *
 * Mirrors the server validators so a bad row is caught locally instead of
 * costing a round trip: extra="forbid", non-blank identifiers, finite
 * strictly-positive amount, parseable timestamp, string bin.
 */
export function sanitizeTransaction(input: unknown): TransactionRecord {
  if (!isRecordObject(input)) {
    throw new TransactionRejected('*', 'transaction must be an object');
  }

  for (const banned of FORBIDDEN_FIELDS) {
    if (banned in input) {
      throw new TransactionRejected(
        banned,
        `'${banned}' is evaluation-only ground truth and must never be sent`
      );
    }
  }

  const out: Record<string, unknown> = {};
  for (const field of TRANSACTION_FIELDS) {
    if (!(field in input) || input[field] === null || input[field] === undefined) {
      throw new TransactionRejected(field, `missing required column '${field}'`);
    }
    out[field] = input[field];
  }

  for (const field of IDENTIFIER_FIELDS) {
    const text = asString(out[field]).trim();
    if (!text) throw new TransactionRejected(field, `'${field}' must not be blank`);
    out[field] = text;
  }

  const amount = asFiniteNumber(out.amount);
  if (amount === null) {
    throw new TransactionRejected('amount', "'amount' must be a finite number");
  }
  if (amount <= 0) {
    throw new TransactionRejected('amount', "'amount' must be strictly positive");
  }
  out.amount = amount;

  const timestamp = asString(out.timestamp).trim();
  if (!timestamp || Number.isNaN(Date.parse(timestamp))) {
    throw new TransactionRejected(
      'timestamp',
      `'timestamp' is not a parseable datetime: ${JSON.stringify(out.timestamp)}`
    );
  }
  out.timestamp = timestamp;

  return out as unknown as TransactionRecord;
}

/* ------------------------------------------------------------------ */
/* Normalisation                                                       */
/* ------------------------------------------------------------------ */

function normalizeSignals(evidence: unknown): TopSignal[] {
  if (!isRecordObject(evidence)) return [];
  const raw = evidence.top_signals;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(isRecordObject)
    .map((s) => ({
      ...s,
      feature: asString(s.feature),
      contribution: asFiniteNumber(s.contribution),
    }))
    .filter((s) => s.feature !== '')
    .sort((a, b) => Math.abs(b.contribution ?? 0) - Math.abs(a.contribution ?? 0));
}

export function normalizeDetection(
  raw: unknown,
  source: 'websocket' | 'rest'
): DetectionView {
  const result: DetectionResult = isRecordObject(raw) ? (raw as DetectionResult) : {};
  const campaigns: Campaign[] = Array.isArray(result.campaigns) ? result.campaigns : [];

  const scored = campaigns
    .map((c) => ({ campaign: c, score: asFiniteNumber(c.score) }))
    .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));

  const totalScoredInRun = scored.filter((s) => s.score !== null).length;

  const candidates: CandidateView[] = scored.map(({ campaign, score }, index) => {
    const transactionIds = (
      Array.isArray(campaign.transaction_ids) ? campaign.transaction_ids : []
    )
      .map(asString)
      .filter(Boolean);
    const candidateId = asString(campaign.candidate_id);
    return {
      // Fall back to candidate_id only when the backend sent no member ids;
      // such a node cannot be matched across runs.
      key: candidateKey(transactionIds) || `candidate:${candidateId}`,
      candidateId,
      transactionIds,
      score,
      tier: asString(campaign.tier),
      band: toRiskBand(campaign.tier, campaign.score),
      action: asString(campaign.action),
      topSignals: normalizeSignals(campaign.evidence),
      rank: index + 1,
      totalScoredInRun,
      raw: campaign,
    };
  });

  const summary: DetectionSummary = isRecordObject(result.summary) ? result.summary : {};
  const byKey = new Map(candidates.map((c) => [c.key, c]));

  return {
    status: asString(result.status) || 'unknown',
    summary,
    candidates,
    byKey,
    receivedAt: Date.now(),
    source,
  };
}

export function normalizeServerError(env: ServerEnvelope): ServerErrorView {
  const code = (asString(env.code) || 'error') as ServerErrorCode;
  const details: ValidationDetail[] = Array.isArray(env.detail)
    ? env.detail.filter(isRecordObject).map((d) => ({
        field: asString(d.field),
        message: asString(d.message),
        type: asString(d.type),
      }))
    : [];
  return {
    code,
    message: asString(env.message) || 'the server reported an error',
    details,
    raw: env,
    blocking: code === 'scorer_unavailable',
  };
}

/* ------------------------------------------------------------------ */
/* Loader                                                              */
/* ------------------------------------------------------------------ */

export interface DataLoaderEvents {
  onConnectionState?(state: ConnectionState, info?: string): void;
  onAck?(payload: ConnectionAckPayload): void;
  onDetection?(view: DetectionView, solicited: boolean): void;
  onServerError?(error: ServerErrorView): void;
  /** depth = queued frames, inFlight = a frame is awaiting its reply. */
  onQueueChange?(depth: number, inFlight: boolean): void;
  onWarning?(message: string): void;
}

export interface DataLoaderOptions {
  apiBase?: string;
  /** Reply timeout. The pipeline is O(population); be generous. */
  requestTimeoutMs?: number;
  maxReconnectAttempts?: number;
}

interface PendingRequest {
  frame: ClientMessage;
  resolve(env: ServerEnvelope): void;
  reject(err: Error): void;
}

export class DataLoader {
  readonly apiBase: string;
  readonly wsUrl: string;

  private socket: WebSocket | null = null;
  private state: ConnectionState = 'idle';
  private readonly queue: PendingRequest[] = [];
  private inFlight: PendingRequest | null = null;
  private timeoutHandle: number | null = null;
  private reconnectAttempts = 0;
  private closedByUser = false;

  /** Ids already accepted by the server. add_transaction() appends without
   *  deduplicating, so resending an id would duplicate the row. */
  private readonly sentIds = new Set<string>();

  private readonly timeoutMs: number;
  private readonly maxReconnectAttempts: number;
  private readonly events: DataLoaderEvents;

  constructor(
    events: DataLoaderEvents,
    options: DataLoaderOptions = {}
  ) {
    this.events = events;
    this.apiBase = (options.apiBase ?? resolveApiBase()).replace(/\/+$/, '');
    this.wsUrl = toWebSocketUrl(this.apiBase);
    this.timeoutMs = options.requestTimeoutMs ?? 45_000;
    this.maxReconnectAttempts = options.maxReconnectAttempts ?? 6;
  }

  get connectionState(): ConnectionState {
    return this.state;
  }

  get queueDepth(): number {
    return this.queue.length;
  }

  get isBusy(): boolean {
    return this.inFlight !== null;
  }

  get sentCount(): number {
    return this.sentIds.size;
  }

  /* -- REST ----------------------------------------------------------- */

  async fetchHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.apiBase}/health`, {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) throw new Error(`GET /health failed: ${response.status}`);
    return (await response.json()) as HealthResponse;
  }

  async resetBackend(): Promise<void> {
    const response = await fetch(`${this.apiBase}/reset`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
    });

    const body: unknown = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(this.describeRestError(response.status, body));
    }
    }

  /** Initial dashboard load. Safe at zero transactions. */
  async fetchCampaigns(): Promise<DetectionView> {
    const response = await fetch(`${this.apiBase}/campaigns`, {
      headers: { Accept: 'application/json' },
    });
    const body: unknown = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(this.describeRestError(response.status, body));
    return normalizeDetection(body, 'rest');
  }

  /** Fallback ingest. Returns a population result, not a per-transaction score. */
  async postTransaction(input: unknown): Promise<DetectionView> {
    const tx = sanitizeTransaction(input);
    if (this.sentIds.has(tx.transaction_id)) {
      throw new TransactionRejected(
        'transaction_id',
        `duplicate transaction_id '${tx.transaction_id}'`
      );
    }
    const response = await fetch(`${this.apiBase}/transactions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(tx),
    });
    const body: unknown = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(this.describeRestError(response.status, body));
    this.sentIds.add(tx.transaction_id);
    const view = normalizeDetection(body, 'rest');
    this.events.onDetection?.(view, true);
    return view;
  }

  private describeRestError(status: number, body: unknown): string {
    if (isRecordObject(body)) {
      if (typeof body.detail === 'string') return `HTTP ${status}: ${body.detail}`;
      if (Array.isArray(body.detail)) {
        const first = body.detail.find(isRecordObject);
        if (first) return `HTTP ${status}: ${asString(first.msg) || JSON.stringify(first)}`;
      }
      if (typeof body.error === 'string') return `HTTP ${status}: ${body.error}`;
    }
    return `HTTP ${status}`;
  }

  /* -- WebSocket ------------------------------------------------------ */

  async connect(): Promise<void> {
    this.closedByUser = false;
    await this.warnIfOtherClients();
    this.openSocket();
  }

  /**
   * Unsolicited broadcasts from a second client would break FIFO reply
   * matching. Detect that up front rather than corrupting silently.
   */
  private async warnIfOtherClients(): Promise<void> {
    try {
      const health = await this.fetchHealth();
      if (!health.scorer_loaded) {
        this.events.onWarning?.(
          `Backend reports scorer_loaded=false (${health.load_error ?? 'no detail'}). ` +
            'Detection will return scorer_unavailable until the Phase 5.5 artifact is built.'
        );
      }
      if ((health.active_websocket_clients ?? 0) > 0) {
        this.events.onWarning?.(
          `${health.active_websocket_clients} other WebSocket client(s) already connected. ` +
            'Detection updates broadcast to all peers; close other tabs for a clean demo.'
        );
      }
    } catch {
      // Health is advisory. Connect anyway and let the socket report failure.
    }
  }

  private openSocket(): void {
    this.setState(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');
    const socket = new WebSocket(this.wsUrl);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.setState('open');
      this.drain();
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        this.events.onWarning?.('discarded a non-JSON frame from the server');
        return;
      }
      if (!isServerEnvelope(parsed)) {
        this.events.onWarning?.('discarded a server frame with no "type" field');
        return;
      }
      this.handleEnvelope(parsed);
    };

    socket.onerror = () => {
      this.events.onWarning?.(`WebSocket error on ${this.wsUrl}`);
    };

    socket.onclose = (event: CloseEvent) => {
      this.socket = null;
      this.failInFlight(
        new Error(`socket closed (code ${event.code}) before a reply arrived`)
      );
      if (this.closedByUser) {
        this.setState('closed');
        return;
      }
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.setState('failed', `gave up after ${this.reconnectAttempts} reconnect attempts`);
        this.failQueue(new Error('connection failed'));
        return;
      }
      const delay = Math.min(500 * 2 ** this.reconnectAttempts, 8_000);
      this.reconnectAttempts += 1;
      this.setState('reconnecting', `retrying in ${delay} ms`);
      window.setTimeout(() => {
        if (!this.closedByUser) this.openSocket();
      }, delay);
    };
  }

  private handleEnvelope(env: ServerEnvelope): void {
    if (env.type === 'connection_ack') {
      const payload = isRecordObject(env.data) ? (env.data as ConnectionAckPayload) : undefined;
      if (payload) this.events.onAck?.(payload);
      return; // Not a reply — never resolves a pending request.
    }

    const pending = this.inFlight;
    if (pending) {
      this.clearTimeout();
      this.inFlight = null;
      pending.resolve(env);
    }

    if (env.type === 'detection_update') {
      this.events.onDetection?.(normalizeDetection(env.data, 'websocket'), pending !== null);
    } else if (env.type === 'error') {
      this.events.onServerError?.(normalizeServerError(env));
    } else if (env.type !== 'pong') {
      this.events.onWarning?.(`unrecognised server message type '${env.type}'`);
    }

    this.emitQueue();
    this.drain();
  }

  /* -- Request discipline --------------------------------------------- */

  private enqueue(frame: ClientMessage): Promise<ServerEnvelope> {
    return new Promise<ServerEnvelope>((resolve, reject) => {
      this.queue.push({ frame, resolve, reject });
      this.emitQueue();
      this.drain();
    });
  }

  private drain(): void {
    if (this.inFlight || this.queue.length === 0) return;
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const next = this.queue.shift()!;
    this.inFlight = next;
    try {
      socket.send(JSON.stringify(next.frame));
    } catch (error) {
      this.inFlight = null;
      next.reject(error instanceof Error ? error : new Error(String(error)));
      this.emitQueue();
      return;
    }
    this.timeoutHandle = window.setTimeout(() => {
      // Never deadlock the queue on a lost reply.
      this.failInFlight(new Error(`no reply within ${this.timeoutMs} ms`));
      this.drain();
    }, this.timeoutMs);
    this.emitQueue();
  }

  private clearTimeout(): void {
    if (this.timeoutHandle !== null) {
      window.clearTimeout(this.timeoutHandle);
      this.timeoutHandle = null;
    }
  }

  private failInFlight(error: Error): void {
    this.clearTimeout();
    const pending = this.inFlight;
    this.inFlight = null;
    pending?.reject(error);
    this.emitQueue();
  }

  private failQueue(error: Error): void {
    while (this.queue.length) this.queue.shift()!.reject(error);
    this.emitQueue();
  }

  private emitQueue(): void {
    this.events.onQueueChange?.(this.queue.length, this.inFlight !== null);
  }

  private setState(state: ConnectionState, info?: string): void {
    this.state = state;
    this.events.onConnectionState?.(state, info);
  }

  /* -- Public sends ---------------------------------------------------- */

  /**
   * Ingest one transaction and receive the resulting detection run.
   * Resolves null when the id was already sent (deduplicated locally).
   */
  async sendTransaction(input: unknown): Promise<DetectionView | null> {
    const tx = sanitizeTransaction(input);
    if (this.sentIds.has(tx.transaction_id)) return null;
    this.sentIds.add(tx.transaction_id);

    const env = await this.enqueue({ type: 'transaction', data: tx });
    if (env.type === 'error') {
      // Server refused it, so it is not in the population. Allow a retry.
      this.sentIds.delete(tx.transaction_id);
      const err = normalizeServerError(env);
      throw new Error(`${err.code}: ${err.message}`);
    }
    return normalizeDetection(env.data, 'websocket');
  }

  /** Re-run detection over the current population without ingesting. */
  async requestSnapshot(): Promise<DetectionView> {
    const env = await this.enqueue({ type: 'snapshot' });
    if (env.type === 'error') {
      const err = normalizeServerError(env);
      throw new Error(`${err.code}: ${err.message}`);
    }
    return normalizeDetection(env.data, 'websocket');
  }

  async ping(): Promise<boolean> {
    const env = await this.enqueue({ type: 'ping' });
    return env.type === 'pong';
  }

  close(): void {
    this.closedByUser = true;
    this.clearTimeout();
    this.failQueue(new Error('loader closed'));
    this.failInFlight(new Error('loader closed'));
    this.socket?.close(1000, 'client shutdown');
    this.socket = null;
    this.setState('closed');
  }

  /** Forget locally-tracked ids. Does NOT clear server state — the API has no
   *  reset route, so restart the backend for a genuinely clean run. */
  resetLocalIds(): void {
    this.sentIds.clear();
  }
}
