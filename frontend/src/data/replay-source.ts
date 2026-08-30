/**
 * Preselected replay population for the CONFLUX demo.
 *
 * The backend ingests transactions but never emits them, so the frontend
 * needs its own source for the transactions it will send. That source is one
 * bounded, label-free file exported from the real dataset ahead of time and
 * served as a static asset from public/data/.
 *
 * Responsibility: load and validate the asset, then hand transactions to
 * story-controller.ts sequentially. Nothing else. No WebSocket logic, no
 * detection, no scoring, no grouping, no tiering, no linking, and no reading
 * of `label` or `campaign_id` — their presence is treated as a fatal export
 * defect rather than something to strip and continue past.
 *
 * There is no file picker, no drag-and-drop, and no upload path.
 */

import {
  FORBIDDEN_FIELDS,
  TRANSACTION_FIELDS,
  asFiniteNumber,
  asString,
  isRecordObject,
  type TransactionRecord,
} from './types';
import { sanitizeTransaction, TransactionRejected } from './data-loader';

/* ------------------------------------------------------------------ */
/* Location of the replay asset                                        */
/* ------------------------------------------------------------------ */

export const DEFAULT_REPLAY_URL = '/data/replay-population.json';

/**
 * Hard ceiling on the demo population. The backend re-runs the whole pipeline
 * over the entire in-memory population on every ingest, so total cost grows
 * quadratically across a full replay. This is a safety net; the exporter
 * should already emit something well below it.
 */
export const MAX_REPLAY_TRANSACTIONS = 600;

export function resolveReplayUrl(): string {
  const override = new URLSearchParams(window.location.search).get('replay');
  if (override) return override;
  const env = (import.meta as { env?: Record<string, string> }).env;
  if (env?.VITE_CONFLUX_REPLAY) return env.VITE_CONFLUX_REPLAY;
  return DEFAULT_REPLAY_URL;
}

/* ------------------------------------------------------------------ */
/* Shape of the asset                                                  */
/* ------------------------------------------------------------------ */

/** Provenance recorded by the exporter, displayed so the source is visible. */
export interface ReplayMeta {
  source_dataset?: string;
  generated_at?: string;
  n_transactions?: number;
  /** How the slice was chosen, in plain language. */
  selection?: string;
  labels_removed?: boolean;
  [extra: string]: unknown;
}

export type ReplayAsset =
  | Array<Record<string, unknown>>
  | { meta?: ReplayMeta; transactions: Array<Record<string, unknown>> };

export class ReplayLoadError extends Error {
  readonly cause?: unknown;
  constructor(message: string, cause?: unknown) {
    super(message);
    this.cause = cause;
    this.name = 'ReplayLoadError';
  }
}

export interface ReplayRowIssue {
  index: number;
  transactionId: string;
  field: string;
  reason: string;
}

export interface ReplayLoadReport {
  accepted: number;
  skipped: ReplayRowIssue[];
  truncatedTo: number | null;
  meta: ReplayMeta;
  url: string;
}

/* ------------------------------------------------------------------ */
/* Parsing                                                             */
/* ------------------------------------------------------------------ */

function extractRows(asset: unknown): {
  rows: Array<Record<string, unknown>>;
  meta: ReplayMeta;
} {
  if (Array.isArray(asset)) {
    return { rows: asset.filter(isRecordObject), meta: {} };
  }
  if (isRecordObject(asset) && Array.isArray(asset.transactions)) {
    return {
      rows: asset.transactions.filter(isRecordObject),
      meta: isRecordObject(asset.meta) ? (asset.meta as ReplayMeta) : {},
    };
  }
  throw new ReplayLoadError(
    'replay asset must be a JSON array of transaction objects, or an object ' +
      'with a "transactions" array'
  );
}

/**
 * Fail loudly on ground-truth leakage. Silently stripping `label` would let a
 * mis-generated export pass unnoticed, and the credibility of the demo rests
 * on the frontend never having seen ground truth.
 */
function assertNoGroundTruth(rows: Array<Record<string, unknown>>): void {
  const offenders = new Set<string>();
  for (const row of rows) {
    for (const banned of FORBIDDEN_FIELDS) {
      if (banned in row) offenders.add(banned);
    }
    if (offenders.size === FORBIDDEN_FIELDS.length) break;
  }
  if (offenders.size > 0) {
    throw new ReplayLoadError(
      `replay asset contains ground-truth column(s): ${[...offenders].join(', ')}. ` +
        'Regenerate the export with these columns dropped. The frontend will ' +
        'not load a population that carries labels.'
    );
  }
}

/**
 * Reduce a row to the nine accepted columns and validate it exactly as the API
 * will, so a row that would 422 is dropped here instead of costing a round trip.
 */
function toTransaction(row: Record<string, unknown>): TransactionRecord {
  const trimmed: Record<string, unknown> = {};
  for (const field of TRANSACTION_FIELDS) {
    if (field in row) trimmed[field] = row[field];
  }
  if ('bin' in trimmed) {
    const numeric = asFiniteNumber(trimmed.bin);
    trimmed.bin =
      numeric !== null && Number.isInteger(numeric) ? String(numeric) : asString(trimmed.bin);
  }
  return sanitizeTransaction(trimmed);
}

/* ------------------------------------------------------------------ */
/* ReplaySource                                                        */
/* ------------------------------------------------------------------ */

export interface ReplaySourceOptions {
  url?: string;
  /** Lower of this and MAX_REPLAY_TRANSACTIONS wins. */
  max?: number;
}

/**
 * Bounded, ordered, single-pass cursor over the replay population.
 * story-controller.ts drives it; pacing lives entirely in the controller.
 */
export class ReplaySource {
  private transactions: TransactionRecord[] = [];
  private meta: ReplayMeta = {};
  private cursor = 0;
  private loaded = false;
  private report: ReplayLoadReport | null = null;

  private readonly url: string;
  private readonly max: number;

  constructor(options: ReplaySourceOptions = {}) {
    this.url = options.url ?? resolveReplayUrl();
    this.max = Math.min(options.max ?? MAX_REPLAY_TRANSACTIONS, MAX_REPLAY_TRANSACTIONS);
  }

  /* -- loading ------------------------------------------------------- */

  async load(): Promise<ReplayLoadReport> {
    if (this.loaded && this.report) return this.report;

    let asset: unknown;
    try {
      const response = await fetch(this.url, { headers: { Accept: 'application/json' } });
      if (!response.ok) {
        throw new ReplayLoadError(
          `could not fetch replay population from ${this.url} (HTTP ${response.status}). ` +
            'Generate it into frontend/public/data/ before starting the demo.'
        );
      }
      asset = await response.json();
    } catch (error) {
      if (error instanceof ReplayLoadError) throw error;
      throw new ReplayLoadError(
        `replay population at ${this.url} could not be read as JSON`,
        error
      );
    }

    const { rows, meta } = extractRows(asset);
    assertNoGroundTruth(rows);

    const accepted: TransactionRecord[] = [];
    const skipped: ReplayRowIssue[] = [];
    const seen = new Set<string>();

    rows.forEach((row, index) => {
      let tx: TransactionRecord;
      try {
        tx = toTransaction(row);
      } catch (error) {
        skipped.push({
          index,
          transactionId: asString(row.transaction_id),
          field: error instanceof TransactionRejected ? error.field : '*',
          reason: (error as Error).message,
        });
        return;
      }
      // The backend appends unconditionally and never deduplicates, so a
      // repeated id would silently inflate the server-side population.
      if (seen.has(tx.transaction_id)) {
        skipped.push({
          index,
          transactionId: tx.transaction_id,
          field: 'transaction_id',
          reason: 'duplicate transaction_id in replay asset',
        });
        return;
      }
      seen.add(tx.transaction_id);
      accepted.push(tx);
    });

    if (accepted.length === 0) {
      throw new ReplayLoadError(
        `replay population at ${this.url} yielded no usable transactions ` +
          `(${rows.length} row(s) read, ${skipped.length} rejected). ` +
          'Check the exporter output against the nine production columns.'
      );
    }

    // Chronological, so the replay reads as a genuine arrival stream.
    accepted.sort((a, b) => {
      const delta = Date.parse(a.timestamp) - Date.parse(b.timestamp);
      return delta !== 0 ? delta : a.transaction_id.localeCompare(b.transaction_id);
    });

    const truncatedTo = accepted.length > this.max ? this.max : null;
    this.transactions = truncatedTo === null ? accepted : accepted.slice(0, this.max);

    this.meta = meta;
    this.cursor = 0;
    this.loaded = true;
    this.report = {
      accepted: this.transactions.length,
      skipped,
      truncatedTo,
      meta,
      url: this.url,
    };
    return this.report;
  }

  /* -- inspection ---------------------------------------------------- */

  get isLoaded(): boolean {
    return this.loaded;
  }

  get size(): number {
    return this.transactions.length;
  }

  get position(): number {
    return this.cursor;
  }

  get remaining(): number {
    return this.transactions.length - this.cursor;
  }

  get hasNext(): boolean {
    return this.cursor < this.transactions.length;
  }

  get provenance(): ReplayMeta {
    return { ...this.meta };
  }

  get loadReport(): ReplayLoadReport | null {
    return this.report;
  }

  all(): readonly TransactionRecord[] {
    return this.transactions;
  }

  /** Look up a transaction the frontend already holds, by id. */
  find(transactionId: string): TransactionRecord | undefined {
    return this.transactions.find((tx) => tx.transaction_id === transactionId);
  }

  private assertLoaded(): void {
    if (!this.loaded) {
      throw new ReplayLoadError('ReplaySource.load() must complete before reading transactions');
    }
  }

  /* -- sequential access --------------------------------------------- */

  next(): TransactionRecord | null {
    this.assertLoaded();
    if (!this.hasNext) return null;
    const tx = this.transactions[this.cursor];
    this.cursor += 1;
    return tx;
  }

  peek(): TransactionRecord | null {
    this.assertLoaded();
    return this.hasNext ? this.transactions[this.cursor] : null;
  }

  /** Bulk slice for the warm start. */
  takeWarmup(count: number): TransactionRecord[] {
    this.assertLoaded();
    const size = Math.max(0, Math.min(count, this.remaining));
    const batch = this.transactions.slice(this.cursor, this.cursor + size);
    this.cursor += size;
    return batch;
  }

  /** Rewind the cursor. Does not re-fetch and does not touch backend state. */
  reset(): void {
    this.cursor = 0;
  }
}

/** Shared instance for the demo. */
export const replaySource = new ReplaySource();
