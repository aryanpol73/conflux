/**
 * Replay sequencing: start, pause, reset, pacing, and narrative progression.
 *
 * Pacing exists because the backend re-runs the full pipeline over the whole
 * accumulated population on every ingest. The controller therefore awaits each
 * detection reply before sending the next transaction — natural back-pressure,
 * one request in flight, no flooding. The UI stays interactive throughout
 * because sends are awaited, not blocking.
 *
 * A warm start bulk-ingests a small prefix so the graph has real structure
 * before the visible live stream begins.
 *
 * The controller owns the cache of transactions this frontend has sent, which
 * is the only transaction data the UI can display: the backend returns ids, not
 * records.
 */

import type { DataLoader } from '../data/data-loader';
import type { ReplaySource } from '../data/replay-source';
import type { DetectionView, ReplayState, StoryBeat, TransactionRecord } from '../data/types';

export interface StoryCallbacks {
  onStateChange?(state: ReplayState): void;
  onBeat?(beat: StoryBeat): void;
  onTransactionSent?(tx: TransactionRecord, sentCount: number): void;
  onProgress?(sent: number, total: number): void;
  onNotice?(message: string, severity: 'info' | 'warn' | 'error'): void;
  onFinished?(): void;
}

export interface StoryOptions {
  /** Transactions ingested before the visible stream starts. */
  warmupCount?: number;
  /** Minimum gap between live sends. Actual pace is max(this, backend latency). */
  liveIntervalMs?: number;
  /** Stop after this many total sends, regardless of population size. */
  budget?: number;
}

export class StoryController {
  private state: ReplayState = 'idle';
  private beat: StoryBeat = 'idle';
  private running = false;
  private aborted = false;

  /** Every transaction this frontend has successfully sent, in order. */
  private readonly sent: TransactionRecord[] = [];

  private readonly warmupCount: number;
  private readonly liveIntervalMs: number;
  private readonly budget: number;

  private consecutiveFailures = 0;

  private readonly loader: DataLoader;
  private readonly replay: ReplaySource;
  private readonly callbacks: StoryCallbacks;

  constructor(
    loader: DataLoader,
    replay: ReplaySource,
    callbacks: StoryCallbacks = {},
    options: StoryOptions = {}
  ) {
    this.loader = loader;
    this.replay = replay;
    this.callbacks = callbacks;
    this.warmupCount = options.warmupCount ?? 60;
    this.liveIntervalMs = options.liveIntervalMs ?? 320;
    this.budget = options.budget ?? 300;
  }

  /* -- accessors ----------------------------------------------------- */

  get replayState(): ReplayState {
    return this.state;
  }

  get currentBeat(): StoryBeat {
    return this.beat;
  }

  get sentCount(): number {
    return this.sent.length;
  }

  /** Transactions the frontend holds locally, for graph and panel rendering. */
  sentTransactions(): readonly TransactionRecord[] {
    return this.sent;
  }

  findSent(transactionId: string): TransactionRecord | undefined {
    return this.sent.find((tx) => tx.transaction_id === transactionId);
  }

  /* -- lifecycle ----------------------------------------------------- */

  async load(): Promise<void> {
    this.setState('loading');
    try {
      const report = await this.replay.load();
      this.setState('ready');
      this.callbacks.onProgress?.(0, Math.min(report.accepted, this.budget));
      if (report.skipped.length > 0) {
        this.callbacks.onNotice?.(
          `${report.skipped.length} replay row(s) skipped during validation.`,
          'warn'
        );
      }
      if (report.truncatedTo !== null) {
        this.callbacks.onNotice?.(
          `Replay population truncated to ${report.truncatedTo} transactions.`,
          'info'
        );
      }
    } catch (error) {
      this.setState('idle');
      throw error;
    }
  }

  async start(): Promise<void> {
    if (this.running) return;
    if (!this.replay.isLoaded) {
      this.callbacks.onNotice?.('Replay population is not loaded.', 'error');
      return;
    }
    if (!this.replay.hasNext) {
      this.callbacks.onNotice?.('Replay population is exhausted. Press RESET.', 'info');
      return;
    }

    // Guard against double-ingest: if the backend still contains transactions,
    // warn the operator to reset the backend before starting a clean replay.
    if (this.sent.length === 0) {
      try {
        const health = await this.loader.fetchHealth();
        if (health.transactions_in_memory > 0) {
            this.callbacks.onNotice?.(
                `Backend already holds ${health.transactions_in_memory} transaction(s). ` +
                'Press RESET before starting a new clean replay.',
                'warn'
            );
        }
      } catch {
        /* health is advisory */
      }
    }

    this.aborted = false;
    this.running = true;
    this.setState('running');
    void this.run();
  }

  pause(): void {
    if (!this.running) return;
    this.running = false;
    this.aborted = true;
    this.setState('paused');
  }

  /**
   * Reset both the frontend replay state and the backend population.
   */
  async reset(): Promise<void> {
    this.running = false;
    this.aborted = true;

    this.setState('loading');

    try {
        // Clear the backend population first so the next replay starts clean.
        await this.loader.resetBackend();

        // Now clear all frontend state.
        this.sent.length = 0;
        this.replay.reset();
        this.loader.resetLocalIds();
        this.consecutiveFailures = 0;

        this.setBeat('idle');
        this.setState(this.replay.isLoaded ? 'ready' : 'idle');

        this.callbacks.onProgress?.(
        0,
        Math.min(this.replay.size, this.budget)
        );

        this.callbacks.onNotice?.(
        'Replay reset successfully. Frontend and backend populations are clean.',
        'info'
        );
    } catch (error) {
        const message =
        error instanceof Error ? error.message : String(error);

        this.setState('paused');

        this.callbacks.onNotice?.(
        `Reset failed: ${message}`,
        'error'
        );
    }
}

  /* -- the replay loop ----------------------------------------------- */

  private async run(): Promise<void> {
    const total = Math.min(this.replay.size, this.budget);

    // Act 0 — warm start.
    if (this.sent.length === 0 && this.warmupCount > 0) {
      this.setBeat('warmup');
      const batch = this.replay.takeWarmup(Math.min(this.warmupCount, this.budget));
      for (const tx of batch) {
        if (this.aborted) break;
        await this.dispatch(tx, total);
      }
    }

    if (!this.aborted) this.setBeat('observe');

    // Live stream.
    while (this.running && !this.aborted && this.replay.hasNext) {
      if (this.sent.length >= this.budget) {
        this.callbacks.onNotice?.(
          `Replay budget of ${this.budget} transactions reached; stopping to keep ` +
            'detection latency bounded.',
          'info'
        );
        break;
      }
      const tx = this.replay.next();
      if (!tx) break;

      const startedAt = performance.now();
      await this.dispatch(tx, total);
      // Pace on the slower of the configured interval and real backend latency.
      const elapsed = performance.now() - startedAt;
      const wait = Math.max(0, this.liveIntervalMs - elapsed);
      if (wait > 0) await this.sleep(wait);
    }

    if (this.running) {
      this.running = false;
      this.setState('done');
      this.callbacks.onFinished?.();
    }
  }

  private async dispatch(tx: TransactionRecord, total: number): Promise<void> {
    try {
      // Awaiting the reply IS the back-pressure: one frame in flight, always.
      await this.loader.sendTransaction(tx);
      this.sent.push(tx);
      this.consecutiveFailures = 0;
      this.callbacks.onTransactionSent?.(tx, this.sent.length);
      this.callbacks.onProgress?.(this.sent.length, total);
    } catch (error) {
      this.consecutiveFailures += 1;
      const message = error instanceof Error ? error.message : String(error);
      this.callbacks.onNotice?.(`${tx.transaction_id}: ${message}`, 'warn');

      // Stop rather than hammer a backend that is clearly not accepting work.
      if (this.consecutiveFailures >= 5) {
        this.running = false;
        this.aborted = true;
        this.setState('paused');
        this.callbacks.onNotice?.(
          'Replay paused after 5 consecutive failures. Check the backend and press START.',
          'error'
        );
      }
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  /* -- narrative ----------------------------------------------------- */

  /**
   * Advance the act based on what has genuinely happened. Acts are never
   * faked on a timer: 'link' requires real shared-entity structure in the
   * replayed data, 'converge' requires a real backend candidate.
   */
  noteDetection(view: DetectionView, derivedLinkCount: number): void {
    if (this.beat === 'investigate' || this.beat === 'score') return;
    if (view.candidates.length > 0) {
      this.setBeat('converge');
    } else if (derivedLinkCount > 0 && this.beat === 'observe') {
      this.setBeat('link');
    }
  }

  noteInvestigation(): void {
    this.setBeat('investigate');
  }

  noteScoreView(): void {
    this.setBeat('score');
  }

  /* -- internals ----------------------------------------------------- */

  private setState(state: ReplayState): void {
    if (this.state === state) return;
    this.state = state;
    this.callbacks.onStateChange?.(state);
  }

  private setBeat(beat: StoryBeat): void {
    if (this.beat === beat) return;
    this.beat = beat;
    this.callbacks.onBeat?.(beat);
  }
}
