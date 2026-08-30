/**
 * All DOM rendering outside the graph surface: header, act banner, status,
 * counters, activity stream, candidate list, investigation panel, controls,
 * toasts and the blocking banner.
 *
 * Displays only values the backend actually returned. Where a field is absent,
 * it renders an explicit empty state rather than a plausible-looking number.
 * Nothing here computes a score, a probability, or an explanation.
 */

import {
  maskIdentifier,
  type ActivityItem,
  type AppStats,
  type CandidateView,
  type ConnectionState,
  type DetectionView,
  type GraphMode,
  type ReplayState,
  type StoryBeat,
  type TransactionRecord,
} from '../data/types';

export interface UICallbacks {
  onStart?(): void;
  onPause?(): void;
  onReset?(): void;
  onModeChange?(mode: GraphMode): void;
  onResetView?(): void;
  onCandidateSelected?(key: string): void;
}

interface ActMeta {
  label: string;
  title: string;
  description: string;
}

/** Deliberately restrained language: candidate, structure, signal. Not "fraud". */
const ACTS: Record<StoryBeat, ActMeta> = {
  idle: {
    label: 'STANDBY',
    title: 'Awaiting replay',
    description: 'Load the replay population and press START to begin ingestion.',
  },
  warmup: {
    label: 'WARM START',
    title: 'Seeding the population',
    description:
      'Bulk-ingesting a prefix so the backend has enough population for structure to exist.',
  },
  observe: {
    label: 'ACT I · OBSERVE',
    title: 'Independent activity',
    description:
      'Transactions arrive one at a time. Nothing here is labelled. Each looks ordinary in isolation.',
  },
  link: {
    label: 'ACT II · LINK',
    title: 'Shared entities appear',
    description:
      'Transactions begin sharing cards, devices and IP signatures. These links are read from the replayed fields, not from detection output.',
  },
  converge: {
    label: 'ACT III · CONVERGE',
    title: 'Coordinated structure observed',
    description:
      'The backend has grouped transactions into a candidate. Ordinary-looking activity becomes notable through its collective structure.',
  },
  investigate: {
    label: 'ACT IV · INVESTIGATE',
    title: 'Candidate under inspection',
    description:
      'Inspecting the evidence the backend returned for this candidate. Members, tier, action and top signals.',
  },
  score: {
    label: 'ACT V · RISK POSITION',
    title: 'Where this candidate sits',
    description:
      'The backend risk score and its rank among scored candidates in this run. A ranking signal, not a probability of fraud.',
  },
};

const MAX_ACTIVITY_ROWS = 40;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing DOM element #${id}`);
  return node as T;
}

export class UIController {
  private readonly activity: ActivityItem[] = [];
  private candidates: CandidateView[] = [];
  private selectedKey: string | null = null;
  private mode: GraphMode = 'detection';
  private toastSeq = 0;
  private readonly callbacks: UICallbacks;

  constructor(callbacks: UICallbacks = {}) {
    this.callbacks = callbacks;
    el('btn-start').addEventListener('click', () => this.callbacks.onStart?.());
    el('btn-pause').addEventListener('click', () => this.callbacks.onPause?.());
    el('btn-reset').addEventListener('click', () => this.callbacks.onReset?.());
    el('btn-reset-view').addEventListener('click', () => this.callbacks.onResetView?.());
    el('blocking-banner-dismiss').addEventListener('click', () => this.hideBlockingBanner());

    document.querySelectorAll<HTMLButtonElement>('.mode-btn').forEach((button) => {
      button.addEventListener('click', () => {
        const mode = (button.dataset.mode as GraphMode) ?? 'detection';
        this.setMode(mode);
        this.callbacks.onModeChange?.(mode);
      });
    });
  }

  /* -- header / status ----------------------------------------------- */

  setConnectionState(state: ConnectionState, info?: string): void {
    const wrapper = el('connection-status');
    const text = el('status-text');
    wrapper.dataset.state = state;
    const labels: Record<ConnectionState, string> = {
      idle: 'IDLE',
      connecting: 'CONNECTING',
      open: 'LIVE',
      reconnecting: 'RECONNECTING',
      closed: 'DISCONNECTED',
      failed: 'BACKEND OFFLINE',
    };
    text.textContent = labels[state];
    wrapper.title = info ?? '';
  }

  setAnalyzing(active: boolean): void {
    el('analyzing-indicator').classList.toggle('hidden', !active);
  }

  setQueueIndicator(depth: number, inFlight: boolean): void {
    const node = el('queue-indicator');
    if (!inFlight && depth === 0) {
      node.textContent = 'idle';
      node.dataset.busy = 'false';
      return;
    }
    node.textContent = depth > 0 ? `detecting · ${depth} queued` : 'detecting';
    node.dataset.busy = 'true';
  }

  setBeat(beat: StoryBeat): void {
    const act = ACTS[beat];
    const block = el('act-block');
    el('act-label').textContent = act.label;
    el('act-title').textContent = act.title;
    el('act-description').textContent = act.description;
    block.dataset.beat = beat;
    block.classList.remove('act-flip');
    void block.offsetWidth; // restart the animation
    block.classList.add('act-flip');
  }

  setReplayState(state: ReplayState): void {
    const start = el<HTMLButtonElement>('btn-start');
    const pause = el<HTMLButtonElement>('btn-pause');
    start.disabled = state === 'running' || state === 'loading';
    pause.disabled = state !== 'running';
    start.textContent = state === 'paused' ? 'RESUME' : 'START';
  }

  /* -- counters ------------------------------------------------------ */

  setStats(stats: AppStats): void {
    el('stat-sent').textContent = String(stats.sent);
    el('stat-remaining').textContent = String(stats.remaining);
    el('stat-candidates').textContent = String(stats.candidates);
    el('stat-scored').textContent = String(stats.scored);
  }

  setProgress(sent: number, total: number): void {
    const pct = total > 0 ? Math.min(100, (sent / total) * 100) : 0;
    el('replay-progress-fill').style.width = `${pct.toFixed(1)}%`;
  }

  setProvenance(text: string): void {
    el('replay-provenance').textContent = text;
  }

  /* -- activity stream ----------------------------------------------- */

  pushActivity(tx: TransactionRecord): void {
    this.activity.unshift({
      transactionId: tx.transaction_id,
      merchant: tx.merchant_id,
      amount: tx.amount,
      outcome: tx.auth_outcome,
      at: Date.now(),
    });
    if (this.activity.length > MAX_ACTIVITY_ROWS) this.activity.length = MAX_ACTIVITY_ROWS;

    const list = el<HTMLUListElement>('activity-list');
    const row = document.createElement('li');
    row.className = 'activity-row entering';
    const declined = /declin|fail|reject/i.test(tx.auth_outcome);
    row.innerHTML =
      `<span class="a-merchant">${escapeHtml(tx.merchant_id)}</span>` +
      `<span class="a-amount">${tx.amount.toFixed(2)}</span>` +
      `<span class="a-outcome ${declined ? 'declined' : 'approved'}">` +
      `${escapeHtml(tx.auth_outcome)}</span>`;
    row.title = tx.transaction_id;
    list.prepend(row);
    window.setTimeout(() => row.classList.remove('entering'), 500);
    while (list.children.length > MAX_ACTIVITY_ROWS) list.lastElementChild?.remove();
  }

  clearActivity(): void {
    this.activity.length = 0;
    el('activity-list').innerHTML = '';
  }

  /* -- graph stage --------------------------------------------------- */

  setGraphEmpty(empty: boolean): void {
    el('graph-empty').classList.toggle('hidden', !empty);
  }

  setMode(mode: GraphMode): void {
    this.mode = mode;
    document.querySelectorAll<HTMLButtonElement>('.mode-btn').forEach((button) => {
      button.classList.toggle('active', button.dataset.mode === mode);
    });
  }

  get currentMode(): GraphMode {
    return this.mode;
  }

  /* -- candidates ---------------------------------------------------- */

  renderCandidates(view: DetectionView): void {
    this.candidates = view.candidates;
    const list = el<HTMLUListElement>('candidate-list');
    const empty = el('candidate-empty');

    empty.classList.toggle('hidden', view.candidates.length > 0);
    list.innerHTML = '';

    for (const candidate of view.candidates) {
      const item = document.createElement('li');
      item.className = `candidate-row band-${candidate.band}`;
      item.dataset.key = candidate.key;
      if (candidate.key === this.selectedKey) item.classList.add('selected');

      const scoreText = candidate.score === null ? '—' : candidate.score.toFixed(3);
      const tierText = candidate.tier || 'untiered';

      item.innerHTML =
        `<div class="c-head"><span class="c-id">${escapeHtml(
          candidate.candidateId || 'candidate'
        )}</span><span class="c-score">${scoreText}</span></div>` +
        `<div class="c-meta"><span class="c-tier">${escapeHtml(tierText)}</span>` +
        `<span class="c-members">${candidate.transactionIds.length} tx</span></div>`;

      item.addEventListener('click', () => this.callbacks.onCandidateSelected?.(candidate.key));
      list.appendChild(item);
    }
  }

  setSelectedCandidate(key: string | null): void {
    this.selectedKey = key;
    document.querySelectorAll<HTMLLIElement>('.candidate-row').forEach((row) => {
      row.classList.toggle('selected', row.dataset.key === key);
    });
  }

  /**
   * Investigation panel. Every value shown is a field the backend sent.
   * Contribution bars are scaled to the largest magnitude within this
   * candidate and are labelled as the backend's own contribution values —
   * they are not SHAP, and no causal claim is made.
   */
  renderCandidateDetail(
    candidate: CandidateView | null,
    resolve: (id: string) => TransactionRecord | undefined
  ): void {
    const panel = el('candidate-detail');

    if (!candidate) {
      panel.innerHTML =
        '<div class="empty-state">Select a candidate to inspect backend evidence.</div>';
      return;
    }

    const scoreText = candidate.score === null ? 'not scored' : candidate.score.toFixed(4);
    const rankText =
      candidate.score === null || candidate.totalScoredInRun === 0
        ? 'unranked'
        : `rank ${candidate.rank} of ${candidate.totalScoredInRun} scored in this run`;

    const parts: string[] = [];

    parts.push(
      `<div class="detail-head band-${candidate.band}">` +
        `<div class="detail-id">${escapeHtml(candidate.candidateId || 'candidate')}</div>` +
        `<div class="detail-score">${scoreText}</div>` +
        `<div class="detail-scorelabel">RISK SCORE · backend</div>` +
        `<div class="detail-rank">${rankText}</div>` +
        '</div>'
    );

    parts.push(
      '<div class="detail-grid">' +
        `<div><span class="dg-label">tier</span><span class="dg-value">${escapeHtml(
          candidate.tier || '—'
        )}</span></div>` +
        `<div><span class="dg-label">action</span><span class="dg-value">${escapeHtml(
          candidate.action || '—'
        )}</span></div>` +
        `<div><span class="dg-label">members</span><span class="dg-value">${candidate.transactionIds.length}</span></div>` +
        '</div>'
    );

    // Top signals.
    const signals = candidate.topSignals.filter((s) => s.contribution !== null);
    if (candidate.topSignals.length === 0) {
      parts.push(
        '<div class="detail-section"><h3>Top signals</h3>' +
          '<div class="empty-state">The backend returned no top_signals for this candidate.</div></div>'
      );
    } else {
      const maxAbs = Math.max(...signals.map((s) => Math.abs(s.contribution ?? 0)), 1e-9);
      const rows = candidate.topSignals
        .map((signal) => {
          const value = signal.contribution;
          if (value === null || value === undefined) {
            return (
              `<li class="signal-row"><span class="s-name">${escapeHtml(signal.feature)}</span>` +
              '<span class="s-none">no contribution value returned</span></li>'
            );
          }
          const pct = (Math.abs(value) / maxAbs) * 100;
          return (
            `<li class="signal-row"><span class="s-name">${escapeHtml(signal.feature)}</span>` +
            `<span class="s-bar"><i style="width:${pct.toFixed(1)}%"></i></span>` +
            `<span class="s-value">${value.toFixed(4)}</span></li>`
          );
        })
        .join('');
      parts.push(
        '<div class="detail-section"><h3>Contribution to score</h3>' +
          `<ul class="signal-list">${rows}</ul>` +
          '<p class="detail-note">Values as returned by the backend scorer. ' +
          'Bar length is relative magnitude within this candidate.</p></div>'
      );
    }

    // Members.
    const memberRows = candidate.transactionIds
      .slice(0, 40)
      .map((id) => {
        const tx = resolve(id);
        if (!tx) {
          return `<li class="member-row unknown"><span>${escapeHtml(id)}</span>` +
            '<span class="m-note">not in local cache</span></li>';
        }
        return (
          `<li class="member-row"><span class="m-id">${escapeHtml(tx.transaction_id)}</span>` +
          `<span class="m-merchant">${escapeHtml(tx.merchant_id)}</span>` +
          `<span class="m-amount">${tx.amount.toFixed(2)}</span>` +
          `<span class="m-card">${escapeHtml(maskIdentifier(tx.card_fingerprint, 6))}</span></li>`
        );
      })
      .join('');
    const overflow =
      candidate.transactionIds.length > 40
        ? `<p class="detail-note">${candidate.transactionIds.length - 40} further member(s) not shown.</p>`
        : '';
    parts.push(
      `<div class="detail-section"><h3>Member transactions</h3>` +
        `<ul class="member-list">${memberRows}</ul>${overflow}</div>`
    );

    panel.innerHTML = parts.join('');
    panel.scrollTop = 0;
  }

  findCandidate(key: string): CandidateView | null {
    return this.candidates.find((c) => c.key === key) ?? null;
  }

  /* -- messaging ----------------------------------------------------- */

  showBlockingBanner(message: string): void {
    el('blocking-banner-text').textContent = message;
    el('blocking-banner').classList.remove('hidden');
  }

  hideBlockingBanner(): void {
    el('blocking-banner').classList.add('hidden');
  }

  toast(message: string, severity: 'info' | 'warn' | 'error' = 'info'): void {
    const container = el('toast-container');
    const id = `toast-${(this.toastSeq += 1)}`;
    const node = document.createElement('div');
    node.className = `toast ${severity}`;
    node.id = id;
    node.textContent = message;
    container.appendChild(node);
    window.setTimeout(() => {
      node.classList.add('leaving');
      window.setTimeout(() => node.remove(), 320);
    }, severity === 'error' ? 8000 : 4500);
  }
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
