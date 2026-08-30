/**
 * Transforms replayed transactions + backend detection state into a graph model.
 *
 * Provenance discipline, enforced structurally:
 *
 *   membership edges  candidate → transaction, straight from
 *                     campaigns[].transaction_ids. BACKEND OUTPUT.
 *
 *   entity edges      transaction ↔ transaction where the two share a card,
 *                     device or IP. Computed here from transactions this
 *                     frontend sent, because the backend returns no edges.
 *                     Flagged backendDerived=false and rendered dashed.
 *
 * This module performs no detection and no scoring. Shared-entity edges are a
 * visual reading of fields already present in the replay data; they are never
 * presented as a detection result.
 */

import {
  maskIdentifier,
  type CandidateView,
  type DetectionView,
  type GraphLink,
  type GraphLinkKind,
  type GraphMode,
  type GraphModel,
  type GraphNode,
  type RiskBand,
  type TransactionRecord,
} from '../data/types';

export interface GraphBuildOptions {
  mode: GraphMode;
  /** Bounded visible window. Candidate members are always retained. */
  maxTransactionNodes?: number;
  /** In investigation mode, restrict the view to this candidate. */
  focusKey?: string | null;
}

const DEFAULT_WINDOW = 130;

/** Entity fields promoted to visual links, in render priority order. */
const ENTITY_FIELDS: ReadonlyArray<{ field: keyof TransactionRecord; kind: GraphLinkKind }> = [
  { field: 'card_fingerprint', kind: 'card' },
  { field: 'device_fingerprint', kind: 'device' },
  { field: 'ip_signature', kind: 'ip' },
];

function makeNode(partial: Partial<GraphNode> & { id: string; kind: GraphNode['kind'] }): GraphNode {
  return {
    label: partial.id,
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    radius: partial.kind === 'candidate' ? 18 : 7,
    fx: null,
    fy: null,
    band: 'unranked' as RiskBand,
    candidateKey: null,
    transaction: null,
    memberCount: 0,
    createdAt: Date.now(),
    ...partial,
  };
}

/**
 * Chain members of a shared-entity group rather than forming a clique.
 * A clique is O(n²) edges and turns any busy entity into an unreadable blob;
 * a chain conveys the same "these are connected" fact at O(n).
 */
function chainGroup(
  members: TransactionRecord[],
  kind: GraphLinkKind,
  value: string,
  out: GraphLink[]
): void {
  if (members.length < 2) return;
  const ordered = [...members].sort(
    (a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp)
  );
  for (let i = 1; i < ordered.length; i += 1) {
    const a = ordered[i - 1].transaction_id;
    const b = ordered[i].transaction_id;
    out.push({
      id: `${kind}:${a}->${b}`,
      source: a,
      target: b,
      kind,
      detail: `${kind} ${maskIdentifier(value)}`,
      backendDerived: false,
    });
  }
}

export function buildGraph(
  transactions: readonly TransactionRecord[],
  detection: DetectionView | null,
  options: GraphBuildOptions
): GraphModel {
  const windowSize = options.maxTransactionNodes ?? DEFAULT_WINDOW;
  const candidates: CandidateView[] = detection?.candidates ?? [];

  /* -- decide which candidates are in scope -------------------------- */

  const inScope =
    options.mode === 'investigation' && options.focusKey
      ? candidates.filter((c) => c.key === options.focusKey)
      : candidates;

  /* -- decide which transactions are visible ------------------------- */

  const byId = new Map(transactions.map((tx) => [tx.transaction_id, tx]));

  // Members of in-scope candidates are always retained, regardless of age.
  const memberIds = new Set<string>();
  for (const candidate of inScope) {
    for (const id of candidate.transactionIds) memberIds.add(id);
  }

  let visible: TransactionRecord[];
  if (options.mode === 'investigation' && options.focusKey) {
    visible = [...memberIds].map((id) => byId.get(id)).filter((t): t is TransactionRecord => !!t);
  } else {
    const recent = transactions.slice(-windowSize);
    const seen = new Set(recent.map((t) => t.transaction_id));
    const pinned = [...memberIds]
      .filter((id) => !seen.has(id))
      .map((id) => byId.get(id))
      .filter((t): t is TransactionRecord => !!t);
    visible = [...pinned, ...recent];
  }

  const visibleIds = new Set(visible.map((t) => t.transaction_id));

  /* -- map transaction -> owning candidate --------------------------- */

  const ownerByTx = new Map<string, CandidateView>();
  for (const candidate of inScope) {
    for (const id of candidate.transactionIds) {
      const existing = ownerByTx.get(id);
      // Highest-scoring candidate wins the colour when a tx appears twice.
      if (!existing || (candidate.score ?? -Infinity) > (existing.score ?? -Infinity)) {
        ownerByTx.set(id, candidate);
      }
    }
  }

  /* -- nodes --------------------------------------------------------- */

  const nodes: GraphNode[] = [];

  for (const tx of visible) {
    const owner = ownerByTx.get(tx.transaction_id) ?? null;
    nodes.push(
      makeNode({
        id: tx.transaction_id,
        kind: 'transaction',
        label: tx.merchant_id,
        radius: owner ? 9 : 6.5,
        band: owner ? owner.band : 'unranked',
        candidateKey: owner ? owner.key : null,
        transaction: tx,
      })
    );
  }

  for (const candidate of inScope) {
    // Only render a candidate hub if at least one member is on screen.
    const present = candidate.transactionIds.filter((id) => visibleIds.has(id));
    if (present.length === 0) continue;
    nodes.push(
      makeNode({
        id: `cand:${candidate.key}`,
        kind: 'candidate',
        label: candidate.candidateId || 'candidate',
        radius: Math.min(26, 13 + Math.sqrt(present.length) * 2.4),
        band: candidate.band,
        candidateKey: candidate.key,
        memberCount: candidate.transactionIds.length,
      })
    );
  }

  /* -- links --------------------------------------------------------- */

  const links: GraphLink[] = [];

  // Backend membership edges.
  for (const candidate of inScope) {
    for (const id of candidate.transactionIds) {
      if (!visibleIds.has(id)) continue;
      links.push({
        id: `member:${candidate.key}:${id}`,
        source: `cand:${candidate.key}`,
        target: id,
        kind: 'membership',
        detail: 'candidate membership (backend)',
        backendDerived: true,
      });
    }
  }

  // Client-derived shared-entity edges.
  for (const { field, kind } of ENTITY_FIELDS) {
    const groups = new Map<string, TransactionRecord[]>();
    for (const tx of visible) {
      const value = String(tx[field] ?? '');
      if (!value) continue;
      const bucket = groups.get(value);
      if (bucket) bucket.push(tx);
      else groups.set(value, [tx]);
    }
    for (const [value, members] of groups) {
      chainGroup(members, kind, value, links);
    }
  }

  return { nodes, links };
}

/** Counts for the legend / act detection, without re-walking the model. */
export function countDerivedLinks(model: GraphModel): number {
  return model.links.reduce((total, link) => total + (link.backendDerived ? 0 : 1), 0);
}
