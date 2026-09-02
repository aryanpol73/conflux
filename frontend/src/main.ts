/**
 * Application bootstrap and wiring. Orchestration only — no rendering logic,
 * no transport logic, no detection logic.
 */

import './styles/global.css';
import './styles/graph.css';
import './styles/animations.css';

import { DataLoader } from './data/data-loader';
import { ReplayLoadError, replaySource } from './data/replay-source';
import type { DetectionView, GraphMode } from './data/types';
import { ConfluxGraph } from './graph/graph';
import { buildGraph, countDerivedLinks } from './graph/graph-builder';
import { GraphInteractions } from './graph/graph-interactions';
import { StoryController } from './story/story-controller';
import { UIController } from './ui/ui-controller';
import { renderExplanation } from './ui/explain-panel';

/* -- application state (view-level only) ----------------------------- */

let latestDetection: DetectionView | null = null;
let selectedKey: string | null = null;
let mode: GraphMode = 'detection';

/* -- construction ---------------------------------------------------- */

const ui = new UIController({
  onStart: () => void story.start(),
  onPause: () => story.pause(),
  onReset: () => handleReset(),
  onResetView: () => graph.resetView(),
  onModeChange: (next) => {
    mode = next;
    redrawGraph();
  },
  onCandidateSelected: (key) => selectCandidate(key),
});

const graph = new ConfluxGraph(document.getElementById('graph-container') as HTMLElement);

new GraphInteractions(graph, {
  onSelectCandidate: (key) => selectCandidate(key),
  onSelectTransaction: (node) => {
    if (node === null) {
      selectCandidate(null);
    }
  },
});

const loader = new DataLoader(
  {
    onConnectionState: (state, info) => ui.setConnectionState(state, info),

    onAck: (payload) => {
      if (!payload.scorer_loaded) {
        ui.showBlockingBanner(
          'The backend reports that the frozen scorer reference is not loaded. ' +
            'Detection will fail until the Phase 5.5 artifact is available.'
        );
      }
      if (payload.transactions_in_memory > 0) {
        ui.toast(
          `Backend already holds ${payload.transactions_in_memory} transaction(s) in memory.`,
          'warn'
        );
      }
    },

    onDetection: (view) => {
      latestDetection = view;
      ui.renderCandidates(view);
      redrawGraph();
      refreshStats();

      const derived = countDerivedLinks(
        buildGraph(story.sentTransactions(), view, { mode: 'detection' })
      );
      story.noteDetection(view, derived);

      // Keep the open investigation panel in sync as candidates evolve.
      if (selectedKey) {
        ui.renderCandidateDetail(ui.findCandidate(selectedKey), (id) => story.findSent(id));
        renderExplanation(ui.findCandidate(selectedKey));
      }
    },

    onServerError: (error) => {
      if (error.blocking) {
        ui.showBlockingBanner(`${error.code}: ${error.message}`);
      } else {
        const detail = error.details.length
          ? ` (${error.details.map((d) => `${d.field}: ${d.message}`).join('; ')})`
          : '';
        ui.toast(`${error.code}: ${error.message}${detail}`, 'error');
      }
    },

    onQueueChange: (depth, inFlight) => {
      ui.setQueueIndicator(depth, inFlight);
      ui.setAnalyzing(inFlight);
    },

    onWarning: (message) => ui.toast(message, 'warn'),
  },
  { requestTimeoutMs: 45_000 }
);

const story = new StoryController(
  loader,
  replaySource,
  {
    onStateChange: (state) => ui.setReplayState(state),
    onBeat: (beat) => ui.setBeat(beat),
    onTransactionSent: (tx) => {
      ui.pushActivity(tx);
      ui.setGraphEmpty(false);
      redrawGraph(); // nodes accumulate between detection replies
      refreshStats();
    },
    onProgress: (sent, total) => ui.setProgress(sent, total),
    onNotice: (message, severity) => ui.toast(message, severity),
    onFinished: () => ui.toast('Replay complete.', 'info'),
  },
  { warmupCount: 100, liveIntervalMs: 120, budget: 500 }
);

/* -- behaviour ------------------------------------------------------- */

function redrawGraph(): void {
  const model = buildGraph(story.sentTransactions(), latestDetection, {
    mode,
    focusKey: selectedKey,
    maxTransactionNodes: 130,
  });
  graph.setModel(model);
  ui.setGraphEmpty(model.nodes.length === 0);
  graph.highlightCandidate(mode === 'investigation' ? null : selectedKey);
}

function refreshStats(): void {
  ui.setStats({
    sent: story.sentCount,
    remaining: replaySource.isLoaded ? replaySource.remaining : 0,
    candidates: latestDetection?.candidates.length ?? 0,
    scored: latestDetection?.candidates[0]?.totalScoredInRun ?? 0,
  });
}

function selectCandidate(key: string | null): void {
  selectedKey = key;
  ui.setSelectedCandidate(key);

  if (key === null) {
    ui.renderCandidateDetail(null, (id) => story.findSent(id));
    renderExplanation(null);
    graph.highlightCandidate(null);
    if (mode === 'investigation') redrawGraph();
    return;
  }

  const candidate = ui.findCandidate(key);
  ui.renderCandidateDetail(candidate, (id) => story.findSent(id));
  renderExplanation(candidate);
  story.noteInvestigation();
  if (candidate && candidate.score !== null) story.noteScoreView();

  if (mode === 'investigation') {
    redrawGraph();
  } else {
    graph.highlightCandidate(key);
  }
  graph.focusCandidate(key);
}

async function handleReset(): Promise<void> {
  await story.reset();

  latestDetection = null;
  selectedKey = null;
  graph.clear();
  ui.clearActivity();
  ui.setSelectedCandidate(null);
  ui.renderCandidateDetail(null, (id) => story.findSent(id));
  renderExplanation(null);
  ui.setGraphEmpty(true);
  ui.setBeat('idle');
  refreshStats();
}

/* -- boot ------------------------------------------------------------ */

async function boot(): Promise<void> {
  ui.setConnectionState('idle');
  ui.setBeat('idle');
  ui.setGraphEmpty(true);
  ui.setReplayState('idle');

  // Replay data and the socket are independent; a failure in one must not
  // prevent the other, and neither may take the UI down.
  try {
    await story.load();
    const report = replaySource.loadReport;
    const meta = replaySource.provenance;
    ui.setProvenance(
      `${report?.accepted ?? 0} transactions · ${meta.source_dataset ?? 'replay population'}` +
        (meta.selection ? ` · ${meta.selection}` : '')
    );
    refreshStats();
  } catch (error) {
    const message =
      error instanceof ReplayLoadError
        ? error.message
        : `replay population failed to load: ${String(error)}`;
    ui.setProvenance('replay population unavailable');
    ui.showBlockingBanner(message);
  }

  try {
    await loader.connect();
  } catch (error) {
    ui.toast(`WebSocket connection failed: ${String(error)}`, 'error');
  }

  // Show whatever the backend already holds, so the console is never blank.
  try {
    const initial = await loader.fetchCampaigns();
    latestDetection = initial;
    ui.renderCandidates(initial);
    refreshStats();
  } catch {
    // Backend offline or scorer unavailable; the UI stays usable either way.
  }
}

window.addEventListener('beforeunload', () => loader.close());

void boot();
