/**
 * Plain-English explanation block for the investigation panel.
 *
 * Deliberately isolated: this module only ever appends its own element to the
 * existing panel and removes it again. It never rewrites markup produced by
 * UIController, so the panel's existing layout, classes and styling are
 * untouched.
 *
 * No wording, percentile or ordering is computed here. Everything comes from
 * the backend explain layer, which inverts the deterministic scorer's own
 * weighted contributions. The UI only formats what it was sent.
 */

import '../styles/explain.css';
import type { CampaignExplanation, CandidateView } from '../data/types';

const PANEL_ID = 'candidate-detail';
const BLOCK_ID = 'cfx-explain-block';

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function clampPercent(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : 0;
}

function signalRows(explanation: CampaignExplanation): string {
  const signals = Array.isArray(explanation.all_signals) ? explanation.all_signals : [];
  return signals
    .map((signal) => {
      const sentence = text(signal?.sentence);
      if (!sentence) return '';
      const pct = clampPercent(signal?.suspicion_percentile);
      const display = text(signal?.percentile_display) || `${Math.round(pct * 100)}%`;
      const strong = signal?.is_strong === true ? ' cfx-strong' : '';
      return (
        `<li class="cfx-row${strong}">` +
        `<span class="cfx-pct">${escapeHtml(display)}</span>` +
        `<span class="cfx-bar"><i style="width:${(pct * 100).toFixed(1)}%"></i></span>` +
        `<span class="cfx-sentence">${escapeHtml(sentence)}</span>` +
        '</li>'
      );
    })
    .join('');
}

function buildHtml(explanation: CampaignExplanation): string {
  const rows = signalRows(explanation);
  const verdict = text(explanation.verdict);
  if (!rows && !verdict) return '';

  const summary = text(explanation.summary);
  const caveat = text(explanation.caveat);

  return (
    '<h3>What this means</h3>' +
    (summary ? `<p class="cfx-summary">${escapeHtml(summary)}</p>` : '') +
    (verdict ? `<p class="cfx-verdict">${escapeHtml(verdict)}</p>` : '') +
    (rows ? `<ul class="cfx-list">${rows}</ul>` : '') +
    (caveat ? `<p class="cfx-caveat">${escapeHtml(caveat)}</p>` : '')
  );
}

/**
 * Call immediately after UIController.renderCandidateDetail(), which replaces
 * the panel's innerHTML and would otherwise discard this block.
 *
 * A null candidate, a backend without the explain layer, or an empty
 * explanation all result in no element at all — never an empty box.
 */
export function renderExplanation(candidate: CandidateView | null | undefined): void {
  const panel = document.getElementById(PANEL_ID);
  if (!panel) return;

  document.getElementById(BLOCK_ID)?.remove();

  const explanation = candidate?.explanation;
  if (!explanation) return;

  const html = buildHtml(explanation);
  if (!html) return;

  const block = document.createElement('div');
  block.id = BLOCK_ID;
  // Reuses the panel's existing section class read-only, so spacing and
  // typography match. This module defines no rule for .detail-section.
  block.className = 'detail-section cfx-explain';
  block.innerHTML = html;
  panel.appendChild(block);
}
