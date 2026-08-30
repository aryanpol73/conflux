/**
 * Pointer interaction layer: hover, select, drag, pan, zoom.
 *
 * Owns no rendering and no data. It reads the DOM markers written by graph.ts
 * (`data-node-id`, `data-link-id`), calls back into the graph for visual
 * effect, and notifies the application when a selection changes.
 */

import { maskIdentifier, type GraphNode } from '../data/types';
import type { ConfluxGraph } from './graph';

export interface GraphInteractionCallbacks {
  /** A transaction node was clicked, or null when the background was clicked. */
  onSelectTransaction?(node: GraphNode | null): void;
  /** A candidate hub was clicked. */
  onSelectCandidate?(candidateKey: string): void;
}

const DRAG_THRESHOLD_PX = 4;

export class GraphInteractions {
  private draggingId: string | null = null;
  private panning = false;
  private moved = false;
  private lastX = 0;
  private lastY = 0;

  private readonly graph: ConfluxGraph;
  private readonly callbacks: GraphInteractionCallbacks;

  constructor(graph: ConfluxGraph, callbacks: GraphInteractionCallbacks) {
    this.graph = graph;
    this.callbacks = callbacks;
    const svg = graph.svg;
    svg.addEventListener('pointerdown', this.onPointerDown);
    svg.addEventListener('pointermove', this.onPointerMove);
    svg.addEventListener('pointerup', this.onPointerUp);
    svg.addEventListener('pointerleave', this.onPointerLeave);
    svg.addEventListener('wheel', this.onWheel, { passive: false });
  }

  private nodeIdFrom(event: PointerEvent): string | null {
    const target = event.target as Element | null;
    return target?.closest('[data-node-id]')?.getAttribute('data-node-id') ?? null;
  }

  private linkIdFrom(event: PointerEvent): string | null {
    const target = event.target as Element | null;
    return target?.closest('[data-link-id]')?.getAttribute('data-link-id') ?? null;
  }

  private onPointerDown = (event: PointerEvent): void => {
    this.moved = false;
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    const nodeId = this.nodeIdFrom(event);
    if (nodeId) {
      this.draggingId = nodeId;
    } else {
      this.panning = true;
    }
    this.graph.svg.setPointerCapture(event.pointerId);
  };

  private onPointerMove = (event: PointerEvent): void => {
    const dx = event.clientX - this.lastX;
    const dy = event.clientY - this.lastY;
    if (Math.abs(dx) > DRAG_THRESHOLD_PX || Math.abs(dy) > DRAG_THRESHOLD_PX) {
      this.moved = true;
    }

    if (this.draggingId) {
      const world = this.graph.clientToWorld(event.clientX, event.clientY);
      this.graph.pinNode(this.draggingId, world.x, world.y);
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      return;
    }

    if (this.panning) {
      this.graph.panBy(dx, dy);
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      return;
    }

    this.updateHover(event);
  };

  private updateHover(event: PointerEvent): void {
    const nodeId = this.nodeIdFrom(event);
    if (nodeId) {
      const node = this.graph.getNode(nodeId);
      this.graph.setHover(nodeId);
      if (node) this.graph.showTooltip(this.describeNode(node), event.clientX, event.clientY);
      return;
    }

    const linkId = this.linkIdFrom(event);
    if (linkId) {
      const link = this.graph.findLink(linkId);
      this.graph.setHover(null);
      if (link) {
        const provenance = link.backendDerived
          ? '<span class="tt-tag backend">backend</span>'
          : '<span class="tt-tag derived">derived locally</span>';
        this.graph.showTooltip(
          `<div class="tt-title">${link.kind}</div>` +
            `<div class="tt-row">${link.detail}</div>${provenance}`,
          event.clientX,
          event.clientY
        );
      }
      return;
    }

    this.graph.setHover(null);
    this.graph.hideTooltip();
  }

  private describeNode(node: GraphNode): string {
    if (node.kind === 'candidate') {
      return (
        `<div class="tt-title">${node.label}</div>` +
        `<div class="tt-row">${node.memberCount} member transaction(s)</div>` +
        `<div class="tt-row">tier band: ${node.band}</div>` +
        '<span class="tt-tag backend">backend candidate</span>'
      );
    }
    const tx = node.transaction;
    if (!tx) {
      return `<div class="tt-title">${node.id}</div><div class="tt-row">not in local cache</div>`;
    }
    return (
      `<div class="tt-title">${tx.transaction_id}</div>` +
      `<div class="tt-row">merchant ${tx.merchant_id}</div>` +
      `<div class="tt-row">amount ${tx.amount.toFixed(2)} · ${tx.auth_outcome}</div>` +
      `<div class="tt-row">card ${maskIdentifier(tx.card_fingerprint)}</div>` +
      `<div class="tt-row">device ${maskIdentifier(tx.device_fingerprint)}</div>` +
      `<div class="tt-row">ip ${maskIdentifier(tx.ip_signature)}</div>`
    );
  }

  private onPointerUp = (event: PointerEvent): void => {
    const wasDragging = this.draggingId;
    const dragged = this.moved;

    if (wasDragging) {
      // A click without movement should not leave the node pinned.
      if (!dragged) this.graph.releaseNode(wasDragging);
      this.draggingId = null;
    }
    this.panning = false;

    try {
      this.graph.svg.releasePointerCapture(event.pointerId);
    } catch {
      /* pointer already released */
    }

    if (dragged) return;

    const nodeId = this.nodeIdFrom(event);
    if (!nodeId) {
      this.graph.setSelection(null);
      this.callbacks.onSelectTransaction?.(null);
      return;
    }

    const node = this.graph.getNode(nodeId);
    if (!node) return;

    this.graph.setSelection(nodeId);
    if (node.kind === 'candidate' && node.candidateKey) {
      this.callbacks.onSelectCandidate?.(node.candidateKey);
    } else if (node.candidateKey) {
      // Clicking a member transaction opens its candidate too — that is the
      // question a user is actually asking.
      this.callbacks.onSelectCandidate?.(node.candidateKey);
      this.callbacks.onSelectTransaction?.(node);
    } else {
      this.callbacks.onSelectTransaction?.(node);
    }
  };

  private onPointerLeave = (): void => {
    this.graph.setHover(null);
    this.graph.hideTooltip();
    this.draggingId = null;
    this.panning = false;
  };

  private onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    this.graph.zoomAt(event.clientX, event.clientY, factor);
  };

  destroy(): void {
    const svg = this.graph.svg;
    svg.removeEventListener('pointerdown', this.onPointerDown);
    svg.removeEventListener('pointermove', this.onPointerMove);
    svg.removeEventListener('pointerup', this.onPointerUp);
    svg.removeEventListener('pointerleave', this.onPointerLeave);
    svg.removeEventListener('wheel', this.onWheel);
  }
}
