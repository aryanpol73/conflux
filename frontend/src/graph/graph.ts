/**
 * Graph rendering surface and lifecycle.
 *
 * SVG rendering plus a small velocity-Verlet force layout written inline: at a
 * bounded window of ~150 nodes the O(n²) repulsion is comfortably under frame
 * budget, and it removes a dependency that would exist for one function.
 *
 * Node positions persist across setModel() by id, so an incoming detection
 * update rearranges the graph instead of resetting it. The previous model stays
 * on screen until a new one replaces it; nothing ever blanks.
 *
 * This module renders. It does not decide what to render — graph-builder.ts
 * does — and it holds no backend knowledge.
 */

import type { GraphLink, GraphModel, GraphNode } from '../data/types';

const SVG_NS = 'http://www.w3.org/2000/svg';

interface Transform {
  k: number;
  x: number;
  y: number;
}

export class ConfluxGraph {
  readonly svg: SVGSVGElement;

  private readonly container: HTMLElement;
  private readonly root: SVGGElement;
  private readonly linkLayer: SVGGElement;
  private readonly nodeLayer: SVGGElement;
  private readonly tooltip: HTMLDivElement;

  private nodes: GraphNode[] = [];
  private links: GraphLink[] = [];
  private nodeById = new Map<string, GraphNode>();

  private nodeEls = new Map<string, SVGGElement>();
  private linkEls = new Map<string, SVGLineElement>();

  private transform: Transform = { k: 1, x: 0, y: 0 };
  private alpha = 0;
  private frame: number | null = null;
  private width = 800;
  private height = 600;

  private selectedId: string | null = null;
  private hoveredId: string | null = null;
  private highlightKey: string | null = null;

  private readonly resizeObserver: ResizeObserver;

  constructor(container: HTMLElement) {
    this.container = container;

    this.svg = document.createElementNS(SVG_NS, 'svg');
    this.svg.setAttribute('class', 'conflux-graph');
    this.svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    const defs = document.createElementNS(SVG_NS, 'defs');
    defs.innerHTML = `
      <radialGradient id="cand-glow">
        <stop offset="0%" stop-color="currentColor" stop-opacity="0.55" />
        <stop offset="100%" stop-color="currentColor" stop-opacity="0" />
      </radialGradient>`;
    this.svg.appendChild(defs);

    this.root = document.createElementNS(SVG_NS, 'g');
    this.linkLayer = document.createElementNS(SVG_NS, 'g');
    this.linkLayer.setAttribute('class', 'link-layer');
    this.nodeLayer = document.createElementNS(SVG_NS, 'g');
    this.nodeLayer.setAttribute('class', 'node-layer');
    this.root.appendChild(this.linkLayer);
    this.root.appendChild(this.nodeLayer);
    this.svg.appendChild(this.root);
    container.appendChild(this.svg);

    this.tooltip = document.createElement('div');
    this.tooltip.className = 'graph-tooltip hidden';
    container.appendChild(this.tooltip);

    this.resizeObserver = new ResizeObserver(() => this.measure());
    this.resizeObserver.observe(container);
    this.measure();
  }

  /* -- geometry ------------------------------------------------------ */

  private measure(): void {
    const rect = this.container.getBoundingClientRect();
    this.width = Math.max(320, rect.width);
    this.height = Math.max(240, rect.height);
    this.svg.setAttribute('viewBox', `0 0 ${this.width} ${this.height}`);
    this.kick(0.3);
  }

  clientToWorld(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.svg.getBoundingClientRect();
    const scaleX = this.width / rect.width;
    const scaleY = this.height / rect.height;
    const px = (clientX - rect.left) * scaleX;
    const py = (clientY - rect.top) * scaleY;
    return {
      x: (px - this.transform.x) / this.transform.k,
      y: (py - this.transform.y) / this.transform.k,
    };
  }

  private applyTransform(): void {
    const { k, x, y } = this.transform;
    this.root.setAttribute('transform', `translate(${x} ${y}) scale(${k})`);
  }

  panBy(dx: number, dy: number): void {
    const rect = this.svg.getBoundingClientRect();
    this.transform.x += dx * (this.width / rect.width);
    this.transform.y += dy * (this.height / rect.height);
    this.applyTransform();
  }

  zoomAt(clientX: number, clientY: number, factor: number): void {
    const before = this.clientToWorld(clientX, clientY);
    this.transform.k = Math.min(4, Math.max(0.25, this.transform.k * factor));
    const rect = this.svg.getBoundingClientRect();
    const px = (clientX - rect.left) * (this.width / rect.width);
    const py = (clientY - rect.top) * (this.height / rect.height);
    this.transform.x = px - before.x * this.transform.k;
    this.transform.y = py - before.y * this.transform.k;
    this.applyTransform();
  }

  resetView(): void {
    this.transform = { k: 1, x: 0, y: 0 };
    this.applyTransform();
    this.kick(0.5);
  }

  /** Centre the view on a candidate hub without changing the model. */
  focusCandidate(candidateKey: string): void {
    const node = this.nodeById.get(`cand:${candidateKey}`);
    if (!node) return;
    this.transform.k = 1.5;
    this.transform.x = this.width / 2 - node.x * this.transform.k;
    this.transform.y = this.height / 2 - node.y * this.transform.k;
    this.applyTransform();
  }

  /* -- model --------------------------------------------------------- */

  setModel(model: GraphModel): void {
    const previous = this.nodeById;
    const next = new Map<string, GraphNode>();

    for (const node of model.nodes) {
      const old = previous.get(node.id);
      if (old) {
        // Preserve simulated state so the layout evolves rather than restarts.
        node.x = old.x;
        node.y = old.y;
        node.vx = old.vx;
        node.vy = old.vy;
        node.fx = old.fx;
        node.fy = old.fy;
        node.createdAt = old.createdAt;
      } else {
        const angle = Math.random() * Math.PI * 2;
        const radius = 40 + Math.random() * Math.min(this.width, this.height) * 0.3;
        node.x = this.width / 2 + Math.cos(angle) * radius;
        node.y = this.height / 2 + Math.sin(angle) * radius;
      }
      next.set(node.id, node);
    }

    this.nodes = model.nodes;
    this.links = model.links.filter((l) => next.has(l.source) && next.has(l.target));
    this.nodeById = next;

    this.syncDom();
    this.kick(0.7);
  }

  clear(): void {
    this.nodes = [];
    this.links = [];
    this.nodeById.clear();
    this.syncDom();
  }

  get nodeCount(): number {
    return this.nodes.length;
  }

  getNode(id: string): GraphNode | undefined {
    return this.nodeById.get(id);
  }

  /* -- DOM sync ------------------------------------------------------ */

  private syncDom(): void {
    // Links.
    const liveLinks = new Set(this.links.map((l) => l.id));
    for (const [id, el] of this.linkEls) {
      if (!liveLinks.has(id)) {
        el.remove();
        this.linkEls.delete(id);
      }
    }
    for (const link of this.links) {
      let el = this.linkEls.get(link.id);
      if (!el) {
        el = document.createElementNS(SVG_NS, 'line');
        el.setAttribute('data-link-id', link.id);
        el.setAttribute(
          'class',
          `graph-link kind-${link.kind} ${link.backendDerived ? 'backend' : 'derived'}`
        );
        this.linkLayer.appendChild(el);
        this.linkEls.set(link.id, el);
      }
    }

    // Nodes.
    const liveNodes = new Set(this.nodes.map((n) => n.id));
    for (const [id, el] of this.nodeEls) {
      if (!liveNodes.has(id)) {
        el.remove();
        this.nodeEls.delete(id);
      }
    }
    for (const node of this.nodes) {
      let el = this.nodeEls.get(node.id);
      if (!el) {
        el = document.createElementNS(SVG_NS, 'g');
        el.setAttribute('data-node-id', node.id);
        el.setAttribute('class', `graph-node kind-${node.kind} band-${node.band} entering`);

        if (node.kind === 'candidate') {
          const halo = document.createElementNS(SVG_NS, 'circle');
          halo.setAttribute('class', 'node-halo');
          halo.setAttribute('r', String(node.radius * 2.4));
          el.appendChild(halo);
        }

        const circle = document.createElementNS(SVG_NS, 'circle');
        circle.setAttribute('class', 'node-core');
        circle.setAttribute('r', String(node.radius));
        el.appendChild(circle);

        if (node.kind === 'candidate') {
          const text = document.createElementNS(SVG_NS, 'text');
          text.setAttribute('class', 'node-label');
          text.setAttribute('fill', '#AAB6C5');
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('dy', String(-node.radius - 8));
          text.textContent = node.label;
          el.appendChild(text);
        }

        this.nodeLayer.appendChild(el);
        this.nodeEls.set(node.id, el);
        window.setTimeout(() => el?.classList.remove('entering'), 620);
      } else {
        el.setAttribute(
          'class',
          `graph-node kind-${node.kind} band-${node.band}` +
            (this.selectedId === node.id ? ' selected' : '') +
            (this.hoveredId === node.id ? ' hovered' : '')
        );
        const core = el.querySelector<SVGCircleElement>('.node-core');
        core?.setAttribute('r', String(node.radius));
      }
    }

    this.applyEmphasis();
  }

  /* -- simulation ---------------------------------------------------- */

  kick(alpha = 0.6): void {
    this.alpha = Math.max(this.alpha, alpha);
    if (this.frame === null) {
      this.frame = requestAnimationFrame(() => this.tick());
    }
  }

  private tick(): void {
    this.frame = null;
    if (this.nodes.length === 0) {
      this.alpha = 0;
      return;
    }

    const centreX = this.width / 2;
    const centreY = this.height / 2;
    const nodes = this.nodes;
    const alpha = this.alpha;

    // Repulsion, O(n²) over a bounded window.
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let distSq = dx * dx + dy * dy;
        if (distSq === 0) {
          dx = (Math.random() - 0.5) * 0.5;
          dy = (Math.random() - 0.5) * 0.5;
          distSq = dx * dx + dy * dy;
        }
        if (distSq > 90_000) continue; // ignore distant pairs
        const dist = Math.sqrt(distSq);
        const strength = (900 * alpha) / distSq;
        const fx = (dx / dist) * strength;
        const fy = (dy / dist) * strength;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    // Springs.
    for (const link of this.links) {
      const a = this.nodeById.get(link.source);
      const b = this.nodeById.get(link.target);
      if (!a || !b) continue;
      const rest = link.kind === 'membership' ? 58 : 42;
      const stiffness = link.kind === 'membership' ? 0.05 : 0.028;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const force = (dist - rest) * stiffness * alpha;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    // Centring + integration.
    for (const node of nodes) {
      if (node.fx !== null && node.fy !== null) {
        node.x = node.fx;
        node.y = node.fy;
        node.vx = 0;
        node.vy = 0;
        continue;
      }
      node.vx += (centreX - node.x) * 0.0016 * alpha;
      node.vy += (centreY - node.y) * 0.0016 * alpha;
      node.vx *= 0.86;
      node.vy *= 0.86;
      node.x += node.vx;
      node.y += node.vy;
      const margin = node.radius + 6;
      node.x = Math.min(this.width - margin, Math.max(margin, node.x));
      node.y = Math.min(this.height - margin, Math.max(margin, node.y));
    }

    this.paint();

    this.alpha *= 0.97;
    if (this.alpha > 0.005) {
      this.frame = requestAnimationFrame(() => this.tick());
    } else {
      this.alpha = 0;
    }
  }

  private paint(): void {
    for (const node of this.nodes) {
      const el = this.nodeEls.get(node.id);
      if (el) el.setAttribute('transform', `translate(${node.x.toFixed(1)} ${node.y.toFixed(1)})`);
    }
    for (const link of this.links) {
      const el = this.linkEls.get(link.id);
      const a = this.nodeById.get(link.source);
      const b = this.nodeById.get(link.target);
      if (!el || !a || !b) continue;
      el.setAttribute('x1', a.x.toFixed(1));
      el.setAttribute('y1', a.y.toFixed(1));
      el.setAttribute('x2', b.x.toFixed(1));
      el.setAttribute('y2', b.y.toFixed(1));
    }
  }

  /* -- emphasis ------------------------------------------------------ */

  setSelection(nodeId: string | null): void {
    this.selectedId = nodeId;
    this.applyEmphasis();
  }

  setHover(nodeId: string | null): void {
    this.hoveredId = nodeId;
    this.applyEmphasis();
  }

  /** Dim everything not belonging to the given candidate. */
  highlightCandidate(candidateKey: string | null): void {
    this.highlightKey = candidateKey;
    this.applyEmphasis();
  }

  private applyEmphasis(): void {
    const key = this.highlightKey;
    for (const node of this.nodes) {
      const el = this.nodeEls.get(node.id);
      if (!el) continue;
      el.classList.toggle('selected', node.id === this.selectedId);
      el.classList.toggle('hovered', node.id === this.hoveredId);
      el.classList.toggle('dimmed', key !== null && node.candidateKey !== key);
    }
    for (const link of this.links) {
      const el = this.linkEls.get(link.id);
      if (!el) continue;
      const a = this.nodeById.get(link.source);
      const b = this.nodeById.get(link.target);
      const inFocus =
        key === null || (a?.candidateKey === key && b?.candidateKey === key);
      el.classList.toggle('dimmed', !inFocus);
    }
  }

  /* -- tooltip ------------------------------------------------------- */

  showTooltip(html: string, clientX: number, clientY: number): void {
    const rect = this.container.getBoundingClientRect();
    this.tooltip.innerHTML = html;
    this.tooltip.classList.remove('hidden');
    const x = Math.min(clientX - rect.left + 14, rect.width - this.tooltip.offsetWidth - 8);
    const y = Math.min(clientY - rect.top + 14, rect.height - this.tooltip.offsetHeight - 8);
    this.tooltip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  hideTooltip(): void {
    this.tooltip.classList.add('hidden');
  }

  /* -- drag support -------------------------------------------------- */

  pinNode(id: string, x: number, y: number): void {
    const node = this.nodeById.get(id);
    if (!node) return;
    node.fx = x;
    node.fy = y;
    this.kick(0.35);
  }

  releaseNode(id: string): void {
    const node = this.nodeById.get(id);
    if (!node) return;
    node.fx = null;
    node.fy = null;
    this.kick(0.35);
  }

  findLink(id: string): GraphLink | undefined {
    return this.links.find((l) => l.id === id);
  }

  destroy(): void {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.svg.remove();
    this.tooltip.remove();
  }
}
