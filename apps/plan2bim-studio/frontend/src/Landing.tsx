import { ArrowRight, Box, Cpu, FileCheck2, Layers3, MousePointer2, ShieldCheck, Sparkles } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";

import { DajoongLogo } from "./components/DajoongLogo";
import { ModelViewport } from "./components/ModelViewport";
import type { PlanGraph } from "./types";

const ConversionDialog = lazy(async () => ({ default: (await import("./components/ConversionDialog")).ConversionDialog }));

export function Landing() {
  const [graph, setGraph] = useState<PlanGraph | null>(null);
  const [sourceUrl, setSourceUrl] = useState("/sample/source.png");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [status, setStatus] = useState("Reviewed sample · Level 1");

  useEffect(() => {
    fetch("/sample/03-plan-graph.json")
      .then((response) => response.json())
      .then((payload: PlanGraph) => setGraph(payload))
      .catch(() => setStatus("Open Studio to load a project"));
  }, []);

  return (
    <main className="landing-page">
      <nav className="landing-nav">
        <a href="/" className="landing-brand"><DajoongLogo /></a>
        <div className="landing-links">
          <a href="#engine">Engine</a><a href="#studio">Studio</a><a href="#platform">Platform</a>
        </div>
        <a className="nav-cta" href="/studio">Open Studio <ArrowRight size={15} /></a>
      </nav>

      <section className="hero-section">
        <div className="hero-copy">
          <p className="section-kicker">CPU NATIVE · REVIEW GATED · IFC 4.3</p>
          <h1>From drawing to<br /><span>editable BIM.</span></h1>
          <p className="hero-lede">Dajoong turns architectural drawings into spatial models that remain connected to their source. Every wall, opening, room, and object is ready to inspect and edit.</p>
          <div className="hero-actions"><button onClick={() => setDialogOpen(true)}>Convert a drawing <ArrowRight size={17} /></button><a href="#engine">See how it works</a></div>
        </div>
        <div className="hero-orbit" aria-hidden="true"><span /><span /><span /></div>
        <div className="hero-workbench">
          <div className="workbench-head">
            <span>{status}</span><b>2D</b><i /><b>3D</b>
          </div>
          <button className="drawing-input" onClick={() => setDialogOpen(true)} aria-label="Choose drawing">
            <img src={sourceUrl} alt="Floor plan drawing input" />
            <span><MousePointer2 size={15} /> Replace drawing</span>
          </button>
          <div className="landing-model">
            {graph ? <ModelViewport graph={graph} levelId={graph.levels[0]?.id ?? "L1"} onSelect={() => undefined} minimal /> : <div className="model-loading"><Layers3 /> Preparing model</div>}
          </div>
        </div>
      </section>

      <section className="proof-ribbon" aria-label="Product facts">
        <div><strong>86,533</strong><span>parameters in the bundled core</span></div>
        <div><strong>CPU</strong><span>portable ONNX inference</span></div>
        <div><strong>IFC + GLB</strong><span>semantic and visual exports</span></div>
        <div><strong>Source linked</strong><span>stable IDs and provenance</span></div>
      </section>

      <figure className="generated-product-visual">
        <img src="/brand/dajoong-plan-to-bim-hero-1600.png" alt="Architectural drawing building into a coordinated color BIM" />
        <figcaption><span>REFERENCE OUTPUT CONTRACT</span><strong>Drawing context, building geometry, and coordinated systems in one editable scene.</strong></figcaption>
      </figure>

      <section className="method-section" id="engine">
        <div className="section-intro"><p className="section-kicker">THE ENGINE</p><h2>A small model proposes.<br />A compiler proves.</h2><p>Compact neural inference reads the page once. Deterministic geometry then resolves dimensions, joins, topology, and export contracts. Low-confidence work stays visible for review.</p></div>
        <div className="method-flow">
          <article><span>01</span><Cpu /><h3>Read the full sheet</h3><p>A global image pyramid preserves large structure while focused tiles retain small symbols and notes.</p></article>
          <article><span>02</span><Sparkles /><h3>Build semantic geometry</h3><p>Walls and openings become metric entities with levels, materials, confidence, and source references.</p></article>
          <article><span>03</span><FileCheck2 /><h3>Verify before export</h3><p>Topology and contradiction checks hold uncertain elements for review before IFC or GLB release.</p></article>
        </div>
      </section>

      <section className="studio-section" id="studio">
        <div className="studio-copy"><p className="section-kicker">DAJOONG STUDIO</p><h2>Correct the model where you see the issue.</h2><p>The plan and model share one selection. Drag equipment, reshape walls, edit dimensions, inspect provenance, and move through a confidence-ranked review queue.</p><a href="/studio">Work in Studio <ArrowRight size={16} /></a></div>
        <div className="studio-illustration">
          <div className="studio-rail"><span /><span /><span /><span /></div>
          <div className="studio-plan-mini"><img src="/sample/source.png" alt="Linked plan view" /><i className="selection-a" /><i className="selection-b" /></div>
          <aside><small>SELECTED WALL</small><strong>Interior partition</strong><label>Height <b>3.00 m</b></label><label>Thickness <b>120 mm</b></label><label>Confidence <b>96%</b></label><button>Accept element</button></aside>
        </div>
      </section>

      <section className="platform-section" id="platform">
        <div><p className="section-kicker">PRODUCTION PLATFORM</p><h2>One conversion contract.<br />Any product surface.</h2></div>
        <div className="platform-points">
          <article><Box /><h3>Portable core</h3><p>The input to output engine remains independent from accounts, queues, billing, and editing.</p></article>
          <article><ShieldCheck /><h3>Auditable by design</h3><p>Source hashes, page references, stable entity IDs, and correction history travel with the project.</p></article>
          <article><Layers3 /><h3>Built for teams</h3><p>Web and mobile clients share project state while heavy jobs can scale through isolated workers.</p></article>
        </div>
      </section>

      <section className="closing-section"><p>Start with the drawing.<br />Leave with a model you can own.</p><a href="/studio">Open Dajoong Studio <ArrowRight /></a></section>
      <footer className="landing-footer"><DajoongLogo inverse /><div><a href="/privacy">Privacy</a><a href="/cookies">Cookies</a><a href="/terms">Terms</a><a href="/support">Support</a></div><small>© 2026 Dajoong</small></footer>

      {dialogOpen ? (
        <Suspense fallback={<div className="studio-tool-loading" role="status"><span />Opening conversion workspace</div>}>
          <ConversionDialog open onClose={() => setDialogOpen(false)} onStatus={setStatus} onComplete={(nextGraph, nextSource) => { setGraph(nextGraph); if (nextSource) setSourceUrl(nextSource); }} />
        </Suspense>
      ) : null}
    </main>
  );
}
