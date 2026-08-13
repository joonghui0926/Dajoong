import { ArrowRight } from "lucide-react";
import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from "react";

import { AsyncFeatureBoundary, reliableLazy } from "./components/AsyncFeatureBoundary";
import { DajoongLogo } from "./components/DajoongLogo";

const ConversionDialog = reliableLazy(async () => ({ default: (await import("./components/ConversionDialog")).ConversionDialog }));

const preloadStudio = () => void import("./StudioRoute");
const preloadConversion = () => void import("./components/ConversionDialog");
const MEASURED_MEDIAN_SECONDS = 2.720126;
const HERO_MODEL = "/marketing/cubi-008-reviewed-bim.png";
const REVIEWED_SOURCE = "/marketing/cubi-014-reviewed-source.webp";
const REVIEWED_MODEL = "/marketing/cubi-014-reviewed-bim.webp";
const COLLABORATION_SOURCE = "/marketing/cubi-005-reviewed-source.png";

export function Landing() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [, setStatus] = useState("Reviewed sample · Level 1");
  const [engineActive, setEngineActive] = useState(false);
  const [displaySeconds, setDisplaySeconds] = useState(0);
  const [comparison, setComparison] = useState(57);
  const landingRef = useRef<HTMLElement | null>(null);
  const landingRestoreGuardRef = useRef(false);
  const resetLandingScrollRef = useRef<() => void>(() => undefined);
  const engineRef = useRef<HTMLElement | null>(null);
  const enginePlayedRef = useRef(false);

  useLayoutEffect(() => {
    const previousScrollRestoration = window.history.scrollRestoration;
    window.history.scrollRestoration = "manual";
    const navigationEntry = window.performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined;
    const guardRestoredPosition = Boolean(window.location.hash)
      || navigationEntry?.type === "reload"
      || navigationEntry?.type === "back_forward";
    landingRestoreGuardRef.current = guardRestoredPosition;

    // A refreshed marketing page must always reopen at the hero. Section links
    // still work during the current visit, but a stale fragment must not turn a
    // reload into an implicit jump to the embedded Studio.
    if (window.location.hash) {
      window.history.replaceState(
        window.history.state,
        "",
        `${window.location.pathname}${window.location.search}`,
      );
    }

    const landingScroller = landingRef.current;
    const previousInlineScrollBehavior = landingScroller?.style.scrollBehavior ?? "";
    if (landingScroller) landingScroller.style.scrollBehavior = "auto";
    const resetScroll = () => {
      window.scrollTo(0, 0);
      if (landingScroller) landingScroller.scrollTop = 0;
    };
    resetLandingScrollRef.current = resetScroll;
    const releaseRestoreGuard = () => {
      landingRestoreGuardRef.current = false;
    };
    resetScroll();
    const animationFrame = window.requestAnimationFrame(resetScroll);
    const postLayoutReset = window.setTimeout(resetScroll, guardRestoredPosition ? 180 : 0);
    const postRestoreReset = guardRestoredPosition ? window.setTimeout(resetScroll, 620) : 0;
    const finalRestoreReset = guardRestoredPosition ? window.setTimeout(resetScroll, 1_400) : 0;
    const releaseScrollBehavior = window.setTimeout(() => {
      if (landingScroller) landingScroller.style.scrollBehavior = previousInlineScrollBehavior;
    }, guardRestoredPosition ? 1_450 : 80);
    if (guardRestoredPosition) {
      window.addEventListener("pageshow", resetScroll);
      window.addEventListener("load", resetScroll);
    }
    landingScroller?.addEventListener("wheel", releaseRestoreGuard, { passive: true });
    landingScroller?.addEventListener("pointerdown", releaseRestoreGuard, { passive: true });
    landingScroller?.addEventListener("touchstart", releaseRestoreGuard, { passive: true });
    window.addEventListener("keydown", releaseRestoreGuard);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      window.clearTimeout(postLayoutReset);
      window.clearTimeout(postRestoreReset);
      window.clearTimeout(finalRestoreReset);
      window.clearTimeout(releaseScrollBehavior);
      if (guardRestoredPosition) {
        window.removeEventListener("pageshow", resetScroll);
        window.removeEventListener("load", resetScroll);
      }
      landingScroller?.removeEventListener("wheel", releaseRestoreGuard);
      landingScroller?.removeEventListener("pointerdown", releaseRestoreGuard);
      landingScroller?.removeEventListener("touchstart", releaseRestoreGuard);
      window.removeEventListener("keydown", releaseRestoreGuard);
      if (landingScroller) landingScroller.style.scrollBehavior = previousInlineScrollBehavior;
      resetLandingScrollRef.current = () => undefined;
      window.history.scrollRestoration = previousScrollRestoration;
    };
  }, []);

  useEffect(() => {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const revealTargets = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (reduceMotion || !("IntersectionObserver" in window)) {
      revealTargets.forEach((target) => target.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        (entry.target as HTMLElement).classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    }, { root: document.querySelector(".landing-page"), rootMargin: "0px 0px -10%", threshold: 0.12 });
    revealTargets.forEach((target) => observer.observe(target));
    return () => observer.disconnect();
  }, []);

  const visitSection = (event: React.MouseEvent<HTMLAnchorElement>, sectionId: string) => {
    event.preventDefault();
    const section = document.getElementById(sectionId);
    if (!section) return;
    const nextHash = `#${sectionId}`;
    if (window.location.hash === nextHash) {
      window.history.replaceState(window.history.state, "", nextHash);
    } else {
      window.history.pushState(window.history.state, "", nextHash);
    }
    section.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    });
  };

  const handleEmbeddedStudioLoad = (event: React.SyntheticEvent<HTMLIFrameElement>) => {
    if (!landingRestoreGuardRef.current) return;
    event.currentTarget.blur();
    resetLandingScrollRef.current();
    window.requestAnimationFrame(() => resetLandingScrollRef.current());
    landingRestoreGuardRef.current = false;
  };

  useEffect(() => {
    const section = engineRef.current;
    if (!section) return;
    let animationFrame = 0;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const startReplay = () => {
      if (enginePlayedRef.current) return;
      enginePlayedRef.current = true;
      setEngineActive(true);
      if (reduceMotion) {
        setDisplaySeconds(MEASURED_MEDIAN_SECONDS);
        return;
      }
      const startedAt = performance.now();
      const tick = (now: number) => {
        const progress = Math.min(1, (now - startedAt) / 2_050);
        const eased = 1 - Math.pow(1 - progress, 3);
        setDisplaySeconds(MEASURED_MEDIAN_SECONDS * eased);
        if (progress < 1) animationFrame = window.requestAnimationFrame(tick);
      };
      animationFrame = window.requestAnimationFrame(tick);
    };
    if (!("IntersectionObserver" in window)) startReplay();
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) startReplay();
    }, { root: document.querySelector(".landing-page"), threshold: 0.28 });
    observer.observe(section);
    return () => {
      observer.disconnect();
      window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  const comparisonStyle = { "--comparison": `${comparison}%` } as CSSProperties;

  return (
    <main ref={landingRef} className="landing-page landing-v3">
      <nav className="landing-nav">
        <a href="/" className="landing-brand"><DajoongLogo /></a>
        <div className="landing-links">
          <a href="#speed" onClick={(event) => visitSection(event, "speed")}>Speed</a>
          <a href="#studio" onClick={(event) => visitSection(event, "studio")}>Studio</a>
          <a href="#teams" onClick={(event) => visitSection(event, "teams")}>Teams</a>
          <a href="#pricing" onClick={(event) => visitSection(event, "pricing")}>Pricing</a>
        </div>
        <a className="nav-cta" href="/studio" onMouseEnter={preloadStudio} onFocus={preloadStudio}>Open Studio <ArrowRight size={15} /></a>
      </nav>

      <section className="landing-hero-v3">
        <svg className="landing-filter-defs" aria-hidden="true">
          <filter id="dajoong-wireframe" colorInterpolationFilters="sRGB">
            <feColorMatrix in="SourceGraphic" type="saturate" values="0" result="mono" />
            <feConvolveMatrix in="mono" order="3" kernelMatrix="-1 -1 -1 -1 8 -1 -1 -1 -1" divisor="1" bias="0" result="edges" />
            <feColorMatrix in="edges" type="luminanceToAlpha" result="edgeAlpha" />
            <feComponentTransfer in="edgeAlpha"><feFuncA type="gamma" amplitude="1" exponent=".72" offset="0" /></feComponentTransfer>
          </filter>
        </svg>
        <div className="hero-wireframe" aria-hidden="true"><img src={HERO_MODEL} alt="" width="1600" height="1000" fetchPriority="high" /></div>
        <div className="landing-hero-copy" data-reveal>
          <p className="landing-kicker">DRAWING TO EDITABLE BIM</p>
          <h1>Your plan.<br /><span>Now spatial.</span></h1>
          <p>Dajoong turns a complete architectural drawing into a model your team can edit, coordinate, and deliver.</p>
          <div className="landing-hero-actions">
            <button onMouseEnter={preloadConversion} onFocus={preloadConversion} onClick={() => setDialogOpen(true)}>Convert a drawing <ArrowRight size={17} /></button>
            <a href="#speed" onClick={(event) => visitSection(event, "speed")}>See the measured run</a>
          </div>
        </div>
      </section>

      <section className="landing-speed-v3" id="speed" ref={engineRef} data-active={engineActive ? "true" : "false"}>
        <div className="speed-v3-copy" data-reveal>
          <p className="landing-kicker">MEASURED END-TO-END CONVERSION</p>
          <div className="speed-v3-reading"><strong>{displaySeconds.toFixed(2)}</strong><span>seconds</span></div>
          <h2><span>Fast enough,</span><span>decisions stay live.</span></h2>
          <p>Dajoong’s research team developed a whole-sheet spatial compiler that preserves drawing context while resolving editable building elements in one measured run.</p>
          <small>Median of 7 cold full-pipeline runs on the bundled benchmark drawing.</small>
        </div>
        <figure className="conversion-compare" style={comparisonStyle} data-reveal>
          <div className="compare-label compare-label-source">SOURCE DRAWING</div>
          <div className="compare-label compare-label-model">EDITABLE COLOR BIM</div>
          <img className="compare-source" src={REVIEWED_SOURCE} alt="Reviewed CUBI-014 source drawing" width="2038" height="1237" loading="lazy" decoding="async" />
          <div className="compare-model"><img src={REVIEWED_MODEL} alt="Latest reviewed CUBI-014 BIM result" width="1600" height="1000" loading="lazy" decoding="async" /></div>
          <div className="compare-divider" aria-hidden="true"><span>↔</span></div>
          <input aria-label="Compare source drawing and editable BIM" type="range" min="12" max="88" value={comparison} onChange={(event) => setComparison(Number(event.target.value))} />
          <figcaption>Drag to move between the reviewed drawing and the latest stored model.</figcaption>
        </figure>
      </section>

      <section className="landing-editor-v3" id="studio">
        <header data-reveal>
          <p className="landing-kicker">DAJOONG STUDIO</p>
          <h2><span>Not a dead export,</span><span>a working building.</span></h2>
          <p>Organize, edit, and review with the same tools available in Dajoong Studio.</p>
        </header>
        <div className="studio-embed-v3" data-reveal>
          <div className="studio-embed-meta"><span><i /> LIVE CUBI-020 WORKSPACE</span><b>Actual Studio tools · collaboration shown below</b></div>
          <iframe
            title="Interactive CUBI-020 Dajoong Studio workspace"
            src="/studio?embed=landing&demo=cubi-020"
            loading="lazy"
            onLoad={handleEmbeddedStudioLoad}
          />
        </div>
        <a className="editor-link-v3" href="/studio" onMouseEnter={preloadStudio} onFocus={preloadStudio}>Work in the full Studio <ArrowRight size={16} /></a>
      </section>

      <section className="landing-teams-v3" id="teams">
        <div className="teams-v3-copy" data-reveal>
          <div>
            <p className="landing-kicker">LIVE PROJECT COORDINATION</p>
            <h2><span>One model,</span><span>everyone on the same page.</span></h2>
          </div>
          <div className="teams-v3-summary">
            <p>Reviewers see the same element, leave decisions in context, and hand off a clean history instead of another marked-up copy.</p>
            <div className="team-presence-v3"><span>PC</span><span>MA</span><span>SK</span><b>3 reviewers active</b></div>
          </div>
        </div>
        <div className="collaboration-stage-v3" data-reveal>
          <div className="team-toolbar-v3"><span><i />LIVE REVIEW</span><button>Share model</button><button>Present</button></div>
          <div className="team-canvas-v3">
            <img src={COLLABORATION_SOURCE} alt="Reviewed CUBI-005 source drawing open for coordinated team review" width="1221" height="1533" loading="lazy" decoding="async" />
            <div className="live-cursor cursor-one"><i />Paul · W-005</div>
            <div className="live-cursor cursor-two"><i />Sara · D-012</div>
            <div className="team-pulse-v3" aria-hidden="true" />
          </div>
          <aside className="team-panel-v3">
            <header><span><small>PROJECT TEAM</small><strong>3 people online</strong></span><button>Invite</button></header>
            <div className="team-members-v3"><span><i>PC</i><b>Paul Cho</b><small>Reviewing W-005</small></span><span><i>MA</i><b>Mike Alvarez</b><small>Viewing Level 1</small></span><span><i>SK</i><b>Sara Kim</b><small>Editing D-012</small></span></div>
            <nav><button className="active"># model-review <b>4</b></button><button>Activity</button></nav>
            <div className="team-message-list-v3">
              <article className="team-thread-v3">
                <i>PC</i>
                <div><header><strong>Paul Cho</strong><time>14:32</time></header><p>Height confirmed against CUBI-005.</p><span>W-005 · Resolved</span></div>
              </article>
              <article className="team-thread-v3 secondary">
                <i>SK</i>
                <div><header><strong>Sara Kim</strong><time>14:28</time></header><p>Door swing updated. Ready to review.</p><span>D-012 · Mentioned you</span></div>
              </article>
              <article className="team-thread-v3 tertiary">
                <i>MA</i>
                <div><header><strong>Mike Alvarez</strong><time>14:21</time></header><p>Opening family checked in the 3D view.</p><span>Level 1 · Reviewed</span></div>
              </article>
            </div>
            <div className="team-compose-v3"><span>Message #model-review</span><button>Send</button></div>
          </aside>
        </div>
      </section>

      <section className="landing-pricing-v3" id="pricing">
        <div className="pricing-v3-copy" data-reveal>
          <p className="landing-kicker">ONE SUBSCRIPTION. UNLIMITED DRAWINGS.</p>
          <h2>Professional BIM without the traditional price.</h2>
          <p className="pricing-lede-v3">
            <span>Your first drawing is free. Then Dajoong Unlimited is <strong>$79/month</strong>.</span>
            <span><b>Comparison:</b> about one fifth of a standard Autodesk Revit monthly subscription.</span>
          </p>
          <button onMouseEnter={preloadConversion} onFocus={preloadConversion} onClick={() => setDialogOpen(true)}>Convert the first drawing <ArrowRight size={17} /></button>
        </div>
        <div className="price-chart-v3" data-reveal aria-label="Monthly subscription price comparison">
          <div className="price-scale"><span>MONTHLY SUBSCRIPTION · USD</span><b>PER USER</b></div>
          <div className="price-row competitor"><label><span>Autodesk Revit</span><small>Standard monthly subscription</small></label><i /><b>$380</b></div>
          <div className="price-row dajoong"><label><span>Dajoong Unlimited</span><small>Unlimited drawing conversions</small></label><i /><b>$79</b></div>
          <p>79% lower monthly price <span>·</span> First drawing free</p>
          <small>
            Monthly list prices checked August 13, 2026. Product scope, billing terms, taxes, and regional pricing differ.{" "}
            <a href="https://www.autodesk.com/solutions/revit-subscription-faq" target="_blank" rel="noreferrer">Autodesk price source</a>
          </small>
        </div>
      </section>

      <footer className="landing-footer"><DajoongLogo inverse /><div><a href="/privacy">Privacy</a><a href="/cookies">Cookies</a><a href="/terms">Terms</a><a href="/support">Support</a></div><small>© 2026 Dajoong</small></footer>

      {dialogOpen ? (
        <AsyncFeatureBoundary label="Opening conversion workspace">
          <ConversionDialog
            open
            onClose={() => setDialogOpen(false)}
            onStatus={setStatus}
            onComplete={(_graph, _sourceUrl, jobId) => {
              setDialogOpen(false);
              window.location.assign(jobId ? `/studio?job=${encodeURIComponent(jobId)}` : "/studio");
            }}
          />
        </AsyncFeatureBoundary>
      ) : null}
    </main>
  );
}
