import { CheckCircle2, ShieldAlert, ShieldCheck, X } from "lucide-react";

import type { PlanGraph, QualificationClaim } from "../types";
import type { ReviewPriority } from "../reviewPlanner";

interface QualityReviewProps {
  graph: PlanGraph;
  open: boolean;
  reviewCount: number;
  reviewPriorities: ReviewPriority[];
  onClose: () => void;
  onReviewNext: () => void;
  onLocateEntities: (entityIds: string[]) => void;
}

function label(value: string): string {
  return value.replaceAll("_", " ");
}

function metric(claim: QualificationClaim): string {
  if (claim.estimate == null) return "Not measured";
  const estimate = `${(claim.estimate * 100).toFixed(1)}%`;
  return claim.conservative_floor == null
    ? estimate
    : `${estimate} · floor ${(claim.conservative_floor * 100).toFixed(1)}%`;
}

export function QualityReview({
  graph,
  open,
  reviewCount,
  reviewPriorities,
  onClose,
  onReviewNext,
  onLocateEntities,
}: QualityReviewProps) {
  if (!open) return null;
  const profile = graph.drawing_profile;
  const qualification = graph.qualification;
  const eligible = Boolean(qualification?.production_release_eligible);
  const claims = qualification?.claims ?? [];
  const verification = graph.verification;
  const violations = verification?.violations ?? [];

  return (
    <div
      className="quality-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section className="quality-review" role="dialog" aria-modal="true" aria-labelledby="quality-title">
        <header>
          <div className={eligible ? "quality-icon eligible" : "quality-icon"}>
            {eligible ? <ShieldCheck size={20} /> : <ShieldAlert size={20} />}
          </div>
          <div>
            <span>MODEL ASSURANCE</span>
            <h2 id="quality-title">{eligible ? "Production gate passed" : "Review before release"}</h2>
          </div>
          <button onClick={onClose} aria-label="Close quality report"><X size={18} /></button>
        </header>

        <div className="quality-summary">
          <article>
            <span>Drawing class</span>
            <strong>{profile?.difficulty_class ?? qualification?.difficulty_class ?? "Unprofiled"}</strong>
            <small>{profile ? `${Math.round(profile.complexity_score * 100)} / 100 complexity` : "No profiler record"}</small>
          </article>
          <article>
            <span>Exact model pair</span>
            <strong>{qualification?.exact_model_match ? "Matched" : "Not qualified"}</strong>
            <small>{qualification?.benchmark_cohort || "No sealed cohort"}</small>
          </article>
          <article>
            <span>Review queue</span>
            <strong>{reviewCount}</strong>
            <small>Editable elements requiring attention</small>
          </article>
        </div>

        {profile?.reasons?.length ? (
          <div className="quality-complexity">
            <span>Complexity drivers</span>
            <p>{profile.reasons.join(" · ")}</p>
          </div>
        ) : null}

        {reviewPriorities.length ? (
          <div className="quality-priorities">
            <div className="quality-section-title">
              <div><span>GUIDED REVIEW</span><h3>Highest consequence items first</h3></div>
              <small>Risk combines evidence and BIM impact</small>
            </div>
            {reviewPriorities.slice(0, 5).map((priority) => (
              <article key={`${priority.selection.collection}:${priority.selection.id}`}>
                <span className={`priority-score ${priority.band}`}>{priority.percent}</span>
                <div>
                  <strong>{priority.selection.id}</strong>
                  <small>{priority.reasons[0]?.label ?? "Review evidence and relationships"}</small>
                </div>
                <code>{priority.selection.collection.replaceAll("_", " ")}</code>
                <button type="button" onClick={() => { onLocateEntities([priority.selection.id]); onClose(); }}>Locate</button>
              </article>
            ))}
          </div>
        ) : null}

        <div className={`quality-integrity ${verification?.release_allowed ? "passed" : "blocked"}`}>
          <div className="quality-section-title">
            <div>
              <span>DETERMINISTIC INTEGRITY</span>
              <h3>{verification?.release_allowed ? "Geometry and topology checks passed" : "Model contradictions"}</h3>
            </div>
            <small>
              {verification
                ? `${verification.passed_invariants} / ${verification.checked_invariants} checks`
                : "No certificate"}
            </small>
          </div>
          {violations.length ? violations.map((violation, index) => (
            <article key={`${violation.code}-${index}`}>
              <code>{violation.code}</code>
              <div>
                <strong>{violation.message}</strong>
                {violation.remediation ? <small>{violation.remediation}</small> : null}
              </div>
              {violation.entity_ids.length ? (
                <button
                  type="button"
                  onClick={() => {
                    onLocateEntities(violation.entity_ids);
                    onClose();
                  }}
                >
                  Locate
                </button>
              ) : null}
            </article>
          )) : (
            <div className="quality-empty">
              {verification ? "No deterministic contradictions were found." : "This graph has no verification certificate."}
            </div>
          )}
        </div>

        <div className="quality-claims">
          <div className="quality-section-title">
            <div><span>SEALED EVIDENCE</span><h3>What has actually been measured</h3></div>
            <small>{qualification?.benchmark_sample_count ?? 0} samples</small>
          </div>
          {claims.length ? claims.map((claim) => (
            <article key={claim.claim}>
              <span className={`claim-state ${claim.status}`}>
                {claim.status === "measured" ? <CheckCircle2 size={13} /> : <ShieldAlert size={13} />}
                {label(claim.status)}
              </span>
              <strong>{label(claim.claim)}</strong>
              <b>{metric(claim)}</b>
              <small>{claim.metric || claim.note || "No sealed metric is available."}</small>
            </article>
          )) : (
            <div className="quality-empty">This imported graph has no model qualification record.</div>
          )}
        </div>

        {!eligible && qualification?.review_reasons?.length ? (
          <div className="quality-reasons">
            <span>Release blockers</span>
            <p>{qualification.review_reasons.map(label).join(" · ")}</p>
          </div>
        ) : null}

        <footer>
          <p>Review risk guides triage. Sealed benchmark metrics remain scoped to their measured BIM claim.</p>
          <button onClick={() => { onReviewNext(); onClose(); }} disabled={!reviewCount}>
            Start guided review <b>{reviewCount}</b>
          </button>
        </footer>
      </section>
    </div>
  );
}
