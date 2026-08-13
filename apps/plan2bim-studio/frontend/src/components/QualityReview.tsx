import { ArrowUpRight, CheckCircle2, ChevronDown, ShieldAlert, ShieldCheck, X } from "lucide-react";

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
  const reviewedReference = graph.pipeline?.demo_kind === "direct_visual_ground_truth";
  const eligible = Boolean(qualification?.production_release_eligible);
  const claims = qualification?.claims ?? [];
  const verification = graph.verification;
  const violations = verification?.violations ?? [];
  const nextPriority = reviewPriorities[0];

  return (
    <div
      className="quality-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section className="quality-review" role="dialog" aria-modal="true" aria-labelledby="quality-title">
        <header>
          <div className={eligible ? "quality-icon eligible" : "quality-icon"}>
            {eligible ? <ShieldCheck size={21} /> : <ShieldAlert size={21} />}
          </div>
          <div>
            <span>MODEL ASSURANCE</span>
            <h2 id="quality-title">{reviewedReference ? "Reviewed reference" : eligible ? "Ready to release" : "Review before release"}</h2>
          </div>
          <button onClick={onClose} aria-label="Close quality report"><X size={19} /></button>
        </header>

        <div className="quality-body">
          <section className="quality-intro">
            <span>GUIDED REVIEW</span>
            <h3>{reviewedReference ? "Checked directly against the source drawing." : eligible ? "Every required check is complete." : "Resolve what changes the model first."}</h3>
            <p>
              {reviewedReference
                ? "Every entity in the demo contract was accepted by whole-sheet visual review, followed by a separate omission scan."
                : profile?.reasons?.length
                ? profile.reasons.join(". ")
                : "Evidence, geometry, and BIM relationships are evaluated together."}
            </p>
          </section>

          {!reviewedReference && nextPriority ? (
            <section className="quality-focus" aria-label="Next guided review item">
              <div className="quality-focus-score">
                <strong>{nextPriority.percent}</strong>
                <span>impact</span>
              </div>
              <div>
                <span>{nextPriority.selection.collection.replaceAll("_", " ")}</span>
                <h3>{nextPriority.selection.id}</h3>
                <p>{nextPriority.reasons[0]?.label ?? "Review evidence and model relationships."}</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  onLocateEntities([nextPriority.selection.id]);
                  onClose();
                }}
              >
                Locate <ArrowUpRight size={15} />
              </button>
            </section>
          ) : !reviewedReference ? (
            <div className="quality-empty">No element needs guided review.</div>
          ) : null}

          {!reviewedReference ? <details className="quality-disclosure">
            <summary>
              <div>
                <span>MODEL CHECKS</span>
                <strong>{verification?.release_allowed ? "No contradictions found" : "Contradictions to resolve"}</strong>
              </div>
              <small>
                {verification
                  ? `${verification.passed_invariants} of ${verification.checked_invariants} passed`
                  : "No certificate"}
              </small>
              <ChevronDown size={17} />
            </summary>
            <div className="quality-detail-list">
              {violations.length ? violations.map((violation, index) => (
                <article key={`${violation.code}-${index}`}>
                  <div>
                    <span>{label(violation.code)}</span>
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
          </details> : null}

          {!reviewedReference ? <details className="quality-disclosure">
            <summary>
              <div>
                <span>MEASURED EVIDENCE</span>
                <strong>Qualification record</strong>
              </div>
              <small>{qualification?.benchmark_sample_count ?? 0} samples</small>
              <ChevronDown size={17} />
            </summary>
            <div className="quality-evidence-list">
              {claims.length ? claims.map((claim) => (
                <article key={claim.claim}>
                  <span className={`claim-state ${claim.status}`}>
                    {claim.status === "measured" ? <CheckCircle2 size={13} /> : <ShieldAlert size={13} />}
                    {label(claim.status)}
                  </span>
                  <div>
                    <strong>{label(claim.claim)}</strong>
                    <small>{claim.metric || claim.note || "No sealed metric is available."}</small>
                  </div>
                  <b>{metric(claim)}</b>
                </article>
              )) : (
                <div className="quality-empty">This imported graph has no model qualification record.</div>
              )}
            </div>
          </details> : null}

          {!reviewedReference && !eligible && qualification?.review_reasons?.length ? (
            <p className="quality-blockers">
              <strong>Release is paused.</strong> {qualification.review_reasons.map(label).join(". ")}
            </p>
          ) : null}
        </div>

        <footer>
          <p>Each decision stays attached to the model history and exported correction record.</p>
          {reviewedReference ? (
            <button onClick={onClose}>Done</button>
          ) : (
            <button onClick={() => { onReviewNext(); onClose(); }} disabled={!reviewCount}>
              Review next <b>{reviewCount}</b>
            </button>
          )}
        </footer>
      </section>
    </div>
  );
}
