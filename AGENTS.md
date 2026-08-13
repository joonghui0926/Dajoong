# Dajoong repository instructions

## Ground-truth annotations

- Never promote model output, pseudo-labels, generated labels, vector draw-order extraction, or an existing candidate graph to ground truth.
- The source drawing itself must be opened at native resolution and visually reviewed in full before any annotation is marked as ground truth.
- Ground truth is created by direct visual annotation of the source drawing. Candidate overlays may be used only as review aids and every entity must still be accepted, corrected, or rejected by looking at the source pixels.
- Record a source image hash, the reviewed sheet bounds, the annotator, the review date, the entity-level evidence region, and whether the whole sheet was scanned for omissions.
- Keep non-commercial public drawings in a sealed evaluation-only corpus. Train commercial models only on Dajoong-owned drawings or data with explicit commercial-training permission.
- Do not report F1 from a corpus that fails these requirements. Treat it as an exploratory diagnostic instead.

