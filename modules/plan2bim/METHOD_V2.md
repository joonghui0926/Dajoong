# Dajoong Method v2 — Global Program Reconstruction

Method v1 treated the drawing mainly as a dense set of local pixel classes. It
could score obvious walls while still missing the building: exterior shape,
room closure, host relationships, and omitted elements were not one joint
decision. Threshold changes cannot repair that failure mode.

Method v2 changes the prediction target from isolated pixels to a constrained
building program.

## 1. Normalize one complete sheet

The complete drawing is rectified and scaled once. The global topology student
always receives a whole-sheet token plus multi-scale drafting evidence. Crops
may be used by element specialists, but never as the primary building view.

## 2. Recover the building program first

The global student predicts six aligned structural targets: exterior boundary,
wall centerline, junction, opening, room seed, and room interior. A deterministic
graph decoder must then form closed rooms and connected walls. An opening is not
valid merely because its pixels look like a door; it needs a compatible host wall.

## 3. Ask local experts only after topology exists

Door, window, fixture, stair, electrical, plumbing, HVAC, and object experts are
conditioned on the recovered room and host context. Their evidence remains
separate in the spatial evidence graph. Disagreement creates review state rather
than invented geometry.

The spatial evidence graph is the sole compiler-proposal source. There is no
parallel detector-to-BIM shortcut; a local detection must survive the graph's
promotion and relation checks before it can become an editable BIM entity.

## 4. Compile and render the candidate BIM

The graph is compiled into metric BIM entities with stable IDs and provenance.
The result is rendered back to the source coordinate system. Missing ink,
impossible intersections, unhosted openings, and unassigned equipment block
release or create bounded repair hypotheses.

## 5. Train without contaminating evaluation

Dajoong's inverse compiler generates varied building programs for topology
pretraining. Generated samples are permanently marked `synthetic_pretrain_only`,
`real_drawing_ground_truth=false`, and `evaluation_eligible=false`. Production
calibration may use only Dajoong-owned drawings or data with explicit commercial
training rights and direct whole-sheet human annotation. Public non-commercial
drawings remain sealed evaluation-only data.

## Release rule

No single F1 score makes a result production-safe. Release requires entity-level
metrics, whole-sheet omission checks, topology validity, independent expert
agreement, and BIM-to-source reprojection. The current runtime remains review
gated until those conditions are satisfied.
