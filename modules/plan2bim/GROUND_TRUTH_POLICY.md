# Plan2BIM ground-truth policy

Only a direct, whole-sheet visual annotation of the native-resolution source drawing is ground truth.

Merely writing `direct_visual_source_annotation` in a manifest is not evidence.
Every accepted entity must have an independent source-pixel annotation event and
must record `evidence_kind: native_source_pixels`. Geometry copied from SVG,
vector draw order, an existing graph, or any candidate overlay is rejected even
when someone later looked at the raster.

## Required workflow

1. Open the original sheet at native resolution and inspect the complete plan area.
2. Make separate visual passes for walls, openings, rooms, fixed equipment, building systems and every drawn furniture or appliance symbol required by the selected content profile.
3. Annotate entities from the source pixels. An existing graph or model overlay is a review aid only.
4. Scan the complete plan area for omissions after entity annotation.
5. Record the source hash, reviewed bounds, annotator, date, evidence region and license scope.
6. Validate the manifest before it can enter a benchmark or training split.

The production `full_editable_bim` profile includes visible movable furniture and
typed appliances. A structural-only benchmark may not be used to claim product
object F1 because excluded furniture would make the product recall denominator
incomplete.

## Prohibited ground-truth sources

- Model predictions or pseudo-labels
- Automatically generated labels
- Candidate graphs accepted without source-image review
- Vector draw order presented as proof of raster recognition
- Synthetic annotations presented as real-drawing ground truth

## Dataset separation

`commercial_train` is restricted to Dajoong-owned drawings or drawings with explicit commercial-training permission. Public datasets with non-commercial or uncertain drawing rights are `research_eval_only` and must never be mixed into a commercial training split.

Every commercial-training manifest also records the rights holder, permission basis,
evidence reference, reviewer, review date, and whether commercial derivative-model use
is allowed. The training index splits by building/collection group so adjacent sheets
from one project cannot leak across train, validation, and test sets.

The validator in `buili_plan2bim.ground_truth` fails closed when these declarations or the native source hash are absent.
