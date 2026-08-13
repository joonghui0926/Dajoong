# Dajoong Plan2BIM

This directory is the standalone drawing-to-BIM portion of Dajoong. It contains the
lightweight AEC checkpoint and the CPU inference path. It also contains the
deterministic metric compiler, verifier, colored IFC exporter, and selectable
GLB derivative exporter.

The module has no dependency on the Dajoong web app, API, upload jobs, or browser
editor. A caller supplies one floor plan image or PDF page and its physical scale. The
module emits an IFC4 model plus the evidence and verification artifacts needed
to audit the conversion. Product-layer review and editing live in
`apps/plan2bim-studio`.

## Active and archived paths

There is one default product runtime. Its filenames, hashes, qualification
manifest, release state, and canonical evaluation are pinned in
`src/buili_plan2bim/models/ACTIVE_RUNTIME.json`. Product code reads that registry;
it does not select a checkpoint by scanning a directory.

- `src/buili_plan2bim/`: active converter code
- `scripts/`: active build, ground-truth, asset, and final-pipeline evaluation commands
- `research/legacy-*`: reproducibility-only code that is never imported by the product
- `artifacts/plan2bim-current/`: the only current full-sheet conversion output
- `artifacts/plan2bim-current-evaluation/`: the only current canonical evaluation
- `artifacts/_superseded-*/`: archived outputs that are never runtime or evaluation inputs

Private ONNX bytes are injected on the server. A private bundle cannot replace
`ACTIVE_RUNTIME.json`, and a hash mismatch fails before inference.

## Install

```powershell
cd C:\Users\jjoon\OneDrive\Documents\Dajoong\modules\plan2bim
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

The runtime uses ONNX Runtime on CPU. PyTorch and a GPU are not required.

## Command line

```powershell
dajoong-plan2bim plan.png out --pixels-per-meter 100
```

PDF pages preserve the original file hash and page number in provenance:

```powershell
dajoong-plan2bim drawing-set.pdf out --page 3 --pdf-dpi 300 --pixels-per-meter 118.11
```

The active content-addressed full-sheet semantic model adds wall masks, openings,
fixed-object boxes, and labels from the same raster input. Server deployments set
`DAJOONG_SEMANTIC_MODEL_PATH`; a private bundle may instead install the pinned
`models/semantic-model.onnx`. The default full-sheet limit is 2048 pixels and is
read from `ACTIVE_RUNTIME.json`, so command-line and server conversions cannot
silently fall back to the old 1024-pixel path.

An explicit model path remains available for isolated research runs:

```powershell
dajoong-plan2bim plan.png out --pixels-per-meter 100 `
  --semantic-model C:\path\to\semantic-model.onnx
```

The semantic model must have a sibling `.onnx.json` manifest. The module checks
its SHA-256 and license/release flags before inference.

Physical scale is required because a raster image alone does not determine real
world dimensions. For a 1:100 drawing rendered at 300 DPI, calculate the exact
pixel scale from the sheet metadata instead of guessing.

Outputs:

- `00-semantic-recognition.json`: optional full-sheet boxes, labels, wall vectors,
  confidence, and promotion state
- `00-spatial-evidence-graph.json`: Method v2 global topology, expert evidence,
  host/containment relations, coverage gaps, and fail-closed release blockers
- `00-semantic-overlay.png`: optional raster-only recognition overlay
- `01-evidence.json`: tile coverage, raw proposals, residual evidence, and timing
- `02-bim-program.json`: metric BIM instructions with source references
- `03-plan-graph.json`: deterministic graph consumed by 3D exporters
- `04-model.ifc`: colored IFC4 model suitable for IFC-compatible CAD/BIM viewers
- `04-model.ifc.certificate.json`: fail-closed verification certificate
- `04-model.glb`: colored browser derivative with one selectable semantic node
  per room, wall, opening, and fixed object
- `04-model.glb.manifest.json`: GLB role, hash, element count, and authoring-format note
- `conversion-manifest.json`: hashes, entity counts, status, and end-to-end timing

## Python API

```python
from buili_plan2bim import convert_image

result = convert_image(
    "plan.png",
    "out",
    pixels_per_meter=100.0,
    project_id="demo",
    level_id="L1",
)
print(result.ifc_path)
```

For repeated conversions, construct `Plan2BimConverter` once so the ONNX session
is reused.

## Multi-level building conversion

`dajoong-building2bim` maps explicit PDF pages into one building PlanGraph, IFC, and GLB. Every level keeps its page, metric scale, elevation, alignment, and source hash.

```json
{
  "project_id": "commercial-building",
  "pdf_dpi": 300,
  "levels": [
    {
      "source_path": "drawing-set.pdf",
      "page_number": 14,
      "level_id": "L1",
      "name": "Ground floor",
      "elevation_m": 0.0,
      "pixels_per_meter": 118.4
    },
    {
      "source_path": "drawing-set.pdf",
      "page_number": 20,
      "level_id": "L2",
      "name": "Second floor",
      "elevation_m": 3.6,
      "pixels_per_meter": 118.4
    }
  ],
  "vertical_connections": []
}
```

```powershell
dajoong-building2bim building.json output/building
```

The empty connection list is deliberate. A stair, ramp, or escalator must join adjacent levels and its footprint must fit both connected floors before release.
The building converter also writes `05-building-consistency.json`. This content-addressed
report records the ordered levels, invariant counts, release state, and structured findings
used by Studio. Export is blocked for reversed or duplicated vertical connections, level or
wall volume overlap, objects crossing the next level, and misaligned or discontinuous named
shafts. A connection may carry `shaft_id` so a riser or lift run can be checked across every
participating floor. Connection evidence is retained from both endpoint drawings.

## Output contract

IFC/PlanGraph is the canonical editable BIM exchange. GLB is a colored browser
view derivative, not a substitute for an authoring model. Semantic GLB nodes
carry stable element IDs, level, family/type, confidence, review state, source
references, editable field values, and the fields the Studio may edit. Doors also
carry operation type, host-wall-relative hinge handing, and swing side. IFC maps
resolved operations to the corresponding door operation enum. A door with
unresolved operation semantics fails the release verifier and remains in review.
PlanGraph also carries level-scoped driving dimensions and explicit geometric
constraints created by Studio. Coincident wall references are verified against
their metric endpoints, and invalid or zero-length dimensions fail closed. These
authoring records remain canonical in PlanGraph even though browser GLB geometry
does not render them as building components.

Full-building assembly is explicit: callers provide each converted level,
elevation, alignment transform, and any verified vertical connection. The
assembler never invents a stair or connects the top level to a non-existent
floor. The verifier also rejects corrected graphs that reverse the endpoints or
move a named shaft segment away from its shared centerline.

The module is fail-closed. It does not promise arbitrary-drawing 100% inference.
An entity is either supported by evidence or kept review-required; the separate
Studio is responsible for audited human acceptance and corrections.

## Family asset contract

Recognition establishes semantic type, metric envelope, pose, and evidence. The
offline family resolver then chooses only among license-audited candidates for
that exact semantic family. `dajoong-context-shape-ranker-v1` compares the
measured 3-D envelope with precomputed asset proportions, then adds room use,
wall proximity, installation context, nearby component families and an optional reviewed override. A
single grid index is built for the sheet, so looking up surrounding components
does not become an all-pairs scan as projects grow. It
records the winning components, margin and three alternates for audit. This is a
small deterministic rank over an in-memory candidate list, not another network
or generative-model call. A generic appliance is never silently upgraded to a
refrigerator or dishwasher.

Asset geometry is content-addressed and stored once in `PlanGraph.family_assets`.
Fixtures carry stable `geometry_ref` and metric scale fields, so a 1,000-object
project does not duplicate the same mesh 1,000 times. IFC, GLB, Studio web, and
the mobile shell all consume this same converter-owned contract. The conversion
path performs no network request; catalog metadata and normalized meshes are
cached in-process. Every licensed fixture retains provider, author, license,
source URI, source hash, selection score, and exact evidence-envelope fit.

The production server pack currently contains 164 unique content-addressed
geometries: 116 audited licensed orientations and 48 Dajoong parametric building
system variants. No mesh bytes ship in the initial browser bundle. Studio first
renders walls and rooms, requests only the visible level's referenced meshes,
limits fetch concurrency to four, and keeps a 64-entry LRU promise cache.
Immutable mesh hashes are cached by Cloudflare for one year; the authoritative
graph and model parameters remain server-only.

`asset_audit` verifies mesh indices, dimensions, base elevation, and license
allowlists before export. Unknown classes remain semantic review markers rather
than deceptive solid boxes. This contract lets the model and family catalog be
versioned and replaced independently from the product applications.

## Bundled lightweight model

The active reconstruction design is documented in [METHOD_V2.md](METHOD_V2.md).
The runtime materializes its compiler proposal only from the complete-sheet
spatial evidence graph. Local element experts can contribute evidence nodes, but
cannot bypass global host, containment, or omission checks. The active registry
rejects legacy local-first execution methods.

For Dajoong-owned topology pretraining data only:

```powershell
python scripts/generate_synthetic_pretraining.py artifacts/topology-source --count 1000
python scripts/build_synthetic_topology_targets.py artifacts/topology-source artifacts/topology-targets
python scripts/train_global_topology_student.py artifacts/topology-source artifacts/topology-targets artifacts/topology-model --device cuda
```

Both manifests are fail-closed as synthetic pretraining material and cannot be
used as real-drawing ground truth or evaluation data.

Large derived crop corpora are ephemeral. Build them outside synced project
folders, retain only the sealed checkpoint and evaluation report, then remove
the corpus after the run. `scripts/prune_ephemeral_training_artifacts.py` lists
only allowlisted derived-corpus directories in dry-run mode; pass `--execute`
to remove that list. Direct source-pixel ground truth, checkpoints, evaluation
reports, and server asset packs are protected from this command by design.
For the drive-level `DajoongTrainingTemp` scratch directory only,
`--purge-exact-scratch-root` broadens the dry run to every immediate child;
combine it with `--execute` only after reviewing that list. The command refuses
that mode for any other directory.

Direct real-sheet calibration uses a separate three-state proposal contract.
Each directly annotated object contributes one best complete-object native
proposal at the same IoU 0.50 boundary used by geometry evaluation. Clear ink
fragments, multi-object envelopes, and unrelated marks are background; partial
or alternate complete-object overlaps are ignored instead of being mislabeled
as background. Hard negatives are sampled across the full candidate ledger so
large sheets do not train only on the first page region.

At runtime, uncertain equipment candidates are no longer discarded before the
building is considered. They may reach the set decoder only with a strong
aligned-run graph signal, and survive only when two independently accepted
neighbors support the same casework, appliance, or plumbing run. Instance
clustering then selects one consensus source extent before BIM compilation.

The bundled `dajoong-aec-global-enclosure-router-v1` checkpoint has 86,533
parameters and is about 380 KB. Its input has four channels: raster ink,
horizontal support, vertical support, and a global enclosure view. This keeps
large drawing context visible while preserving dense local evidence. A Fourier
wall prior assists orientation auditing. It never removes geometry by itself.

The runtime verifies the checkpoint SHA-256 before inference. The current model
manifest sets `authoritative_decisions` to `false`. Therefore exported IFC files
are drafts marked **NOT FOR CONSTRUCTION** and require review. The module does
not claim that arbitrary drawings are converted without error.

The CubiCasa teacher used for the local quality experiment is separately stored
as `non-commercial-research-only`; it is not bundled with this package and is
never presented as a production Dajoong checkpoint. A production semantic model
must be trained from commercially compatible or Dajoong-owned labels while
preserving the same manifest and output contract.

## Ownership and provenance

Ground-truth creation follows [GROUND_TRUTH_POLICY.md](GROUND_TRUTH_POLICY.md). Model output,
pseudo-labels and candidate graphs are never ground truth. Only direct visual annotation of the
native-resolution source drawing, followed by a whole-sheet omission scan, may enter a benchmark.
The validator fails closed and keeps research-only drawings out of commercial training splits.


The extracted compiler source and model are copied from the local
`Dajoong-Spatial-Compiler` workspace. They remain covered by the included
Dajoong Proprietary Software License. The ONNX artifact SHA-256 is:

`36bcfe230be22ed869eb7bc3a940805c516dd0970c66649f944f0d5451ff1817`
