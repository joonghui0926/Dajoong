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

## Install

```powershell
cd C:\Users\jjoon\OneDrive\Documents\Buili\modules\plan2bim
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

An optional content-addressed full-sheet semantic model can add wall masks,
openings, fixed-object boxes, and labels directly from the same raster input:

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

## Bundled lightweight model

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

The extracted compiler source and model are copied from the local
`Dajoong-Spatial-Compiler` workspace. They remain covered by the included
Dajoong Proprietary Software License. The ONNX artifact SHA-256 is:

`36bcfe230be22ed869eb7bc3a940805c516dd0970c66649f944f0d5451ff1817`
