# Public claims register

Every public claim must point to current code, a release manifest, or a sealed
benchmark for the exact production model pair. Marketing may simplify wording;
it may not widen the measured scope.

## Allowed now

| Claim | Evidence | Required qualifier |
| --- | --- | --- |
| Dajoong converts floor-plan images or PDF pages into BIM artifacts | `modules/plan2bim/README.md` and converter API | Physical scale is required |
| The bundled primary checkpoint has 86,533 parameters and is about 380 KB | `modules/plan2bim/README.md` | Refer only to the bundled primary checkpoint |
| Inference uses ONNX Runtime on CPU | module README and runtime dependency | Do not say every full job is real-time |
| IFC is the canonical editable exchange and GLB is a colored browser derivative | module README output contract | GLB is not a substitute authoring model |
| Supported entities retain source and review metadata | PlanGraph/GLB output contract | Say “supported entities,” not every possible symbol |
| Export is fail-closed for specified contradictions | module/building verifier | Name the tested contradiction if specificity matters |
| Web Android and iOS clients use Dajoong-owned services | root README and release architecture | Conversion does not run on the customer device |
| Customer content is not used for model training without separate written permission | current privacy page | Keep the permission separate and explicit |

## Conditional claims

| Claim | Gate before publication |
| --- | --- |
| Accuracy/F1 for walls openings rooms or objects | Sealed benchmark, exact production hashes, class, sample count, metric definition, and confidence interval |
| Conversion time | Hardware, warm/cold state, input resolution, sheet complexity, exact pipeline scope, and percentile distribution |
| Cost per square foot | Measured AWS/support cost including retries storage review and failed jobs; disclose pricing unit |
| “Faster/cheaper/more accurate than” a competitor | Reproducible head-to-head protocol with equivalent input, output LOD, review, and delivery scope |
| Construction-ready | Qualified project review and an acceptance contract for that project |
| Works with any drawing | Never unless an appropriately broad, independently defensible benchmark exists |

## Prohibited now

- `100% accurate`
- `under 0.9% error on any drawing`
- `zero latency`
- `every floor in under two seconds`
- `construction-ready without review`
- `better than Twindo/Plans2BIM/Autodesk/Procore`
- `cheapest in the market`
- `the model runs in the customer's browser or app`
- use of the non-commercial CubiCasa research teacher as a production claim

## Approved public vocabulary

- review-gated
- evidence-linked
- source-linked
- deterministic metric compiler
- compact private CPU inference
- openBIM IFC
- selectable semantic GLB
- private by default
- refuses unsupported guesses

## Avoid

- magic
- perfect
- one-click construction model
- human-free
- autonomous architect
- digital twin, unless the delivered information scope satisfies the actual use case
