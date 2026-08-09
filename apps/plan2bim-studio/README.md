# Dajoong Plan2BIM Studio

Dajoong keeps the conversion core and product surface separate.

`modules/plan2bim` owns one contract: drawing input to proof carrying IFC, GLB, PlanGraph, and audit artifacts.

This project owns the user workflow around that contract:

- PDF and image upload
- queued conversion jobs
- multi-page PDF to multi-level building assembly
- linked 2D and 3D review
- a shared 2D/3D model browser with persistent category and per-element visibility and edit locks
- shared 2D/3D selection isolation with one-click view and lock recovery actions
- one BIM-aware context menu across the plan, model, and Model Browser surfaces
- graph-backed navigation between hosts, contained elements, constraint chains, and MEP systems
- element correction and confidence review
- Ctrl or Command multi-selection with batch properties
- Autodesk-style left-to-right Window and right-to-left Crossing area selection
- cursor-centered wheel zoom, Space or middle-button pan, touch pinch navigation,
  Fit All, and semantic Fit Selection in the 2D editor
- geometry-aware hover preselection and Tab or repeated-click selection cycling for
  stacked rooms, walls, openings, routes, equipment, and dimensions
- persistent Autodesk-style selection filters shared by 2D and 3D picking with
  per-level category counts and one-click reset
- mixed-value-safe batch editing with one-step undo
- room-boundary vertex editing with midpoint insertion, context removal, wall-axis
  snapping, overlap prevention, minimum-edge and area guards, plus contained-object
  reassignment in the same audited Undo transaction
- contextual duplicate, align, distribute, accept, and delete actions
- object rotation handles with precise and coarse angular snapping
- keyboard nudging, duplication, deletion, and selection shortcuts
- configurable 10 mm to 500 mm plan snapping
- endpoint, midpoint, orthogonal, and temporary free placement snapping
- unit-aware length and angle input while drawing walls or measurements
- continuous wall chains that continue from the last exact endpoint
- exact signed X and Y translation for mixed BIM selections with host-aware opening guards
- exact signed group rotation about an editable metric or imperial pivot
- single and multi-selection 3D move and shared-pivot rotation gizmos that commit through the same PlanGraph guards as 2D
- exact component Mirror and Linear Array commands with non-overlapping defaults and one-step undo
- selectable, level-scoped driving dimensions with direct endpoint handles
- persistent coincident wall-endpoint constraints with connected-wall propagation
- new-level creation and relationship-preserving Copy to level
- hosted opening drag constrained to its wall, extents, and clear neighboring span
- wall-relative door swing arcs with direct width handles
- explicit door operation, hinge, and swing-side correction
- driving wall-length dimensions plus two-wall join and corner commands
- searchable BIM family insertion with actual-size cursor ghosts, host-aware snapping,
  room assignment, live collision guards, rotation, and continuous placement, plus batch replacement
- evidence-bound model assurance with drawing difficulty and per-claim metrics
- a fail-closed review gate that opens directly into the guided element queue
- deterministic building-integrity findings with one-click navigation to the affected level or element
- selectable MEP routes and vertical circulation in 2D and 3D
- active-level PDF evidence synchronization
- immutable correction patches
- artifact downloads
- responsive web, PWA, Android, and iOS clients

The converter remains an input-to-output package. Every editing, review,
download, and patch workflow above belongs to Studio so automation can be
embedded elsewhere without pulling in product UI state.

## Editing interaction contract

- A pointer gesture previews continuously but commits as one audit operation and
  one undo step.
- Dragging left-to-right selects only elements fully enclosed by the blue Window.
  Dragging right-to-left selects every element touched by the beige Crossing box.
  Room fills preserve click selection while allowing either area gesture to begin
  directly over the plan, and Ctrl, Command, or Shift adds matches to the current set.
- Wheel zoom preserves the drawing point under the cursor. Space plus primary drag or
  middle-button drag pans without starting a model mutation. Two-finger pinch combines
  zoom and pan on touch devices. `F` frames the selected BIM footprints and `Home`
  restores the active level, while the on-canvas controls expose the same commands.
- Hover preselection evaluates semantic footprints at the current screen scale. When
  several BIM entities occupy the pointer location, the candidate HUD shows type and ID.
  Tab and Shift+Tab preview the next or previous candidate, Enter confirms it, and
  repeated clicks cycle without requiring category visibility changes.
- Selection filters change pick eligibility without hiding model geometry. Walls,
  openings, rooms, objects, systems, circulation, dimensions, and constraints can be
  toggled independently. The same exclusions apply to Window, Crossing, hover cycling,
  Ctrl or Command+A, and 3D raycasting, while Model Browser navigation remains available.
- Family insertion enters a guided placement mode instead of creating an object at an
  arbitrary default point. The cursor carries the actual rotated footprint. Floor,
  wall-hosted, and ceiling families use separate collision bands; wall components align
  to a nearby host; valid placements inherit the smallest containing room. `R` rotates,
  Shift+`R` reverses, click places continuously, and Escape or Finish ends the command.
- The same component placement solver guards every correction path. Canvas dragging,
  property edits, Exact Move, exact rotation, 3D gizmos, family replacement, duplication,
  Mirror, Array, align, and distribute all refresh room and host references before one
  audited commit. Invalid commands retain the original model and report the conflicting
  room, wall, or object. Duplicate searches outward for the nearest clear location.
- Wall and measurement tools accept meters, millimeters, centimeters, inches,
  and feet. Enter places the exact segment. A wall command continues from the
  new endpoint until Escape ends the chain.
- `M` opens Exact Move for supported geometry. The command accepts signed unit
  values, records one undo step, keeps openings on their host wall, and blocks
  collisions instead of silently detaching BIM relationships.
- `R` opens Exact Rotate for placed components. The command accepts signed relative
  angles, exposes common angle presets, and defaults the editable pivot to the current
  selection center. Multi-selection coordinates and yaw values update in one audited
  undo transaction.
- Selected walls, openings, routes, placed components, or vertical connections expose
  one shared 3D move gizmo. Fixtures and vertical connections also expose shared-pivot
  vertical-axis rotation for single or multi-selection.
  Orbit is disabled while dragging, the active snap increment is reused, and a rejected
  host or collision change returns the object to its original position.
- Shift+M opens Mirror and Shift+A opens Linear Array for placed components. Mirror uses
  an exact X or Y line and defaults beyond the selected footprint to avoid stacked copies.
  Array count includes the source set and accepts signed unit-aware X and Y spacing.
  Either operation is one audited undo transaction.
- Selected rooms expose primary circular grips and secondary diamond grips. Circular
  grips move with grid, endpoint, and wall-axis snap. Dragging a diamond inserts a
  vertex; right clicking a circle removes it. New overlaps, self intersections,
  sub-20 mm edges, and degenerate areas are rejected before commit. Objects that
  genuinely cross a changed boundary are reassigned to the smallest containing room,
  or moved to the review queue when no room contains them.
- Object rotation snaps to 1 degree, Shift snaps to 15 degrees, and Alt permits a
  temporary free angle.
- Ctrl or Command+D duplicates supported BIM components. Arrow keys nudge by the
  active grid, Shift multiplies the step by ten, and Alt uses a 10 mm step.
- Alignment uses actual element footprints rather than only their center points.
- Distribution keeps the outer anchors fixed and spaces the intermediate element
  centers evenly.
- Hosted dependencies block destructive edits until the user includes or rehosts
  the dependent elements.
- Door width handles stay on the host wall. H flips the hinge and S reverses the
  host-wall-relative swing side. Each command remains one audited undo step.
- Wall length is a driving metric dimension. Join averages the nearest endpoints,
  while Corner trims or extends two nonparallel wall axes to their exact intersection.
  Both commands persist a coincident constraint, so later endpoint edits propagate.
- Wall geometry is edited as a BIM relationship package, not as an isolated line.
  Endpoint handles, property values, Exact Move, Join, Corner, and the 3D gizmo all
  use the same preflight planner. A valid change updates coincident wall endpoints,
  hosted doors and windows, room polygons, and wall-mounted equipment in one Undo
  step while recording a separate audit operation for every dependent entity. Changes
  that collapse a wall, self-intersect a room, detach an opening, or create a fixture
  clearance conflict are rejected before the graph is mutated.
- Dimensions are PlanGraph entities rather than view-only annotations. Their endpoints
  and exact length can be edited from either the canvas or property palette.
- Model Assurance shows the conversion certificate separately from statistical model
  qualification. `Locate` reveals hidden elements, preserves active isolation, switches to
  the affected level, and selects the exact entity in both 2D and 3D.
- Door and window insertion chooses the largest clear host-wall span. Moving, resizing,
  rehosting, or batch-changing an opening is rejected when it leaves the wall or overlaps
  another hosted opening; rejected property values immediately return to the model value.
- Copy to level expands the selection only where BIM integrity requires it. Hosted
  openings follow their wall, assigned objects bring their room, and copied constraints
  receive new wall references. Vertical connections are excluded because they require
  an explicit source and destination level. The complete copy is one undo transaction.
- Model Browser visibility is view state shared by 2D and 3D, not a model mutation.
  Category and element locks are also saved as view state, but are enforced at every
  mutation entry: direct manipulation, properties, context actions, keyboard commands,
  and creation tools. Indirect edits are expanded before the lock check, so moving a
  constrained wall cannot silently move a locked connected wall or hosted opening.
  Locking during a gesture cancels the preview before the lock takes effect.
- Isolation is one persisted view state shared by the plan and model canvases. The
  selection toolbar, 3D view tool, `I` shortcut, and Model Browser all operate on that
  same state. `Show all`, `Unlock all`, and `Exit isolate` provide visible recovery paths.
  New and duplicated elements remain visible when created inside an isolated context.
- Right click preserves an existing multi-selection and opens the same nearby actions
  in 2D, 3D, and the Model Browser. View actions remain available on locked elements,
  while model mutations are disabled or routed through the existing dependency guards.
  Arrow keys navigate the menu and Escape closes it without discarding the selection.
- Relationship actions query PlanGraph references rather than UI naming conventions.
  They can select an opening host, every opening on a wall, contained room objects,
  assigned room walls, constrained wall chains, and routes in one building system.
  Related targets are revealed and added to the current isolation context automatically.

## Local development

Terminal one:

```powershell
cd apps/plan2bim-studio/backend
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ..\..\..\modules\plan2bim -e ".[dev]"
.\.venv\Scripts\uvicorn buili_plan2bim_studio.main:app --reload --port 8042
```

Terminal two:

```powershell
cd apps/plan2bim-studio/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` for the landing page or `/studio` for the editor.

The Convert dialog supports one level and building set modes. Building set mode maps each PDF page to an explicit level, scale, and elevation, then returns one editable building graph, IFC, and GLB. Vertical connections remain unassigned until confirmed.

The converter writes `00-drawing-complexity.json` and
`00-model-qualification.json` beside every result. Studio exposes those records
through the `REVIEW GATE` control without confusing a class-specific metric with
whole-model accuracy.

## Mobile

The shared React client is wrapped with Capacitor. Native projects live in `frontend/android` and `frontend/ios`.

```powershell
cd apps/plan2bim-studio/frontend
$env:VITE_STUDIO_API_URL="https://your-api.example.com"
npm run mobile:sync
```

Android can then open in Android Studio. The iOS project must be signed and built on macOS with Xcode.

## AWS production foundation

Terraform under `infra/terraform` creates private S3 storage, DynamoDB job state, an encrypted SQS queue, App Runner API, autoscaled Fargate Spot workers, CloudFront delivery, ECR repositories, and a Cognito user pool.

Redis is deliberately excluded from the first production shape. Job state is durable in DynamoDB and artifacts live in S3. Add a managed Redis service when real time multiuser presence or high frequency shared editing is introduced.

The API selects its runtime through `DAJOONG_RUNTIME`:

- `local` uses the file store and local executor
- `aws` uses S3, DynamoDB, and SQS through `AwsJobGateway`

Build the API and worker image from the repository root because the Docker image includes the independent converter package.

```powershell
docker build -f apps/plan2bim-studio/backend/Dockerfile -t dajoong-plan2bim .
```

Platform credentials and final domains are intentionally external. Use immutable image tags in Terraform and enable the Cognito login flow before a public production release.
