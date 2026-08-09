# Dajoong Plan2BIM Studio frontend

The Studio is deliberately separate from the converter. It imports a source
PlanGraph and optionally its drawing image, then produces a correction patch and
a corrected graph. The source conversion bundle remains immutable.

```powershell
cd apps/plan2bim-studio/frontend
npm install
npm run dev
```

The current correction scope includes linked 2D/3D selection, element browsing,
low-confidence review queues, wall endpoint dragging, object dragging, property
editing, host-aware opening insertion, audited add/update/delete/accept actions,
undo/redo, local autosave, section cuts, model isolation, view presets, command
search, shared 2D/3D category and per-element visibility, fail-safe edit locks,
shared selection isolation, global view recovery actions, and patch/graph downloads.
Visibility, isolation, and lock preferences persist independently
of the immutable source model and correction patch. Lock checks include constrained
wall and hosted-opening side effects before a command can mutate the graph.

Doors and windows can be rehosted from the action bar, element context menu,
Properties panel, command search, or `Shift+H`. Pick Host mode reveals wall
candidates in plan, treats the clicked position as the preferred location, finds
the nearest collision-free span, and preserves a door's world-space leaf and
swing orientation even when the replacement wall runs in the opposite direction.
The host change is recorded with its dependent changes as one auditable Undo step.

Multi-selection Arrange follows CAD key-object behavior: the last-selected
component stays fixed while six edge or center alignment modes move the other
items. Horizontal and vertical distribution use equal clear gaps instead of
equal center spacing, keep both outer anchors fixed, and account for rotated
plan footprints. Opening hosts, room containment, object clearances, work-plane
consistency, locks, audit operations, and one-step Undo are validated before the
command commits. The compact action-bar menu, context menu, command search, and
`Alt+X/Y` shortcuts expose the same workflow.

The internal BIM clipboard supports `Ctrl+C` and `Ctrl+V` across the action
bar, global command bar, context menu, and command search. Copying walls also
captures hosted openings, wall-mounted components, and constraints whose wall
references are complete. Paste allocates stable unique IDs, rewires host and
constraint references, searches for the nearest clear shared offset, validates
wall and room overlap, finds usable opening spans, refreshes fixture containment,
and records the full bundle as one auditable Undo transaction. Copying to a
different level prefers the original coordinates when that work plane is clear.

The bottom edit-history timeline exposes every undoable BIM state in chronological
order. Each entry uses the audited command reason, affected category, and entity
ID, while multi-entity commands remain one visible step. Selecting any past or
future state restores its graph and correction patch together. A new edit from
an earlier state starts a clean branch, matching the existing Undo and Redo model.
The timeline can be collapsed and is also available through command search.

The plan canvas, Three.js model canvas, and Model Browser share one accessible
element context menu for view, lock, duplication, review, and delete actions.
The same menu resolves PlanGraph relationships for host, containment, constraint,
and MEP system navigation without scanning raw entity IDs.

The Model Browser is optimized for dense commercial models. Categories start
collapsed, natural numeric ordering keeps generated IDs predictable, and search
or review filters expand only matching sections. A selection made in either
viewport reveals and scrolls to the corresponding browser row. `Shift` selects
a contiguous visible range, `Ctrl` adds or removes individual rows, and the last
clicked row remains the key object used by precise Arrange commands.

Guided Review ranks editable BIM elements by review risk rather than raw model
confidence alone. It combines drawing difficulty, element criticality, human
review state, broken BIM relationships, deterministic verifier findings, and
the qualification status of the measured claim. Review Next, the Model Browser,
Properties, and Model Assurance expose the same ranking and evidence. The score
is a triage signal; sealed benchmark metrics keep their original measured scope.

The `Ctrl+K` command palette supports keyboard-first editing with ranked BIM
aliases, recent-command recall, visible shortcuts, and contextual availability.
Disabled results stay discoverable and explain which selection, lock, host, or
history condition must change before the command can run. Arrow keys, Home, End,
and Enter provide a complete no-mouse command path.

Named BIM selection sets preserve reusable coordination scopes inside the local
project session. A set can span categories and levels, survive reloads, recall
its primary level, or isolate the same members across the linked plan and model.
Model Browser controls support create, rename, recall, isolate, and delete, while
dynamic command-palette entries provide fast keyboard access to every saved set.
Stale entity references are removed whenever model geometry changes.

Optional workspaces use route-local lazy chunks. Conversion, model assurance,
family browsing, exact transforms, pattern tools, and command search load when
opened, reducing the primary application chunk while retaining an immediate
loading state. The Vite CSS pipeline is isolated from machine-level PostCSS
configuration so production builds remain reproducible across developer hosts.

The interface focuses on the review and correction loop that follows automated
conversion. Native Android and iOS projects use the same application through
Capacitor. Run `npm run mobile:sync` after a web build.
