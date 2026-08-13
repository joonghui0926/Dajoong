# The 100 Drawings Without BIM Project

## Campaign premise

Most construction software demos begin with a clean model. Many real projects
do not. Their useful information is still in PDF floor plans, scans, and drawing
archives. Dajoong will convert 100 permissioned sheets and publish an honest,
aggregate account of what automatic drawing-to-BIM can and cannot do.

The first 20 nominations form a private founding cohort. Each selected team gets:

1. one floor-plan sheet converted privately;
2. a colored IFC and selectable semantic GLB when release gates permit;
3. a BIM Build Sheet with source hash, model hashes, entity counts, timings,
   qualification state, and review queue;
4. a 20-minute guided review in Dajoong Studio;
5. the option — never the obligation — to approve a redacted public case card.

## Why this acquires users

- The offer solves a real backlog item instead of giving away generic credits.
- Difficult plans improve audience relevance and expose product limits honestly.
- The Build Sheet gives participants a useful artifact to circulate internally.
- BIM consultants can refer clients and become a delivery channel rather than a displaced labor pool.
- Permissioned cases become durable search content around actual drawing conditions.

## Selection rubric

Prefer nominations that have:

- a real renovation, facility, estimating, or coordination use case;
- a known scale or at least one reliable dimension;
- permission from the drawing owner;
- one representative sheet rather than a full confidential package;
- a reviewer who can attend the handoff;
- complexity that adds new evidence to the campaign.

Reject or defer:

- documents without ownership or processing authority;
- personal residential plans submitted without a professional use case;
- requests for unreviewed construction certification;
- files containing unnecessary personal, security, or tenant information;
- unsupported drawings where physical scale cannot be established.

## Landing page copy

### Metadata

- Title: `The 100 Drawings Without BIM Project | Dajoong`
- Description: `Nominate one legacy floor plan. Dajoong will build a private, review-gated openBIM model and show exactly what was created, what needs review, and where every element came from.`

### Hero

Eyebrow:

`AN OPEN TECHNICAL FIELD TEST`

Headline:

`100 drawings that should have been BIM by now.`

Body:

`We are converting real floor-plan sheets into reviewable IFC and GLB models — without hiding the uncertain parts. Selected teams receive the model, the evidence, and a guided review. Nothing becomes public without written permission.`

Primary CTA:

`Nominate one drawing`

Secondary CTA:

`See what the Build Sheet records`

### Section: This is not a rendering contest

`A good-looking 3D preview is not enough. Dajoong keeps each supported element connected to the source page, model version, confidence, and review state. The output is designed to be inspected, corrected, and handed off.`

Proof labels:

- `IFC authoring exchange`
- `Selectable semantic GLB`
- `Source-linked entities`
- `CPU ONNX runtime`
- `Fail-closed verification`

### Section: What you receive

`The model` — `A colored IFC and browser-ready GLB when the verifier permits export.`

`The Build Sheet` — `Model hashes, source references, element counts, timing, qualification state, and unresolved review work.`

`The review` — `A guided pass through the 2D drawing and 3D model with the highest-risk elements first.`

### Section: What we ask

`Bring one representative sheet, a known scale, and a real use case. You keep ownership of the drawing and generated project data. Public participation and model-training permission are separate, optional decisions.`

### CTA

Headline:

`Give one drawing a model — and a paper trail.`

Button:

`Apply for the founding 20`

Availability note:

`Five active pilots at a time. Applications are reviewed for fit, ownership, and supportability.`

## Nomination form

Required fields:

- Work email
- Organization
- Role
- Building/project type
- What decision will this model support?
- Current source format
- Is physical scale known?
- Number of floors in the eventual project
- Desired downstream software
- Confirmation of authority to process the drawing
- Confidential/private processing acceptance

Optional and separate:

- Upload one redacted preview for selection
- Permission to create a redacted public case card
- Permission to contact about a written training-data agreement

Never combine the last two permissions with service acceptance.

## BIM Build Sheet public schema

The Build Sheet should expose only project-safe metadata:

- opaque project identifier
- source content hash and page number
- physical scale source
- exact primary and semantic model versions/hashes
- difficulty class
- total processing time and hardware description
- counts by entity type
- exported artifact hashes and sizes
- qualification cohort and sample count
- measured/unmeasured/model-mismatch status per claim
- review-required count and reason categories
- verifier release state

Customer identity, S3 paths, source images, and internal exceptions never appear by default.

## Weekly public case format

Title:

`Drawing 07/100 — a renovation sheet with 14 uncertain openings`

Five panels:

1. Redacted source thumbnail, only if explicitly approved
2. 2D recognition overlay
3. Colored model view
4. Build Sheet facts
5. “What the compiler refused to guess” review list

Closing CTA:

`Have a harder one? Nominate it.`

## Consultant referral loop

Offer BIM consultants a co-branded private review session. The consultant keeps
the client relationship and performs or supervises final acceptance. Dajoong
accelerates the first pass and records corrections. The campaign does not frame
automation as replacing the consultant.

## Follow-up offer

After the private pilot:

- portfolio drawing-debt audit;
- paid multi-floor conversion and assembly;
- team Studio access;
- integration discovery for Autodesk/Procore workflows;
- optional design-partner agreement for recurrent drawing types.
