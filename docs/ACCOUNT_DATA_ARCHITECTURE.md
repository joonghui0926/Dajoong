# Account data architecture

Dajoong keeps identity, metadata, and large BIM artifacts in separate boundaries so
the common project list stays fast while drawings and models remain private.

## Request boundary

- Cloudflare accepts the public request and adds the origin-verification secret.
- Cognito verifies the authorization-code session. The API derives the user only
  from the signed `sub` and verified identity claims.
- The client may request an active workspace with `X-Dajoong-Organization`, but
  the API resolves that ID against the server-side membership table on every
  protected request. It is a selector, never an authorization decision.
- Unauthorized and cross-account object lookups both return `404`, which avoids
  confirming whether another account's job identifier exists.

## Metadata access patterns

The DynamoDB job table keeps the random `job_id` primary key for polling and has
two sparse, time-ordered indexes:

- `owner-id-index`: personal recent projects by `owner_id` and
  `created_at_job`.
- `organization-id-index`: organization projects by `organization_id` and
  `created_at_job`.

Recent-project requests use `Query`, descending sort order, a maximum page size
of 100, and an opaque cursor reconstructed against the authenticated partition.
They do not scan the table or return the large conversion result document.

Every job has `created_at`, `updated_at`, `expires_at`, and `version`. Conditional
writes compare `version` before advancing it. A stale browser or worker receives
a conflict instead of silently overwriting newer data.

Conversion submissions carry a client-generated `Idempotency-Key`. In production,
the authenticated account and that key derive a stable opaque job ID, so a browser
retry cannot enqueue and bill the same conversion twice.

## Artifact and revision storage

- S3 is private, blocks public ACLs and policies, requires TLS, enforces bucket
  ownership, encrypts objects, and keeps object versions.
- Job metadata contains no model bytes. Source sheets, PlanGraph JSON, GLB, IFC,
  overlays, and evidence stay below the private `jobs/<job_id>/` prefix.
- A correction is written to an immutable content-addressed revision directory.
  DynamoDB advances `active_revision` only after all revision objects exist.
  This makes interrupted writes recoverable and allows S3 versioning to retain
  an audit trail during the configured retention window.
- Browser and mobile responses omit S3 paths, account IDs, organization IDs, and
  internal exception details.
- Private API responses force `Cache-Control: private, no-store`; authenticated
  uploads are bounded before they can consume conversion or artifact capacity.

## Deletion and retention

Personal account deletion enumerates the authenticated owner's sparse index,
removes every S3 object version and delete marker, deletes personal job metadata,
then deletes the Cognito identity. Organization-owned records follow the customer
retention policy. Non-owner memberships are removed before identity deletion;
workspace owners must transfer ownership first so company data is never orphaned.

If deletion races an in-flight conversion, the worker's conditional save fails.
It rechecks ownership and either preserves an anonymized organization result or
purges the newly uploaded personal prefix. DynamoDB TTL and S3 lifecycle rules
remain defense-in-depth cleanup, not the primary deletion mechanism.

## Scale and cost posture

DynamoDB uses on-demand billing, S3 holds large values, SQS buffers conversion,
and Fargate workers can return to zero. No always-on database or Redis cluster is
required for project listing, polling, invitations, comments, versions, or
presence.

The collaboration table is a composite-key, single-table design. User partitions
resolve workspace memberships without scans; organization partitions contain
members and hashed invitations; job partitions contain recent comments, activity,
versions, and short-lived presence. Queries paginate within one partition and
never scan the table. Presence expires through DynamoDB TTL, while durable model
snapshots remain content-addressed in S3. Optimistic job versions reject stale
autosaves and restores, and idempotency keys collapse network retries into one
comment, activity event, conversion, or payment.

Organization lookup batches up to 100 records per DynamoDB request, so a user
who belongs to many client workspaces does not create a serial request chain at
sign-in. Membership acceptance and ownership transfer use conditional
transactions. A stale invitation cannot lower an existing role, and an
administrator cannot demote a peer administrator to bypass the removal rule.
