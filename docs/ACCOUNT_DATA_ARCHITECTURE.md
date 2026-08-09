# Account data architecture

Dajoong keeps identity, metadata, and large BIM artifacts in separate boundaries so
the common project list stays fast while drawings and models remain private.

## Request boundary

- Cloudflare accepts the public request and adds the origin-verification secret.
- Cognito verifies the authorization-code session. The API derives ownership only
  from the signed `sub` and administrator-issued `custom:organization_id` claims.
- A client-supplied account or organization identifier is never accepted as an
  authorization decision.
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
retention policy, but the deleted user's `owner_id` is removed immediately.

If deletion races an in-flight conversion, the worker's conditional save fails.
It rechecks ownership and either preserves an anonymized organization result or
purges the newly uploaded personal prefix. DynamoDB TTL and S3 lifecycle rules
remain defense-in-depth cleanup, not the primary deletion mechanism.

## Scale and cost posture

DynamoDB uses on-demand billing, S3 holds large values, SQS buffers conversion,
and Fargate workers can return to zero. No always-on database or Redis cluster is
required for project listing, polling, or low-contention corrections. A dedicated
collaboration service should be introduced only when live multi-user cursors or
high-frequency shared editing justify it.
