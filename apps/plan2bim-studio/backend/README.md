# Dajoong Studio API

This is the project layer around the pure `buili-plan2bim` converter. It owns
uploads, asynchronous jobs, downloads, correction patches, and human review
history. None of these concerns are imported back into the converter module.

```powershell
cd apps/plan2bim-studio/backend
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ..\..\..\modules\plan2bim -e ".[dev]"
.\.venv\Scripts\uvicorn buili_plan2bim_studio.main:app --reload --port 8042
```

Set `DAJOONG_STUDIO_DATA` to choose the file-backed job directory and
`DAJOONG_STUDIO_ORIGINS` to allow the frontend origin. Production should replace
the local executor and file store by setting `DAJOONG_RUNTIME=aws`. The AWS
deployment sets `DAJOONG_MAX_UPLOAD_BYTES` to 100 MB so authenticated uploads
cannot consume unbounded conversion capacity. Its adapter keeps job state in
DynamoDB, artifacts in S3, and work in SQS without
changing the REST contract.

Company collaboration is kept in a separate DynamoDB single-table boundary set
by `DAJOONG_COLLABORATION_TABLE`. It stores organization membership, hashed
invitations, element-linked comments, model-version metadata, activity, and
short-lived presence. A signed user ID alone never grants company access: every
request validates the selected organization against an active membership. Set
`DAJOONG_APP_URL` for invitation links and optionally set a verified
`DAJOONG_INVITE_FROM_EMAIL` SES sender; without SES the API returns a secure link
that an administrator can copy.

Production startup fails closed unless `DAJOONG_RUNTIME=aws` and authentication
is enabled. Client-supplied model paths and thread counts are not accepted;
workers read the private model path and bounded CPU settings from their own
environment.

## Reliability invariants

- Conversion and checkout requests carry account-scoped idempotency keys, so a
  browser retry does not create a second job, credit reservation, or order.
- Queue submissions are stored with the job before SQS delivery. A retry can
  safely re-enqueue a submission after an API or network interruption.
- Workers claim an expiring DynamoDB lease and renew both that lease and SQS
  visibility while converting. Duplicate delivery is acknowledged without
  running a second conversion; a terminated worker becomes reclaimable.
- Failed messages remain unacknowledged and move through the queue redrive
  policy to the DLQ instead of killing the worker process.
- API request handlers never run S3 uploads or token-key refreshes on the async
  event loop. Each response includes an `X-Request-ID` for end-to-end tracing.
- Immutable uploaded S3 keys are stored on the job, so source and render
  retrieval cannot drift from the input selected by the winning request.
- Company purchases resolve to the active organization account. Members keep
  individual sign-ins and roles, while projects, credits, versions, and audit
  records remain company-owned.
- Invitation acceptance and ownership transfer are atomic. Raw invitation
  tokens are never persisted or returned in workspace-list responses.

`DAJOONG_JOB_VISIBILITY_SECONDS` must match the SQS visibility setting.
`DAJOONG_CONVERSION_THREADS` and `DAJOONG_CONVERSION_BATCH_SIZE` are worker-side
capacity controls rather than public request parameters.
The optional private model is stored outside the client bundle at
`DAJOONG_SEMANTIC_MODEL_S3_KEY`; each warm worker downloads it once, verifies
`DAJOONG_SEMANTIC_MODEL_SHA256`, and reuses the local copy for subsequent jobs.

Account project history is a paginated DynamoDB `Query`, not a table scan.
Personal and organization indexes are time ordered, job writes use optimistic
versions, and correction patches become immutable S3 revisions before the active
pointer advances. See `../../../docs/ACCOUNT_DATA_ARCHITECTURE.md` for the full
data, retention, conflict, and deletion contract.
