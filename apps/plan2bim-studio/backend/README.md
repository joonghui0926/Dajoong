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

Account project history is a paginated DynamoDB `Query`, not a table scan.
Personal and organization indexes are time ordered, job writes use optimistic
versions, and correction patches become immutable S3 revisions before the active
pointer advances. See `../../../docs/ACCOUNT_DATA_ARCHITECTURE.md` for the full
data, retention, conflict, and deletion contract.
