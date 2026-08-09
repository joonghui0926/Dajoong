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
adapter keeps job state in DynamoDB, artifacts in S3, and work in SQS without
changing the REST contract.
