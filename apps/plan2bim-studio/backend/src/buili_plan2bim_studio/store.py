from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StudioJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str = "queued"
    source_name: str
    source_key: str = ""
    project_id: str = "dajoong-project"
    output_dir: str
    owner_id: str = ""
    organization_id: str = ""
    created_at: int = 0
    updated_at: int = 0
    expires_at: int = 0
    version: int = 0
    active_revision: str = ""
    graph_sha256: str = ""
    lease_until: int = 0
    error: str = ""
    submission: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class StudioJobPublic(BaseModel):
    """Public job state that never exposes account IDs, paths, or service internals."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    source_name: str
    project_id: str
    created_at: int
    updated_at: int
    expires_at: int
    version: int
    graph_sha256: str = ""
    error: str = ""
    result: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_job(cls, job: StudioJob) -> StudioJobPublic:
        return cls(
            id=job.id,
            status=job.status,
            source_name=job.source_name,
            project_id=job.project_id,
            created_at=job.created_at,
            updated_at=job.updated_at,
            expires_at=job.expires_at,
            version=job.version,
            graph_sha256=job.graph_sha256,
            error=(
                "Conversion failed. Please retry or contact support."
                if job.status == "failed" and job.error
                else ""
            ),
            result={
                key: job.result[key]
                for key in (
                    "schema_version",
                    "source_kind",
                    "page_number",
                    "review_required",
                    "release_allowed",
                )
                if key in job.result
            },
        )


class StudioJobPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[StudioJobPublic]
    next_cursor: str = ""


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(
        self,
        source_name: str,
        *,
        project_id: str = "dajoong-project",
        owner_id: str = "",
        organization_id: str = "",
    ) -> StudioJob:
        job_id = uuid.uuid4().hex
        now = int(time.time())
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        job = StudioJob(
            id=job_id,
            source_name=source_name,
            project_id=project_id,
            output_dir=str(job_dir / "output"),
            owner_id=owner_id,
            organization_id=organization_id,
            created_at=now,
            updated_at=now,
        )
        self.save(job)
        return job

    def save(self, job: StudioJob) -> None:
        with self._lock:
            now = int(time.time())
            if not job.created_at:
                job.created_at = now
            job.updated_at = now
            job.version += 1
            target = self.root / job.id / "job.json"
            staging = target.with_suffix(".tmp")
            staging.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
            staging.replace(target)

    def get(self, job_id: str) -> StudioJob:
        if not job_id.isalnum():
            raise KeyError(job_id)
        target = self.root / job_id / "job.json"
        with self._lock:
            if not target.is_file():
                raise KeyError(job_id)
            return StudioJob.model_validate_json(target.read_text(encoding="utf-8"))

    def list_for_identity(
        self,
        *,
        owner_id: str,
        organization_id: str = "",
        scope: str = "personal",
        limit: int = 25,
        cursor: str = "",
    ) -> StudioJobPage:
        if cursor and (not cursor.isdigit() or len(cursor) > 20):
            raise ValueError("invalid cursor")
        jobs: list[StudioJob] = []
        with self._lock:
            for job_file in self.root.glob("*/job.json"):
                try:
                    job = StudioJob.model_validate_json(job_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                permitted = (
                    job.owner_id == owner_id and not job.organization_id
                    if scope == "personal"
                    else bool(organization_id and job.organization_id == organization_id)
                )
                if permitted:
                    jobs.append(job)
        jobs.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        offset = int(cursor) if cursor else 0
        page = jobs[offset : offset + limit]
        next_offset = offset + len(page)
        return StudioJobPage(
            items=[StudioJobPublic.from_job(job) for job in page],
            next_cursor=str(next_offset) if next_offset < len(jobs) else "",
        )

    def write_json(self, job_id: str, name: str, payload: dict[str, Any]) -> Path:
        target = (self.root / job_id / name).resolve()
        if self.root not in target.parents:
            raise ValueError("invalid artifact path")
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_suffix(f"{target.suffix}.tmp")
            staging.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            staging.replace(target)
        return target
