from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StudioJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str = "queued"
    source_name: str
    output_dir: str
    owner_id: str = ""
    organization_id: str = ""
    expires_at: int = 0
    error: str = ""
    result: dict[str, Any] = Field(default_factory=dict)


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(
        self,
        source_name: str,
        *,
        owner_id: str = "",
        organization_id: str = "",
    ) -> StudioJob:
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        job = StudioJob(
            id=job_id,
            source_name=source_name,
            output_dir=str(job_dir / "output"),
            owner_id=owner_id,
            organization_id=organization_id,
        )
        self.save(job)
        return job

    def save(self, job: StudioJob) -> None:
        with self._lock:
            target = self.root / job.id / "job.json"
            staging = target.with_suffix(".tmp")
            staging.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
            staging.replace(target)

    def get(self, job_id: str) -> StudioJob:
        if not job_id.isalnum():
            raise KeyError(job_id)
        target = self.root / job_id / "job.json"
        if not target.is_file():
            raise KeyError(job_id)
        return StudioJob.model_validate_json(target.read_text(encoding="utf-8"))

    def write_json(self, job_id: str, name: str, payload: dict[str, Any]) -> Path:
        target = (self.root / job_id / name).resolve()
        if self.root not in target.parents:
            raise ValueError("invalid artifact path")
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target
