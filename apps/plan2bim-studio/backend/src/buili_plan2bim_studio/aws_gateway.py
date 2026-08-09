from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from buili_plan2bim import BuildingConversionConfig, ConversionConfig

from .corrections import graph_content_hash
from .store import StudioJob, StudioJobPage, StudioJobPublic


class JobVersionConflict(RuntimeError):
    """Another writer advanced the job while this request was in flight."""


class AwsJobGateway:
    """S3, DynamoDB, and SQS adapter for the unchanged Studio job contract."""

    def __init__(self) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - production packaging guard
            raise RuntimeError("Install the backend with the aws extra") from exc
        self.bucket = os.environ["DAJOONG_ARTIFACT_BUCKET"]
        self.table_name = os.environ["DAJOONG_JOB_TABLE"]
        self.owner_index_name = os.environ.get("DAJOONG_OWNER_INDEX_NAME", "owner-id-index")
        self.organization_index_name = os.environ.get(
            "DAJOONG_ORGANIZATION_INDEX_NAME", "organization-id-index"
        )
        self.queue_url = os.environ["DAJOONG_JOB_QUEUE_URL"]
        region = os.environ.get("AWS_REGION", "us-west-2")
        self.s3 = boto3.client("s3", region_name=region)
        self.ddb = boto3.client("dynamodb", region_name=region)
        self.sqs = boto3.client("sqs", region_name=region)
        self.cognito = boto3.client("cognito-idp", region_name=region)
        self.user_pool_id = os.environ.get("DAJOONG_USER_POOL_ID", "")
        retention_days = int(os.environ.get("DAJOONG_ARTIFACT_RETENTION_DAYS", "90"))
        if not 1 <= retention_days <= 3650:
            raise RuntimeError("DAJOONG_ARTIFACT_RETENTION_DAYS must be between 1 and 3650")
        self.retention_seconds = retention_days * 86_400

    @staticmethod
    def _prefix(job_id: str) -> str:
        return f"jobs/{job_id}"

    def _expires_at(self) -> int:
        return int(time.time()) + self.retention_seconds

    @staticmethod
    def _new_job_id(owner_id: str, idempotency_key: str) -> str:
        if owner_id and idempotency_key:
            return uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"https://dajoong.ai/jobs/{owner_id}/{idempotency_key}",
            ).hex
        return uuid.uuid4().hex

    @staticmethod
    def _created_at_job(created_at: int, job_id: str) -> str:
        return f"{created_at:010d}#{job_id}"

    def create_job(
        self,
        source_name: str,
        source: BinaryIO,
        config: ConversionConfig,
        *,
        semantic_model: str = "",
        owner_id: str = "",
        organization_id: str = "",
        idempotency_key: str = "",
    ) -> StudioJob:
        job_id = self._new_job_id(owner_id, idempotency_key)
        if idempotency_key:
            try:
                existing = self.get(job_id)
                if existing.owner_id == owner_id:
                    return existing
            except KeyError:
                pass
        now = int(time.time())
        # A unique immutable key prevents two concurrent retries with the same
        # idempotency key from overwriting the input chosen by the winning job.
        source_key = f"{self._prefix(job_id)}/input/{uuid.uuid4().hex}-{source_name}"
        self.s3.upload_fileobj(
            source,
            self.bucket,
            source_key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        job = StudioJob(
            id=job_id,
            source_name=source_name,
            project_id=config.project_id,
            output_dir=f"s3://{self.bucket}/{self._prefix(job_id)}/output",
            owner_id=owner_id,
            organization_id=organization_id,
            created_at=now,
            updated_at=now,
            expires_at=self._expires_at(),
        )
        try:
            self.save(job)
        except JobVersionConflict:
            return self.get(job_id)
        message = json.dumps(
            {
                "job_id": job_id,
                "source_name": source_name,
                "source_key": source_key,
                "config": config.model_dump(mode="json"),
                "semantic_model": semantic_model,
            }
        )
        parameters = {
            "QueueUrl": self.queue_url,
            "MessageBody": message,
        }
        if self.queue_url.endswith(".fifo"):
            parameters["MessageGroupId"] = "plan2bim"
            parameters["MessageDeduplicationId"] = job_id
        self.sqs.send_message(**parameters)
        return job

    def create_building_job(
        self,
        source_name: str,
        source: BinaryIO,
        config: BuildingConversionConfig,
        *,
        semantic_model: str = "",
        owner_id: str = "",
        organization_id: str = "",
        idempotency_key: str = "",
    ) -> StudioJob:
        job_id = self._new_job_id(owner_id, idempotency_key)
        if idempotency_key:
            try:
                existing = self.get(job_id)
                if existing.owner_id == owner_id:
                    return existing
            except KeyError:
                pass
        now = int(time.time())
        source_key = f"{self._prefix(job_id)}/input/{uuid.uuid4().hex}-{source_name}"
        self.s3.upload_fileobj(
            source,
            self.bucket,
            source_key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        job = StudioJob(
            id=job_id,
            source_name=source_name,
            project_id=config.project_id,
            output_dir=f"s3://{self.bucket}/{self._prefix(job_id)}/output",
            owner_id=owner_id,
            organization_id=organization_id,
            created_at=now,
            updated_at=now,
            expires_at=self._expires_at(),
        )
        try:
            self.save(job)
        except JobVersionConflict:
            return self.get(job_id)
        message = json.dumps(
            {
                "kind": "building",
                "job_id": job_id,
                "source_name": source_name,
                "source_key": source_key,
                "config": config.model_dump(mode="json"),
                "semantic_model": semantic_model,
            }
        )
        parameters = {"QueueUrl": self.queue_url, "MessageBody": message}
        if self.queue_url.endswith(".fifo"):
            parameters["MessageGroupId"] = "plan2bim"
            parameters["MessageDeduplicationId"] = job_id
        self.sqs.send_message(**parameters)
        return job

    def save(self, job: StudioJob) -> None:
        now = int(time.time())
        if not job.created_at:
            job.created_at = now
        job.updated_at = now
        next_version = job.version + 1
        item = {
            "job_id": {"S": job.id},
            "status": {"S": job.status},
            "source_name": {"S": job.source_name},
            "project_id": {"S": job.project_id},
            "output_dir": {"S": job.output_dir},
            "created_at": {"N": str(job.created_at)},
            "created_at_job": {"S": self._created_at_job(job.created_at, job.id)},
            "updated_at": {"N": str(job.updated_at)},
            "expires_at": {"N": str(job.expires_at or self._expires_at())},
            "version": {"N": str(next_version)},
            "active_revision": {"S": job.active_revision},
            "graph_sha256": {"S": job.graph_sha256},
            "error": {"S": job.error},
            "result_json": {"S": json.dumps(job.result, separators=(",", ":"))},
        }
        if job.owner_id:
            item["owner_id"] = {"S": job.owner_id}
        if job.organization_id:
            item["organization_id"] = {"S": job.organization_id}
        try:
            if job.version == 0:
                self.ddb.put_item(
                    TableName=self.table_name,
                    Item=item,
                    ConditionExpression="attribute_not_exists(job_id)",
                )
            else:
                self.ddb.put_item(
                    TableName=self.table_name,
                    Item=item,
                    ConditionExpression="#version = :expected_version",
                    ExpressionAttributeNames={"#version": "version"},
                    ExpressionAttributeValues={
                        ":expected_version": {"N": str(job.version)},
                    },
                )
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code == "ConditionalCheckFailedException":
                raise JobVersionConflict(job.id) from exc
            raise
        job.version = next_version

    def create_imported_graph(
        self,
        source_name: str,
        graph: dict[str, Any],
        *,
        project_id: str = "dajoong-project",
        owner_id: str = "",
        organization_id: str = "",
    ) -> StudioJob:
        job_id = uuid.uuid4().hex
        now = int(time.time())
        job = StudioJob(
            id=job_id,
            status="review_required",
            source_name=source_name,
            project_id=project_id,
            output_dir=f"s3://{self.bucket}/{self._prefix(job_id)}/output",
            owner_id=owner_id,
            organization_id=organization_id,
            created_at=now,
            updated_at=now,
            expires_at=self._expires_at(),
            result={
                "plan_graph_path": f"s3://{self.bucket}/{self._prefix(job_id)}/output/03-plan-graph.json"
            },
            graph_sha256="",
        )
        self.save(job)
        self.write_json(job, "graph", graph)
        job.graph_sha256 = graph_content_hash(graph)
        self.save(job)
        return job

    def get(self, job_id: str) -> StudioJob:
        if not job_id.isalnum():
            raise KeyError(job_id)
        response = self.ddb.get_item(
            TableName=self.table_name,
            Key={"job_id": {"S": job_id}},
            ConsistentRead=True,
        )
        item = response.get("Item")
        if not item:
            raise KeyError(job_id)
        job = StudioJob(
            id=item["job_id"]["S"],
            status=item["status"]["S"],
            source_name=item["source_name"]["S"],
            project_id=item.get("project_id", {}).get("S", "dajoong-project"),
            output_dir=item["output_dir"]["S"],
            owner_id=item.get("owner_id", {}).get("S", ""),
            organization_id=item.get("organization_id", {}).get("S", ""),
            created_at=int(item.get("created_at", {}).get("N", "0")),
            updated_at=int(item.get("updated_at", {}).get("N", "0")),
            expires_at=int(item.get("expires_at", {}).get("N", "0")),
            version=int(item.get("version", {}).get("N", "0")),
            active_revision=item.get("active_revision", {}).get("S", ""),
            graph_sha256=item.get("graph_sha256", {}).get("S", ""),
            error=item.get("error", {}).get("S", ""),
            result=json.loads(item.get("result_json", {}).get("S", "{}")),
        )
        if job.expires_at and job.expires_at <= int(time.time()):
            raise KeyError(job_id)
        return job

    @staticmethod
    def _encode_cursor(last_key: dict[str, Any] | None) -> str:
        if not last_key:
            return ""
        compact = {
            key: value.get("S", "")
            for key, value in last_key.items()
            if key in {"job_id", "created_at_job"}
        }
        raw = json.dumps(compact, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str, partition_name: str, partition_value: str) -> dict[str, Any]:
        if not cursor:
            return {}
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            job_id = str(payload["job_id"])
            created_at_job = str(payload["created_at_job"])
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            binascii.Error,
        ) as exc:
            raise ValueError("invalid cursor") from exc
        if not job_id.isalnum() or len(created_at_job) > 80 or not created_at_job.endswith(job_id):
            raise ValueError("invalid cursor")
        return {
            "job_id": {"S": job_id},
            partition_name: {"S": partition_value},
            "created_at_job": {"S": created_at_job},
        }

    @staticmethod
    def _job_from_projected_item(item: dict[str, Any]) -> StudioJob:
        return StudioJob(
            id=item["job_id"]["S"],
            status=item["status"]["S"],
            source_name=item["source_name"]["S"],
            project_id=item.get("project_id", {}).get("S", "dajoong-project"),
            output_dir="",
            owner_id=item.get("owner_id", {}).get("S", ""),
            organization_id=item.get("organization_id", {}).get("S", ""),
            created_at=int(item.get("created_at", {}).get("N", "0")),
            updated_at=int(item.get("updated_at", {}).get("N", "0")),
            expires_at=int(item.get("expires_at", {}).get("N", "0")),
            version=int(item.get("version", {}).get("N", "0")),
            error=item.get("error", {}).get("S", ""),
        )

    def list_for_identity(
        self,
        *,
        owner_id: str,
        organization_id: str = "",
        scope: str = "personal",
        limit: int = 25,
        cursor: str = "",
    ) -> StudioJobPage:
        if scope == "personal":
            partition_name = "owner_id"
            partition_value = owner_id
            index_name = self.owner_index_name
        elif scope == "organization" and organization_id:
            partition_name = "organization_id"
            partition_value = organization_id
            index_name = self.organization_index_name
        else:
            raise ValueError("invalid identity scope")
        if not partition_value:
            return StudioJobPage(items=[])
        values = {
            ":partition": {"S": partition_value},
            ":now": {"N": str(int(time.time()))},
        }
        parameters: dict[str, Any] = {
            "TableName": self.table_name,
            "IndexName": index_name,
            "KeyConditionExpression": f"{partition_name} = :partition",
            "FilterExpression": "expires_at > :now",
            "ExpressionAttributeValues": values,
            "ProjectionExpression": (
                "job_id, owner_id, organization_id, source_name, project_id, #status, "
                "created_at, updated_at, expires_at, #version, #error"
            ),
            "ExpressionAttributeNames": {
                "#status": "status",
                "#version": "version",
                "#error": "error",
            },
            "ScanIndexForward": False,
            "Limit": limit,
        }
        start_key = self._decode_cursor(cursor, partition_name, partition_value)
        if start_key:
            parameters["ExclusiveStartKey"] = start_key
        page = self.ddb.query(**parameters)
        jobs = [self._job_from_projected_item(item) for item in page.get("Items", [])]
        return StudioJobPage(
            items=[StudioJobPublic.from_job(job) for job in jobs],
            next_cursor=self._encode_cursor(page.get("LastEvaluatedKey")),
        )

    def artifact_key(
        self,
        job: StudioJob,
        artifact_name: str,
        *,
        level_id: str = "",
    ) -> str:
        is_building = str(job.result.get("schema_version", "")).startswith(
            "dajoong.building-conversion"
        )
        output_names = {
            "graph": "03-plan-graph.json",
            "glb": "04-model.glb",
            "ifc": "04-model.ifc",
            "manifest": "conversion-manifest.json",
            "overlay": "00-semantic-overlay.png",
            "corrected-graph": "corrected-plan-graph.json",
            "corrections": "corrections.json",
        }
        if artifact_name == "source":
            return f"{self._prefix(job.id)}/input/{job.source_name}"
        if artifact_name in {"corrected-graph", "corrections"} and job.active_revision:
            filename = output_names[artifact_name]
            return f"{self._prefix(job.id)}/revisions/{job.active_revision}/{filename}"
        if artifact_name == "render":
            if is_building:
                level_results = job.result.get("level_results") or {}
                if level_id and level_id not in level_results:
                    raise KeyError(level_id)
                selected_level_id, level_result = (
                    (level_id, level_results[level_id])
                    if level_id
                    else next(iter(level_results.items()), ("", {}))
                )
                if level_result.get("source_kind") == "raster_image":
                    return f"{self._prefix(job.id)}/input/{job.source_name}"
                page = int(level_result.get("page_number", 1))
                return (
                    f"{self._prefix(job.id)}/output/levels/{selected_level_id}/"
                    f"00-source-page-{page}.png"
                )
            if job.result.get("source_kind") in {"raster_pdf", "vector_pdf"}:
                page = int(job.result.get("page_number", 1))
                return f"{self._prefix(job.id)}/output/00-source-page-{page}.png"
            return f"{self._prefix(job.id)}/input/{job.source_name}"
        if is_building:
            output_names.update(
                {
                    "graph": "05-building-plan-graph.json",
                    "glb": "05-building.glb",
                    "ifc": "05-building.ifc",
                    "consistency": "05-building-consistency.json",
                    "manifest": "building-conversion-manifest.json",
                }
            )
        filename = output_names.get(artifact_name)
        if not filename:
            raise KeyError(artifact_name)
        return f"{self._prefix(job.id)}/output/{filename}"

    def open_artifact(
        self,
        job: StudioJob,
        artifact_name: str,
        *,
        level_id: str = "",
    ) -> tuple[Any, str, int, str]:
        key = self.artifact_key(job, artifact_name, level_id=level_id)
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=key,
        )
        return (
            response["Body"],
            str(response.get("ContentType") or "application/octet-stream"),
            int(response.get("ContentLength") or 0),
            key.rsplit("/", 1)[-1],
        )

    def read_json(self, job: StudioJob, artifact_name: str) -> dict[str, Any]:
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.artifact_key(job, artifact_name),
        )
        return json.loads(response["Body"].read())

    def read_editable_graph(self, job: StudioJob) -> dict[str, Any]:
        return self.read_json(job, "corrected-graph" if job.active_revision else "graph")

    def write_json(self, job: StudioJob, artifact_name: str, payload: dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.artifact_key(job, artifact_name),
            Body=(json.dumps(payload, indent=2) + "\n").encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    def commit_correction_revision(
        self,
        job: StudioJob,
        *,
        corrected_graph: dict[str, Any],
        corrections: dict[str, Any],
        graph_sha256: str,
        release_allowed: bool,
    ) -> None:
        """Write an immutable revision, then atomically advance its DynamoDB pointer."""
        revision = graph_sha256
        prefix = f"{self._prefix(job.id)}/revisions/{revision}"
        payloads = {
            f"{prefix}/corrected-plan-graph.json": corrected_graph,
            f"{prefix}/corrections.json": corrections,
        }
        for key, payload in payloads.items():
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=(json.dumps(payload, separators=(",", ":")) + "\n").encode(),
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
        job.active_revision = revision
        job.graph_sha256 = graph_sha256
        job.status = "complete" if release_allowed else "review_required"
        self.save(job)

    def _delete_prefix_versions(self, prefix: str) -> None:
        key_marker: str | None = None
        version_marker: str | None = None
        while True:
            parameters: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if key_marker:
                parameters["KeyMarker"] = key_marker
            if version_marker:
                parameters["VersionIdMarker"] = version_marker
            listing = self.s3.list_object_versions(**parameters)
            objects = [
                {"Key": entry["Key"], "VersionId": entry["VersionId"]}
                for entry in [
                    *listing.get("Versions", []),
                    *listing.get("DeleteMarkers", []),
                ]
            ]
            if objects:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects, "Quiet": True},
                )
            if not listing.get("IsTruncated"):
                return
            key_marker = listing.get("NextKeyMarker")
            version_marker = listing.get("NextVersionIdMarker")

    def process_message(self, message: dict[str, Any]) -> StudioJob:
        from buili_plan2bim import BuildingPlan2BimConverter, Plan2BimConverter

        job = self.get(str(message["job_id"]))
        job.status = "running"
        job.error = ""
        self.save(job)
        with tempfile.TemporaryDirectory(prefix=f"dajoong-{job.id[:8]}-") as temporary:
            root = Path(temporary)
            source = root / job.source_name
            output = root / "output"
            self.s3.download_file(self.bucket, str(message["source_key"]), str(source))
            if message.get("kind") == "building":
                building_config = BuildingConversionConfig.model_validate(message["config"])
                building_config = building_config.model_copy(
                    update={
                        "levels": [
                            level.model_copy(update={"source_path": str(source)})
                            for level in building_config.levels
                        ]
                    }
                )
                converter = BuildingPlan2BimConverter(
                    threads=building_config.threads,
                    batch_size=building_config.batch_size,
                    semantic_model_path=str(message.get("semantic_model") or "") or None,
                )
                result = converter.convert(output, building_config)
            else:
                config = ConversionConfig.model_validate(message["config"])
                converter = Plan2BimConverter(
                    threads=config.threads,
                    batch_size=config.batch_size,
                    semantic_model_path=str(message.get("semantic_model") or "") or None,
                )
                result = converter.convert(source, output, config)
            for artifact in output.rglob("*"):
                if artifact.is_file():
                    self.s3.upload_file(
                        str(artifact),
                        self.bucket,
                        f"{self._prefix(job.id)}/output/{artifact.relative_to(output).as_posix()}",
                        ExtraArgs={"ServerSideEncryption": "AES256"},
                    )
            job.result = result.model_dump(mode="json")
            job.graph_sha256 = graph_content_hash(
                json.loads(Path(result.plan_graph_path).read_text(encoding="utf-8"))
            )
            job.status = "review_required" if result.review_required else "complete"
            try:
                self.save(job)
            except JobVersionConflict:
                try:
                    current = self.get(job.id)
                except KeyError:
                    # Account deletion won the race after conversion started.
                    self._delete_prefix_versions(f"{self._prefix(job.id)}/")
                    raise
                if current.organization_id and not current.owner_id:
                    # Preserve organization-owned output without restoring deleted identity data.
                    current.result = job.result
                    current.status = job.status
                    current.error = job.error
                    self.save(current)
                    return current
                raise
            return job

    def delete_account(self, *, owner_id: str, username: str) -> dict[str, int]:
        """Delete personal jobs before deleting the authenticated Cognito identity."""
        if not owner_id or not username or not self.user_pool_id:
            raise RuntimeError("account deletion is not configured")
        deleted_jobs = 0
        detached_organization_jobs = 0
        start_key: dict[str, Any] | None = None
        while True:
            parameters: dict[str, Any] = {
                "TableName": self.table_name,
                "IndexName": self.owner_index_name,
                "KeyConditionExpression": "owner_id = :owner_id",
                "ExpressionAttributeValues": {":owner_id": {"S": owner_id}},
                "ProjectionExpression": "job_id, organization_id",
            }
            if start_key:
                parameters["ExclusiveStartKey"] = start_key
            page = self.ddb.query(**parameters)
            for item in page.get("Items", []):
                # Organization-owned construction records follow the customer retention policy.
                if item.get("organization_id", {}).get("S", ""):
                    self.ddb.update_item(
                        TableName=self.table_name,
                        Key={"job_id": item["job_id"]},
                        UpdateExpression=(
                            "SET updated_at = :updated_at, "
                            "#version = if_not_exists(#version, :zero) + :one REMOVE owner_id"
                        ),
                        ConditionExpression="owner_id = :owner_id",
                        ExpressionAttributeNames={"#version": "version"},
                        ExpressionAttributeValues={
                            ":updated_at": {"N": str(int(time.time()))},
                            ":zero": {"N": "0"},
                            ":one": {"N": "1"},
                            ":owner_id": {"S": owner_id},
                        },
                    )
                    detached_organization_jobs += 1
                    continue
                job_id = item["job_id"]["S"]
                prefix = f"{self._prefix(job_id)}/"
                self._delete_prefix_versions(prefix)
                self.ddb.delete_item(TableName=self.table_name, Key={"job_id": {"S": job_id}})
                deleted_jobs += 1
            start_key = page.get("LastEvaluatedKey")
            if not start_key:
                break
        self.cognito.admin_delete_user(UserPoolId=self.user_pool_id, Username=username)
        return {
            "deleted_personal_jobs": deleted_jobs,
            "detached_organization_jobs": detached_organization_jobs,
        }
