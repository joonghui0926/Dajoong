from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from buili_plan2bim import BuildingConversionConfig, ConversionConfig

from .store import StudioJob


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

    def create_job(
        self,
        source_name: str,
        source: BinaryIO,
        config: ConversionConfig,
        *,
        semantic_model: str = "",
        owner_id: str = "",
        organization_id: str = "",
    ) -> StudioJob:
        job_id = uuid.uuid4().hex
        source_key = f"{self._prefix(job_id)}/input/{source_name}"
        self.s3.upload_fileobj(source, self.bucket, source_key)
        job = StudioJob(
            id=job_id,
            source_name=source_name,
            output_dir=f"s3://{self.bucket}/{self._prefix(job_id)}/output",
            owner_id=owner_id,
            organization_id=organization_id,
            expires_at=self._expires_at(),
        )
        self.save(job)
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
    ) -> StudioJob:
        job_id = uuid.uuid4().hex
        source_key = f"{self._prefix(job_id)}/input/{source_name}"
        self.s3.upload_fileobj(source, self.bucket, source_key)
        job = StudioJob(
            id=job_id,
            source_name=source_name,
            output_dir=f"s3://{self.bucket}/{self._prefix(job_id)}/output",
            owner_id=owner_id,
            organization_id=organization_id,
            expires_at=self._expires_at(),
        )
        self.save(job)
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
        item = {
            "job_id": {"S": job.id},
            "status": {"S": job.status},
            "source_name": {"S": job.source_name},
            "output_dir": {"S": job.output_dir},
            "organization_id": {"S": job.organization_id},
            "expires_at": {"N": str(job.expires_at or self._expires_at())},
            "error": {"S": job.error},
            "result_json": {"S": json.dumps(job.result, separators=(",", ":"))},
        }
        if job.owner_id:
            item["owner_id"] = {"S": job.owner_id}
        self.ddb.put_item(
            TableName=self.table_name,
            Item=item,
        )

    def create_imported_graph(
        self,
        source_name: str,
        graph: dict[str, Any],
        *,
        owner_id: str = "",
        organization_id: str = "",
    ) -> StudioJob:
        job_id = uuid.uuid4().hex
        job = StudioJob(
            id=job_id,
            status="review_required",
            source_name=source_name,
            output_dir=f"s3://{self.bucket}/{self._prefix(job_id)}/output",
            owner_id=owner_id,
            organization_id=organization_id,
            expires_at=self._expires_at(),
            result={
                "plan_graph_path": f"s3://{self.bucket}/{self._prefix(job_id)}/output/03-plan-graph.json"
            },
        )
        self.save(job)
        self.write_json(job, "graph", graph)
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
        return StudioJob(
            id=item["job_id"]["S"],
            status=item["status"]["S"],
            source_name=item["source_name"]["S"],
            output_dir=item["output_dir"]["S"],
            owner_id=item.get("owner_id", {}).get("S", ""),
            organization_id=item.get("organization_id", {}).get("S", ""),
            expires_at=int(item.get("expires_at", {}).get("N", "0")),
            error=item.get("error", {}).get("S", ""),
            result=json.loads(item.get("result_json", {}).get("S", "{}")),
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

    def write_json(self, job: StudioJob, artifact_name: str, payload: dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.artifact_key(job, artifact_name),
            Body=(json.dumps(payload, indent=2) + "\n").encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

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
            job.status = "review_required" if result.review_required else "complete"
            self.save(job)
            return job

    def delete_account(self, *, owner_id: str, username: str) -> dict[str, int]:
        """Delete personal jobs before deleting the authenticated Cognito identity."""
        if not owner_id or not username or not self.user_pool_id:
            raise RuntimeError("account deletion is not configured")
        deleted_jobs = 0
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
                    continue
                job_id = item["job_id"]["S"]
                prefix = f"{self._prefix(job_id)}/"
                key_marker: str | None = None
                version_marker: str | None = None
                while True:
                    listing_parameters: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
                    if key_marker:
                        listing_parameters["KeyMarker"] = key_marker
                    if version_marker:
                        listing_parameters["VersionIdMarker"] = version_marker
                    listing = self.s3.list_object_versions(**listing_parameters)
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
                        break
                    key_marker = listing.get("NextKeyMarker")
                    version_marker = listing.get("NextVersionIdMarker")
                self.ddb.delete_item(TableName=self.table_name, Key={"job_id": {"S": job_id}})
                deleted_jobs += 1
            start_key = page.get("LastEvaluatedKey")
            if not start_key:
                break
        self.cognito.admin_delete_user(UserPoolId=self.user_pool_id, Username=username)
        return {"deleted_personal_jobs": deleted_jobs}
