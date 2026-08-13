from __future__ import annotations

from typing import Any

from buili_plan2bim_studio.aws_gateway import AwsJobGateway
from buili_plan2bim_studio.store import StudioJob


class FakeDynamoDb:
    def __init__(self) -> None:
        self.saved: dict[str, Any] | None = None
        self.deleted: list[str] = []
        self.updated: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []

    def put_item(self, **parameters: Any) -> None:
        self.saved = parameters

    def query(self, **parameters: Any) -> dict[str, Any]:
        self.queries.append(parameters)
        assert parameters["IndexName"] == "owner-id-index"
        if "ScanIndexForward" in parameters:
            return {
                "Items": [
                    {
                        "job_id": {"S": "recent1"},
                        "owner_id": {"S": "user-1"},
                        "status": {"S": "complete"},
                        "source_name": {"S": "A1.1.pdf"},
                        "project_id": {"S": "Tower A"},
                        "created_at": {"N": "2000000000"},
                        "updated_at": {"N": "2000000010"},
                        "expires_at": {"N": "2100000000"},
                        "version": {"N": "3"},
                    }
                ],
                "LastEvaluatedKey": {
                    "job_id": {"S": "recent1"},
                    "owner_id": {"S": "user-1"},
                    "created_at_job": {"S": "2000000000#recent1"},
                },
            }
        return {
            "Items": [
                {"job_id": {"S": "personal"}, "organization_id": {"S": ""}},
                {"job_id": {"S": "retained"}, "organization_id": {"S": "org-1"}},
            ]
        }

    def delete_item(self, **parameters: Any) -> None:
        self.deleted.append(parameters["Key"]["job_id"]["S"])

    def update_item(self, **parameters: Any) -> None:
        self.updated.append(parameters)


class FakeS3:
    def __init__(self) -> None:
        self.listed: list[str] = []
        self.deleted: list[list[dict[str, str]]] = []

    def list_object_versions(self, **parameters: Any) -> dict[str, Any]:
        self.listed.append(parameters["Prefix"])
        return {
            "Versions": [{"Key": "jobs/personal/input/plan.pdf", "VersionId": "v1"}],
            "DeleteMarkers": [{"Key": "jobs/personal/output/model.glb", "VersionId": "d1"}],
            "IsTruncated": False,
        }

    def delete_objects(self, **parameters: Any) -> None:
        self.deleted.append(parameters["Delete"]["Objects"])


class FakeSqs:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send_message(self, **parameters: Any) -> None:
        self.sent.append(parameters)


class FakeCognito:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def admin_delete_user(self, **parameters: Any) -> None:
        self.deleted.append((parameters["UserPoolId"], parameters["Username"]))


def gateway() -> AwsJobGateway:
    value = AwsJobGateway.__new__(AwsJobGateway)
    value.bucket = "private-artifacts"
    value.table_name = "jobs"
    value.user_pool_id = "pool-1"
    value.owner_index_name = "owner-id-index"
    value.organization_index_name = "organization-id-index"
    value.retention_seconds = 90 * 86_400
    value.queue_url = "https://sqs.us-west-2.amazonaws.com/123/jobs.fifo"
    value.ddb = FakeDynamoDb()
    value.s3 = FakeS3()
    value.sqs = FakeSqs()
    value.cognito = FakeCognito()
    return value


def test_save_persists_dynamodb_expiry() -> None:
    value = gateway()
    value.save(
        StudioJob(
            id="job1",
            source_name="plan.pdf",
            source_key="jobs/job1/input/immutable-plan.pdf",
            output_dir="s3://private-artifacts/jobs/job1/output",
            expires_at=2_000_000_000,
            lease_until=1_900_000_000,
            submission={"job_id": "job1"},
        )
    )

    assert value.ddb.saved is not None
    assert value.ddb.saved["Item"]["expires_at"] == {"N": "2000000000"}
    assert value.ddb.saved["Item"]["source_key"] == {
        "S": "jobs/job1/input/immutable-plan.pdf"
    }
    assert value.ddb.saved["Item"]["lease_until"] == {"N": "1900000000"}


def test_account_deletion_removes_all_object_versions_and_personal_job() -> None:
    value = gateway()

    result = value.delete_account(owner_id="user-1", username="username-1")

    assert result == {"deleted_personal_jobs": 1, "detached_organization_jobs": 1}
    assert value.s3.listed == ["jobs/personal/"]
    assert value.s3.deleted == [[
        {"Key": "jobs/personal/input/plan.pdf", "VersionId": "v1"},
        {"Key": "jobs/personal/output/model.glb", "VersionId": "d1"},
    ]]
    assert value.ddb.deleted == ["personal"]
    assert value.ddb.updated[0]["Key"] == {"job_id": {"S": "retained"}}
    assert "REMOVE owner_id" in value.ddb.updated[0]["UpdateExpression"]
    assert value.cognito.deleted == [("pool-1", "username-1")]


def test_recent_jobs_use_account_partition_and_opaque_cursor() -> None:
    value = gateway()

    page = value.list_for_identity(owner_id="user-1", limit=1)

    assert [item.project_id for item in page.items] == ["Tower A"]
    assert page.items[0].version == 3
    assert page.next_cursor
    query = value.ddb.queries[0]
    assert query["KeyConditionExpression"] == "owner_id = :partition"
    assert "attribute_not_exists(organization_id)" in query["FilterExpression"]
    assert query["ScanIndexForward"] is False
    decoded = value._decode_cursor(page.next_cursor, "owner_id", "user-1")
    assert decoded["owner_id"] == {"S": "user-1"}
    assert decoded["job_id"] == {"S": "recent1"}


def test_idempotency_key_is_stable_per_account() -> None:
    first = AwsJobGateway._new_job_id("user-1", "request-12345678")
    assert first == AwsJobGateway._new_job_id("user-1", "request-12345678")
    assert first != AwsJobGateway._new_job_id("user-2", "request-12345678")
    assert len(first) == 32


def test_fifo_jobs_use_independent_message_groups() -> None:
    value = gateway()
    job = StudioJob(
        id="job1",
        source_name="plan.pdf",
        output_dir="s3://private-artifacts/jobs/job1/output",
        submission={"job_id": "job1", "source_key": "jobs/job1/input/plan.pdf"},
    )

    value._enqueue(job)

    assert value.sqs.sent[0]["MessageGroupId"] == "job1"
    assert value.sqs.sent[0]["MessageDeduplicationId"] == "job1"


def test_source_artifact_uses_immutable_uploaded_key() -> None:
    value = gateway()
    job = StudioJob(
        id="job1",
        source_name="plan.pdf",
        source_key="jobs/job1/input/nonce-plan.pdf",
        output_dir="s3://private-artifacts/jobs/job1/output",
    )

    assert value.artifact_key(job, "source") == "jobs/job1/input/nonce-plan.pdf"


def test_duplicate_worker_delivery_does_not_reprocess_active_lease() -> None:
    value = gateway()
    active = StudioJob(
        id="job1",
        status="running",
        source_name="plan.pdf",
        source_key="jobs/job1/input/plan.pdf",
        output_dir="s3://private-artifacts/jobs/job1/output",
        lease_until=4_102_444_800,
    )
    value.get = lambda _job_id: active  # type: ignore[method-assign]

    assert value.process_message({"job_id": "job1"}) is active
