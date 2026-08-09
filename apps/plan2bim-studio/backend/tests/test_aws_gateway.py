from __future__ import annotations

from typing import Any

from buili_plan2bim_studio.aws_gateway import AwsJobGateway
from buili_plan2bim_studio.store import StudioJob


class FakeDynamoDb:
    def __init__(self) -> None:
        self.saved: dict[str, Any] | None = None
        self.deleted: list[str] = []

    def put_item(self, **parameters: Any) -> None:
        self.saved = parameters

    def query(self, **parameters: Any) -> dict[str, Any]:
        assert parameters["IndexName"] == "owner-id-index"
        return {
            "Items": [
                {"job_id": {"S": "personal"}, "organization_id": {"S": ""}},
                {"job_id": {"S": "retained"}, "organization_id": {"S": "org-1"}},
            ]
        }

    def delete_item(self, **parameters: Any) -> None:
        self.deleted.append(parameters["Key"]["job_id"]["S"])


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
    value.retention_seconds = 90 * 86_400
    value.ddb = FakeDynamoDb()
    value.s3 = FakeS3()
    value.cognito = FakeCognito()
    return value


def test_save_persists_dynamodb_expiry() -> None:
    value = gateway()
    value.save(
        StudioJob(
            id="job1",
            source_name="plan.pdf",
            output_dir="s3://private-artifacts/jobs/job1/output",
            expires_at=2_000_000_000,
        )
    )

    assert value.ddb.saved is not None
    assert value.ddb.saved["Item"]["expires_at"] == {"N": "2000000000"}


def test_account_deletion_removes_all_object_versions_and_personal_job() -> None:
    value = gateway()

    result = value.delete_account(owner_id="user-1", username="username-1")

    assert result == {"deleted_personal_jobs": 1}
    assert value.s3.listed == ["jobs/personal/"]
    assert value.s3.deleted == [[
        {"Key": "jobs/personal/input/plan.pdf", "VersionId": "v1"},
        {"Key": "jobs/personal/output/model.glb", "VersionId": "d1"},
    ]]
    assert value.ddb.deleted == ["personal"]
    assert value.cognito.deleted == [("pool-1", "username-1")]
