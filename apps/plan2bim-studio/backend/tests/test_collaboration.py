from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from buili_plan2bim_studio import main
from buili_plan2bim_studio.collaboration import (
    AwsCollaborationStore,
    CollaborationService,
    Organization,
    WorkspaceAccessError,
    WorkspaceConflict,
    WorkspaceMember,
)
from buili_plan2bim_studio.corrections import graph_content_hash
from buili_plan2bim_studio.store import JobStore


def _graph() -> dict[str, object]:
    return {
        "schema_version": "buili.plan-graph.v2",
        "project_id": "workspace-test",
        "sheet_id": "A1.1",
        "scale": {"px_per_meter": 100.0, "source": "test", "confidence": 1.0},
        "levels": [
            {
                "id": "L1",
                "name": "Level 1",
                "elevation_m": 0.0,
                "nominal_height_m": 3.0,
                "confidence": 1.0,
                "uncertainty": 0.0,
                "source_ref_ids": [],
                "model_version": "test",
                "review_state": "accepted",
            }
        ],
        "walls": [],
        "rooms": [],
        "openings": [],
        "fixtures": [],
        "routes": [],
        "vertical_connections": [],
        "sources": [],
        "unsupported_features": [],
        "extraction": {"method": "test"},
        "provenance": {"source_hash": "a" * 64, "source_revision_state": "test"},
        "confidence": {"review_required": False},
        "warnings": [],
        "pipeline": {"content_sha256": "old"},
    }


def test_invitation_domain_roles_and_idempotent_comments(tmp_path) -> None:
    service = CollaborationService(tmp_path)
    organization = service.create_organization(
        user_id="owner",
        email="owner@example.com",
        display_name="Owner",
        name="Dajoong Builders",
        approved_domains=["EXAMPLE.COM"],
        domain_join_enabled=True,
    )
    assert service.require_member(organization.id, "owner").role == "owner"

    receipt = service.create_invitation(
        organization_id=organization.id,
        inviter_id="owner",
        email="editor@example.com",
        role="editor",
    )
    with pytest.raises(WorkspaceConflict):
        service.create_invitation(
            organization_id=organization.id,
            inviter_id="owner",
            email="EDITOR@example.com",
            role="viewer",
        )
    persisted = json.loads((tmp_path / "collaboration.json").read_text(encoding="utf-8"))
    assert receipt.token not in json.dumps(persisted)
    with pytest.raises(WorkspaceAccessError):
        service.accept_invitation(
            token=receipt.token,
            user_id="intruder",
            email="other@example.com",
            display_name="Other",
        )

    editor = service.accept_invitation(
        token=receipt.token,
        user_id="editor",
        email="editor@example.com",
        display_name="Editor",
    )
    assert editor.role == "editor"
    with pytest.raises(WorkspaceConflict):
        service.accept_invitation(
            token=receipt.token,
            user_id="editor",
            email="editor@example.com",
            display_name="Editor",
        )
    revoked_receipt = service.create_invitation(
        organization_id=organization.id,
        inviter_id="owner",
        email="viewer@example.com",
        role="viewer",
    )
    revoked = service.revoke_invitation(
        organization_id=organization.id,
        actor_id="owner",
        invitation_id=revoked_receipt.invitation.id,
    )
    assert revoked.status == "revoked"
    with pytest.raises(WorkspaceConflict):
        service.accept_invitation(
            token=revoked_receipt.token,
            user_id="viewer",
            email="viewer@example.com",
            display_name="Viewer",
        )

    domain_member = service.join_by_domain(
        user_id="domain-user",
        email="domain-user@example.com",
        display_name="Domain user",
    )
    assert domain_member.organization_id == organization.id
    assert domain_member.role == "editor"

    first = service.add_comment(
        organization_id=organization.id,
        job_id="job-1",
        actor_id="editor",
        actor_name="Editor",
        body="Please verify this wall.",
        request_id="same-request",
        entity_ref="walls:L1:wall:1",
        assigned_to="domain-user",
    )
    duplicate = service.add_comment(
        organization_id=organization.id,
        job_id="job-1",
        actor_id="editor",
        actor_name="Editor",
        body="This retry must not create another comment.",
        request_id="same-request",
    )
    assert duplicate.id == first.id
    assert duplicate.body == first.body
    assert len(service.store.list_comments("job-1", 100)) == 1

    with pytest.raises(WorkspaceConflict):
        service.remove_member(
            organization_id=organization.id,
            actor_id="owner",
            user_id="owner",
        )
    service.change_role(
        organization_id=organization.id,
        actor_id="owner",
        user_id="editor",
        role="admin",
    )
    previous_owner, next_owner = service.transfer_ownership(
        organization_id=organization.id,
        actor_id="owner",
        user_id="editor",
    )
    assert previous_owner.role == "admin"
    assert next_owner.role == "owner"
    service.remove_member(
        organization_id=organization.id,
        actor_id="editor",
        user_id="owner",
    )
    with pytest.raises(WorkspaceAccessError):
        service.require_member(organization.id, "owner")


def test_company_job_access_is_revoked_with_membership(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(main, "store", JobStore(tmp_path / "jobs"))
    monkeypatch.setattr(main, "_collaboration_service_instance", None)
    monkeypatch.setattr(main, "_collaboration_service_root", None)
    monkeypatch.setenv("DAJOONG_APP_URL", "https://app.example.com/studio")

    identities = {
        "owner": ("owner", "owner@example.com", "Owner"),
        "editor": ("editor", "editor@example.com", "Editor"),
    }

    def actor(request):
        return identities[request.headers.get("X-Test-User", "owner")]

    monkeypatch.setattr(main, "_actor", actor)
    with TestClient(main.app) as client:
        owner_headers = {"X-Test-User": "owner"}
        created = client.post(
            "/api/workspace/organizations",
            headers=owner_headers,
            json={
                "name": "Dajoong Builders",
                "approved_domains": [],
                "domain_join_enabled": False,
            },
        )
        assert created.status_code == 200
        organization_id = created.json()["organization"]["id"]
        company_owner_headers = {
            **owner_headers,
            "X-Dajoong-Organization": organization_id,
        }

        invitation = client.post(
            "/api/workspace/invitations",
            headers=company_owner_headers,
            json={"email": "editor@example.com", "role": "editor"},
        )
        assert invitation.status_code == 200
        assert "token_sha256" not in invitation.json()["invitation"]
        token = parse_qs(urlparse(invitation.json()["accept_url"]).query)["invite"][0]
        accepted = client.post(
            "/api/workspace/invitations/accept",
            headers={"X-Test-User": "editor"},
            json={"token": token},
        )
        assert accepted.status_code == 200

        imported = client.post(
            "/api/jobs/import",
            headers=company_owner_headers,
            json={"source_name": "A1.1-plan-graph.json", "graph": _graph()},
        )
        assert imported.status_code == 200
        job_id = imported.json()["id"]
        personal_page = client.get(
            "/api/jobs?scope=personal",
            headers=owner_headers,
        ).json()
        company_page = client.get(
            "/api/jobs?scope=organization",
            headers=company_owner_headers,
        ).json()
        assert all(item["id"] != job_id for item in personal_page["items"])
        assert [item["id"] for item in company_page["items"]] == [job_id]
        editor_headers = {
            "X-Test-User": "editor",
            "X-Dajoong-Organization": organization_id,
        }
        assert client.get(f"/api/jobs/{job_id}", headers=editor_headers).status_code == 200

        comment = client.post(
            f"/api/jobs/{job_id}/comments",
            headers={**editor_headers, "Idempotency-Key": "comment-1"},
            json={"body": "Check the selected wall", "entity_ref": "walls:L1:wall:1"},
        )
        assert comment.status_code == 200
        retried_comment = client.post(
            f"/api/jobs/{job_id}/comments",
            headers={**editor_headers, "Idempotency-Key": "comment-1"},
            json={"body": "Check the selected wall", "entity_ref": "walls:L1:wall:1"},
        )
        assert retried_comment.json()["comment"]["id"] == comment.json()["comment"]["id"]
        activity = client.get(f"/api/jobs/{job_id}/activity", headers=editor_headers)
        assert len(activity.json()["items"]) == 1

        baseline_versions = client.get(
            f"/api/jobs/{job_id}/versions",
            headers=editor_headers,
        )
        assert baseline_versions.status_code == 200
        baseline = baseline_versions.json()["items"][0]
        state = client.get(f"/api/jobs/{job_id}", headers=editor_headers).json()
        revised_graph = _graph()
        revised_graph["warnings"] = ["reviewed by the project team"]
        revision = client.post(
            f"/api/jobs/{job_id}/revisions",
            headers=editor_headers,
            json={
                "expected_job_version": state["version"],
                "expected_graph_sha256": graph_content_hash(_graph()),
                "reviewer": "Editor",
                "operations": [],
                "graph": revised_graph,
            },
        )
        assert revision.status_code == 200
        restored = client.post(
            f"/api/jobs/{job_id}/versions/{baseline['graph_sha256']}/restore",
            headers=editor_headers,
            json={"expected_job_version": revision.json()["job_version"]},
        )
        assert restored.status_code == 200
        assert restored.json()["graph_sha256"] == baseline["graph_sha256"]

        stale_restore = client.post(
            f"/api/jobs/{job_id}/versions/{baseline['graph_sha256']}/restore",
            headers=editor_headers,
            json={"expected_job_version": revision.json()["job_version"]},
        )
        assert stale_restore.status_code == 409

        removed = client.delete(
            "/api/workspace/members/editor",
            headers=company_owner_headers,
        )
        assert removed.status_code == 200
        assert client.get(f"/api/jobs/{job_id}", headers=editor_headers).status_code in {403, 404}


def test_member_directory_uses_stable_cursor_pages(tmp_path) -> None:
    service = CollaborationService(tmp_path)
    organization = service.create_organization(
        user_id="owner",
        email="owner@example.com",
        display_name="Owner",
        name="Large Builder",
        approved_domains=[],
        domain_join_enabled=False,
    )
    for index in range(205):
        service.store.save_member(
            WorkspaceMember(
                organization_id=organization.id,
                user_id=f"user-{index:04d}",
                email=f"user-{index:04d}@example.com",
                display_name=f"User {index:04d}",
                role="viewer",
                joined_at=1,
                updated_at=1,
            )
        )
    first, first_cursor = service.store.list_members_page(organization.id, 100, "")
    second, second_cursor = service.store.list_members_page(
        organization.id,
        100,
        first_cursor,
    )
    third, third_cursor = service.store.list_members_page(
        organization.id,
        100,
        second_cursor,
    )
    assert [len(first), len(second), len(third)] == [100, 100, 6]
    assert len({member.user_id for member in first + second + third}) == 206
    assert third_cursor == ""


def test_concurrent_retries_collapse_to_one_membership_and_comment(tmp_path) -> None:
    service = CollaborationService(tmp_path)
    organization = service.create_organization(
        user_id="owner",
        email="owner@example.com",
        display_name="Owner",
        name="Concurrent Builder",
        approved_domains=[],
        domain_join_enabled=False,
    )
    receipt = service.create_invitation(
        organization_id=organization.id,
        inviter_id="owner",
        email="editor@example.com",
        role="editor",
    )

    def accept() -> str:
        try:
            return service.accept_invitation(
                token=receipt.token,
                user_id="editor",
                email="editor@example.com",
                display_name="Editor",
            ).user_id
        except WorkspaceConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as executor:
        accepted = list(executor.map(lambda _: accept(), range(8)))
    assert accepted.count("editor") == 1
    assert len(service.store.list_members(organization.id)) == 2

    def comment() -> str:
        return service.add_comment(
            organization_id=organization.id,
            job_id="job-concurrent",
            actor_id="editor",
            actor_name="Editor",
            body="One network action",
            request_id="one-browser-request",
        ).id

    with ThreadPoolExecutor(max_workers=16) as executor:
        comment_ids = list(executor.map(lambda _: comment(), range(32)))
    assert len(set(comment_ids)) == 1
    assert len(service.store.list_comments("job-concurrent", 100)) == 1


def test_invitation_cannot_downgrade_owner_and_admin_cannot_demote_peer(tmp_path) -> None:
    service = CollaborationService(tmp_path)
    organization = service.create_organization(
        user_id="owner",
        email="owner@example.com",
        display_name="Owner",
        name="Permission Safe Builder",
        approved_domains=[],
        domain_join_enabled=False,
    )
    admin_receipt = service.create_invitation(
        organization_id=organization.id,
        inviter_id="owner",
        email="admin@example.com",
        role="admin",
    )
    service.accept_invitation(
        token=admin_receipt.token,
        user_id="admin",
        email="admin@example.com",
        display_name="Admin",
    )
    peer_receipt = service.create_invitation(
        organization_id=organization.id,
        inviter_id="owner",
        email="peer@example.com",
        role="admin",
    )
    service.accept_invitation(
        token=peer_receipt.token,
        user_id="peer",
        email="peer@example.com",
        display_name="Peer admin",
    )

    with pytest.raises(WorkspaceAccessError):
        service.change_role(
            organization_id=organization.id,
            actor_id="admin",
            user_id="peer",
            role="viewer",
        )

    downgrade_receipt = service.create_invitation(
        organization_id=organization.id,
        inviter_id="admin",
        email="owner@example.com",
        role="viewer",
    )
    accepted_owner = service.accept_invitation(
        token=downgrade_receipt.token,
        user_id="owner",
        email="owner@example.com",
        display_name="Owner",
    )
    assert accepted_owner.role == "owner"
    assert service.require_member(organization.id, "owner", "owner").role == "owner"


def test_aws_workspace_directory_batches_organization_reads() -> None:
    members = [
        WorkspaceMember(
            organization_id=f"org_{index:03d}",
            user_id="one-user",
            email="user@example.com",
            display_name="One user",
            role="viewer",
            joined_at=1,
            updated_at=1,
        )
        for index in range(205)
    ]
    organizations = {
        member.organization_id: Organization(
            id=member.organization_id,
            name=f"Workspace {member.organization_id}",
            slug=member.organization_id,
            created_by="creator",
            created_at=1,
            updated_at=1,
        )
        for member in members
    }

    class FakeDynamoDB:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def query(self, **request):
            assert request["ExpressionAttributeValues"][":pk"]["S"] == "USER#one-user"
            return {
                "Items": [
                    AwsCollaborationStore._item(
                        "USER#one-user",
                        f"ORG#{member.organization_id}",
                        "membership",
                        member,
                    )
                    for member in members
                ]
            }

        def batch_get_item(self, **request):
            keys = request["RequestItems"]["collaboration"]["Keys"]
            self.batch_sizes.append(len(keys))
            return {
                "Responses": {
                    "collaboration": [
                        AwsCollaborationStore._item(
                            key["pk"]["S"],
                            "META",
                            "organization",
                            organizations[key["pk"]["S"].removeprefix("ORG#")],
                        )
                        for key in keys
                    ]
                }
            }

    fake = FakeDynamoDB()
    store = AwsCollaborationStore.__new__(AwsCollaborationStore)
    store.ddb = fake
    store.table_name = "collaboration"
    directory = store.organizations_for_user("one-user")
    assert len(directory) == 205
    assert fake.batch_sizes == [100, 100, 5]
