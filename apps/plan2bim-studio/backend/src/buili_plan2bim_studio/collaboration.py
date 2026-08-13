from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

WorkspaceRole = Literal["owner", "admin", "editor", "commenter", "viewer"]
MemberStatus = Literal["active", "invited", "suspended"]

ROLE_WEIGHT: dict[WorkspaceRole, int] = {
    "viewer": 10,
    "commenter": 20,
    "editor": 30,
    "admin": 40,
    "owner": 50,
}


class WorkspaceAccessError(RuntimeError):
    pass


class WorkspaceConflict(RuntimeError):
    pass


class Organization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    slug: str
    created_by: str
    approved_domains: list[str] = Field(default_factory=list)
    domain_join_enabled: bool = False
    created_at: int
    updated_at: int


class WorkspaceMember(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: str
    user_id: str
    email: str = ""
    display_name: str = ""
    role: WorkspaceRole
    status: MemberStatus = "active"
    joined_at: int
    updated_at: int


class WorkspaceInvitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    organization_id: str
    email: str
    role: WorkspaceRole
    invited_by: str
    token_sha256: str
    status: Literal["pending", "accepted", "revoked", "expired"] = "pending"
    created_at: int
    expires_at: int
    accepted_by: str = ""


class InvitationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invitation: WorkspaceInvitation
    token: str


class ProjectComment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    organization_id: str
    job_id: str
    author_id: str
    author_name: str
    body: str
    entity_ref: str = ""
    assigned_to: str = ""
    status: Literal["open", "resolved"] = "open"
    created_at: int
    updated_at: int


class ActivityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    organization_id: str
    job_id: str = ""
    actor_id: str
    actor_name: str
    kind: str
    summary: str
    entity_ref: str = ""
    created_at: int


class ModelVersionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: str
    job_id: str
    version: int
    graph_sha256: str
    created_by: str
    created_by_name: str
    label: str
    summary: dict[str, int] = Field(default_factory=dict)
    release_allowed: bool = False
    created_at: int


class WorkspacePresence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: str
    job_id: str
    user_id: str
    display_name: str
    color: str
    active_entity: str = ""
    updated_at: int
    expires_at: int


class CollaborationStore(Protocol):
    def create_organization(
        self,
        *,
        user_id: str,
        email: str,
        display_name: str,
        name: str,
        approved_domains: list[str],
        domain_join_enabled: bool,
    ) -> Organization: ...

    def organizations_for_user(
        self, user_id: str
    ) -> list[tuple[Organization, WorkspaceMember]]: ...

    def get_organization(self, organization_id: str) -> Organization: ...

    def get_member(self, organization_id: str, user_id: str) -> WorkspaceMember | None: ...

    def list_members(self, organization_id: str) -> list[WorkspaceMember]: ...

    def list_members_page(
        self,
        organization_id: str,
        limit: int,
        cursor: str,
    ) -> tuple[list[WorkspaceMember], str]: ...

    def save_member(self, member: WorkspaceMember) -> None: ...

    def delete_member(self, organization_id: str, user_id: str) -> None: ...

    def transfer_ownership(
        self,
        previous_owner: WorkspaceMember,
        next_owner: WorkspaceMember,
    ) -> None: ...

    def save_invitation(self, invitation: WorkspaceInvitation) -> None: ...

    def revoke_invitation(
        self,
        previous: WorkspaceInvitation,
        revoked: WorkspaceInvitation,
    ) -> None: ...

    def get_invitation(self, organization_id: str, token_sha256: str) -> WorkspaceInvitation: ...

    def list_invitations(self, organization_id: str) -> list[WorkspaceInvitation]: ...

    def accept_invitation(
        self,
        previous: WorkspaceInvitation,
        accepted: WorkspaceInvitation,
        member: WorkspaceMember,
        previous_member: WorkspaceMember | None,
    ) -> None: ...

    def organization_for_domain(self, domain: str) -> Organization | None: ...

    def add_comment(self, comment: ProjectComment) -> ProjectComment: ...

    def list_comments(self, job_id: str, limit: int) -> list[ProjectComment]: ...

    def get_comment(self, job_id: str, comment_id: str) -> ProjectComment: ...

    def save_comment(self, comment: ProjectComment) -> None: ...

    def add_activity(self, activity: ActivityEvent) -> None: ...

    def list_activity(self, job_id: str, limit: int) -> list[ActivityEvent]: ...

    def record_version(self, version: ModelVersionRecord) -> None: ...

    def list_versions(self, job_id: str, limit: int) -> list[ModelVersionRecord]: ...

    def upsert_presence(self, presence: WorkspacePresence) -> None: ...

    def list_presence(self, job_id: str, now: int) -> list[WorkspacePresence]: ...


def _slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part)[:48] or "workspace"


def _normalized_domain(value: str) -> str:
    return value.strip().lower().lstrip("@").rstrip(".")


def _safe_domains(values: list[str]) -> list[str]:
    return sorted(
        {
            domain
            for raw in values
            if (domain := _normalized_domain(raw))
            and "." in domain
            and len(domain) <= 253
            and all(part and part.replace("-", "").isalnum() for part in domain.split("."))
        }
    )[:20]


def _encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(value: str) -> str:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid workspace cursor") from exc


class LocalCollaborationStore:
    """Atomic local adapter. Production uses the DynamoDB implementation below."""

    def __init__(self, root: Path) -> None:
        self.path = root.resolve() / "collaboration.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _empty(self) -> dict[str, dict[str, Any]]:
        return {
            "organizations": {},
            "members": {},
            "invitations": {},
            "domains": {},
            "comments": {},
            "activity": {},
            "versions": {},
            "presence": {},
        }

    def _read(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, Any] = {}
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        return {key: dict(payload.get(key) or {}) for key in self._empty()}

    def _write(self, payload: dict[str, dict[str, Any]]) -> None:
        staging = self.path.with_suffix(".tmp")
        staging.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        staging.replace(self.path)

    @staticmethod
    def _member_key(organization_id: str, user_id: str) -> str:
        return f"{organization_id}:{user_id}"

    def create_organization(
        self,
        *,
        user_id: str,
        email: str,
        display_name: str,
        name: str,
        approved_domains: list[str],
        domain_join_enabled: bool,
    ) -> Organization:
        now = int(time.time())
        organization = Organization(
            id=f"org_{uuid.uuid4().hex[:20]}",
            name=name.strip(),
            slug=f"{_slug(name)}-{secrets.token_hex(3)}",
            created_by=user_id,
            approved_domains=_safe_domains(approved_domains),
            domain_join_enabled=domain_join_enabled,
            created_at=now,
            updated_at=now,
        )
        member = WorkspaceMember(
            organization_id=organization.id,
            user_id=user_id,
            email=email.lower(),
            display_name=display_name or email.split("@", 1)[0],
            role="owner",
            joined_at=now,
            updated_at=now,
        )
        with self._lock:
            payload = self._read()
            for domain in organization.approved_domains:
                if domain in payload["domains"]:
                    raise WorkspaceConflict(f"{domain} is already connected to a workspace")
            payload["organizations"][organization.id] = organization.model_dump(mode="json")
            payload["members"][self._member_key(organization.id, user_id)] = member.model_dump(
                mode="json"
            )
            if organization.domain_join_enabled:
                for domain in organization.approved_domains:
                    payload["domains"][domain] = organization.id
            self._write(payload)
        return organization

    def organizations_for_user(self, user_id: str) -> list[tuple[Organization, WorkspaceMember]]:
        with self._lock:
            payload = self._read()
            result: list[tuple[Organization, WorkspaceMember]] = []
            for raw in payload["members"].values():
                member = WorkspaceMember.model_validate(raw)
                if member.user_id != user_id or member.status != "active":
                    continue
                raw_org = payload["organizations"].get(member.organization_id)
                if raw_org:
                    result.append((Organization.model_validate(raw_org), member))
            return sorted(result, key=lambda pair: pair[0].name.casefold())

    def get_organization(self, organization_id: str) -> Organization:
        with self._lock:
            raw = self._read()["organizations"].get(organization_id)
            if not raw:
                raise KeyError(organization_id)
            return Organization.model_validate(raw)

    def get_member(self, organization_id: str, user_id: str) -> WorkspaceMember | None:
        with self._lock:
            raw = self._read()["members"].get(self._member_key(organization_id, user_id))
            return WorkspaceMember.model_validate(raw) if raw else None

    def list_members(self, organization_id: str) -> list[WorkspaceMember]:
        with self._lock:
            members = [
                WorkspaceMember.model_validate(raw)
                for raw in self._read()["members"].values()
                if raw.get("organization_id") == organization_id
            ]
        return sorted(
            members,
            key=lambda item: (item.status != "active", item.display_name.casefold()),
        )

    def list_members_page(
        self,
        organization_id: str,
        limit: int,
        cursor: str,
    ) -> tuple[list[WorkspaceMember], str]:
        after = _decode_cursor(cursor) if cursor else ""
        with self._lock:
            keyed = sorted(
                (
                    self._member_key(organization_id, str(raw.get("user_id", ""))),
                    WorkspaceMember.model_validate(raw),
                )
                for raw in self._read()["members"].values()
                if raw.get("organization_id") == organization_id
            )
        page = [member for key, member in keyed if key > after][:limit]
        last_key = self._member_key(organization_id, page[-1].user_id) if page else ""
        has_more = bool(page) and any(key > last_key for key, _ in keyed)
        return page, _encode_cursor(last_key) if has_more else ""

    def save_member(self, member: WorkspaceMember) -> None:
        with self._lock:
            payload = self._read()
            payload["members"][self._member_key(member.organization_id, member.user_id)] = (
                member.model_dump(mode="json")
            )
            self._write(payload)

    def delete_member(self, organization_id: str, user_id: str) -> None:
        with self._lock:
            payload = self._read()
            payload["members"].pop(self._member_key(organization_id, user_id), None)
            self._write(payload)

    def transfer_ownership(
        self,
        previous_owner: WorkspaceMember,
        next_owner: WorkspaceMember,
    ) -> None:
        with self._lock:
            payload = self._read()
            current_previous = payload["members"].get(
                self._member_key(previous_owner.organization_id, previous_owner.user_id)
            )
            current_next = payload["members"].get(
                self._member_key(next_owner.organization_id, next_owner.user_id)
            )
            if not current_previous or not current_next:
                raise WorkspaceConflict("workspace membership changed")
            if WorkspaceMember.model_validate(current_previous).role != "owner":
                raise WorkspaceConflict("workspace ownership changed")
            payload["members"][
                self._member_key(previous_owner.organization_id, previous_owner.user_id)
            ] = previous_owner.model_dump(mode="json")
            payload["members"][
                self._member_key(next_owner.organization_id, next_owner.user_id)
            ] = next_owner.model_dump(mode="json")
            self._write(payload)

    def save_invitation(self, invitation: WorkspaceInvitation) -> None:
        with self._lock:
            payload = self._read()
            if invitation.status == "pending":
                duplicate = next(
                    (
                        raw
                        for raw in payload["invitations"].values()
                        if raw.get("organization_id") == invitation.organization_id
                        and raw.get("email", "").casefold() == invitation.email.casefold()
                        and raw.get("status") == "pending"
                        and raw.get("id") != invitation.id
                        and int(raw.get("expires_at", 0)) > int(time.time())
                    ),
                    None,
                )
                if duplicate:
                    raise WorkspaceConflict("this email already has a pending invitation")
            key = f"{invitation.organization_id}:{invitation.token_sha256}"
            payload["invitations"][key] = invitation.model_dump(mode="json")
            self._write(payload)

    def revoke_invitation(
        self,
        previous: WorkspaceInvitation,
        revoked: WorkspaceInvitation,
    ) -> None:
        with self._lock:
            payload = self._read()
            key = f"{previous.organization_id}:{previous.token_sha256}"
            current = payload["invitations"].get(key)
            if not current or WorkspaceInvitation.model_validate(current) != previous:
                raise WorkspaceConflict("invitation changed while it was being revoked")
            payload["invitations"][key] = revoked.model_dump(mode="json")
            self._write(payload)

    def get_invitation(self, organization_id: str, token_sha256: str) -> WorkspaceInvitation:
        with self._lock:
            raw = self._read()["invitations"].get(f"{organization_id}:{token_sha256}")
            if not raw:
                raise KeyError(token_sha256)
            return WorkspaceInvitation.model_validate(raw)

    def list_invitations(self, organization_id: str) -> list[WorkspaceInvitation]:
        with self._lock:
            invitations = [
                WorkspaceInvitation.model_validate(raw)
                for raw in self._read()["invitations"].values()
                if raw.get("organization_id") == organization_id
            ]
        return sorted(invitations, key=lambda item: item.created_at, reverse=True)

    def accept_invitation(
        self,
        previous: WorkspaceInvitation,
        accepted: WorkspaceInvitation,
        member: WorkspaceMember,
        previous_member: WorkspaceMember | None,
    ) -> None:
        with self._lock:
            payload = self._read()
            invitation_key = f"{previous.organization_id}:{previous.token_sha256}"
            raw = payload["invitations"].get(invitation_key)
            if not raw or WorkspaceInvitation.model_validate(raw) != previous:
                raise WorkspaceConflict("invitation changed while it was being accepted")
            member_key = self._member_key(member.organization_id, member.user_id)
            current_member = payload["members"].get(member_key)
            expected_member = (
                previous_member.model_dump(mode="json") if previous_member else None
            )
            if current_member != expected_member:
                raise WorkspaceConflict("membership changed while the invitation was accepted")
            payload["members"][member_key] = (
                member.model_dump(mode="json")
            )
            payload["invitations"][invitation_key] = accepted.model_dump(mode="json")
            self._write(payload)

    def organization_for_domain(self, domain: str) -> Organization | None:
        with self._lock:
            payload = self._read()
            organization_id = payload["domains"].get(_normalized_domain(domain))
            raw = payload["organizations"].get(organization_id) if organization_id else None
            return Organization.model_validate(raw) if raw else None

    def add_comment(self, comment: ProjectComment) -> ProjectComment:
        with self._lock:
            payload = self._read()
            key = f"{comment.job_id}:{comment.id}"
            existing = payload["comments"].get(key)
            if existing:
                return ProjectComment.model_validate(existing)
            payload["comments"][key] = comment.model_dump(mode="json")
            self._write(payload)
        return comment

    def list_comments(self, job_id: str, limit: int) -> list[ProjectComment]:
        with self._lock:
            comments = [
                ProjectComment.model_validate(raw)
                for raw in self._read()["comments"].values()
                if raw.get("job_id") == job_id
            ]
        return sorted(comments, key=lambda item: item.created_at, reverse=True)[:limit]

    def get_comment(self, job_id: str, comment_id: str) -> ProjectComment:
        with self._lock:
            raw = self._read()["comments"].get(f"{job_id}:{comment_id}")
            if not raw:
                raise KeyError(comment_id)
            return ProjectComment.model_validate(raw)

    def save_comment(self, comment: ProjectComment) -> None:
        with self._lock:
            payload = self._read()
            payload["comments"][f"{comment.job_id}:{comment.id}"] = comment.model_dump(mode="json")
            self._write(payload)

    def add_activity(self, activity: ActivityEvent) -> None:
        with self._lock:
            payload = self._read()
            payload["activity"][f"{activity.job_id}:{activity.id}"] = activity.model_dump(
                mode="json"
            )
            self._write(payload)

    def list_activity(self, job_id: str, limit: int) -> list[ActivityEvent]:
        with self._lock:
            items = [
                ActivityEvent.model_validate(raw)
                for raw in self._read()["activity"].values()
                if raw.get("job_id") == job_id
            ]
        return sorted(items, key=lambda item: item.created_at, reverse=True)[:limit]

    def record_version(self, version: ModelVersionRecord) -> None:
        with self._lock:
            payload = self._read()
            key = f"{version.job_id}:{version.version}"
            payload["versions"].setdefault(key, version.model_dump(mode="json"))
            self._write(payload)

    def list_versions(self, job_id: str, limit: int) -> list[ModelVersionRecord]:
        with self._lock:
            items = [
                ModelVersionRecord.model_validate(raw)
                for raw in self._read()["versions"].values()
                if raw.get("job_id") == job_id
            ]
        return sorted(items, key=lambda item: item.version, reverse=True)[:limit]

    def upsert_presence(self, presence: WorkspacePresence) -> None:
        with self._lock:
            payload = self._read()
            payload["presence"][f"{presence.job_id}:{presence.user_id}"] = presence.model_dump(
                mode="json"
            )
            self._write(payload)

    def list_presence(self, job_id: str, now: int) -> list[WorkspacePresence]:
        with self._lock:
            payload = self._read()
            result: list[WorkspacePresence] = []
            dirty = False
            for key, raw in list(payload["presence"].items()):
                presence = WorkspacePresence.model_validate(raw)
                if presence.expires_at <= now:
                    payload["presence"].pop(key, None)
                    dirty = True
                elif presence.job_id == job_id:
                    result.append(presence)
            if dirty:
                self._write(payload)
        return sorted(result, key=lambda item: item.display_name.casefold())


class AwsCollaborationStore:
    """Single-table tenant store using partition-local queries only; no table scans."""

    def __init__(self) -> None:
        import boto3

        self.ddb = boto3.client("dynamodb")
        self.table_name = os.environ["DAJOONG_COLLABORATION_TABLE"]

    @staticmethod
    def _item(pk: str, sk: str, kind: str, value: BaseModel, **extra: Any) -> dict[str, Any]:
        item: dict[str, Any] = {
            "pk": {"S": pk},
            "sk": {"S": sk},
            "kind": {"S": kind},
            "data": {"S": value.model_dump_json()},
        }
        for key, raw in extra.items():
            if isinstance(raw, int):
                item[key] = {"N": str(raw)}
            elif raw:
                item[key] = {"S": str(raw)}
        return item

    def _get(self, pk: str, sk: str, model: type[BaseModel]) -> Any:
        response = self.ddb.get_item(
            TableName=self.table_name,
            Key={"pk": {"S": pk}, "sk": {"S": sk}},
            ConsistentRead=True,
        )
        if "Item" not in response:
            raise KeyError(sk)
        return model.model_validate_json(response["Item"]["data"]["S"])

    def _query(
        self,
        pk: str,
        prefix: str,
        model: type[BaseModel],
        limit: int = 1000,
    ) -> list[Any]:
        items: list[Any] = []
        cursor: dict[str, Any] | None = None
        while len(items) < limit:
            request: dict[str, Any] = {
                "TableName": self.table_name,
                "KeyConditionExpression": "pk = :pk AND begins_with(sk, :prefix)",
                "ExpressionAttributeValues": {
                    ":pk": {"S": pk},
                    ":prefix": {"S": prefix},
                },
                "ScanIndexForward": False,
                "Limit": min(250, limit - len(items)),
            }
            if cursor:
                request["ExclusiveStartKey"] = cursor
            response = self.ddb.query(**request)
            items.extend(
                model.model_validate_json(item["data"]["S"])
                for item in response.get("Items", [])
            )
            cursor = response.get("LastEvaluatedKey")
            if not cursor:
                break
        return items[:limit]

    def create_organization(
        self,
        *,
        user_id: str,
        email: str,
        display_name: str,
        name: str,
        approved_domains: list[str],
        domain_join_enabled: bool,
    ) -> Organization:
        now = int(time.time())
        organization = Organization(
            id=f"org_{uuid.uuid4().hex[:20]}",
            name=name.strip(),
            slug=f"{_slug(name)}-{secrets.token_hex(3)}",
            created_by=user_id,
            approved_domains=_safe_domains(approved_domains),
            domain_join_enabled=domain_join_enabled,
            created_at=now,
            updated_at=now,
        )
        member = WorkspaceMember(
            organization_id=organization.id,
            user_id=user_id,
            email=email.lower(),
            display_name=display_name or email.split("@", 1)[0],
            role="owner",
            joined_at=now,
            updated_at=now,
        )
        writes: list[dict[str, Any]] = [
            {
                "Put": {
                    "TableName": self.table_name,
                    "Item": self._item(
                        f"ORG#{organization.id}", "META", "organization", organization
                    ),
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
            {
                "Put": {
                    "TableName": self.table_name,
                    "Item": self._item(
                        f"ORG#{organization.id}", f"MEMBER#{user_id}", "member", member
                    ),
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
            {
                "Put": {
                    "TableName": self.table_name,
                    "Item": self._item(
                        f"USER#{user_id}", f"ORG#{organization.id}", "membership", member
                    ),
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
        ]
        if organization.domain_join_enabled:
            writes.extend(
                {
                    "Put": {
                        "TableName": self.table_name,
                        "Item": {
                            "pk": {"S": f"DOMAIN#{domain}"},
                            "sk": {"S": "ORG"},
                            "kind": {"S": "domain"},
                            "organization_id": {"S": organization.id},
                        },
                        "ConditionExpression": "attribute_not_exists(pk)",
                    }
                }
                for domain in organization.approved_domains
            )
        try:
            self.ddb.transact_write_items(TransactItems=writes)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"TransactionCanceledException", "ConditionalCheckFailedException"}:
                raise WorkspaceConflict(
                    "a domain is already connected to another workspace"
                ) from exc
            raise
        return organization

    def organizations_for_user(self, user_id: str) -> list[tuple[Organization, WorkspaceMember]]:
        members = self._query(f"USER#{user_id}", "ORG#", WorkspaceMember)
        active_members = [member for member in members if member.status == "active"]
        organizations: dict[str, Organization] = {}
        organization_ids = list(dict.fromkeys(member.organization_id for member in active_members))
        for offset in range(0, len(organization_ids), 100):
            chunk = organization_ids[offset : offset + 100]
            request_items: dict[str, Any] = {
                self.table_name: {
                    "Keys": [
                        {
                            "pk": {"S": f"ORG#{organization_id}"},
                            "sk": {"S": "META"},
                        }
                        for organization_id in chunk
                    ],
                    "ConsistentRead": True,
                }
            }
            for attempt in range(4):
                response = self.ddb.batch_get_item(RequestItems=request_items)
                for item in response.get("Responses", {}).get(self.table_name, []):
                    organization = Organization.model_validate_json(item["data"]["S"])
                    organizations[organization.id] = organization
                unprocessed = response.get("UnprocessedKeys", {}).get(self.table_name)
                if not unprocessed:
                    break
                request_items = {self.table_name: unprocessed}
                time.sleep(0.01 * (2**attempt))
            # BatchGet can legally return unprocessed keys.  After bounded
            # backoff, strongly-consistent point reads keep sign-in reliable
            # instead of silently dropping a workspace from the account menu.
            for organization_id in chunk:
                if organization_id not in organizations:
                    organizations[organization_id] = self.get_organization(organization_id)
        result = [
            (organizations[member.organization_id], member)
            for member in active_members
            if member.organization_id in organizations
        ]
        return sorted(result, key=lambda pair: pair[0].name.casefold())

    def get_organization(self, organization_id: str) -> Organization:
        return self._get(f"ORG#{organization_id}", "META", Organization)

    def get_member(self, organization_id: str, user_id: str) -> WorkspaceMember | None:
        try:
            return self._get(f"ORG#{organization_id}", f"MEMBER#{user_id}", WorkspaceMember)
        except KeyError:
            return None

    def list_members(self, organization_id: str) -> list[WorkspaceMember]:
        members = self._query(f"ORG#{organization_id}", "MEMBER#", WorkspaceMember)
        return sorted(
            members,
            key=lambda item: (item.status != "active", item.display_name.casefold()),
        )

    def list_members_page(
        self,
        organization_id: str,
        limit: int,
        cursor: str,
    ) -> tuple[list[WorkspaceMember], str]:
        request: dict[str, Any] = {
            "TableName": self.table_name,
            "KeyConditionExpression": "pk = :pk AND begins_with(sk, :prefix)",
            "ExpressionAttributeValues": {
                ":pk": {"S": f"ORG#{organization_id}"},
                ":prefix": {"S": "MEMBER#"},
            },
            "ScanIndexForward": True,
            "Limit": limit,
        }
        if cursor:
            last_key = _decode_cursor(cursor)
            if not last_key.startswith("MEMBER#"):
                raise ValueError("invalid workspace cursor")
            request["ExclusiveStartKey"] = {
                "pk": {"S": f"ORG#{organization_id}"},
                "sk": {"S": last_key},
            }
        response = self.ddb.query(**request)
        members = [
            WorkspaceMember.model_validate_json(item["data"]["S"])
            for item in response.get("Items", [])
        ]
        next_key = response.get("LastEvaluatedKey", {}).get("sk", {}).get("S", "")
        return members, _encode_cursor(next_key) if next_key else ""

    def save_member(self, member: WorkspaceMember) -> None:
        self.ddb.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": self.table_name,
                        "Item": self._item(
                            f"ORG#{member.organization_id}",
                            f"MEMBER#{member.user_id}",
                            "member",
                            member,
                        ),
                    }
                },
                {
                    "Put": {
                        "TableName": self.table_name,
                        "Item": self._item(
                            f"USER#{member.user_id}",
                            f"ORG#{member.organization_id}",
                            "membership",
                            member,
                        ),
                    }
                },
            ]
        )

    def delete_member(self, organization_id: str, user_id: str) -> None:
        self.ddb.transact_write_items(
            TransactItems=[
                {
                    "Delete": {
                        "TableName": self.table_name,
                        "Key": {
                            "pk": {"S": f"ORG#{organization_id}"},
                            "sk": {"S": f"MEMBER#{user_id}"},
                        },
                    }
                },
                {
                    "Delete": {
                        "TableName": self.table_name,
                        "Key": {
                            "pk": {"S": f"USER#{user_id}"},
                            "sk": {"S": f"ORG#{organization_id}"},
                        },
                    }
                },
            ]
        )

    def transfer_ownership(
        self,
        previous_owner: WorkspaceMember,
        next_owner: WorkspaceMember,
    ) -> None:
        items = [
            (
                f"ORG#{previous_owner.organization_id}",
                f"MEMBER#{previous_owner.user_id}",
                "member",
                previous_owner,
            ),
            (
                f"USER#{previous_owner.user_id}",
                f"ORG#{previous_owner.organization_id}",
                "membership",
                previous_owner,
            ),
            (
                f"ORG#{next_owner.organization_id}",
                f"MEMBER#{next_owner.user_id}",
                "member",
                next_owner,
            ),
            (
                f"USER#{next_owner.user_id}",
                f"ORG#{next_owner.organization_id}",
                "membership",
                next_owner,
            ),
        ]
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(pk, sk, kind, member),
                            **(
                                {
                                    "ConditionExpression": "contains(#data, :owner)",
                                    "ExpressionAttributeNames": {"#data": "data"},
                                    "ExpressionAttributeValues": {
                                        ":owner": {"S": '\"role\":\"owner\"'}
                                    },
                                }
                                if member.user_id == previous_owner.user_id
                                else {}
                            ),
                        }
                    }
                    for pk, sk, kind, member in items
                ]
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"TransactionCanceledException", "ConditionalCheckFailedException"}:
                raise WorkspaceConflict("workspace ownership changed") from exc
            raise

    def save_invitation(self, invitation: WorkspaceInvitation) -> None:
        email_hash = hashlib.sha256(invitation.email.casefold().encode()).hexdigest()
        invitation_put = {
            "Put": {
                "TableName": self.table_name,
                "Item": self._item(
                    f"ORG#{invitation.organization_id}",
                    f"INVITE#{invitation.token_sha256}",
                    "invitation",
                    invitation,
                    expires_at=invitation.expires_at,
                ),
            }
        }
        guard_key = {
            "pk": {"S": f"ORG#{invitation.organization_id}"},
            "sk": {"S": f"INVITE_EMAIL#{email_hash}"},
        }
        if invitation.status == "pending":
            guard_action: dict[str, Any] = {
                "Put": {
                    "TableName": self.table_name,
                    "Item": {
                        **guard_key,
                        "kind": {"S": "invitation_guard"},
                        "expires_at": {"N": str(invitation.expires_at)},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            }
        else:
            guard_action = {
                "Delete": {
                    "TableName": self.table_name,
                    "Key": guard_key,
                }
            }
        try:
            self.ddb.transact_write_items(
                TransactItems=[invitation_put, guard_action]
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"TransactionCanceledException", "ConditionalCheckFailedException"}:
                raise WorkspaceConflict(
                    "this email already has a pending invitation"
                ) from exc
            raise

    def revoke_invitation(
        self,
        previous: WorkspaceInvitation,
        revoked: WorkspaceInvitation,
    ) -> None:
        email_hash = hashlib.sha256(revoked.email.casefold().encode()).hexdigest()
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(
                                f"ORG#{revoked.organization_id}",
                                f"INVITE#{revoked.token_sha256}",
                                "invitation",
                                revoked,
                                expires_at=revoked.expires_at,
                            ),
                            "ConditionExpression": "#data = :previous",
                            "ExpressionAttributeNames": {"#data": "data"},
                            "ExpressionAttributeValues": {
                                ":previous": {"S": previous.model_dump_json()}
                            },
                        }
                    },
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"ORG#{revoked.organization_id}"},
                                "sk": {"S": f"INVITE_EMAIL#{email_hash}"},
                            },
                        }
                    },
                ]
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"TransactionCanceledException", "ConditionalCheckFailedException"}:
                raise WorkspaceConflict(
                    "invitation changed while it was being revoked"
                ) from exc
            raise

    def get_invitation(self, organization_id: str, token_sha256: str) -> WorkspaceInvitation:
        return self._get(
            f"ORG#{organization_id}", f"INVITE#{token_sha256}", WorkspaceInvitation
        )

    def list_invitations(self, organization_id: str) -> list[WorkspaceInvitation]:
        return self._query(f"ORG#{organization_id}", "INVITE#", WorkspaceInvitation)

    def accept_invitation(
        self,
        previous: WorkspaceInvitation,
        accepted: WorkspaceInvitation,
        member: WorkspaceMember,
        previous_member: WorkspaceMember | None,
    ) -> None:
        member_condition = (
            {
                "ConditionExpression": "#data = :previous_member",
                "ExpressionAttributeNames": {"#data": "data"},
                "ExpressionAttributeValues": {
                    ":previous_member": {"S": previous_member.model_dump_json()}
                },
            }
            if previous_member
            else {"ConditionExpression": "attribute_not_exists(pk)"}
        )
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(
                                f"ORG#{member.organization_id}",
                                f"MEMBER#{member.user_id}",
                                "member",
                                member,
                            ),
                            **member_condition,
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(
                                f"USER#{member.user_id}",
                                f"ORG#{member.organization_id}",
                                "membership",
                                member,
                            ),
                            **member_condition,
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(
                                f"ORG#{accepted.organization_id}",
                                f"INVITE#{accepted.token_sha256}",
                                "invitation",
                                accepted,
                                expires_at=accepted.expires_at,
                            ),
                            "ConditionExpression": "#data = :previous",
                            "ExpressionAttributeNames": {"#data": "data"},
                            "ExpressionAttributeValues": {
                                ":previous": {"S": previous.model_dump_json()}
                            },
                        }
                    },
                    {
                        "Delete": {
                            "TableName": self.table_name,
                            "Key": {
                                "pk": {"S": f"ORG#{accepted.organization_id}"},
                                "sk": {
                                    "S": "INVITE_EMAIL#"
                                    + hashlib.sha256(
                                        accepted.email.casefold().encode()
                                    ).hexdigest()
                                },
                            },
                        }
                    },
                ]
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"TransactionCanceledException", "ConditionalCheckFailedException"}:
                raise WorkspaceConflict(
                    "invitation changed while it was being accepted"
                ) from exc
            raise

    def organization_for_domain(self, domain: str) -> Organization | None:
        response = self.ddb.get_item(
            TableName=self.table_name,
            Key={"pk": {"S": f"DOMAIN#{_normalized_domain(domain)}"}, "sk": {"S": "ORG"}},
            ConsistentRead=True,
        )
        organization_id = response.get("Item", {}).get("organization_id", {}).get("S", "")
        return self.get_organization(organization_id) if organization_id else None

    def add_comment(self, comment: ProjectComment) -> ProjectComment:
        timeline_key = f"COMMENT#{comment.created_at:010d}#{comment.id}"
        lookup_key = f"COMMENT#{comment.job_id}#{comment.id}"
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(
                                f"JOB#{comment.job_id}", timeline_key, "comment", comment
                            ),
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._item(lookup_key, "META", "comment_lookup", comment),
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                ]
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code not in {"ConditionalCheckFailedException", "TransactionCanceledException"}:
                raise
            return self.get_comment(comment.job_id, comment.id)
        return comment

    def list_comments(self, job_id: str, limit: int) -> list[ProjectComment]:
        return self._query(f"JOB#{job_id}", "COMMENT#", ProjectComment, limit)

    def get_comment(self, job_id: str, comment_id: str) -> ProjectComment:
        return self._get(f"COMMENT#{job_id}#{comment_id}", "META", ProjectComment)

    def save_comment(self, comment: ProjectComment) -> None:
        current = self.get_comment(comment.job_id, comment.id)
        timeline_key = f"COMMENT#{current.created_at:010d}#{comment.id}"
        self.ddb.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": self.table_name,
                        "Item": self._item(
                            f"JOB#{comment.job_id}", timeline_key, "comment", comment
                        ),
                    }
                },
                {
                    "Put": {
                        "TableName": self.table_name,
                        "Item": self._item(
                            f"COMMENT#{comment.job_id}#{comment.id}",
                            "META",
                            "comment_lookup",
                            comment,
                        ),
                    }
                },
            ]
        )

    def add_activity(self, activity: ActivityEvent) -> None:
        self.ddb.put_item(
            TableName=self.table_name,
            Item=self._item(
                f"JOB#{activity.job_id}",
                f"ACTIVITY#{activity.created_at:010d}#{activity.id}",
                "activity",
                activity,
            ),
        )

    def list_activity(self, job_id: str, limit: int) -> list[ActivityEvent]:
        return self._query(f"JOB#{job_id}", "ACTIVITY#", ActivityEvent, limit)

    def record_version(self, version: ModelVersionRecord) -> None:
        try:
            self.ddb.put_item(
                TableName=self.table_name,
                Item=self._item(
                    f"JOB#{version.job_id}",
                    f"VERSION#{version.version:010d}",
                    "version",
                    version,
                ),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code != "ConditionalCheckFailedException":
                raise

    def list_versions(self, job_id: str, limit: int) -> list[ModelVersionRecord]:
        return self._query(f"JOB#{job_id}", "VERSION#", ModelVersionRecord, limit)

    def upsert_presence(self, presence: WorkspacePresence) -> None:
        self.ddb.put_item(
            TableName=self.table_name,
            Item=self._item(
                f"JOB#{presence.job_id}",
                f"PRESENCE#{presence.user_id}",
                "presence",
                presence,
                expires_at=presence.expires_at,
            ),
        )

    def list_presence(self, job_id: str, now: int) -> list[WorkspacePresence]:
        return [
            item
            for item in self._query(f"JOB#{job_id}", "PRESENCE#", WorkspacePresence)
            if item.expires_at > now
        ]


class CollaborationService:
    def __init__(self, root: Path) -> None:
        self.store: CollaborationStore = (
            AwsCollaborationStore()
            if os.environ.get("DAJOONG_RUNTIME", "local").lower() == "aws"
            else LocalCollaborationStore(root)
        )

    def create_organization(
        self,
        *,
        user_id: str,
        email: str,
        display_name: str,
        name: str,
        approved_domains: list[str],
        domain_join_enabled: bool,
    ) -> Organization:
        if not 2 <= len(name.strip()) <= 100:
            raise ValueError("workspace name must be between 2 and 100 characters")
        return self.store.create_organization(
            user_id=user_id,
            email=email,
            display_name=display_name,
            name=name,
            approved_domains=approved_domains,
            domain_join_enabled=domain_join_enabled,
        )

    def require_member(
        self,
        organization_id: str,
        user_id: str,
        minimum_role: WorkspaceRole = "viewer",
    ) -> WorkspaceMember:
        member = self.store.get_member(organization_id, user_id)
        if (
            not member
            or member.status != "active"
            or ROLE_WEIGHT[member.role] < ROLE_WEIGHT[minimum_role]
        ):
            raise WorkspaceAccessError("workspace access is not available")
        return member

    def create_invitation(
        self,
        *,
        organization_id: str,
        inviter_id: str,
        email: str,
        role: WorkspaceRole,
    ) -> InvitationReceipt:
        self.require_member(organization_id, inviter_id, "admin")
        if role == "owner":
            raise ValueError("ownership is transferred separately")
        normalized_email = email.strip().lower()
        if "@" not in normalized_email or len(normalized_email) > 320:
            raise ValueError("a valid email address is required")
        now = int(time.time())
        active_invitations = [
            item
            for item in self.store.list_invitations(organization_id)
            if item.status == "pending" and item.expires_at > now
        ]
        if any(item.email.casefold() == normalized_email.casefold() for item in active_invitations):
            raise WorkspaceConflict("this email already has a pending invitation")
        if len(active_invitations) >= 500:
            raise WorkspaceConflict("revoke unused invitations before creating more")
        secret = secrets.token_urlsafe(32)
        token = f"{organization_id}.{secret}"
        invitation = WorkspaceInvitation(
            id=f"inv_{uuid.uuid4().hex[:20]}",
            organization_id=organization_id,
            email=normalized_email,
            role=role,
            invited_by=inviter_id,
            token_sha256=hashlib.sha256(token.encode()).hexdigest(),
            created_at=now,
            expires_at=now + 7 * 24 * 60 * 60,
        )
        self.store.save_invitation(invitation)
        return InvitationReceipt(invitation=invitation, token=token)

    def accept_invitation(
        self,
        *,
        token: str,
        user_id: str,
        email: str,
        display_name: str,
    ) -> WorkspaceMember:
        organization_id, separator, _ = token.partition(".")
        if not separator or not organization_id.startswith("org_"):
            raise ValueError("invitation is invalid")
        invitation = self.store.get_invitation(
            organization_id, hashlib.sha256(token.encode()).hexdigest()
        )
        now = int(time.time())
        if invitation.status != "pending" or invitation.expires_at <= now:
            raise WorkspaceConflict("invitation is no longer active")
        if invitation.email.casefold() != email.strip().casefold():
            raise WorkspaceAccessError("sign in with the invited email address")
        existing = self.store.get_member(organization_id, user_id)
        previous_member = existing.model_copy(deep=True) if existing else None
        member = existing or WorkspaceMember(
            organization_id=organization_id,
            user_id=user_id,
            email=email.lower(),
            display_name=display_name or email.split("@", 1)[0],
            role=invitation.role,
            joined_at=now,
            updated_at=now,
        )
        # Accepting an invitation must never reduce an existing member's
        # authority.  In particular, an administrator must not be able to
        # invite the owner's verified email as a viewer and orphan the
        # workspace when the owner follows the link.
        if not existing or ROLE_WEIGHT[invitation.role] > ROLE_WEIGHT[existing.role]:
            member.role = invitation.role
        member.status = "active"
        member.updated_at = now
        previous = invitation.model_copy(deep=True)
        invitation.status = "accepted"
        invitation.accepted_by = user_id
        self.store.accept_invitation(previous, invitation, member, previous_member)
        return member

    def revoke_invitation(
        self,
        *,
        organization_id: str,
        actor_id: str,
        invitation_id: str,
    ) -> WorkspaceInvitation:
        self.require_member(organization_id, actor_id, "admin")
        invitation = next(
            (
                item
                for item in self.store.list_invitations(organization_id)
                if item.id == invitation_id
            ),
            None,
        )
        if not invitation:
            raise KeyError(invitation_id)
        if invitation.status != "pending":
            raise WorkspaceConflict("invitation is no longer active")
        previous = invitation.model_copy(deep=True)
        invitation.status = "revoked"
        self.store.revoke_invitation(previous, invitation)
        return invitation

    def join_by_domain(
        self, *, user_id: str, email: str, display_name: str
    ) -> WorkspaceMember:
        _, separator, domain = email.strip().lower().partition("@")
        if not separator:
            raise ValueError("a verified work email is required")
        organization = self.store.organization_for_domain(domain)
        if not organization or not organization.domain_join_enabled:
            raise WorkspaceAccessError("this email domain is not open for workspace joining")
        existing = self.store.get_member(organization.id, user_id)
        if existing and existing.status == "active":
            return existing
        now = int(time.time())
        member = WorkspaceMember(
            organization_id=organization.id,
            user_id=user_id,
            email=email.lower(),
            display_name=display_name or email.split("@", 1)[0],
            role="editor",
            joined_at=now,
            updated_at=now,
        )
        self.store.save_member(member)
        return member

    def change_role(
        self,
        *,
        organization_id: str,
        actor_id: str,
        user_id: str,
        role: WorkspaceRole,
    ) -> WorkspaceMember:
        actor = self.require_member(organization_id, actor_id, "admin")
        target = self.require_member(organization_id, user_id)
        if target.role == "owner" or role == "owner":
            raise WorkspaceConflict("workspace ownership must be transferred explicitly")
        if target.role == "admin" and actor.role != "owner" and actor.user_id != target.user_id:
            raise WorkspaceAccessError("only an owner can change another administrator's role")
        target.role = role
        target.updated_at = int(time.time())
        self.store.save_member(target)
        return target

    def remove_member(self, *, organization_id: str, actor_id: str, user_id: str) -> None:
        actor = self.require_member(organization_id, actor_id, "admin")
        target = self.require_member(organization_id, user_id)
        if target.role == "owner":
            raise WorkspaceConflict("the workspace owner cannot be removed")
        if actor.role == "admin" and target.role == "admin" and actor.user_id != target.user_id:
            raise WorkspaceAccessError("only an owner can remove another admin")
        self.store.delete_member(organization_id, user_id)

    def transfer_ownership(
        self,
        *,
        organization_id: str,
        actor_id: str,
        user_id: str,
    ) -> tuple[WorkspaceMember, WorkspaceMember]:
        previous_owner = self.require_member(organization_id, actor_id, "owner")
        next_owner = self.require_member(organization_id, user_id, "admin")
        if previous_owner.user_id == next_owner.user_id:
            raise WorkspaceConflict("select another administrator as the new owner")
        now = int(time.time())
        previous_owner.role = "admin"
        previous_owner.updated_at = now
        next_owner.role = "owner"
        next_owner.updated_at = now
        self.store.transfer_ownership(previous_owner, next_owner)
        return previous_owner, next_owner

    def add_activity(
        self,
        *,
        organization_id: str,
        job_id: str,
        actor_id: str,
        actor_name: str,
        kind: str,
        summary: str,
        entity_ref: str = "",
        request_id: str = "",
        created_at: int | None = None,
    ) -> ActivityEvent:
        event_id = (
            hashlib.sha256(
                f"{organization_id}:{job_id}:{actor_id}:{request_id}".encode()
            ).hexdigest()[:20]
            if request_id
            else uuid.uuid4().hex[:20]
        )
        activity = ActivityEvent(
            id=f"evt_{event_id}",
            organization_id=organization_id,
            job_id=job_id,
            actor_id=actor_id,
            actor_name=actor_name,
            kind=kind,
            summary=summary[:500],
            entity_ref=entity_ref[:300],
            created_at=created_at or int(time.time()),
        )
        self.store.add_activity(activity)
        return activity

    def add_comment(
        self,
        *,
        organization_id: str,
        job_id: str,
        actor_id: str,
        actor_name: str,
        body: str,
        request_id: str,
        entity_ref: str = "",
        assigned_to: str = "",
    ) -> ProjectComment:
        self.require_member(organization_id, actor_id, "commenter")
        if assigned_to:
            self.require_member(organization_id, assigned_to)
        normalized = body.strip()
        if not 1 <= len(normalized) <= 5000:
            raise ValueError("comment must be between 1 and 5000 characters")
        now = int(time.time())
        digest = hashlib.sha256(
            f"{organization_id}:{job_id}:{actor_id}:{request_id}".encode()
        ).hexdigest()[:24]
        comment = ProjectComment(
            id=f"com_{digest}",
            organization_id=organization_id,
            job_id=job_id,
            author_id=actor_id,
            author_name=actor_name,
            body=normalized,
            entity_ref=entity_ref[:300],
            assigned_to=assigned_to,
            created_at=now,
            updated_at=now,
        )
        return self.store.add_comment(comment)

    def set_comment_status(
        self,
        *,
        organization_id: str,
        job_id: str,
        comment_id: str,
        actor_id: str,
        status: Literal["open", "resolved"],
    ) -> ProjectComment:
        self.require_member(organization_id, actor_id, "commenter")
        comment = self.store.get_comment(job_id, comment_id)
        if comment.organization_id != organization_id:
            raise WorkspaceAccessError("comment is not in this workspace")
        comment.status = status
        comment.updated_at = int(time.time())
        self.store.save_comment(comment)
        return comment

    def heartbeat(
        self,
        *,
        organization_id: str,
        job_id: str,
        user_id: str,
        display_name: str,
        active_entity: str,
    ) -> WorkspacePresence:
        self.require_member(organization_id, user_id)
        now = int(time.time())
        colors = ["#39634c", "#7a5c35", "#5a596f", "#50666d", "#684f58"]
        color = colors[int(hashlib.sha256(user_id.encode()).hexdigest()[:4], 16) % len(colors)]
        presence = WorkspacePresence(
            organization_id=organization_id,
            job_id=job_id,
            user_id=user_id,
            display_name=display_name,
            color=color,
            active_entity=active_entity[:300],
            updated_at=now,
            expires_at=now + 75,
        )
        self.store.upsert_presence(presence)
        return presence
