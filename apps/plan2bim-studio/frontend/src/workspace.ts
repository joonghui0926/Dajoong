import { authFetch, getActiveOrganizationId, setActiveOrganizationId } from "./auth";
import { studioApiUrl } from "./serverApi";

export type WorkspaceRole = "owner" | "admin" | "editor" | "commenter" | "viewer";

export interface Organization {
  id: string;
  name: string;
  slug: string;
  approved_domains: string[];
  domain_join_enabled: boolean;
  created_at: number;
}

export interface WorkspaceMember {
  organization_id: string;
  user_id: string;
  email: string;
  display_name: string;
  role: WorkspaceRole;
  status: "active" | "invited" | "suspended";
  joined_at: number;
  updated_at: number;
}

export interface WorkspaceInvitation {
  id: string;
  organization_id: string;
  email: string;
  role: WorkspaceRole;
  status: "pending" | "accepted" | "revoked" | "expired";
  created_at: number;
  expires_at: number;
}

export interface WorkspaceContext {
  organizations: Array<{ organization: Organization; membership: WorkspaceMember }>;
  active_organization_id: string;
  members: WorkspaceMember[];
  member_cursor: string;
  invitations: WorkspaceInvitation[];
}

export interface ProjectComment {
  id: string;
  author_id: string;
  author_name: string;
  body: string;
  entity_ref: string;
  assigned_to: string;
  status: "open" | "resolved";
  created_at: number;
  updated_at: number;
}

export interface ActivityEvent {
  id: string;
  actor_id: string;
  actor_name: string;
  kind: string;
  summary: string;
  entity_ref: string;
  created_at: number;
}

export interface ModelVersionRecord {
  job_id: string;
  version: number;
  graph_sha256: string;
  created_by: string;
  created_by_name: string;
  label: string;
  summary: Record<string, number>;
  release_allowed: boolean;
  created_at: number;
}

export interface WorkspacePresence {
  user_id: string;
  display_name: string;
  color: string;
  active_entity: string;
  updated_at: number;
}

async function checked<T>(response: Response, fallback: string): Promise<T> {
  if (response.ok) return response.json() as Promise<T>;
  const payload = await response.json().catch(() => ({})) as { detail?: string };
  throw new Error(typeof payload.detail === "string" ? payload.detail : fallback);
}

export async function loadWorkspace(signal?: AbortSignal): Promise<WorkspaceContext> {
  const payload = await checked<WorkspaceContext>(
    await authFetch(studioApiUrl("/api/workspace"), { signal }),
    "Could not load the company workspace",
  );
  const currentOrganizationId = getActiveOrganizationId();
  const currentIsAvailable = payload.organizations.some(
    ({ organization }) => organization.id === currentOrganizationId,
  );
  if ((!currentOrganizationId || !currentIsAvailable) && payload.active_organization_id) {
    setActiveOrganizationId(payload.active_organization_id);
  } else if (currentOrganizationId && !currentIsAvailable) {
    setActiveOrganizationId("");
  }
  return payload;
}

export async function loadMoreWorkspaceMembers(cursor: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ member_limit: "100", member_cursor: cursor });
  return checked<WorkspaceContext>(
    await authFetch(studioApiUrl(`/api/workspace?${query}`), { signal }),
    "Could not load more company members",
  );
}

export async function createOrganization(
  name: string,
  approvedDomain: string,
  domainJoinEnabled: boolean,
) {
  const result = await checked<{ organization: Organization }>(
    await authFetch(studioApiUrl("/api/workspace/organizations"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        approved_domains: approvedDomain ? [approvedDomain] : [],
        domain_join_enabled: domainJoinEnabled,
      }),
    }),
    "Could not create the workspace",
  );
  setActiveOrganizationId(result.organization.id);
  return result.organization;
}

export async function joinOrganizationByDomain() {
  const result = await checked<{ membership: WorkspaceMember }>(
    await authFetch(studioApiUrl("/api/workspace/join-domain"), { method: "POST" }),
    "Your email domain is not enabled for automatic joining",
  );
  setActiveOrganizationId(result.membership.organization_id);
  return result.membership;
}

export async function acceptWorkspaceInvitation(token: string) {
  const result = await checked<{ membership: WorkspaceMember }>(
    await authFetch(studioApiUrl("/api/workspace/invitations/accept"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }),
    "The invitation could not be accepted",
  );
  setActiveOrganizationId(result.membership.organization_id);
  return result.membership;
}

export async function inviteWorkspaceMember(email: string, role: Exclude<WorkspaceRole, "owner">) {
  return checked<{
    invitation: WorkspaceInvitation;
    accept_url: string;
    email_delivered: boolean;
  }>(
    await authFetch(studioApiUrl("/api/workspace/invitations"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, role }),
    }),
    "Could not invite this teammate",
  );
}

export async function revokeWorkspaceInvitation(invitationId: string) {
  return checked<{ invitation: WorkspaceInvitation }>(
    await authFetch(
      studioApiUrl(`/api/workspace/invitations/${encodeURIComponent(invitationId)}`),
      { method: "DELETE" },
    ),
    "Could not revoke this invitation",
  );
}

export async function changeWorkspaceMemberRole(
  userId: string,
  role: Exclude<WorkspaceRole, "owner">,
) {
  return checked<{ membership: WorkspaceMember }>(
    await authFetch(studioApiUrl(`/api/workspace/members/${encodeURIComponent(userId)}/role`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    }),
    "Could not update this role",
  );
}

export async function removeWorkspaceMember(userId: string) {
  await checked<{ status: string }>(
    await authFetch(studioApiUrl(`/api/workspace/members/${encodeURIComponent(userId)}`), {
      method: "DELETE",
    }),
    "Could not remove this teammate",
  );
}

export async function transferWorkspaceOwnership(userId: string) {
  return checked<{ previous_owner: WorkspaceMember; next_owner: WorkspaceMember }>(
    await authFetch(studioApiUrl(`/api/workspace/ownership/${encodeURIComponent(userId)}`), {
      method: "POST",
    }),
    "Could not transfer workspace ownership",
  );
}

export async function loadProjectCollaboration(jobId: string, signal?: AbortSignal) {
  const [comments, activity, versions] = await Promise.all([
    checked<{ items: ProjectComment[] }>(
      await authFetch(studioApiUrl(`/api/jobs/${jobId}/comments`), { signal }),
      "Could not load comments",
    ),
    checked<{ items: ActivityEvent[] }>(
      await authFetch(studioApiUrl(`/api/jobs/${jobId}/activity`), { signal }),
      "Could not load activity",
    ),
    checked<{ items: ModelVersionRecord[] }>(
      await authFetch(studioApiUrl(`/api/jobs/${jobId}/versions`), { signal }),
      "Could not load versions",
    ),
  ]);
  return { comments: comments.items, activity: activity.items, versions: versions.items };
}

export async function createProjectComment(
  jobId: string,
  body: string,
  entityRef = "",
  assignedTo = "",
) {
  return checked<{ comment: ProjectComment }>(
    await authFetch(studioApiUrl(`/api/jobs/${jobId}/comments`), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": crypto.randomUUID(),
      },
      body: JSON.stringify({ body, entity_ref: entityRef, assigned_to: assignedTo }),
    }),
    "Could not add this comment",
  );
}

export async function setProjectCommentStatus(
  jobId: string,
  commentId: string,
  status: "open" | "resolved",
) {
  return checked<{ comment: ProjectComment }>(
    await authFetch(studioApiUrl(`/api/jobs/${jobId}/comments/${commentId}/status`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
    "Could not update this comment",
  );
}

export async function restoreModelVersion(
  jobId: string,
  graphSha256: string,
  expectedJobVersion: number,
) {
  return checked<{ graph_sha256: string; job_version: number }>(
    await authFetch(studioApiUrl(`/api/jobs/${jobId}/versions/${graphSha256}/restore`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_job_version: expectedJobVersion }),
    }),
    "Could not restore this model version",
  );
}

export async function sendWorkspaceHeartbeat(jobId: string, activeEntity: string) {
  return checked<{ self: WorkspacePresence; active: WorkspacePresence[] }>(
    await authFetch(studioApiUrl("/api/workspace/presence"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, active_entity: activeEntity }),
    }),
    "Could not update collaboration presence",
  );
}
