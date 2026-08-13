import {
  Activity,
  Building2,
  Check,
  Clipboard,
  Clock3,
  Crown,
  History,
  Link2,
  MessageSquare,
  Plus,
  RefreshCw,
  RotateCcw,
  Send,
  ShieldCheck,
  UserMinus,
  Users,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { setActiveOrganizationId } from "../auth";
import type { PlanGraph } from "../types";
import {
  acceptWorkspaceInvitation,
  changeWorkspaceMemberRole,
  createOrganization,
  createProjectComment,
  inviteWorkspaceMember,
  joinOrganizationByDomain,
  loadProjectCollaboration,
  loadMoreWorkspaceMembers,
  loadWorkspace,
  removeWorkspaceMember,
  revokeWorkspaceInvitation,
  restoreModelVersion,
  setProjectCommentStatus,
  transferWorkspaceOwnership,
  type ActivityEvent,
  type ModelVersionRecord,
  type ProjectComment,
  type WorkspaceContext,
  type WorkspaceRole,
} from "../workspace";
import { authFetch } from "../auth";
import { studioApiUrl } from "../serverApi";

type CollaborationTab = "people" | "comments" | "versions" | "activity";

interface CollaborationDialogProps {
  jobId: string;
  jobVersion: number;
  selectedEntity: string;
  onClose: () => void;
  onWorkspaceChanged: (organizationName: string) => void;
  onEntityRequested: (entityReference: string) => void;
  onVersionRestored: (
    graph: PlanGraph,
    revision: { version: number; graphSha256: string },
  ) => void;
}

const editableRoles: Array<Exclude<WorkspaceRole, "owner">> = [
  "admin",
  "editor",
  "commenter",
  "viewer",
];

function initials(value: string) {
  return value
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function relativeTime(timestamp: number) {
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - timestamp));
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function expiresIn(timestamp: number) {
  const seconds = Math.max(0, Math.round(timestamp - Date.now() / 1000));
  if (seconds < 3600) return "in less than 1h";
  if (seconds < 86400) return `in ${Math.ceil(seconds / 3600)}h`;
  return `in ${Math.ceil(seconds / 86400)}d`;
}

function formatVersionSummary(summary: Record<string, number>) {
  const count = Object.values(summary).reduce((total, value) => total + value, 0);
  return count ? `${count} reviewed change${count === 1 ? "" : "s"}` : "Full model snapshot";
}

export function CollaborationDialog({
  jobId,
  jobVersion,
  selectedEntity,
  onClose,
  onWorkspaceChanged,
  onEntityRequested,
  onVersionRestored,
}: CollaborationDialogProps) {
  const [context, setContext] = useState<WorkspaceContext | null>(null);
  const [tab, setTab] = useState<CollaborationTab>("people");
  const [comments, setComments] = useState<ProjectComment[]>([]);
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [versions, setVersions] = useState<ModelVersionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [teamName, setTeamName] = useState("");
  const [teamDomain, setTeamDomain] = useState("");
  const [domainJoin, setDomainJoin] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<Exclude<WorkspaceRole, "owner">>("editor");
  const [inviteLink, setInviteLink] = useState("");
  const [commentBody, setCommentBody] = useState("");
  const [commentAssignee, setCommentAssignee] = useState("");
  const [restoreCandidate, setRestoreCandidate] = useState("");
  const [ownershipCandidate, setOwnershipCandidate] = useState("");
  const inviteToken = useMemo(() => new URLSearchParams(window.location.search).get("invite") ?? "", []);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError("");
    try {
      const workspace = await loadWorkspace(signal);
      setContext(workspace);
      if (jobId && workspace.active_organization_id) {
        const project = await loadProjectCollaboration(jobId, signal);
        setComments(project.comments);
        setActivity(project.activity);
        setVersions(project.versions);
      } else {
        setComments([]);
        setActivity([]);
        setVersions([]);
      }
    } catch (caught) {
      if ((caught as Error).name !== "AbortError") {
        setError(caught instanceof Error ? caught.message : "Could not load collaboration");
      }
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (
        document.visibilityState !== "visible"
        || !jobId
        || !context?.active_organization_id
      ) return;
      void loadProjectCollaboration(jobId).then((project) => {
        setComments(project.comments);
        setActivity(project.activity);
        setVersions(project.versions);
      }).catch(() => undefined);
    }, 15_000);
    return () => window.clearInterval(timer);
  }, [context?.active_organization_id, jobId]);

  const activePair = context?.organizations.find(
    (item) => item.organization.id === context.active_organization_id,
  );
  const canManage = activePair?.membership.role === "owner" || activePair?.membership.role === "admin";
  const canComment = Boolean(
    activePair && ["owner", "admin", "editor", "commenter"].includes(activePair.membership.role),
  );
  const canEdit = Boolean(
    activePair && ["owner", "admin", "editor"].includes(activePair.membership.role),
  );

  const run = async (action: () => Promise<void>) => {
    setWorking(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The action could not be completed");
    } finally {
      setWorking(false);
    }
  };

  const createTeam = () => run(async () => {
    const organization = await createOrganization(teamName, teamDomain, domainJoin);
    setNotice(`${organization.name} is ready`);
    onWorkspaceChanged(organization.name);
    await refresh();
  });

  const acceptInvite = () => run(async () => {
    const membership = await acceptWorkspaceInvitation(inviteToken);
    const organization = (await loadWorkspace()).organizations.find(
      (item) => item.organization.id === membership.organization_id,
    )?.organization;
    sessionStorage.removeItem("dajoong-pending-invite-v1");
    window.history.replaceState({}, "", "/studio");
    setNotice(`Joined ${organization?.name ?? "the workspace"}`);
    onWorkspaceChanged(organization?.name ?? "the workspace");
    await refresh();
  });

  const joinDomain = () => run(async () => {
    const membership = await joinOrganizationByDomain();
    const workspace = await loadWorkspace();
    const organization = workspace.organizations.find(
      (item) => item.organization.id === membership.organization_id,
    )?.organization;
    setNotice(`Joined ${organization?.name ?? "the workspace"}`);
    onWorkspaceChanged(organization?.name ?? "the workspace");
    await refresh();
  });

  const switchOrganization = (organizationId: string) => {
    const organization = context?.organizations.find(
      (item) => item.organization.id === organizationId,
    )?.organization;
    setActiveOrganizationId(organizationId);
    onWorkspaceChanged(organization?.name ?? "another workspace");
    setNotice(`Switched to ${organization?.name ?? "workspace"}`);
    void refresh();
  };

  const invite = () => run(async () => {
    const result = await inviteWorkspaceMember(inviteEmail, inviteRole);
    setInviteLink(result.accept_url);
    setInviteEmail("");
    setNotice(result.email_delivered ? "Invitation emailed" : "Invitation link is ready to share");
    await refresh();
  });

  const transferOwnership = (userId: string) => run(async () => {
    if (ownershipCandidate !== userId) {
      setOwnershipCandidate(userId);
      setNotice("Click Transfer again to confirm company ownership transfer");
      return;
    }
    await transferWorkspaceOwnership(userId);
    setOwnershipCandidate("");
    setNotice("Workspace ownership transferred");
    await refresh();
  });

  const loadMoreMembers = () => run(async () => {
    if (!context?.member_cursor) return;
    const next = await loadMoreWorkspaceMembers(context.member_cursor);
    setContext((current) => current ? {
      ...current,
      members: [
        ...current.members,
        ...next.members.filter(
          (member) => !current.members.some((item) => item.user_id === member.user_id),
        ),
      ],
      member_cursor: next.member_cursor,
    } : current);
  });

  const addComment = () => run(async () => {
    const result = await createProjectComment(
      jobId,
      commentBody,
      selectedEntity,
      commentAssignee,
    );
    setComments((current) => [result.comment, ...current]);
    setCommentBody("");
    setCommentAssignee("");
    setNotice(selectedEntity ? "Comment linked to the selected element" : "Comment added");
  });

  const restoreVersion = (version: ModelVersionRecord) => run(async () => {
    if (restoreCandidate !== version.graph_sha256) {
      setRestoreCandidate(version.graph_sha256);
      return;
    }
    const restored = await restoreModelVersion(
      jobId,
      version.graph_sha256,
      jobVersion,
    );
    const graphResponse = await authFetch(studioApiUrl(`/api/jobs/${jobId}/artifacts/corrected-graph?delivery=lazy`));
    if (!graphResponse.ok) throw new Error("The restored model could not be loaded");
    const graph = await graphResponse.json() as PlanGraph;
    onVersionRestored(graph, {
      version: restored.job_version,
      graphSha256: restored.graph_sha256,
    });
    setRestoreCandidate("");
    setNotice(`Restored version ${version.version}`);
    await refresh();
  });

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="collaboration-dialog" role="dialog" aria-modal="true" aria-labelledby="collaboration-title">
        <header className="collaboration-header">
          <div>
            <span className="eyebrow">COMPANY WORKSPACE</span>
            <h2 id="collaboration-title">People, versions and decisions</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close collaboration"><X size={18} /></button>
        </header>

        {inviteToken ? (
          <div className="invite-accept-banner">
            <div><Link2 size={17} /><span><strong>You have a team invitation</strong><small>Accept it with the email address that received the link.</small></span></div>
            <button type="button" disabled={working} onClick={acceptInvite}>Accept invitation</button>
          </div>
        ) : null}

        {loading && !context ? <div className="collaboration-loading"><RefreshCw className="spin" size={18} /> Loading workspace</div> : null}

        {!loading && context && !context.organizations.length ? (
          <div className="workspace-onboarding">
            <div className="workspace-onboarding-copy">
              <Building2 size={28} />
              <span className="eyebrow">ONE PURCHASE · ONE TEAM</span>
              <h3>Create your company workspace</h3>
              <p>Projects, conversion credits and model history belong to the company. Teammates sign in with their own accounts.</p>
            </div>
            <div className="workspace-create-form">
              <label>Company or team name<input value={teamName} onChange={(event) => setTeamName(event.target.value)} placeholder="Northstar Construction" autoFocus /></label>
              <label>Work email domain <span>optional</span><input value={teamDomain} onChange={(event) => setTeamDomain(event.target.value)} placeholder="northstar.com" /></label>
              <label className="workspace-checkbox"><input type="checkbox" checked={domainJoin} disabled={!teamDomain} onChange={(event) => setDomainJoin(event.target.checked)} /><span>Let verified users from this domain join without an invitation</span></label>
              <button type="button" className="primary-action" disabled={working || teamName.trim().length < 2} onClick={createTeam}><Plus size={16} /> Create workspace</button>
              <button type="button" className="quiet-action" disabled={working} onClick={joinDomain}>Join with my work email</button>
            </div>
          </div>
        ) : null}

        {context?.organizations.length ? (
          <>
            <div className="workspace-switch-row">
              <div className="workspace-switcher">
                <Building2 size={16} />
                <select value={context.active_organization_id} onChange={(event) => switchOrganization(event.target.value)} aria-label="Active company workspace">
                  {context.organizations.map(({ organization }) => <option key={organization.id} value={organization.id}>{organization.name}</option>)}
                </select>
              </div>
              <div className="workspace-role"><ShieldCheck size={14} /> {activePair?.membership.role}</div>
            </div>

            <nav className="collaboration-tabs" aria-label="Collaboration sections">
              <button className={tab === "people" ? "active" : ""} onClick={() => setTab("people")}><Users size={15} /> People <span>{context.members.length}{context.member_cursor ? "+" : ""}</span></button>
              <button className={tab === "comments" ? "active" : ""} disabled={!jobId} onClick={() => setTab("comments")}><MessageSquare size={15} /> Comments <span>{comments.filter((item) => item.status === "open").length}</span></button>
              <button className={tab === "versions" ? "active" : ""} disabled={!jobId} onClick={() => setTab("versions")}><History size={15} /> Versions <span>{versions.length}</span></button>
              <button className={tab === "activity" ? "active" : ""} disabled={!jobId} onClick={() => setTab("activity")}><Activity size={15} /> Activity</button>
            </nav>

            <div className="collaboration-content">
              {tab === "people" ? (
                <div className="people-pane">
                  {canManage ? (
                    <div className="invite-row">
                      <input type="email" value={inviteEmail} onChange={(event) => setInviteEmail(event.target.value)} placeholder="teammate@company.com" aria-label="Teammate email" />
                      <select value={inviteRole} onChange={(event) => setInviteRole(event.target.value as Exclude<WorkspaceRole, "owner">)} aria-label="Invitation role">
                        {editableRoles.map((role) => <option key={role} value={role}>{role}</option>)}
                      </select>
                      <button type="button" disabled={working || !inviteEmail.includes("@")} onClick={invite}><Send size={15} /> Invite</button>
                    </div>
                  ) : null}
                  {inviteLink ? (
                    <div className="invite-link-row"><span><Check size={14} /> Share this secure link</span><code>{inviteLink}</code><button type="button" onClick={() => void navigator.clipboard.writeText(inviteLink)}><Clipboard size={14} /> Copy</button></div>
                  ) : null}
                  <div className="member-list" role="list">
                    {context.members.map((member) => (
                      <div className="member-row" role="listitem" key={member.user_id}>
                        <span className="member-avatar">{initials(member.display_name || member.email)}</span>
                        <span className="member-identity"><strong>{member.display_name || member.email}</strong><small>{member.email}</small></span>
                        {canManage && member.role !== "owner" ? (
                          <select value={member.role} onChange={(event) => void run(async () => { await changeWorkspaceMemberRole(member.user_id, event.target.value as Exclude<WorkspaceRole, "owner">); await refresh(); })} aria-label={`Role for ${member.display_name}`}>
                            {editableRoles.map((role) => <option key={role} value={role}>{role}</option>)}
                          </select>
                        ) : <span className="role-badge">{member.role}</span>}
                        <span className="member-actions">
                          {activePair?.membership.role === "owner" && member.role === "admin" ? <button type="button" className={ownershipCandidate === member.user_id ? "transfer-owner confirm" : "transfer-owner"} onClick={() => void transferOwnership(member.user_id)} aria-label={`Transfer ownership to ${member.display_name}`} title="Transfer company ownership"><Crown size={14} /> {ownershipCandidate === member.user_id ? "Transfer" : "Owner"}</button> : null}
                          {canManage && member.role !== "owner" ? <button type="button" className="remove-member" onClick={() => void run(async () => { await removeWorkspaceMember(member.user_id); await refresh(); })} aria-label={`Remove ${member.display_name}`}><UserMinus size={15} /></button> : null}
                        </span>
                      </div>
                    ))}
                  </div>
                  {context.member_cursor ? <button type="button" className="load-more-members" disabled={working} onClick={loadMoreMembers}><Users size={14} /> Load more people</button> : null}
                  {context.invitations.length ? <div className="pending-invites"><strong>Pending invitations</strong>{context.invitations.map((invitation) => <span key={invitation.id}><span>{invitation.email}<small>{invitation.role} · expires {expiresIn(invitation.expires_at)}</small></span>{canManage ? <button type="button" onClick={() => void run(async () => { await revokeWorkspaceInvitation(invitation.id); await refresh(); })}>Revoke</button> : null}</span>)}</div> : null}
                </div>
              ) : null}

              {tab === "comments" ? (
                <div className="comments-pane">
                  {canComment ? (
                    <div className="comment-composer">
                      <textarea value={commentBody} onChange={(event) => setCommentBody(event.target.value)} placeholder={selectedEntity ? `Comment on ${selectedEntity}` : "Leave a project comment"} />
                      <div><select value={commentAssignee} onChange={(event) => setCommentAssignee(event.target.value)} aria-label="Assign comment"><option value="">No assignee</option>{context.members.map((member) => <option key={member.user_id} value={member.user_id}>Assign to {member.display_name}</option>)}</select><button type="button" disabled={working || !commentBody.trim()} onClick={addComment}><Send size={14} /> Comment</button></div>
                    </div>
                  ) : null}
                  <div className="comment-list">
                    {comments.map((comment) => <article key={comment.id} className={comment.status === "resolved" ? "resolved" : ""}><div className="comment-meta"><span className="member-avatar small">{initials(comment.author_name)}</span><strong>{comment.author_name}</strong><time>{relativeTime(comment.created_at)}</time>{comment.entity_ref ? <button type="button" className="entity-reference" onClick={() => onEntityRequested(comment.entity_ref)}><Link2 size={12} /> {comment.entity_ref}</button> : null}</div><p>{comment.body}</p><footer>{comment.assigned_to ? <span>Assigned to {context.members.find((member) => member.user_id === comment.assigned_to)?.display_name ?? "team member"}</span> : <span>Project note</span>}{canComment ? <button type="button" onClick={() => void run(async () => { const next = comment.status === "open" ? "resolved" : "open"; const result = await setProjectCommentStatus(jobId, comment.id, next); setComments((items) => items.map((item) => item.id === comment.id ? result.comment : item)); })}>{comment.status === "open" ? <><Check size={13} /> Resolve</> : "Reopen"}</button> : null}</footer></article>)}
                    {!comments.length ? <div className="empty-collaboration"><MessageSquare size={22} /><strong>No comments yet</strong><span>Start a decision thread without leaving the model.</span></div> : null}
                  </div>
                </div>
              ) : null}

              {tab === "versions" ? (
                <div className="version-list">
                  {versions.map((version, index) => <article key={`${version.version}:${version.graph_sha256}`}><div className="version-rail"><span>{version.version}</span>{index < versions.length - 1 ? <i /> : null}</div><div className="version-copy"><header><strong>{version.label}</strong>{index === 0 ? <span className="current-version">Current</span> : null}</header><p>{formatVersionSummary(version.summary)}</p><small>{version.created_by_name} · {relativeTime(version.created_at)} · {version.release_allowed ? "Release ready" : "Review required"}</small></div>{canEdit && index > 0 ? <button type="button" className={restoreCandidate === version.graph_sha256 ? "confirm" : ""} disabled={working} onClick={() => void restoreVersion(version)}>{restoreCandidate === version.graph_sha256 ? <><Check size={14} /> Confirm</> : <><RotateCcw size={14} /> Restore</>}</button> : null}</article>)}
                  {!versions.length ? <div className="empty-collaboration"><History size={22} /><strong>No cloud versions yet</strong><span>Versions appear after the first company model is saved.</span></div> : null}
                </div>
              ) : null}

              {tab === "activity" ? (
                <div className="activity-list">
                  {activity.map((event) => <article key={event.id}><span className="activity-icon">{event.kind.includes("comment") ? <MessageSquare size={14} /> : <Clock3 size={14} />}</span><div><strong>{event.actor_name}</strong><p>{event.summary}</p><small>{relativeTime(event.created_at)}{event.entity_ref ? ` · ${event.entity_ref}` : ""}</small></div></article>)}
                  {!activity.length ? <div className="empty-collaboration"><Activity size={22} /><strong>No activity yet</strong><span>Model saves and team decisions appear here.</span></div> : null}
                </div>
              ) : null}
            </div>
          </>
        ) : null}

        {error ? <div className="collaboration-message error" role="alert">{error}</div> : null}
        {notice ? <div className="collaboration-message" role="status">{notice}</div> : null}
      </section>
    </div>
  );
}
