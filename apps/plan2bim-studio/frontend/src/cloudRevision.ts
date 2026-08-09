import { authFetch } from "./auth";
import { studioApiUrl } from "./serverApi";
import type { CorrectionOperation, PlanGraph } from "./types";

export interface CloudRevisionPayload {
  expectedJobVersion: number;
  expectedGraphSha256: string;
  graph: PlanGraph;
  operations: CorrectionOperation[];
}

export async function saveCloudRevision(jobId: string, payload: CloudRevisionPayload): Promise<{
  conflict: boolean;
  graphSha256: string;
  jobVersion: number;
}> {
  const response = await authFetch(studioApiUrl(`/api/jobs/${jobId}/revisions`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      expected_job_version: payload.expectedJobVersion,
      expected_graph_sha256: payload.expectedGraphSha256,
      reviewer: "studio-user",
      operations: payload.operations,
      graph: payload.graph,
    }),
  });
  if (response.status === 409) return { conflict: true, graphSha256: "", jobVersion: 0 };
  if (!response.ok) throw new Error(`cloud save returned ${response.status}`);
  const saved = (await response.json()) as { graph_sha256: string; job_version: number };
  return {
    conflict: false,
    graphSha256: saved.graph_sha256,
    jobVersion: saved.job_version,
  };
}
