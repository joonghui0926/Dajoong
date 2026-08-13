import type { PlanGraph } from "./types";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
export const BUNDLED_STUDIO_SAMPLE_SOURCE_SHA256 = "2c9092e12dd22207b1ab41d7660534d73f4b341121d279685307ba20597da5d6";

export function graphSourceHashes(graph: PlanGraph): Set<string> {
  const sources = Array.isArray(graph.sources) ? graph.sources : [];
  const provenance = graph.provenance as Record<string, unknown> | undefined;
  return new Set([
    ...sources.flatMap((source) => [source.source_hash, source.sha256]),
    provenance?.source_image_sha256,
    provenance?.input_sha256,
  ].filter((value): value is string => typeof value === "string" && SHA256_PATTERN.test(value.toLowerCase()))
    .map((value) => value.toLowerCase()));
}

export function isBundledStudioSampleGraph(graph: PlanGraph) {
  return graphSourceHashes(graph).has(BUNDLED_STUDIO_SAMPLE_SOURCE_SHA256);
}

interface HashedArtifact {
  path: string;
  sha256: string;
}

export interface StudioSampleManifest {
  schema_version: "dajoong.studio-sample.v1";
  sample_id: string;
  revision: string;
  content_contract: {
    profile: "full_editable_bim";
    whole_sheet_reviewed: true;
    omission_scan_completed: true;
    counts: {
      walls: number;
      openings: number;
      rooms: number;
      fixtures: number;
    };
  };
  reviewed_source: {
    sha256: string;
    width_px: number;
    height_px: number;
  };
  display_source: HashedArtifact & {
    derived_from_reviewed_source: boolean;
  };
  graph: HashedArtifact & {
    source_image_sha256: string;
  };
}

export interface VerifiedStudioSample {
  graph: PlanGraph;
  sourceUrl: string;
  sampleId: string;
}

function requireSha256(value: unknown, field: string): asserts value is string {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value.toLowerCase())) {
    throw new Error(`sample contract has an invalid ${field}`);
  }
}

export function verifyStudioSampleContract(
  manifest: StudioSampleManifest,
  graph: PlanGraph,
  observed: { graphSha256: string; displaySourceSha256: string },
) {
  if (manifest.schema_version !== "dajoong.studio-sample.v1") {
    throw new Error("sample contract version is not supported");
  }
  requireSha256(manifest.reviewed_source.sha256, "reviewed source hash");
  requireSha256(manifest.display_source.sha256, "display source hash");
  requireSha256(manifest.graph.sha256, "graph hash");
  requireSha256(manifest.graph.source_image_sha256, "graph source hash");

  const reviewedSourceSha = manifest.reviewed_source.sha256.toLowerCase();
  const graphSourceSha = manifest.graph.source_image_sha256.toLowerCase();
  const provenance = graph.provenance as Record<string, unknown> | undefined;
  const embeddedSourceSha = String(provenance?.source_image_sha256 ?? "").toLowerCase();
  const pipeline = graph.pipeline as Record<string, unknown> | undefined;

  if (graphSourceSha !== reviewedSourceSha || embeddedSourceSha !== reviewedSourceSha) {
    throw new Error("sample graph belongs to a different reviewed drawing");
  }
  if (observed.graphSha256.toLowerCase() !== manifest.graph.sha256.toLowerCase()) {
    throw new Error("sample graph changed without a reviewed contract update");
  }
  if (observed.displaySourceSha256.toLowerCase() !== manifest.display_source.sha256.toLowerCase()) {
    throw new Error("sample preview changed without a reviewed contract update");
  }
  if (!manifest.display_source.derived_from_reviewed_source) {
    throw new Error("sample preview is not declared as a derivative of the reviewed source");
  }
  if (!manifest.revision || manifest.revision !== manifest.graph.sha256.slice(0, 16)) {
    throw new Error("sample revision does not identify the exact graph bytes");
  }
  if (manifest.content_contract.profile !== "full_editable_bim") {
    throw new Error("sample is not a full editable BIM review packet");
  }
  if (
    manifest.content_contract.whole_sheet_reviewed !== true
    || manifest.content_contract.omission_scan_completed !== true
    || pipeline?.whole_sheet_reviewed !== true
    || pipeline?.omission_scan_completed !== true
  ) {
    throw new Error("sample has not completed whole-sheet omission review");
  }
  const observedCounts = {
    walls: graph.walls.length,
    openings: graph.openings.length,
    rooms: graph.rooms.length,
    fixtures: graph.fixtures.length,
  };
  if (Object.entries(manifest.content_contract.counts).some(
    ([key, expected]) => observedCounts[key as keyof typeof observedCounts] !== expected,
  )) {
    throw new Error("sample entity counts do not match the reviewed content contract");
  }
}

async function sha256Hex(data: BufferSource) {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

export async function loadVerifiedStudioSample(signal?: AbortSignal): Promise<VerifiedStudioSample> {
  const manifestResponse = await fetch("/sample/sample-manifest.json", { signal, cache: "no-cache" });
  if (!manifestResponse.ok) throw new Error(`sample manifest returned ${manifestResponse.status}`);
  const manifest = await manifestResponse.json() as StudioSampleManifest;

  const revision = encodeURIComponent(manifest.revision);
  const [graphResponse, displaySourceResponse] = await Promise.all([
    fetch(`${manifest.graph.path}?revision=${revision}`, { signal, cache: "force-cache" }),
    fetch(`${manifest.display_source.path}?revision=${revision}`, { signal, cache: "force-cache" }),
  ]);
  if (!graphResponse.ok) throw new Error(`sample graph returned ${graphResponse.status}`);
  if (!displaySourceResponse.ok) throw new Error(`sample source returned ${displaySourceResponse.status}`);

  const [graphText, displaySourceBytes] = await Promise.all([
    graphResponse.text(),
    displaySourceResponse.arrayBuffer(),
  ]);
  const graph = JSON.parse(graphText) as PlanGraph;
  const [graphSha256, displaySourceSha256] = await Promise.all([
    sha256Hex(new TextEncoder().encode(graphText)),
    sha256Hex(displaySourceBytes),
  ]);
  verifyStudioSampleContract(manifest, graph, { graphSha256, displaySourceSha256 });
  return {
    graph,
    sourceUrl: `${manifest.display_source.path}?revision=${revision}`,
    sampleId: manifest.sample_id,
  };
}
