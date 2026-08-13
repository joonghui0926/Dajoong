import { isTrustedStudioApiRequest, studioApiUrl } from "./serverApi";
import type { FamilyAssetDefinition, PlanGraph } from "./types";

const MAGIC = "DJMSH001";
const MESH_REF = /^mesh:([a-f0-9]{64})$/;
const MAX_MEMORY_ASSETS = 64;
const assetCache = new Map<string, Promise<FamilyAssetDefinition>>();

type Fetcher = typeof fetch;

function assetUrl(template: string, meshHash: string): string {
  const path = template.replace("{mesh_sha256}", meshHash);
  if (path.startsWith("/api/")) return studioApiUrl(path);
  if (!isTrustedStudioApiRequest(path)) throw new Error("untrusted family asset origin");
  return path;
}

function trimCache() {
  while (assetCache.size > MAX_MEMORY_ASSETS) {
    const oldest = assetCache.keys().next().value;
    if (oldest) assetCache.delete(oldest);
    else break;
  }
}

export function parseFamilyAssetMesh(
  payload: ArrayBuffer,
  geometryRef: string,
  geometryStatus = "licensed_api_asset",
): FamilyAssetDefinition {
  const match = MESH_REF.exec(geometryRef);
  if (!match) throw new Error("invalid content-addressed geometry reference");
  const view = new DataView(payload);
  if (view.byteLength < 16) throw new Error("family asset payload is truncated");
  const magic = String.fromCharCode(...new Uint8Array(payload, 0, 8));
  if (magic !== MAGIC) throw new Error("unsupported family asset payload");
  const vertexCount = view.getUint32(8, true);
  const faceCount = view.getUint32(12, true);
  if (vertexCount < 4 || faceCount < 4) throw new Error("family asset mesh is degenerate");
  const expectedBytes = 16 + vertexCount * 3 * 4 + faceCount * 3 * 4 + faceCount * 3;
  if (expectedBytes !== view.byteLength) throw new Error("family asset payload length mismatch");
  let offset = 16;
  const vertices: [number, number, number][] = [];
  for (let index = 0; index < vertexCount; index += 1) {
    vertices.push([
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
    ]);
    offset += 12;
  }
  const faces: [number, number, number][] = [];
  for (let index = 0; index < faceCount; index += 1) {
    const face: [number, number, number] = [
      view.getUint32(offset, true),
      view.getUint32(offset + 4, true),
      view.getUint32(offset + 8, true),
    ];
    if (face.some((vertex) => vertex >= vertexCount)) throw new Error("family asset face is invalid");
    faces.push(face);
    offset += 12;
  }
  const colors: [number, number, number][] = [];
  for (let index = 0; index < faceCount; index += 1) {
    colors.push([
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
    ]);
    offset += 3;
  }
  return {
    schema_version: "dajoong.family-asset.v1",
    geometry_status: geometryStatus,
    asset_mesh_sha256: match[1],
    normalized_to_unit_envelope: true,
    mesh_vertices: vertices,
    mesh_faces: faces,
    mesh_face_colors: colors,
  };
}

export async function loadFamilyAsset(
  graph: PlanGraph,
  geometryRef: string,
  geometryStatus: string,
  fetcher: Fetcher = fetch,
): Promise<FamilyAssetDefinition> {
  const inline = graph.family_assets?.[geometryRef];
  if (inline) return inline;
  const match = MESH_REF.exec(geometryRef);
  if (!match || graph.asset_delivery?.format !== "dajoong.mesh.v1") {
    throw new Error("family asset is not available from the server library");
  }
  const existing = assetCache.get(geometryRef);
  if (existing) {
    assetCache.delete(geometryRef);
    assetCache.set(geometryRef, existing);
    return existing;
  }
  const pending = fetcher(assetUrl(graph.asset_delivery.mesh_url_template, match[1]), {
    headers: { Accept: "application/vnd.dajoong.mesh" },
  }).then(async (response) => {
    if (!response.ok) throw new Error(`family asset request failed (${response.status})`);
    return parseFamilyAssetMesh(await response.arrayBuffer(), geometryRef, geometryStatus);
  }).catch((error) => {
    assetCache.delete(geometryRef);
    throw error;
  });
  assetCache.set(geometryRef, pending);
  trimCache();
  return pending;
}

export async function loadVisibleFamilyAssets(
  graph: PlanGraph,
  levelId: string,
  fetcher: Fetcher = fetch,
): Promise<Record<string, FamilyAssetDefinition>> {
  const requests = new Map<string, string>();
  for (const fixture of graph.fixtures) {
    if (fixture.level_id !== levelId || !fixture.geometry_ref) continue;
    if (graph.family_assets?.[fixture.geometry_ref]) continue;
    requests.set(fixture.geometry_ref, fixture.geometry_status ?? "licensed_api_asset");
  }
  const queue = [...requests.entries()];
  const result: Record<string, FamilyAssetDefinition> = {};
  const worker = async () => {
    while (queue.length) {
      const next = queue.shift();
      if (!next) return;
      const [reference, status] = next;
      try {
        result[reference] = await loadFamilyAsset(graph, reference, status, fetcher);
      } catch {
        // The viewport keeps a visible semantic marker when one remote asset fails.
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(4, queue.length) }, worker));
  return result;
}

export function clearFamilyAssetCacheForTests() {
  assetCache.clear();
}

