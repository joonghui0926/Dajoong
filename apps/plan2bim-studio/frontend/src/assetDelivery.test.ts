import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearFamilyAssetCacheForTests,
  loadVisibleFamilyAssets,
  parseFamilyAssetMesh,
} from "./assetDelivery";
import type { PlanGraph } from "./types";

function meshPayload() {
  const vertices = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
  ];
  const faces = [
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
  ];
  const buffer = new ArrayBuffer(16 + vertices.length * 12 + faces.length * 12 + faces.length * 3);
  const view = new DataView(buffer);
  "DJMSH001".split("").forEach((value, index) => view.setUint8(index, value.charCodeAt(0)));
  view.setUint32(8, vertices.length, true);
  view.setUint32(12, faces.length, true);
  let offset = 16;
  vertices.flat().forEach((value) => { view.setFloat32(offset, value, true); offset += 4; });
  faces.flat().forEach((value) => { view.setUint32(offset, value, true); offset += 4; });
  faces.forEach(() => {
    view.setUint8(offset, 120); view.setUint8(offset + 1, 100); view.setUint8(offset + 2, 80);
    offset += 3;
  });
  return buffer;
}

function graph(reference: string): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }, { id: "L2", name: "Level 2" }],
    walls: [], rooms: [], openings: [], routes: [],
    fixtures: [
      { id: "a", level_id: "L1", type: "chair", center_m: [0, 0], size_m: [1, 1, 1], geometry_ref: reference },
      { id: "b", level_id: "L1", type: "chair", center_m: [2, 0], size_m: [1, 1, 1], geometry_ref: reference },
      { id: "c", level_id: "L2", type: "chair", center_m: [0, 0], size_m: [1, 1, 1], geometry_ref: "mesh:" + "b".repeat(64) },
    ],
    asset_delivery: {
      schema_version: "dajoong.asset-delivery.v1",
      catalog_url: "/api/assets/v1/catalog",
      mesh_url_template: "/api/assets/v1/{mesh_sha256}.mesh",
      format: "dajoong.mesh.v1",
      content_addressed: true,
      lazy_by_visible_level: true,
    },
  };
}

describe("lazy family asset delivery", () => {
  beforeEach(() => clearFamilyAssetCacheForTests());

  it("parses the compact binary mesh contract", () => {
    const reference = "mesh:" + "a".repeat(64);
    const parsed = parseFamilyAssetMesh(meshPayload(), reference);
    expect(parsed.mesh_vertices).toHaveLength(4);
    expect(parsed.mesh_faces).toHaveLength(4);
    expect(parsed.mesh_face_colors).toHaveLength(4);
    expect(parsed.asset_mesh_sha256).toBe("a".repeat(64));
  });

  it("fetches one deduplicated asset for only the visible level", async () => {
    const reference = "mesh:" + "a".repeat(64);
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => (
      new Response(meshPayload(), { status: 200 })
    ));
    const fetcher = fetchMock as unknown as typeof fetch;
    const result = await loadVisibleFamilyAssets(graph(reference), "L1", fetcher);

    expect(Object.keys(result)).toEqual([reference]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toContain("a".repeat(64));

    await loadVisibleFamilyAssets(graph(reference), "L1", fetcher);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("fails closed and leaves a semantic marker when a fetch fails", async () => {
    const reference = "mesh:" + "a".repeat(64);
    const fetcher = vi.fn(async () => new Response(null, { status: 404 })) as typeof fetch;
    const result = await loadVisibleFamilyAssets(graph(reference), "L1", fetcher);

    expect(result).toEqual({});
  });
});
