import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { verifyStudioSampleContract, type StudioSampleManifest } from "./sampleIntegrity";
import type { PlanGraph } from "./types";

const hash = (character: string) => character.repeat(64);
const manifest: StudioSampleManifest = {
  schema_version: "dajoong.studio-sample.v1",
  sample_id: "sample@source",
  revision: hash("c").slice(0, 16),
  content_contract: {
    profile: "full_editable_bim",
    whole_sheet_reviewed: true,
    omission_scan_completed: true,
    counts: { walls: 0, openings: 0, rooms: 0, fixtures: 0 },
  },
  reviewed_source: { sha256: hash("a"), width_px: 100, height_px: 80 },
  display_source: { path: "/source.webp", sha256: hash("b"), derived_from_reviewed_source: true },
  graph: { path: "/graph.json", sha256: hash("c"), source_image_sha256: hash("a") },
};
const graph = {
  schema_version: "buili.plan-graph.v2",
  levels: [], walls: [], rooms: [], openings: [], fixtures: [], routes: [],
  provenance: { source_image_sha256: hash("a") },
  pipeline: { whole_sheet_reviewed: true, omission_scan_completed: true },
} as PlanGraph;

describe("Studio sample integrity", () => {
  it("accepts a display derivative only when graph and source contracts agree", () => {
    expect(() => verifyStudioSampleContract(manifest, graph, {
      graphSha256: hash("c"),
      displaySourceSha256: hash("b"),
    })).not.toThrow();
  });

  it("rejects an ID collision where the graph came from another source", () => {
    const wrongGraph = { ...graph, provenance: { source_image_sha256: hash("d") } };
    expect(() => verifyStudioSampleContract(manifest, wrongGraph, {
      graphSha256: hash("c"),
      displaySourceSha256: hash("b"),
    })).toThrow(/different reviewed drawing/);
  });

  it("rejects a stale preview or graph", () => {
    expect(() => verifyStudioSampleContract(manifest, graph, {
      graphSha256: hash("e"),
      displaySourceSha256: hash("b"),
    })).toThrow(/graph changed/);
  });

  it("pins the bundled full-editable demo to its exact source and graph bytes", async () => {
    const publicRoot = new URL("../public/", import.meta.url);
    const bundledManifest = JSON.parse(
      await readFile(new URL("sample/sample-manifest.json", publicRoot), "utf8"),
    ) as StudioSampleManifest;
    const graphBytes = await readFile(new URL("sample/03-plan-graph.json", publicRoot));
    const sourceBytes = await readFile(new URL("sample/source.webp", publicRoot));
    const bundledGraph = JSON.parse(graphBytes.toString("utf8")) as PlanGraph;

    verifyStudioSampleContract(bundledManifest, bundledGraph, {
      graphSha256: createHash("sha256").update(graphBytes).digest("hex"),
      displaySourceSha256: createHash("sha256").update(sourceBytes).digest("hex"),
    });
    expect(bundledGraph.fixtures).toHaveLength(65);
    expect(bundledGraph.pipeline?.demo_kind).toBe("reviewed_full_editable_product_demo");
    expect(bundledGraph.pipeline?.evaluation_eligible).toBe(false);
    expect(bundledManifest.content_contract).toEqual({
      profile: "full_editable_bim",
      whole_sheet_reviewed: true,
      omission_scan_completed: true,
      counts: { walls: 17, openings: 22, rooms: 16, fixtures: 65 },
    });
  });
});
