import { describe, expect, it } from "vitest";

import { exactRotationChanges, exactTranslationChanges } from "./exactTransform";
import type { PlanGraph } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }],
    walls: [{
      id: "wall-1",
      level_id: "L1",
      from: [0, 0],
      to: [6, 0],
      thickness_m: 0.12,
      height_m: 3,
    }],
    rooms: [],
    openings: [
      {
        id: "door-1",
        level_id: "L1",
        type: "door",
        wall_id: "wall-1",
        center_m: [2, 0],
        x_m: 2,
        width_m: 0.9,
        height_m: 2.1,
      },
      {
        id: "door-2",
        level_id: "L1",
        type: "door",
        wall_id: "wall-1",
        center_m: [4, 0],
        x_m: 4,
        width_m: 0.9,
        height_m: 2.1,
      },
    ],
    fixtures: [{
      id: "sink-1",
      level_id: "L1",
      type: "sink",
      center_m: [1, 2],
      size_m: [0.6, 0.5, 0.9],
    }],
    routes: [{
      id: "duct-1",
      level_id: "L1",
      points_m: [[0, 1, 2.7], [2, 1, 2.7]],
    }],
    vertical_connections: [],
    dimensions: [{ id: "dim-1", level_id: "L1", from: [0, 0], to: [1, 0] }],
  };
}

describe("exact translation", () => {
  it("moves heterogeneous editable geometry with one metric delta", () => {
    const result = exactTranslationChanges(
      graph(),
      [
        { collection: "fixtures", id: "sink-1" },
        { collection: "routes", id: "duct-1" },
        { collection: "dimensions", id: "dim-1" },
      ],
      [0.25, -0.5],
    );
    expect(result.valid).toBe(true);
    expect(result.changesById["sink-1"].center_m).toEqual([1.25, 1.5]);
    expect(result.changesById["duct-1"].points_m).toEqual([
      [0.25, 0.5, 2.7],
      [2.25, 0.5, 2.7],
    ]);
    expect(result.changesById["dim-1"]).toEqual({ from: [0.25, -0.5], to: [1.25, -0.5] });
  });

  it("moves both wall endpoints so hosted elements can follow in the reducer", () => {
    const result = exactTranslationChanges(
      graph(),
      [{ collection: "walls", id: "wall-1" }],
      [1, 2],
    );
    expect(result.changesById["wall-1"]).toEqual({ from: [1, 2], to: [7, 2] });
  });

  it("projects an opening onto its host wall and preserves host semantics", () => {
    const result = exactTranslationChanges(
      graph(),
      [{ collection: "openings", id: "door-1" }],
      [0.25, 0.5],
    );
    expect(result.valid).toBe(true);
    expect(result.changesById["door-1"].center_m).toEqual([2.25, 0]);
    expect(result.changesById["door-1"].x_m).toBe(2.25);
    expect(result.notices[0]).toContain("projected onto its host wall");
  });

  it("blocks an exact move that would overlap an unselected opening", () => {
    const result = exactTranslationChanges(
      graph(),
      [{ collection: "openings", id: "door-1" }],
      [2, 0],
    );
    expect(result.valid).toBe(false);
    expect(result.changesById).toEqual({});
    expect(result.reason).toContain("door-2");
  });

  it("blocks a perpendicular opening offset that would be a silent no-op", () => {
    const result = exactTranslationChanges(
      graph(),
      [{ collection: "openings", id: "door-1" }],
      [0, 1],
    );
    expect(result.valid).toBe(false);
    expect(result.reason).toContain("no component along its host wall");
  });
});

describe("exact group rotation", () => {
  it("rotates placed objects around one plan pivot and preserves their relative layout", () => {
    const source = graph();
    source.fixtures.push({
      id: "sink-2",
      level_id: "L1",
      type: "sink",
      center_m: [3, 2],
      size_m: [0.6, 0.5, 0.9],
      yaw_deg: 10,
    });
    const result = exactRotationChanges(
      source,
      [
        { collection: "fixtures", id: "sink-1" },
        { collection: "fixtures", id: "sink-2" },
      ],
      [2, 2],
      90,
    );
    expect(result.valid).toBe(true);
    expect(result.changesById["sink-1"]).toEqual({ center_m: [2, 1], yaw_deg: 90 });
    expect(result.changesById["sink-2"]).toEqual({ center_m: [2, 3], yaw_deg: 100 });
  });

  it("normalizes yaw and rejects hosted or structural geometry", () => {
    const source = graph();
    source.fixtures[0].yaw_deg = 170;
    expect(exactRotationChanges(source, [{ collection: "fixtures", id: "sink-1" }], [1, 2], 30).changesById["sink-1"].yaw_deg).toBe(-160);
    expect(exactRotationChanges(source, [{ collection: "walls", id: "wall-1" }], [0, 0], 45).valid).toBe(false);
  });
});
