import { describe, expect, it } from "vitest";

import {
  distanceMeters,
  movePolygonVertex,
  polygonArea,
  setWallLength,
  setSegmentLength,
  smartSnap,
} from "./editorGeometry";
import type { PlanGraph } from "./types";

const graph: PlanGraph = {
  schema_version: "buili.plan-graph.v2",
  levels: [{ id: "L1", name: "Level 1" }],
  walls: [
    {
      id: "wall-1",
      level_id: "L1",
      from: [0, 0],
      to: [4, 0],
      thickness_m: 0.12,
      height_m: 3,
    },
  ],
  rooms: [],
  openings: [],
  fixtures: [],
  routes: [],
  vertical_connections: [],
};

describe("constraint aware editing geometry", () => {
  it("prefers a semantic endpoint inside screen derived tolerance", () => {
    const result = smartSnap([4.04, 0.03], graph, "L1", {
      grid_m: 0.05,
      tolerance_m: 0.1,
    });
    expect(result.point).toEqual([4, 0]);
    expect(result.label).toBe("Wall endpoint");
    expect(result.guides).toHaveLength(2);
  });

  it("locks construction to the dominant orthogonal axis", () => {
    const result = smartSnap([2.03, 0.62], graph, "L1", {
      grid_m: 0.05,
      tolerance_m: 0.01,
      origin: [1, 1],
      orthogonal: true,
    });
    expect(result.point).toEqual([2.05, 1]);
    expect(result.label).toBe("Orthogonal");
  });

  it("allows a temporary free placement override", () => {
    const result = smartSnap([1.237, 2.468], graph, "L1", {
      grid_m: 0.5,
      tolerance_m: 1,
      disabled: true,
    });
    expect(result.point).toEqual([1.237, 2.468]);
    expect(result.guides).toEqual([]);
  });

  it("reports metric distance without drawing scale ambiguity", () => {
    expect(distanceMeters([0, 0], [3, 4])).toBe(5);
  });

  it("uses a wall length as a driving dimension", () => {
    expect(setWallLength(graph.walls[0], 6)).toEqual({ from: [0, 0], to: [6, 0] });
    expect(setWallLength(graph.walls[0], 2, "end")).toEqual({ from: [2, 0], to: [4, 0] });
    expect(setWallLength(graph.walls[0], 0.01)).toBeNull();
    expect(setSegmentLength([1, 1], [1, 3], 4)).toEqual({ from: [1, 1], to: [1, 5] });
  });

  it("moves one room vertex while preserving a valid concave boundary", () => {
    const result = movePolygonVertex(
      [[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]],
      3,
      [2.5, 2],
    );
    expect(result.valid).toBe(true);
    expect(result.polygon[3]).toEqual([2.5, 2]);
    expect(polygonArea(result.polygon)).toBe(12);
  });

  it("rejects a room edit that crosses another boundary edge", () => {
    const original: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]];
    const result = movePolygonVertex(original, 1, [-1, 3]);
    expect(result.valid).toBe(false);
    expect(result.reason).toBe("self_intersection");
    expect(result.polygon).toEqual(original);
  });
});
