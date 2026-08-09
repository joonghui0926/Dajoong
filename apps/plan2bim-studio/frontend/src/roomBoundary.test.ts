import { describe, expect, it } from "vitest";

import {
  insertRoomBoundaryVertex,
  planRoomBoundaryTransform,
  removeRoomBoundaryVertex,
  snapRoomBoundaryPoint,
} from "./roomBoundary";
import type { PlanGraph } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }],
    walls: [
      { id: "wall-diagonal", level_id: "L1", from: [0, 0], to: [4, 4], thickness_m: 0.12, height_m: 3 },
    ],
    rooms: [
      { id: "room-left", level_id: "L1", name: "Office", polygon: [[0, 0], [4, 0], [4, 4], [0, 4]] },
      { id: "room-right", level_id: "L1", name: "Hall", polygon: [[4, 0], [8, 0], [8, 4], [4, 4]] },
    ],
    openings: [],
    fixtures: [
      {
        id: "desk-1",
        level_id: "L1",
        type: "desk",
        room_id: "room-left",
        center_m: [3.5, 2],
        size_m: [0.4, 0.4, 0.75],
        yaw_deg: 0,
      },
    ],
    routes: [],
    vertical_connections: [],
  };
}

describe("room boundary editing", () => {
  it("inserts a midpoint vertex and removes a safe redundant vertex", () => {
    const polygon: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]];
    const inserted = insertRoomBoundaryVertex(polygon, 0);
    expect(inserted).toMatchObject({ valid: true, vertexIndex: 1 });
    expect(inserted.polygon[1]).toEqual([2, 0]);
    expect(removeRoomBoundaryVertex(inserted.polygon, 1)).toMatchObject({
      valid: true,
      polygon,
    });
    expect(removeRoomBoundaryVertex([[0, 0], [1, 0], [0, 1]], 1)).toMatchObject({
      valid: false,
      reason: expect.stringContaining("at least three"),
    });
  });

  it("snaps a room grip to the nearest point along a wall axis", () => {
    const model = graph();
    const snap = snapRoomBoundaryPoint([2, 2.08], model.walls, 0.1);
    expect(snap?.wallId).toBe("wall-diagonal");
    expect(snap?.point[0]).toBeCloseTo(2.04);
    expect(snap?.point[1]).toBeCloseTo(2.04);
    expect(snapRoomBoundaryPoint([2, 2.3], model.walls, 0.1)).toBeNull();
  });

  it("blocks a new room overlap before mutating the graph", () => {
    const plan = planRoomBoundaryTransform(graph(), "room-left", [
      [0, 0], [5, 0], [5, 4], [0, 4],
    ]);
    expect(plan).toMatchObject({
      valid: false,
      reason: expect.stringContaining("overlap room-right"),
    });
  });

  it("marks an object for review when an edited boundary no longer contains it", () => {
    const plan = planRoomBoundaryTransform(graph(), "room-left", [
      [0, 0], [3, 0], [3, 4], [0, 4],
    ]);
    expect(plan.valid).toBe(true);
    expect(plan.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        selection: { collection: "fixtures", id: "desk-1" },
        changes: expect.objectContaining({ room_id: null, review_state: "review_required" }),
      }),
    ]));
    expect(plan.graph?.fixtures[0].room_id).toBeNull();
  });
});
