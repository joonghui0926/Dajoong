import { describe, expect, it } from "vitest";

import {
  selectInRectangle,
  selectionMode,
  selectionRectangle,
} from "./windowSelection";
import type { PlanGraph, Selection } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }],
    walls: [{ id: "wall-1", level_id: "L1", from: [0, 2], to: [10, 2], thickness_m: .2, height_m: 3 }],
    rooms: [{ id: "room-1", level_id: "L1", name: "Room", polygon: [[0, 0], [10, 0], [10, 6], [0, 6]] }],
    openings: [],
    fixtures: [
      { id: "desk-1", level_id: "L1", type: "desk", center_m: [3, 4], size_m: [2, 1, .75], yaw_deg: 0 },
      { id: "desk-2", level_id: "L1", type: "desk", center_m: [7, 4], size_m: [2, 1, .75], yaw_deg: 45 },
    ],
    routes: [{ id: "route-1", level_id: "L1", points_m: [[1, 5, 2.7], [9, 5, 2.7]] }],
    vertical_connections: [],
  };
}

const candidates: Selection[] = [
  { collection: "walls", id: "wall-1" },
  { collection: "rooms", id: "room-1" },
  { collection: "fixtures", id: "desk-1" },
  { collection: "fixtures", id: "desk-2" },
  { collection: "routes", id: "route-1" },
];

describe("Autodesk style window selection", () => {
  it("uses drag direction to distinguish window from crossing mode", () => {
    expect(selectionMode([1, 1], [5, 5])).toBe("window");
    expect(selectionMode([5, 1], [1, 5])).toBe("crossing");
    expect(selectionRectangle([5, 5], [1, 1])).toEqual({ minX: 1, minY: 1, maxX: 5, maxY: 5 });
  });

  it("selects only geometry fully enclosed by a left to right window", () => {
    const selected = selectInRectangle(
      graph(),
      candidates,
      selectionRectangle([1.5, 3.25], [4.5, 4.75]),
      "window",
    );
    expect(selected).toEqual([{ collection: "fixtures", id: "desk-1" }]);
  });

  it("selects geometry touched by a right to left crossing box", () => {
    const selected = selectInRectangle(
      graph(),
      candidates,
      selectionRectangle([6, 1.5], [4, 2.5]),
      "crossing",
    );
    expect(selected).toContainEqual({ collection: "walls", id: "wall-1" });
    expect(selected).toContainEqual({ collection: "rooms", id: "room-1" });
    expect(selected).not.toContainEqual({ collection: "fixtures", id: "desk-1" });
  });

  it("uses the actual rotated component footprint instead of its unrotated bounds", () => {
    const selected = selectInRectangle(
      graph(),
      [{ collection: "fixtures", id: "desk-2" }],
      selectionRectangle([5.8, 2.8], [8.2, 5.2]),
      "window",
    );
    expect(selected).toEqual([{ collection: "fixtures", id: "desk-2" }]);
  });
});
