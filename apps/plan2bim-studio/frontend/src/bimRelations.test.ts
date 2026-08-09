import { describe, expect, it } from "vitest";

import { relatedSelectionGroups } from "./bimRelations";
import type { PlanGraph } from "./types";

const graph: PlanGraph = {
  schema_version: "buili.plan-graph.v2",
  levels: [{ id: "L1", name: "Level 1" }],
  walls: [
    { id: "wall-1", level_id: "L1", room_id: "room-1", from: [0, 0], to: [4, 0], thickness_m: 0.12, height_m: 3 },
    { id: "wall-2", level_id: "L1", room_id: "room-1", from: [4, 0], to: [4, 4], thickness_m: 0.12, height_m: 3 },
  ],
  rooms: [{ id: "room-1", level_id: "L1", name: "Office", polygon: [[0, 0], [4, 0], [4, 4], [0, 4]] }],
  openings: [
    { id: "door-1", level_id: "L1", type: "door", wall_id: "wall-1", center_m: [1, 0], width_m: 0.9, height_m: 2.1 },
    { id: "window-1", level_id: "L1", type: "window", wall_id: "wall-1", center_m: [3, 0], width_m: 1.2, height_m: 1.2 },
  ],
  fixtures: [
    { id: "desk-1", level_id: "L1", type: "desk", room_id: "room-1", center_m: [1, 1], size_m: [1, 0.6, 0.75] },
    { id: "chair-1", level_id: "L1", type: "chair", room_id: "room-1", center_m: [2, 1], size_m: [0.5, 0.5, 0.9] },
  ],
  routes: [
    { id: "route-1", level_id: "L1", system_id: "supply-air", points_m: [[0, 0, 2.6], [2, 0, 2.6]] },
    { id: "route-2", level_id: "L1", system_id: "supply-air", points_m: [[2, 0, 2.6], [4, 0, 2.6]] },
  ],
  vertical_connections: [],
  constraints: [
    {
      id: "constraint-1",
      level_id: "L1",
      type: "coincident",
      references: [
        { collection: "walls", entity_id: "wall-1", handle: "to" },
        { collection: "walls", entity_id: "wall-2", handle: "from" },
      ],
    },
  ],
};

describe("BIM relationship navigation", () => {
  it("moves from an opening to its host and sibling openings", () => {
    const groups = relatedSelectionGroups(graph, { collection: "openings", id: "door-1" });
    expect(groups.map((item) => item.id)).toEqual(["host-wall", "wall-openings"]);
    expect(groups[0].selections).toEqual([{ collection: "walls", id: "wall-1" }]);
    expect(groups[1].selections).toHaveLength(2);
  });

  it("finds hosted openings and a transitive wall constraint chain", () => {
    const groups = relatedSelectionGroups(graph, { collection: "walls", id: "wall-1" });
    expect(groups.find((item) => item.id === "hosted-openings")?.selections).toHaveLength(2);
    expect(groups.find((item) => item.id === "constraint-chain")?.selections).toEqual([
      { collection: "walls", id: "wall-1" },
      { collection: "walls", id: "wall-2" },
    ]);
  });

  it("navigates room membership, constraints, and MEP systems", () => {
    expect(relatedSelectionGroups(graph, { collection: "fixtures", id: "desk-1" }).map((item) => item.id)).toEqual([
      "fixture-room",
      "room-objects",
    ]);
    expect(relatedSelectionGroups(graph, { collection: "rooms", id: "room-1" }).map((item) => item.id)).toEqual([
      "contained-objects",
      "room-walls",
    ]);
    expect(relatedSelectionGroups(graph, { collection: "constraints", id: "constraint-1" })[0].selections).toHaveLength(2);
    expect(relatedSelectionGroups(graph, { collection: "routes", id: "route-1" })[0].selections).toHaveLength(2);
  });
});
