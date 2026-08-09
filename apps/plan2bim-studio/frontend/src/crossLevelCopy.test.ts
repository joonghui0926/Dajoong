import { describe, expect, it } from "vitest";

import { copySelectionsToLevel } from "./crossLevelCopy";
import type { PlanGraph } from "./types";

const graph: PlanGraph = {
  schema_version: "buili.plan-graph.v2",
  levels: [
    { id: "L1", name: "Level 1", elevation_m: 0, nominal_height_m: 3 },
    { id: "L2", name: "Level 2", elevation_m: 3, nominal_height_m: 3 },
  ],
  rooms: [{ id: "L1:room:1", level_id: "L1", name: "Office", polygon: [[0, 0], [5, 0], [5, 4], [0, 4]] }],
  walls: [
    { id: "L1:wall:1", level_id: "L1", room_id: "L1:room:1", from: [0, 0], to: [5, 0], thickness_m: 0.12, height_m: 3 },
    { id: "L1:wall:2", level_id: "L1", room_id: "L1:room:1", from: [5, 0], to: [5, 4], thickness_m: 0.12, height_m: 3 },
  ],
  openings: [{ id: "L1:door:1", level_id: "L1", type: "door", wall_id: "L1:wall:1", center_m: [2.5, 0], width_m: 0.9, height_m: 2.1 }],
  fixtures: [{ id: "L1:desk:1", level_id: "L1", type: "desk", room_id: "L1:room:1", center_m: [2, 2], size_m: [1.4, 0.7, 0.75] }],
  routes: [],
  constraints: [{
    id: "L1:constraint:1",
    level_id: "L1",
    type: "coincident",
    references: [
      { collection: "walls", entity_id: "L1:wall:1", handle: "to" },
      { collection: "walls", entity_id: "L1:wall:2", handle: "from" },
    ],
  }],
  dimensions: [{ id: "L1:dimension:1", level_id: "L1", type: "aligned", from: [0, 0], to: [5, 0] }],
  vertical_connections: [],
};

describe("cross-level BIM copy", () => {
  it("expands hosted dependencies and remaps every copied reference", () => {
    const result = copySelectionsToLevel(
      graph,
      [
        { collection: "walls", id: "L1:wall:1" },
        { collection: "walls", id: "L1:wall:2" },
        { collection: "fixtures", id: "L1:desk:1" },
        { collection: "dimensions", id: "L1:dimension:1" },
      ],
      "L2",
    );
    expect(result.items.map((item) => item.collection)).toEqual([
      "rooms", "walls", "walls", "openings", "fixtures", "dimensions", "constraints",
    ]);
    const wallIds = result.items.filter((item) => item.collection === "walls").map((item) => item.entity.id);
    const room = result.items.find((item) => item.collection === "rooms")?.entity;
    const opening = result.items.find((item) => item.collection === "openings")?.entity;
    const fixture = result.items.find((item) => item.collection === "fixtures")?.entity;
    const constraint = result.items.find((item) => item.collection === "constraints")?.entity;
    expect(opening?.wall_id).toBe(wallIds[0]);
    expect(fixture?.room_id).toBe(room?.id);
    expect(constraint?.references).toEqual([
      { collection: "walls", entity_id: wallIds[0], handle: "to" },
      { collection: "walls", entity_id: wallIds[1], handle: "from" },
    ]);
    expect(result.items.every((item) => item.entity.level_id === "L2")).toBe(true);
    expect(result.items.every((item) => item.entity.copied_from_entity_id)).toBe(true);
    expect(result.conflicts).toEqual([]);
    expect(graph.openings[0].wall_id).toBe("L1:wall:1");
  });

  it("refuses to reinterpret a vertical connection as single-level geometry", () => {
    const withStair: PlanGraph = {
      ...graph,
      vertical_connections: [{
        id: "stair-1",
        type: "stair",
        from_level_id: "L1",
        to_level_id: "L2",
        center_m: [2, 2],
        footprint_m: [2, 4],
      }],
    };
    const result = copySelectionsToLevel(
      withStair,
      [{ collection: "vertical_connections", id: "stair-1" }],
      "L2",
    );
    expect(result.items).toEqual([]);
    expect(result.warnings[0]).toContain("explicit pair of levels");
  });

  it("fails clearly when the target level is missing", () => {
    expect(() => copySelectionsToLevel(graph, [], "L9")).toThrow("Target level does not exist");
  });

  it("blocks a transaction that would stack geometry on the target level", () => {
    const occupied: PlanGraph = {
      ...graph,
      walls: [
        ...graph.walls,
        { id: "L2:wall:existing", level_id: "L2", from: [0, 0], to: [5, 0], thickness_m: 0.12, height_m: 3 },
      ],
    };
    const result = copySelectionsToLevel(
      occupied,
      [{ collection: "walls", id: "L1:wall:1" }],
      "L2",
    );
    expect(result.items).toEqual([]);
    expect(result.conflicts[0]).toMatchObject({
      sourceId: "L1:wall:1",
      targetId: "L2:wall:existing",
      reason: "coincident_wall",
    });
  });
});
