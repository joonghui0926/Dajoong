import { describe, expect, it } from "vitest";

import { graphBounds, updateEntity } from "./graph";
import type { PlanGraph } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1", nominal_height_m: 3 }],
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
    routes: [
      {
        id: "route-1",
        level_id: "L1",
        discipline: "mechanical",
        points_m: [
          [1, 1, 2.5],
          [8, 3, 2.5],
        ],
        section_m: [0.2, 0.1],
      },
    ],
    vertical_connections: [
      {
        id: "stair-1",
        type: "stair",
        from_level_id: "L1",
        to_level_id: "L2",
        center_m: [-2, 2],
        footprint_m: [2, 4],
      },
    ],
  };
}

describe("graph editing", () => {
  it("includes systems and vertical circulation in level bounds", () => {
    expect(graphBounds(graph(), "L1")).toEqual({
      minX: -3,
      minY: 0,
      maxX: 8,
      maxY: 4,
    });
  });

  it("turns a corrected element into accepted auditable geometry", () => {
    const updated = updateEntity(
      graph(),
      { collection: "walls", id: "wall-1" },
      { height_m: 3.4 },
    );
    expect(updated.walls[0]).toMatchObject({
      id: "wall-1",
      height_m: 3.4,
      confidence: 1,
      uncertainty: 0,
      review_state: "accepted",
      model_version: "human-correction",
    });
  });

  it("propagates a coincident endpoint and keeps hosted openings on the moved wall", () => {
    const constrained: PlanGraph = {
      ...graph(),
      walls: [
        { id: "wall-1", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.12, height_m: 3 },
        { id: "wall-2", level_id: "L1", from: [4, 0], to: [4, 4], thickness_m: 0.12, height_m: 3 },
      ],
      openings: [
        { id: "door-1", level_id: "L1", type: "door", wall_id: "wall-2", center_m: [4, 2], x_m: 2, width_m: 0.9, height_m: 2.1 },
      ],
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
    const updated = updateEntity(
      constrained,
      { collection: "walls", id: "wall-1" },
      { to: [5, 1] },
    );
    expect(updated.walls[1].from).toEqual([5, 1]);
    const door = updated.openings[0];
    const wall = updated.walls[1];
    const cross = (door.center_m[0] - wall.from[0]) * (wall.to[1] - wall.from[1])
      - (door.center_m[1] - wall.from[1]) * (wall.to[0] - wall.from[0]);
    expect(Math.abs(cross)).toBeLessThan(1e-9);
    expect(door.x_m).toBe(2);
  });
});
