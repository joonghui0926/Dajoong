import { describe, expect, it } from "vitest";

import type { FixtureFamily } from "./families";
import {
  fixturePlacementAt,
  findNearestValidFixtureCopy,
  polygonContainsPolygon,
  polygonsOverlap,
  rotatedFixtureFootprint,
  validateFixtureEntityChanges,
  validateFixtureTransformChanges,
  validateNewFixtures,
} from "./fixturePlacement";
import type { PlanGraph } from "./types";

const floorFamily: FixtureFamily = {
  id: "test:desk",
  name: "Desk",
  category: "Furniture",
  type: "desk",
  discipline: "architectural",
  size_m: [2, 1, 0.75],
  material: "wood",
  keywords: ["desk"],
  mounting: "floor",
};

const wallFamily: FixtureFamily = {
  ...floorFamily,
  id: "test:panel",
  name: "Panel",
  type: "panel",
  size_m: [0.8, 0.2, 1.1],
  mounting: "wall",
};

const ceilingFamily: FixtureFamily = {
  ...floorFamily,
  id: "test:diffuser",
  name: "Diffuser",
  type: "ceiling-diffuser",
  size_m: [0.6, 0.6, 0.08],
  mounting: "ceiling",
};

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }],
    rooms: [{ id: "room-1", level_id: "L1", name: "Office", polygon: [[0, 0], [10, 0], [10, 8], [0, 8]] }],
    walls: [
      { id: "wall-top", level_id: "L1", from: [0, 0], to: [10, 0], thickness_m: 0.2, height_m: 3 },
      { id: "wall-divider", level_id: "L1", from: [5, 0], to: [5, 8], thickness_m: 0.2, height_m: 3 },
    ],
    openings: [],
    fixtures: [{ id: "desk-existing", level_id: "L1", type: "desk", center_m: [2, 3], size_m: [2, 1, 0.75], yaw_deg: 0 }],
    routes: [],
    vertical_connections: [],
  };
}

describe("fixture placement", () => {
  it("creates an exact rotated footprint and reusable polygon predicates", () => {
    const footprint = rotatedFixtureFootprint([2, 2], [2, 1], 90);
    expect(footprint[0][0]).toBeCloseTo(2.5);
    expect(footprint[0][1]).toBeCloseTo(1);
    expect(polygonContainsPolygon([[0, 0], [4, 0], [4, 4], [0, 4]], footprint)).toBe(true);
    expect(polygonsOverlap(footprint, rotatedFixtureFootprint([2, 3], [2, 1], 0))).toBe(true);
  });

  it("assigns the containing room and preserves requested rotation", () => {
    const placement = fixturePlacementAt(graph(), floorFamily, "L1", [8, 4], 90);
    expect(placement).toMatchObject({ valid: true, roomId: "room-1", yawDeg: 90 });
  });

  it("blocks boundaries, walls, and same-band object collisions", () => {
    expect(fixturePlacementAt(graph(), floorFamily, "L1", [0.3, 4], 0).reason).toBe("outside_room");
    expect(fixturePlacementAt(graph(), floorFamily, "L1", [5, 4], 90)).toMatchObject({
      valid: false,
      reason: "wall_conflict",
      conflictIds: ["wall-divider"],
    });
    expect(fixturePlacementAt(graph(), floorFamily, "L1", [2.4, 3], 0)).toMatchObject({
      valid: false,
      reason: "fixture_conflict",
      conflictIds: ["desk-existing"],
    });
  });

  it("finds and aligns a wall host before placement", () => {
    const placement = fixturePlacementAt(graph(), wallFamily, "L1", [8, 0.35], 0);
    expect(placement.valid).toBe(true);
    expect(placement.hostWallId).toBe("wall-top");
    expect(placement.roomId).toBe("room-1");
    expect(placement.yawDeg).toBe(0);
    expect(placement.center[1]).toBeCloseTo(0.21);
    expect(fixturePlacementAt(graph(), wallFamily, "L1", [8, 4], 0).reason).toBe("wall_host_required");
  });

  it("keeps ceiling and floor collision bands independent", () => {
    const placement = fixturePlacementAt(graph(), ceilingFamily, "L1", [2, 3], 0);
    expect(placement).toMatchObject({ valid: true, roomId: "room-1", mounting: "ceiling" });
  });

  it("chooses the room side when a snapped wall cursor has no signed side", () => {
    const oppositeRoom = graph();
    oppositeRoom.rooms = [{
      id: "room-below",
      level_id: "L1",
      name: "Below",
      polygon: [[0, -4], [10, -4], [10, 0], [0, 0]],
    }];
    oppositeRoom.walls = [oppositeRoom.walls[0]];
    oppositeRoom.fixtures = [];
    const placement = fixturePlacementAt(oppositeRoom, wallFamily, "L1", [8, 0], 0);
    expect(placement.valid).toBe(true);
    expect(placement.roomId).toBe("room-below");
    expect(placement.center[1]).toBeLessThan(0);
  });

  it("revalidates edited fixtures and refreshes room relationships", () => {
    const moved = validateFixtureEntityChanges(graph(), "desk-existing", {
      center_m: [8, 4],
      yaw_deg: 90,
    });
    expect(moved.valid).toBe(true);
    expect(moved.changesById["desk-existing"]).toMatchObject({
      center_m: [8, 4],
      yaw_deg: 90,
      room_id: "room-1",
      mounting: "floor",
    });
    expect(validateFixtureEntityChanges(graph(), "desk-existing", {
      center_m: [5, 4],
    })).toMatchObject({ valid: false, reason: expect.stringContaining("Wall collision") });
  });

  it("rejects invalid dimensions and collisions between a transformed group", () => {
    const invalidSize = validateFixtureEntityChanges(graph(), "desk-existing", {
      size_m: [0, 1, 0.75],
    });
    expect(invalidSize).toMatchObject({ valid: false, reason: expect.stringContaining("invalid component dimensions") });

    const groupGraph = graph();
    groupGraph.fixtures.push({
      id: "desk-second",
      level_id: "L1",
      type: "desk",
      center_m: [8, 3],
      size_m: [2, 1, 0.75],
      yaw_deg: 0,
    });
    const group = validateFixtureTransformChanges(groupGraph, {
      "desk-existing": { center_m: [7, 4] },
      "desk-second": { center_m: [7.4, 4] },
    });
    expect(group).toMatchObject({ valid: false, reason: expect.stringContaining("Object collision") });
  });

  it("validates new objects and finds the nearest clear duplicate offset", () => {
    const clone = {
      ...graph().fixtures[0],
      id: "desk-copy",
    };
    expect(validateNewFixtures(graph(), [clone])).toMatchObject({
      valid: false,
      reason: expect.stringContaining("Object collision"),
    });
    const copy = findNearestValidFixtureCopy(graph(), [clone], 0.05);
    expect(copy.valid).toBe(true);
    expect(copy.fixtures[0].id).toBe("desk-copy");
    expect(copy.fixtures[0].room_id).toBe("room-1");
    expect(copy.offset).not.toEqual([0, 0]);
  });
});
