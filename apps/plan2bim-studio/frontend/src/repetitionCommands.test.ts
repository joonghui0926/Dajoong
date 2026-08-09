import { describe, expect, it } from "vitest";

import { defaultMirrorCoordinates, linearArrayPattern, mirrorPattern, patternCenter } from "./repetitionCommands";
import type { PlanGraph, Selection } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }, { id: "L2", name: "Level 2" }],
    walls: [],
    rooms: [],
    openings: [],
    routes: [],
    fixtures: [
      { id: "desk-1", level_id: "L1", type: "desk", center_m: [2, 3], size_m: [1, .6, .75], yaw_deg: 30 },
      { id: "chair-1", level_id: "L1", type: "chair", center_m: [4, 5], size_m: [.5, .5, .9], yaw_deg: -15 },
    ],
    vertical_connections: [{
      id: "stair-1",
      from_level_id: "L1",
      to_level_id: "L2",
      center_m: [8, 4],
      footprint_m: [2, 4],
      yaw_deg: 90,
    }],
  };
}

const desk: Selection = { collection: "fixtures", id: "desk-1" };
const chair: Selection = { collection: "fixtures", id: "chair-1" };

describe("repetition commands", () => {
  it("computes the centroid used as the non-destructive mirror default", () => {
    expect(patternCenter(graph(), [desk, chair])).toEqual([3, 4]);
  });

  it("places the default mirror lines beyond the selected extents to avoid overlap", () => {
    const coordinates = defaultMirrorCoordinates(graph(), [desk]);
    expect(coordinates?.[0]).toBeGreaterThan(2.7);
    expect(coordinates?.[1]).toBeGreaterThan(3.7);
  });

  it("creates a reviewed mirror copy with reflected position and yaw", () => {
    const result = mirrorPattern(graph(), [desk], "vertical", 5, true);
    expect(result.valid).toBe(true);
    expect(result.items).toHaveLength(1);
    expect(result.items[0].entity.center_m).toEqual([8, 3]);
    expect(result.items[0].entity.yaw_deg).toBe(150);
    expect(result.items[0].entity.review_state).toBe("accepted");
    expect(result.items[0].entity.id).not.toBe("desk-1");
  });

  it("can mirror the source in place when keeping the original is disabled", () => {
    const result = mirrorPattern(graph(), [chair], "horizontal", 2, false);
    expect(result.items).toHaveLength(0);
    expect(result.changesById["chair-1"]).toEqual({ center_m: [4, -1], yaw_deg: 15 });
  });

  it("builds a linear array whose count includes the source", () => {
    const result = linearArrayPattern(graph(), [desk, chair], 3, [1.25, -0.5]);
    expect(result.valid).toBe(true);
    expect(result.items).toHaveLength(4);
    expect(result.items.map((item) => item.entity.center_m)).toEqual([
      [3.25, 2.5],
      [5.25, 4.5],
      [4.5, 2],
      [6.5, 4],
    ]);
  });

  it("rejects zero-spacing and non-component patterns", () => {
    expect(linearArrayPattern(graph(), [desk], 4, [0, 0]).valid).toBe(false);
    expect(mirrorPattern(graph(), [{ collection: "walls", id: "missing" }], "vertical", 0, true).valid).toBe(false);
  });
});
