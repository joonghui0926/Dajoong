import { describe, expect, it } from "vitest";

import {
  cycleSelectionIndex,
  hitTestPlanGraph,
  selectionCandidateLabel,
} from "./planHitTest";
import type { PlanGraph, Selection } from "./types";

const graph: PlanGraph = {
  schema_version: "1",
  levels: [{ id: "L1", name: "Level 1" }],
  rooms: [{ id: "R1", level_id: "L1", name: "Plant room", polygon: [[0, 0], [8, 0], [8, 6], [0, 6]] }],
  walls: [{ id: "W1", level_id: "L1", from: [0, 3], to: [8, 3], thickness_m: 0.2, height_m: 3 }],
  openings: [],
  fixtures: [{ id: "F1", level_id: "L1", type: "air_handler", center_m: [4, 3], size_m: [2, 1, 1.8] }],
  routes: [{ id: "D1", level_id: "L1", kind: "supply duct", points_m: [[1, 1, 2.5], [7, 1, 2.5]] }],
};

const selections: Selection[] = [
  { collection: "rooms", id: "R1" },
  { collection: "walls", id: "W1" },
  { collection: "fixtures", id: "F1" },
  { collection: "routes", id: "D1" },
];

describe("plan hit testing", () => {
  it("returns every stacked element in editing priority order", () => {
    expect(hitTestPlanGraph(graph, selections, [4, 3], 0.08)).toEqual([
      { collection: "fixtures", id: "F1" },
      { collection: "walls", id: "W1" },
      { collection: "rooms", id: "R1" },
    ]);
  });

  it("uses screen-derived tolerance for narrow linework", () => {
    expect(hitTestPlanGraph(graph, selections, [3, 1.07], 0.08)).toEqual([
      { collection: "routes", id: "D1" },
      { collection: "rooms", id: "R1" },
    ]);
    expect(hitTestPlanGraph(graph, selections, [3, 1.2], 0.08)).toEqual([
      { collection: "rooms", id: "R1" },
    ]);
  });

  it("cycles forward and backward without leaving the candidate list", () => {
    expect(cycleSelectionIndex(2, 3)).toBe(0);
    expect(cycleSelectionIndex(0, 3, true)).toBe(2);
  });

  it("uses semantic names for the compact candidate HUD", () => {
    expect(selectionCandidateLabel(graph, { collection: "fixtures", id: "F1" })).toBe("air handler");
    expect(selectionCandidateLabel(graph, { collection: "rooms", id: "R1" })).toBe("Plant room");
  });
});
