import { describe, expect, it } from "vitest";

import {
  includesSelection,
  sanitizeEntityViewState,
  selectionKey,
  toggleSelectionState,
} from "./editorViewState";
import type { PlanGraph, Selection } from "./types";

const wall: Selection = { collection: "walls", id: "L1:wall:1" };
const graph: PlanGraph = {
  schema_version: "buili.plan-graph.v2",
  levels: [{ id: "L1", name: "Level 1" }],
  walls: [
    {
      id: wall.id,
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

describe("editor view state", () => {
  it("uses a collision-safe key and toggles an exact entity", () => {
    expect(selectionKey(wall)).not.toBe(selectionKey({ collection: "openings", id: wall.id }));
    const hidden = toggleSelectionState([], wall);
    expect(includesSelection(hidden, wall)).toBe(true);
    expect(toggleSelectionState(hidden, wall)).toEqual([]);
  });

  it("deduplicates, validates, and drops entities missing from the active graph", () => {
    expect(
      sanitizeEntityViewState(
        [
          wall,
          wall,
          { collection: "fixtures", id: "missing" },
          { collection: "unknown", id: "bad" },
          { collection: "walls", id: 12 },
        ],
        graph,
      ),
    ).toEqual([wall]);
  });
});
