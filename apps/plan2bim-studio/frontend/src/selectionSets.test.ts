import { describe, expect, it } from "vitest";

import {
  createSelectionSet,
  defaultSelectionSetName,
  renameSelectionSet,
  sanitizeSelectionSets,
  selectionSetSummary,
  type BimSelectionSet,
} from "./selectionSets";
import type { PlanGraph } from "./types";

const graph: PlanGraph = {
  schema_version: "buili.plan-graph.v2",
  levels: [{ id: "L1", name: "Level 1" }],
  walls: [{ id: "wall-1", level_id: "L1", from: [0, 0], to: [3, 0], thickness_m: 0.12, height_m: 3 }],
  rooms: [],
  openings: [],
  fixtures: [{ id: "chair-1", level_id: "L1", type: "chair", center_m: [1, 1], size_m: [0.5, 0.5, 0.9] }],
  routes: [],
};

describe("BIM selection sets", () => {
  it("creates a stable deduplicated selection set", () => {
    const set = createSelectionSet([], "  Electrical   review  ", [
      { collection: "walls", id: "wall-1" },
      { collection: "walls", id: "wall-1" },
      { collection: "fixtures", id: "chair-1" },
    ], "2026-08-09T00:00:00.000Z");
    expect(set).toMatchObject({
      id: "selection-set-1",
      name: "Electrical review",
      selections: [
        { collection: "walls", id: "wall-1" },
        { collection: "fixtures", id: "chair-1" },
      ],
    });
    expect(selectionSetSummary(set!)).toBe("2 elements · 2 categories");
  });

  it("sanitizes persisted sets against the current graph", () => {
    expect(sanitizeSelectionSets([
      {
        id: "set-1",
        name: "Core",
        selections: [
          { collection: "walls", id: "wall-1" },
          { collection: "fixtures", id: "missing" },
        ],
        created_at: "now",
        updated_at: "now",
      },
      { id: "empty", name: "Empty", selections: [{ collection: "fixtures", id: "missing" }] },
      { id: "set-1", name: "Duplicate", selections: [{ collection: "walls", id: "wall-1" }] },
    ], graph)).toEqual([
      {
        id: "set-1",
        name: "Core",
        selections: [{ collection: "walls", id: "wall-1" }],
        created_at: "now",
        updated_at: "now",
      },
    ]);
  });

  it("generates an unused default name and preserves identity when renamed", () => {
    const existing: BimSelectionSet[] = [
      {
        id: "selection-set-1",
        name: "Selection set 2",
        selections: [{ collection: "walls", id: "wall-1" }],
        created_at: "old",
        updated_at: "old",
      },
    ];
    expect(defaultSelectionSetName(existing)).toBe("Selection set 3");
    expect(renameSelectionSet(existing, "selection-set-1", "  Exterior   shell  ", "new")[0]).toMatchObject({
      id: "selection-set-1",
      name: "Exterior shell",
      created_at: "old",
      updated_at: "new",
    });
  });
});
