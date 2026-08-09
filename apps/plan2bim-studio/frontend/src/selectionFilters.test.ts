import { describe, expect, it } from "vitest";

import {
  filterSelectableSelections,
  sanitizeSelectionExclusions,
  toggleSelectionExclusion,
} from "./selectionFilters";

describe("selection filters", () => {
  it("sanitizes persisted filter state and removes duplicates", () => {
    expect(sanitizeSelectionExclusions(["walls", "fixtures", "walls", "invalid", 4])).toEqual([
      "walls",
      "fixtures",
    ]);
  });

  it("toggles only supported BIM collections", () => {
    expect(toggleSelectionExclusion(["walls"], "walls")).toEqual([]);
    expect(toggleSelectionExclusion([], "routes")).toEqual(["routes"]);
    expect(toggleSelectionExclusion([], "levels")).toEqual([]);
  });

  it("filters candidates without mutating the current selection array", () => {
    const selections = [
      { collection: "walls" as const, id: "W1" },
      { collection: "fixtures" as const, id: "F1" },
      { collection: "rooms" as const, id: "R1" },
    ];
    expect(filterSelectableSelections(selections, ["walls", "rooms"])).toEqual([
      { collection: "fixtures", id: "F1" },
    ]);
    expect(selections).toHaveLength(3);
  });
});
