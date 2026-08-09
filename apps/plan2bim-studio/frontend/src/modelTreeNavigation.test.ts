import { describe, expect, it } from "vitest";

import {
  collapsedModelTree,
  compareModelTreeItems,
  expandedModelTree,
  modelTreeSelectionRange,
  treeSectionIsExpanded,
} from "./modelTreeNavigation";

describe("model browser navigation", () => {
  it("starts dense model categories collapsed and expands populated categories on demand", () => {
    expect(collapsedModelTree()).toMatchObject({ walls: false, openings: false, rooms: false });
    expect(expandedModelTree({ walls: 52, rooms: 0 })).toMatchObject({ walls: true, rooms: false });
    expect(treeSectionIsExpanded(false, true, 4)).toBe(true);
    expect(treeSectionIsExpanded(true, false, 0)).toBe(false);
  });

  it("sorts generated numeric IDs naturally for predictable range selection", () => {
    const items = [{ id: "L1:wall:10" }, { id: "L1:wall:2" }, { id: "L1:wall:1" }];
    expect(items.sort(compareModelTreeItems).map((item) => item.id)).toEqual([
      "L1:wall:1",
      "L1:wall:2",
      "L1:wall:10",
    ]);
  });

  it("selects an inclusive forward range and keeps the clicked target as key object", () => {
    expect(modelTreeSelectionRange(
      "walls",
      [{ id: "w1" }, { id: "w2" }, { id: "w3" }, { id: "w4" }],
      "w1",
      "w3",
    )).toEqual([
      { collection: "walls", id: "w1" },
      { collection: "walls", id: "w2" },
      { collection: "walls", id: "w3" },
    ]);
  });

  it("keeps a reverse range contiguous while making the clicked target primary", () => {
    expect(modelTreeSelectionRange(
      "openings",
      [{ id: "o1" }, { id: "o2" }, { id: "o3" }, { id: "o4" }],
      "o4",
      "o2",
    )).toEqual([
      { collection: "openings", id: "o3" },
      { collection: "openings", id: "o4" },
      { collection: "openings", id: "o2" },
    ]);
  });

  it("falls back to the clicked row if the previous anchor is filtered out", () => {
    expect(modelTreeSelectionRange("rooms", [{ id: "r2" }], "r1", "r2"))
      .toEqual([{ collection: "rooms", id: "r2" }]);
  });
});
