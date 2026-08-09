import { describe, expect, it } from "vitest";

import { createBimClipboardBundle, planBimPaste } from "./bimClipboard";
import type { PlanGraph } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    project_id: "clipboard-test",
    levels: [
      { id: "L1", name: "Level 1" },
      { id: "L2", name: "Level 2" },
    ],
    rooms: [{ id: "room", level_id: "L1", name: "Office", polygon: [[-1, -1], [18, -1], [18, 10], [-1, 10]] }],
    walls: [
      { id: "wall-a", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.12, height_m: 3 },
      { id: "wall-b", level_id: "L1", from: [4, 0], to: [4, 4], thickness_m: 0.12, height_m: 3 },
    ],
    openings: [{ id: "door", level_id: "L1", type: "door", wall_id: "wall-a", center_m: [2, 0], width_m: 0.9, height_m: 2.1 }],
    fixtures: [{ id: "desk", level_id: "L1", type: "desk", room_id: "room", center_m: [2, 3], size_m: [1, 1, 0.75] }],
    routes: [],
    vertical_connections: [],
    constraints: [{
      id: "corner",
      level_id: "L1",
      type: "coincident",
      references: [
        { collection: "walls", entity_id: "wall-a", handle: "to" },
        { collection: "walls", entity_id: "wall-b", handle: "from" },
      ],
    }],
  };
}

describe("BIM clipboard", () => {
  it("captures hosted dependencies and remaps the complete pasted wall bundle", () => {
    const source = graph();
    const bundle = createBimClipboardBundle(source, [
      { collection: "walls", id: "wall-a" },
      { collection: "walls", id: "wall-b" },
    ]);
    expect(bundle.items.map((item) => item.collection)).toEqual([
      "walls", "walls", "openings", "constraints",
    ]);
    expect(bundle.included_selections).toContainEqual({ collection: "openings", id: "door" });

    const paste = planBimPaste(source, bundle, "L1", 0.05);
    expect(paste.valid).toBe(true);
    expect(paste.offset_m).not.toEqual([0, 0]);
    const walls = paste.items.filter((item) => item.collection === "walls");
    const opening = paste.items.find((item) => item.collection === "openings")?.entity;
    const constraint = paste.items.find((item) => item.collection === "constraints")?.entity;
    expect(opening?.wall_id).toBe(walls[0].entity.id);
    expect(constraint?.references).toEqual([
      { collection: "walls", entity_id: walls[0].entity.id, handle: "to" },
      { collection: "walls", entity_id: walls[1].entity.id, handle: "from" },
    ]);
    expect(paste.items.every((item) => item.entity.pasted_from_entity_id)).toBe(true);
  });

  it("finds a clear in-room placement and refreshes fixture containment", () => {
    const source = graph();
    const bundle = createBimClipboardBundle(source, [{ collection: "fixtures", id: "desk" }]);
    const paste = planBimPaste(source, bundle, "L1", 0.05);
    expect(paste.valid).toBe(true);
    expect(paste.items[0].entity.center_m).not.toEqual([2, 3]);
    expect(paste.items[0].entity.room_id).toBe("room");
  });

  it("pastes in place on another level and refuses an orphaned opening", () => {
    const source = graph();
    const wallBundle = createBimClipboardBundle(source, [{ collection: "walls", id: "wall-a" }]);
    const crossLevel = planBimPaste(source, wallBundle, "L2");
    expect(crossLevel.valid).toBe(true);
    expect(crossLevel.offset_m).toEqual([0, 0]);
    expect(crossLevel.items.every((item) => item.entity.level_id === "L2")).toBe(true);

    const openingOnly = createBimClipboardBundle(source, [{ collection: "openings", id: "door" }]);
    expect(planBimPaste(source, openingOnly, "L2")).toMatchObject({
      valid: false,
      reason: expect.stringContaining("requires its host wall"),
    });
  });
});
