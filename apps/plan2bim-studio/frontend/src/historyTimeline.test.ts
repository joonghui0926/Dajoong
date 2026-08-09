import { describe, expect, it } from "vitest";

import { buildHistoryTimeline, historyOperationDelta, historyReasonLabel } from "./historyTimeline";
import type { CorrectionOperation } from "./types";

function operation(id: string, reason: string, entityId = id): CorrectionOperation {
  return {
    id,
    action: "update",
    collection: "walls",
    entity_id: entityId,
    changes: { height_m: 3.2 },
    reason,
  };
}

describe("Fusion-style edit history", () => {
  it("labels audited BIM commands with product language", () => {
    expect(historyReasonLabel("paste_bim_clipboard")).toBe("Paste BIM package");
    expect(historyReasonLabel("custom_floor_finish")).toBe("Custom floor finish");
  });

  it("builds chronological past, current, and future entries", () => {
    const imported = { operations: [] };
    const moved = { operations: [operation("op-1", "exact_move", "wall-1")] };
    const pasted = {
      operations: [
        ...moved.operations,
        operation("op-2", "paste_bim_clipboard", "wall-2"),
        operation("op-3", "paste_bim_clipboard", "wall-3"),
      ],
    };
    const entries = buildHistoryTimeline([imported], moved, [pasted]);
    expect(entries.map((entry) => [entry.label, entry.state])).toEqual([
      ["Imported model", "past"],
      ["Move precisely", "current"],
      ["Paste BIM package", "future"],
    ]);
    expect(entries[2].detail).toContain("2 related elements");
  });

  it("treats a coalesced property operation as a new visible state", () => {
    const previous = { operations: [operation("op-1", "property_edit")] };
    const current = {
      operations: [{ ...operation("op-1", "property_edit"), changes: { height_m: 3.4 } }],
    };
    expect(historyOperationDelta(previous, current)).toEqual([current.operations[0]]);
  });

  it("describes a recovered session even when its earlier snapshots are unavailable", () => {
    const entries = buildHistoryTimeline([], { operations: [operation("op-1", "property_edit")] }, []);
    expect(entries[0]).toMatchObject({
      label: "Recovered model",
      detail: "1 audited changes",
      state: "current",
    });
  });
});
