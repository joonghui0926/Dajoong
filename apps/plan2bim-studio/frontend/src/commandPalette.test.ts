import { describe, expect, it, vi } from "vitest";

import {
  nextCommandIndex,
  rankStudioCommands,
  recordRecentCommand,
  sanitizeRecentCommandIds,
  type StudioCommand,
} from "./commandPalette";

const command = (id: string, label: string, group = "Modify", aliases: string[] = []): StudioCommand => ({
  id,
  label,
  group,
  aliases,
  run: vi.fn(),
});

describe("command palette ranking", () => {
  const commands = [
    command("move", "Move selected elements by exact offset", "Modify", ["translate", "precision"]),
    command("wall", "Add wall", "Model", ["partition"]),
    command("assurance", "Open model assurance", "Review", ["quality", "verify"]),
    command("history", "Toggle edit history timeline", "History", ["undo timeline"]),
  ];

  it("surfaces recent commands first when the search field is empty", () => {
    expect(rankStudioCommands(commands, "", ["history", "move"]).map((item) => item.command.id)).toEqual([
      "history",
      "move",
      "wall",
      "assurance",
    ]);
  });

  it("matches domain aliases and forgiving subsequences", () => {
    expect(rankStudioCommands(commands, "quality", []).map((item) => item.command.id)).toEqual(["assurance"]);
    expect(rankStudioCommands(commands, "ad wl", []).map((item) => item.command.id)).toContain("wall");
  });

  it("deduplicates and bounds persisted recents", () => {
    expect(sanitizeRecentCommandIds(["move", "move", 2, "wall", "gone"], new Set(["move", "wall"]))).toEqual([
      "move",
      "wall",
    ]);
    expect(recordRecentCommand(["wall", "move"], "move")).toEqual(["move", "wall"]);
  });

  it("wraps keyboard navigation predictably", () => {
    expect(nextCommandIndex(-1, "next", 4)).toBe(0);
    expect(nextCommandIndex(3, "next", 4)).toBe(0);
    expect(nextCommandIndex(0, "previous", 4)).toBe(3);
    expect(nextCommandIndex(2, "first", 4)).toBe(0);
    expect(nextCommandIndex(2, "last", 4)).toBe(3);
    expect(nextCommandIndex(0, "next", 0)).toBe(-1);
  });
});
