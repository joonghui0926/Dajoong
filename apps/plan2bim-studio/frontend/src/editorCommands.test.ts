import { describe, expect, it } from "vitest";

import {
  alignmentChanges,
  distributionChanges,
  cornerWallChanges,
  joinWallEndpointChanges,
  offsetChanges,
} from "./editorCommands";
import type { BaseEntity, WallEntity } from "./types";

const fixtures: BaseEntity[] = [
  { id: "a", center_m: [1, 1], size_m: [2, 1, 1] },
  { id: "b", center_m: [4, 3], size_m: [1, 1, 1] },
  { id: "c", center_m: [9, 5], size_m: [2, 2, 1] },
];

describe("precision editor commands", () => {
  it("aligns component edges to the last-selected key object", () => {
    expect(alignmentChanges(fixtures, "left")).toEqual({
      a: { center_m: [9, 1] },
      b: { center_m: [8.5, 3] },
    });
  });

  it("distributes centers without moving the two anchors", () => {
    expect(distributionChanges(fixtures, "horizontal")).toEqual({
      b: { center_m: [5, 3] },
    });
  });

  it("distributes equal clear gaps and uses rotated plan footprints", () => {
    const varied: BaseEntity[] = [
      { id: "left", center_m: [1, 0], size_m: [2, 1, 1] },
      { id: "middle", center_m: [3, 0], size_m: [4, 1, 1], yaw_deg: 90 },
      { id: "right", center_m: [10, 0], size_m: [2, 1, 1] },
    ];
    expect(distributionChanges(varied, "horizontal")).toEqual({
      middle: { center_m: [5.5, 0] },
    });
    expect(alignmentChanges(varied, "left")).toEqual({
      left: { center_m: [10, 0] },
      middle: { center_m: [9.5, 0] },
    });
  });

  it("does not create no-op changes for an already arranged selection", () => {
    expect(alignmentChanges([
      { id: "a", center_m: [2, 4], size_m: [1, 1, 1] },
      { id: "key", center_m: [6, 4], size_m: [1, 1, 1] },
    ], "center-y")).toEqual({});
    expect(distributionChanges([
      { id: "a", center_m: [1, 0], size_m: [1, 1, 1] },
      { id: "b", center_m: [3, 0], size_m: [1, 1, 1] },
      { id: "c", center_m: [5, 0], size_m: [1, 1, 1] },
    ], "horizontal")).toEqual({});
  });

  it("nudges all positioned entities by one deterministic delta", () => {
    expect(offsetChanges(fixtures.slice(0, 2), [0.05, -0.1])).toEqual({
      a: { center_m: [1.05, 0.9] },
      b: { center_m: [4.05, 2.9] },
    });
  });

  it("joins the nearest pair of wall endpoints at one shared point", () => {
    const walls: WallEntity[] = [
      { id: "w1", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.1, height_m: 3 },
      { id: "w2", level_id: "L1", from: [4.2, 0.2], to: [4.2, 4], thickness_m: 0.1, height_m: 3 },
    ];
    expect(joinWallEndpointChanges(walls)).toEqual({
      w1: { to: [4.1, 0.1] },
      w2: { from: [4.1, 0.1] },
    });
  });

  it("trims or extends two wall axes to their exact corner", () => {
    const walls: WallEntity[] = [
      { id: "w1", level_id: "L1", from: [0, 0], to: [3.8, 0], thickness_m: 0.1, height_m: 3 },
      { id: "w2", level_id: "L1", from: [4, 0.2], to: [4, 4], thickness_m: 0.1, height_m: 3 },
    ];
    expect(cornerWallChanges(walls)).toEqual({
      w1: { to: [4, 0] },
      w2: { from: [4, 0] },
    });
  });
});
