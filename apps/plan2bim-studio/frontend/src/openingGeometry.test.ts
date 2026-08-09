import { describe, expect, it } from "vitest";

import {
  findAvailableOpeningPlacement,
  moveOpeningToPoint,
  openingFrame,
  resizeOpeningFromEdge,
  toggleDoorHanding,
  toggleDoorSwingSide,
  validateOpeningPlacement,
} from "./openingGeometry";
import type { OpeningEntity, WallEntity } from "./types";

const wall: WallEntity = {
  id: "wall-1",
  level_id: "L1",
  from: [0, 0],
  to: [10, 0],
  thickness_m: 0.12,
  height_m: 3,
};

const door: OpeningEntity = {
  id: "door-1",
  level_id: "L1",
  type: "door",
  wall_id: wall.id,
  center_m: [5, 0.2],
  width_m: 1,
  height_m: 2.1,
  handing: "start",
  swing_side: "positive",
};

describe("opening geometry", () => {
  it("orients the opening and swing in the host wall frame", () => {
    const frame = openingFrame(door, wall);
    expect(frame?.center).toEqual([5, 0]);
    expect(frame?.start).toEqual([4.5, 0]);
    expect(frame?.end).toEqual([5.5, 0]);
    expect(frame?.hinge).toEqual([4.5, 0]);
    expect(frame?.openLeafEnd).toEqual([4.5, 1]);
  });

  it("resizes from one edge while keeping the opposite edge fixed", () => {
    const resized = resizeOpeningFromEdge([3, 2], [6, 0], wall);
    expect(resized.valid).toBe(true);
    expect(resized.changes).toEqual({ center_m: [4.5, 0], width_m: 3, x_m: 4.5 });
  });

  it("rejects a degenerate door width", () => {
    expect(resizeOpeningFromEdge([5.05, 0], [5, 0], wall).reason).toBe("too_narrow");
  });

  it("flips semantic controls deterministically", () => {
    expect(toggleDoorHanding("start")).toBe("end");
    expect(toggleDoorHanding("end")).toBe("start");
    expect(toggleDoorSwingSide("positive")).toBe("negative");
    expect(toggleDoorSwingSide("negative")).toBe("positive");
    expect(toggleDoorHanding("unknown")).toBe("start");
    expect(toggleDoorSwingSide("unknown")).toBe("positive");
  });

  it("blocks overlap and placement beyond the host-wall extents", () => {
    const sibling = { ...door, id: "door-2", center_m: [7, 0] as [number, number] };
    expect(
      validateOpeningPlacement({ ...door, center_m: [6.2, 0] }, wall, [door, sibling]),
    ).toMatchObject({ valid: false, reason: "overlap", conflictId: "door-2" });
    expect(
      validateOpeningPlacement({ ...door, center_m: [0.25, 0] }, wall, [door]),
    ).toMatchObject({ valid: false, reason: "outside_wall" });
  });

  it("clamps movement to the wall and finds the largest free insertion span", () => {
    const moved = moveOpeningToPoint(door, wall, [door], [-5, 3]);
    expect(moved.valid).toBe(true);
    expect(moved.changes?.center_m).toEqual([0.5, 0]);

    const placement = findAvailableOpeningPlacement(
      wall,
      [{ ...door, id: "existing", center_m: [2, 0] }],
      0.9,
    );
    expect(placement.valid).toBe(true);
    expect(placement.changes?.x_m).toBeGreaterThan(5);
  });
});
