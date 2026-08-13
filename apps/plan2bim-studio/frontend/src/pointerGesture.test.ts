import { describe, expect, it } from "vitest";

import { exceedsDragThreshold } from "./pointerGesture";

describe("3D viewport pointer gestures", () => {
  it("keeps small pointer jitter eligible for object selection", () => {
    expect(exceedsDragThreshold({ x: 100, y: 100 }, { x: 103, y: 102 })).toBe(false);
  });

  it("classifies an orbit gesture as a drag before the click event", () => {
    expect(exceedsDragThreshold({ x: 100, y: 100 }, { x: 112, y: 106 })).toBe(true);
  });
});
