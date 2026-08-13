import { describe, expect, it } from "vitest";

import { canRestoreModelCamera, type ModelCameraSnapshot } from "./modelCameraState";

const snapshot: ModelCameraSnapshot = {
  levelId: "L1",
  viewRevision: 4,
  position: [12, 8, 10],
  target: [4, 0.8, 3],
};

describe("3D model camera persistence", () => {
  it("restores the camera when model geometry changes within the same view", () => {
    expect(canRestoreModelCamera(snapshot, "L1", 4)).toBe(true);
  });

  it("refits after an explicit view command or a level change", () => {
    expect(canRestoreModelCamera(snapshot, "L1", 5)).toBe(false);
    expect(canRestoreModelCamera(snapshot, "L2", 4)).toBe(false);
  });
});
