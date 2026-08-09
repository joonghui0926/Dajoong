import { describe, expect, it } from "vitest";

import {
  fittedPlanView,
  panPlanViewByPixels,
  planZoomPercent,
  viewForSelections,
  zoomPlanViewAt,
} from "./planNavigation";
import type { PlanGraph } from "./types";

const graph: PlanGraph = {
  schema_version: "1",
  levels: [{ id: "L1", name: "Level 1" }],
  walls: [{ id: "W1", level_id: "L1", from: [0, 0], to: [6, 0], thickness_m: 0.2, height_m: 3 }],
  rooms: [],
  openings: [],
  fixtures: [{ id: "F1", level_id: "L1", type: "desk", center_m: [3, 4], size_m: [2, 1, 0.8], yaw_deg: 90 }],
  routes: [],
};

describe("plan navigation", () => {
  it("fits plan bounds with proportional padding", () => {
    expect(fittedPlanView({ minX: 0, minY: 0, maxX: 10, maxY: 5 }, 0.1)).toEqual({
      x: -1,
      y: -1,
      width: 12,
      height: 7,
    });
  });

  it("keeps the cursor anchor stationary while zooming and respects limits", () => {
    const view = { x: 0, y: 0, width: 10, height: 8 };
    expect(zoomPlanViewAt(view, [2, 3], 0.5, 1, 40)).toEqual({
      x: 1,
      y: 1.5,
      width: 5,
      height: 4,
    });
    expect(zoomPlanViewAt(view, [2, 3], 0.001, 2, 40).width).toBe(2);
  });

  it("pans in screen space at the active world scale", () => {
    expect(panPlanViewByPixels(
      { x: 0, y: 0, width: 10, height: 5 },
      100,
      -50,
      1000,
      500,
    )).toEqual({ x: -1, y: 0.5, width: 10, height: 5 });
  });

  it("fits selected semantic footprints instead of only their centers", () => {
    const fitted = fittedPlanView({ minX: 0, minY: 0, maxX: 10, maxY: 10 });
    const view = viewForSelections(graph, [{ collection: "fixtures", id: "F1" }], fitted, 0);
    expect(view.x).toBeCloseTo(2.5);
    expect(view.y).toBeCloseTo(3);
    expect(view.width).toBeCloseTo(1);
    expect(view.height).toBeCloseTo(2);
  });

  it("reports zoom relative to Fit All", () => {
    expect(planZoomPercent(
      { x: 0, y: 0, width: 12, height: 8 },
      { x: 3, y: 2, width: 3, height: 2 },
    )).toBe(400);
  });
});
