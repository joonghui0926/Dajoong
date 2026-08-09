import type { PlanGraph, Selection } from "./types";
import { elementFootprint } from "./windowSelection";

export interface PlanViewBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PlanBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function fittedPlanView(
  bounds: PlanBounds,
  paddingRatio = 0.06,
): PlanViewBox {
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const height = Math.max(1, bounds.maxY - bounds.minY);
  const padding = Math.max(width, height) * paddingRatio;
  return {
    x: bounds.minX - padding,
    y: bounds.minY - padding,
    width: width + padding * 2,
    height: height + padding * 2,
  };
}

export function zoomPlanViewAt(
  view: PlanViewBox,
  anchor: [number, number],
  requestedFactor: number,
  minWidth: number,
  maxWidth: number,
): PlanViewBox {
  const safeFactor = Number.isFinite(requestedFactor) && requestedFactor > 0
    ? requestedFactor
    : 1;
  const nextWidth = Math.min(maxWidth, Math.max(minWidth, view.width * safeFactor));
  const factor = nextWidth / view.width;
  return {
    x: anchor[0] - (anchor[0] - view.x) * factor,
    y: anchor[1] - (anchor[1] - view.y) * factor,
    width: nextWidth,
    height: view.height * factor,
  };
}

export function panPlanViewByPixels(
  view: PlanViewBox,
  deltaX: number,
  deltaY: number,
  viewportWidth: number,
  viewportHeight: number,
): PlanViewBox {
  const worldPerPixel = Math.max(
    view.width / Math.max(1, viewportWidth),
    view.height / Math.max(1, viewportHeight),
  );
  return {
    ...view,
    x: view.x - deltaX * worldPerPixel,
    y: view.y - deltaY * worldPerPixel,
  };
}

export function viewForSelections(
  graph: PlanGraph,
  selections: Selection[],
  fallback: PlanViewBox,
  paddingRatio = 0.16,
): PlanViewBox {
  const points = selections.flatMap((selection) =>
    elementFootprint(graph, selection)?.points ?? [],
  );
  if (!points.length) return fallback;
  const minX = Math.min(...points.map(([x]) => x));
  const maxX = Math.max(...points.map(([x]) => x));
  const minY = Math.min(...points.map(([, y]) => y));
  const maxY = Math.max(...points.map(([, y]) => y));
  const rawWidth = Math.max(0.25, maxX - minX);
  const rawHeight = Math.max(0.25, maxY - minY);
  const padding = Math.max(rawWidth, rawHeight, 0.5) * paddingRatio;
  return {
    x: (minX + maxX) / 2 - rawWidth / 2 - padding,
    y: (minY + maxY) / 2 - rawHeight / 2 - padding,
    width: rawWidth + padding * 2,
    height: rawHeight + padding * 2,
  };
}

export function planZoomPercent(fitted: PlanViewBox, current: PlanViewBox): number {
  if (current.width <= 0) return 100;
  return Math.max(1, Math.round(fitted.width / current.width * 100));
}

export function planViewBoxValue(view: PlanViewBox): string {
  return `${view.x} ${view.y} ${view.width} ${view.height}`;
}
