export interface PointerPoint {
  x: number;
  y: number;
}

export const MODEL_SELECTION_DRAG_THRESHOLD_PX = 5;

export function exceedsDragThreshold(
  start: PointerPoint,
  current: PointerPoint,
  threshold = MODEL_SELECTION_DRAG_THRESHOLD_PX,
): boolean {
  return Math.hypot(current.x - start.x, current.y - start.y) >= threshold;
}
