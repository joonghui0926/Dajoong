import type { CorrectionOperation } from "./types";

export interface HistorySnapshotLike {
  operations: CorrectionOperation[];
}

export interface HistoryTimelineEntry {
  index: number;
  label: string;
  detail: string;
  operationCount: number;
  state: "past" | "current" | "future";
}

const reasonLabels: Record<string, string> = {
  property_edit: "Edit properties",
  manual_review: "Edit properties",
  batch_edit: "Edit selected properties",
  wall_property_edit: "Edit wall properties",
  room_boundary_property_edit: "Edit room boundary",
  batch_wall_edit: "Edit selected walls",
  batch_fixture_edit: "Edit selected objects",
  replace_fixture_family: "Replace object family",
  direct_manipulation: "Move element",
  direct_bim_manipulation: "Transform BIM elements",
  direct_wall_manipulation: "Edit wall network",
  direct_room_boundary: "Edit room boundary",
  keyboard_nudge: "Nudge selection",
  exact_move: "Move precisely",
  exact_translation: "Move precisely",
  exact_rotate: "Rotate precisely",
  exact_rotation: "Rotate precisely",
  "3d_gizmo_translation": "Move in 3D",
  "3d_gizmo_rotation": "Rotate in 3D",
  align_components: "Align components",
  distribute_components: "Distribute components",
  duplicate_component: "Duplicate components",
  paste_bim_clipboard: "Paste BIM package",
  copy_to_level: "Copy to level",
  linear_array: "Create linear array",
  linear_array_component: "Create linear array",
  mirror_pattern: "Mirror components",
  mirror_component: "Mirror components",
  opening_rehost: "Rehost opening",
  flip_door_handing: "Flip door handing",
  reverse_door_swing: "Reverse door swing",
  trim_extend_wall_corner: "Create wall corner",
  join_wall_endpoint: "Join wall endpoints",
  batch_review: "Accept selected elements",
  visual_review: "Accept element",
  batch_false_positive: "Delete selected elements",
  false_positive: "Delete element",
  missing_element: "Add element",
};

const collectionLabels: Record<string, string> = {
  walls: "walls",
  openings: "openings",
  rooms: "rooms",
  fixtures: "objects",
  routes: "building systems",
  vertical_connections: "vertical circulation",
  dimensions: "dimensions",
  constraints: "constraints",
  levels: "levels",
};

export function historyReasonLabel(reason: string): string {
  if (reasonLabels[reason]) return reasonLabels[reason];
  const words = reason.replaceAll("_", " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : "Edit model";
}

export function historyOperationDelta(
  previous: HistorySnapshotLike | null,
  current: HistorySnapshotLike,
): CorrectionOperation[] {
  if (!previous) return current.operations;
  if (current.operations.length > previous.operations.length) {
    return current.operations.slice(previous.operations.length);
  }
  const currentLast = current.operations.at(-1);
  const previousLast = previous.operations.at(-1);
  if (currentLast && JSON.stringify(currentLast) !== JSON.stringify(previousLast)) {
    return [currentLast];
  }
  return [];
}

export function buildHistoryTimeline(
  past: HistorySnapshotLike[],
  present: HistorySnapshotLike,
  future: HistorySnapshotLike[],
): HistoryTimelineEntry[] {
  const snapshots = [...past, present, ...future];
  const currentIndex = past.length;
  return snapshots.map((snapshot, index) => {
    const delta = historyOperationDelta(index ? snapshots[index - 1] : null, snapshot);
    const reasons = [...new Set(delta.map((operation) => operation.reason))];
    const collections = [...new Set(delta.map((operation) => collectionLabels[operation.collection] ?? operation.collection))];
    const isInitial = index === 0;
    const label = isInitial
      ? snapshot.operations.length
        ? "Recovered model"
        : "Imported model"
      : reasons.length === 1
        ? historyReasonLabel(reasons[0])
        : delta.length
          ? "Edit BIM package"
          : "Model state";
    const detail = isInitial
      ? snapshot.operations.length
        ? `${snapshot.operations.length} audited changes`
        : "Source conversion"
      : delta.length === 1
        ? `${collectionLabels[delta[0].collection] ?? delta[0].collection} · ${delta[0].entity_id}`
        : `${delta.length} related elements${collections.length ? ` · ${collections.join(", ")}` : ""}`;
    return {
      index,
      label,
      detail,
      operationCount: snapshot.operations.length,
      state: index < currentIndex ? "past" : index === currentIndex ? "current" : "future",
    };
  });
}

export function historySnapshots<T>(past: T[], present: T, future: T[]): T[] {
  return [...past, present, ...future];
}
