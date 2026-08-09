import type { CollectionName, Selection } from "./types";

export const selectionFilterOptions: Array<{
  collection: Selection["collection"];
  label: string;
  detail: string;
}> = [
  { collection: "walls", label: "Walls", detail: "Architectural and structural walls" },
  { collection: "openings", label: "Openings", detail: "Doors and windows" },
  { collection: "rooms", label: "Rooms", detail: "Space boundaries and floor regions" },
  { collection: "fixtures", label: "Objects", detail: "Equipment, furniture, and devices" },
  { collection: "routes", label: "Systems", detail: "MEP and fire routes" },
  { collection: "vertical_connections", label: "Circulation", detail: "Stairs, shafts, risers, and lifts" },
  { collection: "dimensions", label: "Dimensions", detail: "Driving and reference dimensions" },
  { collection: "constraints", label: "Constraints", detail: "Persistent geometric relationships" },
];

const filterableCollections = new Set<CollectionName>(
  selectionFilterOptions.map((option) => option.collection),
);

export function sanitizeSelectionExclusions(value: unknown): CollectionName[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter(
    (candidate): candidate is CollectionName =>
      typeof candidate === "string" && filterableCollections.has(candidate as CollectionName),
  ))];
}

export function toggleSelectionExclusion(
  exclusions: CollectionName[],
  collection: CollectionName,
): CollectionName[] {
  if (!filterableCollections.has(collection)) return exclusions;
  return exclusions.includes(collection)
    ? exclusions.filter((item) => item !== collection)
    : [...exclusions, collection];
}

export function filterSelectableSelections(
  selections: Selection[],
  exclusions: CollectionName[],
): Selection[] {
  const excluded = new Set(exclusions);
  return selections.filter((selection) => !excluded.has(selection.collection));
}
