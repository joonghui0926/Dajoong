import type { CollectionName, Selection } from "./types";

const naturalCollator = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

export const modelTreeCollections: CollectionName[] = [
  "walls",
  "openings",
  "rooms",
  "fixtures",
  "routes",
  "vertical_connections",
  "constraints",
  "dimensions",
];

export function collapsedModelTree(): Record<string, boolean> {
  return Object.fromEntries(modelTreeCollections.map((collection) => [collection, false]));
}

export function expandedModelTree(
  counts: Partial<Record<CollectionName, number>>,
): Record<string, boolean> {
  return Object.fromEntries(
    modelTreeCollections.map((collection) => [collection, (counts[collection] ?? 0) > 0]),
  );
}

export function treeSectionIsExpanded(
  explicit: boolean | undefined,
  filterActive: boolean,
  itemCount: number,
): boolean {
  return itemCount > 0 && (Boolean(explicit) || filterActive);
}

export function compareModelTreeItems(
  left: { id: string },
  right: { id: string },
): number {
  return naturalCollator.compare(left.id, right.id);
}

/**
 * Produces a contiguous browser selection while keeping the clicked target last.
 * The last item is the key object for alignment and the primary property panel.
 */
export function modelTreeSelectionRange(
  collection: CollectionName,
  items: Array<{ id: string }>,
  anchorId: string | null,
  targetId: string,
): Selection[] {
  const targetIndex = items.findIndex((item) => item.id === targetId);
  const anchorIndex = anchorId
    ? items.findIndex((item) => item.id === anchorId)
    : -1;
  if (targetIndex < 0) return [];
  if (anchorIndex < 0) return [{ collection, id: targetId }];

  const start = Math.min(anchorIndex, targetIndex);
  const end = Math.max(anchorIndex, targetIndex);
  const range = items
    .slice(start, end + 1)
    .map((item) => ({ collection, id: item.id } as Selection));
  return [
    ...range.filter((selection) => selection.id !== targetId),
    { collection, id: targetId },
  ];
}
