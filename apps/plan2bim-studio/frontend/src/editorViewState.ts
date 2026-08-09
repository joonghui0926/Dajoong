import { collections, findEntity } from "./graph";
import type { PlanGraph, Selection } from "./types";

export function selectionKey(selection: Selection): string {
  return `${selection.collection}\u0000${selection.id}`;
}

export function includesSelection(items: Selection[], selection: Selection): boolean {
  const key = selectionKey(selection);
  return items.some((item) => selectionKey(item) === key);
}

export function toggleSelectionState(items: Selection[], selection: Selection): Selection[] {
  const key = selectionKey(selection);
  return items.some((item) => selectionKey(item) === key)
    ? items.filter((item) => selectionKey(item) !== key)
    : [...items, selection];
}

export function sanitizeEntityViewState(value: unknown, graph?: PlanGraph): Selection[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  const result: Selection[] = [];
  for (const candidate of value) {
    if (!candidate || typeof candidate !== "object") continue;
    const record = candidate as Record<string, unknown>;
    if (
      typeof record.collection !== "string" ||
      !collections.includes(record.collection as Selection["collection"]) ||
      typeof record.id !== "string" ||
      !record.id
    ) {
      continue;
    }
    const selection: Selection = {
      collection: record.collection as Selection["collection"],
      id: record.id,
    };
    const key = selectionKey(selection);
    if (seen.has(key) || (graph && !findEntity(graph, selection))) continue;
    seen.add(key);
    result.push(selection);
  }
  return result;
}
