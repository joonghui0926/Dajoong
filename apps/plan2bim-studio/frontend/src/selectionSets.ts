import { sanitizeEntityViewState } from "./editorViewState";
import type { PlanGraph, Selection } from "./types";

export interface BimSelectionSet {
  id: string;
  name: string;
  selections: Selection[];
  created_at: string;
  updated_at: string;
}

function cleanName(value: unknown): string {
  return typeof value === "string" ? value.trim().replace(/\s+/g, " ").slice(0, 64) : "";
}

export function sanitizeSelectionSets(value: unknown, graph?: PlanGraph): BimSelectionSet[] {
  if (!Array.isArray(value)) return [];
  const result: BimSelectionSet[] = [];
  const seenIds = new Set<string>();
  for (const candidate of value) {
    if (!candidate || typeof candidate !== "object") continue;
    const record = candidate as Record<string, unknown>;
    const id = typeof record.id === "string" ? record.id.trim() : "";
    const name = cleanName(record.name);
    const selections = sanitizeEntityViewState(record.selections, graph);
    if (!id || !name || !selections.length || seenIds.has(id)) continue;
    seenIds.add(id);
    const createdAt = typeof record.created_at === "string" ? record.created_at : "";
    const updatedAt = typeof record.updated_at === "string" ? record.updated_at : createdAt;
    result.push({
      id,
      name,
      selections,
      created_at: createdAt,
      updated_at: updatedAt,
    });
  }
  return result;
}

function nextSelectionSetId(existing: BimSelectionSet[]): string {
  const used = new Set(existing.map((item) => item.id));
  let index = existing.length + 1;
  while (used.has(`selection-set-${index}`)) index += 1;
  return `selection-set-${index}`;
}

export function defaultSelectionSetName(existing: BimSelectionSet[]): string {
  const used = new Set(existing.map((item) => item.name.toLowerCase()));
  let index = existing.length + 1;
  while (used.has(`selection set ${index}`)) index += 1;
  return `Selection set ${index}`;
}

export function createSelectionSet(
  existing: BimSelectionSet[],
  name: string,
  selections: Selection[],
  timestamp = new Date().toISOString(),
): BimSelectionSet | null {
  const cleanSelections = sanitizeEntityViewState(selections);
  if (!cleanSelections.length) return null;
  return {
    id: nextSelectionSetId(existing),
    name: cleanName(name) || defaultSelectionSetName(existing),
    selections: cleanSelections,
    created_at: timestamp,
    updated_at: timestamp,
  };
}

export function renameSelectionSet(
  sets: BimSelectionSet[],
  id: string,
  name: string,
  timestamp = new Date().toISOString(),
): BimSelectionSet[] {
  const nextName = cleanName(name);
  if (!nextName) return sets;
  return sets.map((item) => item.id === id
    ? { ...item, name: nextName, updated_at: timestamp }
    : item);
}

export function selectionSetSummary(set: BimSelectionSet): string {
  const categories = new Set(set.selections.map((selection) => selection.collection));
  return `${set.selections.length} element${set.selections.length === 1 ? "" : "s"} · ${categories.size} categor${categories.size === 1 ? "y" : "ies"}`;
}
