import { findEntity } from "./graph";
import { selectionKey } from "./editorViewState";
import type { PlanGraph, Selection } from "./types";

export interface RelatedSelectionGroup {
  id: string;
  label: string;
  selections: Selection[];
}

function group(
  graph: PlanGraph,
  id: string,
  label: string,
  selections: Selection[],
): RelatedSelectionGroup | null {
  const seen = new Set<string>();
  const valid = selections.filter((selection) => {
    const key = selectionKey(selection);
    if (seen.has(key) || !findEntity(graph, selection)) return false;
    seen.add(key);
    return true;
  });
  return valid.length ? { id, label, selections: valid } : null;
}

function constrainedWallChain(graph: PlanGraph, wallId: string): Selection[] {
  const queue = [wallId];
  const seen = new Set<string>();
  while (queue.length) {
    const current = queue.shift();
    if (!current || seen.has(current)) continue;
    seen.add(current);
    for (const constraint of graph.constraints ?? []) {
      if (!constraint.references.some((reference) => reference.entity_id === current)) continue;
      constraint.references.forEach((reference) => queue.push(reference.entity_id));
    }
  }
  return [...seen].map((id) => ({ collection: "walls" as const, id }));
}

export function relatedSelectionGroups(
  graph: PlanGraph,
  selection: Selection,
): RelatedSelectionGroup[] {
  const entity = findEntity(graph, selection);
  if (!entity) return [];
  const groups: Array<RelatedSelectionGroup | null> = [];

  if (selection.collection === "openings") {
    const wallId = typeof entity.wall_id === "string" ? entity.wall_id : "";
    groups.push(group(graph, "host-wall", "Select host wall", [
      { collection: "walls", id: wallId },
    ]));
    const siblings = graph.openings
      .filter((opening) => opening.wall_id === wallId)
      .map((opening) => ({ collection: "openings" as const, id: opening.id }));
    if (siblings.length > 1) {
      groups.push(group(graph, "wall-openings", `Select ${siblings.length} openings on this wall`, siblings));
    }
  }

  if (selection.collection === "walls") {
    const hosted = graph.openings
      .filter((opening) => opening.wall_id === selection.id)
      .map((opening) => ({ collection: "openings" as const, id: opening.id }));
    if (hosted.length) {
      groups.push(group(graph, "hosted-openings", `Select ${hosted.length} hosted opening${hosted.length === 1 ? "" : "s"}`, hosted));
    }
    const chain = constrainedWallChain(graph, selection.id);
    if (chain.length > 1) {
      groups.push(group(graph, "constraint-chain", `Select ${chain.length} constrained walls`, chain));
    }
    const roomId = typeof entity.room_id === "string" ? entity.room_id : "";
    if (roomId) {
      groups.push(group(graph, "wall-room", "Select assigned room", [
        { collection: "rooms", id: roomId },
      ]));
    }
  }

  if (selection.collection === "fixtures") {
    const roomId = typeof entity.room_id === "string" ? entity.room_id : "";
    if (roomId) {
      groups.push(group(graph, "fixture-room", "Select containing room", [
        { collection: "rooms", id: roomId },
      ]));
      const roomObjects = graph.fixtures
        .filter((fixture) => fixture.room_id === roomId)
        .map((fixture) => ({ collection: "fixtures" as const, id: fixture.id }));
      if (roomObjects.length > 1) {
        groups.push(group(graph, "room-objects", `Select ${roomObjects.length} objects in this room`, roomObjects));
      }
    }
  }

  if (selection.collection === "rooms") {
    const roomObjects = graph.fixtures
      .filter((fixture) => fixture.room_id === selection.id)
      .map((fixture) => ({ collection: "fixtures" as const, id: fixture.id }));
    if (roomObjects.length) {
      groups.push(group(graph, "contained-objects", `Select ${roomObjects.length} contained object${roomObjects.length === 1 ? "" : "s"}`, roomObjects));
    }
    const assignedWalls = graph.walls
      .filter((wall) => wall.room_id === selection.id)
      .map((wall) => ({ collection: "walls" as const, id: wall.id }));
    if (assignedWalls.length) {
      groups.push(group(graph, "room-walls", `Select ${assignedWalls.length} assigned wall${assignedWalls.length === 1 ? "" : "s"}`, assignedWalls));
    }
  }

  if (selection.collection === "constraints") {
    const references = Array.isArray(entity.references)
      ? entity.references as Array<{ entity_id?: unknown }>
      : [];
    const walls = references
      .filter((reference) => typeof reference.entity_id === "string")
      .map((reference) => ({ collection: "walls" as const, id: String(reference.entity_id) }));
    groups.push(group(graph, "constrained-walls", `Select ${walls.length} constrained wall${walls.length === 1 ? "" : "s"}`, walls));
  }

  if (selection.collection === "routes") {
    const systemId = typeof entity.system_id === "string" ? entity.system_id : "";
    if (systemId) {
      const systemRoutes = graph.routes
        .filter((route) => route.system_id === systemId)
        .map((route) => ({ collection: "routes" as const, id: route.id }));
      if (systemRoutes.length > 1) {
        groups.push(group(graph, "system-routes", `Select ${systemRoutes.length} routes in this system`, systemRoutes));
      }
    }
  }

  return groups.filter((item): item is RelatedSelectionGroup => item !== null);
}
