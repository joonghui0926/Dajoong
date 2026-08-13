import { beforeAll, describe, expect, it, vi } from "vitest";

import type { SessionAction, SessionState } from "./Studio";
import type { PlanGraph } from "./types";

let initialSessionState: SessionState;
let studioSessionReducer: (state: SessionState, action: SessionAction) => SessionState;
let actionSelections: (action: SessionAction) => Array<{ collection: string; id: string }>;
let actionMutationSelections: (
  action: SessionAction,
  graph: PlanGraph | null,
) => Array<{ collection: string; id: string }>;
let shouldRecoverStudioSession: typeof import("./Studio").shouldRecoverStudioSession;
let sessionSchemaVersion: number;

beforeAll(async () => {
  vi.stubGlobal("window", { location: { origin: "http://localhost" } });
  const studio = await import("./Studio");
  initialSessionState = studio.initialSessionState;
  studioSessionReducer = studio.studioSessionReducer;
  actionSelections = studio.actionSelections;
  actionMutationSelections = studio.actionMutationSelections;
  shouldRecoverStudioSession = studio.shouldRecoverStudioSession;
  sessionSchemaVersion = studio.STUDIO_SESSION_SCHEMA_VERSION;
});

const graph: PlanGraph = {
  schema_version: "buili.plan-graph.v2",
  levels: [{ id: "L1", name: "Level 1" }],
  walls: [],
  rooms: [],
  openings: [],
  fixtures: [
    {
      id: "chair-1",
      level_id: "L1",
      type: "chair",
      center_m: [1, 1],
      size_m: [0.5, 0.5, 0.9],
    },
  ],
  routes: [],
  vertical_connections: [],
};

describe("Studio edit transactions", () => {
  it("refreshes untouched legacy demos but preserves unsynced edits", () => {
    expect(shouldRecoverStudioSession({ graph, operations: [] })).toBe(false);
    expect(shouldRecoverStudioSession({ graph, operations: [{
      id: "legacy-edit",
      collection: "fixtures",
      entity_id: "chair-1",
      action: "update",
      reason: "test",
      changes: {},
    }] })).toBe(true);
    expect(shouldRecoverStudioSession({
      schema_version: sessionSchemaVersion,
      graph,
      operations: [],
    })).toBe(true);
  });

  it("refreshes the bundled sample even when an old cache has the current schema", () => {
    const bundledGraph = {
      ...graph,
      provenance: {
        source_image_sha256: "2c9092e12dd22207b1ab41d7660534d73f4b341121d279685307ba20597da5d6",
      },
    };
    expect(shouldRecoverStudioSession({
      schema_version: sessionSchemaVersion,
      graph: bundledGraph,
      operations: [],
    })).toBe(false);
    expect(shouldRecoverStudioSession({
      schema_version: sessionSchemaVersion,
      graph: bundledGraph,
      operations: [{
        id: "kept-user-edit",
        collection: "fixtures",
        entity_id: "chair-1",
        action: "update",
        reason: "user edit",
        changes: {},
      }],
    })).toBe(true);
  });

  it("identifies every existing entity touched by a mutation command", () => {
    const selections = [
      { collection: "walls" as const, id: "wall-1" },
      { collection: "walls" as const, id: "wall-2" },
    ];
    expect(actionSelections({
      type: "batchTransform",
      selections,
      changesById: { "wall-1": { from: [1, 1] }, "wall-2": { to: [2, 2] } },
      reason: "test",
    })).toEqual(selections);
    expect(actionSelections({
      type: "edit",
      selection: selections[0],
      changes: { height_m: 3.2 },
    })).toEqual([selections[0]]);
    expect(actionSelections({
      type: "add",
      collection: "walls",
      entity: { id: "wall-new" },
    })).toEqual([]);
  });

  it("expands a wall endpoint edit to constrained walls and hosted openings", () => {
    const constrainedGraph: PlanGraph = {
      ...graph,
      fixtures: [],
      walls: [
        { id: "wall-1", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.12, height_m: 3 },
        { id: "wall-2", level_id: "L1", from: [4, 0], to: [4, 4], thickness_m: 0.12, height_m: 3 },
      ],
      openings: [
        { id: "door-1", level_id: "L1", type: "door", wall_id: "wall-2", center_m: [4, 2], width_m: 0.9, height_m: 2.1 },
      ],
      constraints: [
        {
          id: "constraint-1",
          level_id: "L1",
          type: "coincident",
          references: [
            { collection: "walls", entity_id: "wall-1", handle: "to" },
            { collection: "walls", entity_id: "wall-2", handle: "from" },
          ],
        },
      ],
    };
    expect(actionMutationSelections({
      type: "edit",
      selection: { collection: "walls", id: "wall-1" },
      changes: { to: [5, 0] },
    }, constrainedGraph)).toEqual([
      { collection: "walls", id: "wall-1" },
      { collection: "walls", id: "wall-2" },
      { collection: "openings", id: "door-1" },
    ]);
  });

  it("commits a full pointer gesture as one audited operation and one undo", () => {
    const selection = { collection: "fixtures" as const, id: "chair-1" };
    let state: SessionState = studioSessionReducer(initialSessionState, {
      type: "load",
      graph,
    });
    state = studioSessionReducer(state, { type: "beginGesture", selection });
    state = studioSessionReducer(state, {
      type: "previewGesture",
      selection,
      changes: { center_m: [2, 2] },
    });
    state = studioSessionReducer(state, {
      type: "previewGesture",
      selection,
      changes: { center_m: [3, 4] },
    });
    state = studioSessionReducer(state, { type: "commitGesture" });

    expect(state.present?.graph.fixtures[0].center_m).toEqual([3, 4]);
    expect(state.present?.operations).toHaveLength(1);
    expect(state.present?.operations[0]).toMatchObject({
      action: "update",
      reason: "direct_manipulation",
      changes: { center_m: [3, 4] },
    });
    expect(state.past).toHaveLength(1);

    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.fixtures[0].center_m).toEqual([1, 1]);
  });

  it("records a key-object arrangement as one undo transaction", () => {
    const arrangeGraph: PlanGraph = {
      ...graph,
      fixtures: [
        graph.fixtures[0],
        { ...graph.fixtures[0], id: "chair-key", center_m: [4, 3] },
      ],
    };
    const selections = [
      { collection: "fixtures" as const, id: "chair-1" },
      { collection: "fixtures" as const, id: "chair-key" },
    ];
    let state = studioSessionReducer(initialSessionState, { type: "load", graph: arrangeGraph });
    state = studioSessionReducer(state, {
      type: "batchTransform",
      selections,
      changesById: { "chair-1": { center_m: [1, 3] } },
      reason: "align_center-y",
    });
    expect(state.present?.graph.fixtures[0].center_m).toEqual([1, 3]);
    expect(state.present?.graph.fixtures[1].center_m).toEqual([4, 3]);
    expect(state.present?.operations).toHaveLength(1);
    expect(state.present?.operations[0].reason).toBe("align_center-y");
    expect(state.past).toHaveLength(1);

    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.fixtures[0].center_m).toEqual([1, 1]);
    expect(state.present?.graph.fixtures[1].center_m).toEqual([4, 3]);
  });

  it("adds a remapped BIM clipboard bundle as one audited undo transaction", () => {
    let state = studioSessionReducer(initialSessionState, { type: "load", graph });
    state = studioSessionReducer(state, {
      type: "addMany",
      reason: "paste_bim_clipboard",
      items: [
        {
          collection: "walls",
          entity: { id: "wall:paste", level_id: "L1", from: [0, 0], to: [3, 0], thickness_m: 0.12, height_m: 3 },
        },
        {
          collection: "openings",
          entity: { id: "door:paste", level_id: "L1", type: "door", wall_id: "wall:paste", center_m: [1.5, 0], width_m: 0.9, height_m: 2.1 },
        },
      ],
    });
    expect(state.present?.graph.walls.map((wall) => wall.id)).toContain("wall:paste");
    expect(state.present?.graph.openings[0].wall_id).toBe("wall:paste");
    expect(state.present?.operations).toHaveLength(2);
    expect(state.present?.operations.every((operation) => operation.reason === "paste_bim_clipboard")).toBe(true);
    expect(state.past).toHaveLength(1);

    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.walls).toEqual([]);
    expect(state.present?.graph.openings).toEqual([]);
  });

  it("audits every dependent wall edit while preserving one-step undo", () => {
    const wallGraph: PlanGraph = {
      ...graph,
      fixtures: [],
      walls: [
        { id: "wall-1", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.12, height_m: 3 },
      ],
      openings: [
        { id: "door-1", level_id: "L1", type: "door", wall_id: "wall-1", center_m: [2, 0], width_m: 0.9, height_m: 2.1 },
      ],
      rooms: [
        { id: "room-1", level_id: "L1", name: "Office", polygon: [[0, 0], [4, 0], [4, 3], [0, 3]] },
      ],
    };
    const selection = { collection: "walls" as const, id: "wall-1" };
    let state = studioSessionReducer(initialSessionState, { type: "load", graph: wallGraph });
    state = studioSessionReducer(state, { type: "beginGesture", selection });
    state = studioSessionReducer(state, {
      type: "previewTransform",
      selection,
      reason: "direct_wall_manipulation",
      entries: [
        { selection, changes: { to: [5, 0] } },
        { selection: { collection: "openings", id: "door-1" }, changes: { center_m: [2.5, 0] } },
        {
          selection: { collection: "rooms", id: "room-1" },
          changes: { polygon: [[0, 0], [5, 0], [4, 3], [0, 3]] },
        },
      ],
    });

    expect(state.present?.graph.walls[0].to).toEqual([5, 0]);
    expect(state.present?.graph.openings[0].center_m).toEqual([2.5, 0]);
    expect(state.present?.operations).toEqual([]);

    state = studioSessionReducer(state, { type: "commitGesture" });
    expect(state.present?.operations).toHaveLength(3);
    expect(state.present?.operations.map((operation) => operation.collection)).toEqual([
      "walls",
      "openings",
      "rooms",
    ]);
    expect(state.present?.operations.every((operation) => operation.reason === "direct_wall_manipulation")).toBe(true);
    expect(state.past).toHaveLength(1);

    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.walls[0].to).toEqual([4, 0]);
    expect(state.present?.graph.openings[0].center_m).toEqual([2, 0]);
    expect(state.present?.graph.rooms[0].polygon).toEqual([[0, 0], [4, 0], [4, 3], [0, 3]]);
  });

  it("commits room containment updates as one named correction transaction", () => {
    const roomGraph: PlanGraph = {
      ...graph,
      rooms: [{ id: "room-1", level_id: "L1", name: "Office", polygon: [[0, 0], [4, 0], [4, 4], [0, 4]] }],
      fixtures: [{
        ...graph.fixtures[0],
        room_id: "room-1",
        center_m: [3.5, 2],
      }],
    };
    const selection = { collection: "rooms" as const, id: "room-1" };
    let state = studioSessionReducer(initialSessionState, { type: "load", graph: roomGraph });
    state = studioSessionReducer(state, { type: "beginGesture", selection });
    state = studioSessionReducer(state, {
      type: "previewTransform",
      selection,
      reason: "direct_room_boundary",
      entries: [
        { selection, changes: { polygon: [[0, 0], [3, 0], [3, 4], [0, 4]] } },
        {
          selection: { collection: "fixtures", id: "chair-1" },
          changes: { room_id: null, review_state: "review_required" },
        },
      ],
    });
    state = studioSessionReducer(state, { type: "commitGesture" });
    expect(state.present?.operations).toHaveLength(2);
    expect(state.present?.operations.every((operation) => operation.reason === "direct_room_boundary")).toBe(true);
    expect(state.present?.graph.fixtures[0].room_id).toBeNull();
    expect(state.past).toHaveLength(1);

    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.rooms[0].polygon).toEqual([[0, 0], [4, 0], [4, 4], [0, 4]]);
    expect(state.present?.graph.fixtures[0].room_id).toBe("room-1");
  });

  it("creates a persistent wall constraint and removes the whole command with one undo", () => {
    const wallGraph: PlanGraph = {
      ...graph,
      fixtures: [],
      walls: [
        { id: "wall-1", level_id: "L1", from: [0, 0], to: [3.8, 0], thickness_m: 0.12, height_m: 3 },
        { id: "wall-2", level_id: "L1", from: [4, 0.2], to: [4, 4], thickness_m: 0.12, height_m: 3 },
      ],
      constraints: [],
    };
    const selections = [
      { collection: "walls" as const, id: "wall-1" },
      { collection: "walls" as const, id: "wall-2" },
    ];
    let state = studioSessionReducer(initialSessionState, { type: "load", graph: wallGraph });
    state = studioSessionReducer(state, {
      type: "constrainWalls",
      selections,
      changesById: { "wall-1": { to: [4, 0] }, "wall-2": { from: [4, 0] } },
      constraint: {
        id: "constraint-1",
        level_id: "L1",
        type: "coincident",
        references: [
          { collection: "walls", entity_id: "wall-1", handle: "to" },
          { collection: "walls", entity_id: "wall-2", handle: "from" },
        ],
      },
      reason: "trim_extend_wall_corner",
    });
    expect(state.present?.graph.constraints).toHaveLength(1);
    expect(state.present?.graph.walls[0].to).toEqual([4, 0]);
    expect(state.present?.operations).toHaveLength(3);
    expect(state.past).toHaveLength(1);

    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.constraints).toEqual([]);
    expect(state.present?.graph.walls[0].to).toEqual([3.8, 0]);
    expect(state.present?.graph.walls[1].from).toEqual([4, 0.2]);
  });

  it("can discard a recovered patch and restore the immutable imported graph", () => {
    const changed = structuredClone(graph);
    changed.fixtures[0].center_m = [8, 9];
    let state = studioSessionReducer(initialSessionState, {
      type: "recover",
      source: graph,
      graph: changed,
      operations: [
        {
          id: "edit-1",
          action: "update",
          collection: "fixtures",
          entity_id: "chair-1",
          changes: { center_m: [8, 9] },
          reason: "property_edit",
        },
      ],
    });
    state = studioSessionReducer(state, { type: "resetToSource" });
    expect(state.present?.graph.fixtures[0].center_m).toEqual([1, 1]);
    expect(state.present?.operations).toEqual([]);
    expect(state.past).toEqual([]);
    expect(state.future).toEqual([]);
  });

  it("adds a dependency package as one undoable cross-level transaction", () => {
    const building: PlanGraph = {
      ...graph,
      levels: [...graph.levels, { id: "L2", name: "Level 2", elevation_m: 3 }],
    };
    let state = studioSessionReducer(initialSessionState, { type: "load", graph: building });
    state = studioSessionReducer(state, {
      type: "addMany",
      items: [
        {
          collection: "walls",
          entity: {
            id: "L2:wall:1:copy",
            level_id: "L2",
            from: [0, 0],
            to: [4, 0],
            height_m: 3,
            thickness_m: 0.12,
            copied_from_entity_id: "L1:wall:1",
            confidence: 1,
            review_state: "accepted",
          },
        },
        {
          collection: "openings",
          entity: {
            id: "L2:door:1:copy",
            level_id: "L2",
            type: "door",
            wall_id: "L2:wall:1:copy",
            center_m: [2, 0],
            width_m: 0.9,
            height_m: 2.1,
            copied_from_entity_id: "L1:door:1",
            confidence: 1,
            review_state: "accepted",
          },
        },
      ],
    });
    expect(state.present?.graph.walls).toHaveLength(1);
    expect(state.present?.graph.openings).toHaveLength(1);
    expect(state.present?.operations).toHaveLength(2);
    expect(state.present?.operations[0].changes).toMatchObject({
      level_id: "L2",
      copied_from_entity_id: "L1:wall:1",
    });
    expect(state.past).toHaveLength(1);
    state = studioSessionReducer(state, { type: "undo" });
    expect(state.present?.graph.walls).toEqual([]);
    expect(state.present?.graph.openings).toEqual([]);
  });

  it("jumps across the visible edit timeline and preserves redo chronology", () => {
    const selection = { collection: "fixtures" as const, id: "chair-1" };
    let state = studioSessionReducer(initialSessionState, { type: "load", graph });
    state = studioSessionReducer(state, {
      type: "edit",
      selection,
      changes: { center_m: [2, 2] },
    });
    state = studioSessionReducer(state, {
      type: "edit",
      selection,
      changes: { material: "oak" },
    });
    expect(state.past).toHaveLength(2);
    expect(state.present?.graph.fixtures[0]).toMatchObject({ center_m: [2, 2], material: "oak" });

    state = studioSessionReducer(state, { type: "jumpToHistory", index: 0 });
    expect(state.past).toHaveLength(0);
    expect(state.future).toHaveLength(2);
    expect(state.present?.graph.fixtures[0].center_m).toEqual([1, 1]);

    state = studioSessionReducer(state, { type: "jumpToHistory", index: 2 });
    expect(state.past).toHaveLength(2);
    expect(state.future).toHaveLength(0);
    expect(state.present?.graph.fixtures[0]).toMatchObject({ center_m: [2, 2], material: "oak" });
  });
});
