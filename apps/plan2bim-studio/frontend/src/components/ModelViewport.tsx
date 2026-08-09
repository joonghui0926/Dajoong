import { BoxSelect, Cuboid, Eye, EyeOff, Move, RotateCw, Scissors } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";

import { selectionKey } from "../editorViewState";
import { graphBounds } from "../graph";
import type {
  CollectionName,
  FixtureEntity,
  PlanGraph,
  RouteEntity,
  Selection,
  VerticalConnectionEntity,
  WallEntity,
} from "../types";

interface ModelViewportProps {
  graph: PlanGraph;
  levelId: string;
  selections?: Selection[];
  onSelect: (selection: Selection, additive?: boolean) => void;
  onOpenContextMenu?: (selection: Selection, clientX: number, clientY: number) => void;
  hiddenCollections?: CollectionName[];
  lockedCollections?: CollectionName[];
  hiddenEntities?: Selection[];
  lockedEntities?: Selection[];
  isolatedEntities?: Selection[];
  selectionExclusions?: CollectionName[];
  onIsolateSelection?: () => void;
  onExitIsolation?: () => void;
  snapIncrementM?: number;
  onTransformCommit?: (
    selections: Selection[],
    transform: ModelTransformCommit,
  ) => boolean;
  minimal?: boolean;
}

export interface ModelTransformCommit {
  delta_m?: [number, number];
  rotation_delta_deg?: number;
  pivot_m?: [number, number];
}

const COLORS = {
  wall: 0xfffcf7,
  wood: 0xc99a61,
  wet: 0xbfc9c8,
  service: 0x99aead,
  glass: 0x65b9d7,
  fixture: 0x8b7559,
  appliance: 0x354c53,
  electrical: 0x715e4d,
  mechanical: 0x58756f,
  plumbing: 0x718983,
  fire: 0x9a594a,
  circulation: 0x667b70,
  selected: 0xf06b4f,
};

type ViewPreset = "iso" | "top" | "front";
type TransformMode = "translate" | "rotate";

export function ModelViewport({ graph, levelId, selections = [], onSelect, onOpenContextMenu, hiddenCollections = [], lockedCollections = [], hiddenEntities = [], lockedEntities = [], isolatedEntities = [], selectionExclusions = [], onIsolateSelection, onExitIsolation, snapIncrementM = 0.05, onTransformCommit, minimal = false }: ModelViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [viewPreset, setViewPreset] = useState<ViewPreset>("iso");
  const [sectionEnabled, setSectionEnabled] = useState(false);
  const [sectionHeight, setSectionHeight] = useState(1.45);
  const [transformMode, setTransformMode] = useState<TransformMode>("translate");
  const [transformReadout, setTransformReadout] = useState("");
  const selection = selections.at(-1) ?? null;
  const isolationActive = isolatedEntities.length > 0;
  const transformable = Boolean(
    selections.length > 0
    && selections.every((item) =>
      ["walls", "openings", "fixtures", "routes", "vertical_connections"].includes(item.collection)
      && !lockedCollections.includes(item.collection)
      && !lockedEntities.some(
        (locked) => locked.collection === item.collection && locked.id === item.id,
      )),
  );
  const rotatable = Boolean(
    transformable
    && selections.every((item) => ["fixtures", "vertical_connections"].includes(item.collection)),
  );

  useEffect(() => {
    if (!rotatable && transformMode === "rotate") setTransformMode("translate");
    setTransformReadout("");
  }, [rotatable, selection?.collection, selection?.id, transformMode]);
  const maxHeight = useMemo(
    () => Math.max(2.4, ...graph.walls.filter((item) => item.level_id === levelId).map((item) => item.height_m)),
    [graph, levelId],
  );

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf3eee7);
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.localClippingEnabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.screenSpacePanning = true;
    controls.maxPolarAngle = Math.PI * 0.49;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x9a8c7e, 2.45));
    const sun = new THREE.DirectionalLight(0xfffbf3, 3.2);
    sun.position.set(-14, 20, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xbcecff, 1.1);
    fill.position.set(12, 8, -10);
    scene.add(fill);

    const bounds = graphBounds(graph, levelId);
    const width = Math.max(2, bounds.maxX - bounds.minX);
    const depth = Math.max(2, bounds.maxY - bounds.minY);
    const center = new THREE.Vector3((bounds.minX + bounds.maxX) / 2, 0, (bounds.minY + bounds.maxY) / 2);
    const clippingPlanes = sectionEnabled ? [new THREE.Plane(new THREE.Vector3(0, -1, 0), sectionHeight)] : [];
    const isSelected = (collection: Selection["collection"], id: string) =>
      selections.some((item) => item.collection === collection && item.id === id);
    const hidden = new Set(hiddenCollections);
    const hiddenEntityKeys = new Set(hiddenEntities.map(selectionKey));
    const isolatedEntityKeys = new Set(isolatedEntities.map(selectionKey));
    const selectedOnly = (collection: Selection["collection"], id: string) =>
      !hidden.has(collection) &&
      !hiddenEntityKeys.has(selectionKey({ collection, id })) &&
      (!isolatedEntityKeys.size || isolatedEntityKeys.has(selectionKey({ collection, id })));
    const material = (options: THREE.MeshStandardMaterialParameters) =>
      new THREE.MeshStandardMaterial({ roughness: 0.76, ...options, clippingPlanes });
    const mark = (object: THREE.Object3D, item: Selection) => {
      object.userData.selection = item;
      object.traverse((child) => { child.userData.selection = item; });
      return object;
    };
    const transformTargets = new Map<string, THREE.Object3D>();
    const registerTransformTarget = (item: Selection, object: THREE.Object3D) => {
      transformTargets.set(selectionKey(item), object);
      return object;
    };

    const base = new THREE.Mesh(
      new THREE.BoxGeometry(width + 1, 0.09, depth + 1),
      material({ color: 0xd9d1c6, roughness: 0.94 }),
    );
    base.position.set(center.x, -0.065, center.z);
    base.receiveShadow = true;
    scene.add(base);

    for (const room of graph.rooms.filter((item) => item.level_id === levelId)) {
      if (room.polygon.length < 3 || !selectedOnly("rooms", room.id)) continue;
      const shape = new THREE.Shape();
      room.polygon.forEach(([x, y], index) => (index ? shape.lineTo(x, y) : shape.moveTo(x, y)));
      const geometry = new THREE.ShapeGeometry(shape);
      geometry.rotateX(Math.PI / 2);
      const descriptor = `${room.name ?? ""} ${room.occupancy ?? ""}`.toLowerCase();
      const floorColor = /bath|toilet|utility|mechanical|technical/.test(descriptor) ? COLORS.wet : /balcony|exterior/.test(descriptor) ? COLORS.service : COLORS.wood;
      const selected = isSelected("rooms", room.id);
      const mesh = new THREE.Mesh(
        geometry,
        material({ color: selected ? COLORS.selected : floorColor, roughness: 0.88, side: THREE.DoubleSide }),
      );
      mesh.position.y = 0.006;
      mesh.receiveShadow = true;
      mark(mesh, { collection: "rooms", id: room.id });
      scene.add(mesh);
    }

    for (const wall of graph.walls.filter((item) => item.level_id === levelId)) {
      if (!selectedOnly("walls", wall.id)) continue;
      const selectionRef: Selection = { collection: "walls", id: wall.id };
      const selected = isSelected("walls", wall.id);
      const pieces = wallPieces(wall, graph);
      const wallGroup = new THREE.Group();
      for (const piece of pieces) {
        const mesh = wallPieceMesh(wall, piece.start, piece.end, piece.base, piece.height, material({ color: selected ? COLORS.selected : COLORS.wall }));
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        wallGroup.add(mesh);
      }
      mark(wallGroup, selectionRef);
      registerTransformTarget(selectionRef, wallGroup);
      scene.add(wallGroup);
    }

    for (const opening of graph.openings.filter((item) => item.level_id === levelId)) {
      if (!selectedOnly("openings", opening.id)) continue;
      const hostWall = graph.walls.find((item) => item.id === opening.wall_id);
      const selected = isSelected("openings", opening.id);
      const window = opening.type === "window";
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.18, opening.width_m - 0.08), opening.height_m - 0.04, window ? 0.035 : 0.055),
        material({
          color: selected ? COLORS.selected : window ? COLORS.glass : 0x9b7856,
          roughness: window ? 0.18 : 0.62,
          metalness: window ? 0.12 : 0,
          transparent: window,
          opacity: window ? 0.6 : 1,
        }),
      );
      panel.position.set(opening.center_m[0], (opening.sill_height_m ?? 0) + opening.height_m / 2, opening.center_m[1]);
      if (hostWall) panel.rotation.y = -Math.atan2(hostWall.to[1] - hostWall.from[1], hostWall.to[0] - hostWall.from[0]);
      const selectionRef: Selection = { collection: "openings", id: opening.id };
      mark(panel, selectionRef);
      registerTransformTarget(selectionRef, panel);
      scene.add(panel);
    }

    for (const fixture of graph.fixtures.filter((item) => item.level_id === levelId)) {
      if (!selectedOnly("fixtures", fixture.id)) continue;
      const selected = isSelected("fixtures", fixture.id);
      const group = createFixture(fixture, selected, material);
      const selectionRef: Selection = { collection: "fixtures", id: fixture.id };
      mark(group, selectionRef);
      registerTransformTarget(selectionRef, group);
      scene.add(group);
    }

    for (const route of graph.routes.filter((item) => item.level_id === levelId)) {
      if (!selectedOnly("routes", route.id) || route.points_m.length < 2) continue;
      const selected = isSelected("routes", route.id);
      const routeMesh = createRoute(route, selected, material);
      const selectionRef: Selection = { collection: "routes", id: route.id };
      mark(routeMesh, selectionRef);
      registerTransformTarget(selectionRef, routeMesh);
      scene.add(routeMesh);
    }

    const levelHeight = graph.levels.find((item) => item.id === levelId)?.nominal_height_m ?? 3;
    for (const connection of graph.vertical_connections ?? []) {
      if (connection.from_level_id !== levelId && connection.to_level_id !== levelId) continue;
      if (!selectedOnly("vertical_connections", connection.id)) continue;
      const selected = isSelected("vertical_connections", connection.id);
      const connectionMesh = createVerticalConnection(
        connection,
        levelId,
        levelHeight,
        selected,
        material,
      );
      const selectionRef: Selection = { collection: "vertical_connections", id: connection.id };
      mark(connectionMesh, selectionRef);
      registerTransformTarget(selectionRef, connectionMesh);
      scene.add(connectionMesh);
    }

    const grid = new THREE.GridHelper(Math.ceil(Math.max(width, depth) * 1.5), Math.ceil(Math.max(width, depth) * 1.5), 0x8ca2aa, 0xd7d2ca);
    grid.position.set(center.x, -0.01, center.z);
    scene.add(grid);
    const span = Math.max(width, depth);
    if (viewPreset === "top") camera.position.set(center.x, span * 1.45, center.z + 0.001);
    else if (viewPreset === "front") camera.position.set(center.x, span * 0.38, center.z + span * 1.35);
    else camera.position.set(center.x + span * 0.78, span * 0.72, center.z + span * 0.92);
    controls.target.set(center.x, 0.8, center.z);
    controls.update();

    let transformControls: TransformControls | null = null;
    let transformHelper: THREE.Object3D | null = null;
    let transformTarget: THREE.Object3D | null = null;
    let transformStartPosition = new THREE.Vector3();
    let transformStartRotation = 0;
    let transformDragged = false;
    let suppressNextClick = false;
    if (transformable) {
      const selectedTargets = selections
        .map((item) => transformTargets.get(selectionKey(item)) ?? null)
        .filter((item): item is THREE.Object3D => item !== null);
      if (selectedTargets.length === 1) {
        transformTarget = selectedTargets[0];
      } else if (selectedTargets.length === selections.length) {
        scene.updateMatrixWorld(true);
        const bounds = new THREE.Box3();
        selectedTargets.forEach((target) => bounds.expandByObject(target));
        const pivot = bounds.getCenter(new THREE.Vector3());
        const proxy = new THREE.Group();
        proxy.position.copy(pivot);
        scene.add(proxy);
        scene.updateMatrixWorld(true);
        selectedTargets.forEach((target) => proxy.attach(target));
        transformTarget = proxy;
      }
      if (transformTarget) {
        transformControls = new TransformControls(camera, renderer.domElement);
        transformControls.setMode(transformMode);
        transformControls.setSpace("world");
        transformControls.setSize(0.82);
        transformControls.setTranslationSnap(snapIncrementM > 0 ? snapIncrementM : null);
        transformControls.setRotationSnap(THREE.MathUtils.degToRad(1));
        transformControls.showX = transformMode === "translate";
        transformControls.showY = transformMode === "rotate";
        transformControls.showZ = transformMode === "translate";
        transformControls.attach(transformTarget);
        transformHelper = transformControls.getHelper();
        scene.add(transformHelper);

        transformControls.addEventListener("mouseDown", () => {
          if (!transformTarget) return;
          transformStartPosition = transformTarget.position.clone();
          transformStartRotation = transformTarget.rotation.y;
          transformDragged = false;
          const prefix = selections.length > 1 ? `${selections.length} selected · ` : "";
          setTransformReadout(
            transformMode === "translate"
              ? `${prefix}ΔX 0.000 m  ·  ΔY 0.000 m`
              : `${prefix}ΔYaw 0.0°`,
          );
        });
        transformControls.addEventListener("objectChange", () => {
          if (!transformTarget) return;
          transformDragged = true;
          if (transformMode === "translate") {
            const dx = transformTarget.position.x - transformStartPosition.x;
            const dy = transformTarget.position.z - transformStartPosition.z;
            const prefix = selections.length > 1 ? `${selections.length} selected · ` : "";
            setTransformReadout(`${prefix}ΔX ${signedMetric(dx)}  ·  ΔY ${signedMetric(dy)}`);
          } else {
            const deltaYaw = normalizeDegrees(
              -THREE.MathUtils.radToDeg(transformTarget.rotation.y - transformStartRotation),
            );
            const prefix = selections.length > 1 ? `${selections.length} selected · ` : "";
            setTransformReadout(`${prefix}ΔYaw ${deltaYaw >= 0 ? "+" : ""}${deltaYaw.toFixed(1)}°`);
          }
        });
        transformControls.addEventListener("dragging-changed", (event) => {
          controls.enabled = !Boolean(event.value);
        });
        transformControls.addEventListener("mouseUp", () => {
          if (!transformTarget || !onTransformCommit || !transformDragged) return;
          suppressNextClick = true;
          let accepted = false;
          if (transformMode === "translate") {
            const delta: [number, number] = [
              transformTarget.position.x - transformStartPosition.x,
              transformTarget.position.z - transformStartPosition.z,
            ];
            if (Math.hypot(...delta) > 0.000001) {
              accepted = onTransformCommit(selections, { delta_m: delta });
            }
          } else {
            const deltaYaw = normalizeDegrees(
              -THREE.MathUtils.radToDeg(transformTarget.rotation.y - transformStartRotation),
            );
            if (Math.abs(transformTarget.rotation.y - transformStartRotation) > 0.000001) {
              accepted = onTransformCommit(selections, {
                rotation_delta_deg: deltaYaw,
                pivot_m: [transformStartPosition.x, transformStartPosition.z],
              });
            }
          }
          if (!accepted) {
            transformTarget.position.copy(transformStartPosition);
            transformTarget.rotation.y = transformStartRotation;
            setTransformReadout("Transform blocked · model constraints preserved");
          }
        });
      }
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const selectionExclusionSet = new Set(selectionExclusions);
    const selectionAt = (event: MouseEvent): Selection | null => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(scene.children, true).find((item) => {
        const candidate = item.object.userData.selection as Selection | undefined;
        return candidate && !selectionExclusionSet.has(candidate.collection);
      });
      return hit ? hit.object.userData.selection as Selection : null;
    };
    const click = (event: MouseEvent) => {
      if (suppressNextClick) {
        suppressNextClick = false;
        return;
      }
      const next = selectionAt(event);
      if (next) onSelect(next, event.ctrlKey || event.metaKey);
    };
    const contextMenu = (event: MouseEvent) => {
      const next = selectionAt(event);
      if (!next || !onOpenContextMenu) return;
      event.preventDefault();
      onOpenContextMenu(next, event.clientX, event.clientY);
    };
    renderer.domElement.addEventListener("click", click);
    renderer.domElement.addEventListener("contextmenu", contextMenu);

    const resize = () => {
      const rect = host.getBoundingClientRect();
      camera.aspect = Math.max(0.1, rect.width / Math.max(1, rect.height));
      camera.updateProjectionMatrix();
      renderer.setSize(rect.width, rect.height, false);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let frame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("click", click);
      renderer.domElement.removeEventListener("contextmenu", contextMenu);
      controls.dispose();
      if (transformControls) {
        transformControls.detach();
        transformControls.dispose();
      }
      if (transformHelper) scene.remove(transformHelper);
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((item) => item.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [graph, hiddenCollections, hiddenEntities, isolatedEntities, levelId, onOpenContextMenu, onSelect, onTransformCommit, sectionEnabled, sectionHeight, selectionExclusions, selections, snapIncrementM, transformMode, transformable, viewPreset]);

  return (
    <div className="viewport model-viewport" ref={hostRef} aria-label="3D model editor">
      <div className="viewport-label">3D MODEL</div>
      {!minimal ? (
        <div className={selections.length ? "model-tools with-selection" : "model-tools"} aria-label="3D view tools">
          <div className="view-cube">
            <button className={viewPreset === "top" ? "active" : ""} onClick={() => setViewPreset("top")}>TOP</button>
            <button className={viewPreset === "iso" ? "active" : ""} onClick={() => setViewPreset("iso")}>ISO</button>
            <button className={viewPreset === "front" ? "active" : ""} onClick={() => setViewPreset("front")}>FRONT</button>
          </div>
          <div className="model-transform-tools" aria-label="3D transform tools">
            <button
              className={transformMode === "translate" ? "tool-toggle active" : "tool-toggle"}
              aria-pressed={transformMode === "translate"}
              disabled={!transformable}
              onClick={() => setTransformMode("translate")}
              title="Move selected geometry in model coordinates"
            ><Move size={15} /></button>
            <button
              className={transformMode === "rotate" ? "tool-toggle active" : "tool-toggle"}
              aria-pressed={transformMode === "rotate"}
              disabled={!rotatable}
              onClick={() => setTransformMode("rotate")}
              title="Rotate selected object around its vertical axis"
            ><RotateCw size={15} /></button>
          </div>
          <button className={sectionEnabled ? "tool-toggle active" : "tool-toggle"} onClick={() => setSectionEnabled((value) => !value)} title="Section cut"><Scissors size={15} /></button>
          <button
            className={isolationActive ? "tool-toggle active" : "tool-toggle"}
            disabled={!selection && !isolationActive}
            onClick={isolationActive ? onExitIsolation : onIsolateSelection}
            title={isolationActive ? "Exit 2D and 3D isolation" : "Isolate selected in 2D and 3D (I)"}
          >
            {isolationActive ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
          <button className="tool-toggle" onClick={() => setViewPreset("iso")} title="Fit model"><BoxSelect size={15} /></button>
        </div>
      ) : null}
      {!minimal && sectionEnabled ? <label className={selections.length ? "section-slider with-selection" : "section-slider"}><Scissors size={13} /><input aria-label="Section height" type="range" min="0.25" max={maxHeight} step="0.05" value={sectionHeight} onChange={(event) => setSectionHeight(Number(event.target.value))} /><b>{sectionHeight.toFixed(2)} m</b></label> : null}
      {!minimal && transformReadout ? <output className="model-transform-readout">{transformReadout}</output> : null}
      {!minimal ? <div className="viewport-hint"><Cuboid size={11} /> Orbit · pan · zoom · select</div> : null}
    </div>
  );
}

function normalizeDegrees(value: number): number {
  const normalized = ((value % 360) + 360) % 360;
  return normalized > 180 ? normalized - 360 : normalized;
}

function signedMetric(value: number): string {
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  return `${normalized >= 0 ? "+" : ""}${normalized.toFixed(3)} m`;
}

interface WallPiece { start: number; end: number; base: number; height: number }

function wallPieces(wall: WallEntity, graph: PlanGraph): WallPiece[] {
  const length = Math.hypot(wall.to[0] - wall.from[0], wall.to[1] - wall.from[1]);
  const openings = graph.openings
    .filter((item) => item.wall_id === wall.id)
    .map((item) => {
      const dx = wall.to[0] - wall.from[0];
      const dz = wall.to[1] - wall.from[1];
      const along = ((item.center_m[0] - wall.from[0]) * dx + (item.center_m[1] - wall.from[1]) * dz) / Math.max(0.001, length);
      return { item, start: Math.max(0, along - item.width_m / 2), end: Math.min(length, along + item.width_m / 2) };
    })
    .sort((left, right) => left.start - right.start);
  if (!openings.length) return [{ start: 0, end: length, base: 0, height: wall.height_m }];
  const pieces: WallPiece[] = [];
  let cursor = 0;
  for (const opening of openings) {
    if (opening.start > cursor) pieces.push({ start: cursor, end: opening.start, base: 0, height: wall.height_m });
    const sill = Math.max(0, opening.item.sill_height_m ?? 0);
    const top = Math.min(wall.height_m, sill + opening.item.height_m);
    if (sill > 0.02) pieces.push({ start: opening.start, end: opening.end, base: 0, height: sill });
    if (top < wall.height_m - 0.02) pieces.push({ start: opening.start, end: opening.end, base: top, height: wall.height_m - top });
    cursor = Math.max(cursor, opening.end);
  }
  if (cursor < length) pieces.push({ start: cursor, end: length, base: 0, height: wall.height_m });
  return pieces.filter((piece) => piece.end - piece.start > 0.01 && piece.height > 0.01);
}

function wallPieceMesh(wall: WallEntity, start: number, end: number, base: number, height: number, material: THREE.Material) {
  const dx = wall.to[0] - wall.from[0];
  const dz = wall.to[1] - wall.from[1];
  const length = Math.max(0.001, Math.hypot(dx, dz));
  const ux = dx / length;
  const uz = dz / length;
  const midpoint = (start + end) / 2;
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(end - start, height, Math.max(0.06, wall.thickness_m)), material);
  mesh.position.set(wall.from[0] + ux * midpoint, base + height / 2, wall.from[1] + uz * midpoint);
  mesh.rotation.y = -Math.atan2(dz, dx);
  return mesh;
}

function createFixture(
  fixture: FixtureEntity,
  selected: boolean,
  material: (options: THREE.MeshStandardMaterialParameters) => THREE.MeshStandardMaterial,
) {
  const group = new THREE.Group();
  const [width, depth, height] = fixture.size_m.map((value) => Math.max(0.08, value));
  const type = `${fixture.type} ${fixture.family_id}`.toLowerCase();
  const color = selected ? COLORS.selected : /appliance|electrical/.test(type) ? COLORS.appliance : COLORS.fixture;
  const add = (geometry: THREE.BufferGeometry, position: [number, number, number], shade = color) => {
    const mesh = new THREE.Mesh(geometry, material({ color: shade, roughness: .68 }));
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  };
  if (/toilet/.test(type)) {
    add(new THREE.BoxGeometry(width * .72, height * .42, depth * .72), [0, height * .21, depth * .08], 0xf4f1e9);
    add(new THREE.CylinderGeometry(width * .3, width * .36, height * .2, 22), [0, height * .46, -depth * .12], 0xfaf8f2);
    add(new THREE.BoxGeometry(width * .78, height * .46, depth * .3), [0, height * .69, -depth * .31], 0xf4f1e9);
  } else if (/sink/.test(type)) {
    add(new THREE.BoxGeometry(width, height * .72, depth), [0, height * .36, 0]);
    add(new THREE.BoxGeometry(width * .9, height * .08, depth * .88), [0, height * .78, 0], 0xe7e3db);
    add(new THREE.CylinderGeometry(width * .07, width * .07, height * .2, 12), [0, height * .94, -depth * .16], 0x9ca9aa);
  } else if (/closet|casework/.test(type)) {
    add(new THREE.BoxGeometry(width, height, depth), [0, height / 2, 0]);
    add(new THREE.BoxGeometry(.025, height * .84, depth + .012), [-width * .25, height * .52, 0], 0xb69a75);
    add(new THREE.BoxGeometry(.025, height * .84, depth + .012), [width * .25, height * .52, 0], 0xb69a75);
  } else if (/bench/.test(type)) {
    add(new THREE.BoxGeometry(width, height * .18, depth), [0, height * .72, 0]);
    add(new THREE.BoxGeometry(width * .08, height * .65, depth * .72), [-width * .38, height * .34, 0]);
    add(new THREE.BoxGeometry(width * .08, height * .65, depth * .72), [width * .38, height * .34, 0]);
  } else {
    add(new THREE.BoxGeometry(width, height, depth), [0, height / 2, 0]);
    if (/appliance|electrical/.test(type)) add(new THREE.BoxGeometry(width * .78, height * .44, .015), [0, height * .55, depth * .505], 0x5d777f);
  }
  group.position.set(fixture.center_m[0], fixture.base_elevation_m ?? 0, fixture.center_m[1]);
  group.rotation.y = -((fixture.yaw_deg ?? 0) * Math.PI) / 180;
  return group;
}

function createRoute(
  route: RouteEntity,
  selected: boolean,
  material: (options: THREE.MeshStandardMaterialParameters) => THREE.MeshStandardMaterial,
) {
  const points = route.points_m.map(
    ([x, y, z]) => new THREE.Vector3(x, Math.max(0.015, z), y),
  );
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
  const section = route.section_m ?? [0.05, 0.05];
  const radius = Math.max(0.012, Math.max(...section) / 2);
  const segments = Math.min(512, Math.max(12, points.length * 12));
  const discipline = String(route.discipline ?? "mechanical") as keyof typeof COLORS;
  const shade = selected ? COLORS.selected : COLORS[discipline] ?? COLORS.mechanical;
  const mesh = new THREE.Mesh(
    new THREE.TubeGeometry(curve, segments, radius, 8, false),
    material({ color: shade, metalness: 0.15, roughness: 0.48 }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function createVerticalConnection(
  connection: VerticalConnectionEntity,
  levelId: string,
  levelHeight: number,
  selected: boolean,
  material: (options: THREE.MeshStandardMaterialParameters) => THREE.MeshStandardMaterial,
) {
  const group = new THREE.Group();
  const [width, depth] = connection.footprint_m.map((value) => Math.max(0.25, value));
  const shade = selected ? COLORS.selected : COLORS.circulation;
  const kind = String(connection.type ?? connection.kind ?? "stair").toLowerCase();
  const add = (
    geometry: THREE.BufferGeometry,
    position: [number, number, number],
    color = shade,
  ) => {
    const mesh = new THREE.Mesh(
      geometry,
      material({ color, roughness: 0.7, metalness: /escalator|elevator/.test(kind) ? 0.18 : 0 }),
    );
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  };
  const isArrival = connection.to_level_id === levelId;
  if (isArrival) {
    add(new THREE.BoxGeometry(width, 0.08, depth), [0, 0.04, 0]);
  } else if (/elevator|riser/.test(kind)) {
    add(new THREE.BoxGeometry(width, levelHeight, depth), [0, levelHeight / 2, 0]);
  } else if (kind === "ramp") {
    const ramp = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.12, Math.hypot(depth, levelHeight)),
      material({ color: shade, roughness: 0.76 }),
    );
    ramp.rotation.x = Math.atan2(levelHeight, depth);
    ramp.position.set(0, levelHeight / 2, 0);
    ramp.castShadow = true;
    group.add(ramp);
  } else {
    const stepCount = Math.max(8, Math.min(24, Math.round(levelHeight / 0.18)));
    const treadDepth = depth / stepCount;
    const rise = levelHeight / stepCount;
    for (let index = 0; index < stepCount; index += 1) {
      const stepHeight = rise * (index + 1);
      add(
        new THREE.BoxGeometry(width, stepHeight, treadDepth),
        [0, stepHeight / 2, -depth / 2 + treadDepth * (index + 0.5)],
      );
    }
  }
  group.position.set(connection.center_m[0], 0, connection.center_m[1]);
  group.rotation.y = -((connection.yaw_deg ?? 0) * Math.PI) / 180;
  return group;
}
