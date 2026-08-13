import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

import type { FamilyAssetDefinition, FixtureEntity } from "./types";

export const FIXTURE_COLORS = {
  fixture: 0x8b7559,
  appliance: 0x354c53,
  electrical: 0x715e4d,
  mechanical: 0x58756f,
  plumbing: 0x718983,
  fire: 0x9a594a,
  selected: 0xf06b4f,
  review: 0x9a594a,
};

type MaterialFactory = (
  options: THREE.MeshStandardMaterialParameters,
) => THREE.MeshStandardMaterial;

function validEmbeddedMesh(geometry: FixtureEntity | FamilyAssetDefinition) {
  const vertices = geometry.mesh_vertices ?? [];
  const faces = geometry.mesh_faces ?? [];
  return vertices.length >= 4
    && faces.length >= 4
    && faces.every((face) => face.length === 3 && face.every((index) => (
      Number.isInteger(index) && index >= 0 && index < vertices.length
    )));
}

function embeddedMesh(
  source: FixtureEntity | FamilyAssetDefinition,
  selected: boolean,
  material: MaterialFactory,
  scale: [number, number, number] = [1, 1, 1],
) {
  if (!validEmbeddedMesh(source)) return null;
  const vertices = source.mesh_vertices ?? [];
  const colors = source.mesh_face_colors ?? [];
  const positions: number[] = [];
  const vertexColors: number[] = [];
  for (const [faceIndex, face] of (source.mesh_faces ?? []).entries()) {
    const color = colors[faceIndex] ?? [128, 128, 128];
    // Catalog geometry is Z-up. Three.js is Y-up, so reverse winding while
    // mapping (x, y, z) -> (x, z, y).
    for (const index of [face[0], face[2], face[1]]) {
      const [x, y, z] = vertices[index];
      positions.push(x * scale[0], z * scale[2], y * scale[1]);
      vertexColors.push(color[0] / 255, color[1] / 255, color[2] / 255);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(vertexColors, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(
    geometry,
    material({
      color: selected ? FIXTURE_COLORS.selected : 0xffffff,
      vertexColors: !selected,
      roughness: 0.66,
      metalness: 0.04,
    }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

export function createFixtureGeometry(
  fixture: FixtureEntity,
  selected: boolean,
  material: MaterialFactory,
  sharedGeometry?: FamilyAssetDefinition,
) {
  const group = new THREE.Group();
  const [width, depth, height] = fixture.size_m.map((value) => Math.max(0.035, value));
  const sourceClass = String(fixture.source_class ?? "").toLowerCase();
  const type = `${fixture.type} ${fixture.family_id ?? ""} ${sourceClass}`.toLowerCase();
  const preferSemanticAssembly = /integratedstove|washingmachine|tumbledryer|countertop-sink/.test(type);
  const discipline = String(fixture.discipline ?? "architectural") as keyof typeof FIXTURE_COLORS;
  const baseColor = selected
    ? FIXTURE_COLORS.selected
    : FIXTURE_COLORS[discipline] ?? (/appliance|electrical/.test(type)
      ? FIXTURE_COLORS.appliance
      : FIXTURE_COLORS.fixture);
  const sourceMesh = preferSemanticAssembly
    ? null
    : embeddedMesh(
      sharedGeometry ?? fixture,
      selected,
      material,
      sharedGeometry?.normalized_to_unit_envelope
        ? fixture.geometry_scale_xyz ?? fixture.size_m
        : [1, 1, 1],
    );
  if (sourceMesh) {
    group.add(sourceMesh);
    group.userData.geometryQuality = fixture.geometry_status;
    group.userData.assetUid = fixture.asset_uid;
  } else {
    const add = (
      geometry: THREE.BufferGeometry,
      position: [number, number, number],
      shade = baseColor,
      rotation: [number, number, number] = [0, 0, 0],
      options: THREE.MeshStandardMaterialParameters = {},
    ) => {
      const mesh = new THREE.Mesh(
        geometry,
        material({ color: shade, roughness: 0.68, ...options }),
      );
      mesh.position.set(...position);
      mesh.rotation.set(...rotation);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      return mesh;
    };
    const rounded = (w: number, h: number, d: number, radius = Math.min(w, h, d) * .08) => (
      new RoundedBoxGeometry(w, h, d, 3, Math.max(.006, radius))
    );

    if (/residential-bed|double-bed|\bbed\b/.test(type)) {
      add(rounded(width * .96, height * .22, depth * .96, .025), [0, height * .19, 0], 0x80684f);
      add(rounded(width * .91, height * .24, depth * .89, .055), [0, height * .38, depth * .015], 0xeee8dd);
      add(rounded(width * .96, height * .58, depth * .08, .025), [0, height * .55, -depth * .44], 0x94765c);
      const pillowCount = width > 1.35 ? 2 : 1;
      for (let index = 0; index < pillowCount; index += 1) {
        const x = pillowCount === 1 ? 0 : (index === 0 ? -width * .23 : width * .23);
        add(rounded(width * (pillowCount === 1 ? .58 : .38), height * .13, depth * .20, .045), [x, height * .57, -depth * .28], 0xf8f5ee);
      }
      add(rounded(width * .84, height * .055, depth * .48, .025), [0, height * .54, depth * .18], 0xb9c8c2);
      for (const x of [-width * .38, width * .38]) for (const z of [-depth * .38, depth * .38]) {
        add(rounded(width * .055, height * .15, depth * .055, .008), [x, height * .075, z], 0x5e5143);
      }
    } else if (/sofa|armchair/.test(type)) {
      const seatShade = /armchair/.test(type) ? 0x7e927f : 0x8b9d8f;
      add(rounded(width * .88, height * .25, depth * .76, .055), [0, height * .39, depth * .04], seatShade);
      add(rounded(width * .88, height * .52, depth * .18, .045), [0, height * .66, -depth * .35], 0x708575);
      for (const x of [-width * .46, width * .46]) {
        add(rounded(width * .12, height * .42, depth * .78, .035), [x, height * .45, depth * .02], 0x65796a);
      }
      const cushionCount = /armchair/.test(type) ? 1 : Math.max(2, Math.min(3, Math.round(width / .7)));
      for (let index = 0; index < cushionCount; index += 1) {
        const x = cushionCount === 1 ? 0 : -width * .31 + index * (width * .62 / (cushionCount - 1));
        add(rounded(width * (.68 / cushionCount), height * .12, depth * .58, .035), [x, height * .52, depth * .04], 0x9baca0);
      }
      for (const x of [-width * .36, width * .36]) for (const z of [-depth * .25, depth * .25]) {
        add(rounded(width * .045, height * .18, depth * .045, .006), [x, height * .09, z], 0x4a4d47);
      }
    } else if (/chair/.test(type)) {
      add(rounded(width * .82, height * .10, depth * .78, .025), [0, height * .48, 0], 0x9d805e);
      add(rounded(width * .82, height * .45, depth * .10, .025), [0, height * .73, -depth * .34], 0x927454);
      for (const x of [-width * .33, width * .33]) for (const z of [-depth * .30, depth * .30]) {
        add(rounded(width * .065, height * .47, depth * .065, .008), [x, height * .235, z], 0x4c504b);
      }
    } else if (/toilet/.test(type)) {
      const bowl = add(
        new THREE.SphereGeometry(0.5, 28, 18),
        [0, height * .38, depth * .02],
        0xf6f3ec,
      );
      bowl.scale.set(width * .43, height * .28, depth * .46);
      add(new THREE.TorusGeometry(width * .31, Math.max(.015, width * .045), 10, 32), [0, height * .60, depth * .02], 0xfffdf8, [Math.PI / 2, 0, 0]);
      add(rounded(width * .78, height * .44, depth * .28), [0, height * .73, -depth * .32], 0xf5f1e9);
      add(new THREE.CylinderGeometry(width * .22, width * .28, height * .18, 24), [0, height * .09, -depth * .04], 0xf1eee6);
    } else if (/countertop-sink/.test(type)) {
      add(rounded(width, height * .26, depth, .018), [0, height * .13, 0], 0xd7d5cf);
      const basin = add(
        new THREE.SphereGeometry(.5, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2),
        [0, height * .20, 0],
        0xf8f6f0,
      );
      basin.scale.set(width * .34, height * .23, depth * .33);
      add(new THREE.CylinderGeometry(width * .035, width * .035, height * .62, 14), [0, height * .48, -depth * .27], 0x8d9998);
      add(new THREE.TorusGeometry(width * .10, width * .022, 9, 18, Math.PI), [0, height * .74, -depth * .18], 0x8d9998, [0, Math.PI / 2, 0]);
    } else if (/sink|lavatory/.test(type)) {
      add(rounded(width, height * .11, depth, .025), [0, height * .79, 0], 0xe5e2db);
      const basin = add(new THREE.SphereGeometry(.5, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), [0, height * .73, 0], 0xf7f5ef);
      basin.scale.set(width * .36, height * .16, depth * .34);
      add(new THREE.CylinderGeometry(width * .055, width * .055, height * .25, 14), [0, height * .93, -depth * .22], 0x8d9998);
      add(new THREE.TorusGeometry(width * .12, width * .025, 9, 18, Math.PI), [0, height * 1.04, -depth * .12], 0x8d9998, [0, Math.PI / 2, 0]);
      add(rounded(width * .27, height * .64, depth * .27), [0, height * .34, depth * .05]);
    } else if (/closet|casework|cabinet/.test(type)) {
      add(rounded(width, height, depth, .018), [0, height / 2, 0], 0x9b8264);
      const panelCount = Math.max(1, Math.min(6, Math.round(width / .58)));
      const panelWidth = Math.max(.06, width / panelCount - .016);
      for (let index = 0; index < panelCount; index += 1) {
        const x = -width / 2 + (index + .5) * width / panelCount;
        add(rounded(panelWidth, height * .84, .026, .008), [x, height * .52, depth / 2 + .012], 0xb19672);
        add(new THREE.CylinderGeometry(.012, .012, height * .12, 10), [x + panelWidth * .31, height * .54, depth / 2 + .035], 0x4b4d48);
      }
      add(new THREE.BoxGeometry(width * .92, height * .06, depth * .72), [0, height * .03, 0], 0x675847);
    } else if (/bench/.test(type)) {
      const slatCount = Math.max(3, Math.min(9, Math.round(depth / .08)));
      const slatDepth = depth / slatCount * .82;
      for (let index = 0; index < slatCount; index += 1) {
        const z = -depth / 2 + (index + .5) * depth / slatCount;
        add(rounded(width, height * .09, slatDepth, .01), [0, height * .78, z], 0xb68f62);
      }
      for (const x of [-width * .38, width * .38]) {
        add(rounded(width * .07, height * .70, depth * .72, .012), [x, height * .37, 0], 0x745f46);
      }
    } else if (/bathtub/.test(type)) {
      add(rounded(width, height * .82, depth, .08), [0, height * .41, 0], 0xf5f2eb);
      add(rounded(width * .84, height * .72, depth * .70, .08), [0, height * .53, 0], 0xb9c9c8);
    } else if (/shower/.test(type)) {
      add(new THREE.CylinderGeometry(.018, .018, height * .72, 14), [0, height * .47, -depth * .34], 0x798d91);
      add(new THREE.CylinderGeometry(width * .18, width * .24, .055, 24), [0, height * .83, -depth * .22], 0x91a5a8, [Math.PI / 2.7, 0, 0], { metalness: .38 });
      add(new THREE.CylinderGeometry(width * .11, width * .11, .035, 20), [0, height * .42, -depth * .48], 0x8da0a3, [Math.PI / 2, 0, 0], { metalness: .3 });
    } else if (/stairs?/.test(type)) {
      const stepCount = Math.max(4, Math.min(14, Math.round(Math.max(width, depth) / .22)));
      const runAlongX = width >= depth;
      for (let index = 0; index < stepCount; index += 1) {
        const fraction = (index + 1) / stepCount;
        const stepHeight = height * fraction;
        const stepWidth = runAlongX ? width / stepCount : width;
        const stepDepth = runAlongX ? depth : depth / stepCount;
        const x = runAlongX ? -width / 2 + stepWidth * (index + .5) : 0;
        const z = runAlongX ? 0 : -depth / 2 + stepDepth * (index + .5);
        add(new THREE.BoxGeometry(stepWidth * .98, stepHeight, stepDepth * .98), [x, stepHeight / 2, z], 0xb7aa96);
      }
    } else if (/column/.test(type)) {
      add(rounded(width, height, depth, .012), [0, height / 2, 0], 0xe9e7df);
    } else if (/workstation|conference-table|dining-table|coffee-table|desk|table/.test(type)) {
      add(rounded(width, height * .10, depth, .02), [0, height * .86, 0], 0xa77f54);
      for (const x of [-width * .40, width * .40]) for (const z of [-depth * .34, depth * .34]) {
        add(rounded(width * .055, height * .80, depth * .055, .01), [x, height * .42, z], 0x404b4b);
      }
    } else if (/integratedstove|cooktop/.test(type)) {
      add(rounded(width, height * .24, depth, .018), [0, height * .12, 0], 0x252d2e, [0, 0, 0], { metalness: .34, roughness: .28 });
      for (const x of [-width * .25, width * .25]) for (const z of [-depth * .25, depth * .25]) {
        add(new THREE.TorusGeometry(Math.min(width, depth) * .105, Math.min(width, depth) * .012, 8, 24), [x, height * .26, z], 0x7e8786, [Math.PI / 2, 0, 0], { metalness: .45 });
      }
    } else if (/washingmachine|tumbledryer|washer|dryer/.test(type)) {
      add(rounded(width * .96, height * .96, depth * .96, .025), [0, height * .48, 0], 0xe8e8e4);
      add(new THREE.CylinderGeometry(width * .29, width * .29, .035, 36), [0, height * .45, depth * .49], 0x293537, [Math.PI / 2, 0, 0], { metalness: .18, roughness: .28 });
      add(new THREE.TorusGeometry(width * .31, width * .035, 12, 36), [0, height * .45, depth * .51], 0x8e9998, [Math.PI / 2, 0, 0], { metalness: .32 });
      add(rounded(width * .78, height * .10, .028, .008), [0, height * .83, depth * .50], 0xbcc3c0);
      add(rounded(width * .16, height * .06, .03, .006), [width * .24, height * .83, depth * .52], 0x2d3839);
    } else if (/appliance/.test(type)) {
      // A neutral, review-required appliance family: enough geometry for
      // coordination without pretending the source established a subtype.
      add(rounded(width * .96, height * .94, depth * .96, .025), [0, height * .49, 0], 0x5d6663);
      add(rounded(width * .80, height * .58, .028, .012), [0, height * .48, depth * .49], 0x303b3c, [0, 0, 0], { metalness: .18 });
      add(rounded(width * .74, height * .10, .032, .008), [0, height * .82, depth * .50], 0x85908c);
      for (const x of [-width * .31, width * .31]) add(new THREE.CylinderGeometry(.018, .018, .025, 12), [x, height * .83, depth * .52], 0x252c2c, [Math.PI / 2, 0, 0]);
    } else if (/heat-pump|fan-coil/.test(type)) {
      add(rounded(width, height * .88, depth, .025), [0, height * .48, 0]);
      for (let index = -3; index <= 3; index += 1) add(new THREE.BoxGeometry(width * .72, .012, depth * .025), [0, height * (.38 + index * .045), depth * .51], 0x9aa9a6);
    } else if (/air-terminal|diffuser/.test(type)) {
      add(rounded(width, height * .35, depth, .014), [0, height * .18, 0], 0xd4d8d3);
      for (const fraction of [-.28, -.14, 0, .14, .28]) {
        add(new THREE.BoxGeometry(width * .82, height * .08, depth * .035), [0, height * .38, depth * fraction], 0x82938f);
      }
    } else if (/electrical-panel|panelboard/.test(type)) {
      add(rounded(width, height, depth, .016), [0, height / 2, 0], 0x78827d);
      add(rounded(width * .84, height * .76, .025, .008), [0, height * .51, depth * .51], 0x9aa39e);
      add(new THREE.CylinderGeometry(.012, .012, height * .16, 10), [width * .30, height * .52, depth * .54], 0x353c3b);
    } else if (/light|smoke-detector/.test(type)) {
      add(new THREE.CylinderGeometry(width * .46, width * .42, height, 28), [0, height / 2, 0], 0xf2f0e8);
      add(new THREE.CylinderGeometry(width * .34, width * .34, height * .10, 28), [0, height * .48, 0], 0xc8d5d1);
    } else if (/sprinkler/.test(type)) {
      add(new THREE.CylinderGeometry(width * .09, width * .09, height * .66, 14), [0, height * .36, 0], 0xb68d4c);
      add(new THREE.TorusGeometry(width * .34, width * .06, 8, 24), [0, height * .72, 0], 0xb68d4c, [Math.PI / 2, 0, 0]);
    } else if (/receptacle|thermostat/.test(type)) {
      add(rounded(width, height, depth, .01), [0, height / 2, 0], 0xe8e7df);
      add(rounded(width * .48, height * .28, .012, .004), [0, height * .55, depth * .52], 0x63716f);
    } else {
      const cageGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(width, height, depth));
      const cage = new THREE.LineSegments(
        cageGeometry,
        new THREE.LineBasicMaterial({ color: selected ? FIXTURE_COLORS.selected : FIXTURE_COLORS.review, transparent: true, opacity: .72 }),
      );
      cage.position.y = height / 2;
      group.add(cage);
      group.userData.geometryQuality = "semantic_marker";
    }
  }
  group.position.set(
    fixture.center_m[0],
    fixture.base_elevation_m ?? 0,
    fixture.center_m[1],
  );
  group.rotation.y = -((fixture.yaw_deg ?? 0) * Math.PI) / 180;
  return group;
}
