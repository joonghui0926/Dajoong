import { describe, expect, it } from "vitest";
import * as THREE from "three";

import { createFixtureGeometry } from "./fixtureGeometry";
import type { FixtureEntity } from "./types";

const material = (options: THREE.MeshStandardMaterialParameters) => (
  new THREE.MeshStandardMaterial(options)
);

function fixture(overrides: Partial<FixtureEntity> = {}): FixtureEntity {
  return {
    id: "fixture-1",
    level_id: "L1",
    type: "unknown",
    center_m: [1, 2],
    size_m: [0.6, 0.5, 0.8],
    yaw_deg: 0,
    ...overrides,
  };
}

describe("fixture geometry contract", () => {
  it("renders a converter-owned embedded mesh with its authored colors", () => {
    const group = createFixtureGeometry(fixture({
      geometry_status: "licensed_api_asset",
      asset_uid: "asset-1",
      geometry_ref: "mesh:asset-1",
      geometry_scale_xyz: [0.6, 0.5, 0.8],
    }), false, material, {
      schema_version: "dajoong.family-asset.v1",
      geometry_status: "licensed_api_asset",
      asset_uid: "asset-1",
      asset_mesh_sha256: "asset-1",
      normalized_to_unit_envelope: true,
      mesh_vertices: [
        [-0.3, -0.25, 0], [0.3, -0.25, 0], [0, 0.25, 0], [0, 0, 0.8],
      ],
      mesh_faces: [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
      mesh_face_colors: [[180, 160, 140], [120, 110, 100], [80, 90, 100], [200, 200, 195]],
    });

    const mesh = group.children[0] as THREE.Mesh;
    expect(mesh).toBeInstanceOf(THREE.Mesh);
    expect(mesh.geometry.getAttribute("color")).toBeDefined();
    expect(group.userData.assetUid).toBe("asset-1");
  });

  it("does not turn an unresolved symbol into a deceptive solid box", () => {
    const group = createFixtureGeometry(fixture(), false, material);

    expect(group.children).toHaveLength(1);
    expect(group.children[0]).toBeInstanceOf(THREE.LineSegments);
    expect(group.userData.geometryQuality).toBe("semantic_marker");
  });

  it("keeps a generic appliance neutral while providing coordination detail", () => {
    const group = createFixtureGeometry(fixture({
      family_id: "residential-electrical-appliance",
      type: "electrical appliance",
      geometry_status: "native_bim_parametric",
    }), false, material);

    expect(group.children.length).toBeGreaterThan(3);
    expect(group.children.every((item) => item instanceof THREE.Mesh)).toBe(true);
  });

  it("keeps stacked laundry appliances on their reviewed vertical band", () => {
    const group = createFixtureGeometry(fixture({
      type: "residential-tumble-dryer",
      family_id: "residential-tumble-dryer",
      source_class: "ElectricalAppliance TumbleDryer",
      base_elevation_m: 1,
      size_m: [0.62, 0.6, 1],
    }), false, material);

    expect(group.position.y).toBe(1);
    expect(group.children.length).toBeGreaterThan(4);
  });

  it("uses semantic cooktop geometry instead of a generic full-height asset", () => {
    const group = createFixtureGeometry(fixture({
      type: "residential-integrated-stove",
      family_id: "residential-integrated-stove",
      source_class: "ElectricalAppliance IntegratedStove",
      base_elevation_m: 0.89,
      size_m: [0.56, 0.56, 0.08],
    }), false, material, {
      schema_version: "dajoong.family-asset.v1",
      geometry_status: "licensed_api_asset",
      asset_mesh_sha256: "generic-appliance",
      normalized_to_unit_envelope: true,
      mesh_vertices: [
        [-.5, -.5, 0], [.5, -.5, 0], [0, .5, 0], [0, 0, 1],
      ],
      mesh_faces: [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
    });

    expect(group.position.y).toBe(0.89);
    expect(group.children.length).toBe(5);
    expect(group.userData.assetUid).toBeUndefined();
  });

  it.each([
    ["residential-bed", "reviewed:bed", 7],
    ["sofa", "reviewed:two-seat-sofa", 8],
    ["armchair", "reviewed:armchair", 8],
    ["chair", "reviewed:dining-chair", 6],
    ["desk", "reviewed:desk", 5],
  ])("builds %s as a semantic assembly instead of a placeholder cage", (type, familyId, minimumParts) => {
    const group = createFixtureGeometry(fixture({
      type,
      family_id: familyId,
      geometry_status: "reviewed_semantic_assembly",
      size_m: type === "residential-bed" ? [1.05, 2.1, .62] : [.9, 1.2, .9],
    }), false, material);

    expect(group.children.length).toBeGreaterThanOrEqual(minimumParts);
    expect(group.children.every((item) => item instanceof THREE.Mesh)).toBe(true);
    expect(group.children.some((item) => item instanceof THREE.LineSegments)).toBe(false);
  });
});
