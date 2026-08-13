import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import type { FixtureEntity, PlanGraph } from "./types";

const graph = JSON.parse(readFileSync(
  new URL("../public/sample/03-plan-graph.json", import.meta.url),
  "utf8",
)) as PlanGraph;

function horizontalEnvelope(fixture: FixtureEntity) {
  const quarterTurn = Math.abs(Math.round((fixture.yaw_deg ?? 0) / 90)) % 2 === 1;
  const width = quarterTurn ? fixture.size_m[1] : fixture.size_m[0];
  const depth = quarterTurn ? fixture.size_m[0] : fixture.size_m[1];
  return {
    minX: fixture.center_m[0] - width / 2,
    maxX: fixture.center_m[0] + width / 2,
    minY: fixture.center_m[1] - depth / 2,
    maxY: fixture.center_m[1] + depth / 2,
    area: width * depth,
  };
}

function footprintOverlap(left: FixtureEntity, right: FixtureEntity) {
  const a = horizontalEnvelope(left);
  const b = horizontalEnvelope(right);
  const width = Math.max(0, Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX));
  const depth = Math.max(0, Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY));
  return (width * depth) / Math.max(0.0001, Math.min(a.area, b.area));
}

function verticalOverlap(left: FixtureEntity, right: FixtureEntity) {
  const leftBase = left.base_elevation_m ?? 0;
  const rightBase = right.base_elevation_m ?? 0;
  return Math.min(leftBase + left.size_m[2], rightBase + right.size_m[2])
    - Math.max(leftBase, rightBase);
}

describe("reviewed CUBI-020 object stacking", () => {
  it("stores fixture dimensions in local asset axes before applying yaw", () => {
    expect((graph.provenance as Record<string, unknown> | undefined)?.fixture_axis_contract)
      .toBe("local_size_plus_yaw_v1");

    const verticalStair = graph.fixtures.find((fixture) => fixture.source_entity_id === "svg-fixture-040");
    expect(verticalStair).toBeDefined();
    const stairEnvelope = horizontalEnvelope(verticalStair!);
    expect(stairEnvelope.maxX - stairEnvelope.minX).toBeCloseTo(0.3119, 3);
    expect(stairEnvelope.maxY - stairEnvelope.minY).toBeCloseTo(3.6651, 3);

    const verticalCabinet = graph.fixtures.find((fixture) => fixture.source_entity_id === "svg-fixture-032");
    expect(verticalCabinet).toBeDefined();
    const cabinetEnvelope = horizontalEnvelope(verticalCabinet!);
    expect(cabinetEnvelope.maxX - cabinetEnvelope.minX).toBeCloseTo(0.8388, 3);
    expect(cabinetEnvelope.maxY - cabinetEnvelope.minY).toBeCloseTo(1.8822, 3);
  });

  it("keeps the directly reviewed furniture pass in the delivered graph", () => {
    const count = (sourceClass: string) => graph.fixtures.filter((fixture) => fixture.source_class === sourceClass).length;

    expect(graph.fixtures).toHaveLength(65);
    expect(count("Bed") + count("DoubleBed")).toBe(4);
    expect(count("Desk")).toBe(3);
    expect(count("DeskChair")).toBe(3);
    expect(count("DiningChair")).toBe(6);
    expect(count("TwoSeatSofa")).toBe(1);
    expect(count("Armchair")).toBe(2);
  });

  it("does not leave strongly overlapping objects in the same vertical band", () => {
    const collisions: string[] = [];
    graph.fixtures.forEach((left, index) => {
      for (const right of graph.fixtures.slice(index + 1)) {
        if (left.level_id !== right.level_id) continue;
        if (footprintOverlap(left, right) < 0.72) continue;
        if (verticalOverlap(left, right) <= 0.02) continue;
        collisions.push(`${left.id} <> ${right.id}`);
      }
    });

    expect(collisions).toEqual([]);
  });

  it("keeps the reviewed laundry stack and counter-mounted components supported", () => {
    const bySourceClass = new Map(graph.fixtures.map((fixture) => [String(fixture.source_class), fixture]));
    const washer = bySourceClass.get("ElectricalAppliance WashingMachine");
    const dryer = bySourceClass.get("ElectricalAppliance TumbleDryer");
    const cooktop = bySourceClass.get("ElectricalAppliance IntegratedStove");

    expect(washer?.base_elevation_m).toBe(0);
    expect(dryer?.base_elevation_m).toBeCloseTo(1, 3);
    expect(cooktop?.base_elevation_m).toBeCloseTo(0.89, 3);
    expect(cooktop?.size_m[2]).toBeLessThanOrEqual(0.1);
    expect(graph.fixtures.filter((fixture) => fixture.type === "residential-countertop-sink"))
      .toHaveLength(3);
  });
});
