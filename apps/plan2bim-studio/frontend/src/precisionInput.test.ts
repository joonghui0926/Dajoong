import { describe, expect, it } from "vitest";

import {
  endpointFromLengthAngle,
  lengthAndAngle,
  parseAngleInput,
  parseLengthInput,
  parseOffsetInput,
  parseSignedAngleInput,
} from "./precisionInput";

describe("precision input", () => {
  it("parses common metric and imperial construction units", () => {
    expect(parseLengthInput("2.5")).toBe(2.5);
    expect(parseLengthInput("2500 mm")).toBe(2.5);
    expect(parseLengthInput("250 cm")).toBe(2.5);
    expect(parseLengthInput("8 ft")).toBeCloseTo(2.4384);
    expect(parseLengthInput("8' 2\"")).toBeCloseTo(2.4892);
  });

  it("rejects malformed and non-positive lengths", () => {
    expect(parseLengthInput("")).toBeNull();
    expect(parseLengthInput("2 meters later")).toBeNull();
    expect(parseLengthInput("0 mm")).toBeNull();
    expect(parseLengthInput("-3 m")).toBeNull();
  });

  it("accepts signed offsets without weakening positive length validation", () => {
    expect(parseOffsetInput("-250 mm")).toBe(-0.25);
    expect(parseOffsetInput("0")).toBe(0);
    expect(parseOffsetInput("-1 ft 6 in")).toBeCloseTo(-0.4572);
    expect(parseLengthInput("-250 mm")).toBeNull();
  });

  it("normalizes angles and resolves exact endpoints", () => {
    expect(parseAngleInput("-90 deg")).toBe(270);
    expect(parseAngleInput("450deg")).toBe(90);
    expect(endpointFromLengthAngle([1, 2], 2.5, 0)).toEqual([3.5, 2]);
    expect(endpointFromLengthAngle([1, 2], 2, 90)).toEqual([1, 4]);
  });

  it("preserves signed angles for relative CAD rotation commands", () => {
    expect(parseSignedAngleInput("-90 deg")).toBe(-90);
    expect(parseSignedAngleInput("22.5deg")).toBe(22.5);
    expect(parseSignedAngleInput("quarter turn")).toBeNull();
  });

  it("reports the current segment in the canvas coordinate system", () => {
    const result = lengthAndAngle([1, 1], [4, 5]);
    expect(result.lengthM).toBe(5);
    expect(result.angleDeg).toBeCloseTo(53.130102354);
  });
});
