export type Point2 = [number, number];

const NUMBER = "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)";

export function parseLengthInput(value: string): number | null {
  const meters = parseUnitLength(value);
  return meters !== null && meters > 0 ? meters : null;
}

export function parseOffsetInput(value: string): number | null {
  return parseUnitLength(value);
}

function parseUnitLength(value: string): number | null {
  const normalized = value.trim().toLowerCase().replaceAll(",", "");
  if (!normalized) return null;

  const feetAndInches = normalized.match(
    new RegExp(`^(${NUMBER})\\s*(?:ft|')\\s*(?:(${NUMBER})\\s*(?:in|\"))?$`),
  );
  if (feetAndInches) {
    const feet = Number(feetAndInches[1]);
    const inches = Number(feetAndInches[2] ?? 0);
    const sign = feet < 0 ? -1 : 1;
    const meters = sign * (Math.abs(feet) * 0.3048 + inches * 0.0254);
    return Number.isFinite(meters) ? roundMetric(meters) : null;
  }

  const metric = normalized.match(new RegExp(`^(${NUMBER})\\s*(mm|cm|m|in|ft)?$`));
  if (!metric) return null;
  const amount = Number(metric[1]);
  const factors: Record<string, number> = {
    mm: 0.001,
    cm: 0.01,
    m: 1,
    in: 0.0254,
    ft: 0.3048,
  };
  const meters = amount * factors[metric[2] ?? "m"];
  return Number.isFinite(meters) ? roundMetric(meters) : null;
}

export function parseAngleInput(value: string): number | null {
  const angle = parseSignedAngleInput(value);
  return angle === null ? null : normalizeAngle(angle);
}

export function parseSignedAngleInput(value: string): number | null {
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  const match = normalized.match(new RegExp(`^(${NUMBER})\\s*(?:deg)?$`));
  if (!match) return null;
  const angle = Number(match[1]);
  return Number.isFinite(angle) ? roundMetric(angle) : null;
}

export function lengthAndAngle(from: Point2, to: Point2): {
  lengthM: number;
  angleDeg: number;
} {
  return {
    lengthM: Math.hypot(to[0] - from[0], to[1] - from[1]),
    angleDeg: normalizeAngle(Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI),
  };
}

export function endpointFromLengthAngle(
  origin: Point2,
  lengthM: number,
  angleDeg: number,
): Point2 | null {
  if (!Number.isFinite(lengthM) || lengthM < 0.05 || !Number.isFinite(angleDeg)) {
    return null;
  }
  const radians = angleDeg * Math.PI / 180;
  return [
    roundMetric(origin[0] + Math.cos(radians) * lengthM),
    roundMetric(origin[1] + Math.sin(radians) * lengthM),
  ];
}

function normalizeAngle(value: number): number {
  return roundMetric(((value % 360) + 360) % 360);
}

function roundMetric(value: number): number {
  const rounded = Math.round(value * 1_000_000_000) / 1_000_000_000;
  return Object.is(rounded, -0) ? 0 : rounded;
}
