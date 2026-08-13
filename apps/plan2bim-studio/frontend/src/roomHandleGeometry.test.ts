import { describe, expect, it } from "vitest";

import { MAX_VISIBLE_ROOM_HANDLES, visibleRoomHandleIndices } from "./roomHandleGeometry";

describe("room handle geometry", () => {
  it("keeps every edit point on a deliberately simple room", () => {
    expect(visibleRoomHandleIndices([[0, 0], [4, 0], [4, 3], [0, 3]])).toEqual([0, 1, 2, 3]);
  });

  it("reduces a pixel-traced room to meaningful corners", () => {
    const polygon: [number, number][] = [];
    for (let index = 0; index < 32; index += 1) polygon.push([index / 8, 0]);
    for (let index = 0; index < 32; index += 1) polygon.push([4, index / 8]);
    for (let index = 0; index < 32; index += 1) polygon.push([4 - index / 8, 4]);
    for (let index = 0; index < 32; index += 1) polygon.push([0, 4 - index / 8]);

    const indices = visibleRoomHandleIndices(polygon);
    expect(indices.length).toBeLessThanOrEqual(MAX_VISIBLE_ROOM_HANDLES);
    expect(indices).toEqual(expect.arrayContaining([0, 32, 64, 96]));
  });
});
