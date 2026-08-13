import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DeferredJsonStorage } from "./deferredJsonStorage";

describe("deferred JSON storage", () => {
  const writes: Array<[string, string]> = [];

  beforeEach(() => {
    writes.length = 0;
    vi.useFakeTimers();
    vi.stubGlobal("localStorage", {
      setItem: (key: string, value: string) => writes.push([key, value]),
    });
    vi.stubGlobal("window", {
      clearTimeout,
      setTimeout,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("coalesces rapid editor changes into one write", () => {
    const storage = new DeferredJsonStorage<{ revision: number }>("session", 100);
    storage.schedule({ revision: 1 });
    storage.schedule({ revision: 2 });
    storage.schedule({ revision: 3 });

    vi.advanceTimersByTime(100);

    expect(writes).toEqual([["session", '{"revision":3}']]);
  });

  it("flushes the latest state synchronously during page exit", () => {
    const storage = new DeferredJsonStorage<{ saved: boolean }>("session");
    storage.schedule({ saved: true });

    storage.flush();

    expect(writes).toEqual([["session", '{"saved":true}']]);
  });
});
