import { describe, expect, it } from "vitest";

import { isTrustedStudioApiRequest, resolveStudioApiOrigin, studioApiUrl } from "./serverApi";

describe("Dajoong server-only client boundary", () => {
  it("defaults production builds to the canonical conversion API", () => {
    expect(resolveStudioApiOrigin(undefined, false)).toBe("https://studio-api.builiconstruction.com");
  });

  it("allows a loopback server only in development", () => {
    expect(resolveStudioApiOrigin("http://127.0.0.1:8042", true)).toBe("http://127.0.0.1:8042");
    expect(() => resolveStudioApiOrigin("http://127.0.0.1:8042", false)).toThrow(/only use/i);
  });

  it("rejects alternate production inference services", () => {
    expect(() => resolveStudioApiOrigin("https://example.com", false)).toThrow(/Dajoong/i);
    expect(() => resolveStudioApiOrigin("https://studio-api.builiconstruction.com/proxy", false)).toThrow(/origin only/i);
  });

  it("builds and recognizes only canonical API requests", () => {
    expect(studioApiUrl("/api/jobs")).toMatch(/\/api\/jobs$/);
    expect(isTrustedStudioApiRequest(studioApiUrl("/api/jobs"))).toBe(true);
    expect(isTrustedStudioApiRequest("https://example.com/api/jobs")).toBe(false);
    expect(() => studioApiUrl("/models/private.onnx")).toThrow(/\/api\//i);
  });
});
