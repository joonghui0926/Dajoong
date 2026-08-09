import { describe, expect, it } from "vitest";

import { resolveAppRoute } from "./appRoute";

describe("application routing", () => {
  it("opens the product Studio directly in installed native apps", () => {
    expect(resolveAppRoute("/", true)).toBe("studio");
    expect(resolveAppRoute("/index.html", true)).toBe("studio");
  });

  it("keeps the public landing page at the web root", () => {
    expect(resolveAppRoute("/", false)).toBe("landing");
  });

  it("publishes support, privacy, and self-service deletion routes", () => {
    expect(resolveAppRoute("/privacy", false)).toBe("privacy");
    expect(resolveAppRoute("/support", false)).toBe("support");
    expect(resolveAppRoute("/account-deletion", false)).toBe("accountDeletion");
  });
});
