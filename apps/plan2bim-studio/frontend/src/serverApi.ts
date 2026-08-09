const PRODUCTION_STUDIO_API_ORIGIN = "https://studio-api.builiconstruction.com";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

function normalizedOrigin(value: string): string {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash || (url.pathname && url.pathname !== "/")) {
    throw new Error("Studio API configuration must contain an origin only");
  }
  return url.origin;
}

export function resolveStudioApiOrigin(
  configured: string | undefined,
  development: boolean,
): string {
  const candidate = configured?.trim()
    || (development ? "http://127.0.0.1:8042" : PRODUCTION_STUDIO_API_ORIGIN);
  const origin = normalizedOrigin(candidate);
  const url = new URL(origin);
  if (development && LOOPBACK_HOSTS.has(url.hostname)) return origin;
  if (url.protocol !== "https:" || origin !== PRODUCTION_STUDIO_API_ORIGIN) {
    throw new Error(
      `Studio clients may only use the BU iLI conversion service at ${PRODUCTION_STUDIO_API_ORIGIN}`,
    );
  }
  return origin;
}

export const STUDIO_API_ORIGIN = resolveStudioApiOrigin(
  import.meta.env.VITE_STUDIO_API_URL,
  import.meta.env.DEV,
);

export function studioApiUrl(path: string): string {
  if (!path.startsWith("/api/")) throw new Error("Studio API paths must start with /api/");
  return `${STUDIO_API_ORIGIN}${path}`;
}

export function isTrustedStudioApiRequest(input: RequestInfo | URL): boolean {
  const raw = input instanceof Request ? input.url : String(input);
  try {
    const url = new URL(raw, STUDIO_API_ORIGIN);
    return url.origin === STUDIO_API_ORIGIN && url.pathname.startsWith("/api/");
  } catch {
    return false;
  }
}
