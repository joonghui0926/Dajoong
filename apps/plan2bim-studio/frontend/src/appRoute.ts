export type AppRoute = "landing" | "studio" | "privacy" | "cookies" | "terms" | "support" | "accountDeletion";

export function resolveAppRoute(pathname: string, native: boolean): AppRoute {
  const path = pathname.replace(/\/$/, "") || "/";
  if (path === "/studio" || (native && (path === "/" || path === "/index.html"))) return "studio";
  if (path === "/privacy") return "privacy";
  if (path === "/cookies") return "cookies";
  if (path === "/terms") return "terms";
  if (path === "/support") return "support";
  if (path === "/account-deletion") return "accountDeletion";
  return "landing";
}
