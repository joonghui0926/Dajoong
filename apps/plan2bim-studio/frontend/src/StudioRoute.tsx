import { AuthGate } from "./components/AuthGate";
import { Studio } from "./Studio";

export function StudioRoute() {
  if (new URLSearchParams(window.location.search).get("embed") === "landing") {
    return <Studio />;
  }
  return <AuthGate><Studio /></AuthGate>;
}
