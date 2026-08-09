import { Landing } from "./Landing";
import { Legal } from "./Legal";
import { Studio } from "./Studio";
import { AuthGate } from "./components/AuthGate";

export function App() {
  const path = window.location.pathname.replace(/\/$/, "") || "/";
  if (path === "/studio") return <AuthGate><Studio /></AuthGate>;
  if (path === "/privacy") return <Legal page="privacy" />;
  if (path === "/cookies") return <Legal page="cookies" />;
  if (path === "/terms") return <Legal page="terms" />;
  return <Landing />;
}
