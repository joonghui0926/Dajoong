import { Capacitor } from "@capacitor/core";

import { Landing } from "./Landing";
import { Legal } from "./Legal";
import { Studio } from "./Studio";
import { resolveAppRoute } from "./appRoute";
import { AuthGate } from "./components/AuthGate";
import { CookieBanner } from "./components/CookieBanner";

export function App() {
  const route = resolveAppRoute(window.location.pathname, Capacitor.isNativePlatform());
  let content;
  if (route === "studio") content = <AuthGate><Studio /></AuthGate>;
  else if (route === "privacy") content = <Legal page="privacy" />;
  else if (route === "cookies") content = <Legal page="cookies" />;
  else if (route === "terms") content = <Legal page="terms" />;
  else if (route === "support") content = <Legal page="support" />;
  else if (route === "accountDeletion") content = <Legal page="accountDeletion" />;
  else content = <Landing />;
  return <>{content}<CookieBanner /></>;
}
