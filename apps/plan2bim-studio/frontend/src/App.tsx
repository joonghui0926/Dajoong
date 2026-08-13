import { Capacitor } from "@capacitor/core";

import { resolveAppRoute } from "./appRoute";
import { AsyncFeatureBoundary, reliableLazy } from "./components/AsyncFeatureBoundary";
import { CookieBanner } from "./components/CookieBanner";

const Landing = reliableLazy(async () => ({ default: (await import("./Landing")).Landing }));
const Legal = reliableLazy(async () => ({ default: (await import("./Legal")).Legal }));
const StudioRoute = reliableLazy(async () => ({ default: (await import("./StudioRoute")).StudioRoute }));

function AppLoading() {
  return <main className="app-loading" aria-hidden="true" />;
}

export function App() {
  const route = resolveAppRoute(window.location.pathname, Capacitor.isNativePlatform());
  const embeddedLandingDemo = route === "studio"
    && new URLSearchParams(window.location.search).get("embed") === "landing";
  let content;
  if (route === "studio") content = <StudioRoute />;
  else if (route === "privacy") content = <Legal page="privacy" />;
  else if (route === "cookies") content = <Legal page="cookies" />;
  else if (route === "terms") content = <Legal page="terms" />;
  else if (route === "support") content = <Legal page="support" />;
  else if (route === "accountDeletion") content = <Legal page="accountDeletion" />;
  else content = <Landing />;
  return <><AsyncFeatureBoundary label="Dajoong workspace" fallback={<AppLoading />}>{content}</AsyncFeatureBoundary>{embeddedLandingDemo ? null : <CookieBanner />}</>;
}
