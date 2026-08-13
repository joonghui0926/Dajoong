import { Capacitor } from "@capacitor/core";

import { resolveAppRoute } from "./appRoute";
import { AsyncFeatureBoundary, reliableLazy } from "./components/AsyncFeatureBoundary";
import { CookieBanner } from "./components/CookieBanner";
import { DajoongLogo } from "./components/DajoongLogo";

const Landing = reliableLazy(async () => ({ default: (await import("./Landing")).Landing }));
const Legal = reliableLazy(async () => ({ default: (await import("./Legal")).Legal }));
const StudioRoute = reliableLazy(async () => ({ default: (await import("./StudioRoute")).StudioRoute }));

function AppLoading() {
  return (
    <main className="app-loading" role="status" aria-live="polite">
      <DajoongLogo />
      <span />
      <p>Preparing your workspace</p>
    </main>
  );
}

export function App() {
  const route = resolveAppRoute(window.location.pathname, Capacitor.isNativePlatform());
  let content;
  if (route === "studio") content = <StudioRoute />;
  else if (route === "privacy") content = <Legal page="privacy" />;
  else if (route === "cookies") content = <Legal page="cookies" />;
  else if (route === "terms") content = <Legal page="terms" />;
  else if (route === "support") content = <Legal page="support" />;
  else if (route === "accountDeletion") content = <Legal page="accountDeletion" />;
  else content = <Landing />;
  return <><AsyncFeatureBoundary label="Dajoong workspace" fallback={<AppLoading />}>{content}</AsyncFeatureBoundary><CookieBanner /></>;
}
