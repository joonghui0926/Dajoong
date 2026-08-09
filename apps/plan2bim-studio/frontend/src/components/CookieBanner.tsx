import { useState } from "react";
import { hasGlobalPrivacyControl, readConsent, saveConsent } from "../consent";

export function CookieBanner() {
  const [visible, setVisible] = useState(() => !readConsent());
  const [gpc] = useState(() => hasGlobalPrivacyControl());
  if (!visible) return null;

  const choose = (analytics: boolean) => {
    saveConsent(analytics);
    setVisible(false);
  };

  return (
    <aside className="cookie-banner" aria-label="Cookie choices" role="dialog" aria-live="polite">
      <div>
        <strong>Your privacy, clearly handled.</strong>
        <p>Essential storage keeps your local Studio session. Optional analytics stays off until you allow it and never reads project drawings.</p>
        {gpc ? <small>Global Privacy Control is active. Optional analytics stays off.</small> : null}
        <a href="/cookies">Cookie policy</a><span> / </span><a href="/privacy">Privacy</a>
      </div>
      <div className="cookie-actions">
        <button onClick={() => choose(false)}>Essential only</button>
        <button className="cookie-accept" disabled={gpc} onClick={() => choose(true)}>Allow analytics</button>
      </div>
    </aside>
  );
}
