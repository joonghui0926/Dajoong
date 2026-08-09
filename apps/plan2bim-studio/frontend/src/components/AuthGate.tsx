import { ArrowRight, LoaderCircle, Mail, ShieldCheck } from "lucide-react";
import { siApple, siKakaotalk } from "simple-icons";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { authConfigured, enabledAuthProviders, setBearerToken, signIn, userManager } from "../auth";
import { DajoongLogo } from "./DajoongLogo";

type AuthState = "loading" | "signed-out" | "ready" | "error";

function GoogleMark() {
  return <svg viewBox="0 0 24 24" aria-hidden="true">
    <path fill="#4285F4" d="M21.6 12.23c0-.74-.07-1.45-.19-2.14H12v4.05h5.38a4.6 4.6 0 0 1-2 3.02v2.63h3.24c1.9-1.75 2.98-4.33 2.98-7.56Z" />
    <path fill="#34A853" d="M12 22c2.7 0 4.97-.9 6.62-2.43l-3.24-2.62c-.9.6-2.05.96-3.38.96-2.61 0-4.82-1.76-5.61-4.13H3.04v2.7A10 10 0 0 0 12 22Z" />
    <path fill="#FBBC05" d="M6.39 13.78A6.02 6.02 0 0 1 6.07 12c0-.62.11-1.22.32-1.78v-2.7H3.04A10 10 0 0 0 2 12c0 1.61.38 3.14 1.04 4.48l3.35-2.7Z" />
    <path fill="#EA4335" d="M12 6.09c1.47 0 2.79.5 3.83 1.5l2.87-2.88A9.62 9.62 0 0 0 12 2a10 10 0 0 0-8.96 5.52l3.35 2.7C7.18 7.85 9.39 6.09 12 6.09Z" />
  </svg>;
}

function BrandMark({ path }: { path: string }) {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d={path} /></svg>;
}

export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(authConfigured ? "loading" : "ready");
  const [message, setMessage] = useState("");
  const beginSignIn = async (provider?: "Google" | "SignInWithApple" | "Kakao") => {
    setMessage("");
    try {
      const user = await signIn(provider);
      if (user) setState("ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not complete sign in");
      setState("error");
    }
  };

  useEffect(() => {
    const manager = userManager;
    if (!manager) return;
    const initialize = async () => {
      try {
        const params = new URLSearchParams(window.location.search);
        if (params.has("code") && params.has("state")) {
          await manager.signinRedirectCallback();
          window.history.replaceState({}, document.title, "/studio");
        }
        const user = await manager.getUser();
        if (!user || user.expired) {
          setState("signed-out");
          return;
        }
        if (!user.id_token) throw new Error("Identity token was not returned");
        setBearerToken(user.id_token);
        setState("ready");
      } catch (error) {
        setMessage(error instanceof Error ? error.message : "Could not complete sign in");
        setState("error");
      }
    };
    void initialize();
  }, []);

  if (state === "ready") return children;
  if (state === "loading") return <main className="auth-page"><DajoongLogo /><LoaderCircle className="spin" /><p>Preparing your workspace...</p></main>;
  return (
    <main className="auth-page">
      <div className="auth-backdrop" aria-hidden="true">
        <span className="auth-plane auth-plane-a" />
        <span className="auth-plane auth-plane-b" />
        <span className="auth-plane auth-plane-c" />
      </div>
      <DajoongLogo />
      <div className="auth-copy"><p className="section-kicker">DAJOONG STUDIO</p><h1>Your drawings.<br />Your building data.</h1><p>Sign in to convert plans, review model confidence, and keep corrections connected to the source.</p></div>
      {message ? <p className="auth-error">{message}</p> : null}
      <div className="auth-provider-list" aria-label="Sign-in options">
        <button className="auth-provider-button auth-provider-primary" onClick={() => void beginSignIn()}>
          <span className="auth-provider-icon"><Mail size={18} strokeWidth={1.8} /></span>
          <span>Continue with email</span>
          <ArrowRight className="auth-provider-arrow" size={17} strokeWidth={1.8} />
        </button>
        <div className="auth-provider-divider" aria-hidden="true"><span>or continue with</span></div>
        {enabledAuthProviders.google ? <button className="auth-provider-button" onClick={() => void beginSignIn("Google")}>
          <span className="auth-provider-icon google-mark"><GoogleMark /></span>
          <span>Continue with Google</span>
          <span aria-hidden="true" />
        </button> : null}
        {enabledAuthProviders.apple ? <button className="auth-provider-button" onClick={() => void beginSignIn("SignInWithApple")}>
          <span className="auth-provider-icon auth-brand-symbol apple-mark"><BrandMark path={siApple.path} /></span>
          <span>Continue with Apple</span>
          <span aria-hidden="true" />
        </button> : null}
        {enabledAuthProviders.kakao ? <button className="auth-provider-button kakao-login" onClick={() => void beginSignIn("Kakao")}>
          <span className="auth-provider-icon auth-brand-symbol kakao-mark"><BrandMark path={siKakaotalk.path} /></span>
          <span>Continue with Kakao</span>
          <span aria-hidden="true" />
        </button> : null}
      </div>
      <p className="auth-trust-note"><ShieldCheck size={14} strokeWidth={1.8} />Secure workspace access · Your project data stays private</p>
      <a className="auth-privacy-link" href="/privacy">Privacy and data handling</a>
    </main>
  );
}
