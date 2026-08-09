import { LoaderCircle } from "lucide-react";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { authConfigured, enabledAuthProviders, setBearerToken, signIn, userManager } from "../auth";
import { DajoongLogo } from "./DajoongLogo";

type AuthState = "loading" | "signed-out" | "ready" | "error";

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
      <DajoongLogo />
      <div className="auth-copy"><p className="section-kicker">DAJOONG STUDIO</p><h1>Your drawings.<br />Your building data.</h1><p>Sign in to convert plans, review model confidence, and keep corrections connected to the source.</p></div>
      {message ? <p className="auth-error">{message}</p> : null}
      <div className="auth-provider-list">
        <button onClick={() => void beginSignIn()}>Continue with email</button>
        {enabledAuthProviders.google ? <button onClick={() => void beginSignIn("Google")}>Continue with Google</button> : null}
        {enabledAuthProviders.apple ? <button onClick={() => void beginSignIn("SignInWithApple")}>Continue with Apple</button> : null}
        {enabledAuthProviders.kakao ? <button className="kakao-login" onClick={() => void beginSignIn("Kakao")}>Continue with Kakao</button> : null}
      </div>
      <a href="/privacy">Privacy and data handling</a>
    </main>
  );
}
