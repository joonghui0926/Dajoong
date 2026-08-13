import { ArrowLeft, ArrowRight, Check, Eye, EyeOff, LoaderCircle, LockKeyhole, Mail, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import {
  authConfigured,
  confirmEmailSignUp,
  confirmPasswordReset,
  googleAuthEnabled,
  requestPasswordReset,
  resendEmailConfirmation,
  restoreEmailSession,
  setBearerToken,
  signInWithEmail,
  signInWithGoogle,
  signUpWithEmail,
  userManager,
} from "../auth";
import { DajoongLogo } from "./DajoongLogo";

type AuthState = "loading" | "signed-out" | "ready" | "error";
type AuthView = "choose" | "sign-in" | "sign-up" | "confirm" | "forgot" | "reset";
const PENDING_INVITE_KEY = "dajoong-pending-invite-v1";

function invitationFromLocation() {
  return new URLSearchParams(window.location.search).get("invite")?.trim() ?? "";
}

function GoogleMark() {
  return <svg viewBox="0 0 24 24" aria-hidden="true">
    <path fill="#4285F4" d="M21.6 12.23c0-.74-.07-1.45-.19-2.14H12v4.05h5.38a4.6 4.6 0 0 1-2 3.02v2.63h3.24c1.9-1.75 2.98-4.33 2.98-7.56Z" />
    <path fill="#34A853" d="M12 22c2.7 0 4.97-.9 6.62-2.43l-3.24-2.62c-.9.6-2.05.96-3.38.96-2.61 0-4.82-1.76-5.61-4.13H3.04v2.7A10 10 0 0 0 12 22Z" />
    <path fill="#FBBC05" d="M6.39 13.78A6.02 6.02 0 0 1 6.07 12c0-.62.11-1.22.32-1.78v-2.7H3.04A10 10 0 0 0 2 12c0 1.61.38 3.14 1.04 4.48l3.35-2.7Z" />
    <path fill="#EA4335" d="M12 6.09c1.47 0 2.79.5 3.83 1.5l2.87-2.88A9.62 9.62 0 0 0 12 2a10 10 0 0 0-8.96 5.52l3.35 2.7C7.18 7.85 9.39 6.09 12 6.09Z" />
  </svg>;
}

export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(authConfigured ? "loading" : "ready");
  const [view, setView] = useState<AuthView>("choose");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [notice, setNotice] = useState("");
  const pendingInvitation = Boolean(invitationFromLocation() || sessionStorage.getItem(PENDING_INVITE_KEY));

  const rememberInvitation = () => {
    const invitation = invitationFromLocation();
    if (invitation) sessionStorage.setItem(PENDING_INVITE_KEY, invitation);
  };

  const moveTo = (next: AuthView) => {
    setMessage("");
    setNotice("");
    setPassword("");
    setVerificationCode("");
    setView(next);
  };

  const beginGoogleSignIn = async () => {
    setMessage("");
    try {
      rememberInvitation();
      const user = await signInWithGoogle();
      if (user) setState("ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not complete sign in");
      setState("error");
    }
  };

  const submitEmail = async (event: FormEvent) => {
    event.preventDefault();
    if (!email.trim()) {
      setMessage("Enter your email address.");
      return;
    }
    setBusy(true);
    setMessage("");
    setNotice("");
    try {
      rememberInvitation();
      if (view === "sign-in") {
        await signInWithEmail(email, password);
        setState("ready");
      } else if (view === "sign-up") {
        await signUpWithEmail(email, password);
        setView("confirm");
        setNotice(`We sent a verification code to ${email.trim().toLowerCase()}.`);
      } else if (view === "confirm") {
        await confirmEmailSignUp(email, verificationCode);
        await signInWithEmail(email, password);
        setState("ready");
      } else if (view === "forgot") {
        await requestPasswordReset(email);
        setView("reset");
        setNotice(`We sent a reset code to ${email.trim().toLowerCase()}.`);
      } else if (view === "reset") {
        await confirmPasswordReset(email, verificationCode, password);
        setView("sign-in");
        setPassword("");
        setVerificationCode("");
        setNotice("Password updated. Sign in with your new password.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not complete authentication");
    } finally {
      setBusy(false);
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
          const invitation = sessionStorage.getItem(PENDING_INVITE_KEY) ?? "";
          const restoredPath = invitation ? `/studio?invite=${encodeURIComponent(invitation)}` : "/studio";
          window.history.replaceState({}, document.title, restoredPath);
        }
        const user = await manager.getUser();
        if (!user || user.expired) {
          setState(await restoreEmailSession() ? "ready" : "signed-out");
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
  if (state === "loading") return <main className="auth-page auth-loading"><DajoongLogo /><LoaderCircle className="spin" /><p>Preparing your workspace...</p></main>;

  const hasCode = view === "confirm" || view === "reset";
  const needsPassword = view === "sign-in" || view === "sign-up" || view === "confirm" || view === "reset";
  const title = view === "sign-up" ? "Create your workspace."
    : view === "confirm" ? "Check your inbox."
      : view === "forgot" ? "Reset your password."
        : view === "reset" ? "Choose a new password."
          : "Welcome back.";
  const subtitle = view === "sign-up" ? "One account keeps every drawing, model, and review connected."
    : view === "confirm" ? "Enter the six-digit code we sent to finish creating your account."
      : view === "forgot" ? "We’ll send a secure verification code to your email."
        : view === "reset" ? "Enter the code and a new password for your Dajoong account."
          : pendingInvitation ? "Sign in to join the company workspace that invited you."
            : "Continue where your drawings and building data left off.";

  return (
    <main className="auth-page">
      <div className="auth-backdrop" aria-hidden="true"><span className="auth-plane auth-plane-a" /><span className="auth-plane auth-plane-b" /><span className="auth-plane auth-plane-c" /></div>
      <header className="auth-brand"><DajoongLogo /><span>PLAN2BIM STUDIO</span></header>
      <section className="auth-shell" aria-live="polite">
        <div className="auth-intro">
          <p className="section-kicker">SECURE DAJOONG WORKSPACE</p>
          <h1>{view === "choose" ? <>Your drawings.<br />Your building data.</> : title}</h1>
          <p>{view === "choose" ? (pendingInvitation ? "Join the team workspace that invited you." : "Convert plans, review every element, and keep decisions connected to the source.") : subtitle}</p>
          <div className="auth-trust-list"><span><Check size={14} />Source-linked model history</span><span><Check size={14} />Private company workspaces</span></div>
        </div>

        <div className="auth-card">
          {view === "choose" ? <>
            <div className="auth-card-heading"><span>ACCESS YOUR STUDIO</span><h2>Sign in to Dajoong</h2><p>Choose one secure method to continue.</p></div>
            <div className="auth-provider-list" aria-label="Sign-in options">
              <button className="auth-provider-button auth-provider-primary" onClick={() => moveTo("sign-in")}>
                <span className="auth-provider-icon"><Mail size={19} strokeWidth={1.8} /></span><span>Continue with email</span><ArrowRight className="auth-provider-arrow" size={17} strokeWidth={1.8} />
              </button>
              {googleAuthEnabled ? <button className="auth-provider-button auth-google-button" onClick={() => void beginGoogleSignIn()}>
                <span className="auth-provider-icon google-mark"><GoogleMark /></span><span>Continue with Google</span><span aria-hidden="true" />
              </button> : null}
            </div>
            <p className="auth-switch-copy">New to Dajoong? <button type="button" onClick={() => moveTo("sign-up")}>Create an account</button></p>
          </> : <>
            <button className="auth-back-button" type="button" onClick={() => moveTo("choose")}><ArrowLeft size={15} />All sign-in options</button>
            <div className="auth-card-heading"><span>{view === "sign-up" ? "NEW ACCOUNT" : view === "sign-in" ? "WELCOME BACK" : "VERIFY YOUR EMAIL"}</span><h2>{title}</h2><p>{subtitle}</p></div>
            <form className="auth-email-form" onSubmit={submitEmail}>
              <label>Email address<div><Mail size={17} /><input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@company.com" autoComplete="email" required disabled={view === "confirm" || view === "reset"} /></div></label>
              {hasCode ? <label>Verification code<div><ShieldCheck size={17} /><input value={verificationCode} onChange={(event) => setVerificationCode(event.target.value.replace(/\D/g, "").slice(0, 6))} inputMode="numeric" autoComplete="one-time-code" placeholder="000000" required /></div></label> : null}
              {needsPassword ? <label>{view === "reset" ? "New password" : "Password"}<div><LockKeyhole size={17} /><input type={showPassword ? "text" : "password"} value={password} onChange={(event) => setPassword(event.target.value)} placeholder={view === "sign-in" ? "Enter your password" : "12+ characters"} autoComplete={view === "sign-in" ? "current-password" : "new-password"} minLength={view === "sign-in" ? undefined : 12} required /><button type="button" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? "Hide password" : "Show password"}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div></label> : null}
              {view === "sign-in" ? <button className="auth-forgot-link" type="button" onClick={() => moveTo("forgot")}>Forgot password?</button> : null}
              {notice ? <p className="auth-notice" role="status">{notice}</p> : null}
              {message ? <p className="auth-error" role="alert">{message}</p> : null}
              <button className="auth-submit-button" type="submit" disabled={busy}>{busy ? <><LoaderCircle className="spin" size={17} />Working…</> : view === "sign-in" ? <>Sign in <ArrowRight size={17} /></> : view === "sign-up" ? <>Create account <ArrowRight size={17} /></> : view === "confirm" ? <>Verify and sign in <ArrowRight size={17} /></> : view === "forgot" ? <>Send reset code <ArrowRight size={17} /></> : <>Update password <ArrowRight size={17} /></>}</button>
              {view === "confirm" ? <button className="auth-resend-link" type="button" onClick={() => void resendEmailConfirmation(email).then(() => setNotice("A new code is on its way.")).catch((error: Error) => setMessage(error.message))}>Send a new code</button> : null}
            </form>
            <p className="auth-switch-copy">{view === "sign-up" ? <>Already have an account? <button type="button" onClick={() => moveTo("sign-in")}>Sign in</button></> : view === "sign-in" ? <>New to Dajoong? <button type="button" onClick={() => moveTo("sign-up")}>Create an account</button></> : null}</p>
          </>}
        </div>
      </section>
      <footer className="auth-footer"><p><ShieldCheck size={14} />Secure workspace access · Company data stays with the company</p><a href="/privacy">Privacy and data handling</a></footer>
    </main>
  );
}
