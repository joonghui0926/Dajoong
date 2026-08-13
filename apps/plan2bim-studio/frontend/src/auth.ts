import { App as CapacitorApp } from "@capacitor/app";
import { Browser } from "@capacitor/browser";
import { Capacitor, type PluginListenerHandle } from "@capacitor/core";
import { UserManager, type INavigator, type IWindow, type NavigateParams, type NavigateResponse } from "oidc-client-ts";
import { isTrustedStudioApiRequest } from "./serverApi";

const authority = import.meta.env.VITE_COGNITO_AUTHORITY;
const clientId = import.meta.env.VITE_COGNITO_CLIENT_ID;
const native = Capacitor.isNativePlatform();
const nativeRedirectUri = import.meta.env.VITE_NATIVE_REDIRECT_URI || "com.dajoong.plan2bim://auth/callback";
const nativeLogoutUri = import.meta.env.VITE_NATIVE_LOGOUT_URI || "com.dajoong.plan2bim://auth/logout";
const redirectUri = native
  ? nativeRedirectUri
  : import.meta.env.VITE_COGNITO_REDIRECT_URI || `${window.location.origin}/studio`;

let nativeResponseUrl = "";

class NativeBrowserWindow implements IWindow {
  private listener: PluginListenerHandle | null = null;
  private rejectNavigation: ((reason: Error) => void) | null = null;

  async navigate(params: NavigateParams): Promise<NavigateResponse> {
    return new Promise<NavigateResponse>((resolve, reject) => {
      this.rejectNavigation = reject;
      void CapacitorApp.addListener("appUrlOpen", async ({ url }) => {
        if (!url.startsWith(nativeRedirectUri) && !url.startsWith(nativeLogoutUri)) return;
        nativeResponseUrl = url;
        await this.listener?.remove();
        this.listener = null;
        await Browser.close().catch(() => undefined);
        resolve({ url });
      }).then((listener) => {
        this.listener = listener;
        return Browser.open({ url: params.url, presentationStyle: "popover" });
      }).catch(reject);
    });
  }

  close() {
    void this.listener?.remove();
    this.listener = null;
    void Browser.close().catch(() => undefined);
    this.rejectNavigation?.(new Error("Authentication navigation was cancelled"));
    this.rejectNavigation = null;
  }
}

class NativeRedirectNavigator implements INavigator {
  async prepare(): Promise<IWindow> {
    return new NativeBrowserWindow();
  }

  async callback(): Promise<void> {
    return;
  }
}

export const authConfigured = Boolean(authority && clientId);
export const enabledAuthProviders = {
  email: true,
  google: import.meta.env.VITE_AUTH_GOOGLE_ENABLED === "true",
  apple: import.meta.env.VITE_AUTH_APPLE_ENABLED === "true",
  kakao: import.meta.env.VITE_AUTH_KAKAO_ENABLED === "true",
} as const;

const settings = authConfigured ? {
  authority,
  client_id: clientId,
  redirect_uri: redirectUri,
  post_logout_redirect_uri: native ? nativeLogoutUri : window.location.origin,
  response_type: "code",
  scope: "openid email profile",
  automaticSilentRenew: !native,
} : null;

export const userManager = settings
  ? new UserManager(settings, native ? new NativeRedirectNavigator() : undefined)
  : null;

let bearerToken = "";
const ACTIVE_ORGANIZATION_KEY = "dajoong-active-organization-v1";

export function setBearerToken(token: string) {
  bearerToken = token;
}

export function getActiveOrganizationId(): string {
  return localStorage.getItem(ACTIVE_ORGANIZATION_KEY) ?? "";
}

export function setActiveOrganizationId(organizationId: string) {
  if (organizationId) localStorage.setItem(ACTIVE_ORGANIZATION_KEY, organizationId);
  else localStorage.removeItem(ACTIVE_ORGANIZATION_KEY);
  window.dispatchEvent(new CustomEvent("dajoong:workspace-change", { detail: organizationId }));
}

function waitBeforeRetry(delayMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new DOMException("Request cancelled", "AbortError"));
      return;
    }
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", cancel);
      resolve();
    }, delayMs);
    const cancel = () => {
      window.clearTimeout(timer);
      reject(signal?.reason ?? new DOMException("Request cancelled", "AbortError"));
    };
    signal?.addEventListener("abort", cancel, { once: true });
  });
}

export async function authFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = 30_000,
): Promise<Response> {
  if (!isTrustedStudioApiRequest(input)) {
    throw new Error("Blocked a request outside the Dajoong conversion service");
  }
  const headers = new Headers(init.headers);
  const callerSignal = init.signal ?? undefined;
  if (bearerToken) headers.set("Authorization", `Bearer ${bearerToken}`);
  const organizationId = getActiveOrganizationId();
  if (organizationId) headers.set("X-Dajoong-Organization", organizationId);
  const method = (init.method ?? "GET").toUpperCase();
  const retryable = method === "GET" || method === "HEAD" || headers.has("Idempotency-Key");
  const attempts = retryable ? (method === "GET" || method === "HEAD" ? 3 : 2) : 1;
  let lastError: unknown;
  let lastTimedOut = false;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    let timedOut = false;
    const cancel = () => controller.abort(callerSignal?.reason);
    if (callerSignal?.aborted) cancel();
    else callerSignal?.addEventListener("abort", cancel, { once: true });
    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeoutMs);
    try {
      const response = await fetch(input, { ...init, headers, signal: controller.signal });
      const transient = [429, 502, 503, 504].includes(response.status);
      if (!transient || attempt + 1 >= attempts) return response;
      await response.body?.cancel().catch(() => undefined);
      const retryAfterSeconds = Number(response.headers.get("Retry-After") ?? "");
      const delay = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0
        ? Math.min(5_000, retryAfterSeconds * 1_000)
        : 350 * (2 ** attempt);
      await waitBeforeRetry(delay, callerSignal);
    } catch (error) {
      if (callerSignal?.aborted) throw error;
      lastError = error;
      lastTimedOut = timedOut;
      if (attempt + 1 >= attempts) break;
      await waitBeforeRetry(350 * (2 ** attempt), callerSignal);
    } finally {
      window.clearTimeout(timer);
      callerSignal?.removeEventListener("abort", cancel);
    }
  }
  if (lastTimedOut) throw new Error("The Dajoong service took too long to respond. Please try again.");
  throw lastError;
}

export async function signIn(provider?: "Google" | "SignInWithApple" | "Kakao") {
  if (!userManager) throw new Error("Authentication is not configured");
  nativeResponseUrl = "";
  await userManager.signinRedirect(provider ? { extraQueryParams: { identity_provider: provider } } : undefined);
  if (!native) return null;
  if (!nativeResponseUrl) throw new Error("The identity provider did not return to Dajoong");
  const user = await userManager.signinRedirectCallback(nativeResponseUrl);
  if (!user.id_token) throw new Error("Identity token was not returned");
  setBearerToken(user.id_token);
  return user;
}

export async function signOut() {
  setBearerToken("");
  setActiveOrganizationId("");
  if (!userManager) return;
  if (native) {
    await userManager.removeUser();
    return;
  }
  await userManager.signoutRedirect();
}
