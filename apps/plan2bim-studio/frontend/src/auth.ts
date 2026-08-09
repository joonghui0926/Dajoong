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

export function setBearerToken(token: string) {
  bearerToken = token;
}

export function authFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  if (!isTrustedStudioApiRequest(input)) {
    return Promise.reject(new Error("Blocked a request outside the Dajoong conversion service"));
  }
  if (!bearerToken) return fetch(input, init);
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${bearerToken}`);
  return fetch(input, { ...init, headers });
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
  if (!userManager) return;
  if (native) {
    await userManager.removeUser();
    return;
  }
  await userManager.signoutRedirect();
}
