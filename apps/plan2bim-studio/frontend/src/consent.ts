export const CONSENT_KEY = 'dajoong-cookie-consent-v2';
export const CONSENT_COOKIE_KEY = 'dajoong_cookie_consent';
export const CONSENT_POLICY_VERSION = '2026-08-09';
const CONSENT_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 400;

export type ConsentRecord = {
  essential: true;
  analytics: boolean;
  policyVersion: string;
  recordedAt: string;
};

function parseConsent(raw: string | null): ConsentRecord | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<ConsentRecord>;
    if (value.essential !== true) return null;
    if (typeof value.analytics !== 'boolean' || typeof value.recordedAt !== 'string') return null;
    return {
      essential: true,
      analytics: value.analytics,
      policyVersion: typeof value.policyVersion === 'string' ? value.policyVersion : CONSENT_POLICY_VERSION,
      recordedAt: value.recordedAt,
    };
  } catch {
    return null;
  }
}

function readConsentCookie(cookieHeader: string) {
  const pair = cookieHeader.split(';').map((part) => part.trim()).find((part) => part.startsWith(`${CONSENT_COOKIE_KEY}=`));
  if (!pair) return null;
  try {
    return parseConsent(decodeURIComponent(pair.slice(CONSENT_COOKIE_KEY.length + 1)));
  } catch {
    return null;
  }
}

export function readConsent(
  storage: Pick<Storage, 'getItem'> = localStorage,
  cookieHeader: string = typeof document === 'undefined' ? '' : document.cookie,
): ConsentRecord | null {
  try {
    return parseConsent(storage.getItem(CONSENT_KEY)) ?? readConsentCookie(cookieHeader);
  } catch {
    return readConsentCookie(cookieHeader);
  }
}

export function hasGlobalPrivacyControl(nav: Navigator = navigator) {
  return (nav as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl === true;
}

function writeConsentCookie(value: ConsentRecord) {
  if (typeof document === 'undefined') return;
  const hostname = window.location.hostname.toLowerCase();
  const sharedDomain = hostname === 'dajoongbim.com' || hostname.endsWith('.dajoongbim.com')
    ? '; Domain=.dajoongbim.com'
    : '';
  const secure = window.location.protocol === 'https:' ? '; Secure' : '';
  document.cookie = `${CONSENT_COOKIE_KEY}=${encodeURIComponent(JSON.stringify(value))}; Path=/; Max-Age=${CONSENT_COOKIE_MAX_AGE_SECONDS}; SameSite=Lax${sharedDomain}${secure}`;
}

export function persistConsent(value: ConsentRecord, storage: Pick<Storage, 'setItem'> = localStorage) {
  try {
    storage.setItem(CONSENT_KEY, JSON.stringify(value));
  } catch {
    // The shared first-party cookie remains available when storage is restricted.
  }
  writeConsentCookie(value);
  return value;
}

export function saveConsent(analytics: boolean, storage: Pick<Storage, 'setItem'> = localStorage) {
  const value = persistConsent({
    essential: true,
    analytics: hasGlobalPrivacyControl() ? false : analytics,
    policyVersion: CONSENT_POLICY_VERSION,
    recordedAt: new Date().toISOString(),
  }, storage);
  window.dispatchEvent(new CustomEvent('dajoong:consent-changed', { detail: value }));
  return value;
}
