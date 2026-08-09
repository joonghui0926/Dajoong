export const CONSENT_KEY = 'dajoong-cookie-consent-v2';
export const CONSENT_POLICY_VERSION = '2026-08-09';

export type ConsentRecord = {
  essential: true;
  analytics: boolean;
  policyVersion: string;
  recordedAt: string;
};

export function readConsent(storage: Pick<Storage, 'getItem'> = localStorage): ConsentRecord | null {
  try {
    const raw = storage.getItem(CONSENT_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<ConsentRecord>;
    if (value.essential !== true || value.policyVersion !== CONSENT_POLICY_VERSION) return null;
    if (typeof value.analytics !== 'boolean' || typeof value.recordedAt !== 'string') return null;
    return value as ConsentRecord;
  } catch {
    return null;
  }
}

export function hasGlobalPrivacyControl(nav: Navigator = navigator) {
  return (nav as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl === true;
}

export function saveConsent(analytics: boolean, storage: Pick<Storage, 'setItem'> = localStorage) {
  const value: ConsentRecord = {
    essential: true,
    analytics: hasGlobalPrivacyControl() ? false : analytics,
    policyVersion: CONSENT_POLICY_VERSION,
    recordedAt: new Date().toISOString(),
  };
  storage.setItem(CONSENT_KEY, JSON.stringify(value));
  window.dispatchEvent(new CustomEvent('dajoong:consent-changed', { detail: value }));
  return value;
}
