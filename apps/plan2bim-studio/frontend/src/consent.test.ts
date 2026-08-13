import { describe, expect, it, vi } from 'vitest';
import { CONSENT_COOKIE_KEY, CONSENT_POLICY_VERSION, hasGlobalPrivacyControl, readConsent } from './consent';

describe('privacy consent', () => {
  it('fails closed for malformed records but keeps an earlier explicit choice', () => {
    expect(readConsent({ getItem: () => '{broken' })).toBeNull();
    expect(readConsent({ getItem: () => JSON.stringify({ essential: true, analytics: true, policyVersion: 'old', recordedAt: 'now' }) })?.analytics).toBe(true);
  });

  it('accepts only the current explicit choice', () => {
    const value = { essential: true as const, analytics: false, policyVersion: CONSENT_POLICY_VERSION, recordedAt: '2026-08-09T00:00:00.000Z' };
    expect(readConsent({ getItem: vi.fn(() => JSON.stringify(value)) })).toEqual(value);
  });

  it('detects Global Privacy Control', () => {
    expect(hasGlobalPrivacyControl({ globalPrivacyControl: true } as unknown as Navigator)).toBe(true);
  });

  it('reads the shared first-party cookie when local storage is empty', () => {
    const value = { essential: true as const, analytics: true, policyVersion: CONSENT_POLICY_VERSION, recordedAt: '2026-08-09T00:00:00.000Z' };
    const cookie = `${CONSENT_COOKIE_KEY}=${encodeURIComponent(JSON.stringify(value))}`;
    expect(readConsent({ getItem: () => null }, cookie)).toEqual(value);
  });
});
