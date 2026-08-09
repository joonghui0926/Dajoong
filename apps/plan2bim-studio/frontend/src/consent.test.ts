import { describe, expect, it, vi } from 'vitest';
import { CONSENT_POLICY_VERSION, hasGlobalPrivacyControl, readConsent } from './consent';

describe('privacy consent', () => {
  it('fails closed for stale or malformed records', () => {
    expect(readConsent({ getItem: () => '{broken' })).toBeNull();
    expect(readConsent({ getItem: () => JSON.stringify({ essential: true, analytics: true, policyVersion: 'old', recordedAt: 'now' }) })).toBeNull();
  });

  it('accepts only the current explicit choice', () => {
    const value = { essential: true as const, analytics: false, policyVersion: CONSENT_POLICY_VERSION, recordedAt: '2026-08-09T00:00:00.000Z' };
    expect(readConsent({ getItem: vi.fn(() => JSON.stringify(value)) })).toEqual(value);
  });

  it('detects Global Privacy Control', () => {
    expect(hasGlobalPrivacyControl({ globalPrivacyControl: true } as unknown as Navigator)).toBe(true);
  });
});
