import { Capacitor } from "@capacitor/core";

import { authFetch } from "./auth";
import { studioApiUrl } from "./serverApi";

export type BillingProvider = "stripe" | "toss" | "apple" | "google";
export type PurchasePlan = "per_drawing" | "unlimited_monthly";

export interface CheckoutContext {
  country: string;
  currency: "USD" | "KRW";
  unit_amount: number;
  unit_label: string;
  monthly_amount: number;
  monthly_label: string;
  unlimited_active: boolean;
  unlimited_until: number;
  free_units_remaining: number;
  paid_units: number;
  billing_enforced: boolean;
  configured_providers: BillingProvider[];
  native_provider: "apple" | "google" | "";
  comparison_multiple: number;
  comparison_basis: string;
  comparison_source_url: string;
  monthly_comparison_multiple: number;
  monthly_comparison_basis: string;
  monthly_comparison_source_url: string;
  speed_median_seconds: number;
  speed_p95_seconds: number;
  speed_runs: number;
  speed_comparison_multiple: number;
  speed_benchmark_url: string;
  speed_turnaround_source_url: string;
}

export interface CheckoutResponse {
  kind: "redirect" | "toss_payment";
  order_id: string;
  redirect_url: string;
  toss: {
    client_key?: string;
    customer_key?: string;
    method?: "CARD";
    easy_pay?: string;
    amount?: { value: number; currency: "KRW" };
    order_id?: string;
    order_name?: string;
    success_url?: string;
    fail_url?: string;
  };
}

const benchmarkDefaults = {
  monthly_amount: 7_900,
  monthly_label: "$79 / month",
  unlimited_active: false,
  unlimited_until: 0,
  monthly_comparison_multiple: 4.81,
  monthly_comparison_basis: "Autodesk Revit standard monthly subscription ($380/month)",
  monthly_comparison_source_url: "https://www.autodesk.com/solutions/revit-subscription-faq",
  speed_median_seconds: 2.720126,
  speed_p95_seconds: 12.972123,
  speed_runs: 7,
  speed_comparison_multiple: 21_175,
  speed_benchmark_url: "/benchmarks/plan2bim-speed-2026-08-09.json",
  speed_turnaround_source_url: "https://support.twindo.com/article/721-how-long-will-it-take-to-receive-my-order",
} as const;

function normalizeCheckoutContext(context: CheckoutContext): CheckoutContext {
  return { ...benchmarkDefaults, ...context };
}

function nativePlatform() {
  const platform = Capacitor.getPlatform();
  return platform === "ios" || platform === "android" ? platform : "web";
}

export async function loadCheckoutContext(country = "", signal?: AbortSignal): Promise<CheckoutContext> {
  const query = new URLSearchParams({ platform: nativePlatform() });
  if (country) query.set("country", country);
  const response = await authFetch(studioApiUrl(`/api/billing/context?${query}`), { signal });
  if (!response.ok) throw new Error("Could not load checkout details.");
  return normalizeCheckoutContext(await response.json() as CheckoutContext);
}

export async function createCheckout(
  provider: "stripe" | "toss",
  country: string,
  units: number,
  idempotencyKey: string,
  easyPay = "",
  plan: PurchasePlan = "per_drawing",
): Promise<CheckoutResponse> {
  const response = await authFetch(studioApiUrl("/api/billing/checkout"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify({ provider, country, units, plan, easy_pay: easyPay }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail || "The payment provider is temporarily unavailable.");
  }
  return response.json() as Promise<CheckoutResponse>;
}

export async function completeTossReturn(): Promise<boolean> {
  const query = new URLSearchParams(window.location.search);
  if (query.get("checkout") !== "toss-success") return false;
  const orderId = query.get("orderId") || "";
  const paymentKey = query.get("paymentKey") || "";
  const amount = Number(query.get("amount"));
  if (!orderId || !paymentKey || !Number.isSafeInteger(amount) || amount <= 0) return false;
  const response = await authFetch(studioApiUrl("/api/billing/toss/confirm"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order_id: orderId, payment_key: paymentKey, amount }),
  });
  if (!response.ok) throw new Error("Payment was authorized but could not be confirmed.");
  query.delete("checkout");
  query.delete("paymentType");
  query.delete("paymentKey");
  query.delete("orderId");
  query.delete("amount");
  const suffix = query.toString();
  window.history.replaceState({}, "", `${window.location.pathname}${suffix ? `?${suffix}` : ""}`);
  return true;
}

export async function checkoutContextFromPaymentRequired(
  response: Response,
): Promise<{ context: CheckoutContext; requiredUnits: number } | null> {
  if (response.status !== 402) return null;
  const payload = await response.json().catch(() => null) as {
    detail?: { checkout?: CheckoutContext; required_units?: number };
  } | null;
  if (!payload?.detail?.checkout) return null;
  return {
    context: normalizeCheckoutContext(payload.detail.checkout),
    requiredUnits: Math.max(1, payload.detail.required_units || 1),
  };
}
