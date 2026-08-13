import { loadTossPayments } from "@tosspayments/tosspayments-sdk";
import {
  ArrowRight,
  BadgeCheck,
  Check,
  ChevronDown,
  CreditCard,
  FileBox,
  LoaderCircle,
  LockKeyhole,
  ReceiptText,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Timer,
  X,
} from "lucide-react";
import kakaoMarkUrl from "simple-icons/icons/kakao.svg";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  createCheckout,
  loadCheckoutContext,
  type CheckoutContext,
  type PurchasePlan,
} from "../billing";

type Method = "card" | "kakao" | "toss" | "apple" | "google";

function methodsForContext(context: CheckoutContext): Method[] {
  if (
    context.native_provider &&
    context.configured_providers.includes(context.native_provider)
  ) return [context.native_provider];
  if (context.country === "KR" && context.configured_providers.includes("toss")) {
    return ["kakao", "toss", "card"];
  }
  return ["card"];
}

function defaultMethod(context: CheckoutContext): Method {
  return methodsForContext(context)[0];
}

interface CheckoutDialogProps {
  context: CheckoutContext;
  requiredUnits: number;
  onClose: () => void;
  onPaid: () => void;
}

export function CheckoutDialog({ context, requiredUnits, onClose, onPaid }: CheckoutDialogProps) {
  const creditsToBuy = Math.max(
    1,
    requiredUnits - context.free_units_remaining - context.paid_units,
  );
  const [method, setMethod] = useState<Method>(() => defaultMethod(context));
  const [country, setCountry] = useState(context.country);
  const [activeContext, setActiveContext] = useState(context);
  const [busy, setBusy] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [error, setError] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [plan, setPlan] = useState<PurchasePlan>("per_drawing");
  const onPaidRef = useRef(onPaid);
  const countryRequestRef = useRef(0);
  const paymentInFlightRef = useRef(false);
  const checkoutKeyRef = useRef(crypto.randomUUID());
  const total = plan === "unlimited_monthly"
    ? activeContext.monthly_amount
    : activeContext.unit_amount * creditsToBuy;
  const baselineCredits = context.paid_units;

  useEffect(() => {
    onPaidRef.current = onPaid;
  }, [onPaid]);

  useEffect(() => {
    checkoutKeyRef.current = crypto.randomUUID();
  }, [activeContext.country, creditsToBuy, method, plan]);

  useEffect(() => {
    const available = methodsForContext(activeContext);
    if (!available.includes(method)) setMethod(available[0]);
  }, [activeContext, method]);

  useEffect(() => {
    if (!waiting) return;
    let cancelled = false;
    let timer = 0;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const latest = await loadCheckoutContext(country, controller.signal);
        if (cancelled) return;
        setActiveContext(latest);
        const paid = plan === "unlimited_monthly"
          ? latest.unlimited_active
          : latest.paid_units >= baselineCredits + creditsToBuy;
        if (paid) {
          setWaiting(false);
          onPaidRef.current();
          return;
        }
      } catch {
        if (cancelled) return;
      }
      timer = window.setTimeout(() => void poll(), 1_500);
    };
    timer = window.setTimeout(() => void poll(), 1_500);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [baselineCredits, country, creditsToBuy, plan, waiting]);

  const methods = useMemo(() => methodsForContext(activeContext), [activeContext]);

  const updateCountry = async (next: string) => {
    const requestId = ++countryRequestRef.current;
    setCountry(next);
    setBusy(true);
    setError("");
    try {
      const nextContext = await loadCheckoutContext(next);
      if (requestId !== countryRequestRef.current) return;
      setActiveContext(nextContext);
      setMethod(defaultMethod(nextContext));
    } catch (caught) {
      if (requestId !== countryRequestRef.current) return;
      setError(caught instanceof Error ? caught.message : "Could not update checkout.");
    } finally {
      if (requestId === countryRequestRef.current) setBusy(false);
    }
  };

  const beginPayment = async () => {
    if (paymentInFlightRef.current) return;
    paymentInFlightRef.current = true;
    setBusy(true);
    setError("");
    try {
      if (method === "apple" || method === "google") {
        throw new Error(
          `${method === "apple" ? "App Store" : "Google Play"} purchase becomes available after the store product is approved.`,
        );
      }
      const useToss = activeContext.country === "KR" && activeContext.configured_providers.includes("toss");
      const provider = useToss ? "toss" : "stripe";
      const easyPay = useToss
        ? method === "kakao" ? "KAKAOPAY" : method === "toss" ? "TOSSPAY" : ""
        : "";
      const checkout = await createCheckout(
        provider,
        activeContext.country,
        creditsToBuy,
        checkoutKeyRef.current,
        easyPay,
        plan,
      );
      if (checkout.kind === "redirect") {
        const paymentWindow = window.open(
          checkout.redirect_url,
          "dajoong-secure-checkout",
          "popup,width=560,height=760,resizable=yes,scrollbars=yes",
        );
        if (!paymentWindow) {
          window.location.assign(checkout.redirect_url);
          return;
        }
        setWaiting(true);
        return;
      }
      const toss = checkout.toss;
      if (!toss.client_key || !toss.customer_key || !toss.amount || !toss.order_id) {
        throw new Error("Toss checkout configuration is incomplete.");
      }
      const sdk = await loadTossPayments(toss.client_key);
      const payment = sdk.payment({ customerKey: toss.customer_key });
      await payment.requestPayment({
        method: "CARD",
        amount: toss.amount,
        orderId: toss.order_id,
        orderName: toss.order_name || "Dajoong drawing conversion",
        successUrl: toss.success_url,
        failUrl: toss.fail_url,
        card: easyPay ? { flowMode: "DIRECT", easyPay } : undefined,
      } as unknown as Parameters<typeof payment.requestPayment>[0]);
      setWaiting(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Payment could not be started.");
    } finally {
      paymentInFlightRef.current = false;
      setBusy(false);
    }
  };

  return (
    <div className="checkout-backdrop" role="presentation">
      <section className="checkout-shell" role="dialog" aria-modal="true" aria-labelledby="checkout-title">
        <button className="checkout-close" type="button" onClick={onClose} aria-label="Close checkout">
          <X size={18} />
        </button>

        <aside className="checkout-story">
          <div className="checkout-story-topline"><Sparkles size={14} /> DAJOONG LIGHTWEIGHT BIM</div>
          <h2>Full model.<br /><em>One-fifth</em> the benchmark cost.</h2>
          <p>
            Your drawing becomes a coordinated, editable BIM in one fast conversion. The result keeps
            source links and matched building elements ready for review.
          </p>
          <img
            className="checkout-pipeline-image"
            src="/marketing/cubi-014-reviewed-bim.webp"
            width="1600"
            height="1000"
            decoding="async"
            alt="Reviewed Dajoong color BIM result with matched building elements and objects"
          />
          <div className="checkout-speed-proof">
            <div><Timer size={15} /><span><b>{activeContext.speed_median_seconds.toFixed(2)} s</b> measured conversion median</span></div>
            <ArrowRight size={15} />
            <strong>≥{formatMultiple(activeContext.speed_comparison_multiple)} shorter*</strong>
          </div>
          <div className="checkout-comparison">
            <div><span>Dajoong single drawing</span><strong>{activeContext.unit_label}</strong><i style={{ width: "20%" }} /></div>
            <div><span>Dajoong unlimited</span><strong>{activeContext.monthly_label}</strong><i style={{ width: "20%" }} /></div>
            <div><span>T company minimum equivalent</span><strong>$20 / drawing · $400 / 20</strong><i style={{ width: "100%" }} /></div>
          </div>
          <button className="checkout-basis" type="button" onClick={() => setDetailsOpen((value) => !value)}>
            <BadgeCheck size={14} /> Verified price + speed basis
            <ChevronDown className={detailsOpen ? "open" : ""} size={14} />
          </button>
          {detailsOpen ? (
            <p className="checkout-footnote">
              Price: T-company publishes $0.26/ft² with a $20 minimum 3D order. Dajoong is $3.99 per drawing,
              over 5× below that minimum. Monthly unlimited is $79; at 20 drawings, the T-company minimum-order
              equivalent is $400, also over 5× higher. Scope and human review can differ. Pricing
              checked August 2026. <a href={activeContext.comparison_source_url} target="_blank" rel="noreferrer">Price source</a>.
              Speed: {activeContext.speed_runs} cold end-to-end runs, {activeContext.speed_median_seconds.toFixed(2)} s median
              and {activeContext.speed_p95_seconds.toFixed(2)} s observed slowest run. The ≥{formatMultiple(activeContext.speed_comparison_multiple)} figure
              conservatively compares that median with 16 working hours inside the published ~2-business-day turnaround;
              it is a turnaround comparison, not equal-scope algorithm throughput. <a href={activeContext.speed_benchmark_url} target="_blank" rel="noreferrer">Benchmark record</a>
              {" · "}<a href={activeContext.speed_turnaround_source_url} target="_blank" rel="noreferrer">Turnaround source</a>
            </p>
          ) : null}
        </aside>

        <div className="checkout-panel">
          <header>
            <span className="checkout-step">SECURE CHECKOUT</span>
            <h1 id="checkout-title">Keep the model moving.</h1>
            <p>Your upload and conversion settings stay in place while you pay.</p>
          </header>

          <div className="checkout-free-status">
            <FileBox size={18} />
            <div>
              <strong>{context.free_units_remaining ? "First drawing is on us" : "Free drawing used"}</strong>
              <span>{context.free_units_remaining ? "No card was required for your first conversion." : "Add only the credits this conversion needs."}</span>
            </div>
            <Check size={16} />
          </div>

          <div className="checkout-plan-picker" role="radiogroup" aria-label="Billing plan">
            <button className={plan === "per_drawing" ? "active" : ""} type="button" role="radio" aria-checked={plan === "per_drawing"} onClick={() => setPlan("per_drawing")}>
              <span>PER DRAWING</span><strong>{activeContext.unit_label}</strong><small>Pay only when you convert</small>
            </button>
            <button className={plan === "unlimited_monthly" ? "active" : ""} type="button" role="radio" aria-checked={plan === "unlimited_monthly"} onClick={() => setPlan("unlimited_monthly")}>
              <span>MONTHLY UNLIMITED</span><strong>{activeContext.monthly_label}</strong><small>Unlimited drawings for 31 days</small>
            </button>
          </div>

          <div className="checkout-order-line">
            <div>
              <span>{plan === "unlimited_monthly" ? "Monthly unlimited" : "Drawing conversion"}</span>
              <small>{plan === "unlimited_monthly" ? "Unlimited drawings · 31 days" : `${creditsToBuy} ${creditsToBuy === 1 ? "drawing" : "drawings"} · no subscription`}</small>
            </div>
            <strong>{formatMoney(total, activeContext.currency)}</strong>
          </div>

          <div className="checkout-region">
            <label htmlFor="checkout-country">PAYMENT REGION</label>
            <select id="checkout-country" value={country} onChange={(event) => void updateCountry(event.target.value)}>
              <option value="US">United States</option>
              <option value="KR">대한민국</option>
            </select>
          </div>

          <div className="checkout-methods" role="radiogroup" aria-label="Payment method">
            {methods.map((item) => (
              <PaymentMethod key={item} method={item} active={method === item} onSelect={() => setMethod(item)} />
            ))}
          </div>

          {activeContext.configured_providers.length === 0 ? (
            <div className="checkout-provider-note">
              <RotateCcw size={16} /> Payment keys are not connected in this environment yet.
            </div>
          ) : null}
          {error ? <div className="checkout-error">{error}</div> : null}

          <button
            className="checkout-pay"
            type="button"
            disabled={busy || waiting || activeContext.configured_providers.length === 0}
            onClick={() => void beginPayment()}
          >
            {busy || waiting ? <LoaderCircle className="spin" size={18} /> : <LockKeyhole size={17} />}
            <span>{waiting ? "Waiting for secure confirmation…" : `Pay ${formatMoney(total, activeContext.currency)}`}</span>
            {!busy && !waiting ? <ArrowRight size={17} /> : null}
          </button>

          <div className="checkout-reassurance">
            <span><ShieldCheck size={14} /> Encrypted payment</span>
            <span><ReceiptText size={14} /> Receipt by email</span>
            <span><CreditCard size={14} /> No saved card by default</span>
          </div>
          <p className="checkout-legal">By paying, you agree to the Terms. Need help? jjoonghui@gmail.com</p>
        </div>
      </section>
    </div>
  );
}

function PaymentMethod({ method, active, onSelect }: { method: Method; active: boolean; onSelect: () => void }) {
  const copy = {
    card: ["Card", "Visa, Mastercard, Amex"],
    kakao: ["Kakao Pay", "Open Kakao securely"],
    toss: ["toss pay", "Pay in the Toss app"],
    apple: ["Apple", "App Store purchase"],
    google: ["Google Play", "Play purchase"],
  }[method];
  return (
    <button type="button" className={`checkout-method ${method} ${active ? "active" : ""}`} role="radio" aria-checked={active} onClick={onSelect}>
      <BrandMark method={method} />
      <span><strong>{copy[0]}</strong><small>{copy[1]}</small></span>
      <i>{active ? <Check size={13} /> : null}</i>
    </button>
  );
}

function BrandMark({ method }: { method: Method }) {
  if (method === "kakao") return <img src={kakaoMarkUrl} alt="" />;
  if (method === "card") return <CreditCard className="card-mark" aria-hidden="true" />;
  if (method === "toss") return <b className="toss-mark">toss</b>;
  if (method === "apple") return <b className="apple-mark"></b>;
  return <b className="google-mark">▶</b>;
}

function formatMoney(value: number, currency: "USD" | "KRW") {
  return new Intl.NumberFormat(currency === "KRW" ? "ko-KR" : "en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: currency === "KRW" ? 0 : 2,
  }).format(currency === "USD" ? value / 100 : value);
}

function formatMultiple(value: number) {
  return `${Math.floor(value / 1000).toLocaleString("en-US")},000×`;
}
