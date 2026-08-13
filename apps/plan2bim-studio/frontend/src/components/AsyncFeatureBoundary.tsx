import { Component, lazy, Suspense, useEffect, useState } from "react";
import type { ComponentType, ErrorInfo, ReactNode } from "react";

const FEATURE_LOAD_TIMEOUT_MS = 15_000;

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(
      () => reject(new Error("This part of the workspace took too long to load.")),
      timeoutMs,
    );
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

// The component's exact props are preserved by T; the constraint only accepts React components.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function reliableLazy<T extends ComponentType<any>>(
  loader: () => Promise<{ default: T }>,
) {
  return lazy(() => withTimeout(loader(), FEATURE_LOAD_TIMEOUT_MS));
}

export function AsyncFeatureLoading({ label }: { label: string }) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setSlow(true), 2_500);
    return () => window.clearTimeout(timer);
  }, []);
  return (
    <div className="studio-tool-loading" role="status" aria-live="polite">
      <span />
      <div><strong>{label}</strong>{slow ? <small>Your work is safe. This is taking longer than expected.</small> : null}</div>
      {slow ? <button type="button" onClick={() => window.location.reload()}>Reload safely</button> : null}
    </div>
  );
}

interface RecoverableBoundaryProps {
  children: ReactNode;
  label: string;
  variant?: "overlay" | "inline";
}

class RecoverableBoundary extends Component<RecoverableBoundaryProps, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`Could not load ${this.props.label}`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    if (this.props.variant === "inline") {
      return (
        <div className="async-feature-inline" role="alert">
          <strong>{this.props.label} is unavailable</strong>
          <small>The rest of the workspace is ready.</small>
          <button type="button" onClick={() => window.location.reload()}>Retry</button>
        </div>
      );
    }
    return (
      <div className="studio-tool-loading async-feature-error" role="alert">
        <div>
          <strong>{this.props.label} could not be opened</strong>
          <small>Your local edits were preserved. Reload to reconnect the workspace.</small>
        </div>
        <button type="button" onClick={() => window.location.reload()}>Reload safely</button>
      </div>
    );
  }
}

export function AsyncFeatureBoundary({
  children,
  label,
  fallback,
  variant,
}: RecoverableBoundaryProps & { fallback?: ReactNode }) {
  return (
    <RecoverableBoundary label={label} variant={variant}>
      <Suspense fallback={fallback ?? <AsyncFeatureLoading label={label} />}>
        {children}
      </Suspense>
    </RecoverableBoundary>
  );
}

export { RecoverableBoundary };
