import { useState } from "react";
import { authFetch, signOut } from "../auth";
import { studioApiUrl } from "../serverApi";

export function AccountDialog({ onClose }: { onClose: () => void }) {
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const deleteAccount = async () => {
    setSubmitting(true);
    setMessage("");
    try {
      const response = await authFetch(studioApiUrl("/api/account"), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmation }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(payload.detail || "The account could not be deleted");
      }
      await signOut();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The account could not be deleted");
      setSubmitting(false);
    }
  };

  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="conversion-dialog account-dialog" role="dialog" aria-modal="true" aria-labelledby="account-dialog-title">
      <div className="dialog-header"><div><span className="eyebrow">ACCOUNT</span><h2 id="account-dialog-title">Account and privacy</h2></div><button type="button" onClick={onClose} aria-label="Close account settings">Close</button></div>
      <div className="account-dialog-copy">
        <p>Deleting your account removes your sign-in and personal conversion jobs. Organization-owned project records may remain under the organization&apos;s contract and retention policy.</p>
        <a href="/account-deletion" target="_blank" rel="noreferrer">Read the account deletion policy</a>
        <label>Type <strong>DELETE</strong> to confirm<input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} autoComplete="off" /></label>
        {message ? <p className="auth-error" role="alert">{message}</p> : null}
      </div>
      <div className="dialog-actions"><button type="button" className="secondary-button" onClick={onClose}>Cancel</button><button type="button" className="danger-button" disabled={confirmation !== "DELETE" || submitting} onClick={() => void deleteAccount()}>{submitting ? "Deleting..." : "Delete account"}</button></div>
    </section>
  </div>;
}
