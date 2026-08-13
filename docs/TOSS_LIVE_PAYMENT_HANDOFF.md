# Toss Payments live handoff

Dajoong uses Toss Payments for Korean card, KakaoPay, and TossPay checkout.
Payment approval, amount verification, idempotency, and credit granting happen
on the private Dajoong API. The bank account is never stored in Git or sent to
the browser.

## Owner-only setup

1. Complete the Toss Payments electronic-payment merchant application.
2. In the Toss merchant manager, register and verify the settlement account
   ending in `9419`. Confirm that the account holder matches the contracted
   individual or business. Never enter the full account number in GitHub.
3. Enable card payments plus KakaoPay and TossPay for the contracted MID.
4. Copy the matching **live API individual integration** client and secret keys.
5. From the repository root, run:

   ```powershell
   .\scripts\configure_toss_production.ps1 -Deploy
   ```

6. The command validates the whole release and deploys it only after validation
   passes. Complete one low-value live purchase, then verify both the payment
   and expected settlement account in the Toss merchant manager.

The script accepts only live-key prefixes and stores keys in the protected
GitHub `production` environment. The secret key never enters the browser bundle.

## Implemented payment guarantees

- Server-created amount and order ID are checked again during approval.
- An idempotency key prevents repeated clicks from creating mismatched orders.
- Credits or monthly access are granted once inside a transactional store.
- A stable opaque customer key supports a returning-customer checkout without
  disclosing the Dajoong account ID to the browser.
- Production deployment refuses partial payment credentials.
