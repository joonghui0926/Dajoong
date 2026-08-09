# Security

Report security issues privately to `jjoonghui@gmail.com`. Do not open a public issue
for credentials, model exposure, authentication bypasses, or customer data.

## Non-negotiable release boundaries

- The repository must be private while it contains proprietary compiler source.
- Model checkpoints and signing keys must never be committed.
- Web and mobile clients may call only Dajoong-owned production API origins.
- Conversion artifacts are authorization-scoped and streamed through the Dajoong API.
- Production changes use short-lived GitHub OIDC credentials and protected environments.
