# Commercial release handoff

Production is designed so the owner enters keys once and then runs workflows.

## Required GitHub production environment values

Variables:

- `AWS_DEPLOY_ROLE_ARN`
- `STUDIO_AWS_REGION` (default `us-west-2`)
- `TF_STATE_REGION` (default `us-west-1`)
- `CLOUDFLARE_ACCOUNT_ID`
- `DAJOONG_MODEL_BUNDLE_S3_URI`
- `APPLE_TEAM_ID`
- `ANDROID_APP_LINK_SHA256` (Play signing certificate SHA-256 fingerprint)
- `APP_REVIEW_FIRST_NAME`, `APP_REVIEW_LAST_NAME`, and `APP_REVIEW_NOTES`

Secrets:

- `CLOUDFLARE_API_TOKEN`
- `DAJOONG_MODEL_BUNDLE_SHA256`
- OAuth keys for Google, Apple, and Kakao
- Android keystore, passwords, alias, and Google Play service-account JSON
- Apple distribution certificate, password, App Store Connect key ID, issuer ID, and P8
- App Review phone number and demo-account credentials

The exact secret names are validated by `.github/workflows/release-readiness.yml`.

## Release order

1. Run `deploy-plan2bim-studio` to deploy the private converter, CPU workers,
   Cognito, `studio-api.builiconstruction.com`, and `studio.builiconstruction.com`.
2. Verify a real authenticated conversion and download from both web and mobile.
3. Run `release-mobile` with publishing disabled and inspect the signed artifacts.
4. Re-run `release-mobile` for the Play internal track and TestFlight.
5. Submit the reviewed build from the same workflow.

## One-time console actions

- Change this GitHub repository to **Private** and grant the release operator write access.
- Protect the `production` environment with required reviewers.
- Create the Play Console and App Store Connect records for `com.dajoong.plan2bim`.
- Enable Associated Domains and Sign in with Apple for the Apple identifier.
- Register Cognito callbacks shown in the deployment summary with Google, Apple, and Kakao.
- Point the Cloudflare DNS/custom domains at the deployed Workers when first requested.

Support and privacy contact: `jjoonghui@gmail.com`.
