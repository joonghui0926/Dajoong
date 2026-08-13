# Commercial release handoff

Production is designed so the owner enters credentials once in the protected
GitHub `production` environment and then runs three workflows. No key is stored
in Git, a browser bundle, or a mobile application.

## One-time account setup

1. Make this repository **Private**, give the release operator write access, and
   create a protected GitHub environment named `production`.
2. Deploy `infra/bootstrap/github-oidc.yml` once in the AWS account. Use its
   output as `AWS_DEPLOY_ROLE_ARN`.
3. Keep `dajoongbim.com` in the Cloudflare account used by the token.
   Wrangler creates the apex, `www`, `studio`, `app`, and `studio-api`
   custom-domain records and their certificates.
4. Create Play Console and App Store Connect records for
   `com.dajoong.plan2bim`.
5. Enable Associated Domains and Sign in with Apple on that Apple identifier.
6. Upload the content-addressed private model package described in
   `PRIVATE_MODEL_DELIVERY.md` to the private S3 object named below.

The Cognito provider callback is deterministic:

```text
https://dajoong-plan2bim-production-<AWS_ACCOUNT_ID>.auth.<AWS_REGION>.amazoncognito.com/oauth2/idpresponse
```

Register that URL with Google, Apple, and Kakao before the first deployment.

## GitHub environment variables

| Name | Value |
| --- | --- |
| `AWS_DEPLOY_ROLE_ARN` | Bootstrap stack output |
| `STUDIO_AWS_REGION` | Optional; defaults to `us-west-2` |
| `TF_STATE_REGION` | Optional; defaults to `us-west-1` |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |
| `DAJOONG_MODEL_BUNDLE_S3_URI` | Private `s3://...tar.gz` URI |
| `APPLE_TEAM_ID` | Ten-character Apple team ID |
| `ANDROID_APP_LINK_SHA256` | Play signing certificate SHA-256 fingerprint |
| `APP_REVIEW_FIRST_NAME` | Optional; defaults to `Paul` |
| `APP_REVIEW_LAST_NAME` | Optional; defaults to `Cho` |
| `APP_REVIEW_NOTES` | Optional review instructions |
| `DAJOONG_INVITE_FROM_EMAIL` | Optional verified SES sender for automatic team invitation email |
| `DAJOONG_INVITE_IDENTITY_ARN` | Matching SES identity ARN; configure together with the sender |
| `DAJOONG_ALARM_SNS_TOPIC_ARN` | Optional SNS topic for failed-job and queue-delay alerts |

## GitHub environment secrets

| Area | Exact secret names |
| --- | --- |
| Cloudflare/model | `CLOUDFLARE_API_TOKEN`, `DAJOONG_MODEL_BUNDLE_SHA256` |
| Google login | `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` |
| Apple login | `APPLE_SERVICE_ID`, `APPLE_SIGN_IN_KEY_ID`, `APPLE_SIGN_IN_PRIVATE_KEY` |
| Kakao login | `KAKAO_OIDC_CLIENT_ID`, `KAKAO_OIDC_CLIENT_SECRET` |
| Payments | Complete one provider pair: `TOSS_SECRET_KEY` + `TOSS_CLIENT_KEY`, or `STRIPE_SECRET_KEY` + `STRIPE_WEBHOOK_SECRET` |
| Android signing | `ANDROID_KEYSTORE_BASE64`, `ANDROID_KEY_ALIAS`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_PASSWORD` |
| Google Play | `GOOGLE_PLAY_SERVICE_ACCOUNT_JSON` |
| Apple signing | `APPLE_DISTRIBUTION_CERTIFICATE_BASE64`, `APPLE_DISTRIBUTION_CERTIFICATE_PASSWORD` |
| App Store Connect | `APP_STORE_CONNECT_API_KEY_ID`, `APP_STORE_CONNECT_ISSUER_ID`, `APP_STORE_CONNECT_API_KEY_BASE64` |
| Store review | `APP_REVIEW_PHONE`, `APP_REVIEW_DEMO_USER`, `APP_REVIEW_DEMO_PASSWORD` |

Base64 values must contain the raw file bytes encoded without line wrapping.
The Cloudflare token needs Workers Scripts edit, Workers Routes edit, DNS edit,
and account/zone read access for `dajoongbim.com`.

The deployment copies payment server keys into AWS Secrets Manager and passes
only their ARNs to App Runner. Team invitation tokens are hashed at rest. If an
SES sender is not configured, administrators can still copy the one-time secure
invitation link from the People panel.

For a Korea-first Toss-only launch, Stripe is optional. Configure the matching
Toss live client and secret key with `scripts/configure_toss_production.ps1`.
Register the verified settlement account in the Toss merchant manager; never
store a bank account number in this repository. See
`TOSS_LIVE_PAYMENT_HANDOFF.md` for the minimal owner handoff.

## Release order

1. Run `release-readiness`. It fails with the exact missing key name and then
   validates the private model, Python suite, server image, web suite, bundle
   boundary, and store metadata.
2. Run `deploy-plan2bim-studio`. It deploys the private converter, zero-idle CPU
   workers, Cognito, `studio-api.dajoongbim.com`, the landing page at the
   apex, and the editor at `studio.dajoongbim.com`.
3. Complete one authenticated conversion on web and one installed test build.
4. Run `release-mobile` with both publishing switches off. Inspect the signed
   AAB and IPA artifacts.
5. Run it again for the Play internal track and TestFlight. Only after review,
   enable store submission or production release.

## Public production pages

- Website: `https://dajoongbim.com/`
- Studio: `https://studio.dajoongbim.com/studio`
- Privacy: `https://studio.dajoongbim.com/privacy`
- Cookies: `https://studio.dajoongbim.com/cookies`
- Terms: `https://studio.dajoongbim.com/terms`
- Support: `https://studio.dajoongbim.com/support`
- Account deletion: `https://studio.dajoongbim.com/account-deletion`
- API health: `https://studio-api.dajoongbim.com/api/health`

Support and privacy contact: `jjoonghui@gmail.com`.

Account data, recent-project pagination, immutable correction revisions, and
deletion-race handling are specified in `ACCOUNT_DATA_ARCHITECTURE.md`.
