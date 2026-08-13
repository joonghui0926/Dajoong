# Dajoong AWS deployment

This stack provisions the private production boundary around the converter. Cognito issues user tokens. App Runner receives uploads and writes durable job state to DynamoDB. SQS feeds CPU conversion work to an autoscaled Fargate Spot service. Artifacts remain private in S3 and are returned through short-lived signed URLs. The browser client is a separate static-assets Cloudflare Worker, and the public API is a separate Cloudflare Worker that verifies an origin secret before forwarding to App Runner.

## Required operator inputs

1. Configure an AWS account and run `aws configure` once.
2. Build the API image and push the same immutable tag to both ECR repositories created by this stack.
3. Copy `terraform.tfvars.example` to `terraform.tfvars`, then set the image tags, allowed web origin, and Cognito callback URLs.
4. Run `terraform apply`.
5. Build the web client with `https://studio-api.dajoongbim.com` and the Cognito outputs, then deploy `infra/cloudflare/wrangler.web.jsonc` with Wrangler.

The web Worker owns `dajoongbim.com`, `www.dajoongbim.com`, `studio.dajoongbim.com`, and `app.dajoongbim.com`. The apex serves the marketing and legal pages. `www` redirects to the apex, `/studio` redirects to the canonical Studio host, and `app` redirects to Studio while still serving native association files directly. The API Worker owns `studio-api.dajoongbim.com`.

No database server or Redis cluster is required for the first release. DynamoDB stores job, billing, workspace, comment, version, and short-lived presence state. S3 stores drawings, graphs, IFC, GLB, and review patches. SQS absorbs conversion bursts while App Runner and Fargate scale independently. CloudWatch alarms watch the dead-letter queue and oldest queued job; set `alarm_sns_topic_arn` to route those alarms to the operating team.

The production workflow writes Stripe and Toss server keys to AWS Secrets
Manager, then gives App Runner only the secret ARNs. Configure the matching
protected GitHub secrets described in `docs/COMMERCIAL_RELEASE.md`; no payment
credential belongs in Terraform source, the browser bundle, or a mobile binary.

## Validation

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace hashicorp/terraform:latest fmt -check
docker run --rm -v "${PWD}:/workspace" -w /workspace hashicorp/terraform:latest init -backend=false
docker run --rm -v "${PWD}:/workspace" -w /workspace hashicorp/terraform:latest validate
```

Use a remote encrypted Terraform backend before a team deployment. Keep `terraform.tfvars`, state files, AWS credentials, and signing material out of source control.
