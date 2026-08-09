# Dajoong AWS deployment

This stack provisions the production boundary around the converter. The web app is served from private S3 through CloudFront. Cognito issues user tokens. App Runner receives uploads and writes durable job state to DynamoDB. SQS feeds CPU conversion work to an autoscaled Fargate Spot service. Artifacts remain private in S3 and are returned through short lived signed URLs.

## Required operator inputs

1. Configure an AWS account and run `aws configure` once.
2. Build the API image and push the same immutable tag to both ECR repositories created by this stack.
3. Copy `terraform.tfvars.example` to `terraform.tfvars`, then set the image tags, allowed web origin, and Cognito callback URLs.
4. Run `terraform apply`.
5. Build the web client with the CloudFront API URL and Cognito values from Terraform outputs, then upload `frontend/dist` to the web bucket.

No database server or Redis cluster is required for the first release. DynamoDB stores job state and S3 stores drawings, graphs, IFC, GLB, and review patches. Redis becomes useful only when live multiuser cursors or high frequency collaborative editing are introduced.

## Validation

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace hashicorp/terraform:latest fmt -check
docker run --rm -v "${PWD}:/workspace" -w /workspace hashicorp/terraform:latest init -backend=false
docker run --rm -v "${PWD}:/workspace" -w /workspace hashicorp/terraform:latest validate
```

Use a remote encrypted Terraform backend before a team deployment. Keep `terraform.tfvars`, state files, AWS credentials, and signing material out of source control.
