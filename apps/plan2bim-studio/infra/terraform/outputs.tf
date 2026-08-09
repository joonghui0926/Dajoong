output "web_bucket" { value = aws_s3_bucket.web.id }
output "artifact_bucket" { value = aws_s3_bucket.artifacts.id }
output "cloudfront_domain" { value = aws_cloudfront_distribution.web.domain_name }
output "api_repository" { value = aws_ecr_repository.api.repository_url }
output "worker_repository" { value = aws_ecr_repository.worker.repository_url }
output "api_service_url" { value = aws_apprunner_service.api.service_url }
output "job_queue_url" { value = aws_sqs_queue.jobs.url }
output "user_pool_id" { value = aws_cognito_user_pool.users.id }
output "user_pool_client_id" { value = aws_cognito_user_pool_client.web.id }
output "cognito_authority" { value = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.users.id}" }
output "cognito_hosted_ui" { value = "https://${aws_cognito_user_pool_domain.web.domain}.auth.${var.aws_region}.amazoncognito.com" }
output "origin_secret_arn" {
  value     = var.origin_secret_arn
  sensitive = true
}
