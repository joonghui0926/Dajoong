data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" { state = "available" }

locals {
  name = "dajoong-plan2bim-${var.environment}"
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule { object_ownership = "BucketOwnerEnforced" }
}

data "aws_iam_policy_document" "artifacts_transport" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  policy = data.aws_iam_policy_document.artifacts_transport.json
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket     = aws_s3_bucket.artifacts.id
  depends_on = [aws_s3_bucket_versioning.artifacts]
  rule {
    id     = "expire-unversioned-job-artifacts"
    status = "Enabled"
    filter { prefix = "jobs/" }
    expiration { days = var.artifact_retention_days }
    noncurrent_version_expiration { noncurrent_days = 30 }
    abort_incomplete_multipart_upload { days_after_initiation = 1 }
  }
}

resource "aws_dynamodb_table" "jobs" {
  name                        = "${local.name}-jobs"
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "job_id"
  deletion_protection_enabled = var.environment == "production"
  attribute {
    name = "job_id"
    type = "S"
  }
  attribute {
    name = "owner_id"
    type = "S"
  }
  attribute {
    name = "organization_id"
    type = "S"
  }
  attribute {
    name = "created_at_job"
    type = "S"
  }
  global_secondary_index {
    name = "owner-id-index"
    key_schema {
      attribute_name = "owner_id"
      key_type       = "HASH"
    }
    key_schema {
      attribute_name = "created_at_job"
      key_type       = "RANGE"
    }
    projection_type = "INCLUDE"
    non_key_attributes = [
      "organization_id",
      "source_name",
      "project_id",
      "status",
      "created_at",
      "updated_at",
      "expires_at",
      "version",
      "error",
    ]
  }
  global_secondary_index {
    name = "organization-id-index"
    key_schema {
      attribute_name = "organization_id"
      key_type       = "HASH"
    }
    key_schema {
      attribute_name = "created_at_job"
      key_type       = "RANGE"
    }
    projection_type = "INCLUDE"
    non_key_attributes = [
      "owner_id",
      "source_name",
      "project_id",
      "status",
      "created_at",
      "updated_at",
      "expires_at",
      "version",
      "error",
    ]
  }
  point_in_time_recovery { enabled = true }
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "billing" {
  name                        = "${local.name}-billing"
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "record_id"
  deletion_protection_enabled = var.environment == "production"
  attribute {
    name = "record_id"
    type = "S"
  }
  point_in_time_recovery { enabled = true }
}

resource "aws_dynamodb_table" "collaboration" {
  name                        = "${local.name}-collaboration"
  billing_mode                = "PAY_PER_REQUEST"
  hash_key                    = "pk"
  range_key                   = "sk"
  deletion_protection_enabled = var.environment == "production"
  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }
  point_in_time_recovery { enabled = true }
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

resource "aws_sqs_queue" "dead_letter" {
  name                      = "${local.name}-jobs-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "jobs" {
  name                       = "${local.name}-jobs"
  visibility_timeout_seconds = var.job_visibility_seconds
  message_retention_seconds  = 345600
  receive_wait_time_seconds  = 20
  sqs_managed_sse_enabled    = true
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = 3
  })
}

resource "aws_ecr_repository" "api" {
  name                 = "${local.name}-api"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "worker" {
  name                 = "${local.name}-worker"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_cognito_user_pool" "users" {
  name                     = "${local.name}-users"
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]
  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_numbers                  = true
    require_symbols                  = true
    require_uppercase                = true
    temporary_password_validity_days = 3
  }
}

resource "aws_cognito_user_pool_client" "web" {
  name                                 = "${local.name}-web"
  user_pool_id                         = aws_cognito_user_pool.users.id
  generate_secret                      = false
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]
  supported_identity_providers         = ["COGNITO"]
  callback_urls                        = var.oauth_callback_urls
  logout_urls                          = var.oauth_logout_urls

  lifecycle {
    # The release workflow adds only providers whose external keys are present.
    ignore_changes = [supported_identity_providers]
  }
}

resource "aws_cognito_user_pool_domain" "web" {
  domain       = "${local.name}-${data.aws_caller_identity.current.account_id}"
  user_pool_id = aws_cognito_user_pool.users.id
}

data "aws_iam_policy_document" "runtime_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com", "ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "runtime" {
  name               = "${local.name}-runtime"
  assume_role_policy = data.aws_iam_policy_document.runtime_assume.json
}

data "aws_iam_policy_document" "runtime" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"]
    resources = ["${aws_s3_bucket.artifacts.arn}/jobs/*"]
  }
  statement {
    actions   = ["s3:ListBucket", "s3:ListBucketVersions"]
    resources = [aws_s3_bucket.artifacts.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["jobs/*"]
    }
  }
  statement {
    actions   = ["cognito-idp:AdminDeleteUser"]
    resources = [aws_cognito_user_pool.users.arn]
  }
  statement {
    actions = ["dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem"]
    resources = [
      aws_dynamodb_table.jobs.arn,
      aws_dynamodb_table.billing.arn,
      aws_dynamodb_table.collaboration.arn,
    ]
  }
  statement {
    actions   = ["dynamodb:TransactWriteItems"]
    resources = [aws_dynamodb_table.billing.arn, aws_dynamodb_table.collaboration.arn]
  }
  statement {
    actions   = ["dynamodb:Query"]
    resources = [aws_dynamodb_table.collaboration.arn]
  }
  dynamic "statement" {
    for_each = var.invite_identity_arn == "" ? [] : [var.invite_identity_arn]
    content {
      actions   = ["ses:SendEmail"]
      resources = [statement.value]
    }
  }
  statement {
    actions = ["dynamodb:Query"]
    resources = [
      "${aws_dynamodb_table.jobs.arn}/index/owner-id-index",
      "${aws_dynamodb_table.jobs.arn}/index/organization-id-index",
    ]
  }
  statement {
    actions   = ["sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility", "sqs:GetQueueAttributes"]
    resources = [aws_sqs_queue.jobs.arn]
  }
  dynamic "statement" {
    for_each = toset(compact([
      var.origin_secret_arn,
      var.stripe_secret_key_arn,
      var.stripe_webhook_secret_arn,
      var.toss_secret_key_arn,
    ]))
    content {
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [statement.value]
    }
  }
}

resource "aws_iam_role_policy" "runtime" {
  role   = aws_iam_role.runtime.id
  policy = data.aws_iam_policy_document.runtime.json
}

data "aws_iam_policy_document" "apprunner_ecr_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_ecr" {
  name               = "${local.name}-apprunner-ecr"
  assume_role_policy = data.aws_iam_policy_document.apprunner_ecr_assume.json
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

data "aws_iam_policy_document" "ecs_execution_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_execution_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_apprunner_service" "api" {
  service_name                   = "${local.name}-api"
  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.api.arn
  source_configuration {
    auto_deployments_enabled = false
    authentication_configuration { access_role_arn = aws_iam_role.apprunner_ecr.arn }
    image_repository {
      image_identifier      = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
      image_repository_type = "ECR"
      image_configuration {
        port = "8042"
        runtime_environment_variables = {
          DAJOONG_RUNTIME                 = "aws"
          DAJOONG_ENVIRONMENT             = var.environment
          DAJOONG_ARTIFACT_BUCKET         = aws_s3_bucket.artifacts.id
          DAJOONG_JOB_TABLE               = aws_dynamodb_table.jobs.name
          DAJOONG_BILLING_TABLE           = aws_dynamodb_table.billing.name
          DAJOONG_COLLABORATION_TABLE     = aws_dynamodb_table.collaboration.name
          DAJOONG_JOB_QUEUE_URL           = aws_sqs_queue.jobs.url
          DAJOONG_STUDIO_ORIGINS          = var.cors_origins
          DAJOONG_REQUIRE_AUTH            = "true"
          DAJOONG_AUTH_ISSUER             = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.users.id}"
          DAJOONG_AUTH_AUDIENCE           = aws_cognito_user_pool_client.web.id
          DAJOONG_USER_POOL_ID            = aws_cognito_user_pool.users.id
          DAJOONG_OWNER_INDEX_NAME        = "owner-id-index"
          DAJOONG_ORGANIZATION_INDEX_NAME = "organization-id-index"
          DAJOONG_ARTIFACT_RETENTION_DAYS = tostring(var.artifact_retention_days)
          DAJOONG_MAX_UPLOAD_BYTES        = "104857600"
          DAJOONG_BILLING_ENFORCE         = "true"
          DAJOONG_JOB_VISIBILITY_SECONDS  = tostring(var.job_visibility_seconds)
          DAJOONG_PRICE_USD_CENTS         = tostring(var.price_usd_cents)
          DAJOONG_PRICE_KRW               = tostring(var.price_krw)
          DAJOONG_MONTHLY_PRICE_USD_CENTS = tostring(var.monthly_price_usd_cents)
          DAJOONG_MONTHLY_PRICE_KRW       = tostring(var.monthly_price_krw)
          DAJOONG_CHECKOUT_RETURN_ORIGIN  = var.checkout_return_origin
          DAJOONG_TOSS_CLIENT_KEY         = var.toss_client_key
          DAJOONG_APP_URL                 = var.app_url
          DAJOONG_INVITE_FROM_EMAIL       = var.invite_from_email
          AWS_REGION                      = var.aws_region
        }
        runtime_environment_secrets = merge(
          var.origin_secret_arn == "" ? {} : { DAJOONG_ORIGIN_VERIFY_SECRET = var.origin_secret_arn },
          var.stripe_secret_key_arn == "" ? {} : { DAJOONG_STRIPE_SECRET_KEY = var.stripe_secret_key_arn },
          var.stripe_webhook_secret_arn == "" ? {} : { DAJOONG_STRIPE_WEBHOOK_SECRET = var.stripe_webhook_secret_arn },
          var.toss_secret_key_arn == "" ? {} : { DAJOONG_TOSS_SECRET_KEY = var.toss_secret_key_arn },
        )
      }
    }
  }
  instance_configuration {
    cpu               = "1 vCPU"
    memory            = "2 GB"
    instance_role_arn = aws_iam_role.runtime.arn
  }
  health_check_configuration {
    path     = "/api/health"
    protocol = "HTTP"
  }
}

resource "aws_vpc" "worker" {
  cidr_block           = "10.44.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
}

resource "aws_internet_gateway" "worker" { vpc_id = aws_vpc.worker.id }

resource "aws_subnet" "worker" {
  count                   = 2
  vpc_id                  = aws_vpc.worker.id
  cidr_block              = cidrsubnet(aws_vpc.worker.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
}

resource "aws_route_table" "worker" { vpc_id = aws_vpc.worker.id }
resource "aws_route" "internet" {
  route_table_id         = aws_route_table.worker.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.worker.id
}
resource "aws_route_table_association" "worker" {
  count          = 2
  subnet_id      = aws_subnet.worker[count.index].id
  route_table_id = aws_route_table.worker.id
}

resource "aws_security_group" "worker" {
  name   = "${local.name}-worker-egress"
  vpc_id = aws_vpc.worker.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_cluster" "worker" {
  name = "${local.name}-worker"
}

resource "aws_ecs_cluster_capacity_providers" "worker" {
  cluster_name       = aws_ecs_cluster.worker.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/dajoong/${local.name}/worker"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.worker_cpu)
  memory                   = tostring(var.worker_memory)
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.runtime.arn
  container_definitions = jsonencode([{
    name      = "worker"
    image     = "${aws_ecr_repository.worker.repository_url}:${var.worker_image_tag}"
    essential = true
    command   = ["python", "-m", "buili_plan2bim_studio.aws_worker"]
    environment = [
      { name = "DAJOONG_RUNTIME", value = "aws" },
      { name = "DAJOONG_ENVIRONMENT", value = var.environment },
      { name = "DAJOONG_ARTIFACT_BUCKET", value = aws_s3_bucket.artifacts.id },
      { name = "DAJOONG_JOB_TABLE", value = aws_dynamodb_table.jobs.name },
      { name = "DAJOONG_JOB_QUEUE_URL", value = aws_sqs_queue.jobs.url },
      { name = "DAJOONG_JOB_VISIBILITY_SECONDS", value = tostring(var.job_visibility_seconds) },
      { name = "DAJOONG_SEMANTIC_MODEL_S3_KEY", value = var.semantic_model_s3_key },
      { name = "DAJOONG_SEMANTIC_MODEL_SHA256", value = var.semantic_model_sha256 },
      { name = "DAJOONG_USER_POOL_ID", value = aws_cognito_user_pool.users.id },
      { name = "DAJOONG_ARTIFACT_RETENTION_DAYS", value = tostring(var.artifact_retention_days) },
      { name = "AWS_REGION", value = var.aws_region },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.worker.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "worker"
      }
    }
  }])
}

resource "aws_ecs_service" "worker" {
  name            = "${local.name}-worker"
  cluster         = aws_ecs_cluster.worker.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = 0
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
  network_configuration {
    subnets          = aws_subnet.worker[*].id
    security_groups  = [aws_security_group.worker.id]
    assign_public_ip = true
  }
  depends_on = [aws_ecs_cluster_capacity_providers.worker]
}

resource "aws_appautoscaling_target" "worker" {
  max_capacity       = var.worker_max_tasks
  min_capacity       = 0
  resource_id        = "service/${aws_ecs_cluster.worker.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "worker_out" {
  name               = "${local.name}-worker-out"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 30
    metric_aggregation_type = "Maximum"
    step_adjustment {
      metric_interval_lower_bound = 0
      metric_interval_upper_bound = 4
      scaling_adjustment          = 1
    }
    step_adjustment {
      metric_interval_lower_bound = 4
      metric_interval_upper_bound = 19
      scaling_adjustment          = 4
    }
    step_adjustment {
      metric_interval_lower_bound = 19
      scaling_adjustment          = 10
    }
  }
}

resource "aws_appautoscaling_policy" "worker_in" {
  name               = "${local.name}-worker-in"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  step_scaling_policy_configuration {
    adjustment_type         = "ExactCapacity"
    cooldown                = 180
    metric_aggregation_type = "Maximum"
    step_adjustment {
      metric_interval_upper_bound = 1
      scaling_adjustment          = 0
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "worker_out" {
  alarm_name          = "${local.name}-queue-has-work"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Maximum"
  threshold           = 1
  dimensions          = { QueueName = aws_sqs_queue.jobs.name }
  alarm_actions       = [aws_appautoscaling_policy.worker_out.arn]
}

resource "aws_cloudwatch_metric_alarm" "worker_in" {
  alarm_name          = "${local.name}-queue-idle"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  threshold           = 1
  alarm_actions       = [aws_appautoscaling_policy.worker_in.arn]

  metric_query {
    id          = "backlog"
    expression  = "visible + running"
    label       = "Visible and in-flight jobs"
    return_data = true
  }
  metric_query {
    id = "visible"
    metric {
      metric_name = "ApproximateNumberOfMessagesVisible"
      namespace   = "AWS/SQS"
      period      = 60
      stat        = "Maximum"
      dimensions  = { QueueName = aws_sqs_queue.jobs.name }
    }
  }
  metric_query {
    id = "running"
    metric {
      metric_name = "ApproximateNumberOfMessagesNotVisible"
      namespace   = "AWS/SQS"
      period      = 60
      stat        = "Maximum"
      dimensions  = { QueueName = aws_sqs_queue.jobs.name }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "dead_letter_has_messages" {
  alarm_name          = "${local.name}-dead-letter-has-messages"
  alarm_description   = "A conversion exhausted its retries and requires inspection."
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Maximum"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  dimensions          = { QueueName = aws_sqs_queue.dead_letter.name }
  alarm_actions       = compact([var.alarm_sns_topic_arn])
  ok_actions          = compact([var.alarm_sns_topic_arn])
}

resource "aws_cloudwatch_metric_alarm" "queue_age" {
  alarm_name          = "${local.name}-queue-age"
  alarm_description   = "The oldest conversion has waited longer than the service objective."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApproximateAgeOfOldestMessage"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Maximum"
  threshold           = var.job_queue_age_alarm_seconds
  treat_missing_data  = "notBreaching"
  dimensions          = { QueueName = aws_sqs_queue.jobs.name }
  alarm_actions       = compact([var.alarm_sns_topic_arn])
  ok_actions          = compact([var.alarm_sns_topic_arn])
}

resource "aws_apprunner_auto_scaling_configuration_version" "api" {
  auto_scaling_configuration_name = "${local.name}-api"
  max_concurrency                 = 80
  max_size                        = var.api_max_instances
  min_size                        = 1
}
