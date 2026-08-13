variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "environment" {
  type    = string
  default = "production"
}

variable "api_image_tag" {
  type    = string
  default = "latest"
}

variable "worker_image_tag" {
  type    = string
  default = "latest"
}

variable "worker_cpu" {
  type    = number
  default = 2048
}

variable "worker_memory" {
  type    = number
  default = 4096
}

variable "worker_max_tasks" {
  type    = number
  default = 100
  validation {
    condition     = var.worker_max_tasks >= 1 && var.worker_max_tasks <= 1000
    error_message = "worker_max_tasks must be between 1 and 1000."
  }
}

variable "api_max_instances" {
  type    = number
  default = 25
  validation {
    condition     = var.api_max_instances >= 1 && var.api_max_instances <= 25
    error_message = "api_max_instances must be between 1 and 25."
  }
}

variable "alarm_sns_topic_arn" {
  description = "Optional SNS topic ARN for production reliability alarms."
  type        = string
  default     = ""
}

variable "job_queue_age_alarm_seconds" {
  description = "Maximum acceptable age for the oldest queued conversion."
  type        = number
  default     = 900
  validation {
    condition     = var.job_queue_age_alarm_seconds >= 60 && var.job_queue_age_alarm_seconds <= 86400
    error_message = "job_queue_age_alarm_seconds must be between 60 and 86400."
  }
}

variable "job_visibility_seconds" {
  type    = number
  default = 900
  validation {
    condition     = var.job_visibility_seconds >= 60 && var.job_visibility_seconds <= 43200
    error_message = "job_visibility_seconds must be between 60 and 43200."
  }
}

variable "semantic_model_s3_key" {
  description = "Private ONNX object key in the artifact bucket; its manifest is stored at <key>.json."
  type        = string
  default     = ""
}

variable "semantic_model_sha256" {
  description = "Optional SHA-256 checksum used by workers before loading the private ONNX model."
  type        = string
  default     = ""
}

variable "artifact_retention_days" {
  type    = number
  default = 90
}

variable "origin_secret_arn" {
  description = "Secrets Manager ARN containing the Cloudflare-to-App Runner origin verification value."
  type        = string
  sensitive   = true
  default     = ""
}

variable "stripe_secret_key_arn" {
  description = "Secrets Manager ARN containing the Stripe restricted or secret API key."
  type        = string
  sensitive   = true
  default     = ""
}

variable "stripe_webhook_secret_arn" {
  description = "Secrets Manager ARN containing the Stripe webhook signing secret."
  type        = string
  sensitive   = true
  default     = ""
}

variable "toss_secret_key_arn" {
  description = "Secrets Manager ARN containing the Toss Payments secret key."
  type        = string
  sensitive   = true
  default     = ""
}

variable "toss_client_key" {
  description = "Toss Payments client key used by the browser SDK."
  type        = string
  sensitive   = true
  default     = ""
}

variable "checkout_return_origin" {
  type    = string
  default = "https://studio.dajoongbim.com"
}

variable "app_url" {
  description = "Public Dajoong URL used in workspace invitation links."
  type        = string
  default     = "https://dajoongbim.com"
}

variable "invite_from_email" {
  description = "Verified SES sender used for workspace invitations; empty disables email delivery."
  type        = string
  default     = ""
}

variable "invite_identity_arn" {
  description = "SES identity ARN that authorizes workspace invitation delivery."
  type        = string
  default     = ""
}

variable "price_usd_cents" {
  type    = number
  default = 399
}

variable "price_krw" {
  type    = number
  default = 5000
}

variable "monthly_price_usd_cents" {
  type    = number
  default = 7900
}

variable "monthly_price_krw" {
  type    = number
  default = 99000
}

variable "cors_origins" {
  type    = string
  default = "https://studio.dajoongbim.com,https://app.dajoongbim.com,https://localhost,capacitor://localhost"
}

variable "oauth_callback_urls" {
  description = "Allowed web and native authorization-code callbacks."
  type        = list(string)
  default = [
    "https://studio.dajoongbim.com/studio",
    "https://app.dajoongbim.com/studio",
    "com.dajoong.plan2bim://auth/callback",
  ]
}

variable "oauth_logout_urls" {
  description = "Allowed web and native post-logout redirects."
  type        = list(string)
  default = [
    "https://studio.dajoongbim.com/",
    "https://app.dajoongbim.com/",
    "com.dajoong.plan2bim://auth/logout",
  ]
}
