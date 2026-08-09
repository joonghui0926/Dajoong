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
  default = 10
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

variable "cors_origins" {
  type    = string
  default = "https://studio.builiconstruction.com,https://app.builiconstruction.com,https://localhost,capacitor://localhost"
}

variable "oauth_callback_urls" {
  description = "Allowed web and native authorization-code callbacks."
  type        = list(string)
  default = [
    "https://studio.builiconstruction.com/studio",
    "https://app.builiconstruction.com/studio",
    "com.dajoong.plan2bim://auth/callback",
  ]
}

variable "oauth_logout_urls" {
  description = "Allowed web and native post-logout redirects."
  type        = list(string)
  default = [
    "https://studio.builiconstruction.com/",
    "https://app.builiconstruction.com/",
    "com.dajoong.plan2bim://auth/logout",
  ]
}
