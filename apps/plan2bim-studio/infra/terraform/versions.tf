terraform {
  required_version = ">= 1.8"

  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.80, < 7.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Product   = "Dajoong"
      Component = "Plan2BIM-Studio"
      ManagedBy = "Terraform"
    }
  }
}
