from __future__ import annotations

import argparse
import json

from .aws_gateway import AwsJobGateway


def run(*, once: bool = False) -> None:
    gateway = AwsJobGateway()
    while True:
        response = gateway.sqs.receive_message(
            QueueUrl=gateway.queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=900,
            AttributeNames=["ApproximateReceiveCount"],
        )
        messages = response.get("Messages", [])
        if not messages:
            if once:
                return
            continue
        raw = messages[0]
        payload = json.loads(raw["Body"])
        try:
            gateway.process_message(payload)
        except Exception as exc:
            job = gateway.get(str(payload["job_id"]))
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            gateway.save(job)
            if int(raw.get("Attributes", {}).get("ApproximateReceiveCount", "1")) < 3:
                gateway.sqs.change_message_visibility(
                    QueueUrl=gateway.queue_url,
                    ReceiptHandle=raw["ReceiptHandle"],
                    VisibilityTimeout=30,
                )
                if once:
                    return
                continue
        gateway.sqs.delete_message(
            QueueUrl=gateway.queue_url,
            ReceiptHandle=raw["ReceiptHandle"],
        )
        if once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Dajoong Plan2BIM SQS worker")
    parser.add_argument("--once", action="store_true", help="Exit after one poll or job")
    args = parser.parse_args()
    run(once=args.once)


if __name__ == "__main__":
    main()
