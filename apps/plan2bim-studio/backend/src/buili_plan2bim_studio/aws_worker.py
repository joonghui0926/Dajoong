from __future__ import annotations

import argparse
import json
import logging
import os
import threading

from .aws_gateway import AwsJobGateway

LOGGER = logging.getLogger("dajoong.worker")


def run(*, once: bool = False) -> None:
    gateway = AwsJobGateway()
    visibility_seconds = int(os.environ.get("DAJOONG_JOB_VISIBILITY_SECONDS", "900"))
    if not 60 <= visibility_seconds <= 43_200:
        raise RuntimeError("DAJOONG_JOB_VISIBILITY_SECONDS must be between 60 and 43200")
    while True:
        response = gateway.sqs.receive_message(
            QueueUrl=gateway.queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=visibility_seconds,
            AttributeNames=["ApproximateReceiveCount"],
        )
        messages = response.get("Messages", [])
        if not messages:
            if once:
                return
            continue
        raw = messages[0]
        payload: dict[str, object] | None = None
        stop_heartbeat = threading.Event()
        heartbeat: threading.Thread | None = None
        try:
            decoded = json.loads(raw["Body"])
            if not isinstance(decoded, dict) or not str(decoded.get("job_id") or "").isalnum():
                raise ValueError("queue message has no valid job_id")
            payload = decoded
            job_id = str(payload["job_id"])

            def renew_visibility(
                stop_event: threading.Event = stop_heartbeat,
                receipt_handle: str = raw["ReceiptHandle"],
                active_job_id: str = job_id,
            ) -> None:
                interval = max(15, visibility_seconds // 3)
                while not stop_event.wait(interval):
                    try:
                        gateway.sqs.change_message_visibility(
                            QueueUrl=gateway.queue_url,
                            ReceiptHandle=receipt_handle,
                            VisibilityTimeout=visibility_seconds,
                        )
                        gateway.renew_lease(active_job_id, visibility_seconds)
                    except Exception:
                        # Durable status and SQS redelivery remain the recovery boundary.
                        continue

            heartbeat = threading.Thread(
                target=renew_visibility,
                name=f"dajoong-lease-{job_id[:8]}",
                daemon=True,
            )
            heartbeat.start()
            gateway.process_message(payload)
        except Exception as exc:
            LOGGER.exception(
                "conversion message failed",
                extra={"job_id": str((payload or {}).get("job_id", ""))},
            )
            if payload is not None:
                gateway.fail_job(
                    str(payload["job_id"]),
                    f"{type(exc).__name__}: {exc}",
                )
            if int(raw.get("Attributes", {}).get("ApproximateReceiveCount", "1")) < 3:
                gateway.sqs.change_message_visibility(
                    QueueUrl=gateway.queue_url,
                    ReceiptHandle=raw["ReceiptHandle"],
                    VisibilityTimeout=30,
                )
                if once:
                    return
                continue
            # Do not acknowledge a poison message; the queue redrive policy moves it to the DLQ.
            if once:
                return
            continue
        finally:
            stop_heartbeat.set()
            if heartbeat is not None:
                heartbeat.join(timeout=1)
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
