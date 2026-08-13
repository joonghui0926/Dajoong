from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

Currency = Literal["USD", "KRW"]
Provider = Literal["stripe", "toss", "apple", "google"]
Plan = Literal["per_drawing", "unlimited_monthly"]


class InsufficientCredit(RuntimeError):
    def __init__(self, required: int, available: int) -> None:
        super().__init__("more conversion credits are required")
        self.required = required
        self.available = available


class BillingAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: str
    free_units_remaining: int = 1
    paid_units: int = 0
    unlimited_until: int = 0
    updated_at: int = 0


class BillingOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    request_id: str = ""
    account_id: str
    provider: Provider
    amount: int
    currency: Currency
    plan: Plan = "per_drawing"
    units: int = Field(default=1, ge=1, le=100)
    status: Literal["pending", "paid", "failed", "cancelled"] = "pending"
    provider_payment_id: str = ""
    created_at: int = 0
    updated_at: int = 0


class CreditReservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    account_id: str
    units: int
    free_units: int
    paid_units: int
    status: Literal["reserved", "released"] = "reserved"
    created_at: int = 0


class CheckoutContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str
    currency: Currency
    unit_amount: int
    unit_label: str
    monthly_amount: int
    monthly_label: str
    unlimited_active: bool
    unlimited_until: int
    free_units_remaining: int
    paid_units: int
    billing_enforced: bool
    configured_providers: list[Provider]
    native_provider: Literal["apple", "google", ""] = ""
    comparison_multiple: float = 5.01
    comparison_basis: str = "T-company $20 minimum 3D order"
    monthly_comparison_multiple: float = 4.81
    monthly_comparison_basis: str = (
        "Autodesk Revit standard monthly subscription ($380/month)"
    )
    monthly_comparison_source_url: str = (
        "https://www.autodesk.com/solutions/revit-subscription-faq"
    )
    comparison_source_url: str = (
        "https://support.twindo.com/article/716-how-does-scan-to-cad-pricing-work"
    )
    speed_median_seconds: float = 2.720126
    speed_p95_seconds: float = 12.972123
    speed_runs: int = 7
    speed_comparison_multiple: int = 21_175
    speed_benchmark_url: str = "/benchmarks/plan2bim-speed-2026-08-09.json"
    speed_turnaround_source_url: str = (
        "https://support.twindo.com/article/721-how-long-will-it-take-to-receive-my-order"
    )


class LocalBillingStore:
    """Small durable adapter for local development and single-node demos."""

    def __init__(self, root: Path) -> None:
        self.path = root.resolve() / "billing.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {"accounts": {}, "orders": {}, "reservations": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        return {
            "accounts": dict(payload.get("accounts") or {}),
            "orders": dict(payload.get("orders") or {}),
            "reservations": dict(payload.get("reservations") or {}),
        }

    def _write(self, payload: dict[str, dict[str, Any]]) -> None:
        staging = self.path.with_suffix(".tmp")
        staging.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        staging.replace(self.path)

    def get_account(self, account_id: str) -> BillingAccount:
        with self._lock:
            payload = self._read()
            raw = payload["accounts"].get(account_id)
            if raw:
                return BillingAccount.model_validate(raw)
            return BillingAccount(account_id=account_id)

    def reserve(self, account_id: str, units: int, reservation_id: str) -> CreditReservation:
        key = f"{account_id}:{reservation_id}"
        with self._lock:
            payload = self._read()
            existing = payload["reservations"].get(key)
            if existing:
                reservation = CreditReservation.model_validate(existing)
                if reservation.status == "reserved":
                    return reservation
                payload["reservations"].pop(key, None)
            account = BillingAccount.model_validate(
                payload["accounts"].get(account_id) or {"account_id": account_id}
            )
            if account.unlimited_until > int(time.time()):
                reservation = CreditReservation(
                    id=reservation_id,
                    account_id=account_id,
                    units=units,
                    free_units=0,
                    paid_units=0,
                    created_at=int(time.time()),
                )
                payload["reservations"][key] = reservation.model_dump(mode="json")
                self._write(payload)
                return reservation
            available = account.free_units_remaining + account.paid_units
            if available < units:
                raise InsufficientCredit(units, available)
            free_units = min(account.free_units_remaining, units)
            paid_units = units - free_units
            account.free_units_remaining -= free_units
            account.paid_units -= paid_units
            account.updated_at = int(time.time())
            reservation = CreditReservation(
                id=reservation_id,
                account_id=account_id,
                units=units,
                free_units=free_units,
                paid_units=paid_units,
                created_at=account.updated_at,
            )
            payload["accounts"][account_id] = account.model_dump(mode="json")
            payload["reservations"][key] = reservation.model_dump(mode="json")
            self._write(payload)
            return reservation

    def release(self, account_id: str, reservation_id: str) -> None:
        key = f"{account_id}:{reservation_id}"
        with self._lock:
            payload = self._read()
            raw = payload["reservations"].get(key)
            if not raw:
                return
            reservation = CreditReservation.model_validate(raw)
            if reservation.status == "released":
                return
            account = BillingAccount.model_validate(
                payload["accounts"].get(account_id) or {"account_id": account_id}
            )
            account.free_units_remaining += reservation.free_units
            account.paid_units += reservation.paid_units
            account.updated_at = int(time.time())
            reservation.status = "released"
            payload["accounts"][account_id] = account.model_dump(mode="json")
            payload["reservations"][key] = reservation.model_dump(mode="json")
            self._write(payload)

    def create_order(
        self,
        account_id: str,
        provider: Provider,
        amount: int,
        currency: Currency,
        units: int,
        plan: Plan = "per_drawing",
        request_id: str = "",
    ) -> BillingOrder:
        now = int(time.time())
        order_id = (
            f"DJ-{hashlib.sha256(f'{account_id}:{request_id}'.encode()).hexdigest()[:24]}"
            if request_id
            else f"DJ-{uuid.uuid4().hex[:24]}"
        )
        order = BillingOrder(
            id=order_id,
            request_id=request_id,
            account_id=account_id,
            provider=provider,
            amount=amount,
            currency=currency,
            plan=plan,
            units=units,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            payload = self._read()
            existing = payload["orders"].get(order.id)
            if existing:
                current = BillingOrder.model_validate(existing)
                if (
                    current.account_id,
                    current.provider,
                    current.amount,
                    current.currency,
                    current.units,
                    current.plan,
                ) != (account_id, provider, amount, currency, units, plan):
                    raise ValueError("checkout idempotency key was reused with different details")
                return current
            payload["orders"][order.id] = order.model_dump(mode="json")
            self._write(payload)
        return order

    def get_order(self, order_id: str) -> BillingOrder:
        with self._lock:
            raw = self._read()["orders"].get(order_id)
        if not raw:
            raise KeyError(order_id)
        return BillingOrder.model_validate(raw)

    def complete_order(self, order_id: str, provider_payment_id: str) -> BillingAccount:
        with self._lock:
            payload = self._read()
            raw = payload["orders"].get(order_id)
            if not raw:
                raise KeyError(order_id)
            order = BillingOrder.model_validate(raw)
            account = BillingAccount.model_validate(
                payload["accounts"].get(order.account_id) or {"account_id": order.account_id}
            )
            if order.status != "paid":
                order.status = "paid"
                order.provider_payment_id = provider_payment_id
                order.updated_at = int(time.time())
                if order.plan == "unlimited_monthly":
                    account.unlimited_until = (
                        max(account.unlimited_until, order.updated_at) + 31 * 24 * 60 * 60
                    )
                else:
                    account.paid_units += order.units
                account.updated_at = order.updated_at
                payload["orders"][order.id] = order.model_dump(mode="json")
                payload["accounts"][order.account_id] = account.model_dump(mode="json")
                self._write(payload)
            return account


class AwsBillingStore:
    """DynamoDB adapter with conditional writes for credits and webhook idempotency."""

    def __init__(self) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the backend with the aws extra") from exc
        self.table_name = os.environ["DAJOONG_BILLING_TABLE"]
        self.ddb = boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION", "us-west-2"))

    @staticmethod
    def _account_key(account_id: str) -> str:
        return f"ACCOUNT#{account_id}"

    @staticmethod
    def _order_key(order_id: str) -> str:
        return f"ORDER#{order_id}"

    @staticmethod
    def _reservation_key(account_id: str, reservation_id: str) -> str:
        digest = hashlib.sha256(f"{account_id}:{reservation_id}".encode()).hexdigest()
        return f"RESERVATION#{digest}"

    def _ensure_account(self, account_id: str) -> None:
        now = int(time.time())
        try:
            self.ddb.put_item(
                TableName=self.table_name,
                Item={
                    "record_id": {"S": self._account_key(account_id)},
                    "account_id": {"S": account_id},
                    "kind": {"S": "account"},
                    "free_units_remaining": {"N": "1"},
                    "paid_units": {"N": "0"},
                    "unlimited_until": {"N": "0"},
                    "updated_at": {"N": str(now)},
                },
                ConditionExpression="attribute_not_exists(record_id)",
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code != "ConditionalCheckFailedException":
                raise

    def get_account(self, account_id: str) -> BillingAccount:
        self._ensure_account(account_id)
        response = self.ddb.get_item(
            TableName=self.table_name,
            Key={"record_id": {"S": self._account_key(account_id)}},
            ConsistentRead=True,
        )
        item = response["Item"]
        return BillingAccount(
            account_id=account_id,
            free_units_remaining=int(item["free_units_remaining"]["N"]),
            paid_units=int(item["paid_units"]["N"]),
            unlimited_until=int(item.get("unlimited_until", {"N": "0"})["N"]),
            updated_at=int(item["updated_at"]["N"]),
        )

    def reserve(
        self,
        account_id: str,
        units: int,
        reservation_id: str,
        _attempt: int = 0,
    ) -> CreditReservation:
        key = self._reservation_key(account_id, reservation_id)
        existing = self.ddb.get_item(
            TableName=self.table_name,
            Key={"record_id": {"S": key}},
            ConsistentRead=True,
        ).get("Item")
        if existing:
            reservation = CreditReservation.model_validate_json(existing["data"]["S"])
            if reservation.status == "reserved":
                return reservation
            try:
                self.ddb.delete_item(
                    TableName=self.table_name,
                    Key={"record_id": {"S": key}},
                    ConditionExpression="#data = :released_data",
                    ExpressionAttributeNames={"#data": "data"},
                    ExpressionAttributeValues={
                        ":released_data": {"S": existing["data"]["S"]},
                    },
                )
            except Exception:
                if _attempt >= 4:
                    raise
                return self.reserve(account_id, units, reservation_id, _attempt + 1)
            return self.reserve(account_id, units, reservation_id, _attempt + 1)
        account = self.get_account(account_id)
        if account.unlimited_until > int(time.time()):
            reservation = CreditReservation(
                id=reservation_id,
                account_id=account_id,
                units=units,
                free_units=0,
                paid_units=0,
                created_at=int(time.time()),
            )
            try:
                self.ddb.put_item(
                    TableName=self.table_name,
                    Item={
                        "record_id": {"S": key},
                        "kind": {"S": "reservation"},
                        "data": {"S": reservation.model_dump_json()},
                    },
                    ConditionExpression="attribute_not_exists(record_id)",
                )
            except Exception:
                existing = self.ddb.get_item(
                    TableName=self.table_name,
                    Key={"record_id": {"S": key}},
                    ConsistentRead=True,
                ).get("Item")
                if existing:
                    return CreditReservation.model_validate_json(existing["data"]["S"])
                raise
            return reservation
        available = account.free_units_remaining + account.paid_units
        if available < units:
            raise InsufficientCredit(units, available)
        free_units = min(account.free_units_remaining, units)
        paid_units = units - free_units
        reservation = CreditReservation(
            id=reservation_id,
            account_id=account_id,
            units=units,
            free_units=free_units,
            paid_units=paid_units,
            created_at=int(time.time()),
        )
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {"record_id": {"S": self._account_key(account_id)}},
                            "UpdateExpression": (
                                "SET free_units_remaining = :new_free, paid_units = :new_paid, "
                                "updated_at = :now"
                            ),
                            "ConditionExpression": (
                                "free_units_remaining = :old_free AND paid_units = :old_paid"
                            ),
                            "ExpressionAttributeValues": {
                                ":new_free": {"N": str(account.free_units_remaining - free_units)},
                                ":new_paid": {"N": str(account.paid_units - paid_units)},
                                ":old_free": {"N": str(account.free_units_remaining)},
                                ":old_paid": {"N": str(account.paid_units)},
                                ":now": {"N": str(int(time.time()))},
                            },
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": {
                                "record_id": {"S": key},
                                "kind": {"S": "reservation"},
                                "data": {"S": reservation.model_dump_json()},
                            },
                            "ConditionExpression": "attribute_not_exists(record_id)",
                        }
                    },
                ]
            )
        except Exception as exc:
            existing = self.ddb.get_item(
                TableName=self.table_name,
                Key={"record_id": {"S": key}},
                ConsistentRead=True,
            ).get("Item")
            if existing:
                return CreditReservation.model_validate_json(existing["data"]["S"])
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            retryable = code in {
                "ConditionalCheckFailedException",
                "TransactionCanceledException",
            }
            if not retryable or _attempt >= 4:
                raise
            time.sleep(0.01 * (2**_attempt))
            return self.reserve(account_id, units, reservation_id, _attempt + 1)
        return reservation

    def release(self, account_id: str, reservation_id: str) -> None:
        key = self._reservation_key(account_id, reservation_id)
        response = self.ddb.get_item(
            TableName=self.table_name,
            Key={"record_id": {"S": key}},
            ConsistentRead=True,
        )
        if "Item" not in response:
            return
        reservation = CreditReservation.model_validate_json(response["Item"]["data"]["S"])
        if reservation.status == "released":
            return
        reservation.status = "released"
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {"record_id": {"S": self._account_key(account_id)}},
                            "UpdateExpression": (
                                "SET updated_at = :now "
                                "ADD free_units_remaining :free, paid_units :paid"
                            ),
                            "ExpressionAttributeValues": {
                                ":free": {"N": str(reservation.free_units)},
                                ":paid": {"N": str(reservation.paid_units)},
                                ":now": {"N": str(int(time.time()))},
                            },
                        }
                    },
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {"record_id": {"S": key}},
                            "UpdateExpression": "SET #data = :data",
                            "ConditionExpression": "contains(#data, :reserved)",
                            "ExpressionAttributeNames": {"#data": "data"},
                            "ExpressionAttributeValues": {
                                ":data": {"S": reservation.model_dump_json()},
                                ":reserved": {"S": '\"status\":\"reserved\"'},
                            },
                        }
                    },
                ]
            )
        except Exception:
            return

    def create_order(
        self,
        account_id: str,
        provider: Provider,
        amount: int,
        currency: Currency,
        units: int,
        plan: Plan = "per_drawing",
        request_id: str = "",
    ) -> BillingOrder:
        now = int(time.time())
        order_id = (
            f"DJ-{hashlib.sha256(f'{account_id}:{request_id}'.encode()).hexdigest()[:24]}"
            if request_id
            else f"DJ-{uuid.uuid4().hex[:24]}"
        )
        order = BillingOrder(
            id=order_id,
            request_id=request_id,
            account_id=account_id,
            provider=provider,
            amount=amount,
            currency=currency,
            plan=plan,
            units=units,
            created_at=now,
            updated_at=now,
        )
        try:
            self.ddb.put_item(
                TableName=self.table_name,
                Item={
                    "record_id": {"S": self._order_key(order.id)},
                    "kind": {"S": "order"},
                    "data": {"S": order.model_dump_json()},
                },
                ConditionExpression="attribute_not_exists(record_id)",
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code != "ConditionalCheckFailedException" or not request_id:
                raise
            current = self.get_order(order.id)
            if (
                current.account_id,
                current.provider,
                current.amount,
                current.currency,
                current.units,
                current.plan,
            ) != (account_id, provider, amount, currency, units, plan):
                raise ValueError(
                    "checkout idempotency key was reused with different details"
                ) from exc
            return current
        return order

    def get_order(self, order_id: str) -> BillingOrder:
        response = self.ddb.get_item(
            TableName=self.table_name,
            Key={"record_id": {"S": self._order_key(order_id)}},
            ConsistentRead=True,
        )
        if "Item" not in response:
            raise KeyError(order_id)
        return BillingOrder.model_validate_json(response["Item"]["data"]["S"])

    def complete_order(self, order_id: str, provider_payment_id: str) -> BillingAccount:
        order = self.get_order(order_id)
        if order.status == "paid":
            return self.get_account(order.account_id)
        self._ensure_account(order.account_id)
        order.status = "paid"
        order.provider_payment_id = provider_payment_id
        order.updated_at = int(time.time())
        account_update = (
            {
                "UpdateExpression": "SET updated_at = :now, unlimited_until = :until",
                "ExpressionAttributeValues": {
                        ":until": {
                            "N": str(
                                max(
                                    self.get_account(order.account_id).unlimited_until,
                                    order.updated_at,
                                )
                                + 31 * 24 * 60 * 60
                            )
                        },
                    ":now": {"N": str(order.updated_at)},
                },
            }
            if order.plan == "unlimited_monthly"
            else {
                "UpdateExpression": "SET updated_at = :now ADD paid_units :units",
                "ExpressionAttributeValues": {
                    ":units": {"N": str(order.units)},
                    ":now": {"N": str(order.updated_at)},
                },
            }
        )
        try:
            self.ddb.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {"record_id": {"S": self._order_key(order.id)}},
                            "UpdateExpression": "SET #data = :data",
                            "ConditionExpression": "contains(#data, :pending)",
                            "ExpressionAttributeNames": {"#data": "data"},
                            "ExpressionAttributeValues": {
                                ":data": {"S": order.model_dump_json()},
                                ":pending": {"S": '\"status\":\"pending\"'},
                            },
                        }
                    },
                    {
                        "Update": {
                            "TableName": self.table_name,
                            "Key": {"record_id": {"S": self._account_key(order.account_id)}},
                            **account_update,
                        }
                    },
                ]
            )
        except Exception:
            latest = self.get_order(order_id)
            if latest.status != "paid":
                raise
        return self.get_account(order.account_id)


class BillingService:
    def __init__(self, data_root: Path) -> None:
        self.store: LocalBillingStore | AwsBillingStore
        if os.environ.get("DAJOONG_RUNTIME", "local").lower() == "aws":
            self.store = AwsBillingStore()
        else:
            self.store = LocalBillingStore(data_root)

    @staticmethod
    def billing_enforced() -> bool:
        default = "true" if os.environ.get("DAJOONG_RUNTIME", "local") == "aws" else "false"
        return os.environ.get("DAJOONG_BILLING_ENFORCE", default).lower() == "true"

    @staticmethod
    def _country(country: str) -> str:
        value = country.strip().upper()
        return value if len(value) == 2 and value.isalpha() else "US"

    def context(
        self,
        account_id: str,
        country: str,
        native_platform: str = "",
    ) -> CheckoutContext:
        country = self._country(country)
        account = self.store.get_account(account_id)
        is_korea = country == "KR"
        providers: list[Provider] = []
        if os.environ.get("DAJOONG_STRIPE_SECRET_KEY"):
            providers.append("stripe")
        if is_korea and os.environ.get("DAJOONG_TOSS_CLIENT_KEY") and os.environ.get(
            "DAJOONG_TOSS_SECRET_KEY"
        ):
            providers.append("toss")
        native_provider: Literal["apple", "google", ""] = ""
        if native_platform == "ios":
            native_provider = "apple"
            if os.environ.get("DAJOONG_APPLE_IAP_PRODUCT_ID"):
                providers = ["apple"]
        elif native_platform == "android":
            native_provider = "google"
            if os.environ.get("DAJOONG_GOOGLE_PLAY_PRODUCT_ID"):
                providers = ["google"]
        return CheckoutContext(
            country=country,
            currency="KRW" if is_korea else "USD",
            unit_amount=(
                int(os.environ.get("DAJOONG_PRICE_KRW", "5000"))
                if is_korea
                else int(os.environ.get("DAJOONG_PRICE_USD_CENTS", "399"))
            ),
            unit_label="₩5,000 / drawing" if is_korea else "$3.99 / drawing",
            monthly_amount=(
                int(os.environ.get("DAJOONG_MONTHLY_PRICE_KRW", "99000"))
                if is_korea
                else int(os.environ.get("DAJOONG_MONTHLY_PRICE_USD_CENTS", "7900"))
            ),
            monthly_label="₩99,000 / month" if is_korea else "$79 / month",
            unlimited_active=account.unlimited_until > int(time.time()),
            unlimited_until=account.unlimited_until,
            free_units_remaining=account.free_units_remaining,
            paid_units=account.paid_units,
            billing_enforced=self.billing_enforced(),
            configured_providers=providers,
            native_provider=native_provider,
        )

    def reserve(self, account_id: str, units: int, reservation_id: str) -> CreditReservation:
        return self.store.reserve(account_id, units, reservation_id)

    def release(self, account_id: str, reservation_id: str) -> None:
        self.store.release(account_id, reservation_id)

    def create_order(
        self,
        account_id: str,
        provider: Provider,
        country: str,
        units: int,
        plan: Plan = "per_drawing",
        request_id: str = "",
    ) -> tuple[BillingOrder, CheckoutContext]:
        context = self.context(account_id, country)
        if provider not in context.configured_providers:
            raise RuntimeError(f"{provider} is not configured for this checkout")
        order = self.store.create_order(
            account_id,
            provider,
            context.monthly_amount if plan == "unlimited_monthly" else context.unit_amount * units,
            context.currency,
            units,
            plan,
            request_id,
        )
        return order, context

    def stripe_checkout(self, order: BillingOrder, email: str = "") -> str:
        secret_key = os.environ["DAJOONG_STRIPE_SECRET_KEY"]
        return_origin = os.environ.get(
            "DAJOONG_CHECKOUT_RETURN_ORIGIN", "https://studio.dajoongbim.com"
        ).rstrip("/")
        data: list[tuple[str, str]] = [
            ("mode", "payment"),
            ("success_url", f"{return_origin}/studio?checkout=stripe-success"),
            ("cancel_url", f"{return_origin}/studio?checkout=cancelled"),
            ("client_reference_id", order.id),
            ("metadata[order_id]", order.id),
            ("line_items[0][quantity]", "1"),
            ("line_items[0][price_data][currency]", order.currency.lower()),
            ("line_items[0][price_data][unit_amount]", str(order.amount)),
            (
                "line_items[0][price_data][product_data][name]",
                (
                    "Dajoong monthly unlimited"
                    if order.plan == "unlimited_monthly"
                    else "Dajoong drawing conversion"
                ),
            ),
            (
                "line_items[0][price_data][product_data][description]",
                "31 days of unlimited drawing conversions"
                if order.plan == "unlimited_monthly"
                else f"{order.units} drawing conversion credit{'s' if order.units != 1 else ''}",
            ),
            ("billing_address_collection", "auto"),
            ("payment_method_collection", "if_required"),
        ]
        if email:
            data.append(("customer_email", email[:100]))
        response = httpx.post(
            "https://api.stripe.com/v1/checkout/sessions",
            data=data,
            headers={"Authorization": f"Bearer {secret_key}"},
            timeout=15,
        )
        response.raise_for_status()
        url = str(response.json().get("url") or "")
        if not url.startswith("https://checkout.stripe.com/"):
            raise RuntimeError("Stripe did not return a trusted checkout URL")
        return url

    @staticmethod
    def verify_stripe_signature(payload: bytes, signature: str) -> dict[str, Any]:
        secret = os.environ["DAJOONG_STRIPE_WEBHOOK_SECRET"]
        parts = {}
        for item in signature.split(","):
            key, _, value = item.partition("=")
            parts.setdefault(key, []).append(value)
        timestamp = int(parts.get("t", ["0"])[0])
        if abs(int(time.time()) - timestamp) > 300:
            raise ValueError("stale Stripe webhook")
        signed = str(timestamp).encode() + b"." + payload
        expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        if not any(hmac.compare_digest(expected, value) for value in parts.get("v1", [])):
            raise ValueError("invalid Stripe signature")
        return json.loads(payload)

    def complete_stripe_event(self, event: dict[str, Any]) -> BillingAccount | None:
        if event.get("type") != "checkout.session.completed":
            return None
        session = event.get("data", {}).get("object", {})
        if session.get("payment_status") != "paid":
            return None
        order_id = str(session.get("metadata", {}).get("order_id") or "")
        if not order_id:
            raise ValueError("Stripe checkout is missing the order ID")
        order = self.store.get_order(order_id)
        if order.amount != int(session.get("amount_total") or -1):
            raise ValueError("Stripe amount does not match the server order")
        return self.store.complete_order(
            order_id,
            str(session.get("payment_intent") or session["id"]),
        )

    def toss_prepare(self, order: BillingOrder, easy_pay: str = "") -> dict[str, Any]:
        return_origin = os.environ.get(
            "DAJOONG_CHECKOUT_RETURN_ORIGIN", "https://studio.dajoongbim.com"
        ).rstrip("/")
        customer_key = f"customer_{secrets.token_urlsafe(18)}"
        return {
            "client_key": os.environ["DAJOONG_TOSS_CLIENT_KEY"],
            "customer_key": customer_key,
            "method": "CARD",
            "easy_pay": easy_pay,
            "amount": {"value": order.amount, "currency": "KRW"},
            "order_id": order.id,
            "order_name": (
                "Dajoong monthly unlimited"
                if order.plan == "unlimited_monthly"
                else f"Dajoong drawing conversion × {order.units}"
            ),
            "success_url": f"{return_origin}/studio?checkout=toss-success",
            "fail_url": f"{return_origin}/studio?checkout=toss-failed",
        }

    def confirm_toss(
        self,
        account_id: str,
        order_id: str,
        payment_key: str,
        amount: int,
    ) -> BillingAccount:
        order = self.store.get_order(order_id)
        if order.account_id != account_id or order.provider != "toss":
            raise ValueError("Toss order does not belong to this account")
        if order.amount != amount or order.currency != "KRW":
            raise ValueError("Toss amount does not match the server order")
        encoded = base64.b64encode(
            f"{os.environ['DAJOONG_TOSS_SECRET_KEY']}:".encode()
        ).decode()
        response = httpx.post(
            "https://api.tosspayments.com/v1/payments/confirm",
            json={"paymentKey": payment_key, "orderId": order_id, "amount": order.amount},
            headers={
                "Authorization": f"Basic {encoded}",
                "Idempotency-Key": str(uuid.uuid5(uuid.NAMESPACE_URL, order_id)),
            },
            timeout=15,
        )
        response.raise_for_status()
        payment = response.json()
        if payment.get("status") != "DONE" or int(payment.get("totalAmount") or -1) != order.amount:
            raise ValueError("Toss did not return a completed payment")
        return self.store.complete_order(order_id, payment_key)
