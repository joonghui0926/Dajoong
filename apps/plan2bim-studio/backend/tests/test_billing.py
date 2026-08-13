from __future__ import annotations

from pathlib import Path

import pytest

from buili_plan2bim_studio.billing import BillingService, InsufficientCredit


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BillingService:
    monkeypatch.setenv("DAJOONG_RUNTIME", "local")
    monkeypatch.setenv("DAJOONG_BILLING_ENFORCE", "true")
    monkeypatch.delenv("DAJOONG_STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("DAJOONG_TOSS_SECRET_KEY", raising=False)
    monkeypatch.delenv("DAJOONG_TOSS_CLIENT_KEY", raising=False)
    return BillingService(tmp_path)


def test_first_drawing_is_free_and_reservation_is_idempotent(
    service: BillingService,
) -> None:
    first = service.reserve("account-a", 1, "same-request")
    replay = service.reserve("account-a", 1, "same-request")

    assert first == replay
    assert first.free_units == 1
    assert service.context("account-a", "US").free_units_remaining == 0
    with pytest.raises(InsufficientCredit):
        service.reserve("account-a", 1, "next-request")


def test_releasing_failed_request_restores_the_exact_credit_source(
    service: BillingService,
) -> None:
    service.reserve("account-a", 1, "request-a")
    service.release("account-a", "request-a")
    account = service.context("account-a", "US")

    assert account.free_units_remaining == 1
    assert account.paid_units == 0
    retried = service.reserve("account-a", 1, "request-a")
    assert retried.status == "reserved"
    assert service.context("account-a", "US").free_units_remaining == 0


def test_paid_order_grants_once_and_country_changes_payment_rail(
    service: BillingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAJOONG_TOSS_CLIENT_KEY", "test_ck_example")
    monkeypatch.setenv("DAJOONG_TOSS_SECRET_KEY", "test_sk_example")
    order, context = service.create_order("account-a", "toss", "KR", 2)
    service.store.complete_order(order.id, "payment-key")
    service.store.complete_order(order.id, "payment-key")
    account = service.context("account-a", "KR")

    assert context.currency == "KRW"
    assert context.unit_amount == 5_000
    assert context.monthly_amount == 99_000
    assert context.configured_providers == ["toss"]
    assert account.paid_units == 2


def test_toss_customer_key_is_stable_and_does_not_expose_account_id(
    service: BillingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAJOONG_TOSS_CLIENT_KEY", "test_ck_example")
    monkeypatch.setenv("DAJOONG_TOSS_SECRET_KEY", "test_sk_example")
    first, _ = service.create_order("private-account-id", "toss", "KR", 1)
    second, _ = service.create_order("private-account-id", "toss", "KR", 2)

    first_key = service.toss_prepare(first)["customer_key"]
    second_key = service.toss_prepare(second)["customer_key"]

    assert first_key == second_key
    assert "private-account-id" not in first_key
    assert len(first_key) <= 50


def test_checkout_context_exposes_reproducible_speed_evidence(
    service: BillingService,
) -> None:
    context = service.context("account-a", "US")

    assert context.speed_runs == 7
    assert context.speed_median_seconds == pytest.approx(2.720126)
    assert context.speed_comparison_multiple == 21_175
    assert context.speed_benchmark_url.endswith("plan2bim-speed-2026-08-09.json")


def test_checkout_order_is_idempotent_per_account(
    service: BillingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAJOONG_TOSS_CLIENT_KEY", "test_ck_example")
    monkeypatch.setenv("DAJOONG_TOSS_SECRET_KEY", "test_sk_example")
    first, _ = service.create_order(
        "account-a", "toss", "KR", 2, "per_drawing", "checkout-request-1"
    )
    replay, _ = service.create_order(
        "account-a", "toss", "KR", 2, "per_drawing", "checkout-request-1"
    )

    assert replay.id == first.id
    with pytest.raises(ValueError, match="different details"):
        service.create_order(
            "account-a", "toss", "KR", 1, "per_drawing", "checkout-request-1"
        )


def test_monthly_unlimited_grants_access_without_consuming_credits(
    service: BillingService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAJOONG_STRIPE_SECRET_KEY", "sk_test_example")
    order, context = service.create_order(
        "account-a",
        "stripe",
        "US",
        1,
        "unlimited_monthly",
    )
    assert order.amount == 7_900
    service.store.complete_order(order.id, "payment-id")

    active = service.context("account-a", "US")
    assert active.unlimited_active is True
    reservation = service.reserve("account-a", 4, "unlimited-request")
    assert reservation.free_units == 0
    assert reservation.paid_units == 0
    assert service.context("account-a", "US").free_units_remaining == context.free_units_remaining
