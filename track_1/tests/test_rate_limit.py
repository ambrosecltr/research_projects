from __future__ import annotations

from dataclasses import dataclass

import pytest

from poetry50m.rate_limit import DualTokenBucket


@dataclass
class FakeTime:
    now: float = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_dual_bucket_waits_for_both_limits() -> None:
    fake = FakeTime()
    bucket = DualTokenBucket(
        requests_per_minute=2,
        tokens_per_minute=100,
        clock=fake.clock,
        sleeper=fake.sleep,
    )

    bucket.acquire(50)
    bucket.acquire(50)
    bucket.acquire(50)

    assert fake.now == pytest.approx(30.0)


def test_refund_releases_unused_model_tokens() -> None:
    fake = FakeTime()
    bucket = DualTokenBucket(
        requests_per_minute=2,
        tokens_per_minute=100,
        clock=fake.clock,
        sleeper=fake.sleep,
    )

    bucket.acquire(80)
    bucket.refund_model_tokens(reserved=80, used=20)
    bucket.acquire(80)

    assert fake.now == 0.0


@pytest.mark.parametrize("model_tokens", (0, 101))
def test_dual_bucket_rejects_invalid_reservations(model_tokens: int) -> None:
    bucket = DualTokenBucket(requests_per_minute=2, tokens_per_minute=100)
    with pytest.raises(ValueError):
        bucket.acquire(model_tokens)
