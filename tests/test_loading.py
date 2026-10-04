import pytest

from tensegrity.loading import (
    linear_ramp,
    linear_ramp_rate,
    smoothstep_ramp,
    smoothstep_ramp_rate,
)


def test_linear_ramp_and_hold():
    assert linear_ramp(0.0, 2.0, 4.0) == 0.0
    assert linear_ramp(2.0, 2.0, 4.0) == pytest.approx(1.0)
    assert linear_ramp(4.0, 2.0, 4.0) == pytest.approx(2.0)
    assert linear_ramp(10.0, 2.0, 4.0) == pytest.approx(2.0)
    assert linear_ramp_rate(2.0, 2.0, 4.0) == pytest.approx(0.5)
    assert linear_ramp_rate(5.0, 2.0, 4.0) == 0.0


def test_smoothstep_ramp_has_zero_endpoint_rate():
    assert smoothstep_ramp(0.0, 3.0, 2.0) == 0.0
    assert smoothstep_ramp(1.0, 3.0, 2.0) == pytest.approx(1.5)
    assert smoothstep_ramp(2.0, 3.0, 2.0) == pytest.approx(3.0)
    assert smoothstep_ramp_rate(0.0, 3.0, 2.0) == 0.0
    assert smoothstep_ramp_rate(2.0, 3.0, 2.0) == 0.0


def test_invalid_ramp_time():
    with pytest.raises(ValueError):
        linear_ramp(1.0, 1.0, 0.0)
