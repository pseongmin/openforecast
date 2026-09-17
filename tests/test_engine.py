import numpy as np
import pandas as pd

from agenthon.qfagent.engine import ForecastRequest, Scenario, forecast_draws


def _panel():
    idx = pd.bdate_range("2019-01-01", "2024-05-31")
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.02, len(idx))
    return pd.DataFrame(
        {"A": 4 + np.cumsum(common + rng.normal(0, 0.01, len(idx))),
         "B": 3 + np.cumsum(common + rng.normal(0, 0.01, len(idx)))},
        index=idx,
    )


def test_output_schema_and_row_count():
    out = forecast_draws(_panel(), ForecastRequest(["A", "B"], [21, 63], "level", n_draws=300))
    assert list(out.columns) == ["draw", "asset", "horizon", "value"]
    assert str(out["draw"].dtype) == "int32" and str(out["horizon"].dtype) == "int32"
    assert len(out) == 300 * 2 * 2
    assert out.groupby(["asset", "horizon"])["draw"].nunique().eq(300).all()


def test_joint_sampling_preserves_cross_asset_correlation():
    out = forecast_draws(_panel(), ForecastRequest(["A", "B"], [21], "level", n_draws=2000))
    wide = out.pivot(index="draw", columns="asset", values="value")
    assert wide["A"].corr(wide["B"]) > 0.5


def test_spread_grows_with_horizon():
    out = forecast_draws(_panel(), ForecastRequest(["A"], [5, 60], "level", n_draws=1000))
    sd = out.groupby("horizon")["value"].std()
    assert sd[60] > sd[5] * 2


def test_scenario_shift_and_widening():
    base = forecast_draws(_panel(), ForecastRequest(["A"], [21], "level", n_draws=3000))
    shifted = forecast_draws(_panel(), ForecastRequest(["A"], [21], "level", n_draws=3000,
                                                       scenarios=[Scenario("up", 1.0, drift_in_sd=1.0)]))
    widened = forecast_draws(_panel(), ForecastRequest(["A"], [21], "level", n_draws=3000,
                                                       scenarios=[Scenario("stress", 0.5, spread_multiple=2.0), Scenario("calm", 0.5)]))
    assert shifted["value"].mean() > base["value"].mean() + 0.5 * base["value"].std()
    assert widened["value"].std() > base["value"].std() * 1.2


def test_log_return_target_starts_at_zero():
    idx = pd.bdate_range("2019-01-01", "2024-05-31")
    rng = np.random.default_rng(2)
    panel = pd.DataFrame({"MOM": rng.normal(0.0002, 0.01, len(idx))}, index=idx)
    out = forecast_draws(panel, ForecastRequest(["MOM"], [127], "log_return", n_draws=500))
    assert abs(out["value"].mean()) < 0.1 and out["value"].std() > 0.05


def test_absent_target_is_proxied_not_crashed():
    out = forecast_draws(_panel(), ForecastRequest(["A", "EMX"], [21], "level", n_draws=300))
    assert set(out["asset"]) == {"A", "EMX"}
    assert out[out.asset == "EMX"]["value"].std() > 0
