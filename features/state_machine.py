"""
Per-symbol Double ROC(3) state machine.
Mirrors 76_generate_v2_sweeps.py exactly.
"""

import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
ROC_THRESH = 1.0


class Signal:
    def __init__(self, symbol: str, direction: str, entry_price: float,
                 entry_time_ist: str, prob: float, roc_at_entry: float,
                 delta_value_lakhs: float, theta_caught_time: str,
                 mark1_time: str, between_time: str,
                 daily_open_passed: bool, bypass_used: bool,
                 prob_long: float = 0.0, prob_short: float = 0.0,
                 prob_no_trade: float = 0.0, theta_prob: float = 0.0):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time_ist = entry_time_ist
        self.prob = prob
        self.prob_long = prob_long
        self.prob_short = prob_short
        self.prob_no_trade = prob_no_trade
        self.theta_prob = theta_prob  # SANDBOX FIX: prob at ARMING bar (threshold answer)
        self.roc_at_entry = roc_at_entry
        self.delta_value_lakhs = delta_value_lakhs
        self.theta_caught_time = theta_caught_time
        self.mark1_time = mark1_time
        self.between_time = between_time
        self.daily_open_passed = daily_open_passed
        self.bypass_used = bypass_used


class DoubleROCStateMachine:
    """Per-symbol state machine. Exactly matches 76_generate_v2_sweeps.py."""

    def __init__(self, long_th: float = 0.25, short_th: float = 0.25):
        self.long_th = long_th
        self.short_th = short_th
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.theta_caught_ts: Optional[str] = None
        self.theta_prob: Optional[float] = None  # SANDBOX FIX
        self.mark1_ts: Optional[str] = None
        self.between_ts: Optional[str] = None
        self.roc_crossed = False

    def reset_daily(self):
        self.reset()

    def serialize(self) -> dict:
        return {
            "state": self.state,
            "theta_caught_ts": self.theta_caught_ts,
            "mark1_ts": self.mark1_ts,
            "between_ts": self.between_ts,
            "roc_crossed": self.roc_crossed,
        }

    def deserialize(self, d: dict):
        self.state = d.get("state", "IDLE")
        self.theta_caught_ts = d.get("theta_caught_ts")
        self.mark1_ts = d.get("mark1_ts")
        self.between_ts = d.get("between_ts")
        self.roc_crossed = d.get("roc_crossed", False)

    def step(self, bar: dict, direction: str, prob: float, roc_3: float,
             daily_open: float = 0.0, prob_long: float = 0.0,
             prob_short: float = 0.0, prob_no_trade: float = 0.0) -> Optional[Signal]:
        """One 1-min bar step. Returns Signal if TRIGGERED, else None."""
        theta = self.long_th if direction == "LONG" else self.short_th
        ts_str = self._ts_to_ist_str(bar)
        bar_ltp = bar["ltp"]

        if self.state == "IDLE":
            if not np.isnan(prob) and prob >= theta:
                self.state = "ARMING"
                self.theta_caught_ts = ts_str
                self.theta_prob = float(prob)  # SANDBOX FIX

        elif self.state == "ARMING":
            if not np.isnan(roc_3):
                if (direction == "LONG" and roc_3 >= ROC_THRESH) or \
                   (direction == "SHORT" and roc_3 <= -ROC_THRESH):
                    self.state = "MARK1"
                    self.mark1_ts = ts_str
                    self.roc_crossed = True

        elif self.state == "MARK1":
            if not np.isnan(roc_3):
                below = (direction == "LONG" and roc_3 < ROC_THRESH) or \
                        (direction == "SHORT" and roc_3 > -ROC_THRESH)
                if below:
                    self.state = "BETWEEN"
                    self.between_ts = ts_str

        elif self.state == "BETWEEN":
            if not np.isnan(roc_3):
                if (direction == "LONG" and roc_3 >= ROC_THRESH) or \
                   (direction == "SHORT" and roc_3 <= -ROC_THRESH):
                    self.state = "TRIGGERED"
                    daily_open_passed = (bar_ltp > daily_open) if direction == "LONG" else (bar_ltp < daily_open)
                    delta_val = float(bar.get("delta_1m", 0)) * bar_ltp / 100000.0

                    bypass_used = False
                    if not daily_open_passed:
                        if direction == "LONG" and prob >= 0.35:
                            bypass_used = True
                            daily_open_passed = True

                    return Signal(
                        symbol=bar.get("symbol", ""),
                        direction=direction,
                        entry_price=bar_ltp,
                        entry_time_ist=ts_str,
                        prob=prob,
                        prob_long=prob_long,
                        prob_short=prob_short,
                        prob_no_trade=prob_no_trade,
                        roc_at_entry=roc_3,
                        delta_value_lakhs=delta_val,
                        theta_caught_time=self.theta_caught_ts or "",
                        mark1_time=self.mark1_ts or "",
                        between_time=self.between_ts or "",
                        daily_open_passed=daily_open_passed,
                        bypass_used=bypass_used,
                        theta_prob=self.theta_prob if self.theta_prob is not None else 0.0,
                    )

        return None

    def get_rejection_reason(self) -> str:
        if self.state == "IDLE":
            return "NO_ALERT"
        elif self.state == "ARMING":
            return "MARK1_NEVER"
        elif self.state == "MARK1":
            return "BETWEEN_NEVER"
        elif self.state == "BETWEEN":
            return "MARK2_NEVER"
        return "TRIGGERED"

    @staticmethod
    def _ts_to_ist_str(bar: dict) -> str:
        ts = bar.get("ts", 0)
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            try:
                import pandas as pd
                dt = pd.Timestamp(ts).to_pydatetime()
            except Exception:
                dt = datetime.fromtimestamp(0, tz=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(IST)
        return ist.strftime("%H:%M:%S")
