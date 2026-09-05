"""
ML Engine — orchestrates live LightGBM + Double ROC(3) state machine.
Integrates into collector.py: called from _process() per tick.

Zero impact guarantee: all ML code wrapped in try/except.
If model fails to load or any step errors → logs warning, never blocks tick flow.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class MLEngine:
    """Per-process singleton. Initialized once at collector startup."""

    def __init__(self, symbol_segments=None, sid_to_symbol=None):
        self.model = None
        self.model_loaded = False
        self.model_path = os.getenv("LGBM_MODEL_PATH", "")
        self.symbol_segments = symbol_segments or {}
        self.sid_to_symbol = sid_to_symbol or {}
        self.long_th = float(os.getenv("LONG_TH", "0.25"))
        self.short_th = float(os.getenv("SHORT_TH", "0.25"))
        self.mode = os.getenv("ML_MODE", "VIRTUAL").upper()
        self.pmocata_url = os.getenv("PMOCATA_URL", "").strip()
        self.sl_pct = float(os.getenv("SL_PCT", "0.5"))
        self.tp_pct = float(os.getenv("TP_PCT", "1.0"))
        self.use_super_order = os.getenv("ML_USE_SUPER_ORDER", "false").lower() == "true"
        self.fixed_qty = int(os.getenv("ML_FIXED_QTY", "0"))
        self.order_flow_type = os.getenv("ML_ORDER_FLOW_TYPE", "STOP_LOSS_MARKET").strip().upper()
        self.limit_pct = float(os.getenv("ML_LIMIT_PCT", "2.0" if self.order_flow_type == "STOP_LOSS" else "0.0"))

        self.accumulators = {}
        self.computers = {}
        self.machines_long = {}
        self.machines_short = {}
        self._last_date = None
        self._db_ok = False
        self._state_file = os.getenv("ML_STATE_FILE", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spool", "ml_state.json"))
        self._save_thread = None
        self._daily_open = {}
        self._cycle_bars = 0
        self._cycle_time = 0.0
        self._cycle_predict_time = 0.0
        self._cycle_start = 0.0

    def start(self):
        """Load model, create ml_signals table. Call once at collector startup."""
        if not self.model_path:
            log.warning("ML Engine: LGBM_MODEL_PATH not set — ML scanning disabled.")
            return

        if not os.path.exists(self.model_path):
            log.warning(f"ML Engine: model not found at {self.model_path} — ML scanning disabled.")
            return

        try:
            import lightgbm as lgb
            t0 = time.time()
            self.model = lgb.Booster(model_file=self.model_path)
            t1 = time.time()
            self.model_loaded = True
            log.info(f"ML Engine: LightGBM model loaded in {(t1-t0)*1000:.0f}ms ({self.model_path})")
            log.info(f"ML Engine: thresholds LONG>={self.long_th} SHORT>={self.short_th} mode={self.mode}")
            if self.pmocata_url:
                log.info(f"ML Engine: pmocata webhook at {self.pmocata_url}")
        except ImportError:
            log.warning("ML Engine: lightgbm not installed — ML scanning disabled.")
            return
        except Exception as e:
            log.warning(f"ML Engine: failed to load model: {e} — ML scanning disabled.")
            return

        self._ensure_ml_signals_table()
        self._load_state()
        self._start_save_thread()

    def _start_save_thread(self):
        import threading
        def _loop():
            while True:
                time.sleep(60)
                try:
                    self._save_state()
                except Exception:
                    pass
        self._save_thread = threading.Thread(target=_loop, daemon=True)
        self._save_thread.start()

    def _save_state(self):
        data = {
            "last_date": str(self._last_date) if self._last_date else None,
            "machines_long": {sym: sm.serialize() for sym, sm in self.machines_long.items()},
            "machines_short": {sym: sm.serialize() for sym, sm in self.machines_short.items()},
        }
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        tmp = self._state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self._state_file)
        log.info(f"ML STATE saved: {len(self.machines_long)} long + {len(self.machines_short)} short machines")

    def _load_state(self):
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            ld = data.get("last_date")
            if ld:
                from datetime import date
                parsed = date.fromisoformat(ld)
                if parsed == date.today():
                    self._last_date = parsed
                    from state_machine import DoubleROCStateMachine
                    for sym, d in data.get("machines_long", {}).items():
                        sm = DoubleROCStateMachine(self.long_th, self.short_th)
                        sm.deserialize(d)
                        self.machines_long[sym] = sm
                    for sym, d in data.get("machines_short", {}).items():
                        sm = DoubleROCStateMachine(self.long_th, self.short_th)
                        sm.deserialize(d)
                        self.machines_short[sym] = sm
                    log.info(f"ML Engine: restored state for {len(self.machines_long)} long + {len(self.machines_short)} short machines")
        except Exception as e:
            log.warning(f"ML Engine: failed to load saved state: {e}")

    def _ensure_ml_signals_table(self):
        try:
            import psycopg2
            host = os.getenv("QUESTDB_HOST", "localhost")
            port = int(os.getenv("QUESTDB_PORT", "8812"))
            user = os.getenv("QUESTDB_USER", "admin")
            pwd = os.getenv("QUESTDB_PASS", "quest")
            db = os.getenv("QUESTDB_DB", "qdb")
            conn = psycopg2.connect(host=host, port=port, user=user, password=pwd,
                                    database=db, connect_timeout=5)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_signals (
                    ts TIMESTAMP,
                    symbol SYMBOL,
                    direction SYMBOL,
                    entry_price DOUBLE,
                    prob_long DOUBLE,
                    prob_short DOUBLE,
                    prob_no_trade DOUBLE,
                    prob DOUBLE,
                    roc_at_entry DOUBLE,
                    delta_value_lakhs DOUBLE,
                    theta_caught_time STRING,
                    mark1_time STRING,
                    between_time STRING,
                    entry_time_ist STRING,
                    theta_prob DOUBLE,
                    daily_open_passed BOOLEAN,
                    bypass_used BOOLEAN,
                    rejection_reason STRING,
                    mode SYMBOL
                ) TIMESTAMP(ts) PARTITION BY DAY
            """)
            cur.close()
            conn.close()
            self._db_ok = True
            log.info("ML Engine: ml_signals table ready")
        except Exception as e:
            log.warning(f"ML Engine: could not create ml_signals table: {e}")

    def on_tick(self, tick: dict, symbol: str):
        """Called from collector._process() for every tick. Never raises."""
        if not self.model_loaded:
            return
        try:
            acc = self.accumulators.get(symbol)
            if acc is None:
                from bar_accumulator import BarAccumulator
                acc = BarAccumulator()
                self.accumulators[symbol] = acc

            now = datetime.now(IST)
            current_date = now.date()
            if self._last_date is None:
                self._last_date = current_date
            elif current_date != self._last_date:
                self._last_date = current_date
                self.accumulators.clear()
                self.computers.clear()
                self.machines_long.clear()
                self.machines_short.clear()
                self._daily_open.clear()

            # SANDBOX FIX: do NOT seed from tick "open" (prev-close-like). Seed from
            # first bar open in _on_bar to match backtest day_met open.
            pass

            bar = acc.add_tick(tick)
            if bar:
                bar["symbol"] = symbol
                self._on_bar(bar, symbol)
        except Exception as e:
            log.warning(f"ML on_tick error [{symbol}]: {e}")

    def _on_bar(self, bar: dict, symbol: str):
        t0 = time.time()
        try:
            comp = self.computers.get(symbol)
            if comp is None:
                from live_computer import LiveFeatureComputer
                comp = LiveFeatureComputer()
                self.computers[symbol] = comp

            feat = comp.push_bar(bar)
            if comp.prev_bar is None:
                return

            t1 = time.time()
            probs = self.model.predict([feat])[0]
            t2 = time.time()
            prob_long = float(probs[0])
            prob_short = float(probs[1])
            prob_no_trade = float(probs[2])

            bar["prob_long"] = prob_long
            bar["prob_short"] = prob_short
            bar["prob_no_trade"] = prob_no_trade
            bar["delta_1m"] = float(feat[1])
            roc_3 = comp.roc_3
            bar_close = bar["close"]
            # SANDBOX FIX: seed daily open from first bar's open (match backtest)
            if symbol not in self._daily_open:
                bar_open = float(bar.get("open", 0.0) or 0.0)
                if bar_open > 0.0:
                    self._daily_open[symbol] = bar_open

            daily_open = self._get_daily_open(symbol, bar["ltp"])

            for direction, prob_val, machines in [
                ("LONG", prob_long, self.machines_long),
                ("SHORT", prob_short, self.machines_short)
            ]:
                sm = machines.get(symbol)
                if sm is None:
                    from state_machine import DoubleROCStateMachine
                    sm = DoubleROCStateMachine(self.long_th, self.short_th)
                    machines[symbol] = sm

                sig = sm.step(
                    bar, direction, prob_val, roc_3,
                    daily_open=daily_open,
                    prob_long=prob_long,
                    prob_short=prob_short,
                    prob_no_trade=prob_no_trade,
                )
                if sig:
                    self._on_signal(sig)

            t3 = time.time()
            feat_us = int((t1 - t0) * 1_000_000)
            pred_us = int((t2 - t1) * 1_000_000)
            sm_us = int((t3 - t2) * 1_000_000)
            total_us = int((t3 - t0) * 1_000_000)
            self._cycle_bars += 1
            self._cycle_time += total_us
            self._cycle_predict_time += pred_us
            if self._cycle_start == 0:
                self._cycle_start = time.time()
            elapsed = time.time() - self._cycle_start
            if elapsed >= 60.0:
                avg_total = self._cycle_time / max(self._cycle_bars, 1)
                avg_pred = self._cycle_predict_time / max(self._cycle_bars, 1)
                log.info(f"ML TIMING: {self._cycle_bars} bars in {elapsed:.1f}s — "
                         f"avg {avg_total:.0f}µs/bar (feat {avg_pred:.0f}µs pred) "
                         f"| symbols active: acc={len(self.accumulators)} comp={len(self.computers)} "
                         f"long_sm={len(self.machines_long)} short_sm={len(self.machines_short)}")
                self._cycle_bars = 0
                self._cycle_time = 0.0
                self._cycle_predict_time = 0.0
                self._cycle_start = 0.0

            if total_us > 200_000:
                log.warning(f"ML SLOW BAR [{symbol}]: {total_us}µs (feat={feat_us} pred={pred_us} sm={sm_us})")

        except Exception as e:
            log.warning(f"ML _on_bar error [{symbol}]: {e}")

    def _is_eq_symbol(self, symbol: str) -> bool:
        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"):
            return False
        if "-FUT" in symbol.upper():
            return False
        if "-CE" in symbol.upper() or "-PE" in symbol.upper():
            return False
        if symbol.isdigit():
            return False
        return True

    def _on_signal(self, sig):
        try:
            if not self._is_eq_symbol(sig.symbol):
                log.info(f"ML SIGNAL skipped (non-EQ) [{sig.direction}] {sig.symbol} @ {sig.entry_price}")
                return

            now_utc = datetime.now(timezone.utc)
            bar_ts = datetime.strptime(sig.entry_time_ist, "%H:%M:%S").time()
            now_ist = datetime.now(IST).time()
            bar_dt = datetime.combine(datetime.now(IST).date(), bar_ts)
            now_dt = datetime.combine(datetime.now(IST).date(), now_ist)
            latency_s = (now_dt - bar_dt).total_seconds()
            rejection = "EXECUTED"

            if self._db_ok:
                self._write_signal_to_db(sig, now_utc, rejection)

            if self.mode == "LIVE" and self.pmocata_url:
                self._post_to_pmocata(sig)
            elif self.mode == "LIVE":
                log.info(f"ML SIGNAL [{sig.direction}] {sig.symbol} @ {sig.entry_price} (no pmocata URL)")
            else:
                log.info(f"ML SIGNAL (VIRTUAL) [{sig.direction}] {sig.symbol} @ {sig.entry_price}")
            log.info(f"ML SIGNAL [{sig.direction}] {sig.symbol} @ {sig.entry_price} "
                     f"prob={sig.prob:.4f} "
                     f"roc_3={sig.roc_at_entry:.2f} "
                     f"delta_lakhs={sig.delta_value_lakhs:.2f} "
                     f"latency={latency_s:.0f}s "
                     f"theta={sig.theta_caught_time} "
                     f"mark1={sig.mark1_time} "
                     f"between={sig.between_time} "
                     f"daily_open={sig.daily_open_passed} "
                     f"bypass={sig.bypass_used}")

        except Exception as e:
            log.warning(f"ML signal handler error: {e}")

    def _write_signal_to_db(self, sig, now_utc, rejection):
        try:
            import psycopg2
            host = os.getenv("QUESTDB_HOST", "localhost")
            port = int(os.getenv("QUESTDB_PORT", "8812"))
            user = os.getenv("QUESTDB_USER", "admin")
            pwd = os.getenv("QUESTDB_PASS", "quest")
            db = os.getenv("QUESTDB_DB", "qdb")
            conn = psycopg2.connect(host=host, port=port, user=user, password=pwd,
                                    database=db, connect_timeout=5)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ml_signals (
                    ts, symbol, direction, entry_price,
                    prob_long, prob_short, prob_no_trade, prob,
                    roc_at_entry, delta_value_lakhs,
                    theta_caught_time, mark1_time, between_time,
                    entry_time_ist, theta_prob,
                    daily_open_passed, bypass_used, rejection_reason, mode
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                now_utc.replace(tzinfo=None), sig.symbol, sig.direction, sig.entry_price,
                sig.prob_long, sig.prob_short, sig.prob_no_trade, sig.prob,
                sig.roc_at_entry, sig.delta_value_lakhs,
                sig.theta_caught_time, sig.mark1_time, sig.between_time,
                sig.entry_time_ist, sig.theta_prob,
                sig.daily_open_passed, sig.bypass_used, rejection, self.mode,
            ))
            cur.close()
            conn.close()
        except Exception as e:
            log.warning(f"ML db write error: {e}")

    def _post_to_pmocata(self, sig):
        if not self.pmocata_url:
            return
        try:
            sl_price = round(sig.entry_price * (1 - self.sl_pct / 100.0), 2)
            tp_price = round(sig.entry_price * (1 + self.tp_pct / 100.0), 2)

            payload = {
                "action": "ENTRY",
                "signal": sig.direction,
                "tradingSymbol": sig.symbol,
                "exchangeSegment": "NSE_EQ",
                "entryPrice": sig.entry_price,
                "mode": self.mode,
                "slPrice": sl_price if sig.direction == "LONG" else round(sig.entry_price * (1 + self.sl_pct / 100.0), 2),
                "tpPrice": tp_price if sig.direction == "LONG" else round(sig.entry_price * (1 - self.tp_pct / 100.0), 2),
                "useSuperOrder": self.use_super_order,
                "orderFlowType": self.order_flow_type,
                "limitPct": self.limit_pct,
            }

            if self.fixed_qty > 0:
                payload["quantity"] = self.fixed_qty
                payload["serverComputeQty"] = False

            import requests
            t0 = time.time()
            resp = requests.post(self.pmocata_url.rstrip("/") + "/webhook",
                                 json=payload, timeout=10)
            t1 = time.time()
            log.info(f"ML POST to pmocata [{sig.symbol}] status={resp.status_code} in {(t1-t0)*1000:.0f}ms")
        except Exception as e:
            log.warning(f"ML pmocata POST error [{sig.symbol}]: {e}")

    def _get_daily_open(self, symbol: str, current_close: float) -> float:
        if symbol not in self._daily_open:
            self._daily_open[symbol] = current_close
        return self._daily_open[symbol]

    def state_summary(self) -> dict:
        total = len(self.accumulators)
        armed = sum(1 for sm in self.machines_long.values() if sm.state == "ARMING")
        return {
            "model_loaded": self.model_loaded,
            "mode": self.mode,
            "symbols_active": total,
            "long_armed": armed,
        }
