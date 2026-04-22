import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dateutil import tz

from config import CONFIG

from runtime.runtime_config import load_runtime_config
from runtime.alert_engine import emit_alert
from engine.pipeline import TradingEngine

from database.db import DB
from data.data_store import DataStore, Candle
from data.bootstrap import fetch_klines
from data.binance_ws import run_ws
from data.orderbook_store import OrderBookStore

from features.market_state import volatility_regime
from features.indicators import rr
from features.quantum_state import build_quantum_state

from signal_engine.bias_detector import detect_bias_h1, detect_bias_h4, combined_bias
from signal_engine.setups import trend_pullback, sweep_reclaim, breakout_confirmation
from signal_engine.scoring import compute_score
from signal_engine.ticket import build_ticket
from signal_engine.confluence import confluence_points
from signal_engine.explainer import explain_no_setup
from signal_engine.diagnostics import run_diagnostics
from signal_engine.context_classifier import classify_market_context
from signal_engine.action_label import suggest_action
from signal_engine.liquidity_context import squeeze_risk_label_from_prices
from signal_engine.derivatives_score import derivatives_points
from signal_engine.news_score import news_points
from signal_engine.quantum_score import quantum_points
from signal_engine.strategy_profile import get_strategy_profile, strategy_points

from ai.explanation_engine import generate_explanation
from liquidity.liquidation_engine import fetch_liquidation_map
from market_data.derivatives_context import build_derivatives_context
from market_data.news_context import build_news_context

from snapshot.chart_renderer import save_snapshot
from risk.risk_governor import RiskGovernor

ROME_TZ = tz.gettz("Europe/Rome")
ORDERBOOK_PERSIST_INTERVAL_SEC = 2.0
EVENT_BUS_MAXSIZE = 2048


def ensure_dirs():
    os.makedirs("out", exist_ok=True)
    os.makedirs(CONFIG.snapshot_dir, exist_ok=True)
    os.makedirs(CONFIG.ticket_dir, exist_ok=True)


def bootstrap_history(store: DataStore, db: DB):
    print("[BOOTSTRAP] Downloading historical candles...")

    plan = [
        ("15m", "15m", 500),
        ("1h", "1h", 500),
        ("4h", "4h", 300),
    ]

    for interval, tf, lim in plan:
        klines = fetch_klines(CONFIG.symbol, interval=interval, limit=lim)
        for k in klines:
            c = Candle.from_binance_kline(k)
            store.add_candle(tf, c)
            db.upsert_candle(
                CONFIG.symbol.upper(),
                tf,
                {
                    "open_time": c.open_time,
                    "close_time": c.close_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                },
            )

        print(f"[BOOTSTRAP] Loaded {len(klines)} candles for {interval}")

    print("[BOOTSTRAP] Done.")


def has_min_history(df_m15, df_h1, df_h4):
    return len(df_m15) >= 150 and len(df_h1) >= 200 and len(df_h4) >= 60


async def main():
    ensure_dirs()

    db = DB(CONFIG.db_path, "database/schema.sql")

    store = DataStore(
        CONFIG.max_candles_m15,
        CONFIG.max_candles_h1,
        CONFIG.max_candles_h4,
    )

    ob = OrderBookStore()
    
    bootstrap_history(store, db)
    COMMANDS_FILE = Path("out/commands.txt")
    runtime_cfg = load_runtime_config()
    
    risk = RiskGovernor(
        runtime_cfg["max_signals_per_day"],
        runtime_cfg["cooldown_minutes"],
    )

    def sync_risk_governor(cfg):
        risk.sync_limits(
            cfg["max_signals_per_day"],
            cfg["cooldown_minutes"],
        )
    
    trading_engine = TradingEngine(
        symbol=CONFIG.symbol,
        store=store,
        orderbook_store=ob,
        db=db,
        risk_governor=risk,
        runtime_config_loader=load_runtime_config,
        snapshot_dir=CONFIG.snapshot_dir,
        ticket_dir=CONFIG.ticket_dir,
        timezone=ROME_TZ,
    )

    event_bus: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=EVENT_BUS_MAXSIZE)
    dropped_events = {"count": 0}
    last_orderbook_persist_ts = {"mono": 0.0}

    def publish_event(evt: Dict[str, Any]) -> None:
        try:
            event_bus.put_nowait(evt)
        except asyncio.QueueFull:
            dropped_events["count"] += 1

    async def event_bus_worker() -> None:
        while True:
            evt = await event_bus.get()
            try:
                evt_type = str(evt.get("type", ""))
                payload = evt.get("payload", {})
                if evt_type == "orderbook_snapshot":
                    db.insert_orderbook_snapshot(payload)
                elif evt_type == "decision_log":
                    db.insert_decision_log(payload)
            except Exception as ex:
                print(f"[EVENT_BUS] error: {ex}")
            finally:
                event_bus.task_done()


    async def on_depth(bids, asks):
        ob.update_from_binance_depth(bids, asks)
        now_mono = time.monotonic()
        if now_mono - last_orderbook_persist_ts["mono"] < ORDERBOOK_PERSIST_INTERVAL_SEC:
            return

        last_orderbook_persist_ts["mono"] = now_mono
        stats = ob.snapshot_stats(top_n=10)
        publish_event(
            {
                "type": "orderbook_snapshot",
                "payload": {
                    "timestamp": datetime.now(ROME_TZ).isoformat(),
                    "symbol": CONFIG.symbol.upper(),
                    "imbalance_avg": stats.get("imbalance_avg"),
                    "imbalance_raw": stats.get("imbalance_raw"),
                    "age_ms": stats.get("age_ms"),
                    "bid_notional": stats.get("bid_notional"),
                    "ask_notional": stats.get("ask_notional"),
                    "top_bid": stats.get("top_bid"),
                    "top_ask": stats.get("top_ask"),
                    "spread": stats.get("spread"),
                    "source": "depth20@100ms",
                },
            }
        )

    def orderbook_points(decision: str, imb: float, runtime_cfg) -> int:
        neutral_th = float(runtime_cfg["orderbook_neutral_threshold"])
        full_th = float(runtime_cfg["orderbook_full_score_threshold"])

        if abs(imb) < neutral_th:
            return 0

        span = max(0.0001, full_th - neutral_th)
        strength = min(1.0, (abs(imb) - neutral_th) / span)
        pts = int(round(10 * strength))

        if decision == "BUY":
            return pts if imb > 0 else -pts

        if decision == "SELL":
            return pts if imb < 0 else -pts

        return 0
    
    def select_setup_candidate(df_m15, df_h1, df_h4, combined_bias: str, strategy_mode: str):
        profile = get_strategy_profile(strategy_mode)
        setup_order = list(profile.setup_priority)

        if combined_bias not in ("bullish", "bearish"):
            return sweep_reclaim(df_m15, "neutral")

        for setup_name in setup_order:
            if setup_name == "TREND_PULLBACK":
                res = trend_pullback(df_m15, df_h1, combined_bias)
            elif setup_name == "SWEEP_RECLAIM":
                res = sweep_reclaim(df_m15, combined_bias)
            elif setup_name == "BREAKOUT_CONFIRMATION":
                res = breakout_confirmation(df_m15, df_h1, combined_bias)
            else:
                res = None

            if res is not None:
                return res

        return None

    def quick_setup_status(df_m15, df_h1, df_h4, rr_min, strategy_mode):
        if not has_min_history(df_m15, df_h1, df_h4):
            return "warming_up", "not enough history"

        b_h1 = detect_bias_h1(df_h1)
        b_h4 = detect_bias_h4(df_h4)
        b_comb = combined_bias(b_h1, b_h4)

        setup_res = select_setup_candidate(
            df_m15,
            df_h1,
            df_h4,
            b_comb,
            strategy_mode,
        )

        if setup_res is None:
            why = explain_no_setup(df_m15, df_h1, df_h4)
            return "no_setup", why

        rr_est = rr(setup_res.entry, setup_res.sl, setup_res.tp1)

        if rr_est < rr_min:
            return (
                "weak_candidate",
                f"{setup_res.setup} rejected early: rr too low ({rr_est:.2f} < min {rr_min:.2f})",
            )

        return f"{setup_res.setup} rr={rr_est:.2f} dir={setup_res.decision}", "candidate found"

    def save_status_event(
        event_type: str,
        decision: str,
        setup: str,
        context: str,
        action: str,
        why: str,
        entry,
        sl,
        tp1,
        tp2,
        rr_estimated,
        score,
        ob_imbalance=None,
        ob_raw=None,
        ob_age_ms=None,
        funding_rate=None,
        oi_now=None,
        oi_change_pct=None,
        crowding=None,
        strategy_mode=None,
        strategy_score=None,
        news_bias=None,
        news_sentiment=None,
        news_impact=None,
        news_score=None,
        quantum_state=None,
        quantum_coherence=None,
        quantum_phase_bias=None,
        quantum_interference=None,
        quantum_tunneling=None,
        quantum_energy=None,
        quantum_decoherence_rate=None,
        quantum_transition_rate=None,
        quantum_dominant_mode=None,
        quantum_score=None,
        ticket_path=None,
        snapshot_path=None,
    ):
        last_evt = db.get_last_event()

        if last_evt is not None:
            if (
                last_evt.get("event_type") == event_type
                and last_evt.get("decision") == decision
                and last_evt.get("setup") == setup
                and last_evt.get("context") == context
                and last_evt.get("action") == action
                and last_evt.get("why") == why
            ):
                return

        now = datetime.now(ROME_TZ)
        signal_id = f"evt_{event_type}_{now.strftime('%Y%m%d_%H%M%S_%f')}_{CONFIG.symbol.upper()}"

        db.insert_signal(
            {
                "signal_id": signal_id,
                "timestamp": now.isoformat(),
                "symbol": CONFIG.symbol.upper(),
                "event_type": event_type,
                "decision": decision,
                "setup": setup,
                "context": context,
                "action": action,
                "why": why,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr_estimated": rr_estimated,
                "score": score,
                "ob_imbalance": ob_imbalance,
                "ob_raw": ob_raw,
                "ob_age_ms": ob_age_ms,
                "funding_rate": funding_rate,
                "oi_now": oi_now,
                "oi_change_pct": oi_change_pct,
                "crowding": crowding,
                "strategy_mode": strategy_mode,
                "strategy_score": strategy_score,
                "news_bias": news_bias,
                "news_sentiment": news_sentiment,
                "news_impact": news_impact,
                "news_score": news_score,
                "quantum_state": quantum_state,
                "quantum_coherence": quantum_coherence,
                "quantum_phase_bias": quantum_phase_bias,
                "quantum_interference": quantum_interference,
                "quantum_tunneling": quantum_tunneling,
                "quantum_energy": quantum_energy,
                "quantum_decoherence_rate": quantum_decoherence_rate,
                "quantum_transition_rate": quantum_transition_rate,
                "quantum_dominant_mode": quantum_dominant_mode,
                "quantum_score": quantum_score,
                "snapshot_path": snapshot_path,
                "ticket_path": ticket_path,
            }
        )

    def maybe_emit_news_regime_alert(
        *,
        now,
        runtime_cfg,
        news,
        context,
        action,
        strategy_code,
        trigger,
    ) -> None:
        if not bool(runtime_cfg.get("alerts_enabled", True)):
            return

        impact_threshold = float(runtime_cfg.get("news_alert_min_impact", 0.70))
        if float(news.impact_score) < impact_threshold:
            return

        if str(news.bias) not in ("bullish", "bearish", "high_alert"):
            return

        severity = "critical" if float(news.impact_score) >= 0.85 else "warning"
        emit_alert(
            alert_type="news_regime",
            title=f"High-impact news regime ({news.bias})",
            body=(
                f"Topic={news.dominant_topic} | impact={news.impact_score:.2f} "
                f"| sentiment={news.sentiment_score:.2f} | ctx={context} | action={action} "
                f"| strategy={strategy_code} | trigger={trigger}"
            ),
            severity=severity,
            created_at=now,
            dedup_key=f"news:{CONFIG.symbol.upper()}:{news.bias}:{news.dominant_topic}",
            cooldown_minutes=int(runtime_cfg.get("alert_cooldown_minutes", 30)),
            metadata={
                "symbol": CONFIG.symbol.upper(),
                "news_bias": news.bias,
                "news_impact": news.impact_score,
                "news_sentiment": news.sentiment_score,
                "dominant_topic": news.dominant_topic,
                "headline_count": news.headline_count,
                "strategy_mode": strategy_code,
                "context": context,
                "action": action,
                "trigger": trigger,
            },
        )

    def maybe_emit_trade_alerts(
        *,
        now,
        runtime_cfg,
        trigger,
        ticket_id,
        ticket_path,
        decision,
        setup_name,
        action,
        score,
        rr_est,
        latest_price,
        strategy_profile,
        strategy_pts,
        news,
        news_pts,
        quantum,
        quant_pts,
        context,
        squeeze_risk,
        risk_status,
        why_text,
    ) -> None:
        if not bool(runtime_cfg.get("alerts_enabled", True)):
            return

        cooldown_minutes = int(runtime_cfg.get("alert_cooldown_minutes", 30))
        rr_text = "N/A" if rr_est is None else f"{rr_est:.2f}"
        base_metadata = {
            "symbol": CONFIG.symbol.upper(),
            "decision": decision,
            "setup": setup_name,
            "action": action,
            "score": score,
            "rr_estimated": rr_est,
            "price": latest_price,
            "strategy_mode": strategy_profile.code,
            "strategy_score": strategy_pts,
            "news_bias": news.bias,
            "news_impact": news.impact_score,
            "news_score": news_pts,
            "quantum_state": quantum.state,
            "quantum_coherence": quantum.coherence,
            "quantum_score": quant_pts,
            "context": context,
            "squeeze_risk": squeeze_risk,
            "risk_can_emit": risk_status.get("can_emit"),
            "risk_block_reason": risk_status.get("block_reason"),
            "trigger": trigger,
            "ticket_path": ticket_path,
            "why": why_text,
        }

        if decision in ("BUY", "SELL") and int(score) >= int(runtime_cfg.get("alert_min_score", 74)):
            severity = "critical" if int(score) >= 85 else "info"
            emit_alert(
                alert_type="signal_live",
                title=f"{decision} signal live ({setup_name})",
                body=(
                    f"score={score} | rr={rr_text}"
                    f" | strategy={strategy_profile.code} | news={news.bias}/{news.impact_score:.2f}"
                    f" | quantum={quantum.state}/{quantum.coherence:.2f}"
                    f" | action={action}"
                ),
                severity=severity,
                created_at=now,
                dedup_key=f"signal:{CONFIG.symbol.upper()}:{decision}:{setup_name}:{strategy_profile.code}",
                cooldown_minutes=cooldown_minutes,
                signal_id=ticket_id,
                metadata=base_metadata,
            )

        elif setup_name == "BLOCKED" and int(score) >= int(runtime_cfg.get("blocked_alert_min_score", 68)):
            emit_alert(
                alert_type="blocked_candidate",
                title="Strong candidate blocked",
                body=(
                    f"setup={setup_name} | score={score} | rr={rr_text} "
                    f"| block_reason={risk_status.get('block_reason')} | strategy={strategy_profile.code} "
                    f"| action={action}"
                ),
                severity="warning",
                created_at=now,
                dedup_key=f"blocked:{CONFIG.symbol.upper()}:{setup_name}:{strategy_profile.code}",
                cooldown_minutes=cooldown_minutes,
                signal_id=ticket_id,
                metadata=base_metadata,
            )

    async def analyze_and_emit(trigger: str):
        result = await trading_engine.run_cycle(trigger)
        publish_event(
            {
                "type": "decision_log",
                "payload": {
                    "timestamp": datetime.now(ROME_TZ).isoformat(),
                    "symbol": CONFIG.symbol.upper(),
                    "trigger": trigger,
                    "event_type": result.event_type,
                    "decision": result.decision,
                    "setup": result.setup,
                    "context": result.context,
                    "action": result.action,
                    "score": float(result.score),
                    "ticket_path": result.ticket_path,
                },
            }
        )
        print("=" * 90)
        print(
            f"[PIPELINE:{trigger}] decision={result.decision} "
            f"setup={result.setup} score={result.score} action={result.action} "
            f"event_type={result.event_type}"
        )
        if result.ticket_path:
            print(f"Ticket:   {result.ticket_path}")

    async def on_closed_kline(tf: str, k: dict):
        c = Candle.from_binance_kline(k)

        store.add_candle(tf, c)

        db.upsert_candle(
            CONFIG.symbol.upper(),
            tf,
            {
                "open_time": c.open_time,
                "close_time": c.close_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            },
        )

        if tf == "15m":
            await analyze_and_emit("M15_CLOSE")

    async def process_command(cmd: str):
        cmd = cmd.strip().lower()

        if cmd == "a":
            await analyze_and_emit("MANUAL")

        elif cmd == "s":
            runtime_cfg = load_runtime_config()
            sync_risk_governor(runtime_cfg)
            strategy_profile = get_strategy_profile(str(runtime_cfg.get("strategy_mode", "BALANCED")))
            df_m15 = store.to_df("15m")
            df_h1 = store.to_df("1h")
            df_h4 = store.to_df("4h")

            if len(df_m15) == 0:
                print("[STATUS] no data")
                return

            vol = volatility_regime(df_m15)
            b_h1 = detect_bias_h1(df_h1) if len(df_h1) > 50 else "neutral"
            b_h4 = detect_bias_h4(df_h4) if len(df_h4) > 50 else "neutral"
            b_comb = combined_bias(b_h1, b_h4)
            ctx = classify_market_context(df_m15, df_h1, df_h4)
            quantum = build_quantum_state(df_m15, df_h1, df_h4, runtime_cfg=runtime_cfg)

            last_close = float(df_m15["close"].iloc[-1])

            imb = ob.imbalance(top_n=10)
            raw_imb = ob.raw_imbalance(top_n=10)
            age = ob.age_ms()

            if imb is None:
                ob_txt = "OB_imbalance=N/A"
            else:
                ob_txt = f"OB_imbalance_avg={imb:.3f} raw={raw_imb:.3f} age_ms={age}"

            liq_data = fetch_liquidation_map("BTC")
            top_liq = None
            if liq_data and liq_data.get("clusters"):
                top_liq = liq_data["clusters"][0]["price"]

            squeeze_risk = squeeze_risk_label_from_prices(
                last_close,
                top_liq,
                high_pct=float(runtime_cfg["squeeze_risk_high_pct"]),
                medium_pct=float(runtime_cfg["squeeze_risk_medium_pct"]),
            )

            deriv = build_derivatives_context(CONFIG.symbol)
            news = build_news_context(
                symbol=CONFIG.symbol,
                limit=int(runtime_cfg.get("news_headline_limit", 6)),
                cache_minutes=int(runtime_cfg.get("news_cache_minutes", 15)),
                enabled=bool(runtime_cfg.get("news_enabled", True)),
            )
            risk_status = risk.status_snapshot(datetime.now(ROME_TZ))

            setup_info, setup_why = trading_engine.quick_setup_status(
                df_m15,
                df_h1,
                df_h4,
                float(runtime_cfg["rr_min"]),
                strategy_profile.code,
            )
            action = suggest_action(
                ctx,
                setup_info,
                setup_why,
                b_comb,
                squeeze_risk=squeeze_risk
            )

            print(
                f"[STATUS] price={last_close:.2f} vol={vol} "
                f"H1={b_h1} H4={b_h4} combined={b_comb} ctx={ctx} | {ob_txt} | "
                f"setup={setup_info} | strategy={strategy_profile.code} | action={action} | squeeze_risk={squeeze_risk} | "
                f"news_bias={news.bias} news_sentiment={news.sentiment_score} news_impact={news.impact_score} | "
                f"quantum_state={quantum.state} quantum_coherence={quantum.coherence} "
                f"quantum_phase={quantum.phase_bias} quantum_tunneling={quantum.tunneling_probability} "
                f"quantum_energy={quantum.energy} quantum_decoherence_rate={quantum.decoherence_rate} "
                f"quantum_transition_rate={quantum.transition_rate} dominant_mode={quantum.dominant_mode} | "
                f"risk_can_emit={risk_status['can_emit']} "
                f"risk_reason={risk_status['block_reason']} "
                f"signals_today={risk_status['signals_today']}/{risk_status['max_signals_per_day']} "
                f"cooldown_remaining={risk_status['cooldown_remaining_minutes']} | "
                f"funding={deriv.get('funding_rate')} | oi_change_15m={deriv.get('open_interest_change_pct_15m')} | "
                f"crowding={deriv.get('crowding')} | why={setup_why}"
            )

        elif cmd == "d":
            df_m15 = store.to_df("15m")
            df_h1 = store.to_df("1h")
            df_h4 = store.to_df("4h")

            diag = run_diagnostics(df_m15, df_h1, df_h4)
            ctx = classify_market_context(df_m15, df_h1, df_h4)

            imb = ob.imbalance(top_n=10)
            raw_imb = ob.raw_imbalance(top_n=10)
            age = ob.age_ms()

            deriv = build_derivatives_context(CONFIG.symbol)

            print("=" * 90)
            print("[DIAGNOSTIC]")
            print(f"bias_h1={diag['bias_h1']} bias_h4={diag['bias_h4']} combined={diag['combined_bias']}")
            print(f"context={ctx}")
            print(f"value_zone={diag['value_zone']}")
            print(f"sweep_reclaim={diag['sweep_reclaim']}")
            print(f"breakout_confirmation={diag['breakout_confirmation']}")
            print(f"trend_pullback_rr={diag['trend_pullback_rr']}")
            print(f"sweep_reclaim_rr={diag['sweep_reclaim_rr']}")
            print(f"breakout_rr={diag['breakout_rr']}")
            if imb is None:
                print("orderbook=N/A")
            else:
                print(f"orderbook_avg={imb:.3f} raw={raw_imb:.3f} age_ms={age}")
            print(f"funding_rate={deriv.get('funding_rate')}")
            print(f"oi_now={deriv.get('open_interest_now')}")
            print(f"oi_change_pct_15m={deriv.get('open_interest_change_pct_15m')}")
            print(f"crowding={deriv.get('crowding')}")
            print("notes:")
            for n in diag["notes"]:
                print(f" - {n}")

        elif cmd == "j":
            rows = db.list_recent_signals(limit=10)

            print("=" * 90)
            print("[JOURNAL] Last 10 events")

            if not rows:
                print("No events saved yet.")
                return

            for r in rows:
                print("-" * 90)
                print(
                    f"time={r['timestamp']} | type={r['event_type']} | symbol={r['symbol']} | "
                    f"decision={r['decision']} | setup={r['setup']} | context={r['context']} | "
                    f"action={r['action']} | score={r['score']} | rr={r['rr_estimated']}"
                )
                print(f"entry={r['entry']} sl={r['sl']} tp1={r['tp1']} tp2={r['tp2']}")
                print(
                    f"ob={r.get('ob_imbalance')} raw={r.get('ob_raw')} age={r.get('ob_age_ms')} | "
                    f"funding={r.get('funding_rate')} oi_now={r.get('oi_now')} "
                    f"oi_change={r.get('oi_change_pct')} crowding={r.get('crowding')} | "
                    f"strategy={r.get('strategy_mode')} strategy_score={r.get('strategy_score')} | "
                    f"news_bias={r.get('news_bias')} news_sentiment={r.get('news_sentiment')} "
                    f"news_impact={r.get('news_impact')} news_score={r.get('news_score')} | "
                    f"quantum_state={r.get('quantum_state')} coherence={r.get('quantum_coherence')} "
                    f"phase={r.get('quantum_phase_bias')} tunneling={r.get('quantum_tunneling')} "
                    f"energy={r.get('quantum_energy')} decoherence={r.get('quantum_decoherence_rate')} "
                    f"transition={r.get('quantum_transition_rate')} dominant_mode={r.get('quantum_dominant_mode')} "
                    f"quant_score={r.get('quantum_score')}"
                )
                print(f"why={r['why']}")
                print(f"ticket={r['ticket_path']}")

        elif cmd == "e":
            runtime_cfg = load_runtime_config()
            sync_risk_governor(runtime_cfg)
            strategy_profile = get_strategy_profile(str(runtime_cfg.get("strategy_mode", "BALANCED")))
            df_m15 = store.to_df("15m")
            df_h1 = store.to_df("1h")
            df_h4 = store.to_df("4h")

            if len(df_m15) == 0:
                print("[EXPLAIN] no data")
                return

            vol = volatility_regime(df_m15)
            b_h1 = detect_bias_h1(df_h1) if len(df_h1) > 50 else "neutral"
            b_h4 = detect_bias_h4(df_h4) if len(df_h4) > 50 else "neutral"
            b_comb = combined_bias(b_h1, b_h4)
            ctx = classify_market_context(df_m15, df_h1, df_h4)
            quantum = build_quantum_state(df_m15, df_h1, df_h4, runtime_cfg=runtime_cfg)
            news = build_news_context(
                symbol=CONFIG.symbol,
                limit=int(runtime_cfg.get("news_headline_limit", 6)),
                cache_minutes=int(runtime_cfg.get("news_cache_minutes", 15)),
                enabled=bool(runtime_cfg.get("news_enabled", True)),
            )

            last_close = float(df_m15["close"].iloc[-1])

            imb = ob.imbalance(top_n=10)
            raw_imb = ob.raw_imbalance(top_n=10)

            liq_data = fetch_liquidation_map("BTC")
            top_liq = None
            if liq_data and liq_data.get("clusters"):
                top_liq = liq_data["clusters"][0]["price"]

            squeeze_risk = squeeze_risk_label_from_prices(
                last_close,
                top_liq,
                high_pct=float(runtime_cfg["squeeze_risk_high_pct"]),
                medium_pct=float(runtime_cfg["squeeze_risk_medium_pct"]),
            )

            setup_info, setup_why = trading_engine.quick_setup_status(
                df_m15,
                df_h1,
                df_h4,
                float(runtime_cfg["rr_min"]),
                strategy_profile.code,
            )
            action = suggest_action(
                ctx,
                setup_info,
                setup_why,
                b_comb,
                squeeze_risk=squeeze_risk
            )

            explanation = generate_explanation(
                price=last_close,
                bias_h1=b_h1,
                bias_h4=b_h4,
                combined_bias=b_comb,
                context=ctx,
                volatility=vol,
                setup_info=setup_info,
                action=action,
                why=(
                    f"{setup_why}; strategy_mode={strategy_profile.code}; squeeze_risk={squeeze_risk}; "
                    f"news_bias={news.bias}; "
                    f"news_sentiment={news.sentiment_score:.2f}; "
                    f"news_impact={news.impact_score:.2f}; "
                    f"quantum_state={quantum.state}; "
                    f"quantum_coherence={quantum.coherence:.2f}; "
                    f"quantum_phase_bias={quantum.phase_bias:.2f}; "
                    f"quantum_tunneling={quantum.tunneling_probability:.2f}; "
                    f"quantum_energy={quantum.energy:.2f}; "
                    f"quantum_decoherence_rate={quantum.decoherence_rate:.2f}; "
                    f"quantum_transition_rate={quantum.transition_rate:.2f}; "
                    f"quantum_dominant_mode={quantum.dominant_mode}"
                ),
                ob_avg=imb,
                ob_raw=raw_imb,
                quantum_state=quantum.state,
                quantum_coherence=quantum.coherence,
                quantum_phase_bias=quantum.phase_bias,
                quantum_tunneling=quantum.tunneling_probability,
                quantum_energy=quantum.energy,
                quantum_decoherence_rate=quantum.decoherence_rate,
                quantum_transition_rate=quantum.transition_rate,
                quantum_dominant_mode=quantum.dominant_mode,
            )

            print("=" * 90)
            print("[EXPLANATION]")
            print(explanation)

        elif cmd == "q":
            print("[CMD] quitting...")
            os._exit(0)

        else:
            print("[CMD] unknown command. Use 'a', 's', 'd', 'j', 'e' or 'q'.")

    async def terminal_command_listener():
        print("Commands: [a] analyze | [s] status | [d] diagnostic | [j] journal | [e] explain | [q] quit")
        while True:
            cmd = await asyncio.to_thread(input, "> ")
            cmd = cmd.strip().lower()
            if cmd:
                await process_command(cmd)

    async def file_command_listener():
        while True:
            try:
                if COMMANDS_FILE.exists():
                    lines = COMMANDS_FILE.read_text(encoding="utf-8").splitlines()
                    if lines:
                        COMMANDS_FILE.write_text("", encoding="utf-8")
                        for cmd in lines:
                            cmd = cmd.strip().lower()
                            if cmd:
                                await process_command(cmd)
            except Exception as ex:
                print(f"[COMMAND_FILE] error: {ex}")

            await asyncio.sleep(1)

    asyncio.create_task(event_bus_worker())
    asyncio.create_task(terminal_command_listener())
    asyncio.create_task(file_command_listener())

    await run_ws(CONFIG.symbol, on_closed_kline, on_depth=on_depth)


if __name__ == "__main__":
    asyncio.run(main())

