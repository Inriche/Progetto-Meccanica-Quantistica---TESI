import asyncio
import os
from datetime import datetime
from pathlib import Path

from dateutil import tz

from config import CONFIG

from runtime.runtime_config import load_runtime_config

from database.db import DB
from data.data_store import DataStore, Candle
from data.bootstrap import fetch_klines
from data.binance_ws import run_ws
from data.orderbook_store import OrderBookStore

from features.market_state import volatility_regime
from features.indicators import rr

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

from ai.explanation_engine import generate_explanation
from liquidity.liquidation_engine import fetch_liquidation_map
from market_data.derivatives_context import build_derivatives_context

from snapshot.chart_renderer import save_snapshot
from risk.risk_governor import RiskGovernor

ROME_TZ = tz.gettz("Europe/Rome")


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
    


    async def on_depth(bids, asks):
        ob.update_from_binance_depth(bids, asks)

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
    
    def quick_setup_status(df_m15, df_h1, df_h4):
        if not has_min_history(df_m15, df_h1, df_h4):
            return "warming_up", "not enough history"

        b_h1 = detect_bias_h1(df_h1)
        b_h4 = detect_bias_h4(df_h4)
        b_comb = combined_bias(b_h1, b_h4)

        if b_comb in ("bullish", "bearish"):
            setup_res = (
                trend_pullback(df_m15, df_h1, b_comb)
                or sweep_reclaim(df_m15, b_comb)
                or breakout_confirmation(df_m15, df_h1, b_comb)
            )
        else:
            setup_res = sweep_reclaim(df_m15, "neutral")

        if setup_res is None:
            why = explain_no_setup(df_m15, df_h1, df_h4)
            return "no_setup", why

        rr_est = rr(setup_res.entry, setup_res.sl, setup_res.tp1)

        if rr_est < 1.0:
            return "weak_candidate", f"{setup_res.setup} rejected early: rr too low ({rr_est:.2f})"

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
                "snapshot_path": snapshot_path,
                "ticket_path": ticket_path,
            }
        )

    async def analyze_and_emit(trigger: str):
        runtime_cfg = load_runtime_config()

        df_m15 = store.to_df("15m")
        df_h1 = store.to_df("1h")
        df_h4 = store.to_df("4h")

        if not has_min_history(df_m15, df_h1, df_h4):
            print(
                f"[ANALYZE:{trigger}] warming up... "
                f"(m15={len(df_m15)} h1={len(df_h1)} h4={len(df_h4)})"
            )
            return

        b_h1 = detect_bias_h1(df_h1)
        b_h4 = detect_bias_h4(df_h4)
        b_comb = combined_bias(b_h1, b_h4)
        vol = volatility_regime(df_m15)
        ctx = classify_market_context(df_m15, df_h1, df_h4)

        liq_data = fetch_liquidation_map("BTC")
        top_liq = None
        if liq_data and liq_data.get("clusters"):
            top_liq = liq_data["clusters"][0]["price"]

        latest_price = float(df_m15["close"].iloc[-1]) if not df_m15.empty else None
        squeeze_risk = squeeze_risk_label_from_prices(
            latest_price,
            top_liq,
            high_pct=float(runtime_cfg["squeeze_risk_high_pct"]),
            medium_pct=float(runtime_cfg["squeeze_risk_medium_pct"]),
        )

        deriv = build_derivatives_context(CONFIG.symbol)
        funding_rate = deriv.get("funding_rate")
        oi_now = deriv.get("open_interest_now")
        oi_change_pct = deriv.get("open_interest_change_pct_15m")
        crowding = deriv.get("crowding")

        now = datetime.now(ROME_TZ)

        setup_res = None
        if b_comb in ("bullish", "bearish"):
            setup_res = (
                trend_pullback(df_m15, df_h1, b_comb)
                or sweep_reclaim(df_m15, b_comb)
                or breakout_confirmation(df_m15, df_h1, b_comb)
            )
        else:
            setup_res = sweep_reclaim(df_m15, "neutral")

        decision = "FLAT"
        entry = sl = tp1 = tp2 = None
        setup_name = "NONE"
        reasons = ["No valid setup passed filters."]
        snapshot_path = ""

        score = 0
        grade = "C"
        components = []

        imb = ob.imbalance(top_n=10)
        raw_imb = ob.raw_imbalance(top_n=10)
        imb_age = ob.age_ms()

        rr_est = None
        base_score = 0
        base_components = []
        ob_pts = 0
        conf_pts = 0
        deriv_pts = 0

        if setup_res is not None:
            entry = setup_res.entry
            sl = setup_res.sl
            tp1 = setup_res.tp1
            tp2 = setup_res.tp2

            rr_est = rr(entry, sl, tp1)

            breakdown = compute_score(
                bias_h1=b_h1,
                bias_h4=b_h4,
                combined=b_comb,
                rr_est=rr_est,
                volatility=vol,
                setup_name=setup_res.setup,
                context=ctx,
            )

            base_score = breakdown.score
            base_components = breakdown.components

            if imb is not None and imb_age is not None and imb_age <= 2000:
                ob_pts = orderbook_points(setup_res.decision, imb, runtime_cfg)

            conf_pts = confluence_points(entry, setup_res.decision, df_h1)

            deriv_pts = derivatives_points(
                funding_rate=funding_rate,
                oi_change_pct=oi_change_pct,
                crowding=crowding,
                decision=setup_res.decision,
                extreme_oi_pct=float(runtime_cfg["derivatives_extreme_oi_pct"]),
                mild_oi_pct=float(runtime_cfg["derivatives_mild_oi_pct"]),
            )

            score = max(0, min(100, base_score + ob_pts + conf_pts + deriv_pts))
            grade = "A" if score >= 80 else ("B" if score >= 70 else "C")
            components = base_components + [
                ("orderbook_imbalance", ob_pts),
                ("h1_confluence", conf_pts),
                ("derivatives_context", deriv_pts),
            ]

            passes_rr = rr_est >= runtime_cfg["rr_min"]
            passes_score = score >= runtime_cfg["min_score_for_signal"]
            passes_risk = risk.can_emit(now)

            if passes_rr and passes_score and passes_risk:
                decision = setup_res.decision
                setup_name = setup_res.setup

                reasons = setup_res.reasons.copy()

                if imb is None:
                    reasons.append("OrderBook: N/A")
                else:
                    reasons.append(
                        f"OrderBook imbalance_avg={imb:.3f} raw={raw_imb:.3f} age_ms={imb_age}"
                    )

                reasons.append(f"H1 confluence points={conf_pts}")
                reasons.append(f"derivatives points={deriv_pts}")
                reasons.append(f"context={ctx}")
                reasons.append(f"squeeze_risk={squeeze_risk}")

                if top_liq is not None:
                    reasons.append(f"nearest_liquidation_cluster={top_liq}")

                if funding_rate is not None:
                    reasons.append(f"funding_rate={funding_rate}")

                if oi_now is not None:
                    reasons.append(f"oi_now={oi_now}")

                if oi_change_pct is not None:
                    reasons.append(f"oi_change_pct_15m={oi_change_pct}")

                if crowding is not None:
                    reasons.append(f"crowding={crowding}")

            else:
                setup_name = "BLOCKED"
                decision = "FLAT"

                reasons = [
                    f"Setup found ({setup_res.setup}) but blocked by filters.",
                    f"RR_est={rr_est:.2f} (min {runtime_cfg['rr_min']})",
                    (
                        f"score={score} (min {runtime_cfg['min_score_for_signal']}) "
                        f"base={base_score} ob_pts={ob_pts} conf_pts={conf_pts} deriv_pts={deriv_pts}"
                    ),
                    f"vol={vol}, combined_bias={b_comb}, context={ctx}",
                    f"Risk: can_emit={passes_risk}. Trigger={trigger}",
                ]

                if imb is None:
                    reasons.append("OrderBook: N/A")
                else:
                    reasons.append(
                        f"OrderBook imbalance_avg={imb:.3f} raw={raw_imb:.3f} age_ms={imb_age}"
                    )

                reasons.append(f"squeeze_risk={squeeze_risk}")

                if top_liq is not None:
                    reasons.append(f"nearest_liquidation_cluster={top_liq}")

                if funding_rate is not None:
                    reasons.append(f"funding_rate={funding_rate}")

                if oi_now is not None:
                    reasons.append(f"oi_now={oi_now}")

                if oi_change_pct is not None:
                    reasons.append(f"oi_change_pct_15m={oi_change_pct}")

                if crowding is not None:
                    reasons.append(f"crowding={crowding}")

        if decision == "FLAT" and setup_name == "NONE":
            setup_info, setup_why = quick_setup_status(df_m15, df_h1, df_h4)
            action = suggest_action(ctx, setup_info, setup_why, b_comb, squeeze_risk=squeeze_risk)
            why_text = f"{setup_why}; squeeze_risk={squeeze_risk}"

            save_status_event(
                event_type="status",
                decision="FLAT",
                setup="NONE",
                context=ctx,
                action=action,
                why=why_text,
                entry=None,
                sl=None,
                tp1=None,
                tp2=None,
                rr_estimated=None,
                score=0,
                ob_imbalance=imb,
                ob_raw=raw_imb,
                ob_age_ms=imb_age,
                funding_rate=funding_rate,
                oi_now=oi_now,
                oi_change_pct=oi_change_pct,
                crowding=crowding,
            )

            print(f"[ANALYZE:{trigger}] FLAT (no setup)")
            return

        action = suggest_action(
            ctx,
            setup_name,
            "; ".join(reasons),
            b_comb,
            squeeze_risk=squeeze_risk
        )

        if not any(str(r).startswith("squeeze_risk=") for r in reasons):
            reasons.append(f"squeeze_risk={squeeze_risk}")

        reasons.append(f"action={action}")

        if decision != "FLAT":
            risk.mark_emitted(now)

            snapshot_path = save_snapshot(
                df_m15,
                CONFIG.snapshot_dir,
                CONFIG.symbol,
                now,
                entry,
                sl,
                tp1,
                tp2,
            )

        why_text = "; ".join(reasons)

        event_snapshot = {
            "price": latest_price,
            "context": ctx,
            "action": action,
            "squeeze_risk": squeeze_risk,
            "orderbook": {
                "imbalance_avg": imb,
                "raw": raw_imb,
                "age_ms": imb_age,
            },
            "derivatives": {
                "funding_rate": funding_rate,
                "oi_now": oi_now,
                "oi_change_pct_15m": oi_change_pct,
                "crowding": crowding,
            },
            "liquidity": {
                "nearest_liquidation_cluster": top_liq,
            },
        }

        ticket = build_ticket(
            symbol=CONFIG.symbol,
            timestamp=now,
            decision=decision,
            setup=setup_name,
            bias_h1=b_h1,
            bias_h4=b_h4,
            combined_bias=b_comb,
            volatility=vol,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            score=score,
            grade=grade,
            components=components,
            reasons=[f"Trigger={trigger}"] + reasons,
            snapshot_path=snapshot_path,
            rr_min_required=runtime_cfg["rr_min"],
            context=ctx,
            action=action,
            liquidation_cluster=top_liq,
            event_snapshot=event_snapshot,
        )

        ticket_path = ticket.save(CONFIG.ticket_dir)

        db.insert_signal(
            {
                "signal_id": ticket.payload["id"],
                "timestamp": ticket.payload["timestamp"],
                "symbol": ticket.payload["symbol"],
                "event_type": "signal",
                "decision": decision,
                "setup": setup_name,
                "context": ctx,
                "action": action,
                "why": why_text,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr_estimated": rr_est,
                "score": score,
                "ob_imbalance": imb,
                "ob_raw": raw_imb,
                "ob_age_ms": imb_age,
                "funding_rate": funding_rate,
                "oi_now": oi_now,
                "oi_change_pct": oi_change_pct,
                "crowding": crowding,
                "snapshot_path": snapshot_path,
                "ticket_path": ticket_path,
            }
        )

        print("=" * 90)
        print(
            f"[ANALYZE:{trigger}] [{now.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{ticket.payload['symbol']} -> {decision} | setup={setup_name} "
            f"| score={score} ({grade}) | action={action}"
        )

        if decision != "FLAT":
            print(f"ENTRY={entry} SL={sl} TP1={tp1} TP2={tp2}")
            print(f"Snapshot: {snapshot_path}")

        print(f"Ticket:   {ticket_path}")
        print("Reasons:")
        for r in ticket.payload["evidence"]["key_reasons"]:
            print(f" - {r}")

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

            squeeze_risk = squeeze_risk_label_from_prices(last_close, top_liq)

            deriv = build_derivatives_context(CONFIG.symbol)

            setup_info, setup_why = quick_setup_status(df_m15, df_h1, df_h4)
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
                f"setup={setup_info} | action={action} | squeeze_risk={squeeze_risk} | "
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
                    f"oi_change={r.get('oi_change_pct')} crowding={r.get('crowding')}"
                )
                print(f"why={r['why']}")
                print(f"ticket={r['ticket_path']}")

        elif cmd == "e":
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

            last_close = float(df_m15["close"].iloc[-1])

            imb = ob.imbalance(top_n=10)
            raw_imb = ob.raw_imbalance(top_n=10)

            liq_data = fetch_liquidation_map("BTC")
            top_liq = None
            if liq_data and liq_data.get("clusters"):
                top_liq = liq_data["clusters"][0]["price"]

            squeeze_risk = squeeze_risk_label_from_prices(last_close, top_liq)

            setup_info, setup_why = quick_setup_status(df_m15, df_h1, df_h4)
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
                why=f"{setup_why}; squeeze_risk={squeeze_risk}",
                ob_avg=imb,
                ob_raw=raw_imb,
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

    asyncio.create_task(terminal_command_listener())
    asyncio.create_task(file_command_listener())

    await run_ws(CONFIG.symbol, on_closed_kline, on_depth=on_depth)


if __name__ == "__main__":
    asyncio.run(main())