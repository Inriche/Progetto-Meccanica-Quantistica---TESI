from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import pandas as pd

from execution.outcome_simulator import (
    load_future_candles_after_timestamp,
    simulate_outcome_from_candles,
)
from features.indicators import atr, rr
from features.market_state import volatility_regime
from features.quantum_state import build_quantum_state
from liquidity.liquidation_engine import fetch_liquidation_map
from market_data.derivatives_context import build_derivatives_context
from market_data.news_context import NewsContext, build_news_context
from risk.risk_governor import RiskGovernor
from runtime.alert_engine import emit_alert
from signal_engine.action_label import suggest_action
from signal_engine.bias_detector import combined_bias, detect_bias_h1, detect_bias_h4
from signal_engine.confluence import confluence_points
from signal_engine.context_classifier import classify_market_context
from signal_engine.derivatives_score import derivatives_points
from signal_engine.explainer import explain_no_setup
from signal_engine.liquidity_context import squeeze_risk_label_from_prices
from signal_engine.news_score import news_points
from signal_engine.quantum_score import quantum_points
from signal_engine.scoring import ScoreBreakdown, compute_score
from signal_engine.setups import (
    SetupResult,
    breakout_confirmation,
    sweep_reclaim,
    trend_pullback,
)
from signal_engine.strategy_profile import StrategyProfile, get_strategy_profile, strategy_points
from signal_engine.ticket import build_ticket
from snapshot.chart_renderer import save_snapshot


@dataclass
class MarketState:
    now: datetime
    runtime_cfg: Dict[str, Any]
    strategy_profile: StrategyProfile
    df_m15: pd.DataFrame
    df_h1: pd.DataFrame
    df_h4: pd.DataFrame
    bias_h1: str
    bias_h4: str
    combined_bias: str
    volatility: str
    context: str
    latest_price: Optional[float]
    squeeze_risk: str
    orderbook_imbalance: Optional[float]
    orderbook_raw: Optional[float]
    orderbook_age_ms: Optional[int]
    liquidation_cluster: Optional[float]
    derivatives: Dict[str, Any]
    quantum: Any
    news: NewsContext
    risk_before: Dict[str, Any]


@dataclass
class FeatureState:
    setup: Optional[SetupResult]
    rr_estimated: Optional[float]
    explanation: str


@dataclass
class ScoreState:
    base_score: int
    breakdown: ScoreBreakdown
    orderbook_points: int
    confluence_points: int
    derivatives_points: int
    quantum_points: int
    news_points: int
    strategy_points: int
    strategy_notes: list[str]
    final_score: int
    grade: str


@dataclass
class SignalDraft:
    decision: str
    setup_name: str
    entry: Optional[float]
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    reasons: list[str]
    score: int
    grade: str
    rr_estimated: Optional[float]
    components: list[tuple[str, int]]
    action: str
    risk_after: Dict[str, Any]
    event_type: str


@dataclass
class EngineCycleResult:
    emitted: bool
    decision: str
    setup: str
    score: float
    action: str
    event_type: str
    context: Optional[str]
    ticket_path: Optional[str]
    rr_estimated: Optional[float]
    trigger: str


class TradingEngine:
    def __init__(
        self,
        *,
        symbol: str,
        store: Any,
        orderbook_store: Any,
        db: Any,
        risk_governor: RiskGovernor,
        runtime_config_loader: Callable[[], Dict[str, Any]],
        snapshot_dir: str,
        ticket_dir: str,
        timezone: Any,
        time_provider: Optional[Callable[[], datetime]] = None,
        enable_artifacts: bool = True,
    ) -> None:
        self.symbol = symbol
        self.store = store
        self.orderbook_store = orderbook_store
        self.db = db
        self.risk = risk_governor
        self.runtime_config_loader = runtime_config_loader
        self.snapshot_dir = snapshot_dir
        self.ticket_dir = ticket_dir
        self.timezone = timezone
        self.time_provider = time_provider
        self.enable_artifacts = bool(enable_artifacts)
        self.logger = logging.getLogger("engine.pipeline")

    def _now(self) -> datetime:
        if self.time_provider is None:
            return datetime.now(self.timezone)
        now = self.time_provider()
        if now.tzinfo is None:
            return now.replace(tzinfo=self.timezone)
        return now

    def _quantum_frame(
        self,
        *,
        timeframe: str,
        base_df: pd.DataFrame,
        min_rows: int = 150,
        as_of_close_time_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Prefer in-memory full buffers; if too short, fallback to DB candles to reach
        a structural window length for persistence/Hurst when available.
        """
        if len(base_df) >= min_rows:
            return base_df.copy()

        conn = getattr(self.db, "conn", None)
        if conn is None:
            return base_df.copy()

        try:
            query = """
                SELECT open_time, close_time, open, high, low, close, volume
                FROM candles
                WHERE timeframe = ?
                  AND UPPER(symbol) = UPPER(?)
            """
            params: list[Any] = [timeframe, self.symbol]
            if as_of_close_time_ms is not None:
                query += " AND close_time <= ?"
                params.append(int(as_of_close_time_ms))
            query += " ORDER BY open_time DESC LIMIT ?"
            params.append(int(min_rows))
            hist = pd.read_sql_query(
                query,
                conn,
                params=params,
            )
            if hist.empty:
                return base_df.copy()

            merged = pd.concat([hist, base_df], ignore_index=True)
            if "open_time" not in merged.columns:
                return base_df.copy()
            merged = (
                merged.sort_values("open_time")
                .drop_duplicates(subset=["open_time"], keep="last")
                .tail(int(min_rows))
                .copy()
            )
            return merged
        except Exception as ex:
            self.logger.warning(
                "[Pipeline] quantum_frame_db_fallback_failed timeframe=%s symbol=%s err=%s",
                timeframe,
                self.symbol,
                ex,
            )
            return base_df.copy()

    async def run_cycle(self, trigger: str) -> EngineCycleResult:
        market = self.load_market_state(trigger=trigger)
        if not self._has_min_history(market.df_m15, market.df_h1, market.df_h4):
            self.logger.info(
                "[Pipeline] warming_up trigger=%s m15=%s h1=%s h4=%s",
                trigger,
                len(market.df_m15),
                len(market.df_h1),
                len(market.df_h4),
            )
            return EngineCycleResult(
                emitted=False,
                decision="FLAT",
                setup="NONE",
                score=0.0,
                action="STAND_BY",
                event_type="status",
                context=None,
                ticket_path=None,
                rr_estimated=None,
                trigger=trigger,
            )

        features = self.build_features(market)
        self.classify_context(market)
        scores = self.compute_all_scores(market, features)
        draft = self.generate_signal_ticket(market, features, scores, trigger)
        draft = self.apply_risk_filter(market, features, scores, draft, trigger)
        execution_info = self.optionally_execute_or_simulate(market, draft)
        ticket_path = self.persist_logs_and_metrics(
            market=market,
            features=features,
            scores=scores,
            draft=draft,
            trigger=trigger,
            execution_info=execution_info,
        )

        emitted = draft.decision in ("BUY", "SELL")
        return EngineCycleResult(
            emitted=emitted,
            decision=draft.decision,
            setup=draft.setup_name,
            score=float(draft.score),
            action=draft.action,
            event_type=draft.event_type,
            context=market.context,
            ticket_path=ticket_path,
            rr_estimated=draft.rr_estimated,
            trigger=trigger,
        )

    def load_market_state(self, *, trigger: str) -> MarketState:
        runtime_cfg = self.runtime_config_loader()
        self.risk.sync_limits(
            int(runtime_cfg["max_signals_per_day"]),
            int(runtime_cfg["cooldown_minutes"]),
        )
        now = self._now()
        now_ms = int(now.timestamp() * 1000)

        strategy_profile = get_strategy_profile(str(runtime_cfg.get("strategy_mode", "BALANCED")))
        df_m15 = self.store.to_df("15m")
        df_h1 = self.store.to_df("1h")
        df_h4 = self.store.to_df("4h")
        quantum_df_m15 = self._quantum_frame(timeframe="15m", base_df=df_m15, min_rows=150, as_of_close_time_ms=now_ms)
        quantum_df_h1 = self._quantum_frame(timeframe="1h", base_df=df_h1, min_rows=150, as_of_close_time_ms=now_ms)
        quantum_df_h4 = self._quantum_frame(timeframe="4h", base_df=df_h4, min_rows=150, as_of_close_time_ms=now_ms)

        bias_h1 = detect_bias_h1(df_h1) if len(df_h1) > 0 else "neutral"
        bias_h4 = detect_bias_h4(df_h4) if len(df_h4) > 0 else "neutral"
        bias_comb = combined_bias(bias_h1, bias_h4)
        volatility = volatility_regime(df_m15) if len(df_m15) > 0 else "low"
        context = classify_market_context(df_m15, df_h1, df_h4) if len(df_m15) > 0 else "transition"
        # Structural persistence may require longer history than short-term bias features.
        quantum = build_quantum_state(quantum_df_m15, quantum_df_h1, quantum_df_h4)

        use_external_context = bool(runtime_cfg.get("use_external_context", True))
        liquidation_cluster = None
        if use_external_context:
            liq_data = fetch_liquidation_map("BTC")
            if liq_data and liq_data.get("clusters"):
                liquidation_cluster = liq_data["clusters"][0]["price"]

        latest_price = float(df_m15["close"].iloc[-1]) if not df_m15.empty else None
        squeeze_risk = squeeze_risk_label_from_prices(
            latest_price,
            liquidation_cluster,
            high_pct=float(runtime_cfg["squeeze_risk_high_pct"]),
            medium_pct=float(runtime_cfg["squeeze_risk_medium_pct"]),
        )

        derivatives = {
            "open_interest_now": None,
            "open_interest_change_pct_15m": None,
            "funding_rate": None,
            "mark_price": None,
            "next_funding_time": None,
            "crowding": "neutral",
            "source": "disabled",
        }
        if use_external_context:
            derivatives = build_derivatives_context(self.symbol)
        news = build_news_context(
            symbol=self.symbol,
            limit=int(runtime_cfg.get("news_headline_limit", 6)),
            cache_minutes=int(runtime_cfg.get("news_cache_minutes", 15)),
            enabled=bool(runtime_cfg.get("news_enabled", True) and use_external_context),
        )

        risk_before = self.risk.status_snapshot(now)

        state = MarketState(
            now=now,
            runtime_cfg=runtime_cfg,
            strategy_profile=strategy_profile,
            df_m15=df_m15,
            df_h1=df_h1,
            df_h4=df_h4,
            bias_h1=bias_h1,
            bias_h4=bias_h4,
            combined_bias=bias_comb,
            volatility=volatility,
            context=context,
            latest_price=latest_price,
            squeeze_risk=squeeze_risk,
            orderbook_imbalance=self.orderbook_store.imbalance(top_n=10),
            orderbook_raw=self.orderbook_store.raw_imbalance(top_n=10),
            orderbook_age_ms=self.orderbook_store.age_ms(),
            liquidation_cluster=liquidation_cluster,
            derivatives=derivatives,
            quantum=quantum,
            news=news,
            risk_before=risk_before,
        )
        self.logger.info("[Pipeline] step1 load_market_state trigger=%s done", trigger)
        return state

    def build_features(self, market: MarketState) -> FeatureState:
        setup = self._select_setup_candidate(
            market.df_m15,
            market.df_h1,
            market.df_h4,
            market.combined_bias,
            market.strategy_profile.code,
        )
        rr_est = None
        if setup is not None:
            rr_est = rr(setup.entry, setup.sl, setup.tp1)

        explanation = explain_no_setup(market.df_m15, market.df_h1, market.df_h4)
        self.logger.info("[Pipeline] step2 build_features setup=%s", setup.setup if setup else "NONE")
        return FeatureState(setup=setup, rr_estimated=rr_est, explanation=explanation)

    def classify_context(self, market: MarketState) -> str:
        self.logger.info("[Pipeline] step3 classify_context context=%s", market.context)
        return market.context

    def compute_all_scores(self, market: MarketState, features: FeatureState) -> ScoreState:
        if features.setup is None or features.rr_estimated is None:
            zero_breakdown = ScoreBreakdown(score=0, grade="C", components=[])
            return ScoreState(
                base_score=0,
                breakdown=zero_breakdown,
                orderbook_points=0,
                confluence_points=0,
                derivatives_points=0,
                quantum_points=0,
                news_points=0,
                strategy_points=0,
                strategy_notes=[],
                final_score=0,
                grade="C",
            )

        funding = market.derivatives.get("funding_rate")
        oi_now = market.derivatives.get("open_interest_now")
        oi_change = market.derivatives.get("open_interest_change_pct_15m")
        crowding = market.derivatives.get("crowding")

        ob_pts = 0
        if market.orderbook_imbalance is not None and market.orderbook_age_ms is not None and market.orderbook_age_ms <= 2000:
            ob_pts = self._orderbook_points(
                features.setup.decision,
                market.orderbook_imbalance,
                market.runtime_cfg,
            )

        conf_pts = confluence_points(features.setup.entry, features.setup.decision, market.df_h1)
        deriv_pts = derivatives_points(
            funding_rate=market.derivatives.get("funding_rate"),
            oi_change_pct=market.derivatives.get("open_interest_change_pct_15m"),
            crowding=market.derivatives.get("crowding"),
            decision=features.setup.decision,
            extreme_oi_pct=float(market.runtime_cfg["derivatives_extreme_oi_pct"]),
            mild_oi_pct=float(market.runtime_cfg["derivatives_mild_oi_pct"]),
        )
        quant_pts = quantum_points(
            quantum=market.quantum,
            decision=features.setup.decision,
            coherence_threshold=float(market.runtime_cfg["quantum_coherence_threshold"]),
            tunneling_threshold=float(market.runtime_cfg["quantum_tunneling_threshold"]),
        )
        n_pts = news_points(
            news=market.news,
            decision=features.setup.decision,
            enabled=bool(market.runtime_cfg.get("news_enabled", True)),
        )
        strat_pts, strat_notes = strategy_points(
            profile=market.strategy_profile,
            setup_name=features.setup.setup,
            decision=features.setup.decision,
            context=market.context,
            squeeze_risk=market.squeeze_risk,
            latest_price=market.latest_price,
            liquidation_cluster=market.liquidation_cluster,
            quantum_coherence=float(market.quantum.coherence),
            quantum_tunneling=float(market.quantum.tunneling_probability),
            quantum_phase_bias=float(market.quantum.phase_bias),
            news_sentiment=float(market.news.sentiment_score),
            news_impact=float(market.news.impact_score),
        )

        heuristic_extra_points = int(ob_pts + conf_pts + deriv_pts + quant_pts + n_pts + strat_pts)
        action_hint = suggest_action(
            market.context,
            features.setup.setup,
            features.explanation,
            market.combined_bias,
            squeeze_risk=market.squeeze_risk,
        )
        ml_features = {
            "ob_imbalance": market.orderbook_imbalance,
            "ob_raw": market.orderbook_raw,
            "ob_age_ms": market.orderbook_age_ms,
            "funding_rate": funding,
            "oi_now": oi_now,
            "oi_change_pct": oi_change,
            "crowding": crowding,
            "strategy_mode": market.strategy_profile.code,
            "strategy_score": strat_pts,
            "news_bias": market.news.bias,
            "news_sentiment": market.news.sentiment_score,
            "news_impact": market.news.impact_score,
            "news_score": n_pts,
            "action": action_hint,
            "quantum_state": market.quantum.state,
            "quantum_coherence": market.quantum.coherence,
            "quantum_phase_bias": market.quantum.phase_bias,
            "quantum_interference": market.quantum.interference,
            "quantum_tunneling": market.quantum.tunneling_probability,
            "quantum_score": quant_pts,
        }
        breakdown = compute_score(
            bias_h1=market.bias_h1,
            bias_h4=market.bias_h4,
            combined=market.combined_bias,
            rr_est=features.rr_estimated,
            volatility=market.volatility,
            setup_name=features.setup.setup,
            context=market.context,
            decision=features.setup.decision,
            ml_features=ml_features,
            extra_heuristic_points=heuristic_extra_points,
        )

        score = int(max(0, min(100, breakdown.score)))
        grade = breakdown.grade
        self.logger.info("[Pipeline] step4 compute_all_scores score=%s grade=%s", score, grade)
        return ScoreState(
            base_score=breakdown.heuristic_score,
            breakdown=breakdown,
            orderbook_points=ob_pts,
            confluence_points=conf_pts,
            derivatives_points=deriv_pts,
            quantum_points=quant_pts,
            news_points=n_pts,
            strategy_points=strat_pts,
            strategy_notes=strat_notes,
            final_score=score,
            grade=grade,
        )

    def generate_signal_ticket(
        self,
        market: MarketState,
        features: FeatureState,
        scores: ScoreState,
        trigger: str,
    ) -> SignalDraft:
        if features.setup is None:
            setup_info, setup_why = self.quick_setup_status(
                market.df_m15,
                market.df_h1,
                market.df_h4,
                float(market.runtime_cfg["rr_min"]),
                market.strategy_profile.code,
            )
            action = suggest_action(
                market.context,
                setup_info,
                setup_why,
                market.combined_bias,
                squeeze_risk=market.squeeze_risk,
            )
            if str(action).startswith("WAIT_BREAKOUT"):
                self._log_wait_breakout_debug(
                    market=market,
                    action=action,
                    decision="FLAT",
                )
            reasons = [
                setup_why,
                f"squeeze_risk={market.squeeze_risk}",
                f"strategy_mode={market.strategy_profile.code}",
                f"news_bias={market.news.bias}",
                f"news_sentiment={market.news.sentiment_score:.2f}",
                f"news_impact={market.news.impact_score:.2f}",
                f"quantum_state={market.quantum.state}",
                f"quantum_coherence={market.quantum.coherence:.2f}",
                f"quantum_phase_bias={market.quantum.phase_bias:.2f}",
                f"quantum_tunneling={market.quantum.tunneling_probability:.2f}",
                f"trigger={trigger}",
            ]
            self.logger.info("[Pipeline] step5 generate_signal_ticket setup=NONE action=%s", action)
            return SignalDraft(
                decision="FLAT",
                setup_name="NONE",
                entry=None,
                sl=None,
                tp1=None,
                tp2=None,
                reasons=reasons,
                score=0,
                grade="C",
                rr_estimated=None,
                components=[],
                action=action,
                risk_after=market.risk_before,
                event_type="status",
            )

        setup = features.setup
        reasons = list(setup.reasons)
        reasons.append(
            f"OrderBook imbalance_avg={market.orderbook_imbalance} raw={market.orderbook_raw} age_ms={market.orderbook_age_ms}"
        )
        reasons.append(f"H1 confluence points={scores.confluence_points}")
        reasons.append(f"derivatives points={scores.derivatives_points}")
        reasons.append(
            f"news_bias={market.news.bias} sentiment={market.news.sentiment_score:.2f} impact={market.news.impact_score:.2f} points={scores.news_points}"
        )
        reasons.append(f"strategy_mode={market.strategy_profile.code} strategy_points={scores.strategy_points}")
        reasons.append(f"quantum_state={market.quantum.state}")
        reasons.append(
            "quantum "
            f"coherence={market.quantum.coherence:.2f} "
            f"phase_bias={market.quantum.phase_bias:.2f} "
            f"interference={market.quantum.interference:.2f} "
            f"tunneling={market.quantum.tunneling_probability:.2f} "
            f"points={scores.quantum_points}"
        )
        reasons.append(f"context={market.context}")
        reasons.append(f"squeeze_risk={market.squeeze_risk}")
        if market.liquidation_cluster is not None:
            reasons.append(f"nearest_liquidation_cluster={market.liquidation_cluster}")

        funding = market.derivatives.get("funding_rate")
        oi_now = market.derivatives.get("open_interest_now")
        oi_change = market.derivatives.get("open_interest_change_pct_15m")
        crowding = market.derivatives.get("crowding")

        if funding is not None:
            reasons.append(f"funding_rate={funding}")
        if oi_now is not None:
            reasons.append(f"oi_now={oi_now}")
        if oi_change is not None:
            reasons.append(f"oi_change_pct_15m={oi_change}")
        if crowding is not None:
            reasons.append(f"crowding={crowding}")
        for note in scores.strategy_notes:
            reasons.append(f"strategy_note={note}")
        if market.news.headlines:
            reasons.append(f"news_topic={market.news.dominant_topic}")

        action = suggest_action(
            market.context,
            setup.setup,
            "; ".join(reasons),
            market.combined_bias,
            squeeze_risk=market.squeeze_risk,
        )
        reasons.append(f"action={action}")

        components = list(scores.breakdown.components) + [
            ("orderbook_imbalance", scores.orderbook_points),
            ("h1_confluence", scores.confluence_points),
            ("derivatives_context", scores.derivatives_points),
            ("quantum_context", scores.quantum_points),
            ("news_context", scores.news_points),
            ("strategy_context", scores.strategy_points),
        ]
        self.logger.info(
            "[Pipeline] step5 generate_signal_ticket setup=%s decision=%s score=%s",
            setup.setup,
            setup.decision,
            scores.final_score,
        )
        return SignalDraft(
            decision=setup.decision,
            setup_name=setup.setup,
            entry=setup.entry,
            sl=setup.sl,
            tp1=setup.tp1,
            tp2=setup.tp2,
            reasons=reasons,
            score=scores.final_score,
            grade=scores.grade,
            rr_estimated=features.rr_estimated,
            components=components,
            action=action,
            risk_after=market.risk_before,
            event_type="signal",
        )

    def _log_wait_breakout_debug(
        self,
        *,
        market: MarketState,
        action: str,
        decision: str,
    ) -> None:
        df_m15 = market.df_m15
        if len(df_m15) < 20:
            self.logger.info(
                "[WAIT_BREAKOUT_DEBUG] ts=%s action=%s decision=%s price=%s note=insufficient_m15_history",
                market.now.isoformat(),
                action,
                decision,
                market.latest_price,
            )
            return

        last = df_m15.iloc[-1]
        prev = df_m15.iloc[-2]
        range_high = float(df_m15["high"].iloc[-12:-1].max())
        range_low = float(df_m15["low"].iloc[-12:-1].min())
        candle_range = float(last["high"] - last["low"])
        atr14 = atr(df_m15, 14).iloc[-1]
        atr14_value = float(atr14) if pd.notna(atr14) else None
        atr_ok = bool(pd.notna(atr14) and candle_range <= float(atr14) * 1.8)

        broke_up = bool(last["close"] > range_high)
        broke_down = bool(last["close"] < range_low)
        strong_close_up = bool(last["close"] > last["open"])
        strong_close_down = bool(last["close"] < last["open"])
        prev_not_broken_up = bool(prev["close"] <= range_high)
        prev_not_broken_down = bool(prev["close"] >= range_low)

        price = float(market.latest_price) if market.latest_price is not None else None
        long_distance_abs = (price - range_high) if price is not None else None
        short_distance_abs = (price - range_low) if price is not None else None
        long_distance_pct = ((price / range_high) - 1.0) if (price is not None and range_high > 0) else None
        short_distance_pct = ((price / range_low) - 1.0) if (price is not None and range_low > 0) else None
        atr_ratio = (candle_range / atr14_value) if (atr14_value is not None and atr14_value > 0) else None

        self.logger.info(
            "[WAIT_BREAKOUT_DEBUG] ts=%s price=%.6f breakout_long=%.6f breakout_short=%.6f "
            "dist_long=%.6f dist_long_pct=%.6f dist_short=%.6f dist_short_pct=%.6f "
            "vol_atr14=%.6f candle_range=%.6f range_atr_ratio=%.6f atr_ok=%s "
            "bias=%s broke_up=%s strong_up=%s prev_ok_up=%s broke_down=%s strong_down=%s prev_ok_down=%s "
            "quantum_coherence=%.6f decision=%s action=%s",
            market.now.isoformat(),
            price if price is not None else float("nan"),
            range_high,
            range_low,
            long_distance_abs if long_distance_abs is not None else float("nan"),
            long_distance_pct if long_distance_pct is not None else float("nan"),
            short_distance_abs if short_distance_abs is not None else float("nan"),
            short_distance_pct if short_distance_pct is not None else float("nan"),
            atr14_value if atr14_value is not None else float("nan"),
            candle_range,
            atr_ratio if atr_ratio is not None else float("nan"),
            atr_ok,
            market.combined_bias,
            broke_up,
            strong_close_up,
            prev_not_broken_up,
            broke_down,
            strong_close_down,
            prev_not_broken_down,
            float(market.quantum.coherence),
            decision,
            action,
        )

    def apply_risk_filter(
        self,
        market: MarketState,
        features: FeatureState,
        scores: ScoreState,
        draft: SignalDraft,
        trigger: str,
    ) -> SignalDraft:
        if draft.setup_name == "NONE":
            draft.risk_after = market.risk_before
            self._emit_news_regime_alert(market, trigger, draft.action)
            self.logger.info("[Pipeline] step6 apply_risk_filter status_only")
            return draft

        passes_rr = (
            draft.rr_estimated is not None
            and draft.rr_estimated >= float(market.runtime_cfg["rr_min"])
        )
        passes_score = draft.score >= int(market.runtime_cfg["min_score_for_signal"])
        passes_risk = bool(market.risk_before.get("can_emit"))

        if not (passes_rr and passes_score and passes_risk):
            draft.decision = "FLAT"
            draft.setup_name = "BLOCKED"
            draft.event_type = "signal"
            draft.reasons = [
                f"Setup found but blocked by filters. Trigger={trigger}",
                (
                    f"RR_est={draft.rr_estimated:.2f} (min {float(market.runtime_cfg['rr_min']):.2f})"
                    if draft.rr_estimated is not None
                    else "RR_est=N/A"
                ),
                (
                    f"score={draft.score} (min {int(market.runtime_cfg['min_score_for_signal'])}) "
                    f"base={scores.base_score} ob_pts={scores.orderbook_points} "
                    f"conf_pts={scores.confluence_points} deriv_pts={scores.derivatives_points} "
                    f"quant_pts={scores.quantum_points} news_pts={scores.news_points} "
                    f"strategy_pts={scores.strategy_points}"
                ),
                (
                    f"Risk: can_emit={passes_risk} "
                    f"block_reason={market.risk_before.get('block_reason')} "
                    f"signals_today={market.risk_before.get('signals_today')}/"
                    f"{market.risk_before.get('max_signals_per_day')} "
                    f"cooldown_remaining_minutes={market.risk_before.get('cooldown_remaining_minutes')}"
                ),
                f"context={market.context}",
                f"squeeze_risk={market.squeeze_risk}",
                f"action={draft.action}",
            ]
            draft.risk_after = market.risk_before
        else:
            self.risk.mark_emitted(market.now)
            draft.risk_after = self.risk.status_snapshot(market.now)

        self.logger.info(
            "[Pipeline] step6 apply_risk_filter passes_rr=%s passes_score=%s passes_risk=%s decision=%s",
            passes_rr,
            passes_score,
            passes_risk,
            draft.decision,
        )
        return draft

    def optionally_execute_or_simulate(self, market: MarketState, draft: SignalDraft) -> Dict[str, Any]:
        if draft.decision not in ("BUY", "SELL") or draft.entry is None or draft.sl is None or draft.tp1 is None:
            self.logger.info("[Pipeline] step7 optionally_execute_or_simulate skipped")
            return {
                "mode": "simulate",
                "status": "skipped",
                "reason": "No executable signal.",
            }

        future_df = load_future_candles_after_timestamp(
            market.now.isoformat(),
            timeframe="15m",
            limit=int(market.runtime_cfg.get("validation_horizon_bars", 16)),
        )
        outcome = simulate_outcome_from_candles(
            decision=draft.decision,
            entry=draft.entry,
            sl=draft.sl,
            tp1=draft.tp1,
            tp2=draft.tp2,
            future_df=future_df,
        )
        self.logger.info("[Pipeline] step7 optionally_execute_or_simulate outcome=%s", outcome.get("status"))
        return {
            "mode": "simulate",
            "status": outcome.get("status"),
            "bars_checked": outcome.get("bars_checked"),
            "hit_price": outcome.get("hit_price"),
            "hit_time": outcome.get("hit_time"),
        }

    def persist_logs_and_metrics(
        self,
        *,
        market: MarketState,
        features: FeatureState,
        scores: ScoreState,
        draft: SignalDraft,
        trigger: str,
        execution_info: Dict[str, Any],
    ) -> Optional[str]:
        why_text = "; ".join(draft.reasons)
        funding = market.derivatives.get("funding_rate")
        oi_now = market.derivatives.get("open_interest_now")
        oi_change = market.derivatives.get("open_interest_change_pct_15m")
        crowding = market.derivatives.get("crowding")

        if draft.setup_name == "NONE":
            self._save_status_event(
                now=market.now,
                event_type="status",
                decision=draft.decision,
                setup=draft.setup_name,
                context=market.context,
                action=draft.action,
                why=why_text,
                entry=draft.entry,
                sl=draft.sl,
                tp1=draft.tp1,
                tp2=draft.tp2,
                rr_estimated=draft.rr_estimated,
                score=draft.score,
                ob_imbalance=market.orderbook_imbalance,
                ob_raw=market.orderbook_raw,
                ob_age_ms=market.orderbook_age_ms,
                funding_rate=funding,
                oi_now=oi_now,
                oi_change_pct=oi_change,
                crowding=crowding,
                strategy_mode=market.strategy_profile.code,
                strategy_score=scores.strategy_points,
                news_bias=market.news.bias,
                news_sentiment=market.news.sentiment_score,
                news_impact=market.news.impact_score,
                news_score=scores.news_points,
                quantum_state=market.quantum.state,
                quantum_coherence=market.quantum.coherence,
                quantum_phase_bias=market.quantum.phase_bias,
                quantum_interference=market.quantum.interference,
                quantum_tunneling=market.quantum.tunneling_probability,
                quantum_score=scores.quantum_points,
                ticket_path=None,
                snapshot_path=None,
            )
            self.logger.info("[Pipeline] step8 persist_logs_and_metrics status_saved")
            return None

        snapshot_path = ""
        if self.enable_artifacts and draft.decision in ("BUY", "SELL"):
            snapshot_path = save_snapshot(
                market.df_m15,
                self.snapshot_dir,
                self.symbol,
                market.now,
                draft.entry,
                draft.sl,
                draft.tp1,
                draft.tp2,
            )

        event_snapshot = {
            "price": market.latest_price,
            "context": market.context,
            "action": draft.action,
            "squeeze_risk": market.squeeze_risk,
            "orderbook": {
                "imbalance_avg": market.orderbook_imbalance,
                "raw": market.orderbook_raw,
                "age_ms": market.orderbook_age_ms,
            },
            "derivatives": {
                "funding_rate": funding,
                "oi_now": oi_now,
                "oi_change_pct_15m": oi_change,
                "crowding": crowding,
            },
            "strategy": {
                "mode": market.strategy_profile.code,
                "label": market.strategy_profile.label,
                "description": market.strategy_profile.description,
                "points": scores.strategy_points,
                "notes": scores.strategy_notes,
            },
            "news": market.news.to_dict(),
            "quantum": {
                "state": market.quantum.state,
                "coherence": market.quantum.coherence,
                "phase_bias": market.quantum.phase_bias,
                "interference": market.quantum.interference,
                "tunneling_probability": market.quantum.tunneling_probability,
                "amplitude": market.quantum.amplitude,
                "state_confidence": market.quantum.state_confidence,
                "points": scores.quantum_points,
            },
            "liquidity": {
                "nearest_liquidation_cluster": market.liquidation_cluster,
            },
            "risk": draft.risk_after,
            "execution": execution_info,
        }

        signal_id = f"sig_{market.now.strftime('%Y%m%d_%H%M%S_%f')}_{self.symbol.upper()}"
        signal_timestamp = market.now.isoformat()
        signal_symbol = self.symbol.upper()
        ticket_path: Optional[str] = None
        ticket_payload_id: Optional[str] = None
        if self.enable_artifacts:
            ticket = build_ticket(
                symbol=self.symbol,
                timestamp=market.now,
                decision=draft.decision,
                setup=draft.setup_name,
                bias_h1=market.bias_h1,
                bias_h4=market.bias_h4,
                combined_bias=market.combined_bias,
                volatility=market.volatility,
                entry=draft.entry,
                sl=draft.sl,
                tp1=draft.tp1,
                tp2=draft.tp2,
                score=draft.score,
                grade=draft.grade,
                components=draft.components,
                reasons=[f"Trigger={trigger}"] + draft.reasons + [f"execution_status={execution_info.get('status')}"],
                snapshot_path=snapshot_path,
                rr_min_required=float(market.runtime_cfg["rr_min"]),
                min_score_for_signal=int(market.runtime_cfg["min_score_for_signal"]),
                max_signals_per_day=int(market.runtime_cfg["max_signals_per_day"]),
                cooldown_minutes=int(market.runtime_cfg["cooldown_minutes"]),
                context=market.context,
                action=draft.action,
                liquidation_cluster=market.liquidation_cluster,
                strategy_snapshot=event_snapshot["strategy"],
                news_snapshot=event_snapshot["news"],
                quantum_snapshot=event_snapshot["quantum"],
                event_snapshot=event_snapshot,
                risk_gate_status=draft.risk_after,
            )
            ticket_path = ticket.save(self.ticket_dir)
            signal_id = str(ticket.payload["id"])
            signal_timestamp = str(ticket.payload["timestamp"])
            signal_symbol = str(ticket.payload["symbol"])
            ticket_payload_id = signal_id

        self.db.insert_signal(
            {
                "signal_id": signal_id,
                "timestamp": signal_timestamp,
                "symbol": signal_symbol,
                "event_type": "signal",
                "decision": draft.decision,
                "setup": draft.setup_name,
                "context": market.context,
                "action": draft.action,
                "why": why_text,
                "entry": draft.entry,
                "sl": draft.sl,
                "tp1": draft.tp1,
                "tp2": draft.tp2,
                "rr_estimated": draft.rr_estimated,
                "score": draft.score,
                "ob_imbalance": market.orderbook_imbalance,
                "ob_raw": market.orderbook_raw,
                "ob_age_ms": market.orderbook_age_ms,
                "funding_rate": funding,
                "oi_now": oi_now,
                "oi_change_pct": oi_change,
                "crowding": crowding,
                "strategy_mode": market.strategy_profile.code,
                "strategy_score": scores.strategy_points,
                "news_bias": market.news.bias,
                "news_sentiment": market.news.sentiment_score,
                "news_impact": market.news.impact_score,
                "news_score": scores.news_points,
                "quantum_state": market.quantum.state,
                "quantum_coherence": market.quantum.coherence,
                "quantum_phase_bias": market.quantum.phase_bias,
                "quantum_interference": market.quantum.interference,
                "quantum_tunneling": market.quantum.tunneling_probability,
                "quantum_score": scores.quantum_points,
                "snapshot_path": snapshot_path,
                "ticket_path": ticket_path,
            }
        )

        if ticket_payload_id is not None and ticket_path is not None:
            self._emit_trade_alerts(
                market=market,
                draft=draft,
                scores=scores,
                ticket_id=ticket_payload_id,
                ticket_path=ticket_path,
                trigger=trigger,
                why_text=why_text,
            )
        self._emit_news_regime_alert(market, trigger, draft.action)
        self.logger.info(
            "[Pipeline] step8 persist_logs_and_metrics signal_saved decision=%s setup=%s score=%s",
            draft.decision,
            draft.setup_name,
            draft.score,
        )
        return ticket_path

    def quick_setup_status(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        rr_min: float,
        strategy_mode: str,
    ) -> tuple[str, str]:
        if not self._has_min_history(df_m15, df_h1, df_h4):
            return "warming_up", "not enough history"

        b_h1 = detect_bias_h1(df_h1)
        b_h4 = detect_bias_h4(df_h4)
        b_comb = combined_bias(b_h1, b_h4)
        setup_res = self._select_setup_candidate(df_m15, df_h1, df_h4, b_comb, strategy_mode)

        if setup_res is None:
            return "no_setup", explain_no_setup(df_m15, df_h1, df_h4)

        rr_est = rr(setup_res.entry, setup_res.sl, setup_res.tp1)
        if rr_est < rr_min:
            return (
                "weak_candidate",
                f"{setup_res.setup} rejected early: rr too low ({rr_est:.2f} < min {rr_min:.2f})",
            )
        return f"{setup_res.setup} rr={rr_est:.2f} dir={setup_res.decision}", "candidate found"

    def _has_min_history(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> bool:
        return len(df_m15) >= 150 and len(df_h1) >= 200 and len(df_h4) >= 60

    def _select_setup_candidate(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame,
        combined_bias_value: str,
        strategy_mode: str,
    ) -> Optional[SetupResult]:
        profile = get_strategy_profile(strategy_mode)
        if combined_bias_value not in ("bullish", "bearish"):
            return sweep_reclaim(df_m15, "neutral")

        for setup_name in profile.setup_priority:
            if setup_name == "TREND_PULLBACK":
                res = trend_pullback(df_m15, df_h1, combined_bias_value)
            elif setup_name == "SWEEP_RECLAIM":
                res = sweep_reclaim(df_m15, combined_bias_value)
            elif setup_name == "BREAKOUT_CONFIRMATION":
                res = breakout_confirmation(df_m15, df_h1, combined_bias_value)
            else:
                res = None

            if res is not None:
                return res
        return None

    def _orderbook_points(self, decision: str, imbalance: float, runtime_cfg: Dict[str, Any]) -> int:
        neutral_th = float(runtime_cfg["orderbook_neutral_threshold"])
        full_th = float(runtime_cfg["orderbook_full_score_threshold"])
        if abs(imbalance) < neutral_th:
            return 0

        span = max(0.0001, full_th - neutral_th)
        strength = min(1.0, (abs(imbalance) - neutral_th) / span)
        pts = int(round(10 * strength))

        if decision == "BUY":
            return pts if imbalance > 0 else -pts
        if decision == "SELL":
            return pts if imbalance < 0 else -pts
        return 0

    def _save_status_event(self, *, now: Optional[datetime] = None, **row: Any) -> None:
        last_evt = self.db.get_last_event()
        if last_evt is not None:
            if (
                last_evt.get("event_type") == row.get("event_type")
                and last_evt.get("decision") == row.get("decision")
                and last_evt.get("setup") == row.get("setup")
                and last_evt.get("context") == row.get("context")
                and last_evt.get("action") == row.get("action")
                and last_evt.get("why") == row.get("why")
            ):
                return

        event_now = now or self._now()
        signal_id = (
            f"evt_{row.get('event_type')}_"
            f"{event_now.strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{self.symbol.upper()}"
        )
        payload = {
            "signal_id": signal_id,
            "timestamp": event_now.isoformat(),
            "symbol": self.symbol.upper(),
        }
        payload.update(row)
        self.db.insert_signal(payload)

    def _emit_news_regime_alert(self, market: MarketState, trigger: str, action: str) -> None:
        cfg = market.runtime_cfg
        if not bool(cfg.get("alerts_enabled", True)):
            return
        impact_threshold = float(cfg.get("news_alert_min_impact", 0.70))
        if float(market.news.impact_score) < impact_threshold:
            return
        if str(market.news.bias) not in ("bullish", "bearish", "high_alert"):
            return

        severity = "critical" if float(market.news.impact_score) >= 0.85 else "warning"
        emit_alert(
            alert_type="news_regime",
            title=f"High-impact news regime ({market.news.bias})",
            body=(
                f"Topic={market.news.dominant_topic} | impact={market.news.impact_score:.2f} "
                f"| sentiment={market.news.sentiment_score:.2f} | ctx={market.context} | action={action} "
                f"| strategy={market.strategy_profile.code} | trigger={trigger}"
            ),
            severity=severity,
            created_at=market.now,
            dedup_key=f"news:{self.symbol.upper()}:{market.news.bias}:{market.news.dominant_topic}",
            cooldown_minutes=int(cfg.get("alert_cooldown_minutes", 30)),
            metadata={
                "symbol": self.symbol.upper(),
                "news_bias": market.news.bias,
                "news_impact": market.news.impact_score,
                "news_sentiment": market.news.sentiment_score,
                "dominant_topic": market.news.dominant_topic,
                "headline_count": market.news.headline_count,
                "strategy_mode": market.strategy_profile.code,
                "context": market.context,
                "action": action,
                "trigger": trigger,
            },
        )

    def _emit_trade_alerts(
        self,
        *,
        market: MarketState,
        draft: SignalDraft,
        scores: ScoreState,
        ticket_id: str,
        ticket_path: str,
        trigger: str,
        why_text: str,
    ) -> None:
        cfg = market.runtime_cfg
        if not bool(cfg.get("alerts_enabled", True)):
            return

        cooldown_minutes = int(cfg.get("alert_cooldown_minutes", 30))
        rr_text = "N/A" if draft.rr_estimated is None else f"{draft.rr_estimated:.2f}"
        metadata = {
            "symbol": self.symbol.upper(),
            "decision": draft.decision,
            "setup": draft.setup_name,
            "action": draft.action,
            "score": draft.score,
            "rr_estimated": draft.rr_estimated,
            "price": market.latest_price,
            "strategy_mode": market.strategy_profile.code,
            "strategy_score": scores.strategy_points,
            "news_bias": market.news.bias,
            "news_impact": market.news.impact_score,
            "news_score": scores.news_points,
            "quantum_state": market.quantum.state,
            "quantum_coherence": market.quantum.coherence,
            "quantum_score": scores.quantum_points,
            "context": market.context,
            "squeeze_risk": market.squeeze_risk,
            "risk_can_emit": draft.risk_after.get("can_emit"),
            "risk_block_reason": draft.risk_after.get("block_reason"),
            "trigger": trigger,
            "ticket_path": ticket_path,
            "why": why_text,
        }

        if draft.decision in ("BUY", "SELL") and int(draft.score) >= int(cfg.get("alert_min_score", 74)):
            severity = "critical" if int(draft.score) >= 85 else "info"
            emit_alert(
                alert_type="signal_live",
                title=f"{draft.decision} signal live ({draft.setup_name})",
                body=(
                    f"score={draft.score} | rr={rr_text} | strategy={market.strategy_profile.code} "
                    f"| news={market.news.bias}/{market.news.impact_score:.2f} "
                    f"| quantum={market.quantum.state}/{market.quantum.coherence:.2f} | action={draft.action}"
                ),
                severity=severity,
                created_at=market.now,
                dedup_key=f"signal:{self.symbol.upper()}:{draft.decision}:{draft.setup_name}:{market.strategy_profile.code}",
                cooldown_minutes=cooldown_minutes,
                signal_id=ticket_id,
                metadata=metadata,
            )
        elif draft.setup_name == "BLOCKED" and int(draft.score) >= int(cfg.get("blocked_alert_min_score", 68)):
            emit_alert(
                alert_type="blocked_candidate",
                title="Strong candidate blocked",
                body=(
                    f"setup={draft.setup_name} | score={draft.score} | rr={rr_text} "
                    f"| block_reason={draft.risk_after.get('block_reason')} | strategy={market.strategy_profile.code} "
                    f"| action={draft.action}"
                ),
                severity="warning",
                created_at=market.now,
                dedup_key=f"blocked:{self.symbol.upper()}:{draft.setup_name}:{market.strategy_profile.code}",
                cooldown_minutes=cooldown_minutes,
                signal_id=ticket_id,
                metadata=metadata,
            )
