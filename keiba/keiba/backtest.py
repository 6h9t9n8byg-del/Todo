"""ウォークフォワード検証と馬券シミュレーション。

競馬モデルの評価でランダム分割の交差検証を使うのは誤り。時間を跨いだ情報が
漏れる上、実運用（過去だけで未来を当てる）と条件が違うため。ここでは
「ある時点までのデータで学習 → 次の期間を予測」を繰り返す方式のみを提供する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import summarize
from .pipeline import Predictor, PredictorConfig


@dataclass
class BacktestConfig:
    initial_train_days: int = 420   # 最初の学習に確保する日数
    step_days: int = 60             # 1フォールドで予測する期間
    train_window_days: int | None = None  # None なら全過去を使う（expanding）
    min_train_races: int = 300      # これ未満のフォールドは学習せず飛ばす
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    verbose: bool = True


def walk_forward(feat: pd.DataFrame, config: BacktestConfig | None = None) -> pd.DataFrame:
    """時系列ウォークフォワードで全期間のアウトオブサンプル予測を作る。"""
    cfg = config or BacktestConfig()
    feat = feat.sort_values(["date", "race_id"]).reset_index(drop=True)
    labeled = feat[feat["finish_pos"].notna()]
    start = labeled["date"].min() + pd.Timedelta(days=cfg.initial_train_days)
    end = labeled["date"].max()

    preds: list[pd.DataFrame] = []
    fold_start = start
    fold = 0
    while fold_start <= end:
        fold_end = fold_start + pd.Timedelta(days=cfg.step_days)
        train_mask = labeled["date"] < fold_start
        if cfg.train_window_days is not None:
            train_mask &= labeled["date"] >= fold_start - pd.Timedelta(days=cfg.train_window_days)
        train = labeled[train_mask]
        test = labeled[(labeled["date"] >= fold_start) & (labeled["date"] < fold_end)]

        if test.empty or train["race_id"].nunique() < cfg.min_train_races:
            fold_start = fold_end
            continue

        fold += 1
        predictor = Predictor(config=cfg.predictor)
        predictor.fit(train)
        out = predictor.predict(test)
        out["fold"] = fold
        out["blend_weight"] = predictor.blend_weight_
        preds.append(out)

        if cfg.verbose:
            s = summarize(out["p_final"].to_numpy(), out["y_win"].to_numpy(), out["race_id"])
            print(
                f"fold {fold:>2} {fold_start.date()}..{(fold_end - pd.Timedelta(days=1)).date()} "
                f"train={train['race_id'].nunique():>5}R test={s['races']:>4}R "
                f"top1={s['top1_accuracy']:.3f} NLL={s['nll']:.4f} "
                f"w_model={predictor.blend_weight_:.2f}"
            )
        fold_start = fold_end

    if not preds:
        raise ValueError("バックテスト可能な期間がありません（データ期間または設定を確認）")
    return pd.concat(preds, ignore_index=True)


# --------------------------------------------------------------------- 馬券戦略
@dataclass
class BettingConfig:
    """単勝馬券のシミュレーション設定。"""

    ev_threshold: float = 0.15      # 期待値のしきい値（0.15 = +15% 以上を狙う）
    min_prob: float = 0.03          # これ未満の勝率は買わない（推定誤差が大きい）
    min_odds: float = 1.5
    max_odds: float = 50.0          # 極端な人気薄は分散が大きすぎる
    stake_mode: str = "kelly"       # "kelly" または "flat"
    kelly_fraction: float = 0.25    # フラクショナルケリー（1.0 は破産リスクが高い）
    max_stake_fraction: float = 0.01
    flat_stake: float = 100.0
    initial_bankroll: float = 1_000_000.0
    unit: float = 100.0             # 馬券は100円単位


def simulate_bets(preds: pd.DataFrame, config: BettingConfig | None = None) -> dict:
    """予測結果に対して単勝馬券を模擬購入する。

    Returns
    -------
    dict: ``summary``（集計）, ``bets``（購入明細）, ``curve``（資金推移）
    """
    cfg = config or BettingConfig()
    df = preds.sort_values(["date", "race_id"]).copy()
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")

    eligible = (
        (df["ev"] > cfg.ev_threshold)
        & (df["p_final"] >= cfg.min_prob)
        & (df["odds"].between(cfg.min_odds, cfg.max_odds))
    )
    candidates = df[eligible]

    bankroll = cfg.initial_bankroll
    peak = bankroll
    max_dd = 0.0
    rows = []
    curve = []

    for (date, race_id), grp in candidates.groupby(["date", "race_id"], sort=False):
        for r in grp.itertuples():
            if cfg.stake_mode == "kelly":
                f = min(float(r.kelly) * cfg.kelly_fraction, cfg.max_stake_fraction)
                stake = np.floor(bankroll * f / cfg.unit) * cfg.unit
            else:
                stake = cfg.flat_stake
            if stake < cfg.unit or stake > bankroll:
                continue
            won = bool(r.y_win > 0.5)
            payout = stake * float(r.odds) if won else 0.0
            bankroll += payout - stake
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
            rows.append({
                "date": date, "race_id": race_id, "horse_id": r.horse_id,
                "odds": float(r.odds), "p_final": float(r.p_final),
                "p_market": float(r.p_market), "ev": float(r.ev),
                "stake": stake, "payout": payout, "won": won, "bankroll": bankroll,
            })
        curve.append({"date": date, "race_id": race_id, "bankroll": bankroll})

    bets = pd.DataFrame(rows)
    if bets.empty:
        return {"summary": {"n_bets": 0, "note": "条件を満たす馬券がありません"},
                "bets": bets, "curve": pd.DataFrame(curve)}

    stake_sum = float(bets["stake"].sum())
    payout_sum = float(bets["payout"].sum())
    summary = {
        "n_bets": int(len(bets)),
        "n_races_bet": int(bets["race_id"].nunique()),
        "total_stake": stake_sum,
        "total_payout": payout_sum,
        "profit": payout_sum - stake_sum,
        "roi": payout_sum / stake_sum - 1.0,
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "avg_odds_hit": float(bets.loc[bets["won"], "odds"].mean()) if bets["won"].any() else float("nan"),
        "final_bankroll": float(bankroll),
        "max_drawdown": float(max_dd),
    }
    summary["roi_ci95"] = _bootstrap_roi_ci(bets)
    return {"summary": summary, "bets": bets, "curve": pd.DataFrame(curve)}


def _bootstrap_roi_ci(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """レース単位のブートストラップで ROI の 95% 信頼区間を出す。

    馬券の収益は分散が非常に大きく、点推定の ROI だけを見ると危険なため。
    """
    rng = np.random.default_rng(seed)
    grouped = bets.groupby("race_id")[["stake", "payout"]].sum().to_numpy()
    if len(grouped) < 20:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))
    sampled = grouped[idx]
    roi = sampled[:, :, 1].sum(axis=1) / np.maximum(sampled[:, :, 0].sum(axis=1), 1e-9) - 1.0
    return (float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975)))


def strategy_grid(preds: pd.DataFrame, thresholds=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
                  stake_mode: str = "flat") -> pd.DataFrame:
    """期待値しきい値を振って戦略の感度を見る。"""
    rows = []
    for t in thresholds:
        res = simulate_bets(preds, BettingConfig(ev_threshold=t, stake_mode=stake_mode))
        s = res["summary"]
        if s["n_bets"] == 0:
            rows.append({"ev_threshold": t, "n_bets": 0})
            continue
        rows.append({
            "ev_threshold": t, "n_bets": s["n_bets"], "hit_rate": s["hit_rate"],
            "roi": s["roi"], "roi_lo": s["roi_ci95"][0], "roi_hi": s["roi_ci95"][1],
            "profit": s["profit"], "max_drawdown": s["max_drawdown"],
        })
    return pd.DataFrame(rows)
