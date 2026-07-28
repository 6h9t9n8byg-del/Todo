"""コマンドラインインターフェース。

    python -m keiba generate --out data/races.csv
    python -m keiba backtest --data data/races.csv
    python -m keiba train    --data data/races.csv --out model.joblib
    python -m keiba predict  --history data/races.csv --entries data/entries.csv \
                             --model model.joblib --html sheet.html
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BettingConfig, simulate_bets, strategy_grid, walk_forward
from .data import GeneratorConfig, generate_dataset
from .features import FeatureBuilder
from .metrics import calibration_table, format_summary, summarize
from .pipeline import Predictor, PredictorConfig
from .report import write_report
from .ratings import REST_NAME, RatingConfig, fit_ratings
from .realdata import check_consistency, load_horses, load_races, races_for_rating
from .schema import coerce, load_csv, validate
from .tickets import build_proposals
from .webapp import build_payload, write_app

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


# ------------------------------------------------------------------ generate
def cmd_generate(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig(seed=args.seed, n_race_days=args.days, n_horses=args.horses)
    df = generate_dataset(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"生成: {len(df):,} 行 / {df['race_id'].nunique():,} レース "
          f"({df['date'].min()} 〜 {df['date'].max()}) -> {args.out}")
    return 0


# --------------------------------------------------------------------- split
def cmd_split(args: argparse.Namespace) -> int:
    """データを「過去成績」と「出馬表（結果を伏せたもの）」に分割する。"""
    df = load_csv(args.data, require_results=True)
    cutoff = df["date"].max() - pd.Timedelta(days=args.entry_days)
    history, entries = df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()
    if args.races:
        keep = entries["race_id"].drop_duplicates().head(args.races)
        entries = entries[entries["race_id"].isin(set(keep))]
    entries[["finish_pos", "finish_time", "corner_pos", "last_3f"]] = np.nan

    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history_out, index=False)
    entries.to_csv(args.entries_out, index=False)
    print(f"過去成績: {history['race_id'].nunique():,} レース -> {args.history_out}")
    print(f"出馬表  : {entries['race_id'].nunique():,} レース -> {args.entries_out}")
    return 0


# --------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    df = load_csv(args.data, require_results=True)
    feat = FeatureBuilder().build(df)
    predictor = Predictor(config=PredictorConfig(
        use_market=not args.no_market, valid_days=args.valid_days, verbose=True
    ))
    predictor.fit(feat)
    report = predictor.valid_report_
    print("\n=== 学習サマリ ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("\n=== 重要度上位 ===")
    print(predictor.feature_importance(args.top).round(4).to_string())
    predictor.save(args.out)
    print(f"\nモデルを保存しました -> {args.out}")
    return 0


# ------------------------------------------------------------------ backtest
def cmd_backtest(args: argparse.Namespace) -> int:
    df = load_csv(args.data, require_results=True)
    feat = FeatureBuilder().build(df)
    cfg = BacktestConfig(
        initial_train_days=args.initial_train_days,
        step_days=args.step_days,
        train_window_days=args.train_window_days,
        predictor=PredictorConfig(use_market=not args.no_market),
    )
    print("=== ウォークフォワード検証 ===")
    preds = walk_forward(feat, cfg)

    y = preds["y_win"].to_numpy()
    races = preds["race_id"]
    print("\n=== アウトオブサンプル精度 ===")
    print(format_summary("モデルのみ", summarize(preds["p_model"].to_numpy(), y, races)))
    if preds["p_market"].notna().any():
        print(format_summary("市場(オッズ)のみ", summarize(preds["p_market"].to_numpy(), y, races)))
    print(format_summary("最終(ブレンド)", summarize(preds["p_final"].to_numpy(), y, races)))

    print("\n=== キャリブレーション（最終確率） ===")
    print(calibration_table(preds["p_final"].to_numpy(), y).round(4).to_string(index=False))

    print("\n=== 単勝シミュレーション（期待値しきい値ごと / 均一100円） ===")
    print(strategy_grid(preds).round(4).to_string(index=False))

    res = simulate_bets(preds, BettingConfig(ev_threshold=args.ev_threshold, stake_mode="kelly"))
    print(f"\n=== ケリー基準（EV>{args.ev_threshold}, 1/4ケリー） ===")
    for k, v in res["summary"].items():
        print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"\n予測結果を保存しました -> {args.out}")
    return 0


# ------------------------------------------------------------------- predict
def cmd_predict(args: argparse.Namespace) -> int:
    history = load_csv(args.history, require_results=True)
    entries = coerce(pd.read_csv(args.entries))
    validate(entries)

    target_races = set(entries["race_id"].dropna().unique())
    combined = pd.concat([history[~history["race_id"].isin(target_races)], entries],
                         ignore_index=True)
    feat = FeatureBuilder().build(combined)

    if args.model and Path(args.model).exists():
        predictor = Predictor.load(args.model)
        print(f"モデルを読み込みました: {args.model}")
    else:
        print("モデル未指定のため履歴から学習します...")
        predictor = Predictor(config=PredictorConfig(verbose=True))
        predictor.fit(feat[feat["finish_pos"].notna()])

    target = feat[feat["race_id"].isin(target_races)]
    if target.empty:
        print("予測対象のレースが見つかりません", file=sys.stderr)
        return 1

    preds = predictor.predict(target)
    cols = ["race_id", "draw", "horse_name", "jockey_id", "odds",
            "p_model", "p_market", "p_final", "ev", "kelly"]
    for race_id, grp in preds.groupby("race_id"):
        print(f"\n=== {race_id} ===")
        print(grp[cols].round(4).to_string(index=False))

    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"\n-> {args.out}")
    if args.html:
        write_report(preds, args.html)
        print(f"-> {args.html}")
    return 0


# -------------------------------------------------------------------- webapp
def cmd_webapp(args: argparse.Namespace) -> int:
    """予測結果をスマホで見られる単一HTMLに書き出す。"""
    preds = pd.read_csv(args.preds)
    payload = build_payload(preds, n_races=args.races, metrics_source=preds)
    size = write_app(payload, args.out, fragment_only=args.fragment_only)
    print(f"レース {len(payload['races'])} 件 / {size / 1024:.0f} KB -> {args.out}")
    print("スマートフォンに転送して開くか、静的ホスティングに置いてください。")
    return 0


# ---------------------------------------------------------------------- real
def cmd_real(args: argparse.Namespace) -> int:
    """実在馬のレーティングと、指定した出走馬に対する買い目。"""
    races, race_meta = load_races()
    horses, _ = load_horses()
    problems = check_consistency(races, horses)
    if problems:
        print("データに問題があります:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 1

    ratings = fit_ratings(races_for_rating(races),
                          config=RatingConfig(prior_sd=args.prior_sd),
                          as_of=date.today())

    print(f"収集: {race_meta.get('collected')} / レース {len(races)} 件 / 馬 {len(horses)} 頭")
    print("出典は data/real/*.yaml を参照。買う前に必ず裏を取ること。\n")

    if not args.horses:
        print("=== レーティング（大きいほど強い。se は推定の不確かさ） ===")
        table = ratings.table()
        table = table[table["horse"] != REST_NAME] if args.hide_rest else table
        print(table.round(3).to_string(index=False))
        print("\n--horses 馬名,馬名,... を付けると、その顔ぶれでの勝率と買い目を出します。")
        return 0

    field = [h.strip() for h in args.horses.split(",") if h.strip()]
    unknown = [h for h in field if ratings.index(h) is None]
    if unknown:
        print(f"※ 記録が無い馬（事前分布で扱います）: {', '.join(unknown)}\n")

    table = ratings.field_table(field)
    print("=== 勝率 ===")
    print(table.round(4).to_string(index=False))

    order = table["horse"].tolist()
    probs = table["win_prob"].to_numpy()
    draws = list(range(1, len(order) + 1))
    proposals = build_proposals(probs, draws, draws, n_runners=len(order))

    print("\n=== 買い目（番号は上の並び順） ===")
    for bet in proposals:
        sep = "→" if bet.ordered else "-"
        legs = " / ".join(sep.join(map(str, t["legs"])) for t in bet.tickets)
        print(f"  {bet.name:<5} 当たる確率 {bet.hit_prob:6.1%}  {bet.points:>2}点  "
              f"必要オッズ {bet.breakeven_odds:6.1f}倍  [{legs}]")
    return 0


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args: argparse.Namespace) -> int:
    preds = pd.read_csv(args.preds)
    y = preds["y_win"].to_numpy()
    races = preds["race_id"]
    for col, label in [("p_model", "モデルのみ"), ("p_market", "市場のみ"), ("p_final", "最終")]:
        if col in preds and preds[col].notna().any():
            print(format_summary(label, summarize(preds[col].to_numpy(), y, races)))
    print("\n=== キャリブレーション ===")
    print(calibration_table(preds["p_final"].to_numpy(), y).round(4).to_string(index=False))
    print("\n=== 戦略グリッド ===")
    print(strategy_grid(preds).round(4).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keiba", description="競馬予想システム")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="動作確認用の合成レースデータを生成")
    g.add_argument("--out", default="data/races.csv")
    g.add_argument("--days", type=int, default=220, help="開催日数")
    g.add_argument("--horses", type=int, default=1200)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("split", help="データを過去成績と出馬表に分割（predict の動作確認用）")
    s.add_argument("--data", required=True)
    s.add_argument("--history-out", default="data/history.csv")
    s.add_argument("--entries-out", default="data/entries.csv")
    s.add_argument("--entry-days", type=int, default=1, help="末尾の何日分を出馬表にするか")
    s.add_argument("--races", type=int, default=None, help="出馬表に含めるレース数の上限")
    s.set_defaults(func=cmd_split)

    t = sub.add_parser("train", help="全履歴でモデルを学習して保存")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="model.joblib")
    t.add_argument("--valid-days", type=int, default=150)
    t.add_argument("--no-market", action="store_true", help="オッズを使わない")
    t.add_argument("--top", type=int, default=30)
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backtest", help="ウォークフォワード検証と馬券シミュレーション")
    b.add_argument("--data", required=True)
    b.add_argument("--out", default=None, help="予測結果CSVの出力先")
    b.add_argument("--step-days", type=int, default=60)
    b.add_argument("--initial-train-days", type=int, default=420)
    b.add_argument("--train-window-days", type=int, default=None)
    b.add_argument("--ev-threshold", type=float, default=0.15)
    b.add_argument("--no-market", action="store_true")
    b.set_defaults(func=cmd_backtest)

    pr = sub.add_parser("predict", help="出馬表（未来のレース）を予測")
    pr.add_argument("--history", required=True, help="過去成績CSV")
    pr.add_argument("--entries", required=True, help="予測対象の出馬表CSV")
    pr.add_argument("--model", default=None)
    pr.add_argument("--out", default=None)
    pr.add_argument("--html", default=None, help="HTML予想紙の出力先")
    pr.set_defaults(func=cmd_predict)

    w = sub.add_parser("webapp", help="予測結果をスマホ用の単一HTMLに書き出す")
    w.add_argument("--preds", required=True, help="backtest --out で保存した予測CSV")
    w.add_argument("--out", default="app.html")
    w.add_argument("--races", type=int, default=120, help="収録するレース数（新しい順）")
    w.add_argument("--fragment-only", action="store_true",
                   help="head/body を含まない断片として出力する")
    w.set_defaults(func=cmd_webapp)

    rl = sub.add_parser("real", help="実在馬のレーティングと買い目")
    rl.add_argument("--horses", default=None,
                    help="出走馬をカンマ区切りで指定（例: クロワデュノール,メイショウタバル）")
    rl.add_argument("--prior-sd", type=float, default=1.2,
                    help="レーティングの事前分布の広さ（小さいほど慎重）")
    rl.add_argument("--hide-rest", action="store_true", help="仮想の「着順不明の出走馬」を隠す")
    rl.set_defaults(func=cmd_real)

    e = sub.add_parser("evaluate", help="保存済み予測CSVを評価")
    e.add_argument("--preds", required=True)
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
