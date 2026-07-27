"""競馬予想システム。

レース単位の確率モデル（条件付きロジット + LightGBM ランカー/分類器のアンサンブル）と
市場オッズのブレンドで単勝勝率を推定し、ウォークフォワードで検証する。
"""

from .backtest import BacktestConfig, BettingConfig, simulate_bets, strategy_grid, walk_forward
from .data import GeneratorConfig, SyntheticRaceGenerator, generate_dataset
from .features import FeatureBuilder, build_features
from .market import blend, implied_probabilities
from .metrics import summarize
from .pipeline import Predictor, PredictorConfig
from .schema import load_csv

__all__ = [
    "BacktestConfig",
    "BettingConfig",
    "FeatureBuilder",
    "GeneratorConfig",
    "Predictor",
    "PredictorConfig",
    "SyntheticRaceGenerator",
    "blend",
    "build_features",
    "generate_dataset",
    "implied_probabilities",
    "load_csv",
    "simulate_bets",
    "strategy_grid",
    "summarize",
    "walk_forward",
]

__version__ = "0.1.0"
