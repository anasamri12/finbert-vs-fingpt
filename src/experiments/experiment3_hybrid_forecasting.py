from __future__ import annotations

import argparse
import copy
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATE_CANDIDATES = ("market_date", "date", "Date", "published_date", "published_at", "datetime")
PRICE_CANDIDATES = {
    "close": ("close", "Adj Close", "Close", "price"),
    "open": ("open", "Open"),
    "high": ("high", "High"),
    "low": ("low", "Low"),
    "volume": ("volume", "Volume"),
}
DEFAULT_MODEL_TYPES = ("lstm", "gru")
DEFAULT_SENTIMENT_FEATURES = (
    "sentiment_score_mean",
    "sentiment_score_std",
    "article_count_log1p",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "source_day_count",
)
DEFAULT_PRICE_FEATURES = (
    "close",
    "return_1d",
    "log_return_1d",
    # open_close_pct and high_low_pct removed: the merged CSV has Adjusted Close in the
    # "close" column but unadjusted OHLC from Yahoo Finance, making cross-column ratios
    # unreliable (e.g., intraday gap would read as ~-19% due to the adj/unadj mismatch).
    "volume_log1p",
    "volume_change_1d",
    "sma_5_gap",
    "sma_10_gap",
    "ema_5_gap",
    "ema_10_gap",
    "volatility_5",
    "volatility_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
)


@dataclass
class SequenceSplit:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    test_actual: np.ndarray
    test_current_close: np.ndarray
    test_market_dates: list[str]
    test_target_dates: list[str]
    n_rows_clean: int
    n_rows_train: int
    n_rows_val: int
    n_rows_test: int
    n_sequences_train: int
    n_sequences_val: int
    n_sequences_test: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def _normalize_col_name(col: Any) -> str:
    if isinstance(col, tuple):
        parts = [str(x).strip() for x in col if str(x).strip() and str(x).strip().lower() != "nan"]
        return " ".join(parts).strip()
    return str(col).strip()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_names = [_normalize_col_name(c) for c in out.columns]

    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in flat_names:
        base = name if name else "col"
        count = seen.get(base, 0)
        unique = base if count == 0 else f"{base}_{count+1}"
        seen[base] = count + 1
        unique_names.append(unique)

    out.columns = unique_names
    return out


def _split_repeated_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if item:
                out.append(item)
    return out


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def safe_name(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_").replace("^", "").replace("/", "_").replace("\\", "_")


def infer_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    normalized = {_normalize_col_name(c).lower(): c for c in df.columns}

    for cand in candidates:
        col = normalized.get(cand.lower())
        if col is not None:
            return str(col)

    for cand in candidates:
        cand_low = cand.lower()
        for norm_name, raw_col in normalized.items():
            if norm_name.startswith(cand_low + " ") or norm_name.startswith(cand_low + "_"):
                return str(raw_col)
            if cand_low in norm_name and cand_low in {"adj close", "close", "open", "high", "low", "volume", "date"}:
                return str(raw_col)

    raise ValueError(f"Could not infer {label} column. Available columns: {list(df.columns)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def resolve_input_files(args: argparse.Namespace) -> list[str]:
    if args.input_csv:
        files = [str(Path(p)) for p in args.input_csv]
    else:
        files = sorted(glob.glob(args.input_glob))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No merged market files matched. input_csv={args.input_csv}, input_glob={args.input_glob}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: benchmark price-only and sentiment-augmented sequence models on merged market data."
    )
    parser.add_argument(
        "--input-glob",
        default="results/experiment2_finbert_run2_*_filtered/*_merged_market.csv",
        help="Glob for merged market CSVs from Experiment 2. Ignored if --input-csv is provided.",
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=[],
        help="Specific merged market CSV to evaluate. Repeat or pass comma-separated paths.",
    )
    parser.add_argument("--out-dir", default="results/experiment3", help="Output folder.")
    parser.add_argument(
        "--target-mode",
        choices=("next_close", "next_return"),
        default="next_return",
        help="Forecast next close price or next return.",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument("--sequence-length", type=int, default=10, help="Number of past market days per input sequence.")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Chronological train fraction.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Chronological validation fraction.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs per model.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for LSTM/GRU.")
    parser.add_argument("--num-layers", type=int, default=2, help="Recurrent layers.")
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout for the recurrent head.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience on validation loss.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm (0 to disable).")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-types",
        default="lstm,gru",
        help="Comma-separated model types to train. Supported: lstm,gru",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on processed merged market CSVs.")
    args = parser.parse_args()

    args.input_csv = _unique_keep_order(_split_repeated_args(args.input_csv))
    args.model_types = tuple(_unique_keep_order(_split_repeated_args([args.model_types])))

    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be >= 2")
    if not (0.0 < args.train_frac < 1.0):
        parser.error("--train-frac must be between 0 and 1")
    if not (0.0 < args.val_frac < 1.0):
        parser.error("--val-frac must be between 0 and 1")
    if args.train_frac + args.val_frac >= 1.0:
        parser.error("train_frac + val_frac must be < 1")
    invalid_models = sorted(set(args.model_types) - set(DEFAULT_MODEL_TYPES))
    if invalid_models:
        parser.error(f"Unsupported --model-types values: {invalid_models}")
    return args


def load_merged_market(csv_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(csv_path)
    raw = _flatten_columns(raw)

    date_col = infer_column(raw, DATE_CANDIDATES, "market date")
    close_col = infer_column(raw, PRICE_CANDIDATES["close"], "close price")
    open_col = infer_column(raw, PRICE_CANDIDATES["open"], "open price")
    high_col = infer_column(raw, PRICE_CANDIDATES["high"], "high price")
    low_col = infer_column(raw, PRICE_CANDIDATES["low"], "low price")
    volume_col = infer_column(raw, PRICE_CANDIDATES["volume"], "volume")

    df = raw.copy()
    df["market_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df["open"] = pd.to_numeric(df[open_col], errors="coerce")
    df["high"] = pd.to_numeric(df[high_col], errors="coerce")
    df["low"] = pd.to_numeric(df[low_col], errors="coerce")
    df["volume"] = pd.to_numeric(df[volume_col], errors="coerce")

    df = df.dropna(subset=["market_date", "close", "open", "high", "low", "volume"]).copy()
    df = df.sort_values("market_date").drop_duplicates(subset=["market_date"], keep="first").reset_index(drop=True)

    meta = {
        "input_csv": csv_path,
        "rows_input": int(len(raw)),
        "rows_after_clean": int(len(df)),
        "date_col": date_col,
        "close_col": close_col,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "volume_col": volume_col,
        "date_min": str(df["market_date"].min().date()) if len(df) else None,
        "date_max": str(df["market_date"].max().date()) if len(df) else None,
    }
    return df, meta


def engineer_features(df: pd.DataFrame, target_mode: str, horizon: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy().sort_values("market_date").reset_index(drop=True)

    out["return_1d"] = out["close"].pct_change()
    out["log_return_1d"] = np.log(out["close"]).diff()
    out["open_close_pct"] = (out["close"] - out["open"]) / out["open"].replace(0.0, np.nan)
    out["high_low_pct"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["volume_log1p"] = np.log1p(out["volume"].clip(lower=0.0))
    out["volume_change_1d"] = out["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    sma_5 = out["close"].rolling(5).mean()
    sma_10 = out["close"].rolling(10).mean()
    ema_5 = out["close"].ewm(span=5, adjust=False).mean()
    ema_10 = out["close"].ewm(span=10, adjust=False).mean()
    ema_12 = out["close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["close"].ewm(span=26, adjust=False).mean()

    out["sma_5_gap"] = (out["close"] / sma_5) - 1.0
    out["sma_10_gap"] = (out["close"] / sma_10) - 1.0
    out["ema_5_gap"] = (out["close"] / ema_5) - 1.0
    out["ema_10_gap"] = (out["close"] / ema_10) - 1.0
    out["volatility_5"] = out["return_1d"].rolling(5).std()
    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["rsi_14"] = compute_rsi(out["close"], period=14)
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    if "article_count" in out.columns:
        out["article_count_log1p"] = np.log1p(pd.to_numeric(out["article_count"], errors="coerce").clip(lower=0.0))
    else:
        out["article_count_log1p"] = np.nan

    out["current_close"] = out["close"]
    out["target_date"] = out["market_date"].shift(-horizon)
    if target_mode == "next_close":
        out["target"] = out["close"].shift(-horizon)
    else:
        out["target"] = out["close"].shift(-horizon) / out["close"] - 1.0

    feature_groups = {
        "price_only": [feature for feature in DEFAULT_PRICE_FEATURES if feature in out.columns],
        "price_plus_sentiment": [
            feature
            for feature in (*DEFAULT_PRICE_FEATURES, *DEFAULT_SENTIMENT_FEATURES)
            if feature in out.columns
        ],
    }
    return out, feature_groups


def prepare_sequence_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    train_frac: float,
    val_frac: float,
    batch_size: int,
) -> SequenceSplit:
    required_cols = list(feature_cols) + ["target", "target_date", "current_close", "market_date"]
    clean = df.dropna(subset=required_cols).copy().sort_values("market_date").reset_index(drop=True)
    if len(clean) < max(sequence_length + 20, 80):
        raise ValueError(
            f"Not enough clean rows after feature engineering ({len(clean)}) for sequence_length={sequence_length}."
        )

    n_rows = len(clean)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))

    train_end = max(train_end, sequence_length + 5)
    val_end = max(val_end, train_end + 5)
    val_end = min(val_end, n_rows - 5)

    if train_end >= val_end or val_end >= n_rows:
        raise ValueError(
            f"Chronological split failed for n_rows={n_rows}, train_end={train_end}, val_end={val_end}."
        )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_feature_frame = clean.loc[: train_end - 1, feature_cols].astype(float)
    train_target_frame = clean.loc[: train_end - 1, ["target"]].astype(float)

    clean_scaled = clean.copy()
    clean_scaled[feature_cols] = clean_scaled[feature_cols].astype(float)
    clean_scaled.loc[: train_end - 1, feature_cols] = feature_scaler.fit_transform(train_feature_frame)
    if train_end < n_rows:
        clean_scaled.loc[train_end:, feature_cols] = feature_scaler.transform(clean.loc[train_end:, feature_cols].astype(float))

    clean_scaled["target_scaled"] = np.nan
    clean_scaled.loc[: train_end - 1, "target_scaled"] = target_scaler.fit_transform(train_target_frame).ravel()
    if train_end < n_rows:
        clean_scaled.loc[train_end:, "target_scaled"] = target_scaler.transform(clean.loc[train_end:, ["target"]].astype(float)).ravel()

    x_train: list[np.ndarray] = []
    y_train: list[float] = []
    x_val: list[np.ndarray] = []
    y_val: list[float] = []
    x_test: list[np.ndarray] = []
    y_test: list[float] = []
    test_actual: list[float] = []
    test_current_close: list[float] = []
    test_market_dates: list[str] = []
    test_target_dates: list[str] = []

    feature_values = clean_scaled[feature_cols].to_numpy(dtype=np.float32)
    target_values = clean_scaled["target_scaled"].to_numpy(dtype=np.float32)

    for target_idx in range(sequence_length - 1, n_rows):
        start_idx = target_idx - sequence_length + 1
        x_seq = feature_values[start_idx : target_idx + 1]
        y_item = target_values[target_idx]

        if target_idx < train_end:
            x_train.append(x_seq)
            y_train.append(float(y_item))
        elif target_idx < val_end:
            x_val.append(x_seq)
            y_val.append(float(y_item))
        else:
            x_test.append(x_seq)
            y_test.append(float(y_item))
            test_actual.append(float(clean.loc[target_idx, "target"]))
            test_current_close.append(float(clean.loc[target_idx, "current_close"]))
            test_market_dates.append(str(pd.Timestamp(clean.loc[target_idx, "market_date"]).date()))
            test_target_dates.append(str(pd.Timestamp(clean.loc[target_idx, "target_date"]).date()))

    if not x_train or not x_val or not x_test:
        raise ValueError("Sequence split produced an empty split. Try a smaller --sequence-length.")

    x_train_tensor = torch.tensor(np.stack(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_val_tensor = torch.tensor(np.stack(x_val), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.float32)
    x_test_tensor = torch.tensor(np.stack(x_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return SequenceSplit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_actual=np.array(test_actual, dtype=float),
        test_current_close=np.array(test_current_close, dtype=float),
        test_market_dates=test_market_dates,
        test_target_dates=test_target_dates,
        n_rows_clean=int(n_rows),
        n_rows_train=int(train_end),
        n_rows_val=int(val_end - train_end),
        n_rows_test=int(n_rows - val_end),
        n_sequences_train=int(len(x_train)),
        n_sequences_val=int(len(x_val)),
        n_sequences_test=int(len(x_test)),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


class SequenceRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, model_type: str) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if model_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
            )
        head_hidden = max(16, hidden_size // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def train_model(
    split: SequenceSplit,
    model_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple[nn.Module, dict[str, Any]]:
    model = SequenceRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in split.train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in split.val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    meta = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "history": history,
    }
    return model, meta


def invert_predictions(pred_scaled: np.ndarray, target_scaler: StandardScaler) -> np.ndarray:
    return target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    current_close: np.ndarray,
    target_mode: str,
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    # MAPE is only meaningful for next_close (price levels); for next_return the denominator
    # is near-zero returns which make MAPE explode to meaningless thousands of percent.
    if target_mode == "next_close":
        non_zero = np.abs(actual) > 1e-8
        mape = float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100.0) if non_zero.any() else np.nan
    else:
        mape = np.nan
    r2 = float(r2_score(actual, predicted))

    if target_mode == "next_close":
        true_dir = np.sign(actual - current_close)
        pred_dir = np.sign(predicted - current_close)
    else:
        true_dir = np.sign(actual)
        pred_dir = np.sign(predicted)

    valid = (true_dir != 0) & (pred_dir != 0)
    directional_accuracy = float(np.mean(true_dir[valid] == pred_dir[valid])) if valid.any() else np.nan
    directional_coverage = float(np.mean(valid)) if len(valid) else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "directional_coverage": directional_coverage,
    }


def predict_loader(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def build_persistence_baseline(
    actual: np.ndarray,
    current_close: np.ndarray,
    target_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    if target_mode == "next_close":
        predicted = current_close.copy()
    else:
        predicted = np.zeros_like(actual)
    metrics = evaluate_predictions(actual=actual, predicted=predicted, current_close=current_close, target_mode=target_mode)
    return predicted, metrics


def choose_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def run_single_input(csv_path: str, args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict[str, Any]:
    dataset_name = safe_name(csv_path)
    dataset_out_dir = out_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    merged_df, load_meta = load_merged_market(csv_path)
    feature_df, feature_groups = engineer_features(merged_df, target_mode=args.target_mode, horizon=args.horizon)

    engineered_csv = dataset_out_dir / "engineered_dataset.csv"
    feature_df.to_csv(engineered_csv, index=False)

    metrics_rows: list[dict[str, Any]] = []
    summary_models: list[dict[str, Any]] = []

    for feature_set_name, feature_cols in feature_groups.items():
        split = prepare_sequence_split(
            df=feature_df,
            feature_cols=feature_cols,
            sequence_length=args.sequence_length,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            batch_size=args.batch_size,
        )

        if feature_set_name == "price_only":
            persistence_pred, persistence_metrics = build_persistence_baseline(
                actual=split.test_actual,
                current_close=split.test_current_close,
                target_mode=args.target_mode,
            )
            pd.DataFrame(
                {
                    "market_date": split.test_market_dates,
                    "target_date": split.test_target_dates,
                    "current_close": split.test_current_close,
                    "actual_target": split.test_actual,
                    "predicted_target": persistence_pred,
                }
            ).to_csv(dataset_out_dir / "predictions_persistence_price_only.csv", index=False)
            metrics_rows.append(
                {
                    "dataset": dataset_name,
                    "input_csv": csv_path,
                    "feature_set": feature_set_name,
                    "model_family": "persistence",
                    "target_mode": args.target_mode,
                    "horizon": args.horizon,
                    "sequence_length": args.sequence_length,
                    "n_rows_clean": split.n_rows_clean,
                    "n_sequences_train": split.n_sequences_train,
                    "n_sequences_val": split.n_sequences_val,
                    "n_sequences_test": split.n_sequences_test,
                    **persistence_metrics,
                }
            )

        for model_type in args.model_types:
            model, train_meta = train_model(
                split=split,
                model_type=model_type,
                input_size=len(feature_cols),
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                grad_clip=args.grad_clip,
            )

            pred_scaled = predict_loader(model, split.test_loader, device=device)
            pred = invert_predictions(pred_scaled, split.target_scaler)
            metrics = evaluate_predictions(
                actual=split.test_actual,
                predicted=pred,
                current_close=split.test_current_close,
                target_mode=args.target_mode,
            )

            model_label = f"{model_type}_{feature_set_name}"
            torch.save(model.state_dict(), dataset_out_dir / f"{model_label}.pt")

            pd.DataFrame(
                {
                    "market_date": split.test_market_dates,
                    "target_date": split.test_target_dates,
                    "current_close": split.test_current_close,
                    "actual_target": split.test_actual,
                    "predicted_target": pred,
                }
            ).to_csv(dataset_out_dir / f"predictions_{model_label}.csv", index=False)
            pd.DataFrame(train_meta["history"]).to_csv(dataset_out_dir / f"training_history_{model_label}.csv", index=False)

            metrics_rows.append(
                {
                    "dataset": dataset_name,
                    "input_csv": csv_path,
                    "feature_set": feature_set_name,
                    "model_family": model_type,
                    "target_mode": args.target_mode,
                    "horizon": args.horizon,
                    "sequence_length": args.sequence_length,
                    "n_rows_clean": split.n_rows_clean,
                    "n_sequences_train": split.n_sequences_train,
                    "n_sequences_val": split.n_sequences_val,
                    "n_sequences_test": split.n_sequences_test,
                    "best_epoch": train_meta["best_epoch"],
                    "best_val_loss": train_meta["best_val_loss"],
                    **metrics,
                }
            )
            summary_models.append(
                {
                    "model_label": model_label,
                    "feature_cols": feature_cols,
                    "train_meta": train_meta,
                    "metrics": metrics,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["feature_set", "model_family"]).reset_index(drop=True)
    metrics_csv = dataset_out_dir / "experiment3_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "dataset": dataset_name,
        "input_csv": csv_path,
        "device": str(device),
        "target_mode": args.target_mode,
        "horizon": args.horizon,
        "sequence_length": args.sequence_length,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "patience": args.patience,
        "load_meta": load_meta,
        "feature_groups": feature_groups,
        "engineered_dataset_csv": str(engineered_csv),
        "metrics_csv": str(metrics_csv),
        "models": summary_models,
    }
    summary_json = dataset_out_dir / "experiment3_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_txt = dataset_out_dir / "experiment3_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("Experiment 3: Hybrid Forecasting\n")
        handle.write(f"Input CSV: {csv_path}\n")
        handle.write(f"Target mode: {args.target_mode}\n")
        handle.write(f"Horizon: {args.horizon}\n")
        handle.write(f"Sequence length: {args.sequence_length}\n")
        handle.write(f"Device: {device}\n\n")
        handle.write(metrics_df.to_string(index=False))
        handle.write("\n")

    print(f"[experiment3] Saved {metrics_csv}")
    return {
        "dataset": dataset_name,
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = resolve_input_files(args)
    device = choose_device(args.device)

    run_manifest: list[dict[str, Any]] = []
    for csv_path in input_files:
        run_manifest.append(run_single_input(csv_path=csv_path, args=args, out_dir=out_dir, device=device))

    manifest_path = out_dir / "experiment3_manifest.json"
    manifest_path.write_text(json.dumps({"runs": run_manifest}, indent=2), encoding="utf-8")
    print(f"[experiment3] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
