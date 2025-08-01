
from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("./data_cache")

class ParquetCache:
    def __init__(self, root: Path | str = DATA_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str, interval: str) -> Path:
        fn = f"{symbol}_{interval}.parquet"
        return self.root / fn

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        p = self.path(symbol, interval)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}. Place OHLCV parquet with columns [open,high,low,close,volume].")
        df = pd.read_parquet(p)
        required = {"open","high","low","close","volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {p}. Need {required}.")
        df = df.sort_index()
        return df
