
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.backtest import vector_backtest
from factor_factory.selection import compute_metrics

def main(argv: list[str] | None=None):
    p=argparse.ArgumentParser()
    p.add_argument("--program", required=True, help="JSON file with {'tokens':[...]}")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--outdir", default="rlc_out")
    args=p.parse_args(argv)

    with open(args.program,"r") as f:
        prog = json.load(f)["tokens"]

    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)
    sig = eval_prefix(prog, df).dropna().rename("signal")
    price = df["close"].reindex(sig.index)

    equity, pnl = vector_backtest(price, sig)
    m = compute_metrics(pnl, equity, sig)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sig.to_csv(outdir/"signal.csv", header=True)
    equity.to_csv(outdir/"equity.csv", header=True)
    with open(outdir/"metrics.json","w") as f:
        import json as _json
        _json.dump(m, f, indent=2)
    print("Metrics:", m)

if __name__ == "__main__":
    main()
