import argparse
import json
from pathlib import Path

import pandas as pd
from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.backtest import vector_backtest
from factor_factory.selection import compute_metrics


def _load_tokens(program_arg: str, model_path: str | None) -> list[int]:
    """Load token list from a json file or from best_program.json next to model."""
    if program_arg == "best":
        if model_path is None:
            raise ValueError("--model 경로가 필요합니다 (program=best).")
        best_json = Path(model_path).with_name("best_program.json")
        if not best_json.exists():
            raise FileNotFoundError(f"{best_json} 파일이 없습니다. 학습 시 SaveBestProgramCallback 이 제대로 동작했는지 확인하세요.")
        with open(best_json) as f:
            return json.load(f)["tokens"]
    # else: file path 지정
    with open(program_arg) as f:
        return json.load(f)["tokens"]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True, help="'best' 또는 {'tokens':[...]} JSON 파일")
    parser.add_argument("--model", help="program=best 일 때 PPO 모델 경로")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--outdir", default="rlc_out")
    parser.add_argument("--export_csv", help="price/signal/equity/pnl 통합 CSV 저장 경로")
    args = parser.parse_args(argv)

    # 1. 토큰 로드
    tokens = _load_tokens(args.program, args.model)

    # 2. 데이터 로드
    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)
    sig = eval_prefix(tokens, df).dropna().rename("signal")
    price = df["close"].reindex(sig.index)

    if sig.empty:
        raise ValueError("⚠️ 시그널이 비어 있습니다. 프로그램 토큰을 확인하세요.")

    # 3. 백테스트
    equity, pnl = vector_backtest(price, sig)
    metrics = compute_metrics(pnl, equity, sig)

    # 4. 결과 저장
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sig.to_csv(outdir / "signal.csv")
    equity.to_csv(outdir / "equity.csv")
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.export_csv:
        df_export = pd.DataFrame({
            "price": price,
            "signal": sig,
            "equity": equity,
            "pnl": pnl,
        }).dropna()
        Path(args.export_csv).parent.mkdir(parents=True, exist_ok=True)
        df_export.to_csv(args.export_csv)
        print(f"[저장] 통합 CSV → {args.export_csv}")

    # 5. 콘솔 출력
    print("📊 Metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"✅ Signal 수={len(sig)} | 누적 PnL={pnl.sum():.4f}")


if __name__ == "__main__":
    main()
