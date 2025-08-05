import argparse
import json
from pathlib import Path

import pandas as pd
from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc.compiler import eval_prefix
from factor_factory.rlc.utils import tokens_to_infix, calc_tree_depth
from factor_factory.backtest import vector_backtest
from factor_factory.selection import compute_metrics
from factor_factory.visualization import create_trading_report


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
    parser.add_argument("--charts", action="store_true", help="트레이딩 차트 생성")
    parser.add_argument("--chart_dir", default="charts", help="차트 저장 디렉토리")
    args = parser.parse_args(argv)

    # 1. 토큰 로드
    tokens = _load_tokens(args.program, args.model)
    
    # 프로그램 정보 출력
    infix_formula = tokens_to_infix(tokens)
    tree_depth = calc_tree_depth(tokens)
    
    print("=" * 80)
    print("📊 PROGRAM EVALUATION REPORT")
    print("=" * 80)
    print(f"🔢 토큰 수: {len(tokens)}")
    print(f"📏 트리 깊이: {tree_depth}")
    print(f"🧮 수식: {infix_formula}")
    print(f"🔢 토큰 리스트: {tokens}")
    print("=" * 80)

    # 2. 데이터 로드
    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)
    print(f"📈 데이터: {args.symbol}_{args.interval}")
    print(f"📅 기간: {df.index[0]} ~ {df.index[-1]} ({len(df):,} bars)")
    
    sig = eval_prefix(tokens, df).dropna().rename("signal")
    price = df["close"].reindex(sig.index)

    if sig.empty:
        raise ValueError("⚠️ 시그널이 비어 있습니다. 프로그램 토큰을 확인하세요.")

    print(f"📊 시그널 통계:")
    print(f"   - 범위: [{sig.min():.4f}, {sig.max():.4f}]")
    print(f"   - 평균: {sig.mean():.4f}")
    print(f"   - 표준편차: {sig.std():.4f}")
    print(f"   - 롱 신호 비율: {(sig > 0.1).mean():.2%}")
    print(f"   - 숏 신호 비율: {(sig < -0.1).mean():.2%}")
    print(f"   - 플랫 비율: {(abs(sig) <= 0.1).mean():.2%}")

    # 3. 백테스트
    print("\n🔄 백테스트 실행 중...")
    equity, pnl = vector_backtest(price, sig)
    metrics = compute_metrics(pnl, equity, sig)

    # 성능 메트릭 출력
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE METRICS")
    print("=" * 80)
    
    print(f"💰 CAGR (연평균 성장률):        {metrics['cagr']:8.2%}")
    print(f"📈 Sharpe Ratio:              {metrics['sharpe']:8.4f}")
    print(f"📉 Max Drawdown:              {metrics['mdd']:8.2%}")
    print(f"🔄 Turnover:                  {metrics['turnover']:8.2f}")
    print(f"🏆 Calmar Ratio:              {metrics['calmar']:8.4f}")
    print(f"🎯 Win Rate:                  {metrics['win_rate']:8.2%}")
    print(f"💎 Profit Factor:             {metrics['profit_factor']:8.4f}")
    print(f"📊 Information Ratio:         {metrics['information_ratio']:8.4f}")
    print(f"📉 Max Consecutive Losses:    {metrics['max_consecutive_losses']:8d} days")
    print(f"🔢 Total Trades:              {metrics['total_trades']:8d}")
    print(f"💵 Avg Trade PnL:             {metrics['avg_trade_pnl']:8.6f}")
    
    # 최종 수익률
    total_return = (equity.iloc[-1] - 1) * 100
    print(f"🎊 Total Return:              {total_return:8.2f}%")
    print("=" * 80)

    # 4. 결과 저장
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 프로그램 정보도 함께 저장
    program_info = {
        "tokens": tokens,
        "formula": infix_formula,
        "tree_depth": tree_depth,
        "token_count": len(tokens)
    }
    
    sig.to_csv(outdir / "signal.csv")
    equity.to_csv(outdir / "equity.csv")
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / "program_info.json", "w") as f:
        json.dump(program_info, f, indent=2)
    
    print(f"\n💾 결과 저장:")
    print(f"   - 시그널: {outdir}/signal.csv")
    print(f"   - 수익곡선: {outdir}/equity.csv") 
    print(f"   - 성능지표: {outdir}/metrics.json")
    print(f"   - 프로그램정보: {outdir}/program_info.json")

    # 5. 차트 생성 (옵션)
    if args.charts:
        print(f"\n📊 트레이딩 차트 생성 중...")
        try:
            create_trading_report(
                price=price,
                signal=sig,
                equity=equity,
                pnl=pnl,
                metrics=metrics,
                formula=infix_formula,
                output_dir=args.chart_dir
            )
        except Exception as e:
            print(f"⚠️ 차트 생성 중 오류: {e}")
            print("   matplotlib 설치 여부를 확인하세요: pip install matplotlib")

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
