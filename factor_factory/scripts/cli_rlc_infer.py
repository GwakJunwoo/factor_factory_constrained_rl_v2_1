from __future__ import annotations
import argparse, json
from pathlib import Path
from stable_baselines3 import PPO

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import ProgramEnv, RLCConfig, tokens_to_infix, calc_tree_depth

def rollout_once(model: PPO, env: ProgramEnv, deterministic: bool = True):
    obs, _ = env.reset()
    done = False
    info_last = {}
    prog = None
    rew = 0.0
    while not done:
        act, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, done, trunc, info = env.step(int(act))
        if done:
            info_last = info
            prog = info.get("program")
            break
    return rew, prog, info_last

def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_program.zip")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--tries", type=int, default=256)
    p.add_argument("--outdir", default="rlc_out")
    p.add_argument("--eval_stride", type=int, default=2)
    p.add_argument("--max_eval_bars", type=int, default=20000)
    p.add_argument("--long_threshold", type=float, default=1.5)
    p.add_argument("--short_threshold", type=float, default=-1.5)
    p.add_argument("--rolling_window", type=int, default=252)
    args = p.parse_args(argv)

    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)
    cfg = RLCConfig(
        eval_stride=args.eval_stride, 
        max_eval_bars=args.max_eval_bars,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        rolling_window=args.rolling_window
    )
    env = ProgramEnv(df, cfg)
    model = PPO.load(args.model)

    print(f"🤖 모델에서 최적 프로그램 탐색 중... ({args.tries} tries)")
    best = (-1e9, None, {})
    for i in range(args.tries):
        rew, prog, info = rollout_once(model, env, deterministic=False)
        if prog is not None and rew > best[0]:
            best = (rew, prog, info)
            # 새로운 최적 프로그램 발견 시 출력
            formula = tokens_to_infix(prog)
            print(f"🏆 새로운 최적 발견! (시도 {i+1}/{args.tries}) reward={rew:.4f}")
            print(f"   수식: {formula}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if best[1] is None:
        print("❌ 유효한 프로그램을 찾지 못했습니다.")
        return

    prog = best[1]
    formula = tokens_to_infix(prog)
    reward = best[0]
    info = best[2]
    
    print("\n" + "="*80)
    print("🎉 BEST PROGRAM FOUND!")
    print("="*80)
    print(f"🏆 최고 보상: {reward:.4f}")
    print(f"🧮 수식: {formula}")
    print(f"🔢 토큰: {prog}")
    print(f"📏 트리 깊이: {calc_tree_depth(prog)}")
    print(f"📊 PnL: {info.get('pnl', 'N/A'):.4f}")
    print(f"🔄 거래 수: {info.get('trades', 'N/A')}")
    print("="*80)
    
    # 프로그램 정보와 함께 저장
    program_data = {
        "tokens": prog,
        "formula": formula,
        "reward": reward,
        "tree_depth": calc_tree_depth(prog),
        "info": info
    }
    
    with open(outdir / "best_program.json", "w") as f:
        json.dump(program_data, f, indent=2)
    with open(outdir / "best_program.txt", "w") as f:
        f.write(formula)

    print(f"\n✅ 결과 저장 → {outdir}")
    print(f"   - best_program.json (상세 정보)")
    print(f"   - best_program.txt (수식만)")

if __name__ == "__main__":
    main()
