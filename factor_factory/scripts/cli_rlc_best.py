
from __future__ import annotations
import argparse, json
from pathlib import Path
from stable_baselines3 import PPO
from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import ProgramEnv, RLCConfig, tokens_to_infix

def rollout_once(model:PPO, env:ProgramEnv, deterministic:bool=True):
    obs, _ = env.reset()
    done=False; info_last={}; prog=None; rew=0.0
    while not done:
        act, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, done, trunc, info = env.step(int(act))
        if done:
            info_last = info
            prog = info.get("program")
            break
    return rew, prog, info_last

def main(argv: list[str] | None=None):
    p=argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_program.zip")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--tries", type=int, default=256)
    p.add_argument("--outdir", default="rlc_out")
    p.add_argument("--eval_stride", type=int, default=2)
    p.add_argument("--max_eval_bars", type=int, default=20000)
    args=p.parse_args(argv)

    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)
    env = ProgramEnv(df, RLCConfig(eval_stride=args.eval_stride, max_eval_bars=args.max_eval_bars))
    model = PPO.load(args.model)

    best = (-1e9, None, {})
    for _ in range(args.tries):
        rew, prog, info = rollout_once(model, env, deterministic=False)
        if prog is not None and rew > best[0]:
            best = (rew, prog, info)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if best[1] is None:
        print("No valid program found.")
        return
    prog = best[1]
    with open(outdir/"best_program.json","w") as f:
        json.dump({"tokens": prog}, f, indent=2)
    with open(outdir/"best_program.txt","w") as f:
        f.write(tokens_to_infix(prog))
    print("Saved best program to", outdir)

if __name__ == "__main__":
    main()
