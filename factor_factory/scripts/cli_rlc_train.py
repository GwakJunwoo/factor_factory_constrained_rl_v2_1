
from __future__ import annotations
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import ProgramEnv, RLCConfig

def main(argv: list[str] | None=None):
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--timesteps", type=int, default=150_000)  # smaller by default for faster feedback
    p.add_argument("--save", default="models/ppo_program.zip")
    p.add_argument("--eval_stride", type=int, default=2)
    p.add_argument("--max_eval_bars", type=int, default=20000)
    args = p.parse_args(argv)

    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)

    def env_fn(): 
        return ProgramEnv(df, RLCConfig(eval_stride=args.eval_stride, max_eval_bars=args.max_eval_bars))

    env = DummyVecEnv([env_fn])
    model = PPO(
        "MlpPolicy", env, verbose=1, seed=42,
        learning_rate=3e-4, clip_range=0.2, target_kl=0.03,
        n_steps=1024, batch_size=1024, n_epochs=8, ent_coef=0.01
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"Saved model: {args.save}")

if __name__ == "__main__":
    main()
