from __future__ import annotations
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import ProgramEnv, RLCConfig
from factor_factory.rlc.callback import SaveBestProgramCallback


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train PPO on ProgramEnv (RL‑C)")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save", default="models/ppo_program.zip")
    parser.add_argument("--eval_stride", type=int, default=2)
    parser.add_argument("--max_eval_bars", type=int, default=20_000)
    args = parser.parse_args(argv)

    # ── 데이터 로드 ────────────────────────────────────────────
    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)

    # ── 환경 설정 ──────────────────────────────────────────────
    cfg = RLCConfig(eval_stride=args.eval_stride, max_eval_bars=args.max_eval_bars)
    env = DummyVecEnv([lambda: ProgramEnv(df, cfg)])

    # ── 모델 & 콜백 ────────────────────────────────────────────
    save_dir = Path(args.save).parent
    callback = SaveBestProgramCallback(str(save_dir), verbose=1)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # ── 저장 ──────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.save)
    print(f"✅ PPO 모델 저장 → {args.save}")


if __name__ == "__main__":
    main()
