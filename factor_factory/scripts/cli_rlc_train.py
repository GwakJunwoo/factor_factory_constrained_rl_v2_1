from __future__ import annotations
import argparse
import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import ProgramEnv, RLCConfig, clear_program_cache
from factor_factory.rlc.callback import SaveBestProgramCallback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train PPO on Realistic ProgramEnv (미래 정보 누출 방지)")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save", default="models/ppo_program.zip")
    parser.add_argument("--eval_stride", type=int, default=2)
    parser.add_argument("--max_eval_bars", type=int, default=20_000)
    parser.add_argument("--long_threshold", type=float, default=1.5)
    parser.add_argument("--short_threshold", type=float, default=-1.5)
    parser.add_argument("--rolling_window", type=int, default=252)
    # 추가 현실적 매개변수
    parser.add_argument("--commission", type=float, default=0.0008, help="거래 수수료")
    parser.add_argument("--slippage", type=float, default=0.0015, help="슬리피지")
    parser.add_argument("--signal_delay", type=int, default=1, help="신호 지연")
    parser.add_argument("--execution_delay", type=int, default=1, help="체결 지연")
    parser.add_argument("--rebalance_freq", default="D", help="리밸런싱 빈도")
    parser.add_argument("--validate_signals", action="store_true", help="신호 검증 활성화")
    args = parser.parse_args(argv)

    # 캐시 초기화
    clear_program_cache()
    logging.info("프로그램 캐시 초기화 완료")

    # ── 데이터 로드 ────────────────────────────────────────────
    logging.info(f"데이터 로딩: {args.symbol}_{args.interval}")
    df = ParquetCache(DATA_ROOT).load(args.symbol, args.interval)
    logging.info(f"데이터 크기: {df.shape}, 기간: {df.index[0]} ~ {df.index[-1]}")

    # ── 환경 설정 (현실적 거래 조건) ──────────────────────────────────────────
    cfg = RLCConfig(
        eval_stride=args.eval_stride, 
        max_eval_bars=args.max_eval_bars,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        rolling_window=args.rolling_window,
        # 현실적 거래 조건
        commission=args.commission,
        slippage=args.slippage,
        signal_delay=args.signal_delay,
        execution_delay=args.execution_delay,
        rebalance_frequency=args.rebalance_freq
    )
    env = DummyVecEnv([lambda: ProgramEnv(df, cfg)])
    
    logging.info(f"환경 설정 완료 (현실적 조건):")
    logging.info(f"  - 수수료: {cfg.commission:.2%}, 슬리피지: {cfg.slippage:.2%}")
    logging.info(f"  - 신호 지연: {cfg.signal_delay}, 체결 지연: {cfg.execution_delay}")
    logging.info(f"  - 리밸런싱: {cfg.rebalance_frequency}")
    
    if args.validate_signals:
        logging.info("  - 신호 검증 활성화: 미래 정보 누출 감지")
    else:
        logging.info("  - 신호 검증 비활성화")

    # ── 모델 & 콜백 ────────────────────────────────────────────
    save_dir = Path(args.save).parent
    callback = SaveBestProgramCallback(str(save_dir), verbose=1)

    logging.info("PPO 모델 초기화")
    model = PPO("MlpPolicy", env, verbose=1)
    
    logging.info(f"학습 시작: {args.timesteps:,} timesteps")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # ── 저장 ──────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.save)
    logging.info(f"✅ PPO 모델 저장 → {args.save}")
    
    # 환경에서 성능 통계 출력
    if hasattr(env.envs[0], 'total_programs_evaluated'):
        logging.info(f"학습 통계:")
        logging.info(f"  - 총 평가된 프로그램: {env.envs[0].total_programs_evaluated:,}")
        logging.info(f"  - 캐시 히트: {env.envs[0].cache_hits:,}")
        if env.envs[0].total_programs_evaluated > 0:
            hit_rate = env.envs[0].cache_hits / env.envs[0].total_programs_evaluated
            logging.info(f"  - 캐시 히트율: {hit_rate:.2%}")


if __name__ == "__main__":
    main()
