#!/usr/bin/env python3
"""
MCTS 기반 Factor Discovery CLI

AlphaZero 스타일 MCTS를 사용한 팩터 발견 학습
PPO와 병렬로 실행 가능한 추가적인 강화학습 모델
"""

import argparse
import logging
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcts_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import RLCConfig
from factor_factory.mcts import (
    PolicyValueNetwork, 
    MCTSFactorEnv, 
    AlphaZeroTrainer
)
from factor_factory.pool import FactorPool


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MCTS 기반 Factor Discovery (AlphaZero 스타일)"
    )
    
    # 데이터 설정
    parser.add_argument("--symbol", required=True, help="심볼 (예: BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="시간 간격")
    
    # MCTS 학습 설정
    parser.add_argument("--iterations", type=int, default=50, 
                       help="학습 반복 횟수")
    parser.add_argument("--episodes-per-iter", type=int, default=100,
                       help="반복당 에피소드 수")
    parser.add_argument("--mcts-simulations", type=int, default=800,
                       help="MCTS 시뮬레이션 횟수")
    parser.add_argument("--c-puct", type=float, default=1.0,
                       help="UCB 탐색 상수")
    
    # 신경망 설정
    parser.add_argument("--hidden-dims", type=int, nargs='+', 
                       default=[256, 256, 128],
                       help="신경망 은닉층 크기")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="학습률")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="학습 배치 크기")
    parser.add_argument("--training-epochs", type=int, default=10,
                       help="신경망 학습 에포크")
    
    # 평가 설정
    parser.add_argument("--eval-episodes", type=int, default=50,
                       help="평가 에피소드 수")
    parser.add_argument("--eval-interval", type=int, default=5,
                       help="평가 주기")
    
    # 저장 설정
    parser.add_argument("--save-dir", default="mcts_results",
                       help="결과 저장 디렉토리")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="저장 주기")
    parser.add_argument("--factor-pool-dir", default="mcts_factor_pool",
                       help="Factor Pool 디렉토리")
    
    # 환경 설정 (기존 PPO와 동일)
    parser.add_argument("--max-len", type=int, default=21,
                       help="최대 프로그램 길이")
    parser.add_argument("--eval-stride", type=int, default=2,
                       help="평가 스트라이드")
    parser.add_argument("--max-eval-bars", type=int, default=20000,
                       help="최대 평가 바 수")
    parser.add_argument("--long-threshold", type=float, default=1.5,
                       help="롱 포지션 임계값")
    parser.add_argument("--short-threshold", type=float, default=-1.5,
                       help="숏 포지션 임계값")
    parser.add_argument("--rolling-window", type=int, default=252,
                       help="롤링 윈도우")
    
    # 현실적 거래 설정
    parser.add_argument("--commission", type=float, default=0.0008,
                       help="거래 수수료")
    parser.add_argument("--slippage", type=float, default=0.0015,
                       help="슬리피지")
    parser.add_argument("--leverage", type=float, default=1.0,
                       help="레버리지")
    
    # 기타
    parser.add_argument("--device", default="auto",
                       help="연산 장치 (auto/cpu/cuda)")
    parser.add_argument("--resume-from", type=str,
                       help="체크포인트에서 재시작")
    
    args = parser.parse_args(argv)
    
    # ── 데이터 로딩 ──────────────────────────────────────────
    logging.info("데이터 로딩 시작")
    cache = ParquetCache(DATA_ROOT)
    
    try:
        df = cache.load(args.symbol, args.interval)
        logging.info(f"데이터 로딩 완료: {args.symbol}_{args.interval}")
        logging.info(f"데이터 크기: {df.shape}, 기간: {df.index[0]} ~ {df.index[-1]}")
    except Exception as e:
        logging.error(f"데이터 로딩 실패: {e}")
        return
    
    # ── 환경 설정 ──────────────────────────────────────────
    logging.info("MCTS 환경 설정")
    
    rlc_config = RLCConfig(
        max_len=args.max_len,
        eval_stride=args.eval_stride,
        max_eval_bars=args.max_eval_bars,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        rolling_window=args.rolling_window,
        commission=args.commission,
        slippage=args.slippage,
        leverage=args.leverage
    )
    
    # MCTS 환경 생성
    mcts_env = MCTSFactorEnv(df, rlc_config)
    
    # ── 신경망 초기화 ──────────────────────────────────────
    logging.info("Policy-Value Network 초기화")
    
    network = PolicyValueNetwork(
        input_dim=23,
        hidden_dims=args.hidden_dims,
        action_dim=25,
        dropout_rate=0.1
    )
    
    logging.info(f"네트워크 구조: {args.hidden_dims}")
    
    # ── Factor Pool 초기화 ──────────────────────────────────
    logging.info("Factor Pool 초기화")
    factor_pool = FactorPool(args.factor_pool_dir)
    
    # ── AlphaZero 트레이너 초기화 ────────────────────────────
    logging.info("AlphaZero Trainer 초기화")
    
    trainer = AlphaZeroTrainer(
        env=mcts_env,
        network=network,
        factor_pool=factor_pool,
        # MCTS 설정
        mcts_simulations=args.mcts_simulations,
        c_puct=args.c_puct,
        # 학습 설정
        episodes_per_iteration=args.episodes_per_iter,
        training_batch_size=args.batch_size,
        training_epochs=args.training_epochs,
        # 평가 설정
        evaluation_episodes=args.eval_episodes,
        evaluation_interval=args.eval_interval,
        # 저장 설정
        save_interval=args.save_interval,
        checkpoint_dir=args.save_dir
    )
    
    # ── 체크포인트 복원 ─────────────────────────────────────
    if args.resume_from:
        logging.info(f"체크포인트에서 복원: {args.resume_from}")
        try:
            trainer.trainer.load_model(args.resume_from)
            logging.info("체크포인트 복원 완료")
        except Exception as e:
            logging.warning(f"체크포인트 복원 실패: {e}")
    
    # ── 학습 실행 ──────────────────────────────────────────
    logging.info(f"MCTS 기반 Factor Discovery 학습 시작")
    logging.info(f"설정:")
    logging.info(f"  - 반복 횟수: {args.iterations}")
    logging.info(f"  - 반복당 에피소드: {args.episodes_per_iter}")
    logging.info(f"  - MCTS 시뮬레이션: {args.mcts_simulations}")
    logging.info(f"  - 신경망 학습 에포크: {args.training_epochs}")
    logging.info(f"  - 결과 저장: {args.save_dir}")
    
    try:
        trainer.train(num_iterations=args.iterations)
        
        # ── 최종 통계 출력 ───────────────────────────────────
        logging.info("학습 완료! 최종 통계:")
        
        env_stats = mcts_env.get_statistics()
        logging.info(f"환경 통계: {env_stats}")
        
        if factor_pool:
            pool_stats = factor_pool.get_statistics()
            logging.info(f"Factor Pool 통계: {pool_stats}")
        
        logging.info(f"최고 성능: {trainer.best_performance:.4f}")
        logging.info(f"발견된 팩터 수: {len(trainer.discovered_factors)}")
        
    except KeyboardInterrupt:
        logging.info("사용자에 의해 학습이 중단되었습니다.")
        trainer._save_checkpoint()
    except Exception as e:
        logging.error(f"학습 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
