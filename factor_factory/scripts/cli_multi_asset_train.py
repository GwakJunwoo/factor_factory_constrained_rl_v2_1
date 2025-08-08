#!/usr/bin/env python3
"""
다종목 팩터 모델 학습 CLI

크로스 섹션 정규화를 통한 진정한 팩터 모델 학습
MCTS와 PPO 모두 지원
"""

import argparse
import logging
import sys
from pathlib import Path
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 로깅 설정 - Windows 호환 UTF-8 인코딩
import locale
import os

# Windows에서 안전한 인코딩 설정
try:
    if os.name == 'nt':  # Windows
        # Windows에서는 기본 로케일 유지하고 인코딩만 처리
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        import codecs
        import sys
        # stdout/stderr을 UTF-8로 설정
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    else:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except Exception as e:
    pass  # 로케일 설정 실패해도 계속 진행

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_asset_training.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)

# Windows 콘솔에서 유니코드 문자 출력 방지
import builtins
original_print = builtins.print

def safe_print(*args, **kwargs):
    """Windows에서 안전한 출력을 위한 print 함수"""
    try:
        # 유니코드 문자를 안전한 문자로 변환
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # 문제될 수 있는 유니코드 문자들을 안전한 문자로 교체
                safe_arg = arg.replace('📊', '[CHART]').replace('🎯', '[TARGET]').replace('⚡', '[FAST]')
                safe_arg = safe_arg.replace('🔍', '[SEARCH]').replace('💡', '[IDEA]').replace('✨', '[STAR]')
                safe_args.append(safe_arg)
            else:
                safe_args.append(arg)
        return original_print(*safe_args, **kwargs)
    except UnicodeEncodeError:
        # 모든 유니코드 오류를 ASCII로 처리
        ascii_args = []
        for arg in args:
            if isinstance(arg, str):
                ascii_args.append(arg.encode('ascii', 'replace').decode('ascii'))
            else:
                ascii_args.append(str(arg))
        return original_print(*ascii_args, **kwargs)

# print 함수를 안전한 버전으로 교체
builtins.print = safe_print

from factor_factory.data import ParquetCache, DATA_ROOT
from factor_factory.rlc import RLCConfig
from factor_factory.multi_asset import (
    MultiAssetDataManager,
    CrossSectionNormalizer,
    PortfolioManager,
    VectorizedBacktest
)
from factor_factory.multi_asset.factor_env import MultiAssetFactorEnv
from factor_factory.mcts import (
    PolicyValueNetwork, 
    AlphaZeroTrainer
)
try:
    from factor_factory.pool.factor_pool import FactorPool
except ImportError:
    FactorPool = None


def run_multi_asset_backtest(args):
    """다종목 백테스트만 실행"""
    logging.info("[BACKTEST] 다종목 백테스트 모드")
    
    # 데이터 로딩
    cache = ParquetCache(DATA_ROOT)
    data_dict = {}
    
    for symbol in args.symbols:
        try:
            df = cache.load(symbol, args.interval)
            data_dict[symbol] = df
            logging.info(f"[OK] {symbol} 데이터 로딩 완료: {df.shape}")
        except Exception as e:
            logging.error(f"[ERROR] {symbol} 데이터 로딩 실패: {e}")
            return
    
    if len(data_dict) < 2:
        logging.error("[ERROR] 최소 2개 종목이 필요합니다.")
        return
    
    # 데이터 매니저 초기화
    data_manager = MultiAssetDataManager(args.symbols, args.interval)
    data_manager.data_dict = data_dict
    aligned_data = data_manager.align_data(method='inner')
    
    logging.info(f"정렬된 데이터: {aligned_data.shape}")
    
    # 간단한 팩터로 테스트 (RSI14만 사용)
    test_program = [11]  # RSI14만 사용하는 단순한 팩터
    
    # 팩터 계산
    factor_matrix = data_manager.calculate_factor_matrix(test_program)
    price_matrix = data_manager.get_price_matrix('close')
    
    logging.info(f"팩터 행렬: {factor_matrix.shape}")
    
    # 백테스트 실행
    backtest = VectorizedBacktest(
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        max_position_pct=args.max_position_pct,
        rebalance_frequency=args.rebalance_freq
    )
    
    results = backtest.run_backtest(
        factor_matrix=factor_matrix,
        price_matrix=price_matrix,
        normalizer_method=args.normalizer,
        market_neutral=args.market_neutral,
        target_long_ratio=args.long_ratio,
        target_short_ratio=args.short_ratio
    )
    
    # 결과 출력
    metrics = results['performance_metrics']
    logging.info("백테스트 결과:")
    logging.info(f"  총 수익률: {metrics['total_return']:.2%}")
    logging.info(f"  연환산 수익률: {metrics['annualized_return']:.2%}")
    logging.info(f"  변동성: {metrics['volatility']:.2%}")
    logging.info(f"  샤프 비율: {metrics['sharpe_ratio']:.3f}")
    logging.info(f"  최대 낙폭: {metrics['max_drawdown']:.2%}")
    logging.info(f"  롱 비율: {metrics['long_ratio']:.2%}")
    logging.info(f"  숏 비율: {metrics['short_ratio']:.2%}")
    
    # 결과 저장
    if args.save_dir:
        backtest.save_results(args.save_dir)
        logging.info(f"결과 저장: {args.save_dir}")


def run_multi_asset_mcts_training(args):
    """다종목 MCTS 학습 실행"""
    logging.info("[MCTS] 다종목 MCTS 학습 모드")
    
    # 데이터 로딩 (단일 종목 데이터를 다종목으로 시뮬레이션)
    cache = ParquetCache(DATA_ROOT)
    
    try:
        # 실제 데이터 로딩 (현재는 BTCUSDT만 사용 가능)
        btc_df = cache.load('BTCUSDT', args.interval)
        logging.info(f"[OK] BTCUSDT 데이터 로딩: {btc_df.shape}")
        
        # 시뮬레이션 데이터 생성
        data_dict = {'BTCUSDT': btc_df}
        
        # 추가 종목 시뮬레이션 (노이즈 추가)
        import numpy as np
        np.random.seed(42)
        
        for i, symbol in enumerate(args.symbols[1:], 1):
            sim_df = btc_df.copy()
            
            # 가격에 노이즈 추가
            noise_factor = 0.02 + i * 0.01
            price_multiplier = 0.1 + i * 0.05  # 가격 스케일 다양화
            
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(1, noise_factor, len(sim_df))
                sim_df[col] = sim_df[col] * noise * price_multiplier
            
            # 거래량도 조정
            volume_noise = np.random.normal(1, 0.1, len(sim_df))
            sim_df['volume'] = sim_df['volume'] * volume_noise * (0.5 + i * 0.2)
            
            data_dict[symbol] = sim_df
            logging.info(f"[OK] {symbol} 시뮬레이션 데이터 생성")
        
    except Exception as e:
        logging.error(f"[ERROR] 데이터 로딩 실패: {e}")
        return
    
    # RLC 설정
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
    
    # 다종목 환경 생성
    env = MultiAssetFactorEnv(
        symbols=list(data_dict.keys()),
        data_dict=data_dict,
        config=rlc_config,
        normalizer_method=args.normalizer,
        market_neutral=args.market_neutral,
        target_long_ratio=args.long_ratio,
        target_short_ratio=args.short_ratio
    )
    
    # 신경망 초기화
    network = PolicyValueNetwork(
        input_dim=23,  # 기존과 동일하게 유지
        hidden_dims=args.hidden_dims,
        action_dim=25,
        dropout_rate=0.1
    )
    
    # Factor Pool 초기화
    factor_pool = None
    if FactorPool:
        factor_pool = FactorPool(args.factor_pool_dir)
    
    # AlphaZero 트레이너 초기화
    trainer = AlphaZeroTrainer(
        env=env,
        network=network,
        factor_pool=factor_pool,
        mcts_simulations=args.mcts_simulations,
        c_puct=args.c_puct,
        episodes_per_iteration=args.episodes_per_iter,
        training_batch_size=args.batch_size,
        training_epochs=args.training_epochs,
        evaluation_episodes=args.eval_episodes,
        evaluation_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.save_dir
    )
    
    # 학습 실행
    logging.info(f"[START] 다종목 MCTS 학습 시작")
    logging.info(f"  종목 수: {len(data_dict)}")
    logging.info(f"  정규화: {args.normalizer}")
    logging.info(f"  마켓뉴트럴: {args.market_neutral}")
    logging.info(f"  반복 횟수: {args.iterations}")
    
    try:
        trainer.train(num_iterations=args.iterations)
        
        # 최종 통계
        env_stats = env.get_statistics()
        logging.info("[COMPLETE] 학습 완료! 최종 통계:")
        logging.info(f"  최고 성능: {env_stats.get('best_performance', 0):.4f}")
        logging.info(f"  평균 보상: {env_stats.get('avg_reward', 0):.4f}")
        logging.info(f"  평균 수익률: {env_stats.get('avg_return', 0):.2%}")
        logging.info(f"  평균 샤프: {env_stats.get('avg_sharpe', 0):.3f}")
        
        # 발견된 최고 전략 출력
        best_strategy = env.get_best_strategy()
        if best_strategy:
            print("\n" + "="*60)
            print("🏆 발견된 최고 전략")
            print("="*60)
            logging.info("[STRATEGY] 발견된 최고 전략:")
            logging.info(f"  프로그램 토큰: {best_strategy.get('program', 'N/A')}")
            
            # 프로그램을 인간이 읽기 쉬운 형태로 변환
            try:
                from factor_factory.rlc.compiler import RLCCompiler
                compiler = RLCCompiler()
                human_readable = compiler.decompile_program(best_strategy.get('program', []))
                print(f"📊 팩터 공식: {human_readable}")
                logging.info(f"    - 읽기 쉬운 형태: {human_readable}")
            except:
                print(f"📊 팩터 공식: Program_{len(best_strategy.get('program', []))}_tokens")
            
            metrics = best_strategy.get('metrics', {})
            print(f"🎯 성능 지표:")
            print(f"    샤프 비율: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"    총 수익률: {metrics.get('total_return', 0):.2%}")
            print(f"    연환산 수익률: {metrics.get('annualized_return', 0):.2%}")
            print(f"    변동성: {metrics.get('volatility', 0):.2%}")
            print(f"    최대 낙폭: {metrics.get('max_drawdown', 0):.2%}")
            print(f"    보상 점수: {best_strategy.get('reward', 0):.4f}")
            
            logging.info(f"  성능 지표:")
            logging.info(f"    - 샤프 비율: {metrics.get('sharpe_ratio', 0):.4f}")
            logging.info(f"    - 총 수익률: {metrics.get('total_return', 0):.2%}")
            logging.info(f"    - 연환산 수익률: {metrics.get('annualized_return', 0):.2%}")
            logging.info(f"    - 변동성: {metrics.get('volatility', 0):.2%}")
            logging.info(f"    - 최대 낙폭: {metrics.get('max_drawdown', 0):.2%}")
        else:
            print("\n❌ 유효한 전략을 발견하지 못했습니다.")
        
        # Factor Pool 통계
        if factor_pool:
            pool_stats = factor_pool.get_statistics()
            print(f"\n📈 Factor Pool 통계:")
            print(f"  발견된 팩터 수: {len(trainer.discovered_factors)}")
            print(f"  Factor Pool: {pool_stats}")
            logging.info(f"  발견된 팩터: {len(trainer.discovered_factors)}")
            logging.info(f"  Factor Pool: {pool_stats}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        logging.info("사용자에 의해 학습이 중단되었습니다.")
        trainer._save_checkpoint()
    except Exception as e:
        logging.error(f"학습 중 오류: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="다종목 팩터 모델 학습 및 백테스트"
    )
    
    # 모드 선택
    parser.add_argument("--mode", choices=['backtest', 'mcts'], default='backtest',
                       help="실행 모드")
    
    # 종목 설정
    parser.add_argument("--symbols", nargs='+', 
                       default=['BTCUSDT', 'ETHUSDT_SIM', 'ADAUSDT_SIM'],
                       help="종목 리스트")
    parser.add_argument("--interval", default="1h", help="시간 간격")
    
    # 크로스 섹션 설정
    parser.add_argument("--normalizer", 
                       choices=['z_score', 'rank', 'percentile', 'mad'],
                       default='z_score', help="정규화 방법")
    parser.add_argument("--market-neutral", action='store_true', default=True,
                       help="마켓 뉴트럴 전략 사용")
    parser.add_argument("--long-ratio", type=float, default=0.3,
                       help="롱 포지션 목표 비율")
    parser.add_argument("--short-ratio", type=float, default=0.3,
                       help="숏 포지션 목표 비율")
    
    # 포트폴리오 설정
    parser.add_argument("--initial-capital", type=float, default=100000,
                       help="초기 자본금")
    parser.add_argument("--max-position-pct", type=float, default=0.2,
                       help="종목별 최대 포지션 비율")
    parser.add_argument("--rebalance-freq", choices=['D', 'W', 'M'], default='D',
                       help="리밸런싱 빈도")
    
    # 거래 비용
    parser.add_argument("--commission", type=float, default=0.0008,
                       help="거래 수수료")
    parser.add_argument("--slippage", type=float, default=0.0015,
                       help="슬리피지")
    parser.add_argument("--leverage", type=float, default=1.0,
                       help="레버리지")
    
    # MCTS 설정 (MCTS 모드일 때만 사용)
    parser.add_argument("--iterations", type=int, default=100,
                       help="MCTS 학습 반복 횟수")
    parser.add_argument("--episodes-per-iter", type=int, default=50,
                       help="반복당 에피소드 수")
    parser.add_argument("--mcts-simulations", type=int, default=400,
                       help="MCTS 시뮬레이션 횟수")
    parser.add_argument("--c-puct", type=float, default=1.0,
                       help="UCB 탐색 상수")
    
    # 신경망 설정
    parser.add_argument("--hidden-dims", type=int, nargs='+', 
                       default=[256, 256, 128],
                       help="신경망 은닉층 크기")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="학습률")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="학습 배치 크기")
    parser.add_argument("--training-epochs", type=int, default=10,
                       help="신경망 학습 에포크")
    
    # 평가 설정
    parser.add_argument("--eval-episodes", type=int, default=30,
                       help="평가 에피소드 수")
    parser.add_argument("--eval-interval", type=int, default=10,
                       help="평가 주기")
    
    # 환경 설정
    parser.add_argument("--max-len", type=int, default=21,
                       help="최대 프로그램 길이")
    parser.add_argument("--eval-stride", type=int, default=5,
                       help="평가 스트라이드")
    parser.add_argument("--max-eval-bars", type=int, default=5000,
                       help="최대 평가 바 수")
    parser.add_argument("--long-threshold", type=float, default=1.5,
                       help="롱 포지션 임계값")
    parser.add_argument("--short-threshold", type=float, default=-1.5,
                       help="숏 포지션 임계값")
    parser.add_argument("--rolling-window", type=int, default=252,
                       help="롤링 윈도우")
    
    # 저장 설정
    parser.add_argument("--save-dir", default="multi_asset_results",
                       help="결과 저장 디렉토리")
    parser.add_argument("--save-interval", type=int, default=20,
                       help="저장 주기")
    parser.add_argument("--factor-pool-dir", default="multi_asset_factor_pool",
                       help="Factor Pool 디렉토리")
    
    args = parser.parse_args()
    
    # 실행
    if args.mode == 'backtest':
        run_multi_asset_backtest(args)
    elif args.mode == 'mcts':
        run_multi_asset_mcts_training(args)
    else:
        logging.error(f"지원하지 않는 모드: {args.mode}")


if __name__ == "__main__":
    main()
