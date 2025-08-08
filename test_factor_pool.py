#!/usr/bin/env python3
"""
Factor Pool 시스템 테스트 스크립트
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from factor_factory.pool import FactorPool, FactorAnalyzer, FactorVisualizer

def test_factor_pool():
    """Factor Pool 시스템 테스트"""
    
    print("🧪 Factor Pool 시스템 테스트 시작")
    print("=" * 50)
    
    # Factor Pool 생성
    pool = FactorPool("test_factor_pool")
    
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', periods=2000, freq='H')
    
    # 여러 가상 팩터 생성
    factor_scenarios = [
        ("High Return High Risk", np.random.normal(0.002, 0.015, 2000)),
        ("Stable Low Risk", np.random.normal(0.0008, 0.005, 2000)),
        ("Volatile Strategy", np.random.normal(0.001, 0.025, 2000)),
        ("Consistent Winner", np.random.normal(0.0012, 0.008, 2000)),
        ("Boom Bust Cycle", np.concatenate([
            np.random.normal(0.005, 0.01, 1000),
            np.random.normal(-0.002, 0.01, 1000)
        ]))
    ]
    
    saved_ids = []
    
    for i, (name, pnl_pattern) in enumerate(factor_scenarios):
        print(f"\n📈 {name} 팩터 생성 중...")
        
        # PnL 시계열
        pnl = pd.Series(pnl_pattern, index=dates)
        
        # 수익률 곡선
        equity = (1 + pnl).cumprod()
        
        # 랜덤 신호 (실제로는 팩터 공식에서 생성)
        signal = pd.Series(np.random.normal(0, 1.5, 2000), index=dates)
        
        # 토큰 (가상)
        tokens = list(np.random.randint(1, 20, size=np.random.randint(5, 15)))
        formula = f"FACTOR_{i+1}_{name.replace(' ', '_')}"
        
        # 보상 정보 (가상)
        reward_info = {
            'total_reward': np.random.uniform(-0.5, 2.0),
            'components': {
                'main_reward': pnl.sum() * 10,
                'complexity_penalty': -len(tokens) * 0.01,
                'stability_bonus': 0.1
            },
            'future_leak': False,
            'validation': {'score': np.random.uniform(0.7, 1.0)}
        }
        
        # 팩터 저장
        factor_id = pool.add_factor(
            tokens=tokens,
            formula=formula,
            pnl=pnl,
            equity=equity,
            signal=signal,
            reward_info=reward_info,
            model_version=f"test_v{i//2 + 1}",
            training_episode=i * 100
        )
        
        saved_ids.append(factor_id)
    
    print(f"\n✅ {len(saved_ids)}개 팩터 저장 완료!")
    
    # 통계 확인
    stats = pool.get_statistics()
    print(f"\n📊 Pool 통계:")
    print(f"  총 팩터 수: {stats['total_factors']}")
    print(f"  평균 수익률: {stats['avg_return']:.2%}")
    print(f"  최고 수익률: {stats['best_return']:.2%}")
    
    # 상위 팩터 조회
    top_factors = pool.get_top_factors(3)
    print(f"\n🏆 상위 3개 팩터:")
    for i, factor in enumerate(top_factors, 1):
        print(f"  {i}. {factor.factor_id[:12]} - Return: {factor.total_return:.2%}, Sharpe: {factor.sharpe_ratio:.3f}")
    
    # 분석 테스트
    print(f"\n🔍 분석 시스템 테스트...")
    analyzer = FactorAnalyzer(pool)
    
    if top_factors:
        best_factor = top_factors[0]
        scorecard = analyzer.create_factor_scorecard(best_factor.factor_id)
        print(f"  최고 팩터 점수: {scorecard['total_score']:.1f}/10 (등급: {scorecard['grade']})")
        
        if len(top_factors) >= 2:
            comparison = analyzer.compare_factors([f.factor_id for f in top_factors[:2]])
            print(f"  상위 2개 팩터 비교 완료")
    
    # 시각화 테스트
    print(f"\n📊 시각화 시스템 테스트...")
    visualizer = FactorVisualizer(pool)
    
    try:
        # 간단한 차트 생성 (저장 없이)
        import matplotlib.pyplot as plt
        plt.ioff()  # 화면 출력 비활성화
        
        visualizer.plot_performance_comparison(n_top=3, save_path="test_performance.png")
        visualizer.plot_risk_return_scatter(n_top=5, save_path="test_risk_return.png")
        
        print(f"  ✅ 시각화 차트 생성 완료")
        
        # 파일 정리
        for file in ["test_performance.png", "test_risk_return.png"]:
            if os.path.exists(file):
                os.remove(file)
        
    except Exception as e:
        print(f"  ⚠️ 시각화 테스트 건너뜀: {e}")
    
    print(f"\n🎉 Factor Pool 시스템 테스트 완료!")
    print(f"실제 사용 시 env.py에서 자동으로 상위 팩터들이 저장됩니다.")
    
    return pool, saved_ids

def demo_factor_management():
    """팩터 관리 데모"""
    
    print(f"\n🎮 팩터 관리 데모")
    print("=" * 30)
    
    pool, factor_ids = test_factor_pool()
    
    print(f"\n명령어 사용 예시:")
    print(f"  python factor_pool_manager.py list --n 10")
    print(f"  python factor_pool_manager.py analyze {factor_ids[0][:16]}")
    print(f"  python factor_pool_manager.py compare {factor_ids[0][:16]},{factor_ids[1][:16]}")
    print(f"  python factor_pool_manager.py report")
    print(f"  python factor_pool_manager.py visualize")
    
    print(f"\n미래 기능 (슈퍼팩터):")
    print(f"  - 상위 팩터들을 결합한 앙상블 전략")
    print(f"  - 팩터 간 상관관계 기반 포트폴리오 구성")
    print(f"  - 동적 가중치 조정 시스템")
    print(f"  - 리스크 패리티 팩터 결합")

if __name__ == "__main__":
    demo_factor_management()
