#!/usr/bin/env python3
"""
MultiAssetDataManager 테스트 스크립트

다종목 데이터 매니저의 기본 기능을 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_factory.multi_asset import MultiAssetDataManager
from factor_factory.rlc.grammar import TOKENS


def test_single_asset():
    """단일 종목으로 기본 기능 테스트"""
    print("🧪 단일 종목 테스트 시작")
    
    # 데이터 매니저 생성
    symbols = ['BTCUSDT']
    manager = MultiAssetDataManager(symbols, interval='1h')
    
    # 데이터 로딩
    data_dict = manager.load_data()
    print(f"데이터 로딩 결과: {list(data_dict.keys())}")
    
    # 데이터 정렬
    aligned_data = manager.align_data(method='inner')
    print(f"정렬된 데이터 크기: {aligned_data.shape}")
    print(f"컬럼: {aligned_data.columns.tolist()}")
    
    # 가격 행렬 추출
    close_matrix = manager.get_price_matrix('close')
    print(f"종가 행렬 크기: {close_matrix.shape}")
    print(f"최근 5개 종가:\n{close_matrix.tail()}")
    
    # 수익률 행렬 계산
    returns_matrix = manager.get_returns_matrix(periods=1)
    print(f"수익률 행렬 크기: {returns_matrix.shape}")
    print(f"수익률 통계:\n{returns_matrix.describe()}")
    
    # 간단한 팩터 계산 (RSI14)
    simple_program = [11]  # RSI14 토큰
    factor_matrix = manager.calculate_factor_matrix(simple_program)
    print(f"팩터 행렬 크기: {factor_matrix.shape}")
    print(f"팩터 통계:\n{factor_matrix.describe()}")
    
    # 데이터 품질 검증
    validation_results = manager.validate_data_quality()
    print(f"데이터 품질 검증 결과:\n{validation_results}")
    
    # 통계 정보
    stats = manager.get_statistics()
    print(f"데이터 매니저 통계:\n{stats}")
    
    print("✅ 단일 종목 테스트 완료")
    return manager


def test_multi_asset_simulation():
    """다종목 시뮬레이션 테스트 (같은 데이터로 여러 종목 시뮬레이션)"""
    print("\n🧪 다종목 시뮬레이션 테스트 시작")
    
    # 임시로 BTCUSDT 데이터를 다른 이름으로 사용
    symbols = ['BTCUSDT', 'ETHUSDT_SIM', 'ADAUSDT_SIM']
    
    try:
        manager = MultiAssetDataManager(['BTCUSDT'], interval='1h')
        
        # 실제 데이터 로딩
        data_dict = manager.load_data()
        btc_data = data_dict['BTCUSDT'].copy()
        
        # 시뮬레이션 데이터 생성 (노이즈 추가)
        import numpy as np
        np.random.seed(42)
        
        # ETH 시뮬레이션 (BTC 대비 약간 다른 패턴)
        eth_data = btc_data.copy()
        noise_factor = 0.02
        for col in ['open', 'high', 'low', 'close']:
            noise = np.random.normal(1, noise_factor, len(eth_data))
            eth_data[col] = eth_data[col] * noise
        
        # ADA 시뮬레이션 (더 높은 변동성)
        ada_data = btc_data.copy()
        noise_factor = 0.05
        for col in ['open', 'high', 'low', 'close']:
            noise = np.random.normal(1, noise_factor, len(ada_data))
            ada_data[col] = ada_data[col] * noise * 0.1  # 가격 스케일 조정
        
        # 매니저에 시뮬레이션 데이터 추가
        manager.symbols = symbols
        manager.data_dict = {
            'BTCUSDT': btc_data,
            'ETHUSDT_SIM': eth_data,
            'ADAUSDT_SIM': ada_data
        }
        
        # 데이터 정렬
        aligned_data = manager.align_data(method='inner')
        print(f"다종목 정렬된 데이터 크기: {aligned_data.shape}")
        
        # 가격 행렬
        close_matrix = manager.get_price_matrix('close')
        print(f"종가 행렬:\n{close_matrix.tail()}")
        
        # 팩터 계산 (SMA20 - SMA10)
        sma_program = [10, 9, 1]  # SMA20 - SMA10
        factor_matrix = manager.calculate_factor_matrix(sma_program)
        print(f"팩터 행렬 통계:\n{factor_matrix.describe()}")
        
        # 상관관계 분석
        factor_corr = factor_matrix.corr()
        print(f"종목간 팩터 상관관계:\n{factor_corr}")
        
        print("✅ 다종목 시뮬레이션 테스트 완료")
        return manager
        
    except Exception as e:
        print(f"❌ 다종목 시뮬레이션 테스트 실패: {e}")
        return None


def main():
    """메인 테스트 실행"""
    print("🚀 MultiAssetDataManager 테스트 시작\n")
    
    # 단일 종목 테스트
    single_manager = test_single_asset()
    
    # 다종목 시뮬레이션 테스트
    multi_manager = test_multi_asset_simulation()
    
    print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()
