"""
실제 백테스트 데이터 로더

실제 가격 데이터와 백테스트 결과를 로드하여 실시간 차트에 사용할 수 있도록 처리
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from typing import Tuple, List, Optional, Dict


class RealDataLoader:
    """실제 백테스트 데이터 로더"""
    
    def __init__(self, data_cache_dir: str = "data_cache", result_dir: str = "rlc_out"):
        self.data_cache_dir = Path(data_cache_dir)
        self.result_dir = Path(result_dir)
        
    def load_price_data(self, symbol: str = "BTCUSDT") -> Optional[pd.DataFrame]:
        """가격 데이터 로드"""
        try:
            # Parquet 파일에서 가격 데이터 로드
            price_file = self.data_cache_dir / f"{symbol}_1h.parquet"
            
            if price_file.exists():
                df = pd.read_parquet(price_file)
                # open_time을 datetime으로 변환하고 시간대 정보 제거
                if 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time'])
                    # 시간대 정보가 있으면 제거
                    if hasattr(df['open_time'].dtype, 'tz') and df['open_time'].dt.tz is not None:
                        df['open_time'] = df['open_time'].dt.tz_localize(None)
                    df = df.set_index('open_time')
                
                print(f"가격 데이터 로드 성공: {symbol}, {len(df)}개 행")
                return df
            else:
                print(f"가격 데이터 파일이 없습니다: {price_file}")
                return None
                
        except Exception as e:
            print(f"가격 데이터 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_signal_data(self) -> Optional[pd.DataFrame]:
        """시그널 데이터 로드"""
        try:
            signal_file = self.result_dir / "signal.csv"
            
            if signal_file.exists():
                df = pd.read_csv(signal_file)
                df['open_time'] = pd.to_datetime(df['open_time'])
                # 시간대 정보가 있으면 제거
                if hasattr(df['open_time'].dtype, 'tz') and df['open_time'].dt.tz is not None:
                    df['open_time'] = df['open_time'].dt.tz_localize(None)
                df = df.set_index('open_time')
                print(f"시그널 데이터 로드 성공: {len(df)}개 행")
                return df
            else:
                print(f"시그널 데이터 파일이 없습니다: {signal_file}")
                return None
                
        except Exception as e:
            print(f"시그널 데이터 로드 오류: {e}")
            return None
    
    def load_equity_data(self) -> Optional[pd.DataFrame]:
        """에쿼티 (누적 PNL) 데이터 로드"""
        try:
            equity_file = self.result_dir / "equity.csv"
            
            if equity_file.exists():
                df = pd.read_csv(equity_file)
                df['open_time'] = pd.to_datetime(df['open_time'])
                # 시간대 정보가 있으면 제거
                if hasattr(df['open_time'].dtype, 'tz') and df['open_time'].dt.tz is not None:
                    df['open_time'] = df['open_time'].dt.tz_localize(None)
                df = df.set_index('open_time')
                print(f"에쿼티 데이터 로드 성공: {len(df)}개 행")
                return df
            else:
                print(f"에쿼티 데이터 파일이 없습니다: {equity_file}")
                return None
                
        except Exception as e:
            print(f"에쿼티 데이터 로드 오류: {e}")
            return None
    
    def get_combined_data(self, symbol: str = "BTCUSDT", 
                         start_points: int = 0, 
                         max_points: int = 200) -> Tuple[List, List, List, List]:
        """
        실제 데이터를 결합하여 차트용 데이터 반환
        
        Returns:
            Tuple[timestamps, prices, signals, pnl] - 차트용 데이터
        """
        try:
            # 각 데이터 로드
            price_df = self.load_price_data(symbol)
            signal_df = self.load_signal_data()
            equity_df = self.load_equity_data()
            
            if price_df is None:
                return [], [], [], []
            
            # 시그널과 에쿼티가 없으면 기본값 사용
            if signal_df is None:
                # 가격 데이터 시간 범위에 맞는 빈 시그널 생성
                signal_df = pd.DataFrame(
                    index=price_df.index,
                    data={'signal': 0.0}
                )
            
            if equity_df is None:
                # 가격 데이터 시간 범위에 맞는 에쿼티 생성 (가격 변화율 기반)
                returns = price_df['close'].pct_change().fillna(0)
                cumulative_returns = (1 + returns).cumprod()
                equity_df = pd.DataFrame(
                    index=price_df.index,
                    data={'equity': cumulative_returns}
                )
            
            # 공통 시간 범위 찾기
            common_index = price_df.index.intersection(signal_df.index)
            if len(common_index) == 0:
                # 시간 범위가 다르면 가격 데이터 기준으로 정렬
                common_index = price_df.index
                
                # 시그널 데이터를 가격 데이터에 맞게 리샘플링
                signal_df = signal_df.reindex(common_index, method='ffill').fillna(0)
                equity_df = equity_df.reindex(common_index, method='ffill')
            
            # 시간대 정보 제거 (datetime timezone 비교 오류 방지)
            if hasattr(common_index, 'tz') and common_index.tz is not None:
                common_index = common_index.tz_localize(None)
            
            # 지정된 범위의 데이터 추출
            end_points = min(start_points + max_points, len(common_index))
            selected_index = common_index[start_points:end_points]
            
            if len(selected_index) == 0:
                return [], [], [], []
            
            # 데이터 추출 - 시간대 정보 제거
            timestamps = [ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts 
                         for ts in selected_index.tolist()]
            prices = price_df.loc[selected_index, 'close'].tolist()
            
            # 시그널 처리 (연속값을 이산값으로 변환)
            signal_values = signal_df.loc[selected_index, 'signal'].fillna(0).tolist()
            signals = []
            for sig in signal_values:
                if sig > 0.5:
                    signals.append(1)  # Long
                elif sig < -0.5:
                    signals.append(-1)  # Short
                else:
                    signals.append(0)  # Neutral
            
            # PNL 처리 (에쿼티를 퍼센트 변화로 변환)
            if len(equity_df.columns) > 0:
                equity_col = equity_df.columns[0]
                equity_values = equity_df.loc[selected_index, equity_col].fillna(1.0)
                # 에쿼티를 퍼센트 변화로 변환
                pnl = ((equity_values / equity_values.iloc[0] - 1) * 100).tolist()
            else:
                pnl = [0.0] * len(timestamps)
            
            print(f"실제 데이터 로드 완료:")
            print(f"- 시간 범위: {timestamps[0]} ~ {timestamps[-1]}")
            print(f"- 가격 범위: {min(prices):.2f} ~ {max(prices):.2f}")
            print(f"- 시그널 수: Long {signals.count(1)}, Short {signals.count(-1)}, Neutral {signals.count(0)}")
            print(f"- PNL 범위: {min(pnl):.2f}% ~ {max(pnl):.2f}%")
            
            return timestamps, prices, signals, pnl
            
        except Exception as e:
            print(f"데이터 결합 오류: {e}")
            return [], [], [], []
    
    def get_latest_backtest_data(self, symbol: str = "BTCUSDT", 
                                max_points: int = 100) -> Tuple[List, List, List, List]:
        """최신 백테스트 결과 데이터 반환"""
        try:
            # 가장 최근 백테스트 결과 디렉토리 찾기
            possible_dirs = ['rlc_out', 'rlc_out_v2', 'evaluation_v2', 'multi_asset_test']
            
            latest_result_dir = None
            latest_time = 0
            
            for dir_name in possible_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    # 디렉토리의 최신 수정 시간 확인
                    signal_file = dir_path / "signal.csv"
                    if signal_file.exists():
                        mtime = signal_file.stat().st_mtime
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_result_dir = dir_path
            
            if latest_result_dir:
                print(f"최신 백테스트 결과 사용: {latest_result_dir}")
                # 임시로 result_dir 변경
                original_result_dir = self.result_dir
                self.result_dir = latest_result_dir
                
                # 데이터 로드
                result = self.get_combined_data(symbol, 0, max_points)
                
                # result_dir 복원
                self.result_dir = original_result_dir
                
                return result
            else:
                print("백테스트 결과를 찾을 수 없습니다. 샘플 데이터를 생성합니다.")
                return self._generate_sample_data(symbol, max_points)
                
        except Exception as e:
            print(f"최신 백테스트 데이터 로드 오류: {e}")
            return self._generate_sample_data(symbol, max_points)
    
    def _generate_sample_data(self, symbol: str, max_points: int) -> Tuple[List, List, List, List]:
        """백테스트 결과가 없을 때 가격 데이터 기반 샘플 생성"""
        try:
            price_df = self.load_price_data(symbol)
            
            if price_df is not None and len(price_df) > 0:
                # 최근 데이터 사용
                recent_data = price_df.tail(max_points)
                
                # 시간대 정보 제거
                timestamps = recent_data.index.tolist()
                timestamps = [ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts 
                             for ts in timestamps]
                prices = recent_data['close'].tolist()
                
                # 가격 변화율 기반 간단한 시그널 생성
                price_changes = recent_data['close'].pct_change().fillna(0)
                signals = []
                for change in price_changes:
                    if change > 0.02:  # 2% 이상 상승
                        signals.append(1)  # Long
                    elif change < -0.02:  # 2% 이상 하락
                        signals.append(-1)  # Short
                    else:
                        signals.append(0)  # Neutral
                
                # 누적 수익률 계산
                cumulative_returns = (1 + price_changes).cumprod()
                pnl = ((cumulative_returns - 1) * 100).tolist()
                
                print(f"가격 데이터 기반 샘플 생성 완료 ({len(timestamps)}개 포인트)")
                return timestamps, prices, signals, pnl
            else:
                print("가격 데이터도 없습니다. 완전한 가상 데이터를 생성합니다.")
                return self._generate_virtual_data(max_points)
                
        except Exception as e:
            print(f"샘플 데이터 생성 오류: {e}")
            return self._generate_virtual_data(max_points)
    
    def _generate_virtual_data(self, max_points: int) -> Tuple[List, List, List, List]:
        """완전한 가상 데이터 생성 (마지막 수단)"""
        from datetime import timedelta
        
        # 시간 데이터
        start_time = datetime.now() - timedelta(hours=max_points)
        timestamps = [start_time + timedelta(hours=i) for i in range(max_points)]
        
        # 가격 데이터
        base_price = 45000
        trend = np.linspace(0, 2000, max_points)
        noise = np.random.normal(0, 200, max_points)
        prices = (base_price + trend + noise).tolist()
        
        # 시그널과 PNL
        signals = [np.random.choice([0, 1, -1], p=[0.7, 0.15, 0.15]) for _ in range(max_points)]
        returns = np.random.normal(0.02, 0.5, max_points)
        pnl = np.cumsum(returns).tolist()
        
        print(f"가상 데이터 생성 완료 ({max_points}개 포인트)")
        return timestamps, prices, signals, pnl


# 전역 데이터 로더 인스턴스
_data_loader = None

def get_data_loader() -> RealDataLoader:
    """전역 데이터 로더 인스턴스 반환"""
    global _data_loader
    if _data_loader is None:
        _data_loader = RealDataLoader()
    return _data_loader


def load_real_chart_data(symbol: str = "BTCUSDT", 
                        max_points: int = 200,
                        use_latest: bool = True) -> Tuple[List, List, List, List]:
    """
    실제 백테스트 데이터를 로드하여 차트용 데이터 반환
    
    Args:
        symbol: 심볼 (예: BTCUSDT)
        max_points: 최대 포인트 수
        use_latest: 최신 백테스트 결과 사용 여부
        
    Returns:
        Tuple[timestamps, prices, signals, pnl]
    """
    loader = get_data_loader()
    
    if use_latest:
        return loader.get_latest_backtest_data(symbol, max_points)
    else:
        return loader.get_combined_data(symbol, 0, max_points)
