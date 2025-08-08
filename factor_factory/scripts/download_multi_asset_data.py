#!/usr/bin/env python3
"""
다종목 데이터 다운로드 모듈

Binance API를 통해 여러 종목의 OHLCV 데이터를 다운로드하고
기존 데이터 저장 형식에 맞게 저장하는 모듈
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import yaml
from datetime import datetime, timedelta
import argparse
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class MultiAssetDataDownloader:
    """다종목 데이터 다운로더"""
    
    def __init__(self, data_cache_dir: str = "data_cache"):
        """
        Args:
            data_cache_dir: 데이터 저장 디렉토리
        """
        self.data_cache_dir = Path(data_cache_dir)
        self.data_cache_dir.mkdir(exist_ok=True)
        
        # Binance API 엔드포인트
        self.base_url = "https://api.binance.com"
        
        # 지원하는 시간 간격
        self.supported_intervals = {
            '1m': '1m',
            '3m': '3m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        
        # 인기 암호화폐 심볼들
        self.popular_symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
            'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'EOSUSDT', 'TRXUSDT',
            'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT',
            'ATOMUSDT', 'LUNAUSDT', 'FTMUSDT', 'ALGOUSDT', 'VETUSDT'
        ]
    
    def get_available_symbols(self) -> List[str]:
        """Binance에서 사용 가능한 USDT 페어 심볼 목록 조회"""
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            usdt_symbols = []
            
            for symbol_info in data['symbols']:
                symbol = symbol_info['symbol']
                if (symbol.endswith('USDT') and 
                    symbol_info['status'] == 'TRADING' and
                    symbol_info['quoteAsset'] == 'USDT'):
                    usdt_symbols.append(symbol)
            
            logging.info(f"📊 사용 가능한 USDT 페어: {len(usdt_symbols)}개")
            return sorted(usdt_symbols)
            
        except Exception as e:
            logging.error(f"❌ 심볼 목록 조회 실패: {e}")
            return self.popular_symbols
    
    def download_symbol_data(self, 
                           symbol: str, 
                           interval: str = '1h',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        특정 심볼의 OHLCV 데이터 다운로드
        
        Args:
            symbol: 심볼 (예: 'ETHUSDT')
            interval: 시간 간격
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            limit: 한 번에 가져올 최대 캔들 수
            
        Returns:
            DataFrame with OHLCV data
        """
        if interval not in self.supported_intervals:
            logging.error(f"❌ 지원하지 않는 시간 간격: {interval}")
            return None
        
        try:
            logging.info(f"📥 {symbol} 데이터 다운로드 시작 ({interval})")
            
            # 파라미터 설정
            params = {
                'symbol': symbol,
                'interval': self.supported_intervals[interval],
                'limit': limit
            }
            
            # 날짜 범위 설정
            if start_date:
                start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
                params['startTime'] = start_timestamp
            
            if end_date:
                end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
                params['endTime'] = end_timestamp
            
            all_data = []
            
            # 데이터가 충분할 때까지 반복 다운로드
            while True:
                url = f"{self.base_url}/api/v3/klines"
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # 다음 배치를 위한 시작 시간 업데이트
                last_timestamp = data[-1][0]
                params['startTime'] = last_timestamp + 1
                
                # 종료 조건 확인
                if end_date:
                    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
                    if last_timestamp >= end_timestamp:
                        break
                
                # 레이트 리미트 방지
                time.sleep(0.1)
                
                # 진행 상황 출력
                if len(all_data) % 5000 == 0:
                    logging.info(f"    다운로드 진행: {len(all_data)}개 캔들")
            
            if not all_data:
                logging.warning(f"⚠️ {symbol}: 데이터가 없습니다")
                return None
            
            # DataFrame 생성
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # 필요한 컬럼만 선택하고 타입 변환
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # 타임스탬프를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 가격 및 거래량을 숫자형으로 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 중복 제거 및 정렬
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            
            logging.info(f"✅ {symbol}: {len(df)}개 캔들 다운로드 완료")
            logging.info(f"    기간: {df.index[0]} ~ {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logging.error(f"❌ {symbol} 다운로드 실패: {e}")
            return None
    
    def save_data(self, symbol: str, interval: str, df: pd.DataFrame):
        """데이터를 기존 형식에 맞게 저장"""
        try:
            # 파일명 생성
            filename = f"{symbol}_{interval}"
            parquet_path = self.data_cache_dir / f"{filename}.parquet"
            meta_path = self.data_cache_dir / f"{filename}.meta.yaml"
            
            # Parquet 파일 저장
            df.to_parquet(parquet_path, engine='pyarrow')
            
            # 메타데이터 생성
            meta_data = {
                'symbol': symbol,
                'interval': interval,
                'start_date': df.index[0].isoformat(),
                'end_date': df.index[-1].isoformat(),
                'total_records': len(df),
                'columns': df.columns.tolist(),
                'download_timestamp': datetime.now().isoformat(),
                'data_source': 'Binance API',
                'file_size_mb': parquet_path.stat().st_size / (1024 * 1024)
            }
            
            # 메타데이터 저장
            with open(meta_path, 'w') as f:
                yaml.dump(meta_data, f, default_flow_style=False)
            
            logging.info(f"💾 {symbol} 데이터 저장 완료")
            logging.info(f"    Parquet: {parquet_path}")
            logging.info(f"    메타데이터: {meta_path}")
            
        except Exception as e:
            logging.error(f"❌ {symbol} 저장 실패: {e}")
    
    def download_multiple_symbols(self,
                                symbols: List[str],
                                interval: str = '1h',
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                max_workers: int = 3) -> Dict[str, bool]:
        """
        여러 심볼의 데이터를 순차적으로 다운로드
        
        Args:
            symbols: 심볼 리스트
            interval: 시간 간격
            start_date: 시작 날짜
            end_date: 종료 날짜
            max_workers: 동시 다운로드 수 (API 제한으로 인해 낮게 설정)
            
        Returns:
            다운로드 결과 딕셔너리 {symbol: success}
        """
        results = {}
        
        logging.info(f"🚀 다종목 데이터 다운로드 시작")
        logging.info(f"    종목 수: {len(symbols)}")
        logging.info(f"    시간 간격: {interval}")
        logging.info(f"    기간: {start_date} ~ {end_date}")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logging.info(f"[{i}/{len(symbols)}] {symbol} 처리 중...")
                
                # 기존 파일 확인
                filename = f"{symbol}_{interval}"
                parquet_path = self.data_cache_dir / f"{filename}.parquet"
                
                if parquet_path.exists():
                    logging.info(f"    📁 기존 파일 존재: {parquet_path}")
                    choice = input(f"    {symbol} 기존 데이터를 덮어쓸까요? (y/N): ").lower()
                    if choice not in ['y', 'yes']:
                        results[symbol] = True
                        continue
                
                # 데이터 다운로드
                df = self.download_symbol_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    # 데이터 저장
                    self.save_data(symbol, interval, df)
                    results[symbol] = True
                else:
                    results[symbol] = False
                
                # API 레이트 리미트 방지
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"❌ {symbol} 처리 실패: {e}")
                results[symbol] = False
        
        # 결과 요약
        success_count = sum(results.values())
        total_count = len(symbols)
        
        logging.info(f"🎯 다운로드 완료!")
        logging.info(f"    성공: {success_count}/{total_count}")
        logging.info(f"    실패: {total_count - success_count}/{total_count}")
        
        return results
    
    def get_recommended_symbols(self, count: int = 10) -> List[str]:
        """추천 심볼 목록 반환"""
        available_symbols = self.get_available_symbols()
        
        # 인기 심볼들 중 사용 가능한 것들 우선 선택
        recommended = []
        
        for symbol in self.popular_symbols:
            if symbol in available_symbols:
                recommended.append(symbol)
            if len(recommended) >= count:
                break
        
        # 부족하면 다른 심볼들로 채움
        if len(recommended) < count:
            for symbol in available_symbols:
                if symbol not in recommended:
                    recommended.append(symbol)
                if len(recommended) >= count:
                    break
        
        return recommended[:count]


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(description="다종목 암호화폐 데이터 다운로드")
    
    parser.add_argument("--symbols", nargs='+', 
                       help="다운로드할 심볼 목록 (예: BTCUSDT ETHUSDT)")
    parser.add_argument("--interval", default="1h",
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
                       help="시간 간격")
    parser.add_argument("--start-date", 
                       help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date",
                       help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data_cache",
                       help="데이터 저장 디렉토리")
    parser.add_argument("--recommend", action="store_true",
                       help="추천 심볼 목록 표시")
    parser.add_argument("--count", type=int, default=10,
                       help="추천 심볼 개수")
    
    args = parser.parse_args()
    
    # 다운로더 초기화
    downloader = MultiAssetDataDownloader(args.data_dir)
    
    if args.recommend:
        # 추천 심볼 표시
        recommended = downloader.get_recommended_symbols(args.count)
        print("\n📊 추천 심볼 목록:")
        for i, symbol in enumerate(recommended, 1):
            print(f"  {i:2d}. {symbol}")
        
        print(f"\n💡 사용 예시:")
        print(f"python -m factor_factory.scripts.download_multi_asset_data --symbols {' '.join(recommended[:5])} --interval 1h --start-date 2023-01-01")
        return
    
    if not args.symbols:
        print("❌ 다운로드할 심볼을 지정해주세요.")
        print("💡 추천 심볼을 보려면: --recommend 옵션을 사용하세요")
        return
    
    # 다운로드 실행
    results = downloader.download_multiple_symbols(
        symbols=args.symbols,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 결과 출력
    print("\n📋 다운로드 결과:")
    for symbol, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {symbol}")


if __name__ == "__main__":
    main()
