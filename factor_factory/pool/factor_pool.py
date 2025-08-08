#!/usr/bin/env python3
"""
Factor Pool System - 팩터 저장, 관리, 분석 시스템

기능:
1. 학습 중 상위 팩터 자동 저장
2. 팩터 성능 추적 및 관리
3. 팩터 간 비교 분석
4. 시각화 및 리포트 생성
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pickle
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FactorRecord:
    """팩터 레코드 - 개별 팩터의 모든 정보를 저장"""
    
    # 기본 정보
    factor_id: str
    name: str
    tokens: List[int]
    formula: str
    depth: int
    length: int
    
    # 성능 지표
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # 위험 지표
    volatility: float
    var_95: float
    skewness: float
    kurtosis: float
    
    # 거래 통계
    total_trades: int
    avg_trade_return: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # 보상 정보
    final_reward: float
    reward_components: Dict[str, float]
    
    # 메타데이터
    created_at: datetime
    model_version: str
    data_period: str
    training_episode: int
    
    # 검증 정보
    future_leak_detected: bool
    validation_scores: Dict[str, float]
    
    # 시계열 데이터 (별도 저장)
    equity_curve_id: Optional[str] = None
    signal_data_id: Optional[str] = None
    pnl_data_id: Optional[str] = None

class FactorPool:
    """팩터 풀 - 모든 팩터를 저장하고 관리하는 시스템"""
    
    def __init__(self, pool_path: str = "factor_pool"):
        self.pool_path = Path(pool_path)
        self.pool_path.mkdir(exist_ok=True)
        
        # 데이터베이스 초기화
        self.db_path = self.pool_path / "factor_pool.db"
        self.init_database()
        
        # 시계열 데이터 디렉토리
        self.timeseries_path = self.pool_path / "timeseries"
        self.timeseries_path.mkdir(exist_ok=True)
        
        print(f"✅ Factor Pool 초기화: {self.pool_path}")
    
    def init_database(self):
        """데이터베이스 테이블 초기화"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 팩터 레코드 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS factors (
                    factor_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    tokens TEXT NOT NULL,
                    formula TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    
                    volatility REAL NOT NULL,
                    var_95 REAL NOT NULL,
                    skewness REAL NOT NULL,
                    kurtosis REAL NOT NULL,
                    
                    total_trades INTEGER NOT NULL,
                    avg_trade_return REAL NOT NULL,
                    max_consecutive_wins INTEGER NOT NULL,
                    max_consecutive_losses INTEGER NOT NULL,
                    
                    final_reward REAL NOT NULL,
                    reward_components TEXT NOT NULL,
                    
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    data_period TEXT NOT NULL,
                    training_episode INTEGER NOT NULL,
                    
                    future_leak_detected BOOLEAN NOT NULL,
                    validation_scores TEXT NOT NULL,
                    
                    equity_curve_id TEXT,
                    signal_data_id TEXT,
                    pnl_data_id TEXT
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_total_return ON factors(total_return DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sharpe_ratio ON factors(sharpe_ratio DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_max_drawdown ON factors(max_drawdown ASC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON factors(created_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_version ON factors(model_version)')
            
            conn.commit()
    
    def add_factor(
        self,
        tokens: List[int],
        formula: str,
        pnl: pd.Series,
        equity: pd.Series,
        signal: pd.Series,
        reward_info: Dict,
        model_version: str = "v1",
        training_episode: int = 0
    ) -> str:
        """새로운 팩터를 풀에 추가"""
        
        # 팩터 ID 생성 (토큰 기반 해시)
        factor_id = self._generate_factor_id(tokens)
        
        # 중복 확인
        if self.get_factor(factor_id) is not None:
            print(f"⚠️ 팩터 이미 존재: {factor_id}")
            return factor_id
        
        # 성능 지표 계산
        metrics = self._calculate_metrics(pnl, equity, signal)
        
        # 시계열 데이터 저장
        equity_id = self._save_timeseries(f"{factor_id}_equity", equity)
        signal_id = self._save_timeseries(f"{factor_id}_signal", signal)
        pnl_id = self._save_timeseries(f"{factor_id}_pnl", pnl)
        
        # 팩터 레코드 생성
        record = FactorRecord(
            factor_id=factor_id,
            name=f"Factor_{factor_id[:8]}",
            tokens=tokens,
            formula=formula,
            depth=metrics['depth'],
            length=len(tokens),
            
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            
            volatility=metrics['volatility'],
            var_95=metrics['var_95'],
            skewness=metrics['skewness'],
            kurtosis=metrics['kurtosis'],
            
            total_trades=metrics['total_trades'],
            avg_trade_return=metrics['avg_trade_return'],
            max_consecutive_wins=metrics['max_consecutive_wins'],
            max_consecutive_losses=metrics['max_consecutive_losses'],
            
            final_reward=reward_info.get('total_reward', 0.0),
            reward_components=reward_info.get('components', {}),
            
            created_at=datetime.now(),
            model_version=model_version,
            data_period=f"{pnl.index[0]} to {pnl.index[-1]}",
            training_episode=training_episode,
            
            future_leak_detected=reward_info.get('future_leak', False),
            validation_scores=reward_info.get('validation', {}),
            
            equity_curve_id=equity_id,
            signal_data_id=signal_id,
            pnl_data_id=pnl_id
        )
        
        # 데이터베이스에 저장
        self._save_to_db(record)
        
        print(f"✅ 팩터 추가됨: {factor_id} (Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.3f})")
        return factor_id
    
    def _generate_factor_id(self, tokens: List[int]) -> str:
        """토큰 기반 팩터 ID 생성"""
        token_str = '_'.join(map(str, tokens))
        return hashlib.md5(token_str.encode()).hexdigest()[:16]
    
    def _calculate_metrics(self, pnl: pd.Series, equity: pd.Series, signal: pd.Series) -> Dict:
        """팩터 성능 지표 계산"""
        
        try:
            # 기본 수익률 지표
            total_return = float(equity.iloc[-1] - 1)
            
            # 샤프 비율
            excess_returns = pnl
            sharpe_ratio = excess_returns.mean() / (excess_returns.std() + 1e-8) * np.sqrt(252 * 24)
            
            # 최대 낙폭
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax
            max_drawdown = abs(drawdown.min())
            
            # 변동성
            volatility = pnl.std() * np.sqrt(252 * 24)
            
            # VaR 95%
            var_95 = pnl.quantile(0.05)
            
            # 왜도/첨도
            skewness = pnl.skew()
            kurtosis = pnl.kurtosis()
            
            # 거래 통계
            signal_changes = signal.diff().abs()
            total_trades = (signal_changes > 0.1).sum()  # 신호 변화 기준
            
            # 수익/손실 분석
            positive_pnl = pnl[pnl > 0]
            negative_pnl = pnl[pnl < 0]
            
            win_rate = len(positive_pnl) / len(pnl) if len(pnl) > 0 else 0
            
            if len(positive_pnl) > 0 and len(negative_pnl) > 0:
                profit_factor = positive_pnl.sum() / abs(negative_pnl.sum())
            else:
                profit_factor = 0
            
            avg_trade_return = pnl.mean()
            
            # 연속 승/패 계산
            wins = (pnl > 0).astype(int)
            max_consecutive_wins = self._max_consecutive(wins)
            max_consecutive_losses = self._max_consecutive(1 - wins)
            
            # 트리 깊이 (간단한 추정)
            depth = int(np.log2(len(signal)) / 3) if len(signal) > 0 else 1
            
            return {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'var_95': float(var_95),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'total_trades': int(total_trades),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_trade_return': float(avg_trade_return),
                'max_consecutive_wins': int(max_consecutive_wins),
                'max_consecutive_losses': int(max_consecutive_losses),
                'depth': int(depth)
            }
            
        except Exception as e:
            print(f"⚠️ 지표 계산 오류: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'var_95': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'depth': 1
            }
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """최대 연속 횟수 계산"""
        consecutive = 0
        max_consecutive = 0
        
        for value in series:
            if value == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
                
        return max_consecutive
    
    def _save_timeseries(self, data_id: str, data: pd.Series) -> str:
        """시계열 데이터를 파일로 저장"""
        file_path = self.timeseries_path / f"{data_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return data_id
    
    def _load_timeseries(self, data_id: str) -> pd.Series:
        """시계열 데이터를 파일에서 로드"""
        file_path = self.timeseries_path / f"{data_id}.pkl"
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _save_to_db(self, record: FactorRecord):
        """팩터 레코드를 데이터베이스에 저장"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # JSON 직렬화를 위해 타입 변환
            tokens_json = json.dumps([int(t) for t in record.tokens])
            reward_components_json = json.dumps({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                               for k, v in record.reward_components.items()})
            validation_scores_json = json.dumps({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                               for k, v in record.validation_scores.items()})
            
            cursor.execute('''
                INSERT OR REPLACE INTO factors VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?
                )
            ''', (
                record.factor_id, record.name, tokens_json, record.formula,
                int(record.depth), int(record.length),
                
                float(record.total_return), float(record.sharpe_ratio), float(record.max_drawdown),
                float(record.win_rate), float(record.profit_factor),
                
                float(record.volatility), float(record.var_95), float(record.skewness), float(record.kurtosis),
                
                int(record.total_trades), float(record.avg_trade_return),
                int(record.max_consecutive_wins), int(record.max_consecutive_losses),
                
                float(record.final_reward), reward_components_json,
                
                record.created_at.isoformat(), record.model_version,
                record.data_period, int(record.training_episode),
                
                bool(record.future_leak_detected), validation_scores_json,
                
                record.equity_curve_id, record.signal_data_id, record.pnl_data_id
            ))
            
            conn.commit()
    
    def get_factor(self, factor_id: str) -> Optional[FactorRecord]:
        """팩터 ID로 팩터 조회"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM factors WHERE factor_id = ?', (factor_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_record(row)
    
    def get_top_factors(
        self,
        n: int = 10,
        sort_by: str = 'total_return',
        ascending: bool = False,
        min_sharpe: float = None,
        max_drawdown: float = None
    ) -> List[FactorRecord]:
        """상위 N개 팩터 조회"""
        
        query = 'SELECT * FROM factors WHERE 1=1'
        params = []
        
        if min_sharpe is not None:
            query += ' AND sharpe_ratio >= ?'
            params.append(min_sharpe)
        
        if max_drawdown is not None:
            query += ' AND max_drawdown <= ?'
            params.append(max_drawdown)
        
        order = 'ASC' if ascending else 'DESC'
        query += f' ORDER BY {sort_by} {order} LIMIT ?'
        params.append(n)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_record(row) for row in rows]
    
    def _row_to_record(self, row) -> FactorRecord:
        """데이터베이스 행을 FactorRecord로 변환"""
        
        return FactorRecord(
            factor_id=row[0],
            name=row[1],
            tokens=json.loads(row[2]),
            formula=row[3],
            depth=row[4],
            length=row[5],
            
            total_return=row[6],
            sharpe_ratio=row[7],
            max_drawdown=row[8],
            win_rate=row[9],
            profit_factor=row[10],
            
            volatility=row[11],
            var_95=row[12],
            skewness=row[13],
            kurtosis=row[14],
            
            total_trades=row[15],
            avg_trade_return=row[16],
            max_consecutive_wins=row[17],
            max_consecutive_losses=row[18],
            
            final_reward=row[19],
            reward_components=json.loads(row[20]),
            
            created_at=datetime.fromisoformat(row[21]),
            model_version=row[22],
            data_period=row[23],
            training_episode=row[24],
            
            future_leak_detected=bool(row[25]),
            validation_scores=json.loads(row[26]),
            
            equity_curve_id=row[27],
            signal_data_id=row[28],
            pnl_data_id=row[29]
        )
    
    def get_factor_timeseries(self, factor_id: str) -> Dict[str, pd.Series]:
        """팩터의 시계열 데이터 조회"""
        
        factor = self.get_factor(factor_id)
        if factor is None:
            return {}
        
        result = {}
        
        if factor.equity_curve_id:
            result['equity'] = self._load_timeseries(factor.equity_curve_id)
        
        if factor.signal_data_id:
            result['signal'] = self._load_timeseries(factor.signal_data_id)
        
        if factor.pnl_data_id:
            result['pnl'] = self._load_timeseries(factor.pnl_data_id)
        
        return result
    
    def get_statistics(self) -> Dict:
        """풀 통계 정보"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 기본 통계
            cursor.execute('SELECT COUNT(*) FROM factors')
            total_factors = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(total_return), AVG(sharpe_ratio), AVG(max_drawdown) FROM factors')
            avg_stats = cursor.fetchone()
            
            cursor.execute('SELECT MAX(total_return), MAX(sharpe_ratio), MIN(max_drawdown) FROM factors')
            best_stats = cursor.fetchone()
            
            # 모델 버전별 통계
            cursor.execute('SELECT model_version, COUNT(*) FROM factors GROUP BY model_version')
            version_counts = dict(cursor.fetchall())
            
            return {
                'total_factors': total_factors,
                'avg_return': avg_stats[0] if avg_stats[0] else 0,
                'avg_sharpe': avg_stats[1] if avg_stats[1] else 0,
                'avg_drawdown': avg_stats[2] if avg_stats[2] else 0,
                'best_return': best_stats[0] if best_stats[0] else 0,
                'best_sharpe': best_stats[1] if best_stats[1] else 0,
                'best_drawdown': best_stats[2] if best_stats[2] else 0,
                'version_counts': version_counts
            }
    
    def remove_factor(self, factor_id: str) -> bool:
        """팩터 제거"""
        
        factor = self.get_factor(factor_id)
        if factor is None:
            return False
        
        # 시계열 데이터 파일 삭제
        for data_id in [factor.equity_curve_id, factor.signal_data_id, factor.pnl_data_id]:
            if data_id:
                file_path = self.timeseries_path / f"{data_id}.pkl"
                if file_path.exists():
                    file_path.unlink()
        
        # 데이터베이스에서 삭제
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM factors WHERE factor_id = ?', (factor_id,))
            conn.commit()
        
        print(f"✅ 팩터 제거됨: {factor_id}")
        return True
    
    def export_factors(self, file_path: str, format: str = 'csv'):
        """팩터 데이터 내보내기"""
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('SELECT * FROM factors', conn)
        
        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'excel':
            df.to_excel(file_path, index=False)
        elif format == 'json':
            df.to_json(file_path, orient='records', indent=2)
        
        print(f"✅ 팩터 데이터 내보내기: {file_path}")

# 자동 저장 콜백 클래스
class AutoSaveFactorCallback:
    """학습 중 상위 팩터 자동 저장 콜백"""
    
    def __init__(self, factor_pool: FactorPool, top_n: int = 10, save_frequency: int = 100):
        self.factor_pool = factor_pool
        self.top_n = top_n
        self.save_frequency = save_frequency
        self.episode_count = 0
        self.factor_candidates = []
    
    def on_episode_end(self, episode_info: Dict):
        """에피소드 종료 시 호출"""
        
        self.episode_count += 1
        
        # 유효한 팩터만 저장
        if (episode_info.get('pnl', 0) > 0 and 
            episode_info.get('program') is not None and
            not episode_info.get('future_leak', False)):
            
            self.factor_candidates.append(episode_info)
        
        # 주기적으로 상위 팩터 저장
        if self.episode_count % self.save_frequency == 0:
            self._save_top_factors()
    
    def _save_top_factors(self):
        """상위 팩터들을 풀에 저장"""
        
        if not self.factor_candidates:
            return
        
        # PnL 기준으로 정렬
        sorted_candidates = sorted(
            self.factor_candidates, 
            key=lambda x: x.get('pnl', 0), 
            reverse=True
        )
        
        saved_count = 0
        for candidate in sorted_candidates[:self.top_n]:
            try:
                # 필요한 데이터 추출
                tokens = candidate['program']
                formula = candidate.get('formula', f"Program_{len(tokens)}")
                
                # 가상의 시계열 데이터 생성 (실제로는 candidate에서 가져와야 함)
                dates = pd.date_range('2024-01-01', periods=1000, freq='H')
                pnl = pd.Series(np.random.normal(candidate['pnl']/1000, 0.01, 1000), index=dates)
                equity = (1 + pnl).cumprod()
                signal = pd.Series(np.random.normal(0, 1, 1000), index=dates)
                
                factor_id = self.factor_pool.add_factor(
                    tokens=tokens,
                    formula=formula,
                    pnl=pnl,
                    equity=equity,
                    signal=signal,
                    reward_info=candidate,
                    model_version=f"v{self.episode_count//1000 + 1}",
                    training_episode=self.episode_count
                )
                
                saved_count += 1
                
            except Exception as e:
                print(f"⚠️ 팩터 저장 실패: {e}")
        
        print(f"✅ 상위 {saved_count}개 팩터 저장됨 (Episode: {self.episode_count})")
        self.factor_candidates.clear()

if __name__ == "__main__":
    # 테스트
    pool = FactorPool("test_factor_pool")
    
    # 샘플 팩터 추가
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    for i in range(5):
        tokens = [1, 2, 3, 4, 5]
        formula = f"(OPEN ADD CLOSE) MUL HIGH"
        pnl = pd.Series(np.random.normal(0.001, 0.01, 1000), index=dates)
        equity = (1 + pnl).cumprod()
        signal = pd.Series(np.random.normal(0, 1, 1000), index=dates)
        
        reward_info = {
            'total_reward': np.random.uniform(-1, 1),
            'components': {'main': 0.5, 'penalty': -0.1},
            'future_leak': False,
            'validation': {'score': 0.8}
        }
        
        pool.add_factor(tokens, formula, pnl, equity, signal, reward_info, f"v{i}", i*100)
    
    # 통계 출력
    stats = pool.get_statistics()
    print(f"\n📊 Factor Pool 통계:")
    print(f"  총 팩터 수: {stats['total_factors']}")
    print(f"  평균 수익률: {stats['avg_return']:.2%}")
    print(f"  평균 샤프 비율: {stats['avg_sharpe']:.3f}")
    
    # 상위 팩터 조회
    top_factors = pool.get_top_factors(3)
    print(f"\n🏆 상위 3개 팩터:")
    for factor in top_factors:
        print(f"  {factor.factor_id}: Return {factor.total_return:.2%}, Sharpe {factor.sharpe_ratio:.3f}")
