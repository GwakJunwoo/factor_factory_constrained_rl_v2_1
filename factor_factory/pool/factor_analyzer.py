#!/usr/bin/env python3
"""
Factor Analyzer - 팩터 분석 및 성능 평가

기능:
1. 팩터 성능 분석
2. 백테스트 검증
3. 안정성 분석
4. 팩터 랭킹 시스템
5. 성능 예측 모델
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

from .factor_pool import FactorPool, FactorRecord

class FactorAnalyzer:
    """팩터 분석 클래스"""
    
    def __init__(self, factor_pool: FactorPool):
        self.pool = factor_pool
        self.scaler = StandardScaler()
        self.performance_model = None
    
    def analyze_factor_performance(self, factor_id: str) -> Dict:
        """개별 팩터 성능 분석"""
        
        factor = self.pool.get_factor(factor_id)
        if factor is None:
            return {"error": "Factor not found"}
        
        timeseries = self.pool.get_factor_timeseries(factor_id)
        
        if not timeseries:
            return {"error": "No timeseries data"}
        
        analysis = {
            "basic_info": {
                "factor_id": factor.factor_id,
                "name": factor.name,
                "formula": factor.formula,
                "created_at": factor.created_at.isoformat(),
                "model_version": factor.model_version
            }
        }
        
        # PnL 분석
        if 'pnl' in timeseries:
            pnl = timeseries['pnl']
            analysis["pnl_analysis"] = self._analyze_pnl(pnl)
        
        # 수익률 분석
        if 'equity' in timeseries:
            equity = timeseries['equity']
            analysis["equity_analysis"] = self._analyze_equity(equity)
        
        # 신호 분석
        if 'signal' in timeseries:
            signal = timeseries['signal']
            analysis["signal_analysis"] = self._analyze_signal(signal)
        
        # 리스크 분석
        analysis["risk_analysis"] = self._analyze_risk(factor, timeseries)
        
        # 안정성 분석
        analysis["stability_analysis"] = self._analyze_stability(factor, timeseries)
        
        # 랭킹 정보
        analysis["ranking"] = self._get_factor_ranking(factor)
        
        return analysis
    
    def _analyze_pnl(self, pnl: pd.Series) -> Dict:
        """PnL 분석"""
        
        return {
            "total_pnl": float(pnl.sum()),
            "mean_daily_pnl": float(pnl.mean()),
            "std_daily_pnl": float(pnl.std()),
            "skewness": float(pnl.skew()),
            "kurtosis": float(pnl.kurtosis()),
            "positive_days_ratio": float((pnl > 0).mean()),
            "best_day": float(pnl.max()),
            "worst_day": float(pnl.min()),
            "consecutive_positive_max": self._max_consecutive(pnl > 0),
            "consecutive_negative_max": self._max_consecutive(pnl <= 0)
        }
    
    def _analyze_equity(self, equity: pd.Series) -> Dict:
        """수익률 곡선 분석"""
        
        returns = equity.pct_change().dropna()
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        
        return {
            "total_return": float(equity.iloc[-1] - 1),
            "annualized_return": float((equity.iloc[-1] ** (252*24/len(equity)) - 1)),
            "volatility": float(returns.std() * np.sqrt(252*24)),
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252*24)) if returns.std() > 0 else 0,
            "max_drawdown": float(abs(drawdown.min())),
            "current_drawdown": float(drawdown.iloc[-1]),
            "drawdown_duration_max": self._max_drawdown_duration(drawdown),
            "calmar_ratio": float((equity.iloc[-1] ** (252*24/len(equity)) - 1) / abs(drawdown.min())) if abs(drawdown.min()) > 1e-6 else 0,
            "underwater_time_ratio": float((drawdown < -0.01).mean())  # 1% 이상 드로우다운 비율
        }
    
    def _analyze_signal(self, signal: pd.Series) -> Dict:
        """신호 분석"""
        
        signal_changes = signal.diff().abs()
        turnover = signal_changes.sum() / len(signal)
        
        return {
            "mean_signal": float(signal.mean()),
            "std_signal": float(signal.std()),
            "signal_range": float(signal.max() - signal.min()),
            "turnover_rate": float(turnover),
            "extreme_signals_ratio": float((abs(signal) > 2).mean()),
            "signal_autocorr": float(signal.autocorr(lag=1)) if len(signal) > 1 else 0,
            "signal_persistence": self._calculate_signal_persistence(signal)
        }
    
    def _analyze_risk(self, factor: FactorRecord, timeseries: Dict) -> Dict:
        """리스크 분석"""
        
        risk_metrics = {
            "max_drawdown": factor.max_drawdown,
            "volatility": factor.volatility,
            "var_95": factor.var_95,
            "skewness": factor.skewness,
            "kurtosis": factor.kurtosis
        }
        
        if 'pnl' in timeseries:
            pnl = timeseries['pnl']
            
            # VaR 계산
            risk_metrics["var_99"] = float(pnl.quantile(0.01))
            risk_metrics["cvar_95"] = float(pnl[pnl <= pnl.quantile(0.05)].mean())
            risk_metrics["cvar_99"] = float(pnl[pnl <= pnl.quantile(0.01)].mean())
            
            # 꼬리 리스크
            risk_metrics["tail_ratio"] = float(abs(pnl.quantile(0.95)) / abs(pnl.quantile(0.05))) if pnl.quantile(0.05) != 0 else 0
            
            # 하방 편차
            negative_returns = pnl[pnl < pnl.mean()]
            risk_metrics["downside_deviation"] = float(negative_returns.std()) if len(negative_returns) > 0 else 0
        
        return risk_metrics
    
    def _analyze_stability(self, factor: FactorRecord, timeseries: Dict) -> Dict:
        """안정성 분석"""
        
        stability = {}
        
        if 'pnl' in timeseries and 'equity' in timeseries:
            pnl = timeseries['pnl']
            equity = timeseries['equity']
            
            # 기간별 성과 분석
            periods = self._split_into_periods(pnl, equity)
            stability["period_analysis"] = periods
            
            # 롤링 성과 안정성
            rolling_returns = equity.pct_change().rolling(window=30).sum()
            stability["rolling_stability"] = {
                "rolling_mean": float(rolling_returns.mean()),
                "rolling_std": float(rolling_returns.std()),
                "rolling_sharpe": float(rolling_returns.mean() / rolling_returns.std()) if rolling_returns.std() > 0 else 0
            }
            
            # 시계열 안정성 테스트
            stability["stationarity_test"] = self._test_stationarity(pnl)
        
        return stability
    
    def _split_into_periods(self, pnl: pd.Series, equity: pd.Series, n_periods: int = 4) -> Dict:
        """기간별 성과 분석"""
        
        period_length = len(pnl) // n_periods
        periods_analysis = {}
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(pnl)
            
            period_pnl = pnl.iloc[start_idx:end_idx]
            period_equity = equity.iloc[start_idx:end_idx]
            
            period_return = period_equity.iloc[-1] / period_equity.iloc[0] - 1
            period_sharpe = period_pnl.mean() / period_pnl.std() if period_pnl.std() > 0 else 0
            
            periods_analysis[f"period_{i+1}"] = {
                "return": float(period_return),
                "sharpe": float(period_sharpe),
                "win_rate": float((period_pnl > 0).mean())
            }
        
        # 기간별 일관성 측정
        returns = [periods_analysis[f"period_{i+1}"]["return"] for i in range(n_periods)]
        sharpes = [periods_analysis[f"period_{i+1}"]["sharpe"] for i in range(n_periods)]
        
        periods_analysis["consistency"] = {
            "return_consistency": float(1 / (np.std(returns) + 1e-6)),
            "sharpe_consistency": float(1 / (np.std(sharpes) + 1e-6)),
            "positive_periods_ratio": float(sum(1 for r in returns if r > 0) / len(returns))
        }
        
        return periods_analysis
    
    def _test_stationarity(self, series: pd.Series) -> Dict:
        """정상성 테스트"""
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(series.dropna())
            
            return {
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "is_stationary": result[1] < 0.05,
                "critical_values": {k: float(v) for k, v in result[4].items()}
            }
        except:
            return {"error": "Stationarity test failed"}
    
    def _max_consecutive(self, boolean_series: pd.Series) -> int:
        """최대 연속 횟수"""
        
        consecutive = 0
        max_consecutive = 0
        
        for value in boolean_series:
            if value:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """최대 드로우다운 지속 기간"""
        
        underwater = drawdown < -0.001  # 0.1% 이상 드로우다운
        return self._max_consecutive(underwater)
    
    def _calculate_signal_persistence(self, signal: pd.Series) -> float:
        """신호 지속성 계산"""
        
        # 신호 방향의 지속성
        signal_direction = np.sign(signal)
        direction_changes = (signal_direction.diff() != 0).sum()
        
        # 변화가 적을수록 지속성이 높음
        persistence = 1 - (direction_changes / len(signal))
        return float(persistence)
    
    def _get_factor_ranking(self, factor: FactorRecord) -> Dict:
        """팩터 랭킹 정보"""
        
        # 전체 팩터 대비 순위 계산
        all_factors = self.pool.get_top_factors(1000)  # 충분히 많이 가져오기
        
        rankings = {}
        
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        for metric in metrics:
            values = [getattr(f, metric) for f in all_factors]
            factor_value = getattr(factor, metric)
            
            if metric == 'max_drawdown':  # 낮을수록 좋음
                rank = sum(1 for v in values if v > factor_value) + 1
            else:  # 높을수록 좋음
                rank = sum(1 for v in values if v < factor_value) + 1
            
            percentile = (1 - rank / len(values)) * 100
            
            rankings[metric] = {
                "rank": rank,
                "total_factors": len(values),
                "percentile": round(percentile, 1)
            }
        
        return rankings
    
    def compare_factors(self, factor_ids: List[str]) -> Dict:
        """팩터 간 비교 분석"""
        
        factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if len(factors) < 2:
            return {"error": "Need at least 2 factors for comparison"}
        
        comparison = {
            "factor_count": len(factors),
            "comparison_metrics": {},
            "correlation_analysis": {},
            "dominance_analysis": {}
        }
        
        # 성능 지표 비교
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'volatility']
        
        for metric in metrics:
            values = [getattr(f, metric) for f in factors]
            
            comparison["comparison_metrics"][metric] = {
                "values": {f.factor_id[:8]: getattr(f, metric) for f in factors},
                "best": max(values) if metric != 'max_drawdown' else min(values),
                "worst": min(values) if metric != 'max_drawdown' else max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        # 상관관계 분석
        correlation_matrix = self._calculate_factor_correlations(factor_ids)
        if correlation_matrix is not None:
            comparison["correlation_analysis"] = {
                "mean_correlation": float(np.mean(correlation_matrix.values)),
                "max_correlation": float(np.max(correlation_matrix.values)),
                "min_correlation": float(np.min(correlation_matrix.values)),
                "correlation_matrix": correlation_matrix.to_dict()
            }
        
        # 지배 분석 (어떤 팩터가 다른 팩터들을 지배하는지)
        dominance_scores = self._calculate_dominance_scores(factors)
        comparison["dominance_analysis"] = dominance_scores
        
        return comparison
    
    def _calculate_factor_correlations(self, factor_ids: List[str]) -> Optional[pd.DataFrame]:
        """팩터 간 상관관계 계산"""
        
        pnl_data = {}
        
        for factor_id in factor_ids:
            timeseries = self.pool.get_factor_timeseries(factor_id)
            if 'pnl' in timeseries:
                pnl_data[factor_id[:8]] = timeseries['pnl']
        
        if len(pnl_data) < 2:
            return None
        
        df = pd.DataFrame(pnl_data)
        return df.corr()
    
    def _calculate_dominance_scores(self, factors: List[FactorRecord]) -> Dict:
        """지배 점수 계산"""
        
        dominance = {}
        metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
        
        for factor in factors:
            score = 0
            comparisons = 0
            
            for other_factor in factors:
                if factor.factor_id == other_factor.factor_id:
                    continue
                
                for metric in metrics:
                    factor_value = getattr(factor, metric)
                    other_value = getattr(other_factor, metric)
                    
                    if factor_value > other_value:
                        score += 1
                    comparisons += 1
            
            dominance_ratio = score / comparisons if comparisons > 0 else 0
            
            dominance[factor.factor_id[:8]] = {
                "wins": score,
                "total_comparisons": comparisons,
                "dominance_ratio": round(dominance_ratio, 3)
            }
        
        return dominance
    
    def create_factor_scorecard(self, factor_id: str) -> Dict:
        """팩터 스코어카드 생성"""
        
        analysis = self.analyze_factor_performance(factor_id)
        
        if "error" in analysis:
            return analysis
        
        # 종합 점수 계산
        scores = {}
        
        # 수익성 점수 (40%)
        returns_score = min(analysis["equity_analysis"]["total_return"] * 10, 10)  # 10% = 10점
        sharpe_score = min(analysis["equity_analysis"]["sharpe_ratio"] * 2, 10)   # 5.0 = 10점
        profitability_score = (returns_score * 0.6 + sharpe_score * 0.4) * 0.4
        
        # 안정성 점수 (30%)
        drawdown_score = max(0, 10 - analysis["equity_analysis"]["max_drawdown"] * 50)  # 20% DD = 0점
        volatility_score = max(0, 10 - analysis["equity_analysis"]["volatility"] * 5)    # 20% vol = 0점
        stability_score = (drawdown_score * 0.6 + volatility_score * 0.4) * 0.3
        
        # 일관성 점수 (20%)
        win_rate = analysis["pnl_analysis"]["positive_days_ratio"]
        consistency_score = abs(win_rate - 0.5) * 20 * 0.2  # 50%에서 멀수록 일관성
        
        # 복잡도 점수 (10%)
        factor = self.pool.get_factor(factor_id)
        complexity_penalty = min(factor.depth * 0.5, 5)  # 깊이 10 = 5점 감점
        complexity_score = max(0, 10 - complexity_penalty) * 0.1
        
        total_score = profitability_score + stability_score + consistency_score + complexity_score
        
        scorecard = {
            "factor_id": factor_id,
            "total_score": round(total_score, 2),
            "grade": self._score_to_grade(total_score),
            "component_scores": {
                "profitability": round(profitability_score, 2),
                "stability": round(stability_score, 2),
                "consistency": round(consistency_score, 2),
                "complexity": round(complexity_score, 2)
            },
            "strengths": self._identify_strengths(analysis),
            "weaknesses": self._identify_weaknesses(analysis),
            "recommendations": self._generate_recommendations(analysis)
        }
        
        return scorecard
    
    def _score_to_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        
        if score >= 8.5:
            return "A+"
        elif score >= 7.5:
            return "A"
        elif score >= 6.5:
            return "B+"
        elif score >= 5.5:
            return "B"
        elif score >= 4.5:
            return "C+"
        elif score >= 3.5:
            return "C"
        else:
            return "D"
    
    def _identify_strengths(self, analysis: Dict) -> List[str]:
        """강점 식별"""
        
        strengths = []
        
        if analysis["equity_analysis"]["sharpe_ratio"] > 1.5:
            strengths.append("High Sharpe Ratio (>1.5)")
        
        if analysis["equity_analysis"]["max_drawdown"] < 0.1:
            strengths.append("Low Maximum Drawdown (<10%)")
        
        if analysis["pnl_analysis"]["positive_days_ratio"] > 0.6:
            strengths.append("High Win Rate (>60%)")
        
        if analysis["equity_analysis"]["calmar_ratio"] > 2.0:
            strengths.append("Excellent Risk-Adjusted Returns")
        
        return strengths
    
    def _identify_weaknesses(self, analysis: Dict) -> List[str]:
        """약점 식별"""
        
        weaknesses = []
        
        if analysis["equity_analysis"]["max_drawdown"] > 0.2:
            weaknesses.append("High Maximum Drawdown (>20%)")
        
        if analysis["equity_analysis"]["volatility"] > 0.3:
            weaknesses.append("High Volatility (>30%)")
        
        if analysis["pnl_analysis"]["skewness"] < -1.0:
            weaknesses.append("Negative Skewness (Tail Risk)")
        
        if analysis["signal_analysis"]["turnover_rate"] > 2.0:
            weaknesses.append("High Turnover Rate")
        
        return weaknesses
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """개선 권장사항"""
        
        recommendations = []
        
        if analysis["equity_analysis"]["max_drawdown"] > 0.15:
            recommendations.append("Consider position sizing to reduce maximum drawdown")
        
        if analysis["signal_analysis"]["turnover_rate"] > 1.5:
            recommendations.append("Implement signal smoothing to reduce transaction costs")
        
        if analysis["pnl_analysis"]["skewness"] < -0.5:
            recommendations.append("Add tail risk management to improve return distribution")
        
        return recommendations

if __name__ == "__main__":
    # 테스트
    from factor_pool import FactorPool
    
    pool = FactorPool("test_factor_pool")
    analyzer = FactorAnalyzer(pool)
    
    # 팩터 분석 테스트
    factors = pool.get_top_factors(3)
    if factors:
        factor_id = factors[0].factor_id
        analysis = analyzer.analyze_factor_performance(factor_id)
        scorecard = analyzer.create_factor_scorecard(factor_id)
        
        print(f"📊 Factor Analysis: {factor_id}")
        print(f"Total Score: {scorecard['total_score']}/10 (Grade: {scorecard['grade']})")
        print(f"Strengths: {scorecard['strengths']}")
        print(f"Weaknesses: {scorecard['weaknesses']}")
