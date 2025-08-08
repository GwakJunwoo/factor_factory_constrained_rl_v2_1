#!/usr/bin/env python3
"""
Factor Pool 관리 및 분석 스크립트

사용법:
    python factor_pool_manager.py --list           # 팩터 목록 조회
    python factor_pool_manager.py --analyze ID     # 팩터 분석
    python factor_pool_manager.py --compare ID1,ID2,ID3  # 팩터 비교
    python factor_pool_manager.py --report         # 종합 리포트 생성
    python factor_pool_manager.py --visualize      # 시각화 차트 생성
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from factor_factory.pool import FactorPool, FactorAnalyzer, FactorVisualizer

class FactorPoolManager:
    """Factor Pool 관리 클래스"""
    
    def __init__(self, pool_path: str = "factor_pool"):
        self.pool = FactorPool(pool_path)
        self.analyzer = FactorAnalyzer(self.pool)
        self.visualizer = FactorVisualizer(self.pool)
    
    def list_factors(self, n: int = 20, sort_by: str = 'total_return'):
        """팩터 목록 조회"""
        
        factors = self.pool.get_top_factors(n, sort_by=sort_by)
        
        if not factors:
            print("📭 저장된 팩터가 없습니다.")
            return
        
        print(f"🏆 상위 {len(factors)}개 팩터 (정렬: {sort_by})")
        print("=" * 100)
        print(f"{'ID':16s} {'Name':15s} {'Return':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Win%':>6s} {'Depth':>5s} {'Created':>12s}")
        print("-" * 100)
        
        for factor in factors:
            print(f"{factor.factor_id[:16]:16s} {factor.name[:15]:15s} "
                  f"{factor.total_return:8.2%} {factor.sharpe_ratio:8.3f} "
                  f"{factor.max_drawdown:8.2%} {factor.win_rate:6.1%} "
                  f"{factor.depth:5d} {factor.created_at.strftime('%Y-%m-%d'):>12s}")
    
    def show_statistics(self):
        """풀 통계 정보 출력"""
        
        stats = self.pool.get_statistics()
        
        print("\n📊 Factor Pool 통계")
        print("=" * 50)
        print(f"총 팩터 수: {stats['total_factors']}")
        print(f"평균 수익률: {stats['avg_return']:.2%}")
        print(f"평균 샤프 비율: {stats['avg_sharpe']:.3f}")
        print(f"평균 MDD: {stats['avg_drawdown']:.2%}")
        print()
        print(f"최고 수익률: {stats['best_return']:.2%}")
        print(f"최고 샤프 비율: {stats['best_sharpe']:.3f}")
        print(f"최저 MDD: {stats['best_drawdown']:.2%}")
        
        if stats['version_counts']:
            print(f"\n모델 버전별 분포:")
            for version, count in stats['version_counts'].items():
                print(f"  {version}: {count}개")
    
    def analyze_factor(self, factor_id: str):
        """개별 팩터 분석"""
        
        factor = self.pool.get_factor(factor_id)
        if factor is None:
            print(f"❌ 팩터 '{factor_id}'를 찾을 수 없습니다.")
            return
        
        print(f"🔍 팩터 분석: {factor_id}")
        print("=" * 60)
        
        # 기본 정보
        print(f"이름: {factor.name}")
        print(f"공식: {factor.formula}")
        print(f"생성일: {factor.created_at}")
        print(f"모델 버전: {factor.model_version}")
        print(f"깊이: {factor.depth}, 길이: {factor.length}")
        
        # 성능 지표
        print(f"\n📈 성능 지표:")
        print(f"  총 수익률: {factor.total_return:.2%}")
        print(f"  샤프 비율: {factor.sharpe_ratio:.3f}")
        print(f"  최대 낙폭: {factor.max_drawdown:.2%}")
        print(f"  승률: {factor.win_rate:.1%}")
        print(f"  수익 인수: {factor.profit_factor:.2f}")
        print(f"  변동성: {factor.volatility:.2%}")
        
        # 스코어카드
        scorecard = self.analyzer.create_factor_scorecard(factor_id)
        print(f"\n🏅 종합 평가:")
        print(f"  총점: {scorecard['total_score']:.1f}/10 (등급: {scorecard['grade']})")
        print(f"  수익성: {scorecard['component_scores']['profitability']:.1f}")
        print(f"  안정성: {scorecard['component_scores']['stability']:.1f}")
        print(f"  일관성: {scorecard['component_scores']['consistency']:.1f}")
        print(f"  복잡도: {scorecard['component_scores']['complexity']:.1f}")
        
        if scorecard['strengths']:
            print(f"\n✅ 강점:")
            for strength in scorecard['strengths']:
                print(f"  - {strength}")
        
        if scorecard['weaknesses']:
            print(f"\n⚠️ 약점:")
            for weakness in scorecard['weaknesses']:
                print(f"  - {weakness}")
        
        if scorecard['recommendations']:
            print(f"\n💡 개선 권장사항:")
            for rec in scorecard['recommendations']:
                print(f"  - {rec}")
    
    def compare_factors(self, factor_ids: list):
        """팩터 비교 분석"""
        
        print(f"⚖️ 팩터 비교 분석 ({len(factor_ids)}개)")
        print("=" * 60)
        
        # 기본 비교
        factors = [self.pool.get_factor(fid) for fid in factor_ids if self.pool.get_factor(fid)]
        
        if len(factors) != len(factor_ids):
            print("⚠️ 일부 팩터를 찾을 수 없습니다.")
        
        if len(factors) < 2:
            print("❌ 비교를 위해 최소 2개 팩터가 필요합니다.")
            return
        
        # 비교 테이블
        print(f"{'Metric':15s}", end="")
        for factor in factors:
            print(f"{factor.factor_id[:12]:>12s}", end="")
        print()
        print("-" * (15 + 12 * len(factors)))
        
        metrics = [
            ('Total Return', 'total_return', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Win Rate', 'win_rate', '%'),
            ('Volatility', 'volatility', '%'),
            ('Depth', 'depth', ''),
        ]
        
        for metric_name, attr, unit in metrics:
            print(f"{metric_name:15s}", end="")
            for factor in factors:
                value = getattr(factor, attr)
                if unit == '%':
                    print(f"{value:12.2%}", end="")
                else:
                    print(f"{value:12.1f}", end="")
            print()
        
        # 상세 비교 분석
        comparison = self.analyzer.compare_factors(factor_ids)
        
        if 'correlation_analysis' in comparison:
            corr = comparison['correlation_analysis']
            print(f"\n🔗 상관관계 분석:")
            print(f"  평균 상관계수: {corr['mean_correlation']:.3f}")
            print(f"  최대 상관계수: {corr['max_correlation']:.3f}")
            print(f"  최소 상관계수: {corr['min_correlation']:.3f}")
        
        if 'dominance_analysis' in comparison:
            dom = comparison['dominance_analysis']
            print(f"\n👑 지배 분석:")
            for factor_id, data in dom.items():
                print(f"  {factor_id}: {data['dominance_ratio']:.1%} 지배율 "
                      f"({data['wins']}/{data['total_comparisons']})")
    
    def create_report(self, output_dir: str = "factor_reports"):
        """종합 리포트 생성"""
        
        print(f"📋 종합 팩터 리포트 생성 중... ({output_dir})")
        
        # 시각화 리포트
        self.visualizer.create_factor_report(save_dir=output_dir)
        
        # 텍스트 리포트
        stats = self.pool.get_statistics()
        top_factors = self.pool.get_top_factors(10)
        
        report_path = f"{output_dir}/factor_summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🏆 FACTOR POOL SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"생성일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 전체 통계:\n")
            f.write(f"  총 팩터 수: {stats['total_factors']}\n")
            f.write(f"  평균 수익률: {stats['avg_return']:.2%}\n")
            f.write(f"  평균 샤프 비율: {stats['avg_sharpe']:.3f}\n")
            f.write(f"  평균 MDD: {stats['avg_drawdown']:.2%}\n\n")
            
            f.write("🏆 상위 10개 팩터:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'ID':16s} {'Return':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Win%':>6s} {'Grade':>6s}\n")
            f.write("-" * 80 + "\n")
            
            for factor in top_factors:
                scorecard = self.analyzer.create_factor_scorecard(factor.factor_id)
                f.write(f"{factor.factor_id[:16]:16s} {factor.total_return:8.2%} "
                       f"{factor.sharpe_ratio:8.3f} {factor.max_drawdown:8.2%} "
                       f"{factor.win_rate:6.1%} {scorecard['grade']:>6s}\n")
        
        print(f"✅ 텍스트 리포트 저장: {report_path}")
    
    def create_visualizations(self, output_dir: str = "factor_charts"):
        """시각화 차트 생성"""
        
        print(f"📈 시각화 차트 생성 중... ({output_dir})")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 성능 비교
        self.visualizer.plot_performance_comparison(
            n_top=15, save_path=f"{output_dir}/performance_comparison.png"
        )
        
        # 리스크-리턴 산점도
        self.visualizer.plot_risk_return_scatter(
            n_top=25, save_path=f"{output_dir}/risk_return_scatter.png"
        )
        
        # 특성 히트맵
        self.visualizer.plot_factor_characteristics_heatmap(
            n_top=20, save_path=f"{output_dir}/characteristics_heatmap.png"
        )
        
        print(f"✅ 시각화 차트 생성 완료: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Factor Pool 관리 도구")
    parser.add_argument("--pool-path", default="factor_pool", help="Factor Pool 경로")
    
    subparsers = parser.add_subparsers(dest='command', help='사용할 명령')
    
    # list 명령
    list_parser = subparsers.add_parser('list', help='팩터 목록 조회')
    list_parser.add_argument('--n', type=int, default=20, help='표시할 팩터 수')
    list_parser.add_argument('--sort-by', default='total_return', 
                           choices=['total_return', 'sharpe_ratio', 'max_drawdown'],
                           help='정렬 기준')
    
    # analyze 명령
    analyze_parser = subparsers.add_parser('analyze', help='팩터 분석')
    analyze_parser.add_argument('factor_id', help='분석할 팩터 ID')
    
    # compare 명령
    compare_parser = subparsers.add_parser('compare', help='팩터 비교')
    compare_parser.add_argument('factor_ids', help='비교할 팩터 ID들 (쉼표로 구분)')
    
    # report 명령
    report_parser = subparsers.add_parser('report', help='종합 리포트 생성')
    report_parser.add_argument('--output-dir', default='factor_reports', help='출력 디렉토리')
    
    # visualize 명령
    viz_parser = subparsers.add_parser('visualize', help='시각화 차트 생성')
    viz_parser.add_argument('--output-dir', default='factor_charts', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Factor Pool Manager 초기화
    manager = FactorPoolManager(args.pool_path)
    
    try:
        if args.command == 'list':
            manager.list_factors(args.n, args.sort_by)
            manager.show_statistics()
            
        elif args.command == 'analyze':
            manager.analyze_factor(args.factor_id)
            
        elif args.command == 'compare':
            factor_ids = [fid.strip() for fid in args.factor_ids.split(',')]
            manager.compare_factors(factor_ids)
            
        elif args.command == 'report':
            manager.create_report(args.output_dir)
            
        elif args.command == 'visualize':
            manager.create_visualizations(args.output_dir)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
