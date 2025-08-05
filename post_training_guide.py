#!/usr/bin/env python3
"""
학습 완료 후 후속 작업 가이드 및 자동화 스크립트
"""

def print_post_training_guide():
    """학습 완료 후 작업 가이드 출력"""
    
    print("🎯 FACTOR FACTORY 학습 완료 후 작업 가이드")
    print("=" * 60)
    
    print("\n1️⃣ 새로운 모델에서 최적 프로그램 탐색")
    print("-" * 40)
    print("python -m factor_factory.scripts.cli_rlc_infer \\")
    print("  --model models/ppo_program_v2.zip \\")
    print("  --symbol BTCUSDT --interval 1h \\")
    print("  --tries 512 \\")
    print("  --outdir best_results_v2 \\")
    print("  --eval_stride 2 \\")
    print("  --max_eval_bars 15000")
    print("📌 목적: 훈련된 모델에서 가장 우수한 프로그램들을 탐색")
    
    print("\n2️⃣ 발견된 프로그램들 성과 비교 평가")
    print("-" * 40)
    print("python -m factor_factory.scripts.cli_rlc_eval \\")
    print("  --program best_results_v2/best_program.json \\")
    print("  --symbol BTCUSDT --interval 1h \\")
    print("  --outdir evaluation_v2 \\")
    print("  --charts \\")
    print("  --chart_dir charts_v2")
    print("📌 목적: 새로운 프로그램의 상세 성과 분석 및 시각화")
    
    print("\n3️⃣ 기존 모델과 성과 비교")
    print("-" * 40)
    print("python compare_models.py")
    print("📌 목적: v1 vs v2 모델 성과 비교 분석")
    
    print("\n4️⃣ 실거래 타당성 검증")
    print("-" * 40)
    print("python validate_for_production.py")
    print("📌 목적: 미래 정보 누출, 실거래 적합성 최종 검증")
    
    print("\n5️⃣ 프로덕션 배포 준비")
    print("-" * 40)
    print("python prepare_production.py")
    print("📌 목적: 실거래용 코드 패키징 및 문서화")
    
    print("\n⚡ 빠른 시작 (추천):")
    print("python post_training_workflow.py")
    print("📌 위의 모든 단계를 자동으로 실행")

if __name__ == "__main__":
    print_post_training_guide()
