#!/usr/bin/env python3
"""
Korean to English text replacement for Windows CP949 compatibility
"""

import re

# Dictionary of Korean text replacements
replacements = {
    # Comments
    "# MCTS 탐색기": "# MCTS Searcher",
    "# 네트워크 트레이너": "# Network Trainer", 
    "# 데이터 수집기": "# Data Collector",
    "# 학습 설정": "# Training Configuration",
    "# 체크포인트 관리": "# Checkpoint Management",
    "# 학습 통계": "# Training Statistics",
    "# 성능 추적": "# Performance Tracking",
    "# 전체 진행률 바": "# Overall Progress Bar",
    "# 1. 자기 대국 (Self-Play)": "# 1. Self-Play",
    "# 2. 데이터 수집": "# 2. Data Collection",
    "# 3. 신경망 학습": "# 3. Neural Network Training",
    "# 4. 성능 평가": "# 4. Performance Evaluation",
    "# 5. 체크포인트 저장": "# 5. Checkpoint Save",
    "# 통계 업데이트": "# Statistics Update",
    "# 이번 반복에서 발견된 팩터들 요약": "# Summary of factors discovered in this iteration",
    
    # Function docstrings
    "AlphaZero 학습 실행": "Execute AlphaZero Training",
    "num_iterations: 학습 반복 횟수": "num_iterations: Number of training iterations",
    
    # Print statements
    "[CHECK] AlphaZero Trainer 초기화 완료": "[CHECK] AlphaZero Trainer initialization complete",
    "MCTS 시뮬레이션: ": "MCTS Simulations: ",
    "반복당 에피소드: ": "Episodes per iteration: ",
    "체크포인트 디렉토리: ": "Checkpoint directory: ",
    "[START] AlphaZero 학습 시작": "[START] AlphaZero training started",
    "반복)": "iterations)",
    "[TARGET] 자기 대국 중...": "[TARGET] Self-play in progress...",
    "(반복 ": "(iteration ",
    "[CHART] 학습 데이터 수집 중...": "[CHART] Collecting training data...",
    "🧠 신경망 학습 중...": "[BRAIN] Neural network training...",
    "신경망 학습 중...": "Neural network training...",
    "성능 평가 중...": "Performance evaluation in progress...",
    "평가 결과: ": "Evaluation result: ",
    "완료 요약": "completion summary",
    
    # Progress bar descriptions
    "전체 학습 진행": "Overall Training Progress",
    "반복 ": "Iteration ",
    
    # Bar format
    "반복 {n}": "iter {n}",
}

def fix_file_encoding(file_path):
    """Fix Korean text in the file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply replacements
    for korean, english in replacements.items():
        content = content.replace(korean, english)
    
    # Save back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed encoding in {file_path}")

if __name__ == "__main__":
    fix_file_encoding("factor_factory/mcts/alphazero_trainer.py")
