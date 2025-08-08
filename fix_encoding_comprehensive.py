#!/usr/bin/env python3
"""
Comprehensive Korean to English text replacement for Windows CP949 compatibility
"""

import re

def fix_file_encoding(file_path):
    """Fix Korean text in the file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # More comprehensive replacements - in order of specificity
    replacements = [
        # Docstrings and function descriptions
        ("자기 대국 에피소드 실행", "Execute self-play episodes"),
        ("자기 대국", "self-play"),
        ("[COMPLETE] AlphaZero 학습 완료!", "[COMPLETE] AlphaZero training completed!"),
        ("MCTS 트리에서 완전한 프로그램 생성", "Generate complete program from MCTS tree"),
        ("episode 데이터를 training 데이터로 변환", "Convert episode data to training data"),
        
        # Comments and technical terms
        ("최대 길이 제한", "maximum length limit"),
        ("온도 조절된 액션 선택", "temperature-controlled action selection"),
        ("유효한 액션들에 대해서만 확률 재정규화", "renormalize probabilities only for valid actions"),
        ("터미널 체크", "terminal check"),
        
        # Time and status messages
        ("⏱소요 시간:", "[TIME] Duration:"),
        ("소요 시간:", "Duration:"),
        ("[CHART] 수집된 데이터:", "[CHART] Collected data:"),
        ("수집된 데이터:", "Collected data:"),
        ("[SEARCH] 이번 Iteration 발견 팩터:", "[SEARCH] Factors discovered this iteration:"),
        ("이번 Iteration 발견 팩터:", "Factors discovered this iteration:"),
        ("발견된 팩터들:", "Discovered factors:"),
        ("[TROPHY] 총 발견 팩터:", "[TROPHY] Total discovered factors:"),
        ("총 발견 팩터:", "Total discovered factors:"),
        ("[TARGET] 최고 성능:", "[TARGET] Best performance:"),
        ("최고 성능:", "Best performance:"),
        ("Success률", "Success Rate"),
        
        # Episode progress messages
        ("# 프로그레스 바 추가", "# Add progress bar"),
        ("에피소드 진행", "Episode Progress"),
        ("# 초기 상태에서 MCTS 탐색", "# MCTS search from initial state"),
        ("[SEARCH] 에피소드", "[SEARCH] Episode"),
        ("MCTS 탐색 중...", "MCTS searching..."),
        ("완료", "completed"),
        
        # Action and program generation
        ("# 액션 선택 (온도 조절)", "# Action selection (temperature control)"),
        ("점진적 감소", "gradual decrease"),
        ("# 프로그램 생성 (시뮬레이션)", "# Program generation (simulation)"),
        ("생성된 프로그램 길이:", "Generated program length:"),
        ("토큰", "tokens"),
        ("# 프로그램 평가", "# Program evaluation"),
        ("# 프로그램을 인간이 읽기 쉬운 형태로 변환", "# Convert program to human-readable format"),
        ("# 실시간 팩터 정보 출력", "# Real-time factor information output"),
        
        # Factor discovery messages
        ("[발견]", "[FOUND]"),
        ("보상:", "Reward:"),
        ("샤프:", "Sharpe:"),
        ("수익률:", "Return:"),
        ("낙폭:", "Drawdown:"),
        ("부정적이지만 완전히 나쁘지 않은 경우", "negative but not completely bad case"),
        ("[시도]", "[TRY]"),
        ("# 로깅에도 기록", "# Also record in logging"),
        ("[FACTOR] 발견:", "[FACTOR] Found:"),
        ("# Factor Pool에 저장", "# Save to Factor Pool"),
        ("평가 실패:", "Evaluation failed:"),
        ("프로그램 생성 실패", "Program generation failed"),
        
        # Episode data and progress updates
        ("# 에피소드 데이터 저장", "# Save episode data"),
        ("# 프로그레스 바 업데이트", "# Update progress bar"),
        ("성공", "Success"),
        ("성공률", "Success Rate"),
        ("⚠️ 에피소드", "[WARNING] Episode"),
        ("오류:", "Error:"),
        ("자기 대국 완료:", "Self-play completed:"),
        ("개 에피소드,", "episodes,"),
        
        # Comments and numbers
        ("상위 3개만", "top 3 only"),
        ("초", "sec"),
        ("개", ""),
        
        # More specific phrases
        ("tokens 시퀀스 evaluation (MCTS용)", "Token sequence evaluation (for MCTS)"),
        ("우수한 program을 Factor Pool에 저장", "Save excellent programs to Factor Pool"),
        ("가상의 시계열 data 생성 (실제로는 evaluation에서 가져와야 함)", "Generate virtual time series data (should actually be fetched from evaluation)"),
        ("Factor Pool 저장 실패:", "Factor Pool save failed:"),
        ("check포인트 저장", "Checkpoint save"),
        ("메타data 저장", "Metadata save"),
        ("최고 performance 모델 저장", "Save best performance model"),
        
        # More word-level replacements
        ("시퀀스", "sequence"),
        ("우수한", "excellent"),
        ("가상의", "virtual"),
        ("시계열", "time series"),
        ("실제로는", "actually"),
        ("가져와야", "should fetch"),
        ("함", ""),
        ("저장", "save"),
        ("실패", "failed"),
        ("포인트", "point"),
        ("메타", "meta"),
        ("모델", "model"),
        ("최고", "best"),
        ("program 생성 과정의 각 상태에서 training data 생성", "Generate training data at each state in program generation process"),
        ("신경망 training", "Neural network training"),
        ("배치 샘플링", "Batch sampling"),
        ("training 스텝", "Training step"),
        ("에포크", "Epoch"),
        ("손실", "Loss"),
        ("현재 네트워크 성능 evaluation", "Current network performance evaluation"),
        ("evaluation용 MCTS (더 적은 시뮬레이션)", "MCTS for evaluation (fewer simulations)"),
        ("빠른 evaluation", "Fast evaluation"),
        ("최고 성능 업데이트", "Best performance update"),
        ("더", "more"),
        ("적은", "fewer"),
        ("시뮬레이션", "simulation"),
        ("빠른", "fast"),
        ("네트워크", "network"),
        ("성능", "performance"),
        ("상태", "state"),
        ("과정", "process"),
        ("각", "each"),
        ("현재", "current"),
        ("업데이트", "update"),
        ("배치", "batch"),
        ("샘플링", "sampling"),
        ("스텝", "step"),
        ("학습", "training"),
        ("완료", "completed"),
        ("평가", "evaluation"),
        ("결과", "result"),
        ("팩터", "factor"),
        ("발견", "discovery"),
        ("탐색", "search"),
        ("진행", "progress"),
        ("에피소드", "episode"),
        ("트리", "tree"),
        ("프로그램", "program"),
        ("액션", "action"),
        ("확률", "probability"),
        ("재정규화", "renormalization"),
        ("터미널", "terminal"),
        ("체크", "check"),
        ("데이터", "data"),
        ("변환", "conversion"),
        ("길이", "length"),
        ("제한", "limit"),
        ("조절", "control"),
        ("선택", "selection"),
        ("유효한", "valid"),
        ("대해서만", "only for"),
        ("완전한", "complete"),
        ("최대", "maximum"),
        ("온도", "temperature"),
    ]
    
    # Apply replacements
    for korean, english in replacements:
        content = content.replace(korean, english)
    
    # Remove any remaining Unicode emojis that might cause issues
    content = re.sub(r'[⏱🧠⚠️📊🚀✅🏆🎯🔍]', '', content)
    
    # Save back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed encoding in {file_path}")

if __name__ == "__main__":
    fix_file_encoding("factor_factory/mcts/alphazero_trainer.py")
    fix_file_encoding("multi_asset_gui.py")
