#!/usr/bin/env python3
"""
실시간 로그 모니터링 스크립트
"""

import os
import time
import sys
from pathlib import Path

def watch_log(log_file="multi_asset_training.log", follow=True):
    """로그 파일을 실시간으로 모니터링"""
    
    if not os.path.exists(log_file):
        print(f"로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    print(f"📊 로그 모니터링 시작: {log_file}")
    print("=" * 60)
    
    # 기존 내용 출력
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if content:
            print(content)
    
    if not follow:
        return
    
    # 실시간 모니터링
    print("\n🔄 실시간 모니터링 중... (Ctrl+C로 중단)")
    print("=" * 60)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        f.seek(0, 2)  # 파일 끝으로 이동
        
        try:
            while True:
                line = f.readline()
                if line:
                    # 특별한 키워드 하이라이팅
                    if "[발견]" in line or "FACTOR" in line:
                        print(f"🎯 {line.strip()}")
                    elif "ERROR" in line or "오류" in line:
                        print(f"❌ {line.strip()}")
                    elif "완료" in line:
                        print(f"✅ {line.strip()}")
                    else:
                        print(line.strip())
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n모니터링을 중단했습니다.")

def filter_factors(log_file="multi_asset_training.log"):
    """팩터 관련 로그만 필터링해서 출력"""
    
    if not os.path.exists(log_file):
        print(f"로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    print("🔍 발견된 팩터들:")
    print("=" * 60)
    
    factor_count = 0
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "[FACTOR]" in line or "[발견]" in line:
                factor_count += 1
                print(f"{factor_count}. {line.strip()}")
    
    if factor_count == 0:
        print("아직 팩터가 발견되지 않았습니다.")
    else:
        print(f"\n총 {factor_count}개의 팩터가 발견되었습니다.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="로그 모니터링 도구")
    parser.add_argument("--file", "-f", default="multi_asset_training.log", 
                       help="로그 파일 경로")
    parser.add_argument("--follow", "-F", action="store_true", default=True,
                       help="실시간 모니터링")
    parser.add_argument("--factors-only", action="store_true",
                       help="팩터 관련 로그만 출력")
    
    args = parser.parse_args()
    
    if args.factors_only:
        filter_factors(args.file)
    else:
        watch_log(args.file, args.follow)
