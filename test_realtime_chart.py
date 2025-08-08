#!/usr/bin/env python3
"""
실시간 차트 테스트 스크립트

차트 위젯이 올바르게 동작하는지 테스트하기 위한 가상 데이터 생성 및 시각화
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QTimer

# 프로젝트 루트를 path에 추가
sys.path.append('.')

from factor_factory.visualization.realtime_chart import RealtimeChartWidget


class ChartTestWindow(QMainWindow):
    """차트 테스트 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 실시간 차트 테스트 - 영화같은 시각화")
        self.setGeometry(100, 100, 1400, 900)
        
        # 가상 데이터 생성
        self.generate_sample_data()
        self.current_index = 0
        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """UI 설정"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # 제어 버튼들
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️ 시뮬레이션 시작")
        self.play_btn.clicked.connect(self.start_simulation)
        
        self.pause_btn = QPushButton("⏸️ 일시정지")
        self.pause_btn.clicked.connect(self.pause_simulation)
        
        self.reset_btn = QPushButton("🔄 리셋")
        self.reset_btn.clicked.connect(self.reset_simulation)
        
        self.speed_btn = QPushButton("⚡ 고속 모드")
        self.speed_btn.clicked.connect(self.toggle_speed)
        
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.speed_btn)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # 차트 위젯
        self.chart_widget = RealtimeChartWidget()
        layout.addWidget(self.chart_widget)
        
        central_widget.setLayout(layout)
        
    def setup_timer(self):
        """타이머 설정"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.update_interval = 100  # ms
        
    def generate_sample_data(self):
        """실제 데이터 기반 샘플 생성"""
        try:
            from factor_factory.visualization.real_data_loader import load_real_chart_data
            
            # 실제 데이터 로드 시도
            print("실제 백테스트 데이터 로드 시도...")
            timestamps, prices, signals, pnl = load_real_chart_data(
                symbol="BTCUSDT",
                max_points=100,
                use_latest=True
            )
            
            if timestamps and prices:
                self.full_timestamps = timestamps
                self.full_prices = prices
                self.full_signals = signals
                self.full_pnl = pnl
                
                print(f"실제 데이터 로드 성공:")
                print(f"- 시점: {len(timestamps)}개")
                print(f"- 시간 범위: {timestamps[0]} ~ {timestamps[-1]}")
                print(f"- 가격 범위: {min(prices):.0f} ~ {max(prices):.0f}")
                print(f"- 시그널 수: Long {signals.count(1)}, Short {signals.count(-1)}")
                print(f"- PNL 범위: {min(pnl):.2f}% ~ {max(pnl):.2f}%")
                return
                
        except Exception as e:
            print(f"실제 데이터 로드 실패: {e}")
        
        # 실제 데이터 로드 실패 시 가상 데이터 생성
        print("가상 데이터 생성...")
        n_points = 100
        
        # 시간 데이터
        start_time = datetime.now() - timedelta(hours=n_points)
        self.full_timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
        
        # 가격 데이터 (트렌드 + 노이즈)
        trend = np.linspace(45000, 48000, n_points)  # 상승 트렌드
        noise = np.random.normal(0, 500, n_points)   # 노이즈
        self.full_prices = trend + noise
        
        # 시그널 데이터 (랜덤 롱/숏)
        self.full_signals = []
        for i in range(n_points):
            if np.random.random() < 0.15:
                signal = np.random.choice([1, -1])
            else:
                signal = 0
            self.full_signals.append(signal)
        
        # PNL 데이터 (누적 수익률)
        returns = np.random.normal(0.05, 0.8, n_points)
        self.full_pnl = np.cumsum(returns)
        
        print(f"가상 데이터 생성 완료:")
        print(f"- 시점: {n_points}개")
        print(f"- 가격 범위: {self.full_prices.min():.0f} ~ {self.full_prices.max():.0f}")
        print(f"- 시그널 수: Long {sum(1 for s in self.full_signals if s == 1)}, Short {sum(1 for s in self.full_signals if s == -1)}")
        print(f"- 최종 PNL: {self.full_pnl[-1]:.2f}%")
        
    def start_simulation(self):
        """시뮬레이션 시작"""
        if not self.timer.isActive():
            self.timer.start(self.update_interval)
            self.play_btn.setText("⏸️ 실행 중...")
            print("🎬 실시간 백테스트 시뮬레이션 시작!")
        
    def pause_simulation(self):
        """시뮬레이션 일시정지"""
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶️ 시뮬레이션 재개")
            print("⏸️ 시뮬레이션 일시정지")
        
    def reset_simulation(self):
        """시뮬레이션 리셋"""
        self.timer.stop()
        self.current_index = 0
        self.chart_widget.clear()
        self.play_btn.setText("▶️ 시뮬레이션 시작")
        print("🔄 시뮬레이션 리셋됨")
        
    def toggle_speed(self):
        """속도 모드 토글"""
        if self.update_interval == 100:
            self.update_interval = 20  # 고속 모드
            self.speed_btn.setText("🐌 일반 모드")
            print("⚡ 고속 모드 활성화")
        else:
            self.update_interval = 100  # 일반 모드  
            self.speed_btn.setText("⚡ 고속 모드")
            print("🐌 일반 모드 활성화")
            
        if self.timer.isActive():
            self.timer.setInterval(self.update_interval)
    
    def update_data(self):
        """데이터 업데이트 (한 시점씩 추가)"""
        if self.current_index < len(self.full_timestamps):
            # 현재까지의 데이터 전송
            timestamps = self.full_timestamps[:self.current_index + 1]
            prices = self.full_prices[:self.current_index + 1]
            signals = self.full_signals[:self.current_index + 1]
            pnl = self.full_pnl[:self.current_index + 1]
            
            # 차트 업데이트
            self.chart_widget.update_data(timestamps, prices, signals, pnl)
            
            self.current_index += 1
            
            # 진행률 출력
            if self.current_index % 10 == 0:
                progress = (self.current_index / len(self.full_timestamps)) * 100
                print(f"📈 진행률: {progress:.0f}% ({self.current_index}/{len(self.full_timestamps)})")
        else:
            # 시뮬레이션 완료
            self.timer.stop()
            self.play_btn.setText("✅ 완료")
            print("🎯 시뮬레이션 완료!")


def main():
    """메인 실행 함수"""
    app = QApplication(sys.argv)
    
    # 윈도우 생성 및 표시
    window = ChartTestWindow()
    window.show()
    
    print("🎬 실시간 차트 테스트 애플리케이션 시작")
    print("💡 사용법:")
    print("   - '시뮬레이션 시작' 버튼을 클릭하여 가상 백테스트 실행")
    print("   - '고속 모드'로 빠르게 확인 가능")
    print("   - 실시간으로 가격, 시그널, PNL이 업데이트됩니다")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
