"""
실시간 차트 위젯 - 영화같은 멋진 백테스트 시각화

백테스트가 진행되면서 실시간으로 가격 차트, 롱/숏 시그널, 누적 PNL을 
애니메이션으로 보여주는 위젯입니다.
"""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 영화같은 스타일 설정
plt.style.use('dark_background')
colors = {
    'bg': '#0d1421',
    'grid': '#1e2b3a', 
    'price': '#00d4aa',
    'long': '#00ff88',
    'short': '#ff4444',
    'pnl_positive': '#00cc88',
    'pnl_negative': '#ff6666',
    'text': '#ffffff',
    'accent': '#00ccff'
}


class RealtimeChartWidget(QWidget):
    """실시간 차트 위젯 - 영화같은 백테스트 시각화"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 데이터 저장
        self.timestamps = []
        self.prices = []
        self.signals = []  # 1: long, -1: short, 0: neutral
        self.pnl = []
        
        # 표시 모드
        self.display_mode = "전체 보기"  # "가격 + 시그널", "누적 PNL", "전체 보기"
        
        # 애니메이션 설정
        self.animation_speed = 50  # ms
        self.max_points_display = 200  # 표시할 최대 포인트 수
        
        self.setup_ui()
        self.setup_chart()
        self.setup_animation()
        
    def setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout()
        
        # 정보 라벨
        info_layout = QHBoxLayout()
        self.info_label = QLabel("🎬 실시간 백테스트 차트 - 대기 중...")
        self.info_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.info_label.setStyleSheet(f"color: {colors['accent']}; padding: 5px;")
        
        self.stats_label = QLabel("통계: -")
        self.stats_label.setFont(QFont("Arial", 9))
        self.stats_label.setStyleSheet(f"color: {colors['text']}; padding: 5px;")
        
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.stats_label)
        
        layout.addLayout(info_layout)
        
        # 차트 캔버스
        self.figure = Figure(figsize=(12, 8), facecolor=colors['bg'])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def setup_chart(self):
        """차트 설정"""
        self.figure.clear()
        
        if self.display_mode == "가격 + 시그널":
            self.ax_price = self.figure.add_subplot(111)
            self.setup_price_chart()
            
        elif self.display_mode == "누적 PNL":
            self.ax_pnl = self.figure.add_subplot(111)
            self.setup_pnl_chart()
            
        else:  # 전체 보기
            self.ax_price = self.figure.add_subplot(211)
            self.ax_pnl = self.figure.add_subplot(212)
            self.setup_price_chart()
            self.setup_pnl_chart()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def setup_price_chart(self):
        """가격 차트 설정"""
        self.ax_price.set_facecolor(colors['bg'])
        self.ax_price.grid(True, color=colors['grid'], alpha=0.3)
        self.ax_price.set_title('💰 가격 & 트레이딩 시그널', 
                               color=colors['text'], fontsize=14, fontweight='bold')
        self.ax_price.set_ylabel('가격', color=colors['text'])
        
        # 스타일링
        self.ax_price.spines['bottom'].set_color(colors['grid'])
        self.ax_price.spines['top'].set_color(colors['grid'])
        self.ax_price.spines['right'].set_color(colors['grid'])
        self.ax_price.spines['left'].set_color(colors['grid'])
        self.ax_price.tick_params(colors=colors['text'])
        
        # 초기 빈 라인들
        self.price_line, = self.ax_price.plot([], [], color=colors['price'], 
                                            linewidth=2, label='가격', alpha=0.9)
        self.long_signals = self.ax_price.scatter([], [], c=colors['long'], 
                                                marker='^', s=100, label='🚀 Long', 
                                                alpha=0.8, zorder=5)
        self.short_signals = self.ax_price.scatter([], [], c=colors['short'], 
                                                 marker='v', s=100, label='📉 Short', 
                                                 alpha=0.8, zorder=5)
        
        self.ax_price.legend(loc='upper left', facecolor=colors['bg'], 
                           edgecolor=colors['grid'], labelcolor=colors['text'])
    
    def setup_pnl_chart(self):
        """PNL 차트 설정"""
        self.ax_pnl.set_facecolor(colors['bg'])
        self.ax_pnl.grid(True, color=colors['grid'], alpha=0.3)
        self.ax_pnl.set_title('📈 누적 손익 (PNL)', 
                             color=colors['text'], fontsize=14, fontweight='bold')
        self.ax_pnl.set_ylabel('누적 PNL (%)', color=colors['text'])
        
        # 스타일링
        self.ax_pnl.spines['bottom'].set_color(colors['grid'])
        self.ax_pnl.spines['top'].set_color(colors['grid'])
        self.ax_pnl.spines['right'].set_color(colors['grid'])
        self.ax_pnl.spines['left'].set_color(colors['grid'])
        self.ax_pnl.tick_params(colors=colors['text'])
        
        # 제로 라인
        self.ax_pnl.axhline(y=0, color=colors['grid'], linestyle='--', alpha=0.5)
        
        # 초기 빈 라인
        self.pnl_line, = self.ax_pnl.plot([], [], color=colors['pnl_positive'], 
                                         linewidth=3, label='누적 PNL', alpha=0.9)
        self.pnl_fill = None  # 나중에 추가
        
        self.ax_pnl.legend(loc='upper left', facecolor=colors['bg'], 
                          edgecolor=colors['grid'], labelcolor=colors['text'])
    
    def setup_animation(self):
        """애니메이션 설정"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.current_frame = 0
        self.total_frames = 0
        
    def set_display_mode(self, mode):
        """표시 모드 변경"""
        if mode != self.display_mode:
            self.display_mode = mode
            self.setup_chart()
            if self.timestamps:  # 데이터가 있으면 다시 그리기
                self.update_chart_display()
    
    def update_data(self, timestamps, prices, signals, pnl):
        """차트 데이터 업데이트"""
        # 데이터 변환
        if isinstance(timestamps, (list, np.ndarray)):
            self.timestamps = list(timestamps)
        if isinstance(prices, (list, np.ndarray)):
            self.prices = list(prices)
        if isinstance(signals, (list, np.ndarray)):
            self.signals = list(signals)
        if isinstance(pnl, (list, np.ndarray)):
            self.pnl = list(pnl)
        
        # 최대 표시 포인트 제한
        if len(self.timestamps) > self.max_points_display:
            self.timestamps = self.timestamps[-self.max_points_display:]
            self.prices = self.prices[-self.max_points_display:]
            self.signals = self.signals[-self.max_points_display:]
            self.pnl = self.pnl[-self.max_points_display:]
        
        self.total_frames = len(self.timestamps)
        
        # 실시간 업데이트
        self.update_chart_display()
        
        # 통계 업데이트
        self.update_stats()
        
        # 정보 라벨 업데이트
        self.info_label.setText(f"🎬 실시간 백테스트 차트 - 데이터 포인트: {len(self.timestamps)}")
    
    def update_chart_display(self):
        """차트 표시 업데이트"""
        if not self.timestamps:
            return
            
        try:
            # 시간 데이터 처리
            if isinstance(self.timestamps[0], str):
                time_data = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in self.timestamps]
            else:
                time_data = self.timestamps
            
            # 가격 차트 업데이트
            if hasattr(self, 'ax_price'):
                self.update_price_chart(time_data)
            
            # PNL 차트 업데이트
            if hasattr(self, 'ax_pnl'):
                self.update_pnl_chart(time_data)
                
            self.canvas.draw()
            
        except Exception as e:
            print(f"차트 업데이트 오류: {e}")
    
    def update_price_chart(self, time_data):
        """가격 차트 업데이트"""
        # 가격 라인 업데이트
        self.price_line.set_data(time_data, self.prices)
        
        # 시그널 업데이트
        long_times = [time_data[i] for i in range(len(self.signals)) if self.signals[i] == 1]
        long_prices = [self.prices[i] for i in range(len(self.signals)) if self.signals[i] == 1]
        
        short_times = [time_data[i] for i in range(len(self.signals)) if self.signals[i] == -1]
        short_prices = [self.prices[i] for i in range(len(self.signals)) if self.signals[i] == -1]
        
        # 기존 scatter plot 제거하고 새로 그리기
        for collection in self.ax_price.collections:
            collection.remove()
        
        if long_times:
            self.ax_price.scatter(long_times, long_prices, c=colors['long'], 
                                marker='^', s=100, label='🚀 Long', alpha=0.8, zorder=5)
        if short_times:
            self.ax_price.scatter(short_times, short_prices, c=colors['short'], 
                                marker='v', s=100, label='📉 Short', alpha=0.8, zorder=5)
        
        # 축 범위 조정
        self.ax_price.relim()
        self.ax_price.autoscale_view()
        
        # x축 포맷팅
        if len(time_data) > 1:
            self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    def update_pnl_chart(self, time_data):
        """PNL 차트 업데이트"""
        if not self.pnl:
            return
            
        # PNL 라인 업데이트
        self.pnl_line.set_data(time_data, self.pnl)
        
        # PNL 색상 변경 (양수/음수에 따라)
        current_pnl = self.pnl[-1] if self.pnl else 0
        pnl_color = colors['pnl_positive'] if current_pnl >= 0 else colors['pnl_negative']
        self.pnl_line.set_color(pnl_color)
        
        # 영역 채우기 (기존 제거 후 새로 그리기)
        if hasattr(self, 'pnl_fill') and self.pnl_fill:
            self.pnl_fill.remove()
        
        zero_line = [0] * len(time_data)
        self.pnl_fill = self.ax_pnl.fill_between(time_data, self.pnl, zero_line, 
                                                alpha=0.3, color=pnl_color)
        
        # 축 범위 조정
        self.ax_pnl.relim()
        self.ax_pnl.autoscale_view()
        
        # x축 포맷팅
        if len(time_data) > 1:
            self.ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax_pnl.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    def update_stats(self):
        """통계 정보 업데이트"""
        if not self.pnl or not self.signals:
            self.stats_label.setText("통계: 대기 중...")
            return
        
        try:
            current_pnl = self.pnl[-1] if self.pnl else 0
            max_pnl = max(self.pnl) if self.pnl else 0
            min_pnl = min(self.pnl) if self.pnl else 0
            
            long_signals = sum(1 for s in self.signals if s == 1)
            short_signals = sum(1 for s in self.signals if s == -1)
            
            stats_text = (f"현재 PNL: {current_pnl:.2f}% | "
                         f"최고: {max_pnl:.2f}% | 최저: {min_pnl:.2f}% | "
                         f"Long: {long_signals} | Short: {short_signals}")
            
            self.stats_label.setText(stats_text)
            
            # PNL에 따른 색상 변경
            color = colors['pnl_positive'] if current_pnl >= 0 else colors['pnl_negative']
            self.stats_label.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")
            
        except Exception as e:
            self.stats_label.setText(f"통계 오류: {e}")
    
    def start_animation(self):
        """애니메이션 시작"""
        if not self.timer.isActive():
            self.timer.start(self.animation_speed)
    
    def stop_animation(self):
        """애니메이션 중지"""
        self.timer.stop()
    
    def update_animation(self):
        """애니메이션 프레임 업데이트"""
        # 실시간 업데이트이므로 별도 애니메이션 불필요
        pass
    
    def clear(self):
        """차트 초기화"""
        self.timestamps.clear()
        self.prices.clear()
        self.signals.clear()
        self.pnl.clear()
        
        self.current_frame = 0
        self.total_frames = 0
        
        self.setup_chart()
        self.info_label.setText("🎬 실시간 백테스트 차트 - 초기화됨")
        self.stats_label.setText("통계: 대기 중...")
        self.stats_label.setStyleSheet(f"color: {colors['text']}; padding: 5px;")
    
    def save_chart(self, filepath):
        """차트를 이미지로 저장"""
        self.figure.savefig(filepath, dpi=300, bbox_inches='tight', 
                           facecolor=colors['bg'], edgecolor='none')
    
    def set_animation_speed(self, speed_ms):
        """애니메이션 속도 설정"""
        self.animation_speed = speed_ms
        if self.timer.isActive():
            self.timer.setInterval(speed_ms)
    
    def set_max_points(self, max_points):
        """최대 표시 포인트 수 설정"""
        self.max_points_display = max_points
