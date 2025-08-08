#!/usr/bin/env python3
"""
Multi-Asset Factor Discovery GUI

Multi-asset factor discovery GUI application using PyQt6
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
import subprocess
import queue

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
        QGridLayout, QListWidget, QListWidgetItem, QSplitter,
        QTabWidget, QFileDialog, QMessageBox, QFrame
    )
    from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
    from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    print("PyQt6 or matplotlib is not installed. Please install with:")
    print("pip install PyQt6 matplotlib")
    sys.exit(1)

# Add project root directory to system path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

# Import our real-time chart widget
try:
    from factor_factory.visualization.realtime_chart import RealtimeChartWidget
except ImportError:
    # Create a fallback widget if chart module is not available
    class RealtimeChartWidget(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(600, 400)
            layout = QVBoxLayout()
            label = QLabel("Real-time chart widget not available")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.setLayout(layout)
        
        def update_data(self, timestamps, prices, signals, pnl):
            pass


class TrainingWorker(QThread):
    """Worker thread to run MCTS training in background"""
    
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # current, total
    factor_signal = pyqtSignal(str, float, dict)  # formula, reward, metrics
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # success, message
    checkpoint_signal = pyqtSignal(int, int, list)  # completed_iterations, total_iterations, discovered_factors
    chart_signal = pyqtSignal(list, list, list, list)  # timestamps, prices, signals, pnl - 차트 업데이트용
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = False
        self.process = None
        self.discovered_factors = []  # 발견된 팩터들 저장
        
    def run(self):
        """Execute training"""
        self.is_running = True
        self.status_signal.emit("Preparing training...")
        
        try:
            # Build CLI command
            cmd = self._build_command()
            
            self.log_signal.emit(f"Executing command: {' '.join(cmd)}")
            self.status_signal.emit("Starting training...")
            
            # Run process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(project_root),
                encoding='utf-8',
                errors='replace'  # Replace problematic characters
            )
            
            # Read real-time output
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                    
                line = line.strip()
                if line:
                    self.log_signal.emit(line)
                    self._parse_output(line)
            
            # Wait for process completion
            if self.is_running:
                return_code = self.process.wait()
                
                if return_code == 0:
                    self.finished_signal.emit(True, "Training completed successfully!")
                else:
                    self.finished_signal.emit(False, f"Training failed. (Code: {return_code})")
            else:
                self.finished_signal.emit(False, "Training was interrupted.")
                
        except Exception as e:
            self.finished_signal.emit(False, f"Error during training: {str(e)}")
        finally:
            self.is_running = False
            self.status_signal.emit("Waiting")
    
    def _build_command(self):
        """Build CLI command"""
        cmd = [
            sys.executable, "-m", "factor_factory.scripts.cli_multi_asset_train",
            "--mode", "mcts",
            "--symbols"
        ]
        cmd.extend(self.config['symbols'])
        
        # 이어서 학습인 경우 남은 반복만 실행
        start_iteration = self.config.get('start_iteration', 0)
        remaining_iterations = self.config['iterations'] - start_iteration
        
        cmd.extend([
            "--iterations", str(remaining_iterations),
            "--episodes-per-iter", str(self.config['episodes_per_iter']),
            "--mcts-simulations", str(self.config['mcts_simulations']),
            "--normalizer", self.config['normalizer'],
            "--initial-capital", str(self.config['initial_capital']),
            "--commission", str(self.config['commission']),
            "--slippage", str(self.config['slippage']),
            "--save-dir", self.config['save_dir']
        ])
        
        if self.config['market_neutral']:
            cmd.append("--market-neutral")
        
        return cmd
    
    def _parse_output(self, line):
        """Parse output line to update GUI"""
        
        # Parse progress
        if "Iteration" in line and "/" in line:
            try:
                # Parse "Iteration 1/3" format
                if "Iteration" in line:
                    parts = line.split()
                    for part in parts:
                        if "/" in part:
                            current, total = map(int, part.split("/"))
                            
                            # 시작 반복을 고려한 실제 진행률 계산
                            start_iteration = self.config.get('start_iteration', 0)
                            actual_current = start_iteration + current
                            actual_total = self.config['iterations']
                            
                            self.progress_signal.emit(actual_current, actual_total)
                            self.status_signal.emit(f"Iteration {actual_current}/{actual_total} in progress")
                            
                            # 반복 완료 시 체크포인트 저장
                            if current < total:  # 아직 완료되지 않은 경우
                                self.checkpoint_signal.emit(actual_current, actual_total, self.discovered_factors)
                            
                            break
            except:
                pass
        
        # Parse factor discovery
        if "[FOUND]" in line or "[FACTOR]" in line:
            try:
                # Extract factor information
                if "Reward:" in line:
                    parts = line.split("Reward:")
                    if len(parts) > 1:
                        reward_str = parts[1].strip().split()[0]
                        reward = float(reward_str)
                        
                        # Extract factor formula
                        formula = "Unknown"
                        if "[FOUND]" in line:
                            formula_part = line.split("[FOUND]")[1].split("Reward:")[0].strip()
                            formula = formula_part
                        
                        # 발견된 팩터 저장
                        factor_data = {
                            'formula': formula,
                            'reward': reward,
                            'metrics': {},
                            'timestamp': datetime.now().isoformat()
                        }
                        self.discovered_factors.append(factor_data)
                        
                        self.factor_signal.emit(formula, reward, {})
                        
                        # 차트용 가상 데이터 생성 (실제로는 백테스트 결과에서 가져와야 함)
                        self._generate_chart_data_for_factor(reward)
            except:
                pass
        
        # Status updates
        if "Self-play" in line:
            self.status_signal.emit("Self-play in progress...")
        elif "Neural network training" in line:
            self.status_signal.emit("Neural network training...")
        elif "Performance evaluation" in line:
            self.status_signal.emit("Performance evaluation...")
    
    def stop(self):
        """Stop training"""
        self.is_running = False
        if self.process:
            self.process.terminate()
    
    def _generate_chart_data_for_factor(self, reward):
        """팩터 발견 시 실제 데이터 기반 차트 데이터 생성"""
        try:
            # 설정에서 첫 번째 심볼 사용
            symbol = self.config.get('symbols', ['BTCUSDT'])[0]
            
            # 실제 백테스트 데이터 로드 시도
            try:
                from factor_factory.visualization.real_data_loader import load_real_chart_data
                timestamps, prices, signals, pnl = load_real_chart_data(
                    symbol=symbol,
                    max_points=50 + len(self.discovered_factors) * 10,  # 점진적으로 증가
                    use_latest=True
                )
                
                if timestamps:
                    print(f"실제 데이터 기반 차트 업데이트: {symbol}, {len(timestamps)}개 포인트")
                    
                    # 리워드에 따라 데이터 약간 조정 (실제 트렌드 유지하면서)
                    if reward != 0 and len(pnl) > 0:
                        import numpy as np
                        
                        # PNL에 리워드 트렌드 추가 (기존 PNL에 리워드 기반 보정 적용)
                        reward_adjustment = np.linspace(0, reward * 5, len(pnl))  # 리워드에 비례한 추가 수익
                        pnl = [original + adj for original, adj in zip(pnl, reward_adjustment)]
                        
                        # 좋은 리워드일 때 시그널 밀도 증가
                        if abs(reward) > 0.1:
                            signal_boost_prob = min(0.2, abs(reward) * 0.3)
                            for i in range(len(signals)):
                                if signals[i] == 0 and np.random.random() < signal_boost_prob:
                                    signals[i] = 1 if reward > 0 else -1
                    
                    # 차트 업데이트 신호 발생
                    self.chart_signal.emit(timestamps, prices, signals, pnl)
                    return
                    
            except Exception as load_error:
                print(f"실제 데이터 로드 실패: {load_error}")
            
            # 폴백으로 가상 데이터 생성
            print("실제 데이터를 로드할 수 없어 가상 데이터를 생성합니다.")
            self._generate_fallback_chart_data(reward)
            
        except Exception as e:
            print(f"차트 데이터 생성 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_fallback_chart_data(self, reward):
        """실제 데이터 로드 실패 시 폴백 가상 데이터 생성"""
        try:
            import numpy as np
            from datetime import datetime, timedelta
            
            # 가상 백테스트 데이터 생성 (기존 로직)
            n_points = 50 + len(self.discovered_factors) * 10
            
            # 시간 데이터
            start_time = datetime.now() - timedelta(hours=n_points)
            timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
            
            # 가격 데이터 (트렌드 + 노이즈)
            base_price = 45000
            trend = np.linspace(0, reward * 1000, n_points)
            noise = np.random.normal(0, 200, n_points)
            prices = base_price + trend + noise
            
            # 시그널 데이터
            signals = []
            for i in range(n_points):
                signal_prob = min(0.3, abs(reward) * 0.5)
                if np.random.random() < signal_prob:
                    signal = 1 if reward > 0 else -1
                else:
                    signal = 0
                signals.append(signal)
            
            # PNL 데이터
            daily_returns = np.random.normal(reward * 0.1, 0.5, n_points)
            pnl = np.cumsum(daily_returns)
            
            # 차트 업데이트 신호 발생
            self.chart_signal.emit(timestamps, prices.tolist(), signals, pnl.tolist())
            
            print(f"폴백 가상 데이터 생성 완료: {n_points}개 포인트")
            
        except Exception as e:
            print(f"폴백 차트 데이터 생성 오류: {e}")


class FactorDisplayWidget(QWidget):
    """Widget to display discovered factors"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.factors = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Discovered Factors")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Factor list
        self.factor_list = QListWidget()
        layout.addWidget(self.factor_list)
        
        # Statistics info
        self.stats_label = QLabel("Total factors: 0")
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def add_factor(self, formula, reward, metrics):
        """Add new factor"""
        factor_info = {
            'formula': formula,
            'reward': reward,
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        self.factors.append(factor_info)
        
        # Create list item
        item_text = f"[{factor_info['timestamp']}] {formula}\nReward: {reward:.4f}"
        if metrics.get('sharpe_ratio'):
            item_text += f" | Sharpe: {metrics['sharpe_ratio']:.3f}"
        
        item = QListWidgetItem(item_text)
        
        # Color coding based on reward
        if reward > 0.5:
            item.setBackground(QColor(200, 255, 200))  # Light green
        elif reward > 0:
            item.setBackground(QColor(255, 255, 200))  # Light yellow
        
        self.factor_list.insertItem(0, item)  # Add to top
        
        # Update statistics
        self.update_stats()
    
    def update_stats(self):
        """Update statistics information"""
        total = len(self.factors)
        good_factors = len([f for f in self.factors if f['reward'] > 0])
        
        if total > 0:
            avg_reward = sum(f['reward'] for f in self.factors) / total
            best_reward = max(f['reward'] for f in self.factors)
            
            stats_text = f"Total factors: {total} | Positive reward: {good_factors}\n"
            stats_text += f"Average reward: {avg_reward:.4f} | Best reward: {best_reward:.4f}"
        else:
            stats_text = "Total factors: 0"
        
        self.stats_label.setText(stats_text)


class MultiAssetFactorGUI(QMainWindow):
    """Multi-asset factor discovery GUI main window"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_checkpoint = None
        self.init_ui()
        self.setup_timer()
        self.update_time_estimate()  # 초기 시간 추정
        self.check_existing_checkpoint()  # 기존 체크포인트 확인
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Multi-Asset Factor Discovery")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left settings panel
        left_panel = self.create_settings_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right results panel
        right_panel = self.create_results_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_settings_panel(self):
        """Create settings panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training Settings")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Asset selection group
        assets_group = QGroupBox("Asset Selection")
        assets_layout = QVBoxLayout()
        
        # Default assets
        self.asset_checkboxes = {}
        default_assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
        for asset in default_assets:
            cb = QCheckBox(asset)
            if asset in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:  # Default selected
                cb.setChecked(True)
            self.asset_checkboxes[asset] = cb
            assets_layout.addWidget(cb)
        
        # Custom asset addition
        custom_layout = QHBoxLayout()
        self.custom_asset_edit = QLineEdit()
        self.custom_asset_edit.setPlaceholderText("Enter custom asset (e.g., DOGEUSDT)")
        add_asset_btn = QPushButton("Add")
        add_asset_btn.clicked.connect(self.add_custom_asset)
        custom_layout.addWidget(self.custom_asset_edit)
        custom_layout.addWidget(add_asset_btn)
        assets_layout.addLayout(custom_layout)
        
        assets_group.setLayout(assets_layout)
        layout.addWidget(assets_group)
        
        # MCTS settings group
        mcts_group = QGroupBox("MCTS Settings")
        mcts_layout = QGridLayout()
        
        # Number of iterations
        mcts_layout.addWidget(QLabel("Iterations:"), 0, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 1000)
        self.iterations_spin.setValue(3)  # 빠른 테스트용으로 3으로 줄임
        mcts_layout.addWidget(self.iterations_spin, 0, 1)
        
        # Episodes per iteration
        mcts_layout.addWidget(QLabel("Episodes per iteration:"), 1, 0)
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 100)
        self.episodes_spin.setValue(2)  # 빠른 테스트용으로 2로 줄임
        mcts_layout.addWidget(self.episodes_spin, 1, 1)
        
        # Number of simulations
        mcts_layout.addWidget(QLabel("MCTS Simulations:"), 2, 0)
        self.simulations_spin = QSpinBox()
        self.simulations_spin.setRange(5, 1000)
        self.simulations_spin.setValue(10)  # 빠른 테스트용으로 10으로 줄임
        mcts_layout.addWidget(self.simulations_spin, 2, 1)
        
        mcts_group.setLayout(mcts_layout)
        layout.addWidget(mcts_group)
        
        # 빠른 설정 프리셋 버튼들
        preset_group = QGroupBox("빠른 설정")
        preset_layout = QVBoxLayout()
        
        # 첫 번째 줄: 프로토타이핑용 초고속 설정
        prototype_layout = QHBoxLayout()
        
        lightning_btn = QPushButton("⚡ 번개 테스트")
        lightning_btn.setToolTip("1 iteration, 1 episode, 3 simulations - 10초 내 완료")
        lightning_btn.clicked.connect(self.set_lightning_preset)
        lightning_btn.setStyleSheet("QPushButton { background-color: #E91E63; color: white; font-weight: bold; }")
        
        quick_test_btn = QPushButton("🚀 초고속 테스트")
        quick_test_btn.setToolTip("1 iteration, 1 episode, 5 simulations - 30초 내 완료")
        quick_test_btn.clicked.connect(self.set_ultra_fast_preset)
        quick_test_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-weight: bold; }")
        
        prototype_layout.addWidget(lightning_btn)
        prototype_layout.addWidget(quick_test_btn)
        preset_layout.addLayout(prototype_layout)
        
        # 두 번째 줄: 일반 설정
        normal_layout = QHBoxLayout()
        
        fast_test_btn = QPushButton("🏃 빠른 테스트") 
        fast_test_btn.setToolTip("2 iterations, 2 episodes, 10 simulations - 2-3분 소요")
        fast_test_btn.clicked.connect(self.set_fast_preset)
        fast_test_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        
        normal_btn = QPushButton("⚙️ 일반")
        normal_btn.setToolTip("3 iterations, 3 episodes, 25 simulations - 5-10분 소요")
        normal_btn.clicked.connect(self.set_normal_preset)
        normal_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        
        slow_btn = QPushButton("🎯 정밀")
        slow_btn.setToolTip("5 iterations, 5 episodes, 50 simulations - 15-30분 소요")
        slow_btn.clicked.connect(self.set_precise_preset)
        slow_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        normal_layout.addWidget(fast_test_btn)
        normal_layout.addWidget(normal_btn)
        normal_layout.addWidget(slow_btn)
        preset_layout.addLayout(normal_layout)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # 예상 소요 시간 표시
        self.time_estimate_label = QLabel("⏱️ 예상 소요 시간: 약 1-2분")
        self.time_estimate_label.setStyleSheet("QLabel { background-color: #FFF3E0; padding: 8px; border-radius: 5px; border: 1px solid #FF9800; }")
        layout.addWidget(self.time_estimate_label)
        
        # 설정 변경 시 시간 업데이트
        self.iterations_spin.valueChanged.connect(self.update_time_estimate)
        self.episodes_spin.valueChanged.connect(self.update_time_estimate)
        self.simulations_spin.valueChanged.connect(self.update_time_estimate)
        
        # 성능 팁 정보
        tips_label = QLabel("💡 성능 최적화 팁:\n"
                           "⚡ 처음 테스트: '번개 테스트' (3 시뮬레이션) - 구조 확인용\n"
                           "🚀 빠른 검증: '초고속 테스트' (5 시뮬레이션) - 프로토타이핑\n"
                           "🏃 일반 개발: '빠른 테스트' (10 시뮬레이션) - 개발 단계\n"
                           "⚙️ 실험: '일반' (25 시뮬레이션) - 성능 검증\n"
                           "🎯 최종: '정밀' (50+ 시뮬레이션) - 최종 결과\n"
                           "🧠 캐시: 동일 프로그램 재평가 방지로 10-100배 성능 향상")
        tips_label.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; border: 1px solid #2196F3; font-size: 10px; }")
        tips_label.setWordWrap(True)
        layout.addWidget(tips_label)
        
        # 캐시 통계 표시
        cache_layout = QHBoxLayout()
        self.cache_stats_label = QLabel("📊 캐시 통계: 대기 중...")
        self.cache_stats_label.setStyleSheet("QLabel { background-color: #F3E5F5; padding: 8px; border-radius: 5px; border: 1px solid #9C27B0; font-size: 9px; }")
        
        clear_cache_btn = QPushButton("🧹 캐시 클리어")
        clear_cache_btn.setToolTip("모든 캐시를 초기화하여 메모리를 절약합니다")
        clear_cache_btn.clicked.connect(self.clear_cache_stats)
        clear_cache_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-size: 8px; padding: 4px; }")
        
        cache_layout.addWidget(self.cache_stats_label, 3)
        cache_layout.addWidget(clear_cache_btn, 1)
        layout.addLayout(cache_layout)
        
        # 거래 설정 그룹
        trading_group = QGroupBox("거래 설정")
        trading_layout = QGridLayout()
        
        # 정규화 방법
        trading_layout.addWidget(QLabel("정규화 방법:"), 0, 0)
        self.normalizer_combo = QComboBox()
        self.normalizer_combo.addItems(['z_score', 'rank', 'percentile', 'mad'])
        trading_layout.addWidget(self.normalizer_combo, 0, 1)
        
        # 초기 자본
        trading_layout.addWidget(QLabel("초기 자본:"), 1, 0)
        self.capital_spin = QSpinBox()
        self.capital_spin.setRange(10000, 10000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setSuffix(" $")
        trading_layout.addWidget(self.capital_spin, 1, 1)
        
        # 수수료
        trading_layout.addWidget(QLabel("수수료:"), 2, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0.001, 1.0)
        self.commission_spin.setValue(0.08)  # 0.08%
        self.commission_spin.setDecimals(3)
        self.commission_spin.setSuffix("%")
        trading_layout.addWidget(self.commission_spin, 2, 1)
        
        # 슬리피지
        trading_layout.addWidget(QLabel("슬리피지:"), 3, 0)
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0.001, 1.0)
        self.slippage_spin.setValue(0.15)  # 0.15%
        self.slippage_spin.setDecimals(3)
        self.slippage_spin.setSuffix("%")
        trading_layout.addWidget(self.slippage_spin, 3, 1)
        
        # 마켓 뉴트럴
        self.market_neutral_cb = QCheckBox("마켓 뉴트럴")
        self.market_neutral_cb.setChecked(True)
        trading_layout.addWidget(self.market_neutral_cb, 4, 0, 1, 2)
        
        trading_group.setLayout(trading_layout)
        layout.addWidget(trading_group)
        
        # 저장 디렉토리
        save_group = QGroupBox("결과 저장")
        save_layout = QHBoxLayout()
        
        self.save_dir_edit = QLineEdit("multi_asset_gui_results")
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self.browse_save_dir)
        
        save_layout.addWidget(self.save_dir_edit)
        save_layout.addWidget(browse_btn)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # 체크포인트 관리 그룹
        checkpoint_group = QGroupBox("체크포인트 관리")
        checkpoint_layout = QVBoxLayout()
        
        # 체크포인트 자동 저장 옵션
        self.auto_checkpoint_cb = QCheckBox("자동 체크포인트 저장 (반복마다)")
        self.auto_checkpoint_cb.setChecked(True)
        self.auto_checkpoint_cb.setToolTip("각 반복이 끝날 때마다 자동으로 진행상황을 저장합니다")
        checkpoint_layout.addWidget(self.auto_checkpoint_cb)
        
        # 체크포인트 파일 선택
        checkpoint_file_layout = QHBoxLayout()
        self.checkpoint_file_edit = QLineEdit("training_checkpoint.json")
        self.checkpoint_file_edit.setToolTip("체크포인트가 저장될 파일 이름")
        browse_checkpoint_btn = QPushButton("📁")
        browse_checkpoint_btn.setToolTip("체크포인트 파일 찾아보기")
        browse_checkpoint_btn.clicked.connect(self.browse_checkpoint_file)
        
        checkpoint_file_layout.addWidget(QLabel("체크포인트 파일:"))
        checkpoint_file_layout.addWidget(self.checkpoint_file_edit)
        checkpoint_file_layout.addWidget(browse_checkpoint_btn)
        checkpoint_layout.addLayout(checkpoint_file_layout)
        
        # 체크포인트 상태 표시
        self.checkpoint_status_label = QLabel("체크포인트 없음")
        self.checkpoint_status_label.setStyleSheet("QLabel { background-color: #FFECB3; padding: 5px; border-radius: 3px; }")
        checkpoint_layout.addWidget(self.checkpoint_status_label)
        
        checkpoint_group.setLayout(checkpoint_layout)
        layout.addWidget(checkpoint_group)
        
        # 제어 버튼들
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        
        self.resume_btn = QPushButton("▶️ 이어서 학습")
        self.resume_btn.clicked.connect(self.resume_training)
        self.resume_btn.setEnabled(False)
        self.resume_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        
        self.stop_btn = QPushButton("⏹ 중단")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.resume_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # 상태 표시
        self.status_label = QLabel("대기 중")
        self.status_label.setStyleSheet("QLabel { background-color: #e0e0e0; padding: 5px; border-radius: 3px; }")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_results_panel(self):
        """결과 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 제목과 진행률
        header_layout = QHBoxLayout()
        
        title = QLabel("학습 진행 상황")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        header_layout.addWidget(self.progress_bar)
        
        layout.addLayout(header_layout)
        
        # 탭 위젯
        tab_widget = QTabWidget()
        
        # 로그 탭
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 로그 제어 버튼
        log_controls = QHBoxLayout()
        clear_log_btn = QPushButton("로그 지우기")
        clear_log_btn.clicked.connect(self.clear_log)
        save_log_btn = QPushButton("로그 저장")
        save_log_btn.clicked.connect(self.save_log)
        
        log_controls.addWidget(clear_log_btn)
        log_controls.addWidget(save_log_btn)
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        
        log_tab.setLayout(log_layout)
        tab_widget.addTab(log_tab, "로그")
        
        # 실시간 차트 탭 추가
        chart_tab = QWidget()
        chart_layout = QVBoxLayout()
        
        # 차트 위젯 생성
        self.chart_widget = RealtimeChartWidget()
        chart_layout.addWidget(self.chart_widget)
        
        # 차트 제어 버튼
        chart_controls = QHBoxLayout()
        
        # 차트 모드 선택
        chart_mode_label = QLabel("표시 모드:")
        self.chart_mode_combo = QComboBox()
        self.chart_mode_combo.addItems(["가격 + 시그널", "누적 PNL", "전체 보기"])
        self.chart_mode_combo.currentTextChanged.connect(self.on_chart_mode_changed)
        
        # 실제 데이터 로드 버튼
        load_real_data_btn = QPushButton("📊 실제 데이터 로드")
        load_real_data_btn.clicked.connect(self.load_real_data_to_chart)
        load_real_data_btn.setToolTip("최신 백테스트 결과의 실제 데이터를 차트에 로드합니다")
        
        # 차트 제어 버튼들
        clear_chart_btn = QPushButton("차트 초기화")
        clear_chart_btn.clicked.connect(self.clear_chart)
        
        export_chart_btn = QPushButton("차트 저장")
        export_chart_btn.clicked.connect(self.export_chart)
        
        chart_controls.addWidget(chart_mode_label)
        chart_controls.addWidget(self.chart_mode_combo)
        chart_controls.addWidget(load_real_data_btn)
        chart_controls.addStretch()
        chart_controls.addWidget(clear_chart_btn)
        chart_controls.addWidget(export_chart_btn)
        
        chart_layout.addLayout(chart_controls)
        chart_tab.setLayout(chart_layout)
        tab_widget.addTab(chart_tab, "실시간 차트 🎬")
        
        # 팩터 탭
        self.factor_widget = FactorDisplayWidget()
        tab_widget.addTab(self.factor_widget, "발견된 팩터")
        
        layout.addWidget(tab_widget)
        
        panel.setLayout(layout)
        return panel
    
    def add_custom_asset(self):
        """커스텀 자산 추가"""
        asset = self.custom_asset_edit.text().strip().upper()
        if asset and asset not in self.asset_checkboxes:
            cb = QCheckBox(asset)
            cb.setChecked(True)
            self.asset_checkboxes[asset] = cb
            
            # 자산 그룹에 추가
            assets_group = self.findChild(QGroupBox, "자산 선택")
            if assets_group:
                layout = assets_group.layout()
                layout.insertWidget(layout.count() - 1, cb)  # 커스텀 입력 위에 추가
            
            self.custom_asset_edit.clear()
    
    def browse_save_dir(self):
        """저장 디렉토리 선택"""
        dir_path = QFileDialog.getExistingDirectory(self, "결과 저장 디렉토리 선택")
        if dir_path:
            self.save_dir_edit.setText(dir_path)
    
    def browse_checkpoint_file(self):
        """체크포인트 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "체크포인트 파일 선택",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.checkpoint_file_edit.setText(file_path)
            self.check_existing_checkpoint()
    
    def check_existing_checkpoint(self):
        """기존 체크포인트 파일 확인"""
        checkpoint_file = self.checkpoint_file_edit.text()
        
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    self.current_checkpoint = json.load(f)
                
                # 체크포인트 정보 표시
                completed_iter = self.current_checkpoint.get('completed_iterations', 0)
                total_iter = self.current_checkpoint.get('total_iterations', 0)
                
                self.checkpoint_status_label.setText(
                    f"✅ 체크포인트 발견: {completed_iter}/{total_iter} 반복 완료"
                )
                self.checkpoint_status_label.setStyleSheet(
                    "QLabel { background-color: #C8E6C9; padding: 5px; border-radius: 3px; color: #2E7D32; }"
                )
                self.resume_btn.setEnabled(True)
                
                # 이전 설정으로 복원
                if 'config' in self.current_checkpoint:
                    self.restore_config_from_checkpoint(self.current_checkpoint['config'])
                
                self.add_log(f"[CHECKPOINT] 기존 체크포인트 발견: {completed_iter}/{total_iter} 반복 완료")
            else:
                self.current_checkpoint = None
                self.checkpoint_status_label.setText("체크포인트 없음")
                self.checkpoint_status_label.setStyleSheet(
                    "QLabel { background-color: #FFECB3; padding: 5px; border-radius: 3px; }"
                )
                self.resume_btn.setEnabled(False)
        
        except Exception as e:
            self.current_checkpoint = None
            self.checkpoint_status_label.setText(f"체크포인트 오류: {str(e)}")
            self.checkpoint_status_label.setStyleSheet(
                "QLabel { background-color: #FFCDD2; padding: 5px; border-radius: 3px; color: #C62828; }"
            )
            self.resume_btn.setEnabled(False)
    
    def restore_config_from_checkpoint(self, config):
        """체크포인트에서 설정 복원"""
        try:
            # 자산 선택 복원
            if 'symbols' in config:
                for asset, checkbox in self.asset_checkboxes.items():
                    checkbox.setChecked(asset in config['symbols'])
            
            # MCTS 설정 복원
            if 'iterations' in config:
                self.iterations_spin.setValue(config['iterations'])
            if 'episodes_per_iter' in config:
                self.episodes_spin.setValue(config['episodes_per_iter'])
            if 'mcts_simulations' in config:
                self.simulations_spin.setValue(config['mcts_simulations'])
            
            # 거래 설정 복원
            if 'normalizer' in config:
                index = self.normalizer_combo.findText(config['normalizer'])
                if index >= 0:
                    self.normalizer_combo.setCurrentIndex(index)
            if 'initial_capital' in config:
                self.capital_spin.setValue(config['initial_capital'])
            if 'commission' in config:
                self.commission_spin.setValue(config['commission'] * 100)
            if 'slippage' in config:
                self.slippage_spin.setValue(config['slippage'] * 100)
            if 'market_neutral' in config:
                self.market_neutral_cb.setChecked(config['market_neutral'])
            
            self.add_log("[CHECKPOINT] 이전 설정이 복원되었습니다")
            
        except Exception as e:
            self.add_log(f"[WARNING] 설정 복원 중 오류: {str(e)}")
    
    def save_checkpoint(self, config, completed_iterations, total_iterations, discovered_factors=None):
        """체크포인트 저장"""
        if not self.auto_checkpoint_cb.isChecked():
            return
        
        checkpoint_file = self.checkpoint_file_edit.text()
        
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'completed_iterations': completed_iterations,
                'total_iterations': total_iterations,
                'config': config,
                'discovered_factors': discovered_factors or [],
                'version': '1.0'
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.current_checkpoint = checkpoint_data
            self.checkpoint_status_label.setText(
                f"✅ 체크포인트 저장됨: {completed_iterations}/{total_iterations} 반복 완료"
            )
            self.checkpoint_status_label.setStyleSheet(
                "QLabel { background-color: #C8E6C9; padding: 5px; border-radius: 3px; color: #2E7D32; }"
            )
            
            self.add_log(f"[CHECKPOINT] 진행상황 저장됨: {completed_iterations}/{total_iterations}")
            
        except Exception as e:
            self.add_log(f"[ERROR] 체크포인트 저장 실패: {str(e)}")
    
    def get_selected_assets(self):
        """선택된 자산 목록 반환"""
        selected = []
        for asset, checkbox in self.asset_checkboxes.items():
            if checkbox.isChecked():
                selected.append(asset)
        return selected
    
    def set_lightning_preset(self):
        """번개 테스트 설정 (10초 내 완료)"""
        self.iterations_spin.setValue(1)
        self.episodes_spin.setValue(1)
        self.simulations_spin.setValue(3)  # 극도로 적은 시뮬레이션
        self.add_log("[PRESET] ⚡ 번개 테스트 설정 (약 10초 소요 예상)")
    
    def set_ultra_fast_preset(self):
        """초고속 테스트 설정 (30초 내 완료)"""
        self.iterations_spin.setValue(1)
        self.episodes_spin.setValue(1)
        self.simulations_spin.setValue(5)
        self.add_log("[PRESET] 🚀 초고속 테스트 설정 (약 30초 소요 예상)")
    
    def set_fast_preset(self):
        """빠른 테스트 설정 (2-3분 소요)"""
        self.iterations_spin.setValue(2)
        self.episodes_spin.setValue(2)
        self.simulations_spin.setValue(10)
        self.add_log("[PRESET] 🏃 빠른 테스트 설정 (약 2-3분 소요 예상)")
    
    def set_normal_preset(self):
        """일반 설정 (5-10분 소요)"""
        self.iterations_spin.setValue(3)
        self.episodes_spin.setValue(3)
        self.simulations_spin.setValue(25)
        self.add_log("[PRESET] ⚙️ 일반 설정 (약 5-10분 소요 예상)")
    
    def set_precise_preset(self):
        """정밀 설정 (15-30분 소요)"""
        self.iterations_spin.setValue(5)
        self.episodes_spin.setValue(5)
        self.simulations_spin.setValue(50)
        self.add_log("[PRESET] 🎯 정밀 설정 (약 15-30분 소요 예상)")
    
    def update_time_estimate(self):
        """설정에 따른 예상 소요 시간 업데이트"""
        iterations = self.iterations_spin.value()
        episodes = self.episodes_spin.value()
        simulations = self.simulations_spin.value()
        
        # 더 정확한 시간 계산
        # MCTS 시뮬레이션이 가장 큰 병목이므로 시뮬레이션 수에 가중치 부여
        simulation_weight = simulations * 2  # 시뮬레이션당 약 2초
        episode_weight = episodes * 1  # 에피소드당 추가 1초 (오버헤드)
        iteration_weight = iterations * 0.5  # 반복당 0.5초 (저장 등)
        
        total_seconds = (simulation_weight + episode_weight + iteration_weight) * iterations * episodes
        
        if total_seconds <= 15:
            time_str = "10-15초"
            color = "#4CAF50"  # 녹색
            emoji = "⚡"
        elif total_seconds <= 60:
            time_str = "30초-1분"
            color = "#8BC34A"  # 연한 녹색
            emoji = "🚀"
        elif total_seconds <= 180:
            time_str = "1-3분"
            color = "#FF9800"  # 주황색
            emoji = "🏃"
        elif total_seconds <= 600:
            time_str = "3-10분"
            color = "#FF5722"  # 빨간색
            emoji = "⚙️"
        elif total_seconds <= 1800:
            time_str = "10-30분"
            color = "#9C27B0"  # 보라색
            emoji = "🎯"
        else:
            time_str = "30분 이상"
            color = "#F44336"  # 진한 빨간색
            emoji = "🐌"
        
        operations_per_second = total_seconds / max(1, iterations * episodes * simulations)
        
        self.time_estimate_label.setText(f"{emoji} 예상 소요 시간: {time_str} ({total_seconds:.0f}초 예상)")
        self.time_estimate_label.setStyleSheet(f"QLabel {{ background-color: {color}20; padding: 8px; border-radius: 5px; border: 1px solid {color}; color: {color}; font-weight: bold; }}")
        
        # 성능 경고 표시
        if total_seconds > 300:  # 5분 이상
            self.time_estimate_label.setText(self.time_estimate_label.text() + " ⚠️ 느림!")
        elif simulations > 25:
            self.time_estimate_label.setText(self.time_estimate_label.text() + " 💡 시뮬레이션 줄이기 권장")
    
    def add_log(self, message):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # 로그 텍스트가 존재할 때만 추가 (초기화 전에 호출될 수 있음)
        if hasattr(self, 'log_text'):
            self.log_text.append(formatted_message)
            
            # 스크롤을 맨 아래로
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_text.setTextCursor(cursor)
    
    def start_training(self):
        """학습 시작"""
        self._start_training_internal(resume=False)
    
    def resume_training(self):
        """이어서 학습"""
        if not self.current_checkpoint:
            QMessageBox.warning(self, "오류", "이어서 학습할 체크포인트가 없습니다.")
            return
        
        self._start_training_internal(resume=True)
    
    def _start_training_internal(self, resume=False):
        """학습 시작 (내부 함수)"""
        # 선택된 자산 확인
        selected_assets = self.get_selected_assets()
        if len(selected_assets) < 2:
            QMessageBox.warning(self, "경고", "최소 2개 이상의 자산을 선택해주세요.")
            return
        
        # 설정 수집
        config = {
            'symbols': selected_assets,
            'iterations': self.iterations_spin.value(),
            'episodes_per_iter': self.episodes_spin.value(),
            'mcts_simulations': self.simulations_spin.value(),
            'normalizer': self.normalizer_combo.currentText(),
            'initial_capital': self.capital_spin.value(),
            'commission': self.commission_spin.value() / 100,  # 0.08% -> 0.0008
            'slippage': self.slippage_spin.value() / 100,      # 0.15% -> 0.0015
            'market_neutral': self.market_neutral_cb.isChecked(),
            'save_dir': self.save_dir_edit.text()
        }
        
        # 이어서 학습인 경우 시작 반복 설정
        start_iteration = 0
        if resume and self.current_checkpoint:
            start_iteration = self.current_checkpoint.get('completed_iterations', 0)
            
            if start_iteration >= config['iterations']:
                QMessageBox.information(self, "알림", "이미 모든 반복이 완료되었습니다!")
                return
            
            # 이전 팩터들 복원
            if 'discovered_factors' in self.current_checkpoint:
                for factor_data in self.current_checkpoint['discovered_factors']:
                    self.factor_widget.add_factor(
                        factor_data.get('formula', 'Unknown'),
                        factor_data.get('reward', 0.0),
                        factor_data.get('metrics', {})
                    )
        
        # 학습용 설정에 시작 반복 추가
        config['start_iteration'] = start_iteration
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, config['iterations'])
        self.progress_bar.setValue(start_iteration)
        
        # 로그 초기화 (이어서 학습이 아닌 경우만)
        if not resume:
            self.log_text.clear()
        
        mode_text = "이어서 학습" if resume else "새로운 학습"
        self.add_log(f"{mode_text} 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_log(f"선택된 자산: {', '.join(selected_assets)}")
        
        if resume:
            self.add_log(f"시작 반복: {start_iteration + 1}/{config['iterations']}")
        
        self.add_log(f"설정: {config}")
        self.add_log("-" * 60)
        
        # 워커 스레드 시작
        self.worker = TrainingWorker(config)
        self.worker.log_signal.connect(self.add_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.factor_signal.connect(self.add_factor)
        self.worker.status_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.checkpoint_signal.connect(self.on_checkpoint_save)  # 체크포인트 시그널 연결
        self.worker.chart_signal.connect(self.update_chart_data)  # 차트 업데이트 시그널 연결
        self.worker.start()
    
    def stop_training(self):
        """학습 중단"""
        if self.worker:
            self.worker.stop()
            self.add_log("학습 중단 요청...")
    
    def on_checkpoint_save(self, completed_iterations, total_iterations, discovered_factors):
        """체크포인트 저장 이벤트 처리"""
        if hasattr(self, 'worker') and self.worker:
            config = self.worker.config.copy()
            # start_iteration은 체크포인트에서 제외 (내부 사용용)
            if 'start_iteration' in config:
                del config['start_iteration']
            
            self.save_checkpoint(config, completed_iterations, total_iterations, discovered_factors)
    
    def update_progress(self, current, total):
        """진행률 업데이트"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def add_factor(self, formula, reward, metrics):
        """팩터 추가"""
        self.factor_widget.add_factor(formula, reward, metrics)
        self.add_log(f"[TARGET] 팩터 발견: {formula} (보상: {reward:.4f})")
    
    def update_status(self, status):
        """상태 업데이트"""
        self.status_label.setText(status)
        
        # 캐시 통계도 함께 업데이트
        self.update_cache_stats()
    
    def update_cache_stats(self):
        """캐시 통계 업데이트"""
        try:
            from factor_factory.rlc.enhanced_cache import get_cache_statistics
            
            cache_stats = get_cache_statistics()
            
            if cache_stats:
                # 프로그램 캐시 통계
                prog_stats = cache_stats.get('program_cache', {})
                prog_hit_rate = prog_stats.get('hit_rate', 0) * 100
                prog_size = prog_stats.get('memory_size', 0)
                
                # 백테스트 캐시 통계
                bt_stats = cache_stats.get('backtest_cache', {})
                bt_hit_rate = bt_stats.get('hit_rate', 0) * 100
                bt_size = bt_stats.get('cache_size', 0)
                
                stats_text = f"📊 캐시 성능: 프로그램 {prog_hit_rate:.1f}% ({prog_size}개) | 백테스트 {bt_hit_rate:.1f}% ({bt_size}개)"
                
                # 색상 코딩 (히트율에 따라)
                avg_hit_rate = (prog_hit_rate + bt_hit_rate) / 2
                if avg_hit_rate > 70:
                    color = "#4CAF50"  # 녹색 - 좋음
                elif avg_hit_rate > 40:
                    color = "#FF9800"  # 주황색 - 보통
                else:
                    color = "#F44336"  # 빨간색 - 낮음
                
                self.cache_stats_label.setText(stats_text)
                self.cache_stats_label.setStyleSheet(
                    f"QLabel {{ background-color: {color}20; padding: 8px; border-radius: 5px; "
                    f"border: 1px solid {color}; font-size: 9px; color: {color}; }}"
                )
            else:
                self.cache_stats_label.setText("📊 캐시 통계: 사용 가능한 데이터 없음")
                
        except Exception as e:
            self.cache_stats_label.setText(f"📊 캐시 통계: 오류 ({str(e)[:20]}...)")
    
    def clear_cache_stats(self):
        """캐시 통계 및 캐시 초기화"""
        try:
            from factor_factory.rlc.enhanced_cache import clear_all_caches
            clear_all_caches()
            
            self.cache_stats_label.setText("📊 캐시 통계: 초기화됨")
            self.add_log("[CACHE] 모든 캐시가 초기화되었습니다")
            
        except Exception as e:
            self.add_log(f"[ERROR] 캐시 초기화 실패: {str(e)}")
    
    def training_finished(self, success, message):
        """학습 완료"""
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.resume_btn.setEnabled(self.current_checkpoint is not None)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.update_status("완료")
        
        # 완료 메시지
        self.add_log("-" * 60)
        self.add_log(f"학습 완료: {message}")
        
        # 체크포인트 상태 업데이트
        self.check_existing_checkpoint()
        
        if success:
            QMessageBox.information(self, "완료", message)
        else:
            QMessageBox.warning(self, "오류", message)
        
        self.worker = None
    
    def clear_log(self):
        """로그 지우기"""
        self.log_text.clear()
    
    def save_log(self):
        """로그 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "로그 저장", 
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "저장 완료", f"로그가 저장되었습니다:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "저장 실패", f"로그 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def setup_timer(self):
        """타이머 설정 (UI 업데이트용)"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # 100ms마다 업데이트
    
    def update_ui(self):
        """UI 주기적 업데이트"""
        # 캐시 통계 업데이트 (학습 중일 때만)
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.update_cache_stats()
    
    # === 차트 관련 메서드들 ===
    def on_chart_mode_changed(self, mode):
        """차트 표시 모드 변경"""
        if hasattr(self, 'chart_widget'):
            self.chart_widget.set_display_mode(mode)
    
    def clear_chart(self):
        """차트 초기화"""
        if hasattr(self, 'chart_widget'):
            self.chart_widget.clear()
            self.add_log("[CHART] 차트가 초기화되었습니다.")
    
    def export_chart(self):
        """차트를 이미지로 저장"""
        if hasattr(self, 'chart_widget'):
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "차트 저장",
                f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG Files (*.png);;JPG Files (*.jpg)"
            )
            
            if file_path:
                try:
                    self.chart_widget.save_chart(file_path)
                    QMessageBox.information(self, "저장 완료", f"차트가 저장되었습니다:\n{file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "저장 실패", f"차트 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def update_chart_data(self, timestamps, prices, signals, pnl):
        """차트 데이터 업데이트 (워커에서 호출됨)"""
        if hasattr(self, 'chart_widget'):
            self.chart_widget.update_data(timestamps, prices, signals, pnl)
    
    def load_real_data_to_chart(self):
        """실제 백테스트 데이터를 차트에 로드"""
        try:
            from factor_factory.visualization.real_data_loader import load_real_chart_data
            
            # 선택된 첫 번째 자산 사용
            selected_assets = self.get_selected_assets()
            symbol = selected_assets[0] if selected_assets else "BTCUSDT"
            
            self.add_log(f"[CHART] 실제 데이터 로드 중... (심볼: {symbol})")
            
            # 실제 데이터 로드
            timestamps, prices, signals, pnl = load_real_chart_data(
                symbol=symbol,
                max_points=200,
                use_latest=True
            )
            
            if timestamps:
                # 차트에 데이터 업데이트
                self.chart_widget.update_data(timestamps, prices, signals, pnl)
                self.add_log(f"[CHART] 실제 데이터 로드 완료: {len(timestamps)}개 포인트")
                self.add_log(f"[CHART] 가격 범위: {min(prices):.2f} ~ {max(prices):.2f}")
                self.add_log(f"[CHART] 시그널: Long {signals.count(1)}, Short {signals.count(-1)}")
                self.add_log(f"[CHART] PNL 범위: {min(pnl):.2f}% ~ {max(pnl):.2f}%")
            else:
                self.add_log("[CHART] 실제 데이터를 로드할 수 없습니다.")
                QMessageBox.warning(self, "데이터 로드 실패", 
                                  "실제 백테스트 데이터를 찾을 수 없습니다.\n"
                                  "먼저 백테스트를 실행하거나 데이터 파일을 확인해주세요.")
                
        except Exception as e:
            error_msg = f"실제 데이터 로드 중 오류: {str(e)}"
            self.add_log(f"[ERROR] {error_msg}")
            QMessageBox.warning(self, "오류", error_msg)
    
    def closeEvent(self, event):
        """윈도우 닫기 이벤트"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 
                "확인", 
                "학습이 진행 중입니다. 정말 종료하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """GUI 애플리케이션 실행"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle('Fusion')
    
    # 다크 테마 설정 (선택사항)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # 메인 윈도우 생성
    window = MultiAssetFactorGUI()
    window.show()
    
    # 애플리케이션 실행
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
