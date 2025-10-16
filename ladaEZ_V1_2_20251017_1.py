#!/usr/bin/env python3
"""
LADA REALTIME PLAYER V1.2
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time
import json
import gc
import queue 
from collections import OrderedDict, deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QDialog, QSpinBox, QFormLayout, QDialogButtonBox, QSlider, QSizePolicy, QMessageBox, QComboBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMutex, QMutexLocker, QTimer, QPoint
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QShortcut, QKeySequence, QDragEnterEvent, QDropEvent, QMouseEvent, QCursor
from OpenGL.GL import *

# カレントディレクトリを確実に取得（xxx.pyの場所）
LADA_BASE_PATH = Path(__file__).parent  # 変更: __file__を使ってスクリプトのディレクトリを絶対的に取得（cwd()より安定）
PYTHON_PATH = LADA_BASE_PATH / "python" / "Lib" / "site-packages"

# 変更: カレントディレクトリをsys.pathの先頭に追加（frame_restorer.pyがここにあるため優先）
sys.path.insert(0, str(LADA_BASE_PATH))

# site-packagesを次に追加（torchなどのライブラリ用）
if PYTHON_PATH.exists():
    sys.path.insert(1, str(PYTHON_PATH))

CONFIG_FILE = Path("lada_config.json")

LADA_AVAILABLE = False
try:
    import torch
    from frame_restorer import load_models
    from lada.lib import video_utils
    LADA_AVAILABLE = True
    print("✓ LADA利用可能")
except ImportError as e:
    print(f"✗ LADA: {e}")

# VLCのインポートを試みる
VLC_AVAILABLE = False
try:
    import vlc
    VLC_AVAILABLE = True
    print("✓ VLC利用可能")
except ImportError as e:
    print(f"✗ VLC: {e} - 音声機能は無効化されます")


class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.settings = current_settings or {}
        
        layout = QFormLayout(self)
        
        # 音声同期設定セクション
        layout.addRow(QLabel("<b>音声同期設定</b>"))
        
        self.audio_offset_spin = QDoubleSpinBox()
        self.audio_offset_spin.setRange(-2.0, 2.0)
        self.audio_offset_spin.setSingleStep(0.1)
        self.audio_offset_spin.setValue(self.settings.get('audio_offset', 0.3))
        self.audio_offset_spin.setSuffix(" 秒")
        self.audio_offset_spin.setToolTip(
            "音声の先行オフセット時間\n"
            "LADA処理による遅延を補正します\n"
            "• 0.3秒: 標準的なLADA処理遅延\n"
            "• 0.0秒: 同期なし\n"
            "• 負の値: 音声を遅らせる"
        )
        layout.addRow("音声先行オフセット:", self.audio_offset_spin)
        
        # モザイク検知モデル選択セクション
        layout.addRow(QLabel("<b>モザイク検知モデル設定</b>"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("lada_mosaic_detection_model_v2.pt", "lada_mosaic_detection_model_v2.pt")
        self.model_combo.addItem("lada_mosaic_detection_model_v3.1_fast.pt", "lada_mosaic_detection_model_v3.1_fast.pt")
        self.model_combo.addItem("lada_mosaic_detection_model_v3.1_accurate.pt", "lada_mosaic_detection_model_v3.1_accurate.pt")
        
        # 現在の設定を選択
        current_model = self.settings.get('detection_model', 'lada_mosaic_detection_model_v3.1_fast.pt')
        index = self.model_combo.findData(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
            
        self.model_combo.setToolTip("使用するモザイク検知モデルを選択\n• v2: \n• v3.1_fast: \n• v3.1_Accurate: ")
        layout.addRow("検知モデル:", self.model_combo)
        
        # RESTORER専用設定セクション
        layout.addRow(QLabel("<b>RESTORER設定</b>"))
        
        # 並列処理数設定
        self.parallel_clips_spin = QSpinBox()
        self.parallel_clips_spin.setRange(1, 128)
        self.parallel_clips_spin.setValue(self.settings.get('parallel_clips', 4))
        self.parallel_clips_spin.setToolTip(
            "同時に処理するクリップ数\n"
            "推奨設定:\n"
            "• 4並列: 標準的な並列処理\n" 
            "• 8並列: 高性能GPU向け\n"
            "• 16並列: 最高性能（メモリ注意）"
        )
        layout.addRow("並列処理数:", self.parallel_clips_spin)
        
        # バッチサイズ設定
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(self.settings.get('batch_size', 16))
        self.batch_size_spin.setSuffix(" frames")
        self.batch_size_spin.setToolTip("一度に処理するフレーム数\n大きいほど高速だがメモリ消費が増加")
        layout.addRow("バッチサイズ:", self.batch_size_spin)
        
        # キューサイズ設定
        self.queue_size_spin = QSpinBox()
        self.queue_size_spin.setRange(256, 16384)
        self.queue_size_spin.setValue(self.settings.get('queue_size_mb', 12288))
        self.queue_size_spin.setSuffix(" MB")
        self.queue_size_spin.setToolTip("処理キューのメモリサイズ\n大きいほど安定するがメモリ消費が増加")
        layout.addRow("キューサイズ:", self.queue_size_spin)
        
        # 最大クリップ長設定
        self.max_clip_length_spin = QSpinBox()
        self.max_clip_length_spin.setRange(1, 180)
        self.max_clip_length_spin.setValue(self.settings.get('max_clip_length', 8))
        self.max_clip_length_spin.setSuffix(" frames")
        self.max_clip_length_spin.setToolTip("1クリップあたりの最大フレーム数")
        layout.addRow("最大クリップ長:", self.max_clip_length_spin)
        
        # キャッシュ設定セクション
        layout.addRow(QLabel("<b>キャッシュ設定</b>"))
        
        self.cache_enabled_check = QCheckBox("キャッシュ管理を有効にする")
        self.cache_enabled_check.setChecked(self.settings.get('cache_enabled', True))
        self.cache_enabled_check.setToolTip(
            "キャッシュ管理を有効にすると、モザイク検出時にフレームをキャッシュします\n"
            "無効にするとキャッシュを使用せず、メモリ使用量を削減できます"
        )
        layout.addRow("キャッシュ管理:", self.cache_enabled_check)
        
        # キャッシュサイズ設定
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(1024, 32768)
        self.cache_size_spin.setValue(self.settings.get('cache_size_mb', 12288))
        self.cache_size_spin.setSuffix(" MB")
        self.cache_size_spin.setToolTip("フレームキャッシュの最大サイズ")
        layout.addRow("キャッシュサイズ:", self.cache_size_spin)
        
        # チャンクフレーム数設定
        self.chunk_frames_spin = QSpinBox()
        self.chunk_frames_spin.setRange(50, 500)
        self.chunk_frames_spin.setValue(self.settings.get('chunk_frames', 150))
        self.chunk_frames_spin.setSuffix(" frames")
        self.chunk_frames_spin.setToolTip("1チャンクあたりのフレーム数\n小さいほど細かい管理だがオーバーヘッド増加")
        layout.addRow("チャンクサイズ:", self.chunk_frames_spin)
        
        # ボタンボックス
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    
    def get_settings(self):
        """設定値を取得"""
        return {
            'cache_enabled': self.cache_enabled_check.isChecked(),
            'audio_offset': self.audio_offset_spin.value(),
            'detection_model': self.model_combo.currentData(),
            'parallel_clips': self.parallel_clips_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'queue_size_mb': self.queue_size_spin.value(),
            'max_clip_length': self.max_clip_length_spin.value(),
            'cache_size_mb': self.cache_size_spin.value(),
            'chunk_frames': self.chunk_frames_spin.value(),
            'audio_volume': self.settings.get('audio_volume', 100),
            'audio_muted': self.settings.get('audio_muted', False)
        }

        
class SmartChunkBasedCache:
    """30FPS最適化スマートキャッシュ - 完全無効化対応版"""
    
    def __init__(self, max_size_mb=12288, chunk_frames=150, enabled=True):
        # enabled属性を最初に設定
        self.enabled = enabled
        
        # 共通の基本属性
        self.chunk_frames = chunk_frames
        self.max_size_mb = max_size_mb
        
        # 共通の基本属性を常に初期化（無効時も必要）
        self.mutex = QMutex()
        self.current_size_mb = 0
        self.chunks = {}
        self.access_order = deque()
        
        # 再生位置関連の属性を常に初期化（無効時も必要）
        self.previous_playhead = 0
        self.current_playhead = 0
        
        if not enabled:
            print("[CACHE] キャッシュ管理: 無効")
            return
            
        # キャッシュ有効時のみの追加初期化
        # 処理コスト追跡
        self.processing_costs = {}  # chunk_id -> cost_data
        self.cache_policies = {}    # chunk_id -> policy_dict
        
        # パフォーマンス統計
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_frames': 0,
            'total_processing_time': 0.0
        }
        
        # モザイク検出システム
        self.mosaic_detected = False
        self.consecutive_slow_frames = 0
        self.consecutive_fast_frames = 0
        self.slow_frame_threshold = 3
        self.fast_frame_threshold = 5
        self.mosaic_threshold_ms = 40.0
        self.fast_threshold_ms = 20.0
        self.last_mosaic_change_time = 0
        
        # インテリジェント削除用データ
        self.chunk_access_count = {}
        
        # 非同期クリーンアップ
        self.cleanup_timer = QTimer()
        self.cleanup_timer.setSingleShot(True)
        self.pending_cleanup = False
        
        print(f"[SMART-CACHE] 最適化版 初期化: {max_size_mb}MB, 閾値={self.mosaic_threshold_ms}ms")

    def get_chunk_id(self, frame_num):
        """フレーム番号からチャンクIDを計算"""
        return frame_num // self.chunk_frames

    def should_cache_frame(self, frame_num, frame_data=None):
        """基本FALSE、モザイク検出時のみTRUE"""
        return self.mosaic_detected

    def record_frame_processing_time(self, frame_num, processing_time):
        # キャッシュ無効時は何もしない
        if not self.enabled:
            return
            
        if not self.mutex.tryLock(10):
            return
            
        try:
            chunk_id = self.get_chunk_id(frame_num)
            
            if chunk_id not in self.processing_costs:
                self.processing_costs[chunk_id] = {
                    'frame_times': [],
                    'total_time': 0.0,
                    'sample_count': 0,
                    'last_sample_time': time.time()
                }
            
            cost_data = self.processing_costs[chunk_id]
            cost_data['frame_times'].append(processing_time)
            cost_data['total_time'] += processing_time
            cost_data['sample_count'] += 1
            cost_data['last_sample_time'] = time.time()
            
            # スマートモザイク検出
            current_ms = processing_time * 1000
            mosaic_state_changed = self._update_mosaic_state(current_ms, frame_num)
            
            if cost_data['sample_count'] >= 2:
                self._update_chunk_policy(chunk_id)
            
            self.performance_stats['total_frames'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            if mosaic_state_changed:
                self._debug_mosaic_state()
                
        finally:
            self.mutex.unlock()

    def _update_mosaic_state(self, current_ms, frame_num):
        """モザイク状態を更新"""
        # 必要な属性が存在することを確認
        if not hasattr(self, 'mosaic_detected'):
            return False
            
        previous_state = self.mosaic_detected
        state_changed = False
        
        if current_ms >= self.mosaic_threshold_ms:
            self.consecutive_slow_frames += 1
            self.consecutive_fast_frames = 0
            
            if (self.consecutive_slow_frames >= self.slow_frame_threshold and 
                not self.mosaic_detected):
                self.mosaic_detected = True
                state_changed = True
                self.last_mosaic_change_time = time.time()
                
        elif current_ms <= self.fast_threshold_ms:
            self.consecutive_fast_frames += 1
            self.consecutive_slow_frames = 0
            
            if (self.consecutive_fast_frames >= self.fast_frame_threshold and 
                self.mosaic_detected):
                self.mosaic_detected = False
                state_changed = True
                self.last_mosaic_change_time = time.time()
        
        return state_changed

    def _debug_mosaic_state(self):
        """モザイク状態のデバッグ出力"""
        state = "🔍 モザイク検出" if self.mosaic_detected else "✅ モザイクなし"
        slow_str = f"遅:{self.consecutive_slow_frames}" if self.consecutive_slow_frames > 0 else ""
        fast_str = f"速:{self.consecutive_fast_frames}" if self.consecutive_fast_frames > 0 else ""
        counter_str = f" ({slow_str}{fast_str})".strip()
        print(f"[CACHE] {state}{counter_str}")

    def _update_chunk_policy(self, chunk_id):
        """互換性のためのポリシー更新"""
        # 必要な属性が存在することを確認
        if not hasattr(self, 'processing_costs') or not hasattr(self, 'cache_policies'):
            return
            
        if chunk_id not in self.processing_costs:
            return
            
        cost_data = self.processing_costs[chunk_id]
        avg_ms_per_frame = (cost_data['total_time'] / cost_data['sample_count']) * 1000
        
        if self.mosaic_detected:
            if avg_ms_per_frame <= 100.0:
                policy, priority = 'standard_cache', 2
            else:
                policy, priority = 'priority_cache', 3
        else:
            policy, priority = 'no_cache', 0
        
        self.cache_policies[chunk_id] = {
            'policy': policy,
            'priority': priority,
            'avg_ms_per_frame': avg_ms_per_frame,
            'sample_size': cost_data['sample_count'],
            'last_updated': time.time()
        }

    def get(self, frame_num):
        # キャッシュ無効時は常にNoneを返す
        if not self.enabled:
            return None
            
        if not self.mutex.tryLock(5):
            return None
            
        try:
            chunk_id = self.get_chunk_id(frame_num)
            
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                if frame_num in chunk['frames']:
                    chunk['last_access'] = time.time()
                    self._update_access_order(chunk_id)
                    self.chunk_access_count[chunk_id] = self.chunk_access_count.get(chunk_id, 0) + 1
                    self.performance_stats['cache_hits'] += 1
                    return chunk['frames'][frame_num]
            
            self.performance_stats['cache_misses'] += 1
            return None
        finally:
            self.mutex.unlock()

    def put(self, frame_num, frame):
        # キャッシュ無効時は何もしない
        if not self.enabled:
            return
            
        """モザイク検出時のみフレームをキャッシュ"""
        if not self.mutex.tryLock(5):
            return
            
        try:
            if frame is None:
                self._remove_frame(frame_num)
                return
                
            # should_cache_frameのチェックを削除 - 常にキャッシュする
            # （should_cache_frameはputメソッド内では使用しない）
            chunk_id = self.get_chunk_id(frame_num)
            
            if chunk_id not in self.chunks:
                self.chunks[chunk_id] = {
                    'frames': {},
                    'size_mb': 0,
                    'last_access': time.time()
                }
            
            chunk = self.chunks[chunk_id]
            frame_size_mb = frame.nbytes / (1024 * 1024)
            
            if frame_num in chunk['frames']:
                old_frame = chunk['frames'][frame_num]
                old_size_mb = old_frame.nbytes / (1024 * 1024)
                chunk['size_mb'] -= old_size_mb
                self.current_size_mb -= old_size_mb
            
            chunk['frames'][frame_num] = frame
            chunk['size_mb'] += frame_size_mb
            chunk['last_access'] = time.time()
            self.current_size_mb += frame_size_mb
            
            self._update_access_order(chunk_id)
            
            if self.current_size_mb > self.max_size_mb:
                self._schedule_async_cleanup()
                
        finally:
            self.mutex.unlock()

    def _update_access_order(self, chunk_id):
        """LRU順序を更新"""
        if chunk_id in self.access_order:
            self.access_order.remove(chunk_id)
        self.access_order.append(chunk_id)

    def _schedule_async_cleanup(self):
        """非同期クリーンアップをスケジュール"""
        if not self.pending_cleanup:
            self.pending_cleanup = True
            QTimer.singleShot(50, self._async_cleanup)

    def _async_cleanup(self):
        """インテリジェントな非同期クリーンアップ"""
        if not self.pending_cleanup:
            return
            
        if not self.mutex.tryLock(50):
            QTimer.singleShot(25, self._async_cleanup)
            return
            
        try:
            if self.current_size_mb <= self.max_size_mb * 0.8:
                self.pending_cleanup = False
                return
            
            protected_chunks = self._get_protected_chunks()
            candidate_chunks = self._get_cleanup_candidates(protected_chunks)
            
            removed_count = 0
            for chunk_id, priority_score in candidate_chunks:
                if self._remove_chunk(chunk_id):
                    removed_count += 1
                    print(f"[CACHE] クリーンアップ: チャンク{chunk_id}削除 (優先度: {priority_score:.3f})")
                    
                    if self.current_size_mb <= self.max_size_mb * 0.7:
                        break
                    if removed_count >= 3:
                        break
            
            if self.current_size_mb > self.max_size_mb * 0.8:
                print(f"[CACHE] クリーンアップ継続: {self.current_size_mb:.1f}MB > {self.max_size_mb * 0.8:.1f}MB")
                QTimer.singleShot(25, self._async_cleanup)
            else:
                self.pending_cleanup = False
                print(f"[CACHE] クリーンアップ完了: {removed_count}チャンク削除, 現在 {self.current_size_mb:.1f}MB")
                
        finally:
            self.mutex.unlock()

    def _get_cleanup_candidates(self, protected_chunks):
        """削除候補のチャンクを優先度順にソート"""
        candidates = []
        
        for chunk_id in list(self.access_order):
            if chunk_id in protected_chunks:
                continue
                
            priority_score = self._calculate_cleanup_priority(chunk_id)
            candidates.append((chunk_id, priority_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _calculate_cleanup_priority(self, chunk_id):
        """チャンクの削除優先度を計算"""
        base_score = 0.0
        
        mosaic_ratio = self._get_chunk_mosaic_ratio(chunk_id)
        base_score += (1.0 - mosaic_ratio) * 0.5
        
        access_count = self.chunk_access_count.get(chunk_id, 0)
        access_factor = 1.0 / (access_count + 1)
        base_score += access_factor * 0.3
        
        if chunk_id in self.chunks:
            time_since_access = time.time() - self.chunks[chunk_id]['last_access']
            time_factor = min(time_since_access / 300.0, 1.0)
            base_score += time_factor * 0.2
        
        return base_score

    def _get_chunk_mosaic_ratio(self, chunk_id):
        """チャンク内のモザイクフレームの割合を計算"""
        if chunk_id not in self.chunks:
            return 0.0
        
        chunk = self.chunks[chunk_id]
        mosaic_frames = 0
        total_frames = len(chunk['frames'])
        
        if total_frames == 0:
            return 0.0
        
        for frame_num in chunk['frames']:
            frame_chunk_id = self.get_chunk_id(frame_num)
            if frame_chunk_id in self.processing_costs:
                cost_data = self.processing_costs[frame_chunk_id]
                if cost_data['sample_count'] > 0:
                    avg_time = (cost_data['total_time'] / cost_data['sample_count']) * 1000
                    if avg_time >= self.mosaic_threshold_ms:
                        mosaic_frames += 1
        
        return mosaic_frames / total_frames

    def _get_protected_chunks(self):
        """動的保护範囲を計算"""
        current_chunk = self.get_chunk_id(self.current_playhead)
        protected = set()
        
        for offset in range(-2, 3):
            protected.add(current_chunk + offset)
        
        seek_direction = self.current_playhead - self.previous_playhead
        if abs(seek_direction) > self.chunk_frames:
            if seek_direction > 0:
                for offset in range(1, 4):
                    protected.add(current_chunk + offset)
            elif seek_direction < 0:
                for offset in range(-4, 0):
                    protected.add(current_chunk + offset)
        
        return protected

    def _remove_chunk(self, chunk_id):
        """チャンク全体を削除"""
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            self.current_size_mb -= chunk['size_mb']
            
            if chunk_id in self.access_order:
                self.access_order.remove(chunk_id)
            if chunk_id in self.chunk_access_count:
                del self.chunk_access_count[chunk_id]
            
            del self.chunks[chunk_id]
            return True
        return False

    def _remove_frame(self, frame_num):
        """単一フレームを削除"""
        chunk_id = self.get_chunk_id(frame_num)
        if chunk_id in self.chunks:
            chunk = self.chunks[chunk_id]
            if frame_num in chunk['frames']:
                frame = chunk['frames'][frame_num]
                frame_size_mb = frame.nbytes / (1024 * 1024)
                
                del chunk['frames'][frame_num]
                chunk['size_mb'] -= frame_size_mb
                self.current_size_mb -= frame_size_mb
                
                if not chunk['frames']:
                    self._remove_chunk(chunk_id)

    def update_playhead(self, frame_num):
        """再生位置を更新"""
        self.previous_playhead = self.current_playhead
        self.current_playhead = frame_num

    def clear(self):
        """キャッシュ全クリア"""
        if not self.enabled:
            return
            
        if not self.mutex.tryLock(100):
            return
            
        try:
            self.chunks.clear()
            self.access_order.clear()
            self.current_size_mb = 0
            self.pending_cleanup = False
            
            # 以下の属性が存在する場合のみクリア
            if hasattr(self, 'processing_costs'):
                self.processing_costs.clear()
            if hasattr(self, 'cache_policies'):
                self.cache_policies.clear()
            if hasattr(self, 'chunk_access_count'):
                self.chunk_access_count.clear()
            
            if hasattr(self, 'mosaic_detected'):
                self.mosaic_detected = False
                self.consecutive_slow_frames = 0
                self.consecutive_fast_frames = 0
                self.last_mosaic_change_time = 0
                self.previous_playhead = 0
                self.current_playhead = 0
            
            if hasattr(self, 'performance_stats'):
                self.performance_stats = {
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'total_frames': 0,
                    'total_processing_time': 0.0
                }
            
            print("[CACHE] キャッシュ完全クリア")
        finally:
            self.mutex.unlock()

    def get_stats(self):
        # キャッシュ無効時はゼロ統計を返す
        if not self.enabled:
            return {
                'chunk_count': 0,
                'total_frames': 0,
                'size_mb': 0,
                'max_mb': self.max_size_mb,
                'chunk_frames': self.chunk_frames,
                'hit_ratio': 0.0,
                'avg_processing_time': 0.0,
                'policy_distribution': {},
                'mosaic_detected': False,
                'consecutive_slow': 0,
                'consecutive_fast': 0,
                'mosaic_chunks': 0,
                'avg_mosaic_ratio': 0.0,
                'enabled': False
            }
            
        """詳細なキャッシュ統計を取得"""
        if not self.mutex.tryLock(10):
            return self._get_default_stats()
            
        try:
            chunk_count = len(self.chunks)
            total_frames = sum(len(chunk['frames']) for chunk in self.chunks.values())
            
            mosaic_chunks = 0
            total_mosaic_ratio = 0.0
            for chunk_id in self.chunks:
                mosaic_ratio = self._get_chunk_mosaic_ratio(chunk_id)
                total_mosaic_ratio += mosaic_ratio
                if mosaic_ratio > 0.5:
                    mosaic_chunks += 1
            
            avg_mosaic_ratio = total_mosaic_ratio / chunk_count if chunk_count > 0 else 0.0
            
            stats = {
                'chunk_count': chunk_count,
                'total_frames': total_frames,
                'size_mb': self.current_size_mb,
                'max_mb': self.max_size_mb,
                'chunk_frames': self.chunk_frames,
                'mosaic_detected': self.mosaic_detected,
                'consecutive_slow': self.consecutive_slow_frames,
                'consecutive_fast': self.consecutive_fast_frames,
                'mosaic_chunks': mosaic_chunks,
                'avg_mosaic_ratio': avg_mosaic_ratio,
                'enabled': True
            }
            
            total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            if total_requests > 0:
                stats['hit_ratio'] = self.performance_stats['cache_hits'] / total_requests
            else:
                stats['hit_ratio'] = 0.0
                
            if self.performance_stats['total_frames'] > 0:
                stats['avg_processing_time'] = (self.performance_stats['total_processing_time'] / self.performance_stats['total_frames']) * 1000
            else:
                stats['avg_processing_time'] = 0.0
                
            stats['policy_distribution'] = {}
            for policy in self.cache_policies.values():
                policy_name = policy['policy']
                stats['policy_distribution'][policy_name] = stats['policy_distribution'].get(policy_name, 0) + 1
            
            return stats
        finally:
            self.mutex.unlock()
            
    def _get_default_stats_disabled(self):
        """キャッシュ無効時のデフォルト統計"""
        return {
            'chunk_count': 0,
            'total_frames': 0,
            'size_mb': 0,
            'max_mb': 0,
            'chunk_frames': 0,
            'hit_ratio': 0.0,
            'avg_processing_time': 0.0,
            'policy_distribution': {},
            'mosaic_detected': False,
            'consecutive_slow': 0,
            'consecutive_fast': 0,
            'mosaic_chunks': 0,
            'avg_mosaic_ratio': 0.0,
            'enabled': False
        }

    def _get_default_stats(self):
        """デフォルト統計"""
        return {
            'chunk_count': 0,
            'total_frames': 0,
            'size_mb': 0,
            'max_mb': self.max_size_mb,
            'chunk_frames': self.chunk_frames,
            'hit_ratio': 0.0,
            'avg_processing_time': 0.0,
            'policy_distribution': {},
            'mosaic_detected': False,
            'consecutive_slow': 0,
            'consecutive_fast': 0,
            'mosaic_chunks': 0,
            'avg_mosaic_ratio': 0.0
        }


class VideoGLWidget(QOpenGLWidget):
    playback_toggled = pyqtSignal()
    video_dropped = pyqtSignal(str)
    seek_requested = pyqtSignal(int)
    toggle_mute_signal = pyqtSignal()
    toggle_ai_processing_signal = pyqtSignal()
    set_range_start_signal = pyqtSignal()
    set_range_end_signal = pyqtSignal()
    reset_range_signal = pyqtSignal()
    seek_to_start_signal = pyqtSignal()
    seek_to_end_signal = pyqtSignal()
    seek_to_percentage_signal = pyqtSignal(int)
    toggle_range_mode_signal = pyqtSignal()
    frame_step_forward_signal = pyqtSignal()
    frame_step_backward_signal = pyqtSignal()
    change_playback_speed_signal = pyqtSignal()
    reset_playback_speed_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = None
        self.is_fullscreen = False
        self.normal_parent_geometry = None
        self.parent_widget = None
        self.setMinimumSize(800, 450)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # D&Dを有効化
        self.setAcceptDrops(True)
        
        # フルスクリーン用UI
        self.fs_progress_bar = QProgressBar(self)
        self.fs_progress_bar.setTextVisible(False)
        self.fs_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(40, 40, 40, 200);
                border: none;
                height: 38px;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
            }
        """)
        self.fs_progress_bar.hide()
        
        self.fs_time_label = QLabel("00:00:00 / 00:00:00", self)
        self.fs_time_label.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 40, 40, 200);
                color: white;
                padding: 4px 12px;
                font-size: 14px;
                border-radius: 4px;
            }
        """)
        self.fs_time_label.hide()
        
        # UI自動非表示タイマー
        self.ui_hide_timer = QTimer()
        self.ui_hide_timer.timeout.connect(self.hide_fs_ui)
        self.ui_hide_timer.setSingleShot(True)
        
        # 進捗情報
        self.total_frames = 0
        self.current_frame_num = 0
        self.video_fps = 30.0
        
        self.setMouseTracking(True)
    
    def set_video_info(self, total_frames, fps):
        """動画情報を設定"""
        self.total_frames = total_frames
        self.video_fps = fps
        self.fs_progress_bar.setMaximum(total_frames)
    
    def update_progress(self, frame_num):
        """進捗更新"""
        self.current_frame_num = frame_num
        self.fs_progress_bar.setValue(frame_num)
        
        current_sec = frame_num / self.video_fps if self.video_fps > 0 else 0
        total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        
        current_time = self.format_time(current_sec)
        total_time = self.format_time(total_sec)
        self.fs_time_label.setText(f"{current_time} / {total_time}")
    
    def format_time(self, seconds):
        """秒を HH:MM:SS 形式に変換"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def show_fs_ui(self):
        """フルスクリーンUI表示"""
        if self.is_fullscreen:
            self.update_fs_ui_position()
            self.fs_progress_bar.show()
            self.fs_time_label.show()
            QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
            self.ui_hide_timer.start(3000)
    
    def update_fs_ui_position(self):
        """フルスクリーンUI位置更新"""
        if not self.is_fullscreen:
            return
            
        bar_height = 38
        bar_margin = 20
        self.fs_progress_bar.setGeometry(
            bar_margin, 
            self.height() - bar_height - bar_margin, 
            self.width() - bar_margin * 2, 
            bar_height
        )
        
        self.fs_time_label.adjustSize()
        self.fs_time_label.move(
            (self.width() - self.fs_time_label.width()) // 2,
            self.height() - bar_height - bar_margin - self.fs_time_label.height() - 10
        )
    
    def hide_fs_ui(self):
        """フルスクリーンUI非表示"""
        if self.is_fullscreen:
            self.fs_progress_bar.hide()
            self.fs_time_label.hide()
            QApplication.restoreOverrideCursor()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.is_fullscreen:
            self.update_fs_ui_position()
    
    def mouseMoveEvent(self, event):
        if self.is_fullscreen:
            self.show_fs_ui()
        super().mouseMoveEvent(event)
    
    def fs_progress_click(self, event: QMouseEvent):
        """フルスクリーン進捗バークリック"""
        if self.total_frames > 0:
            pos = event.pos().x()
            bar_margin = 20
            bar_width = self.width() - bar_margin * 2
            relative_pos = pos - bar_margin
            
            if 0 <= relative_pos <= bar_width:
                target_frame = int((relative_pos / bar_width) * self.total_frames)
                self.seek_requested.emit(target_frame)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.video_dropped.emit(file_path)
                event.acceptProposedAction()
    
    def is_video_file(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts']
        file_ext = Path(file_path).suffix.lower()
        return file_ext in video_extensions
    
    def get_main_window(self):
        """メインウィンドウを安全に取得"""
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'seek_relative'):
                return parent
            parent = parent.parent()
        
        for widget in QApplication.topLevelWidgets():
            if hasattr(widget, 'seek_relative'):
                return widget
        
        return None
    
    def keyPressEvent(self, event):
        if self.is_fullscreen:
            key = event.key()
            if key == Qt.Key.Key_F or key == Qt.Key.Key_Escape:
                self.toggle_fullscreen()
            elif key == Qt.Key.Key_Space or key == Qt.Key.Key_K:
                self.playback_toggled.emit()
            elif key == Qt.Key.Key_Right or key == Qt.Key.Key_L:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(300)
            elif key == Qt.Key.Key_Left or key == Qt.Key.Key_J:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(-300)
            elif key == Qt.Key.Key_Semicolon:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(30)
            elif key == Qt.Key.Key_H:
                main_window = self.get_main_window()
                if main_window:
                    main_window.seek_relative(-30)
            elif key == Qt.Key.Key_M:
                self.toggle_mute_signal.emit()
            elif key == Qt.Key.Key_X:
                self.toggle_ai_processing_signal.emit()
            elif key == Qt.Key.Key_S:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.set_range_start_signal.emit()
                else:
                    self.seek_to_start_signal.emit()
            elif key == Qt.Key.Key_E:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.set_range_end_signal.emit()
                else:
                    self.seek_to_end_signal.emit()
            elif key == Qt.Key.Key_R and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.reset_range_signal.emit()
            elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
                percent = key - Qt.Key.Key_0
                self.seek_to_percentage_signal.emit(percent)
            elif key == Qt.Key.Key_P and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.toggle_range_mode_signal.emit()
            # 新しいショートカット
            elif key == Qt.Key.Key_Comma:  # < キー
                self.frame_step_backward_signal.emit()
            elif key == Qt.Key.Key_Period:  # > キー
                self.frame_step_forward_signal.emit()
            elif key == Qt.Key.Key_Z:
                self.change_playback_speed_signal.emit()
            elif key == Qt.Key.Key_A:
                self.reset_playback_speed_signal.emit()
        else:
            key = event.key()
            if key == Qt.Key.Key_M:
                self.toggle_mute_signal.emit()
            elif key == Qt.Key.Key_X:
                self.toggle_ai_processing_signal.emit()
            elif key == Qt.Key.Key_P and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.toggle_range_mode_signal.emit()
            # 新しいショートカット
            elif key == Qt.Key.Key_Comma:  # < キー
                self.frame_step_backward_signal.emit()
            elif key == Qt.Key.Key_Period:  # > キー
                self.frame_step_forward_signal.emit()
            elif key == Qt.Key.Key_Z:
                self.change_playback_speed_signal.emit()
            elif key == Qt.Key.Key_A:
                self.reset_playback_speed_signal.emit()
            else:
                super().keyPressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        parent = self.window()
        if hasattr(parent, 'is_paused'):
            current_pause_state = parent.is_paused
        
        self.toggle_fullscreen()
        
        if hasattr(parent, 'is_paused') and hasattr(parent, 'process_thread'):
            QTimer.singleShot(100, lambda: self.restore_playback_state(parent, current_pause_state))
    
    def restore_playback_state(self, parent, original_pause_state):
        if hasattr(parent, 'process_thread') and parent.process_thread and parent.process_thread.isRunning():
            if original_pause_state:
                parent.process_thread.pause()
                parent.is_paused = True
                parent.play_pause_btn.setText("▶ 再開")
                parent.mode_label.setText("📊 モード: ⏸ 一時停止中")
                self.set_progress_bar_color('red')
            else:
                parent.process_thread.resume()
                parent.is_paused = False
                parent.play_pause_btn.setText("⏸ 一時停止")
                parent.mode_label.setText("📊 モード: 🔄 AI処理中")
                self.set_progress_bar_color('#00ff00')
    
    def mousePressEvent(self, event):
        if self.is_fullscreen and self.fs_progress_bar.isVisible():
            bar_geom = self.fs_progress_bar.geometry()
            if bar_geom.contains(event.pos()):
                self.fs_progress_click(event)
                return
        
        if event.button() == Qt.MouseButton.LeftButton:
            self.playback_toggled.emit()
        super().mousePressEvent(event)
    
    def toggle_fullscreen(self):
        if not self.is_fullscreen:
            self.parent_widget = self.parentWidget()
            parent_window = self.window()
            self.normal_parent_geometry = parent_window.geometry()
            
            self.setParent(None)
            self.setWindowFlags(
                Qt.WindowType.Window | 
                Qt.WindowType.FramelessWindowHint | 
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.showFullScreen()
            self.setFocus(Qt.FocusReason.OtherFocusReason)
            self.activateWindow()
            self.raise_()
            self.is_fullscreen = True
            
            QApplication.processEvents()
            self.update_fs_ui_position()
            self.show_fs_ui()
        else:
            self.hide_fs_ui()
            self.ui_hide_timer.stop()
            
            if self.parent_widget:
                self.setParent(self.parent_widget)
                self.setWindowFlags(Qt.WindowType.Widget)
                
                parent_window = self.parent_widget.window()
                if hasattr(parent_window, 'video_layout'):
                    parent_window.video_layout.insertWidget(0, self)
                
                self.showNormal()
                
                if self.normal_parent_geometry and parent_window:
                    parent_window.setGeometry(self.normal_parent_geometry)
            
            self.is_fullscreen = False
            self.parent_widget = None
    
    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        if self.current_frame is not None and self.texture_id is not None:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            w, h = self.width(), self.height()
            if h == 0:
                h = 1
            window_aspect = w / h
            
            if self.frame_width > 0 and self.frame_height > 0:
                video_aspect = self.frame_width / self.frame_height
            else:
                video_aspect = 16.0 / 9.0
            
            if window_aspect > video_aspect:
                x_scale = video_aspect / window_aspect
                x1, x2 = -x_scale, x_scale
                y1, y2 = -1.0, 1.0
            else:
                y_scale = window_aspect / video_aspect
                x1, x2 = -1.0, 1.0
                y1, y2 = -y_scale, y_scale
            
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0); glVertex2f(x1, y1)
            glTexCoord2f(1.0, 1.0); glVertex2f(x2, y1)
            glTexCoord2f(1.0, 0.0); glVertex2f(x2, y2)
            glTexCoord2f(0.0, 0.0); glVertex2f(x1, y2)
            glEnd()
    
    def update_frame(self, frame):
        if frame is None:
            return
            
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.frame_width != w or self.frame_height != h:
            self.frame_width = w
            self.frame_height = h
        
        self.makeCurrent()
        
        if self.texture_id is None:
            self.texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)
        
        rgb_contiguous = np.ascontiguousarray(rgb)
        
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            w, h, 
            0, 
            GL_RGB, 
            GL_UNSIGNED_BYTE, 
            rgb_contiguous.tobytes()
        )
        
        self.current_frame = rgb
        self.update()
    
    def clear_frame(self):
        self.current_frame = None
        self.frame_width = 0
        self.frame_height = 0
        self.update()
    
    def set_progress_bar_color(self, color):
        self.fs_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(40, 40, 40, 200);
                border: none;
                height: 38px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

class MarkedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.range_start = None
        self.range_end = None
        
    def set_range_marks(self, start, end):
        self.range_start = start
        self.range_end = end
        self.update()
        
    def paintEvent(self, event):
        # まず標準のプログレスバーを描画
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 開始位置マーク（赤色▼）
        if self.range_start is not None and self.maximum() > 0:
            start_pos = int((self.range_start / self.maximum()) * self.width())
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            
            # ▼マークを描画（上部中央）
            points = [
                QPoint(start_pos - 5, 0),    # 左上
                QPoint(start_pos + 5, 0),    # 右上  
                QPoint(start_pos, 8)         # 下中央
            ]
            painter.drawPolygon(points)
        
        # 終了位置マーク（青色▼）
        if self.range_end is not None and self.maximum() > 0:
            end_pos = int((self.range_end / self.maximum()) * self.width())
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            painter.setBrush(QBrush(QColor(0, 0, 255)))
            
            # ▼マークを描画（上部中央）
            points = [
                QPoint(end_pos - 5, 0),      # 左上
                QPoint(end_pos + 5, 0),      # 右上
                QPoint(end_pos, 8)           # 下中央
            ]
            painter.drawPolygon(points)

class OptimizedFrameRestorer:
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length,
                 mosaic_restoration_model_name, mosaic_detection_model, 
                 mosaic_restoration_model, preferred_pad_mode,
                 batch_size=16, queue_size_mb=12288, mosaic_detection=False,
                 parallel_clips=2):
        
        try:
            from frame_restorer import OptimizedFrameRestorer as OFR
            
            self._parent = OFR(
                device=device, 
                video_file=video_file,
                preserve_relative_scale=preserve_relative_scale,
                max_clip_length=max_clip_length,
                mosaic_restoration_model_name=mosaic_restoration_model_name,
                mosaic_detection_model=mosaic_detection_model,
                mosaic_restoration_model=mosaic_restoration_model,
                preferred_pad_mode=preferred_pad_mode,
                mosaic_detection=mosaic_detection,
                batch_size=batch_size,
                parallel_clips=parallel_clips
            )
            
            print(f"[OPTIMIZE] 最適化FrameRestorerの作成成功 - 並列数: {parallel_clips}")
            
        except Exception as e:
            print(f"[OPTIMIZE] 最適化FrameRestorerの作成に失敗: {e}")
            print("[OPTIMIZE] 通常版のFrameRestorerを使用します")
            
            from frame_restorer import FrameRestorer
            
            self._parent = FrameRestorer(
                device=device, 
                video_file=video_file,
                preserve_relative_scale=preserve_relative_scale,
                max_clip_length=max_clip_length,
                mosaic_restoration_model_name=mosaic_restoration_model_name,
                mosaic_detection_model=mosaic_detection_model,
                mosaic_restoration_model=mosaic_restoration_model,
                preferred_pad_mode=preferred_pad_mode,
                mosaic_detection=mosaic_detection
            )
        
        w = self._parent.video_meta_data.video_width
        h = self._parent.video_meta_data.video_height
        
        frame_size_bytes = w * h * 3
        clip_size_bytes = max_clip_length * 256 * 256 * 4
        
        max_frames = max(100, (queue_size_mb * 1024 * 1024) // frame_size_bytes)
        max_clips = max(10, (queue_size_mb * 1024 * 1024) // clip_size_bytes)
        
        self._parent.frame_restoration_queue = queue.Queue(maxsize=max_frames)
        self._parent.mosaic_clip_queue = queue.Queue(maxsize=max_clips)
        self._parent.restored_clip_queue = queue.Queue(maxsize=max_clips)
        self._parent.mosaic_detector.mosaic_clip_queue = self._parent.mosaic_clip_queue
        
        self._parent.batch_size = min(batch_size, max_clip_length)
        
        print(f"[OPTIMIZE] Queue: {max_frames}f, {max_clips}c ({queue_size_mb}MB)")
        print(f"[OPTIMIZE] Batch size: {self._parent.batch_size}")

    def start(self, start_ns=0):
        return self._parent.start(start_ns)
    
    def stop(self):
        return self._parent.stop()
    
    def __iter__(self):
        return self._parent.__iter__()
    
    def __next__(self):
        return self._parent.__next__()


class AudioThread(QThread):
    """音声再生スレッド - 修正安定版"""
    
    def __init__(self, vlc_instance, initial_volume=100, is_muted=False, audio_offset=0.3):
        super().__init__()
        self.vlc_instance = vlc_instance
        self.player = self.vlc_instance.media_player_new()
        self._stop_flag = False
        self._is_paused = True
        self.volume = initial_volume
        self.user_muted = is_muted
        self.internal_muted = False
        self._seek_in_progress = False
        self._seek_mutex = QMutex()
        self._operation_mutex = QMutex()
        self.current_media_path = None
        
        # 音声同期用の変数
        self.last_sync_time = 0
        self.sync_interval = 2.0
        
        # 音声先行オフセット
        self.audio_offset = audio_offset
        
        # AIモード用ミュート管理（シンプル化）
        self.ai_seek_muted = False
        self.ai_seek_start_time = 0
        self.ai_mute_duration = 2.0  # 2秒間ミュート
        
        self.player.audio_set_volume(self.volume)
        self._update_vlc_mute_state()
        
        print(f"[AUDIO] AudioThread初期化: Volume={self.volume}, Mute={self.user_muted}, Offset={self.audio_offset}秒")

    def run(self):
        """メインループ - AIモードミュート管理を追加"""
        while not self._stop_flag:
            current_time = time.time()
            
            # AIモードミュートの自動解除チェック
            if (self.ai_seek_muted and 
                current_time - self.ai_seek_start_time >= self.ai_mute_duration):
                self._unmute_after_seek()
            
            # 音声同期チェック
            if current_time - self.last_sync_time >= self.sync_interval:
                self._check_audio_sync()
                self.last_sync_time = current_time
            
            time.sleep(0.1)

    def _unmute_after_seek(self):
        """AIモードシーク後のミュート解除"""
        if not VLC_AVAILABLE or self._stop_flag:
            return
            
        if not self._operation_mutex.tryLock(50):
            return
            
        try:
            self.ai_seek_muted = False
            self._update_vlc_mute_state()
            print("[AUDIO] AIモードシーク後ミュート解除")
        except Exception as e:
            print(f"[AUDIO] ミュート解除エラー: {e}")
        finally:
            self._operation_mutex.unlock()

    def _check_audio_sync(self):
        """音声同期状態をチェック"""
        if not VLC_AVAILABLE or self._stop_flag or self._is_paused:
            return
        
        try:
            audio_time_ms = self.player.get_time()
            if audio_time_ms < 0:
                return
        except Exception as e:
            print(f"[AUDIO] 同期チェックエラー: {e}")

    def _safe_operation(self, operation, operation_name=""):
        """安全な操作ラッパー"""
        if not self._operation_mutex.tryLock(50):
            print(f"[AUDIO] {operation_name}: 操作ミューテックス取得失敗")
            return False
            
        try:
            operation()
            return True
        except Exception as e:
            print(f"[AUDIO] {operation_name}エラー: {e}")
            return False
        finally:
            self._operation_mutex.unlock()

    def _update_vlc_mute_state(self):
        if not VLC_AVAILABLE:
            return
        should_be_muted = self.user_muted or self.internal_muted or self.ai_seek_muted
        try:
            self.player.audio_set_mute(should_be_muted)
        except Exception as e:
            print(f"[AUDIO] ミュート状態更新エラー: {e}")

    def set_internal_mute(self, is_muted):
        if not VLC_AVAILABLE:
            return
        self.internal_muted = is_muted
        self._update_vlc_mute_state()

    def start_playback(self, video_path, start_sec=0.0, ai_mode=False):
        """再生開始 - 安定化版"""
        if not VLC_AVAILABLE or self._stop_flag:
            return False
            
        def _start():
            try:
                # 現在の再生を完全停止
                if self.player.get_state() != vlc.State.Stopped:
                    self.player.stop()
                    time.sleep(0.05)
                
                self.current_media_path = video_path
                media = self.vlc_instance.media_new(video_path)
                self.player.set_media(media)
                
                # AIモードの場合はミュート設定
                if ai_mode:
                    self.ai_seek_muted = True
                    self.ai_seek_start_time = time.time()
                    print("[AUDIO] AIモード: 2秒間ミュート開始")
                else:
                    self.set_internal_mute(True)
                
                self.player.play()
                
                # 再生開始を待機
                for i in range(50):  # タイムアウト延長
                    state = self.player.get_state()
                    if state in (vlc.State.Playing, vlc.State.Paused):
                        break
                    if state == vlc.State.Error:
                        print("[AUDIO] 再生開始エラー状態")
                        return False
                    time.sleep(0.05)
                
                # シーク処理
                if start_sec > 0.0:
                    audio_start_sec = max(0.0, start_sec - self.audio_offset)
                    success = self._safe_seek(audio_start_sec)
                    if not success:
                        print("[AUDIO] 初期シーク失敗")
                
                # 内部ミュート解除（AIモードでない場合）
                if not ai_mode:
                    self.set_internal_mute(False)
                
                self._is_paused = False
                self.last_sync_time = time.time()
                
                print(f"[AUDIO] 再生開始成功: {Path(video_path).name}, 位置: {start_sec:.2f}秒, AIモード: {ai_mode}")
                return True
                
            except Exception as e:
                print(f"[AUDIO] 再生開始例外: {e}")
                return False
        
        return self._safe_operation(_start, "再生開始")

    def _safe_seek(self, seconds):
        """安全なシーク処理"""
        if not self._seek_mutex.tryLock(100):
            return False
            
        try:
            self._seek_in_progress = True
            msec = int(seconds * 1000)
            
            state = self.player.get_state()
            if state not in (vlc.State.Playing, vlc.State.Paused):
                return False
            
            if not self.player.is_seekable():
                return False
            
            # 内部ミュートを設定してシーク
            self.set_internal_mute(True)
            
            # シーク実行
            result = self.player.set_time(msec)
            time.sleep(0.03)
            
            # 実際の位置を確認
            actual_time = self.player.get_time() / 1000.0
            time_diff = abs(seconds - actual_time)
            
            if time_diff > 0.1:
                print(f"[AUDIO] シークずれ検出: 目標{seconds:.2f}s, 実際{actual_time:.2f}s, 差{time_diff:.2f}s")
                for retry in range(2):
                    self.player.set_time(msec)
                    time.sleep(0.02)
                    actual_time = self.player.get_time() / 1000.0
                    time_diff = abs(seconds - actual_time)
                    if time_diff <= 0.1:
                        break
            
            self.set_internal_mute(False)
            return time_diff <= 0.2
            
        except Exception as e:
            print(f"[AUDIO] シーク例外: {e}")
            return False
        finally:
            self._seek_in_progress = False
            self._seek_mutex.unlock()

    def stop_playback(self):
        """再生停止"""
        if not VLC_AVAILABLE:
            return
            
        def _stop():
            try:
                self._is_paused = True
                self.player.stop()
                time.sleep(0.05)
                self.ai_seek_muted = False  # ミュート状態リセット
            except Exception as e:
                print(f"[AUDIO] 停止例外: {e}")
        
        self._safe_operation(_stop, "再生停止")

    def pause_audio(self):
        """一時停止"""
        if not VLC_AVAILABLE or self._is_paused or self._stop_flag:
            return
            
        def _pause():
            try:
                state = self.player.get_state()
                if state == vlc.State.Playing:
                    self.player.pause()
                    self._is_paused = True
                    print("[AUDIO] 音声一時停止")
            except Exception as e:
                print(f"[AUDIO] 一時停止例外: {e}")
        
        self._safe_operation(_pause, "一時停止")

    def resume_audio(self, start_sec, ai_mode=False):
        """再生再開"""
        if not VLC_AVAILABLE or not self._is_paused or self._stop_flag:
            return False
            
        def _resume():
            try:
                state = self.player.get_state()
                
                if state == vlc.State.Paused:
                    # AIモードの場合はミュート設定
                    if ai_mode:
                        self.ai_seek_muted = True
                        self.ai_seek_start_time = time.time()
                        print("[AUDIO] AIモード再開: 2秒間ミュート開始")
                    else:
                        self.set_internal_mute(True)
                    
                    self.player.play()
                    time.sleep(0.03)
                    
                    # 位置調整
                    if start_sec > 0.0:
                        audio_start_sec = max(0.0, start_sec - self.audio_offset)
                        self._safe_seek(audio_start_sec)
                    
                    # 内部ミュート解除（AIモードでない場合）
                    if not ai_mode:
                        self.set_internal_mute(False)
                        
                elif state == vlc.State.Stopped:
                    if self.current_media_path:
                        return self.start_playback(self.current_media_path, start_sec, ai_mode)
                    else:
                        print("[AUDIO] 再生再開エラー: メディアパス不明")
                        return False
                
                self._is_paused = False
                self.last_sync_time = time.time()
                print(f"[AUDIO] 音声再生再開: 位置 {start_sec:.2f}秒, AIモード: {ai_mode}")
                return True
                
            except Exception as e:
                print(f"[AUDIO] 再生再開例外: {e}")
                return False
        
        return self._safe_operation(_resume, "再生再開")

    def seek_to_time(self, seconds, ai_mode=False):
        """時間指定シーク"""
        if not VLC_AVAILABLE or self._stop_flag:
            return
            
        # AIモードの場合はシーク前にミュート
        if ai_mode:
            self.ai_seek_muted = True
            self.ai_seek_start_time = time.time()
            print("[AUDIO] AIモードシーク: 2秒間ミュート開始")
        
        audio_seconds = max(0.0, seconds - self.audio_offset)
        self._safe_seek(audio_seconds)

    def get_current_time(self):
        """現在の再生時間を取得（秒）"""
        if not VLC_AVAILABLE or self._stop_flag:
            return 0.0
        
        try:
            time_ms = self.player.get_time()
            if time_ms < 0:
                return 0.0
            return time_ms / 1000.0
        except:
            return 0.0

    def sync_with_video(self, video_time_sec):
        """映像に音声を同期"""
        if not VLC_AVAILABLE or self._stop_flag:
            return False
        
        try:
            current_audio_time = self.get_current_time()
            target_audio_time = max(0.0, video_time_sec - self.audio_offset)
            time_diff = target_audio_time - current_audio_time
            
            if abs(time_diff) > 0.05:
                was_playing = not self._is_paused
                if was_playing:
                    self.player.pause()
                
                success = self._safe_seek(target_audio_time)
                
                if was_playing and success:
                    self.player.play()
                
                return success
            
            return True
        except Exception as e:
            print(f"[AUDIO-SYNC] 同期エラー: {e}")
            return False

    def set_volume(self, volume):
        """音量設定"""
        if not VLC_AVAILABLE:
            return
        try:
            self.volume = max(0, min(100, volume))
            self.player.audio_set_volume(self.volume)
        except Exception as e:
            print(f"[AUDIO] 音量設定エラー: {e}")

    def toggle_mute(self, is_muted):
        """ミュート切り替え"""
        if not VLC_AVAILABLE:
            return
        try:
            self.user_muted = is_muted
            self._update_vlc_mute_state()
        except Exception as e:
            print(f"[AUDIO] ミュート切り替えエラー: {e}")

    def set_audio_offset(self, offset):
        """音声先行オフセットを設定"""
        self.audio_offset = offset
        print(f"[AUDIO] 音声先行オフセット設定: {offset:.2f}秒")

    def safe_stop(self):
        """安全な停止"""
        print("[AUDIO] 安全停止開始")
        self._stop_flag = True
        
        self.stop_playback()
        
        if not self.wait(1000):
            print("[AUDIO] スレッド終了待機タイムアウト")
            self.terminate()
            self.wait(500)
        
        print("[AUDIO] 安全停止完了")


class ProcessThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, bool)
    fps_updated = pyqtSignal(float)
    progress_updated = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    
    def __init__(self, video_path, model_dir, detection_model_name, frame_cache, 
                start_frame, thread_id, settings, audio_thread=None, video_fps=30.0):
        super().__init__()
        self.video_path = Path(video_path)
        self.model_dir = Path(model_dir)
        self.detection_model_name = detection_model_name
        self.frame_cache = frame_cache
        self.start_frame = start_frame
        self.thread_id = thread_id
        
        self.batch_size = settings.get('batch_size', 16)
        self.queue_size_mb = settings.get('queue_size_mb', 12288)
        self.max_clip_length = settings.get('max_clip_length', 8)
        self.parallel_clips = settings.get('parallel_clips', 4)
        self.playback_speed = settings.get('playback_speed', 1.0)
        
        self.frame_restorer = None
        self.is_running = False
        self._stop_flag = False
        self.is_paused = False
        self.pause_mutex = QMutex()
        
        self.audio_thread = audio_thread
        self.video_fps = video_fps
        self.total_frames = 0
        
        self._seek_requested = False
        self._seek_target = 0
        self._seek_mutex = QMutex()
        self._safe_stop = False
        
        # 音声同期用
        self.last_audio_sync_time = 0
        self.audio_sync_interval = 2.0
        
        print(f"[THREAD-{thread_id}] プロセススレッド初期化完了")
        print(f"[THREAD-{thread_id}] 使用モデル: {self.detection_model_name}")
        print(f"[THREAD-{thread_id}] 再生速度: {self.playback_speed}x")

    def request_seek(self, target_frame):
        if not self._seek_mutex.tryLock(10):
            return False
            
        try:
            self._seek_requested = True
            self._seek_target = target_frame
            print(f"[THREAD-{self.thread_id}] シークリクエスト受信: フレーム{target_frame}")
            return True
        finally:
            self._seek_mutex.unlock()

    def pause(self):
        if not self.pause_mutex.tryLock(10):
            return
            
        try:
            self.is_paused = True
            if self.audio_thread:
                self.audio_thread.pause_audio()
            print(f"[THREAD-{self.thread_id}] 一時停止完了")
        finally:
            self.pause_mutex.unlock()

    def resume(self):
        if not self.pause_mutex.tryLock(10):
            return
            
        try:
            self.is_paused = False
            if self.audio_thread:
                start_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
                # AIモードなのでTrueを渡す
                self.audio_thread.resume_audio(start_sec, True)
            print(f"[THREAD-{self.thread_id}] 再開完了")
        finally:
            self.pause_mutex.unlock()

    def safe_stop(self):
        print(f"[THREAD-{self.thread_id}] 安全停止開始")
        self._safe_stop = True
        self._stop_flag = True
        self.is_running = False
        self.is_paused = False
        
        if self.frame_restorer:
            try:
                self.frame_restorer.stop()
            except Exception as e:
                print(f"[THREAD-{self.thread_id}] フレームレストーラー停止中の例外: {e}")
        
        if self.audio_thread:
            try:
                self.audio_thread.stop_playback()
                time.sleep(0.05)
            except Exception as e:
                print(f"[THREAD-{self.thread_id}] 音声停止中の例外: {e}")
        
        if not self.wait(1000):
            print(f"[THREAD-{self.thread_id}] スレッド終了待機タイムアウト、強制終了")
            self.terminate()
            self.wait(500)
        
        print(f"[THREAD-{self.thread_id}] 安全停止完了")

    def run(self):
        print(f"[THREAD-{self.thread_id}] スレッド開始")
        
        self.is_running = True
        self._stop_flag = False
        self._safe_stop = False
        
        try:
            if not LADA_AVAILABLE:
                print(f"[THREAD-{self.thread_id}] LADA利用不可")
                return
            
            video_meta = video_utils.get_video_meta_data(self.video_path)
            self.total_frames = video_meta.frames_count
            self.video_fps = video_meta.video_fps
            
            print(f"[THREAD-{self.thread_id}] 動画情報: {self.total_frames}フレーム, {self.video_fps}FPS")
            
            if self._stop_flag or self._safe_stop:
                return
            
            # 音声再生開始 - AIモードなのでTrueを渡す
            if self.audio_thread and not self._safe_stop:
                start_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
                audio_success = self.audio_thread.start_playback(str(self.video_path), start_sec, True)
                if not audio_success:
                    print(f"[THREAD-{self.thread_id}] 音声再生開始失敗")
            
            # モデルファイルのパスを構築
            detection_path = self.model_dir / self.detection_model_name
            restoration_path = self.model_dir / "lada_mosaic_restoration_model_generic_v1.2.pth"
            
            print(f"[THREAD-{self.thread_id}] 検知モデルパス: {detection_path}")
            print(f"[THREAD-{self.thread_id}] 復元モデルパス: {restoration_path}")
            
            if not detection_path.exists():
                print(f"[THREAD-{self.thread_id}] エラー: 検知モデルファイルが見つかりません: {detection_path}")
                return
            
            if not restoration_path.exists():
                print(f"[THREAD-{self.thread_id}] エラー: 復元モデルファイルが見つかりません: {restoration_path}")
                return
            
            detection_model, restoration_model, pad_mode = load_models(
                device="cuda:0",
                mosaic_restoration_model_name="basicvsrpp-v1.2",
                mosaic_restoration_model_path=str(restoration_path),
                mosaic_restoration_config_path=None,
                mosaic_detection_model_path=str(detection_path)
            )
            
            if self._stop_flag or self._safe_stop:
                return
            
            # 最適化されたFrameRestorerを作成
            try:
                print(f"[THREAD-{self.thread_id}] 最適化FrameRestorerを作成中...")
                self.frame_restorer = OptimizedFrameRestorer(
                    device="cuda:0",
                    video_file=self.video_path,
                    preserve_relative_scale=True,
                    max_clip_length=self.max_clip_length,
                    mosaic_restoration_model_name="basicvsrpp-v1.2",
                    mosaic_detection_model=detection_model,
                    mosaic_restoration_model=restoration_model,
                    preferred_pad_mode=pad_mode,
                    batch_size=self.batch_size,
                    queue_size_mb=self.queue_size_mb,
                    mosaic_detection=False,
                    parallel_clips=self.parallel_clips
                )
                print(f"[THREAD-{self.thread_id}] 最適化FrameRestorerの作成成功")
                
            except Exception as e:
                print(f"[THREAD-{self.thread_id}] 最適化FrameRestorerの作成に失敗: {e}")
                from frame_restorer import FrameRestorer
                self.frame_restorer = FrameRestorer(
                    device="cuda:0",
                    video_file=self.video_path,
                    preserve_relative_scale=True,
                    max_clip_length=self.max_clip_length,
                    mosaic_restoration_model_name="basicvsrpp-v1.2",
                    mosaic_detection_model=detection_model,
                    mosaic_restoration_model=restoration_model,
                    preferred_pad_mode=pad_mode,
                    mosaic_detection=False
                )
            
            # フレームレストーラーの開始
            start_ns = int((self.start_frame / self.video_fps) * 1_000_000_000)
            print(f"[THREAD-{self.thread_id}] フレームレストーラー開始: フレーム{self.start_frame}, {start_ns}ns")
            self.frame_restorer.start(start_ns=start_ns)
            
            # メイン処理ループ
            frame_count = self.start_frame
            start_time = time.time()
            pause_start_time = 0
            total_pause_duration = 0
            frame_interval = 1.0 / (self.video_fps * self.playback_speed) if self.video_fps > 0 else 0.033
            
            print(f"[THREAD-{self.thread_id}] 再生速度: {self.playback_speed}x, フレーム間隔: {frame_interval:.3f}秒")
            
            frame_restorer_iter = iter(self.frame_restorer)
            pending_ai_frame = None
            last_mode_was_cached = False
            frame_count_at_reset = self.start_frame
            
            self.frame_cache.update_playhead(frame_count)
            
            cache_frames_during_pause = 1800
            paused_cache_count = 0
            
            consecutive_cached_frames = 0
            max_consecutive_cached = 30
            
            print(f"[THREAD-{self.thread_id}] メイン処理ループ開始")
            
            while self.is_running and not self._stop_flag and not self._safe_stop and frame_count < self.total_frames:
                if self._safe_stop:
                    break
                    
                # シークリクエストチェック
                seek_processed = False
                if self._seek_mutex.tryLock(1):
                    try:
                        if self._seek_requested:
                            print(f"[THREAD-{self.thread_id}] シーク処理開始: {self._seek_target}")
                            frame_count = self._seek_target
                            self.start_frame = frame_count
                            start_ns = int((frame_count / self.video_fps) * 1_000_000_000)
                            
                            try:
                                self.frame_restorer.stop()
                            except:
                                pass
                            
                            self.frame_restorer.start(start_ns=start_ns)
                            frame_restorer_iter = iter(self.frame_restorer)
                            pending_ai_frame = None
                            
                            start_time = time.time()
                            total_pause_duration = 0
                            frame_count_at_reset = frame_count
                            last_mode_was_cached = False
                            paused_cache_count = 0
                            pause_start_time = 0
                            
                            self.frame_cache.update_playhead(frame_count)
                            
                            # 音声シーク - AIモードなのでTrueを渡す
                            if self.audio_thread and not self._safe_stop:
                                target_sec = frame_count / self.video_fps
                                self.audio_thread.seek_to_time(target_sec, True)
                                time.sleep(0.05)
                                self.audio_thread.sync_with_video(target_sec)
                            
                            self._seek_requested = False
                            seek_processed = True
                    finally:
                        self._seek_mutex.unlock()
                
                if seek_processed:
                    continue
                
                frame_start_time = time.time()
                
                if frame_count % 30 == 0:
                    self.frame_cache.update_playhead(frame_count)
                
                # 音声同期チェック
                current_time = time.time()
                if (current_time - self.last_audio_sync_time >= 1.0 and
                    self.audio_thread and not self.is_paused and 
                    frame_count % 5 == 0):
                    video_time_sec = frame_count / self.video_fps
                    sync_success = self.audio_thread.sync_with_video(video_time_sec)
                    if sync_success:
                        self.last_audio_sync_time = current_time
                
                is_paused_check = False
                if self.pause_mutex.tryLock(1):
                    try:
                        is_paused_check = self.is_paused
                    finally:
                        self.pause_mutex.unlock()
                
                if is_paused_check and not self._stop_flag and not self._safe_stop:
                    if pause_start_time == 0:
                        pause_start_time = time.time()
                        paused_cache_count = 0
                    
                    # 一時停止中のキャッシュ処理
                    if self.frame_cache.enabled and paused_cache_count < cache_frames_during_pause:
                        if self.frame_cache.get(frame_count + paused_cache_count) is None:
                            try:
                                item = next(frame_restorer_iter)
                                if item is not None:
                                    restored_frame, _ = item
                                    self.frame_cache.put(frame_count + paused_cache_count, restored_frame)
                                    paused_cache_count += 1
                            except StopIteration:
                                break
                        else:
                            paused_cache_count += 1
                    
                    time.sleep(0.01)
                    continue
                
                if pause_start_time > 0:
                    pause_duration = time.time() - pause_start_time
                    total_pause_duration += pause_duration
                    pause_start_time = 0
                    paused_cache_count = 0
                
                if self._stop_flag or self._safe_stop:
                    break
                
                # フレーム処理
                cached_frame = None
                if self.frame_cache.enabled:
                    cached_frame = self.frame_cache.get(frame_count)

                if cached_frame is not None:
                    final_frame = cached_frame
                    is_cached = True
                    consecutive_cached_frames += 1
                    processing_time = 0.0
                    
                    if consecutive_cached_frames > max_consecutive_cached:
                        if self.frame_cache.enabled:
                            self.frame_cache.put(frame_count, None)
                        cached_frame = None
                        consecutive_cached_frames = 0
                    
                    if not last_mode_was_cached:
                        start_time = time.time()
                        total_pause_duration = 0
                        frame_count_at_reset = frame_count
                    
                else:
                    consecutive_cached_frames = 0
                    
                    if last_mode_was_cached:
                        start_time = time.time()
                        total_pause_duration = 0
                        frame_count_at_reset = frame_count
                    
                    if pending_ai_frame is not None:
                        restored_frame, frame_pts = pending_ai_frame
                        pending_ai_frame = None
                    else:
                        try:
                            item = next(frame_restorer_iter)
                            if item is None:
                                break
                            restored_frame, frame_pts = item
                        except StopIteration:
                            break
                    
                    final_frame = restored_frame
                    is_cached = False
                    processing_time = time.time() - frame_start_time
                    
                    if self.frame_cache.enabled:
                        self.frame_cache.record_frame_processing_time(frame_count, processing_time)
                        self.frame_cache.put(frame_count, final_frame)
                
                last_mode_was_cached = is_cached
                
                # フレームレート制御
                frames_since_reset = frame_count - frame_count_at_reset
                target_time = frames_since_reset * frame_interval
                elapsed = time.time() - start_time - total_pause_duration
                wait_time = target_time - elapsed
                
                if wait_time < -0.5:
                    start_time = time.time() - (frames_since_reset * frame_interval)
                    total_pause_duration = 0
                    wait_time = 0
                
                if wait_time > 0:
                    time.sleep(min(wait_time, 0.1))
                
                # フレーム準備シグナル発行
                if not self._safe_stop:
                    self.frame_ready.emit(final_frame, frame_count, is_cached)
                
                # 音声同期
                if (self.audio_thread and frame_count % (int(self.video_fps) * 2) == 0 and 
                    not self._safe_stop and not self.is_paused):
                    current_sec = frame_count / self.video_fps
                    self.audio_thread.sync_with_video(current_sec)
                
                frame_count += 1
                if not self._safe_stop:
                    self.progress_updated.emit(frame_count, self.total_frames)
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time - total_pause_duration
                    actual_fps = (frame_count - self.start_frame) / elapsed if elapsed > 0 else 0
                    if not self._safe_stop:
                        self.fps_updated.emit(actual_fps)
            
            if not self._stop_flag and not self._safe_stop:
                self.finished_signal.emit()
            
        except Exception as e:
            print(f"[THREAD-{self.thread_id}] AI処理エラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[THREAD-{self.thread_id}] スレッド終了処理開始")
            if self.frame_restorer and not self._safe_stop:
                try:
                    self.frame_restorer.stop()
                except Exception as e:
                    print(f"[THREAD-{self.thread_id}] フレームレストーラー停止中の例外: {e}")
            
            self.is_running = False
            print(f"[THREAD-{self.thread_id}] スレッド終了処理完了")


class LadaFinalPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setAcceptDrops(True)
        
        self.settings = self.load_settings()
        
        # キャッシュ設定を確実に含める（追加）
        if 'cache_enabled' not in self.settings:
            self.settings['cache_enabled'] = True
        
        # 再生速度設定を確実に含める
        if 'playback_speed' not in self.settings:
            self.settings['playback_speed'] = 1.0
        
        self.progress_bar = MarkedProgressBar()
        
        # 再生速度制御用変数
        self.playback_speeds = [1.0, 0.5, 0.25, 0.1, 0.05]
        self.current_speed_index = 0
        self.playback_speed = self.settings['playback_speed']  # settingsから読み込み
        
        # デフォルト設定に検知モデルを追加
        if 'detection_model' not in self.settings:
            self.settings['detection_model'] = 'lada_mosaic_detection_model_v3.1_fast.pt'
        
        # 音声先行オフセットを設定に追加
        if 'audio_offset' not in self.settings:
            self.settings['audio_offset'] = 0.3
        
        # スマートキャッシュで初期化
        chunk_frames = self.settings.get('chunk_frames', 150)
        cache_size_mb = self.settings.get('cache_size_mb', 12288)
        cache_enabled = self.settings.get('cache_enabled', True)
        self.frame_cache = SmartChunkBasedCache(
            max_size_mb=cache_size_mb, 
            chunk_frames=chunk_frames,
            enabled=cache_enabled 
        )
        
        self.current_video = None
        self.total_frames = 0
        self.current_frame = 0
        self.video_fps = 30.0
        self.is_playing = False
        self.is_paused = False
        self.thread_counter = 0
        self._seeking = False
        self.ai_processing_enabled = True
        
        # process_threadをNoneで明示的に初期化
        self.process_thread = None
        
        # 範囲再生用変数
        self.range_start = None
        self.range_end = None
        self.range_mode = False
        
        # 再生速度制御用変数 - 修正: 低速再生を追加
        self.playback_speeds = [1.0, 0.5, 0.25, 0.1, 0.05]  # 100%, 50%, 25%, 10%, 5%
        self.current_speed_index = 0  # 1.0から開始
        self.playback_speed = 1.0
        
        # VLCの初期化
        self.vlc_instance = vlc.Instance('--no-video') if VLC_AVAILABLE else None
        self.audio_thread = None
        if VLC_AVAILABLE:
            initial_volume = self.settings.get('audio_volume', 100)
            initial_mute = self.settings.get('audio_muted', False)
            audio_offset = self.settings.get('audio_offset', 0.3)
            
            if isinstance(initial_volume, float):
                initial_volume = int(initial_volume * 100)
            initial_volume = max(0, min(100, initial_volume))
            
            self.audio_thread = AudioThread(self.vlc_instance, initial_volume, initial_mute, audio_offset)
            self.audio_thread.start()
        
        # 音声同期タイマー（2秒間隔）
        self.audio_sync_timer = QTimer()
        self.audio_sync_timer.timeout.connect(self.sync_audio_with_video)
        self.audio_sync_timer.start(2000)  # 2秒間隔
        
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)
        
        self.init_ui()
        print(f"[MAIN] プレイヤー初期化完了 - 音声同期強化版 (オフセット: {self.settings.get('audio_offset', 0.3):.1f}秒)")

    def sync_audio_with_video(self):
        """音声と映像の同期を実行"""
        if not self.is_playing or not self.audio_thread or self.is_paused:
            return
        
        # 同期頻度を制限（2秒に1回）
        current_time = time.time()
        if hasattr(self, '_last_sync_time'):
            if current_time - self._last_sync_time < 2.0:
                return
        
        self._last_sync_time = current_time        

        try:
            # 現在の映像位置（秒）
            video_time_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
            
            # 音声同期を実行
            success = self.audio_thread.sync_with_video(video_time_sec)
            
            if success:
                # 同期成功時のデバッグ情報
                audio_time = self.audio_thread.get_current_time()
                time_diff = abs(video_time_sec - audio_time)
                if time_diff > 0.1:  # 0.1秒以上のずれがある場合のみ表示
                    print(f"[SYNC] 同期完了: 映像{video_time_sec:.2f}s, 音声{audio_time:.2f}s, 差{time_diff:.3f}s")
            else:
                print(f"[SYNC] 同期失敗: 映像{video_time_sec:.2f}s")
                
        except Exception as e:
            print(f"[SYNC] 同期エラー: {e}")

    def init_ui(self):
        """UIの初期化"""
        self.setWindowTitle("LADA REALTIME PLAYER V1.2")
        self.setGeometry(100, 100, 1200, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.setMinimumSize(800, 600)
        
        self.filename_label = QLabel("")
        self.filename_label.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 40, 40, 220);
                color: #00ff00;
                padding: 2px 15px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 3px;
            }
        """)
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.filename_label.setFixedHeight(24)
        self.filename_label.hide()
        layout.addWidget(self.filename_label)
        
        self.video_layout = QVBoxLayout()
        self.video_widget = VideoGLWidget()
        self.video_widget.playback_toggled.connect(self.toggle_playback)
        self.video_widget.video_dropped.connect(self.load_video)
        self.video_widget.seek_requested.connect(self.seek_to_frame)
        self.video_widget.toggle_mute_signal.connect(self.toggle_mute_shortcut)
        self.video_widget.toggle_ai_processing_signal.connect(self.toggle_ai_processing)
        
        # 新しいシグナル接続
        self.video_widget.set_range_start_signal.connect(self.set_range_start)
        self.video_widget.set_range_end_signal.connect(self.set_range_end)
        self.video_widget.reset_range_signal.connect(self.reset_range)
        self.video_widget.seek_to_start_signal.connect(self.seek_to_start)
        self.video_widget.seek_to_end_signal.connect(self.seek_to_end)
        self.video_widget.seek_to_percentage_signal.connect(self.seek_to_percentage)
        self.video_widget.toggle_range_mode_signal.connect(self.toggle_range_mode)
        self.video_widget.frame_step_forward_signal.connect(self.frame_step_forward)
        self.video_widget.frame_step_backward_signal.connect(self.frame_step_backward)
        self.video_widget.change_playback_speed_signal.connect(self.change_playback_speed)
        self.video_widget.reset_playback_speed_signal.connect(self.reset_playback_speed)
        
        self.video_layout.addWidget(self.video_widget)
        layout.addLayout(self.video_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.mousePressEvent = self.seek_click
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(40, 40, 40, 200);
                border: none;
                height: 19px;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        time_audio_layout = QHBoxLayout()
        
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #aaa;
                background-color: rgba(40, 40, 40, 150);
                padding: 2px 8px;
                border-radius: 4px;
                min-width: 150px;
            }
        """)
        self.time_label.setMaximumHeight(20)
        
        # 範囲制御ボタンを追加
        self.range_start_btn = QPushButton("範囲開始")
        self.range_start_btn.setFixedWidth(70)
        self.range_start_btn.clicked.connect(self.set_range_start)
        self.range_start_btn.setToolTip("Ctrl+S: 範囲開始点を現在位置に設定")
        
        self.range_end_btn = QPushButton("範囲終了")
        self.range_end_btn.setFixedWidth(70)
        self.range_end_btn.clicked.connect(self.set_range_end)
        self.range_end_btn.setToolTip("Ctrl+E: 範囲終了点を現在位置に設定")
        
        self.range_reset_btn = QPushButton("範囲リセット")
        self.range_reset_btn.setFixedWidth(80)
        self.range_reset_btn.clicked.connect(self.reset_range)
        self.range_reset_btn.setToolTip("Ctrl+R: 範囲設定をリセット")
        
        self.range_play_btn = QPushButton("範囲再生")
        self.range_play_btn.setFixedWidth(70)
        self.range_play_btn.clicked.connect(self.toggle_range_mode)
        self.range_play_btn.setCheckable(True)
        self.range_play_btn.setToolTip("Ctrl+P: 範囲再生モードをトグル")
        
        # 範囲表示ラベル
        self.range_label = QLabel("範囲: 未設定")
        self.range_label.setStyleSheet("""
            QLabel {
                font-size: 11px; 
                color: #ccc;
                background-color: rgba(60, 60, 80, 150);
                padding: 2px 6px;
                border-radius: 3px;
                min-width: 120px;
            }
        """)
        self.range_label.setMaximumHeight(20)
        
        time_audio_layout.addWidget(self.time_label)
        time_audio_layout.addWidget(self.range_start_btn)
        time_audio_layout.addWidget(self.range_end_btn)
        time_audio_layout.addWidget(self.range_reset_btn)
        time_audio_layout.addWidget(self.range_play_btn)
        time_audio_layout.addWidget(self.range_label)
        time_audio_layout.addStretch(1)
        
        # 音声コントロール
        if VLC_AVAILABLE:
            self.mute_btn = QPushButton("🔇")
            self.mute_btn.setCheckable(True)
            self.mute_btn.setChecked(self.settings.get('audio_muted', False))
            self.mute_btn.setFixedWidth(40)
            self.mute_btn.clicked.connect(self.toggle_user_mute)
            
            self.volume_slider = QSlider(Qt.Orientation.Horizontal)
            self.volume_slider.setRange(0, 100)
            initial_volume_ui = self.settings.get('audio_volume', 100)
            if isinstance(initial_volume_ui, float):
                initial_volume_ui = int(initial_volume_ui * 100)
            self.volume_slider.setValue(max(0, min(100, initial_volume_ui)))
            
            self.volume_slider.setFixedWidth(150)
            self.volume_slider.valueChanged.connect(self.set_volume_slider)
            self.volume_slider.sliderReleased.connect(self.save_audio_settings)
            
            time_audio_layout.addWidget(self.mute_btn)
            time_audio_layout.addWidget(self.volume_slider)
        
        layout.addLayout(time_audio_layout)
        
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("動画を開く")
        self.open_btn.clicked.connect(self.open_video)
        
        self.play_pause_btn = QPushButton("⏸ 一時停止")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setEnabled(False)
        
        self.settings_btn = QPushButton("⚙️ 設定")
        self.settings_btn.clicked.connect(self.open_settings)
        
        self.ai_toggle_btn = QPushButton("🤖 AI: ON")
        self.ai_toggle_btn.setCheckable(True)
        self.ai_toggle_btn.setChecked(True)
        self.ai_toggle_btn.clicked.connect(self.toggle_ai_processing)
        
        # 削除: フレームステップボタン（完全に削除）
        
        # 再生速度ボタン
        self.speed_btn = QPushButton("🎵 速度: 100%")
        self.speed_btn.clicked.connect(self.change_playback_speed)
        self.speed_btn.setToolTip("Zキー: 再生速度変更 (100%→50%→25%→10%→5%)")
        
        self.speed_reset_btn = QPushButton("🎵 速度リセット")
        self.speed_reset_btn.clicked.connect(self.reset_playback_speed)
        self.speed_reset_btn.setToolTip("Aキー: 再生速度を100%にリセット")
        
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.play_pause_btn)
        btn_layout.addWidget(self.settings_btn)
        btn_layout.addWidget(self.ai_toggle_btn)
        # 削除: フレームステップボタンの追加（完全に削除）
        btn_layout.addWidget(self.speed_btn)
        btn_layout.addWidget(self.speed_reset_btn)
        layout.addLayout(btn_layout)
        
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("⚡ FPS: --")
        self.mode_label = QLabel("📊 モード: 待機中")
        self.cache_label = QLabel("💾 キャッシュ: 0 MB")
        self.smart_cache_label = QLabel("🤖 スマート: --")
        self.sync_label = QLabel("🔊 同期: --")  # 同期状態表示追加
        
        for label in [self.fps_label, self.mode_label, self.cache_label, self.smart_cache_label, self.sync_label]:
            label.setMaximumHeight(20)
        
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.mode_label)
        stats_layout.addWidget(self.cache_label)
        stats_layout.addWidget(self.smart_cache_label)
        stats_layout.addWidget(self.sync_label)
        layout.addLayout(stats_layout)
        
        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(100)
        info.setText("""
    V1.2 20251017-1 : 
    操作: F=フルスクリーントグル | Space=再生/停止 | M=ミュートトグル | X=AI処理トグル | 進捗バークリックでシーク
    　　: S=先頭/範囲開始 | E=末尾/範囲終了 | 1-9=10%-90%移動 | Ctrl+S=範囲開始点 | Ctrl+E=範囲終了点 | Ctrl+R=範囲リセット | Ctrl+P=範囲再生モードトグル
    　　: Z=再生速度変更(100%→50%→25%→10%→5%) | A=再生速度100%にリセット
    """)
        layout.addWidget(info)
        
        self.setup_shortcuts()
        
        if self.audio_thread:
            initial_volume_thread = self.settings.get('audio_volume', 100)
            if isinstance(initial_volume_thread, float):
                initial_volume_thread = int(initial_volume_thread * 100)
            initial_volume_thread = max(0, min(100, initial_volume_thread))
            
            self.audio_thread.set_volume(initial_volume_thread)
            self.audio_thread.toggle_mute(self.settings.get('audio_muted', False))
            self.mute_btn.setText("🔇" if self.settings.get('audio_muted', False) else "🔊")

    def update_stats(self):
        """キャッシュ統計を更新"""
        try:
            stats = self.frame_cache.get_stats()
            self.cache_label.setText(f"💾 キャッシュ: {stats['size_mb']:.1f}MB ({stats['total_frames']}f)")
            
            if 'hit_ratio' in stats and 'policy_distribution' in stats:
                hit_ratio = stats['hit_ratio'] * 100
                
                policy_summary = ""
                total_chunks = sum(stats['policy_distribution'].values())
                for policy, count in stats['policy_distribution'].items():
                    percentage = (count / total_chunks) * 100 if total_chunks > 0 else 0
                    if percentage >= 5.0:
                        policy_summary += f"{policy[:2]}:{percentage:.0f}% "
                
                self.smart_cache_label.setText(f"🤖 Hit:{hit_ratio:.0f}% {policy_summary.strip()}")
            
            # 同期状態表示
            if self.audio_thread and self.is_playing and not self.is_paused:
                video_time = self.current_frame / self.video_fps if self.video_fps > 0 else 0
                audio_time = self.audio_thread.get_current_time()
                time_diff = abs(video_time - audio_time)
                
                if time_diff < 0.1:
                    self.sync_label.setText("🔊 同期: ✅良好")
                    self.sync_label.setStyleSheet("color: #00ff00;")
                elif time_diff < 0.5:
                    self.sync_label.setText(f"🔊 同期: ⚠️{time_diff:.2f}s")
                    self.sync_label.setStyleSheet("color: #ffff00;")
                else:
                    self.sync_label.setText(f"🔊 同期: ❌{time_diff:.2f}s")
                    self.sync_label.setStyleSheet("color: #ff0000;")
            else:
                self.sync_label.setText("🔊 同期: --")
                self.sync_label.setStyleSheet("color: #888888;")
                
        except Exception as e:
            pass

    def load_settings(self):
        """設定の読み込み"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    settings = json.load(f)
                    print(f"[MAIN] 設定読み込み: 音量={settings.get('audio_volume')}, ミュート={settings.get('audio_muted')}, オフセット={settings.get('audio_offset')}")
                    default_settings = {
                        'cache_enabled': True,  # デフォルトで有効
                        'audio_offset': 0.3,
                        'detection_model': 'lada_mosaic_detection_model_v3.1_fast.pt', 
                        'batch_size': 16,
                        'queue_size_mb': 12288,
                        'max_clip_length': 8,
                        'cache_size_mb': 12288,
                        'chunk_frames': 150,
                        'audio_volume': 100, 
                        'audio_muted': False,
                        'parallel_clips': 4
                    }
                    for key, value in default_settings.items():
                        if key not in settings:
                            settings[key] = value
                    return settings
            except Exception as e:
                print(f"[MAIN] 設定読み込みエラー: {e}")
        
        return {
            'cache_enabled': True,  # 追加
            'audio_offset': 0.3,
            'detection_model': 'lada_mosaic_detection_model_v3.1_fast.pt',
            'batch_size': 16,
            'queue_size_mb': 12288,
            'max_clip_length': 8,
            'cache_size_mb': 12288,
            'chunk_frames': 150,
            'audio_volume': 100, 
            'audio_muted': False,
            'parallel_clips': 4
        }

    def setup_shortcuts(self):
        """ショートカットの設定"""
        self.shortcut_fullscreen = QShortcut(QKeySequence('F'), self)
        self.shortcut_fullscreen.activated.connect(self.toggle_fullscreen_shortcut)
        
        self.shortcut_escape = QShortcut(QKeySequence('Esc'), self)
        self.shortcut_escape.activated.connect(self.escape_fullscreen_shortcut)
        
        self.shortcut_space = QShortcut(QKeySequence('Space'), self)
        self.shortcut_space.activated.connect(self.toggle_playback)

        self.shortcut_right = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.seek_relative(300))
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.seek_relative(-300))
        self.shortcut_semicolon = QShortcut(QKeySequence(Qt.Key.Key_Semicolon), self)
        self.shortcut_semicolon.activated.connect(lambda: self.seek_relative(30))
        self.shortcut_h = QShortcut(QKeySequence('H'), self)
        self.shortcut_h.activated.connect(lambda: self.seek_relative(-30))
        self.shortcut_l = QShortcut(QKeySequence('L'), self)
        self.shortcut_l.activated.connect(lambda: self.seek_relative(300))
        self.shortcut_j = QShortcut(QKeySequence('J'), self)
        self.shortcut_j.activated.connect(lambda: self.seek_relative(-300))
        self.shortcut_k = QShortcut(QKeySequence('K'), self)
        self.shortcut_k.activated.connect(self.toggle_playback)
        
        self.shortcut_mute = QShortcut(QKeySequence('M'), self)
        self.shortcut_mute.activated.connect(self.toggle_mute_shortcut)
        
        self.shortcut_ai_toggle = QShortcut(QKeySequence('X'), self)
        self.shortcut_ai_toggle.activated.connect(self.toggle_ai_processing)
        
        # 新しいショートカット
        self.shortcut_s = QShortcut(QKeySequence('S'), self)
        self.shortcut_s.activated.connect(self.seek_to_start)
        
        self.shortcut_e = QShortcut(QKeySequence('E'), self)
        self.shortcut_e.activated.connect(self.seek_to_end)
        
        # 数字キーショートカット (1-9)
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda checked=False, percent=i: self.seek_to_percentage(percent))
        
        # 範囲再生ショートカット
        self.shortcut_ctrl_s = QShortcut(QKeySequence('Ctrl+S'), self)
        self.shortcut_ctrl_s.activated.connect(self.set_range_start)
        
        self.shortcut_ctrl_e = QShortcut(QKeySequence('Ctrl+E'), self)
        self.shortcut_ctrl_e.activated.connect(self.set_range_end)
        
        self.shortcut_ctrl_r = QShortcut(QKeySequence('Ctrl+R'), self)
        self.shortcut_ctrl_r.activated.connect(self.reset_range)
        
        self.shortcut_ctrl_p = QShortcut(QKeySequence('Ctrl+P'), self)
        self.shortcut_ctrl_p.activated.connect(self.toggle_range_mode)
        
        # 新しいショートカット
        self.shortcut_frame_back = QShortcut(QKeySequence(','), self)  # < キー
        self.shortcut_frame_back.activated.connect(self.frame_step_backward)
        
        self.shortcut_frame_forward = QShortcut(QKeySequence('.'), self)  # > キー
        self.shortcut_frame_forward.activated.connect(self.frame_step_forward)
        
        self.shortcut_speed_change = QShortcut(QKeySequence('Z'), self)
        self.shortcut_speed_change.activated.connect(self.change_playback_speed)
        
        self.shortcut_speed_reset = QShortcut(QKeySequence('A'), self)
        self.shortcut_speed_reset.activated.connect(self.reset_playback_speed)

    def seek_to_start(self):
        """Sキー：先頭または範囲開始点へ移動"""
        if self.range_mode and self.range_start is not None:
            target_frame = self.range_start
        else:
            target_frame = 0
        
        self.fast_seek_to_frame(target_frame)

    def seek_to_end(self):
        """Eキー：末尾または範囲終了点へ移動"""
        if self.range_mode and self.range_end is not None:
            target_frame = self.range_end
        else:
            target_frame = self.total_frames - 1 if self.total_frames > 0 else 0
        
        self.fast_seek_to_frame(target_frame)

    def seek_to_percentage(self, percent):
        """1-9キー：指定パーセント位置へ移動"""
        if self.total_frames > 0:
            target_frame = int((percent * 0.1) * self.total_frames)
            self.fast_seek_to_frame(target_frame)

    def set_range_start(self):
        """Ctrl+S：範囲再生開始点設定"""
        if self.total_frames == 0:
            return
            
        self.range_start = self.current_frame
        print(f"[RANGE] 開始点設定: {self.range_start}")
        
        if self.range_end is not None and self.range_start > self.range_end:
            self.range_end = self.total_frames - 1
            print(f"[RANGE] RS>REのためREをEEに設定: {self.range_end}")
        
        if self.range_end is None:
            self.range_end = self.total_frames - 1
            print(f"[RANGE] RE未設定のためEEをREに設定: {self.range_end}")
        
        self.update_range_display()
        self.update_progress_bar_marks()
        self.update_mode_label()

    def set_range_end(self):
        """Ctrl+E：範囲再生終了点設定"""
        if self.total_frames == 0:
            return
            
        self.range_end = self.current_frame
        print(f"[RANGE] 終了点設定: {self.range_end}")
        
        if self.range_start is not None and self.range_end < self.range_start:
            self.range_start = 0
            print(f"[RANGE] RE<RSのためSSをRSに設定: {self.range_start}")
        
        if self.range_start is None:
            self.range_start = 0
            print(f"[RANGE] RS未設定のためSSをRSに設定: {self.range_start}")
        
        self.update_range_display()
        self.update_progress_bar_marks()
        self.update_mode_label()

    def reset_range(self):
        """Ctrl+R：範囲再生リセット"""
        self.range_start = None
        self.range_end = None
        self.range_mode = False
        self.range_play_btn.setChecked(False)
        self.update_range_display()
        self.update_progress_bar_marks()
        self.update_mode_label()
        print("[RANGE] 範囲再生リセット")

    def toggle_range_mode(self):
        """Ctrl+P：範囲再生モードトグル"""
        if self.range_start is not None and self.range_end is not None:
            self.range_mode = not self.range_mode
            self.range_play_btn.setChecked(self.range_mode)
            self.update_range_display()
            self.update_progress_bar_marks()
            self.update_mode_label()
            print(f"[RANGE] 範囲再生モード: {'ON' if self.range_mode else 'OFF'}")
        else:
            print("[RANGE] 範囲が設定されていません。先にCtrl+SとCtrl+Eで範囲を設定してください。")

    def update_range_display(self):
        """範囲表示を更新"""
        if self.range_start is not None and self.range_end is not None:
            start_sec = self.range_start / self.video_fps if self.video_fps > 0 else 0
            end_sec = self.range_end / self.video_fps if self.video_fps > 0 else 0
            start_time = self.format_time(start_sec)
            end_time = self.format_time(end_sec)
            mode_status = "🔁 ON" if self.range_mode else "⏸ OFF"
            self.range_label.setText(f"範囲: {start_time}-{end_time} {mode_status}")
        else:
            self.range_label.setText("範囲: 未設定")

    def update_mode_label(self):
        """モード表示を更新"""
        if self.range_mode:
            if self.range_start is not None and self.range_end is not None:
                start_sec = self.range_start / self.video_fps if self.video_fps > 0 else 0
                end_sec = self.range_end / self.video_fps if self.video_fps > 0 else 0
                start_time = self.format_time(start_sec)
                end_time = self.format_time(end_sec)
                self.mode_label.setText(f"📊 モード: 🔄 範囲再生中 ({start_time}-{end_time})")
            else:
                self.mode_label.setText("📊 モード: 🔄 範囲再生中")
        else:
            if self.range_start is not None and self.range_end is not None:
                start_sec = self.range_start / self.video_fps if self.video_fps > 0 else 0
                end_sec = self.range_end / self.video_fps if self.video_fps > 0 else 0
                start_time = self.format_time(start_sec)
                end_time = self.format_time(end_sec)
                self.mode_label.setText(f"📊 モード: 🔄 AI処理中 [範囲設定済: {start_time}-{end_time}]")
            else:
                self.mode_label.setText("📊 モード: 🔄 AI処理中")

    def update_progress_bar_marks(self):
        """進捗バーに範囲マークを更新"""
        if hasattr(self.progress_bar, 'set_range_marks'):
            self.progress_bar.set_range_marks(self.range_start, self.range_end)
        self.progress_bar.update()

    def check_range_loop(self):
        """範囲再生のループチェック"""
        if (self.range_mode and 
            self.range_start is not None and 
            self.range_end is not None and 
            self.current_frame >= self.range_end):
            self.fast_seek_to_frame(self.range_start)

    def frame_step_forward(self):
        """1フレーム進む"""
        if not self.current_video:
            return
            
        # 再生中なら一時停止
        if self.is_playing and not self.is_paused:
            self.toggle_playback()
        
        target_frame = min(self.current_frame + 1, self.total_frames - 1)
        if self.range_mode and self.range_end is not None:
            target_frame = min(target_frame, self.range_end)
        
        self.fast_seek_to_frame(target_frame)

    def frame_step_backward(self):
        """1フレーム戻る"""
        if not self.current_video:
            return
            
        # 再生中なら一時停止
        if self.is_playing and not self.is_paused:
            self.toggle_playback()
        
        target_frame = max(self.current_frame - 1, 0)
        if self.range_mode and self.range_start is not None:
            target_frame = max(target_frame, self.range_start)
        
        self.fast_seek_to_frame(target_frame)

    def change_playback_speed(self):
        """再生速度を変更 - 修正版"""
        # 再生速度の配列を循環
        self.current_speed_index = (self.current_speed_index + 1) % len(self.playback_speeds)
        self.playback_speed = self.playback_speeds[self.current_speed_index]
        
        speed_percent = int(self.playback_speed * 100)
        self.speed_btn.setText(f"🎵 速度: {speed_percent}%")
        
        print(f"[SPEED] 再生速度を {self.playback_speed:.2f}x ({speed_percent}%) に変更")
        
        # 再生速度変更を適用
        if self.is_playing and not self.is_paused:
            if not self.ai_processing_enabled and hasattr(self, 'original_timer'):
                # 原画再生の場合、タイマー間隔を変更
                self.original_timer.stop()
                frame_interval = int(1000 / (self.video_fps * self.playback_speed)) if self.video_fps > 0 else 33
                self.original_timer.start(frame_interval)
                print(f"[SPEED] 原画再生速度変更: {frame_interval}ms間隔")
            else:
                # AI処理の場合、再生を再開
                self.safe_restart_playback(self.current_frame)

    def reset_playback_speed(self):
        """再生速度を100%にリセット - 修正版"""
        self.current_speed_index = 0  # 1.0のインデックス
        self.playback_speed = 1.0
        self.speed_btn.setText("🎵 速度: 100%")
        
        print("[SPEED] 再生速度を 1.0x (100%) にリセット")
        
        # 再生速度変更を適用
        if self.is_playing and not self.is_paused:
            if not self.ai_processing_enabled and hasattr(self, 'original_timer'):
                # 原画再生の場合、タイマー間隔を変更
                self.original_timer.stop()
                frame_interval = int(1000 / (self.video_fps * self.playback_speed)) if self.video_fps > 0 else 33
                self.original_timer.start(frame_interval)
                print(f"[SPEED] 原画再生速度リセット: {frame_interval}ms間隔")
            else:
                # AI処理の場合、再生を再開
                self.safe_restart_playback(self.current_frame)

    def toggle_mute_shortcut(self):
        if self.audio_thread:
            new_mute_state = not self.audio_thread.user_muted
            self.audio_thread.toggle_mute(new_mute_state)
            self.mute_btn.setChecked(new_mute_state)
            self.mute_btn.setText("🔇" if new_mute_state else "🔊")
            
            if new_mute_state:
                self.settings['last_volume'] = self.audio_thread.volume
                self.volume_slider.setValue(0)
            else:
                unmuted_volume = self.settings.get('last_volume', self.settings.get('audio_volume', 100))
                if isinstance(unmuted_volume, float):
                    unmuted_volume = int(unmuted_volume * 100)
                unmuted_volume = max(1, min(100, unmuted_volume))
                
                self.volume_slider.setValue(unmuted_volume)
                self.audio_thread.set_volume(unmuted_volume)
            
            self.save_audio_settings()

    def toggle_user_mute(self, checked):
        if self.audio_thread:
            self.audio_thread.toggle_mute(checked)
            self.mute_btn.setText("🔇" if checked else "🔊")
            
            if checked:
                self.settings['last_volume'] = self.audio_thread.volume
                self.volume_slider.setValue(0)
            else:
                unmuted_volume = self.settings.get('last_volume', self.settings.get('audio_volume', 100))
                if isinstance(unmuted_volume, float):
                    unmuted_volume = int(unmuted_volume * 100)
                unmuted_volume = max(1, min(100, unmuted_volume))
                
                self.volume_slider.setValue(unmuted_volume)
                self.audio_thread.set_volume(unmuted_volume)
            
            self.save_audio_settings()

    def set_volume_slider(self, value):
        if self.audio_thread:
            self.audio_thread.set_volume(value)
            
            if value > 0 and self.audio_thread.user_muted:
                self.audio_thread.toggle_mute(False)
                self.mute_btn.setChecked(False)
                self.mute_btn.setText("🔊")
            
            self.settings['audio_volume'] = value
            self.save_audio_settings()

    def toggle_ai_processing(self):
        """AI処理切り替え - 音声同期維持"""
        current_frame = self.current_frame
        
        self.ai_processing_enabled = not self.ai_processing_enabled
        
        if self.ai_processing_enabled:
            self.ai_toggle_btn.setText("🤖 AI: ON")
            self.ai_toggle_btn.setChecked(True)
            self.mode_label.setText("📊 モード: 🔄 AI処理有効")
        else:
            self.ai_toggle_btn.setText("🎥 原画: ON")
            self.ai_toggle_btn.setChecked(False)
            self.mode_label.setText("📊 モード: 🎥 原画再生")
        
        if self.current_video:
            self.safe_restart_playback(current_frame)

    def safe_restart_playback(self, start_frame):
        """安全な再生再開 - 音声同期確保"""
        print(f"[MAIN] 安全な再生再開: フレーム{start_frame}")
        
        self.safe_stop()
        
        if self.range_mode and self.range_start is not None and self.range_end is not None:
            start_frame = max(self.range_start, min(start_frame, self.range_end))
        else:
            start_frame = max(0, min(start_frame, self.total_frames - 1))
        
        self.start_processing_from_frame(start_frame)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.load_video(file_path)
                event.acceptProposedAction()

    def is_video_file(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ts']
        file_ext = Path(file_path).suffix.lower()
        return file_ext in video_extensions

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_frame_ready(self, frame, frame_num, is_cached, thread_id):
        if self.process_thread and thread_id == self.process_thread.thread_id:
            self.current_frame = frame_num
            self.video_widget.update_frame(frame)
            
            self.frame_cache.update_playhead(frame_num)
            
            self.check_range_loop()
            
            current_sec = frame_num / self.video_fps if self.video_fps > 0 else 0
            total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            
            current_time = self.format_time(current_sec)
            total_time = self.format_time(total_sec)
            self.time_label.setText(f"{current_time} / {total_time}")
            
            if is_cached:
                if self.range_mode:
                    start_sec = self.range_start / self.video_fps if self.range_start is not None and self.video_fps > 0 else 0
                    end_sec = self.range_end / self.video_fps if self.range_end is not None and self.video_fps > 0 else 0
                    start_time = self.format_time(start_sec)
                    end_time = self.format_time(end_sec)
                    self.mode_label.setText(f"📊 モード: 💾 範囲再生中 ({start_time}-{end_time})")
                else:
                    self.mode_label.setText("📊 モード: 💾 キャッシュ再生")
                if not self.is_paused:
                    self.video_widget.set_progress_bar_color('yellow')
            else:
                if self.range_mode:
                    start_sec = self.range_start / self.video_fps if self.range_start is not None and self.video_fps > 0 else 0
                    end_sec = self.range_end / self.video_fps if self.range_end is not None and self.video_fps > 0 else 0
                    start_time = self.format_time(start_sec)
                    end_time = self.format_time(end_sec)
                    self.mode_label.setText(f"📊 モード: 🔄 範囲再生中 ({start_time}-{end_time})")
                else:
                    self.mode_label.setText("📊 モード: 🔄 AI処理中")
                if not self.is_paused:
                    if self.range_mode:
                        self.video_widget.set_progress_bar_color('#00ff00')
                    else:
                        self.video_widget.set_progress_bar_color('#0088ff')

    def on_progress_update(self, current, total):
        self.current_frame = current
        self.progress_bar.setValue(current)
        self.video_widget.update_progress(current)

    def on_processing_finished(self):
        print("[MAIN] AI処理が完了しました")
        self.safe_stop()
        self.mode_label.setText("📊 モード: 完了")

    def seek_relative(self, delta):
        """高速相対シーク - 音声同期"""
        if self.total_frames == 0 or not self.current_video:
            return
        
        if self.range_mode and self.range_start is not None and self.range_end is not None:
            target_frame = max(self.range_start, min(self.current_frame + delta, self.range_end))
        else:
            target_frame = max(0, min(self.current_frame + delta, self.total_frames - 1))
        
        self.current_frame = target_frame
        self.progress_bar.setValue(target_frame)
        self.video_widget.update_progress(target_frame)
        
        cached_frame = self.frame_cache.get(target_frame)
        if cached_frame is not None:
            self.video_widget.update_frame(cached_frame)
        
        self.fast_seek_to_frame(target_frame)

    def fast_seek_to_frame(self, target_frame):
        """高速シーク処理 - 音声同期強化"""
        if not self.current_video or self._seeking:
            return
        
        self._seeking = True
        
        if self.range_mode and self.range_start is not None and self.range_end is not None:
            target_frame = max(self.range_start, min(target_frame, self.range_end))
        
        # 音声シーク（AIモード情報を渡す）
        if self.audio_thread:
            target_sec = target_frame / self.video_fps if self.video_fps > 0 else 0
            # AIモードかどうかを渡す（追加）
            ai_mode = self.ai_processing_enabled
            self.audio_thread.seek_to_time(target_sec, ai_mode)
            # シーク後の同期確認
            self.audio_thread.sync_with_video(target_sec)
        
        if self.process_thread and self.process_thread.isRunning():
            success = self.process_thread.request_seek(target_frame)
            if not success:
                print("[MAIN] シークリクエスト送信失敗")
        else:
            self.start_processing_from_frame(target_frame)
        
        self._seeking = False

    def seek_to_frame(self, target_frame):
        """互換性のためのシーク処理"""
        if self.range_mode and self.range_start is not None and self.range_end is not None:
            target_frame = max(self.range_start, min(target_frame, self.range_end))
        
        self.fast_seek_to_frame(target_frame)

    def closeEvent(self, event):
        """終了処理"""
        print("=== 安全な終了処理 ===")
        
        self.safe_stop()
        
        if self.audio_thread:
            time.sleep(0.1)
            self.audio_thread.safe_stop()
        
        if hasattr(self, 'video_widget') and self.video_widget.texture_id:
            try:
                self.video_widget.makeCurrent()
                glDeleteTextures([self.video_widget.texture_id])
            except:
                pass
        
        self.frame_cache.clear()
        
        self.save_settings()
        
        print("=== 終了処理完了 ===")
        event.accept()

    def seek_click(self, event):
        if self.total_frames > 0:
            pos = event.pos().x()
            width = self.progress_bar.width()
            target_frame = int((pos / width) * self.total_frames)
            
            if self.range_mode and self.range_start is not None and self.range_end is not None:
                target_frame = max(self.range_start, min(target_frame, self.range_end))
            
            self.fast_seek_to_frame(target_frame)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "動画選択", "", "Videos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)"
        )
        if path:
            self.load_video(path)

    def open_settings(self):
        dialog = SettingsDialog(self, self.settings)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()
            
            needs_restart = False
            needs_cache_rebuild = False
            
            # 設定変更の検出
            cache_related_settings = [
                'batch_size', 'queue_size_mb', 'max_clip_length',
                'cache_size_mb', 'chunk_frames', 'parallel_clips'
            ]
            
            for key in cache_related_settings:
                if new_settings.get(key) != self.settings.get(key):
                    needs_restart = True
                    if key in ['chunk_frames', 'cache_size_mb', 'cache_enabled']:  # cache_enabledを追加
                        needs_cache_rebuild = True
                    break
            
            # 検知モデル変更の検出を追加
            if new_settings.get('detection_model') != self.settings.get('detection_model'):
                needs_restart = True
                needs_cache_rebuild = True
                print(f"[MAIN] 検知モデル変更検出: {self.settings.get('detection_model')} -> {new_settings.get('detection_model')}")
            
            # 音声オフセット設定の更新
            if 'audio_offset' in new_settings and new_settings['audio_offset'] != self.settings.get('audio_offset', 0.3):
                if self.audio_thread:
                    self.audio_thread.set_audio_offset(new_settings['audio_offset'])
                    print(f"[MAIN] 音声先行オフセットを {new_settings['audio_offset']:.2f}秒に更新")
            
            # キャッシュ有効/無効設定の変更検出を追加
            if new_settings.get('cache_enabled') != self.settings.get('cache_enabled', True):
                needs_restart = True
                needs_cache_rebuild = True
                print(f"[MAIN] キャッシュ設定変更検出: {self.settings.get('cache_enabled')} -> {new_settings.get('cache_enabled')}")
            
            if needs_restart:
                # 現在の設定を保存
                self.settings.update(new_settings)
                self.save_settings()

                print("[MAIN] 設定変更 - 安全なリセット実行")
                self.safe_stop()
                
                if needs_cache_rebuild:
                    # 新しい設定でキャッシュを完全に再構築
                    chunk_frames = self.settings['chunk_frames']
                    cache_size_mb = self.settings['cache_size_mb']
                    cache_enabled = self.settings['cache_enabled']
                    
                    # 古いキャッシュを完全にクリア
                    if hasattr(self, 'frame_cache'):
                        self.frame_cache.clear()
                    
                    # 新しいキャッシュインスタンスを作成
                    self.frame_cache = SmartChunkBasedCache(
                        max_size_mb=cache_size_mb,
                        chunk_frames=chunk_frames,
                        enabled=cache_enabled
                    )
                    print(f"[MAIN] キャッシュ再構築: サイズ={cache_size_mb}MB, チャンク={chunk_frames}, 有効={cache_enabled}")
                else:
                    # 設定のみ更新してキャッシュは維持
                    self.frame_cache = SmartChunkBasedCache(
                        max_size_mb=self.settings['cache_size_mb'],
                        chunk_frames=self.settings.get('chunk_frames', 150)
                    )
                
                # 現在の動画があれば再読み込み
                if self.current_video:
                    current_frame = self.current_frame  # 現在のフレームを保存
                    self.load_video(self.current_video)
                    # 同じフレーム位置から再開
                    QTimer.singleShot(100, lambda: self.fast_seek_to_frame(current_frame))
                
                msg = QMessageBox(self)
                msg.setWindowTitle("設定変更")
                if needs_cache_rebuild:
                    if new_settings.get('detection_model') != self.settings.get('detection_model'):
                        msg.setText(f"検知モデルを変更しました: {new_settings.get('detection_model')}\n再生を再開します。")
                    elif new_settings.get('cache_enabled') != self.settings.get('cache_enabled'):
                        state = "有効" if new_settings.get('cache_enabled') else "無効"
                        msg.setText(f"キャッシュ管理を{state}に変更しました。\n再生を再開します。")
                    else:
                        msg.setText("キャッシュ設定を変更しました。\nキャッシュを再構築します。")
                else:
                    msg.setText("処理設定を変更しました。\n再生を再開します。")
                msg.setIcon(QMessageBox.Icon.Information)
                msg.exec()
            else:
                # 再起動不要な設定変更（音声設定など）
                self.settings.update(new_settings)
                self.save_settings()
                print("[MAIN] 設定を更新しました（再起動不要）")
                
    def save_settings(self):
        """設定を保存"""
        if self.audio_thread:
            if not self.audio_thread.user_muted:
                self.settings['audio_volume'] = self.audio_thread.volume
            self.settings['audio_muted'] = self.audio_thread.user_muted
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"[MAIN] 設定を保存: 音量={self.settings.get('audio_volume')}, ミュート={self.settings.get('audio_muted')}, オフセット={self.settings.get('audio_offset')}")
        except Exception as e:
            print(f"[MAIN] 設定保存失敗: {e}")

    def toggle_fullscreen_shortcut(self):
        self.video_widget.toggle_fullscreen()

    def escape_fullscreen_shortcut(self):
        if self.video_widget.is_fullscreen:
            self.video_widget.toggle_fullscreen()

    def save_audio_settings(self):
        self.save_settings()

    def load_video(self, path):
        print(f"[MAIN] 動画読み込み: {path}")
        self.safe_stop()
        self.frame_cache.clear()
        self.video_widget.clear_frame()
        
        self.reset_range()
        
        self.current_video = path
        
        fullpath = str(Path(path).resolve())
        max_length = 100
        if len(fullpath) > max_length:
            fullpath = "..." + fullpath[-(max_length-3):]
        self.filename_label.setText(f"🎬 {fullpath}")
        self.filename_label.show()
        
        self.original_capture = None
        if not self.ai_processing_enabled:
            try:
                self.original_capture = cv2.VideoCapture(str(path))
                if not self.original_capture.isOpened():
                    print("[MAIN] 元動画の読み込みに失敗")
                    self.original_capture = None
            except Exception as e:
                print(f"[MAIN] 元動画キャプチャ作成失敗: {e}")
                self.original_capture = None
        
        try:
            if self.ai_processing_enabled and LADA_AVAILABLE:
                video_meta = video_utils.get_video_meta_data(path)
                self.total_frames = video_meta.frames_count
                self.video_fps = video_meta.video_fps
            else:
                temp_capture = cv2.VideoCapture(str(path))
                if temp_capture.isOpened():
                    self.total_frames = int(temp_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.video_fps = temp_capture.get(cv2.CAP_PROP_FPS)
                    temp_capture.release()
                else:
                    self.total_frames = 0
                    self.video_fps = 30.0
            
            self.progress_bar.setMaximum(self.total_frames)
            self.video_widget.set_video_info(self.total_frames, self.video_fps)
            
        except Exception as e:
            print(f"[MAIN] 動画メタデータ取得失敗: {e}")
            self.total_frames = 0
            self.video_fps = 30.0
        
        self.start_processing_from_frame(0)
        mode_text = "🎥 原画" if not self.ai_processing_enabled else "🤖 AI"
        self.mode_label.setText(f"📊 選択: {Path(path).name} ({mode_text})")

    def start_processing_from_frame(self, start_frame):
        if not self.current_video:
            return
        
        print(f"[MAIN] フレーム{start_frame}から再生開始 (AI処理: {self.ai_processing_enabled})")
        
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.isRunning():
            print("[MAIN] 既存のAIスレッドが動作中です。安全停止します。")
            self.process_thread.safe_stop()
        
        if hasattr(self, 'original_timer') and self.original_timer and self.original_timer.isActive():
            print("[MAIN] 既存の原画タイマーが動作中です。停止します。")
            self.original_timer.stop()
        
        if not self.ai_processing_enabled:
            self.start_original_playback(start_frame)
            return
        
        if not LADA_AVAILABLE:
            self.mode_label.setText("エラー: LADA利用不可")
            return
        
        if self.process_thread and self.process_thread.isRunning():
            print("[MAIN] スレッドがまだ動作しています。処理を中止します。")
            return
        
        model_dir = LADA_BASE_PATH / "model_weights"
        
        detection_model_name = self.settings.get('detection_model', 'lada_mosaic_detection_model_v3.1_fast.pt')
        detection_path = model_dir / detection_model_name
        restoration_path = model_dir / "lada_mosaic_restoration_model_generic_v1.2.pth"
        
        print(f"[MAIN] 選択された検知モデル: {detection_model_name}")
        print(f"[MAIN] 検知モデルパス: {detection_path}")
        print(f"[MAIN] 復元モデルパス: {restoration_path}")
        
        if not detection_path.exists():
            self.mode_label.setText(f"エラー: 検知モデルなし - {detection_model_name}")
            print(f"[MAIN] 検知モデルファイルが見つかりません: {detection_path}")
            return
        
        if not restoration_path.exists():
            self.mode_label.setText("エラー: 復元モデルなし")
            print(f"[MAIN] 復元モデルファイルが見つかりません: {restoration_path}")
            return
        
        self.thread_counter += 1
        current_id = self.thread_counter
        
        current_settings = self.settings.copy()
        current_settings['playback_speed'] = self.playback_speed
        
        self.process_thread = ProcessThread(
            self.current_video, 
            model_dir,
            detection_model_name,
            self.frame_cache, 
            start_frame, 
            current_id, 
            current_settings,  # 修正: current_settingsを使用
            audio_thread=self.audio_thread, 
            video_fps=self.video_fps
        )
        
        self.process_thread.frame_ready.connect(
            lambda frame, num, cached: self.on_frame_ready(frame, num, cached, current_id)
        )
        self.process_thread.fps_updated.connect(
            lambda fps: self.fps_label.setText(f"⚡ FPS: {fps:.1f}")
        )
        self.process_thread.progress_updated.connect(
            lambda c, t: self.on_progress_update(c, t)
        )
        self.process_thread.finished_signal.connect(self.on_processing_finished)
        
        self.process_thread.start()
        self.is_playing = True
        self.is_paused = False
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ 一時停止")
        self.update_mode_label()
        if self.range_mode:
            self.video_widget.set_progress_bar_color('#0088ff')
        else:
            self.video_widget.set_progress_bar_color('#00ff00')
        
        print(f"[MAIN] AI処理スレッド開始完了: ID{current_id}")

    def start_original_playback(self, start_frame):
        """AI処理無効時の元動画再生 - 音声同期強化"""
        print(f"[MAIN] 原画再生開始: フレーム{start_frame}")
        
        if hasattr(self, 'original_capture') and self.original_capture:
            self.original_capture.release()
            self.original_capture = None
        
        if hasattr(self, 'original_timer') and self.original_timer:
            self.original_timer.stop()
            self.original_timer = None
        
        try:
            self.original_capture = cv2.VideoCapture(str(self.current_video))
            if not self.original_capture.isOpened():
                print("[MAIN] 元動画の読み込みに失敗")
                self.mode_label.setText("エラー: 動画読み込み失敗")
                return
        except Exception as e:
            print(f"[MAIN] 元動画キャプチャ作成失敗: {e}")
            self.mode_label.setText("エラー: 動画読み込み失敗")
            return
        
        self.original_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.current_frame = start_frame
        
        actual_pos = self.original_capture.get(cv2.CAP_PROP_POS_FRAMES)
        print(f"[MAIN] 原画再生: 要求フレーム={start_frame}, 実際の位置={actual_pos}")
        
        ret, first_frame = self.original_capture.read()
        if ret:
            self.video_widget.update_frame(first_frame)
            self.current_frame = start_frame + 1
        else:
            print("[MAIN] 最初のフレーム読み込み失敗、先頭にリセット")
            self.original_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            ret, first_frame = self.original_capture.read()
            if ret:
                self.video_widget.update_frame(first_frame)
                self.current_frame = 1
        
        self.progress_bar.setValue(self.current_frame)
        self.video_widget.update_progress(self.current_frame)
        
        self.original_timer = QTimer()
        self.original_timer.timeout.connect(self.update_original_frame)
        frame_interval = int(1000 / (self.video_fps * self.playback_speed)) if self.video_fps > 0 else 33
        self.original_timer.start(frame_interval)
        
        self.is_playing = True
        self.is_paused = False
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ 一時停止")
        self.update_mode_label()
        
        if self.range_mode:
            self.video_widget.set_progress_bar_color('#0088ff')
        else:
            self.video_widget.set_progress_bar_color('#00ff00')
        
        self.frame_cache.update_playhead(self.current_frame)
        
        if self.audio_thread:
            start_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
            self.audio_thread.start_playback(str(self.current_video), start_sec)
        
        print(f"[MAIN] 原画再生開始完了: フレーム{self.current_frame}, 間隔{frame_interval}ms")

    def update_original_frame(self):
        """原画フレーム更新"""
        if not hasattr(self, 'original_capture') or not self.original_capture or not self.is_playing or self.is_paused:
            return
        
        ret, frame = self.original_capture.read()
        if ret:
            self.video_widget.update_frame(frame)
            
            self.progress_bar.setValue(self.current_frame)
            self.video_widget.update_progress(self.current_frame)
            
            self.check_range_loop()
            
            if self.current_frame % 30 == 0:
                self.frame_cache.update_playhead(self.current_frame)
            
            current_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
            total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            current_time = self.format_time(current_sec)
            total_time = self.format_time(total_sec)
            self.time_label.setText(f"{current_time} / {total_time}")
            
            self.current_frame += 1
            
            if self.current_frame >= self.total_frames:
                self.original_timer.stop()
                self.is_playing = False
                self.play_pause_btn.setText("▶ 再生")
                self.mode_label.setText("📊 モード: 🎥 再生完了")
        else:
            self.original_timer.stop()
            self.is_playing = False
            self.play_pause_btn.setText("▶ 再生")
            self.mode_label.setText("📊 モード: 🎥 再生完了")

    def toggle_playback(self):
        """安全な再生/一時停止トグル"""
        if not self.ai_processing_enabled and hasattr(self, 'original_timer'):
            if self.is_paused:
                self.original_timer.start()
                self.is_paused = False
                self.play_pause_btn.setText("⏸ 一時停止")
                self.update_mode_label()
                if self.range_mode:
                    self.video_widget.set_progress_bar_color('#0088ff')
                else:
                    self.video_widget.set_progress_bar_color('#00ff00')
                
                if self.audio_thread:
                    start_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
                    # 原画モードなのでAIモードはFalse（変更）
                    self.audio_thread.resume_audio(start_sec, False)
            else:
                self.original_timer.stop()
                self.is_paused = True
                self.play_pause_btn.setText("▶ 再開")
                self.mode_label.setText("📊 モード: 🎥 一時停止中")
                self.video_widget.set_progress_bar_color('red')
                
                if self.audio_thread:
                    self.audio_thread.pause_audio()
            return
        
        if not self.process_thread or not self.process_thread.isRunning():
            if self.current_video:
                self.start_processing_from_frame(self.current_frame)
            return
        
        if self.is_paused:
            self.process_thread.resume()
            self.is_paused = False
            self.play_pause_btn.setText("⏸ 一時停止")
            self.update_mode_label()
            if self.range_mode:
                self.video_widget.set_progress_bar_color('#0088ff')
            else:
                self.video_widget.set_progress_bar_color('#00ff00')
                
            # AIモード再開時に音声も再開（AIモード情報を渡す）
            if self.audio_thread:
                start_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
                self.audio_thread.resume_audio(start_sec, True)
        else:
            self.process_thread.pause()
            self.is_paused = True
            self.play_pause_btn.setText("▶ 再開")
            self.mode_label.setText("📊 モード: ⏸ 一時停止中")
            self.video_widget.set_progress_bar_color('red')

            if self.audio_thread:
                self.audio_thread.pause_audio()  # ← この行を追加

    def safe_stop(self):
        """安全な停止"""
        print("[MAIN] 安全停止開始")
        
        self.is_playing = False
        self.is_paused = False
        
        if hasattr(self, 'original_timer') and self.original_timer:
            self.original_timer.stop()
        
        if hasattr(self, 'original_capture') and self.original_capture:
            try:
                self.original_capture.release()
            except Exception as e:
                print(f"[MAIN] 原画キャプチャ解放エラー: {e}")
            self.original_capture = None
        
        if hasattr(self, 'process_thread') and self.process_thread:
            # 修正: 再生停止時に音声を完全に停止
            if self.audio_thread:
                self.audio_thread.stop_playback()
                time.sleep(0.03)
                
            self.process_thread.safe_stop()
            self.process_thread = None
        
        self.play_pause_btn.setText("▶ 再生")
        self.play_pause_btn.setEnabled(self.current_video is not None)
        
        print("[MAIN] 安全停止完了")


def main():
    app = QApplication(sys.argv)
    player = LadaFinalPlayer()
    player.show()
    
    def safe_quit():
        player.close()
    
    app.aboutToQuit.connect(safe_quit)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
