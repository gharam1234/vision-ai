"""
시각화 렌더러 모듈
감지 결과(MediaPipe Pose 뼈대, Hands 손가락 관절, 바박), 위험구역, 긴급 배너 오버레이
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Dict
from loguru import logger

from core.tracker import TrackedObject
from core.zone_manager import DangerZone
from processing.intrusion_detector import IntrusionEvent, IntrusionState
from utils.geometry import get_bottom_center


class FrameRenderer:
    """프레임 시각화 렌더러 (MediaPipe 지원 버전)"""

    # BGR 색상 정의
    COLOR_SAFE = (0, 200, 0)          # 녹색 - 안전한 작업자
    COLOR_DANGER = (0, 0, 255)        # 빨강 - 위험구역 내 또는 긴급 위협 작업자
    COLOR_WARNING = (0, 140, 255)     # 주황 - 일반 제스처 감지
    COLOR_WARNING_BG = (0, 0, 180)    # 경고 배경
    COLOR_TEXT = (255, 255, 255)      # 흰색 텍스트
    COLOR_FPS = (0, 255, 255)         # 노란색 FPS
    COLOR_ZONE_FILL_ALPHA = 0.25      # 위험구역 채우기 투명도

    def __init__(self):
        self._fps_history: list[float] = []
        self._last_frame_time = time.time()
        self._current_fps: float = 0.0

    def render(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        danger_zones: list[DangerZone],
        intrusion_events: list[IntrusionEvent],
        active_threats: Optional[list[dict]] = None
    ) -> np.ndarray:
        """
        프레임에 모든 시각화 요소(위험구역, 전신 뼈대, 손가락 마디, 바운딩 박스) 오버레이
        """
        display = frame.copy()
        threats = active_threats or []

        # 1. 위험구역 그리기
        display = self._draw_zones(display, danger_zones)

        # 2. 작업자별 Pose 스켈레톤 및 Hands 손가락 관절 그리기
        for obj in tracked_objects:
            # MediaPipe Pose 그리기 (33개 관절)
            self._draw_skeleton(display, obj.keypoints)
            
            # MediaPipe Hands 그리기 (각 손당 21개 랜드마크)
            self._draw_hand_landmarks(display, obj.hand_landmarks)
            
        # 3. 작업자 바운딩 박스 및 위협 라벨 텍스트 표출
        display = self._draw_tracked_objects(display, tracked_objects, threats)

        # 4. 화면 외곽 점멸 및 상단 긴급 배너 알림 그리기
        if threats:
            display = self._draw_threat_warnings(display, threats, danger_zones)

        # 5. 실시간 상태보드 그리기 (FPS/통계)
        display = self._draw_status_info(display, tracked_objects, danger_zones, threats)

        self._update_fps()

        return display

    def _draw_skeleton(self, frame: np.ndarray, keypoints: Optional[np.ndarray]) -> None:
        """MediaPipe Pose 33개 관절 뼈대 그리기"""
        if keypoints is None or len(keypoints) < 33:
            return

        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]

        # MediaPipe Pose 뼈대 연결 정의 (i, j)
        SKELETON_CONNECTIONS = [
            # 얼굴 부근
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # 어깨와 몸통
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 왼팔
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # 오른팔
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # 왼다리
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # 오른다리
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]

        conf_thresh = 0.4
        
        # 1. 뼈대 라인 그리기
        for p1, p2 in SKELETON_CONNECTIONS:
            x1, y1, c1 = keypoints[p1]
            x2, y2, c2 = keypoints[p2]
            if c1 >= conf_thresh and c2 >= conf_thresh:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 120), 2)

        # 2. 관절 서클 그리기
        for x, y, conf in keypoints:
            if conf >= conf_thresh:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 180, 0), -1)

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_lms: Optional[List]) -> None:
        """MediaPipe Hands 21개 손가락 관절 랜드마크 그리기"""
        if hand_lms is None or len(hand_lms) == 0:
            return

        # 손가락 뼈대 연결 정의
        HAND_CONNECTIONS = [
            # 엄지
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 검지
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 중지
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 약지
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 새끼
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        for hand in hand_lms:
            landmarks = hand["landmarks"]
            score = hand["score"]
            label = hand["label"]  # "Left" or "Right"
            
            # 손목 라벨 색상 (왼손: 연보라, 오른손: 연녹색)
            color_joint = (200, 100, 255) if label == "Left" else (100, 255, 200)
            color_line = (240, 240, 240)

            # 1. 손 뼈대 라인 그리기
            for p1, p2 in HAND_CONNECTIONS:
                x1, y1, _ = landmarks[p1]
                x2, y2, _ = landmarks[p2]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_line, 1)

            # 2. 손 관절 노드 그리기
            for x, y, _ in landmarks:
                cv2.circle(frame, (int(x), int(y)), 3, color_joint, -1)

    def _draw_zones(
        self, frame: np.ndarray, zones: list[DangerZone]
    ) -> np.ndarray:
        """위험구역 폴리곤 표시"""
        overlay = frame.copy()

        for zone in zones:
            pts = zone.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(overlay, [pts.reshape(-1, 2)], zone.color)
            cv2.polylines(frame, [pts], True, zone.color, 2)

            centroid = zone.polygon.mean(axis=0).astype(int)
            label = f"{zone.name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            cv2.rectangle(
                frame,
                (centroid[0] - tw // 2 - 5, centroid[1] - th - 5),
                (centroid[0] + tw // 2 + 5, centroid[1] + 5),
                zone.color, -1
            )
            cv2.putText(
                frame, label,
                (centroid[0] - tw // 2, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1
            )

        cv2.addWeighted(overlay, self.COLOR_ZONE_FILL_ALPHA, frame,
                        1 - self.COLOR_ZONE_FILL_ALPHA, 0, frame)
        return frame

    def _draw_tracked_objects(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        active_threats: list[dict]
    ) -> np.ndarray:
        """추적 작업자 바운딩박스 및 위험 상태 라벨 인쇄"""
        threat_map: Dict[int, List[dict]] = {}
        for t in active_threats:
            tid = t["tracker_id"]
            if tid not in threat_map:
                threat_map[tid] = []
            threat_map[tid].append(t)

        for obj in objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            is_threat = obj.tracker_id in threat_map

            color = self.COLOR_SAFE
            threat_label = ""
            
            if is_threat:
                threat_list = threat_map[obj.tracker_id]
                has_critical = any(t["severity"] == "critical" for t in threat_list)
                color = self.COLOR_DANGER if has_critical else self.COLOR_WARNING
                
                msgs = [t["msg"] for t in threat_list]
                threat_label = " | ".join(msgs)

            # 바운딩박스
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 위협 시 네 모서리 하이라이트
            if is_threat:
                length = min(15, int((x2 - x1) * 0.2))
                cv2.line(frame, (x1, y1), (x1 + length, y1), color, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + length), color, 4)
                cv2.line(frame, (x2, y1), (x2 - length, y1), color, 4)
                cv2.line(frame, (x2, y1), (x2, y1 + length), color, 4)
                cv2.line(frame, (x1, y2), (x1 + length, y2), color, 4)
                cv2.line(frame, (x1, y2), (x1, y2 - length), color, 4)
                cv2.line(frame, (x2, y2), (x2 - length, y2), color, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - length), color, 4)

            foot = get_bottom_center(obj.bbox)
            cv2.circle(frame, foot, 5, color, -1)

            base_label = f"ID:{obj.tracker_id} {obj.confidence:.0%}"
            label = f"[{threat_label}] {base_label}" if is_threat else base_label

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw + 10, y1),
                color, -1
            )
            cv2.putText(
                frame, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEXT, 1
            )

        return frame

    def _draw_threat_warnings(
        self,
        frame: np.ndarray,
        active_threats: list[dict],
        danger_zones: list[DangerZone]
    ) -> np.ndarray:
        """위험 상태 테두리 블링킹 및 중앙 상단 경보 패널 연출"""
        h, w = frame.shape[:2]
        now = time.time()

        has_critical = any(t["severity"] == "critical" for t in active_threats)
        flash_freq = 5.0 if has_critical else 3.0
        is_flash_on = int(now * flash_freq) % 2 == 0

        if is_flash_on:
            border_color = self.COLOR_DANGER if has_critical else self.COLOR_WARNING
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 10)

        banner_w = min(680, int(w * 0.8))
        banner_h = 75
        bx1 = (w - banner_w) // 2
        by1 = 20
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        overlay = frame.copy()
        bg_color = (0, 0, 150) if has_critical else (0, 80, 180)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        border_color = (0, 0, 255) if has_critical else (0, 255, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, 2)

        if has_critical:
            title_text = "*** EMERGENCY STOP : WORKER FALL DOWN ***"
        else:
            title_text = "*** WARNING : WORKER DANGER BEHAVIOR ***"

        (tw1, th1), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx1 = bx1 + (banner_w - tw1) // 2
        cv2.putText(
            frame, title_text,
            (tx1, by1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if not has_critical else (255, 255, 255), 2, cv2.LINE_AA
        )

        details = []
        for t in active_threats[:2]:
            details.append(f"Worker#{t['tracker_id']} ({t['msg']})")
        
        detail_text = ", ".join(details)
        if len(active_threats) > 2:
            detail_text += f" (+{len(active_threats) - 2} more)"

        (tw2, th2), _ = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx2 = bx1 + (banner_w - tw2) // 2
        cv2.putText(
            frame, detail_text,
            (tx2, by1 + 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

        return frame

    def _draw_status_info(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        zones: list[DangerZone],
        active_threats: list[dict]
    ) -> np.ndarray:
        """우측 상단 모니터링 보드"""
        h, w = frame.shape[:2]
        
        intrusion_cnt = sum(1 for t in active_threats if t["threat_type"] == "intrusion")
        gesture_cnt = sum(1 for t in active_threats if t["threat_type"] in ["gesture_x", "gesture_wave"])
        fall_cnt = sum(1 for t in active_threats if t["threat_type"] == "fall_down")

        info_lines = [
            f"FPS: {self._current_fps:.1f}",
            f"Workers: {len(objects)}",
            f"Zones: {len(zones)}",
            f"Intrusion: {intrusion_cnt}",
            f"Gesture: {gesture_cnt}",
            f"Fall Down: {fall_cnt}"
        ]

        line_height = 22
        box_width = 180
        box_height = len(info_lines) * line_height + 15
        x_start = w - box_width - 10
        y_start = 10

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + box_width, y_start + box_height),
            (10, 10, 10), -1
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(info_lines):
            color = self.COLOR_FPS
            if "Intrusion" in line and intrusion_cnt > 0:
                color = self.COLOR_DANGER
            elif "Gesture" in line and gesture_cnt > 0:
                color = self.COLOR_WARNING
            elif "Fall Down" in line and fall_cnt > 0:
                color = self.COLOR_DANGER

            cv2.putText(
                frame, line,
                (x_start + 10, y_start + 20 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        return frame

    def _update_fps(self) -> None:
        """FPS 연산"""
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        if dt > 0:
            self._fps_history.append(1.0 / dt)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self._current_fps = sum(self._fps_history) / len(self._fps_history)

    @property
    def current_fps(self) -> float:
        return self._current_fps
