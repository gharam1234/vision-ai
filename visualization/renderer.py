"""
시각화 렌더러 모듈
감지 결과, 위험구역, 침입 경고를 프레임에 오버레이
"""

import cv2
import numpy as np
import time
from typing import Optional
from loguru import logger

from core.tracker import TrackedObject
from core.zone_manager import DangerZone
from processing.intrusion_detector import IntrusionEvent, IntrusionState
from utils.geometry import get_bottom_center


class FrameRenderer:
    """프레임 시각화 렌더러"""

    # 색상 상수 (BGR)
    COLOR_SAFE = (0, 200, 0)          # 녹색 - 안전한 작업자
    COLOR_DANGER = (0, 0, 255)        # 빨강 - 위험구역 내 작업자
    COLOR_WARNING_BG = (0, 0, 180)    # 경고 배경
    COLOR_TEXT = (255, 255, 255)      # 흰색 텍스트
    COLOR_FPS = (0, 255, 255)         # 노란색 FPS
    COLOR_ZONE_FILL_ALPHA = 0.25      # 위험구역 채우기 투명도

    def __init__(self):
        self._fps_history: list[float] = []
        self._last_frame_time = time.time()
        self._current_fps: float = 0.0
        self._warning_flash_start: float = 0.0
        self._warning_active: bool = False

    def render(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
        danger_zones: list[DangerZone],
        intrusion_events: list[IntrusionEvent],
        active_intrusions: Optional[dict] = None
    ) -> np.ndarray:
        """
        프레임에 모든 시각화 요소를 오버레이

        Args:
            frame: 원본 BGR 프레임
            tracked_objects: 추적된 작업자 리스트
            danger_zones: 위험구역 리스트
            intrusion_events: 이번 프레임의 침입 이벤트
            active_intrusions: 현재 활성 침입 상태

        Returns:
            시각화가 적용된 프레임
        """
        display = frame.copy()

        # 1. 위험구역 표시
        display = self._draw_zones(display, danger_zones)

        # 2. 추적 작업자 표시
        display = self._draw_tracked_objects(
            display, tracked_objects, active_intrusions
        )

        # 3. 침입 경고 표시
        if active_intrusions:
            display = self._draw_intrusion_warnings(
                display, active_intrusions, danger_zones
            )

        # 4. 상태 정보 (FPS, 감지 수 등)
        display = self._draw_status_info(
            display, tracked_objects, danger_zones, active_intrusions
        )

        # FPS 업데이트
        self._update_fps()

        return display

    def _draw_zones(
        self, frame: np.ndarray, zones: list[DangerZone]
    ) -> np.ndarray:
        """위험구역 폴리곤 표시"""
        overlay = frame.copy()

        for zone in zones:
            pts = zone.polygon.reshape((-1, 1, 2)).astype(np.int32)

            # 반투명 채우기
            cv2.fillPoly(overlay, [pts.reshape(-1, 2)], zone.color)

            # 테두리
            cv2.polylines(frame, [pts], True, zone.color, 2)

            # 구역 이름 표시
            centroid = zone.polygon.mean(axis=0).astype(int)
            label = f"{zone.name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # 배경 박스
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

        # 반투명 합성
        cv2.addWeighted(overlay, self.COLOR_ZONE_FILL_ALPHA, frame,
                        1 - self.COLOR_ZONE_FILL_ALPHA, 0, frame)
        return frame

    def _draw_tracked_objects(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        active_intrusions: Optional[dict] = None
    ) -> np.ndarray:
        """추적 작업자 바운딩박스 및 ID 표시"""
        intrusion_tracker_ids = set()
        if active_intrusions:
            intrusion_tracker_ids = {key[0] for key in active_intrusions}

        for obj in objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            is_intruding = obj.tracker_id in intrusion_tracker_ids

            # 색상 결정: 침입 중이면 빨강, 아니면 녹색
            color = self.COLOR_DANGER if is_intruding else self.COLOR_SAFE

            # 바운딩박스
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 침입자일 때 네 모서리(Corner brackets) 강조
            if is_intruding:
                length = min(15, int((x2 - x1) * 0.2))  # 모서리 선 길이
                # 좌상단 모서리
                cv2.line(frame, (x1, y1), (x1 + length, y1), color, 4)
                cv2.line(frame, (x1, y1), (x1, y1 + length), color, 4)
                # 우상단 모서리
                cv2.line(frame, (x2, y1), (x2 - length, y1), color, 4)
                cv2.line(frame, (x2, y1), (x2, y1 + length), color, 4)
                # 좌하단 모서리
                cv2.line(frame, (x1, y2), (x1 + length, y2), color, 4)
                cv2.line(frame, (x1, y2), (x1, y2 - length), color, 4)
                # 우하단 모서리
                cv2.line(frame, (x2, y2), (x2 - length, y2), color, 4)
                cv2.line(frame, (x2, y2), (x2, y2 - length), color, 4)

            # 하단 중심점 (발 위치)
            foot = get_bottom_center(obj.bbox)
            cv2.circle(frame, foot, 5, color, -1)

            # 라벨: 추적 ID + 신뢰도
            label = f"ID:{obj.tracker_id} {obj.confidence:.0%}"
            if is_intruding:
                label = f"!! {label} !!"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # 라벨 배경
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw + 10, y1),
                color, -1
            )
            cv2.putText(
                frame, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2
            )

        return frame

    def _draw_intrusion_warnings(
        self,
        frame: np.ndarray,
        active_intrusions: dict,
        danger_zones: list[DangerZone]
    ) -> np.ndarray:
        """침입 경고 표시"""
        h, w = frame.shape[:2]
        now = time.time()

        # 깜빡임 주기 계산 (약 0.3초 주기로 켜짐/꺼짐 반복)
        is_flash_on = int(now * 3.33) % 2 == 0

        if active_intrusions:
            # 1. 화면 테두리 깜빡임 효과 (ON 타이밍일 때)
            if is_flash_on:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

            # 2. 상단 중앙 대형 경고 패널 그리기
            banner_w = min(680, int(w * 0.8))
            banner_h = 75
            bx1 = (w - banner_w) // 2
            by1 = 20
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h

            # 반투명 붉은색 배경 합성
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 180), -1) # 진한 빨간색
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # 노란색 패널 테두리
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

            # 대표적인 경고 정보 추출 (첫 번째 활성 경고)
            (tracker_id, zone_id), state = list(active_intrusions.items())[0]
            
            # danger_zones에서 구역 이름 검색
            zone_name = zone_id
            for zone in danger_zones:
                if zone.zone_id == zone_id:
                    zone_name = zone.name
                    break

            # 텍스트 라인 1: 대형 경고 타이틀
            title_text = "WARNING: EMERGENCY INTRUSION DETECTED"
            (tw1, th1), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx1 = bx1 + (banner_w - tw1) // 2
            cv2.putText(
                frame, title_text,
                (tx1, by1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

            # 텍스트 라인 2: 세부 침입 내용
            duration = now - state.enter_time
            detail_text = (
                f"Worker #{tracker_id} in "
                f"[{zone_name}]"
            )
            if duration > 0:
                detail_text += f" for {duration:.0f}s"
                
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
        active_intrusions: Optional[dict] = None
    ) -> np.ndarray:
        """상태 정보 표시 (우측 상단)"""
        h, w = frame.shape[:2]
        intrusion_count = len(active_intrusions) if active_intrusions else 0

        info_lines = [
            f"FPS: {self._current_fps:.1f}",
            f"Workers: {len(objects)}",
            f"Zones: {len(zones)}",
            f"Intrusions: {intrusion_count}",
        ]

        # 배경 박스
        line_height = 25
        box_width = 180
        box_height = len(info_lines) * line_height + 20
        x_start = w - box_width - 10
        y_start = 10

        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + box_width, y_start + box_height),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(info_lines):
            color = self.COLOR_FPS
            if "Intrusions" in line and intrusion_count > 0:
                color = self.COLOR_DANGER

            cv2.putText(
                frame, line,
                (x_start + 10, y_start + 25 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1
            )

        return frame

    def _update_fps(self) -> None:
        """FPS 계산 (이동 평균)"""
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        if dt > 0:
            self._fps_history.append(1.0 / dt)
            # 최근 30프레임 이동 평균
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self._current_fps = sum(self._fps_history) / len(self._fps_history)

    @property
    def current_fps(self) -> float:
        return self._current_fps
