"""
프레임 처리 파이프라인 모듈
감지 → 추적 → 침입 판정 → 이벤트 전송 → 시각화 전체 플로우 관리
"""

import time
import numpy as np
from typing import Optional
from loguru import logger

from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.zone_manager import DangerZoneManager
from processing.intrusion_detector import IntrusionDetector, IntrusionEvent
from visualization.renderer import FrameRenderer
from communication.event_sender import EventSender


class FrameProcessor:
    """
    프레임 처리 파이프라인
    하나의 프레임에 대해 전체 감지-추적-침입판정-시각화 파이프라인을 실행
    """

    def __init__(
        self,
        detector: PersonDetector,
        tracker: PersonTracker,
        zone_manager: DangerZoneManager,
        intrusion_detector: IntrusionDetector,
        renderer: FrameRenderer,
        event_sender: Optional[EventSender] = None
    ):
        """
        Args:
            detector: 사람 감지기
            tracker: 사람 추적기
            zone_manager: 위험구역 관리자
            intrusion_detector: 침입 감지기
            renderer: 시각화 렌더러
            event_sender: 이벤트 전송기 (없으면 로그만 출력)
        """
        self._detector = detector
        self._tracker = tracker
        self._zone_manager = zone_manager
        self._intrusion_detector = intrusion_detector
        self._renderer = renderer
        self._event_sender = event_sender

        # 성능 측정
        self._frame_count = 0
        self._total_detect_time = 0.0
        self._total_track_time = 0.0
        self._total_intrusion_time = 0.0
        self._total_render_time = 0.0

        logger.info("FrameProcessor 초기화 완료")

    def process(
        self,
        frame: np.ndarray,
        show_display: bool = True
    ) -> tuple[np.ndarray, list[IntrusionEvent]]:
        """
        단일 프레임 처리 파이프라인

        Args:
            frame: BGR 이미지
            show_display: 시각화 적용 여부

        Returns:
            (시각화 프레임, 침입 이벤트 리스트)
        """
        self._frame_count += 1

        # 1단계: 객체 감지 (YOLO)
        t0 = time.time()
        yolo_result = self._detector.detect_raw(frame)
        detect_time = time.time() - t0
        self._total_detect_time += detect_time

        # 2단계: 객체 추적 (ByteTrack)
        t1 = time.time()
        tracked_objects = self._tracker.update(yolo_result)
        track_time = time.time() - t1
        self._total_track_time += track_time

        # 3단계: 위험구역 가져오기
        danger_zones = self._zone_manager.get_zones()

        # 4단계: 침입 감지
        t2 = time.time()
        intrusion_events = self._intrusion_detector.check_intrusions(
            tracked_objects, danger_zones
        )
        intrusion_time = time.time() - t2
        self._total_intrusion_time += intrusion_time

        # 5단계: 이벤트 전송
        if intrusion_events and self._event_sender:
            for event in intrusion_events:
                self._event_sender.send_event(event, frame)

        # 6단계: 시각화
        display_frame = frame
        if show_display:
            t3 = time.time()
            active_intrusions = self._intrusion_detector.get_active_intrusions()
            display_frame = self._renderer.render(
                frame,
                tracked_objects,
                danger_zones,
                intrusion_events,
                active_intrusions
            )
            render_time = time.time() - t3
            self._total_render_time += render_time

        # 주기적 성능 로그 (100프레임마다)
        if self._frame_count % 100 == 0:
            self._log_performance()

        return display_frame, intrusion_events

    def _log_performance(self) -> None:
        """성능 통계 로그 출력"""
        n = self._frame_count
        avg_detect = (self._total_detect_time / n) * 1000
        avg_track = (self._total_track_time / n) * 1000
        avg_intrusion = (self._total_intrusion_time / n) * 1000
        avg_render = (self._total_render_time / n) * 1000
        avg_total = avg_detect + avg_track + avg_intrusion + avg_render

        logger.info(
            f"📊 성능 통계 ({n}프레임) | "
            f"감지: {avg_detect:.1f}ms | "
            f"추적: {avg_track:.1f}ms | "
            f"침입: {avg_intrusion:.1f}ms | "
            f"렌더: {avg_render:.1f}ms | "
            f"총: {avg_total:.1f}ms ({1000/avg_total:.1f} FPS)"
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def performance_stats(self) -> dict:
        """성능 통계 반환"""
        n = max(self._frame_count, 1)
        return {
            "frame_count": n,
            "avg_detect_ms": (self._total_detect_time / n) * 1000,
            "avg_track_ms": (self._total_track_time / n) * 1000,
            "avg_intrusion_ms": (self._total_intrusion_time / n) * 1000,
            "avg_render_ms": (self._total_render_time / n) * 1000,
        }
