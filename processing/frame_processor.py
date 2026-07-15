"""
프레임 처리 파이프라인 모듈
감지(YOLO) → 추적(ByteTrack) → 랜드마크 추출(MediaPipe) → 침입/제스처 판정 → 이벤트 전송 및 렌더링
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
from core.mediapipe_detector import MediaPipeDetector


class FrameProcessor:
    """
    프레임 처리 파이프라인
    단일 프레임에 대해 전체 YOLO-ByteTrack-MediaPipe-감지엔진-렌더러를 순차 실행
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
            detector: YOLO 사람 감지기
            tracker: ByteTrack 사람 추적기
            zone_manager: 위험구역 관리자
            intrusion_detector: 침입 및 제스처 감지기
            renderer: 시각화 렌더러
            event_sender: 이벤트 전송기
        """
        self._detector = detector
        self._tracker = tracker
        self._zone_manager = zone_manager
        self._intrusion_detector = intrusion_detector
        self._renderer = renderer
        self._event_sender = event_sender

        # MediaPipe 감지기 연동
        self._mp_detector = MediaPipeDetector()

        # 성능 측정
        self._frame_count = 0
        self._total_detect_time = 0.0
        self._total_track_time = 0.0
        self._total_mediapipe_time = 0.0
        self._total_intrusion_time = 0.0
        self._total_render_time = 0.0

        logger.info("FrameProcessor 초기화 완료 (MediaPipe 하이브리드 연동)")

    def process(
        self,
        frame: np.ndarray,
        show_display: bool = True
    ) -> tuple[np.ndarray, list[IntrusionEvent]]:
        """
        단일 프레임 처리 파이프라인
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

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

        # 3단계: 작업자 ROI 크롭 및 MediaPipe Pose/Hands 검출
        t_mp = time.time()
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            
            # 가로, 세로 45% 마진(Padding) 부여하여 손목/팔꿈치가 잘리는 것을 최소화
            pad_w = int((x2 - x1) * 0.45)
            pad_h = int((y2 - y1) * 0.45)
            
            cx1 = max(0, int(x1 - pad_w))
            cy1 = max(0, int(y1 - pad_h))
            cx2 = min(w, int(x2 + pad_w))
            cy2 = min(h, int(y2 + pad_h))
            
            crop_img = frame[cy1:cy2, cx1:cx2]
            if crop_img.size > 0:
                # 33개 Pose 관절 랜드마크 추출 및 전역 좌표 매핑
                obj.keypoints = self._mp_detector.detect_pose(crop_img, (cx1, cy1), obj.bbox)
                # 각 손 21개 랜드마크 추출 및 전역 좌표 매핑
                obj.hand_landmarks = self._mp_detector.detect_hands(crop_img, (cx1, cy1))
                
        self._total_mediapipe_time += (time.time() - t_mp)

        # 4단계: 위험구역 목록 조회
        danger_zones = self._zone_manager.get_zones()

        # 5단계: 침입 및 제스처/쓰러짐 감지
        t2 = time.time()
        intrusion_events = self._intrusion_detector.check_intrusions(
            tracked_objects, danger_zones
        )
        intrusion_time = time.time() - t2
        self._total_intrusion_time += intrusion_time

        # 6단계: 알림 이벤트 백엔드 전송
        if intrusion_events and self._event_sender:
            for event in intrusion_events:
                self._event_sender.send_event(event, frame)

        # 7단계: 화면 시각화 렌더링
        display_frame = frame
        if show_display:
            t3 = time.time()
            active_threats = self._intrusion_detector.get_active_threats()
            display_frame = self._renderer.render(
                frame,
                tracked_objects,
                danger_zones,
                intrusion_events,
                active_threats
            )
            render_time = time.time() - t3
            self._total_render_time += render_time

        # 100프레임마다 성능 분석 지표 출력
        if self._frame_count % 100 == 0:
            self._log_performance()

        return display_frame, intrusion_events

    def _log_performance(self) -> None:
        """성능 통계 로그"""
        n = self._frame_count
        avg_detect = (self._total_detect_time / n) * 1000
        avg_track = (self._total_track_time / n) * 1000
        avg_mp = (self._total_mediapipe_time / n) * 1000
        avg_intrusion = (self._total_intrusion_time / n) * 1000
        avg_render = (self._total_render_time / n) * 1000
        avg_total = avg_detect + avg_track + avg_mp + avg_intrusion + avg_render

        logger.info(
            f"📊 성능 통계 ({n}프레임) | "
            f"YOLO감지: {avg_detect:.1f}ms | "
            f"추적: {avg_track:.1f}ms | "
            f"MediaPipe: {avg_mp:.1f}ms | "
            f"위협연산: {avg_intrusion:.1f}ms | "
            f"렌더: {avg_render:.1f}ms | "
            f"평균: {avg_total:.1f}ms ({1000/avg_total:.1f} FPS)"
        )

    def __del__(self):
        """소멸 시 리소스 해제"""
        if hasattr(self, '_mp_detector'):
            self._mp_detector.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def performance_stats(self) -> dict:
        n = max(self._frame_count, 1)
        return {
            "frame_count": self._frame_count,
            "avg_detect_ms": (self._total_detect_time / n) * 1000,
            "avg_track_ms": (self._total_track_time / n) * 1000,
            "avg_mediapipe_ms": (self._total_mediapipe_time / n) * 1000,
            "avg_intrusion_ms": (self._total_intrusion_time / n) * 1000,
            "avg_render_ms": (self._total_render_time / n) * 1000
        }
