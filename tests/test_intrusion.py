"""
침입 감지 로직 단위 테스트
"""

import pytest
import numpy as np

from core.tracker import TrackedObject
from core.zone_manager import DangerZone
from processing.intrusion_detector import (
    IntrusionDetector,
    IntrusionState,
    IntrusionEvent,
)
from utils.geometry import (
    get_bottom_center,
    is_point_in_polygon,
    polygon_from_points,
    calculate_iou,
)


# =============================================================
# geometry 유틸리티 테스트
# =============================================================

class TestGeometry:
    """기하학 유틸리티 함수 테스트"""

    def test_get_bottom_center(self):
        """바운딩박스 하단 중심점 계산"""
        bbox = (100.0, 50.0, 200.0, 300.0)
        cx, cy = get_bottom_center(bbox)
        assert cx == 150  # (100 + 200) / 2
        assert cy == 300  # y2 = 하단

    def test_get_bottom_center_zero(self):
        """원점 바운딩박스"""
        bbox = (0.0, 0.0, 100.0, 100.0)
        cx, cy = get_bottom_center(bbox)
        assert cx == 50
        assert cy == 100

    def test_point_inside_polygon(self):
        """폴리곤 내부 점 판정"""
        square = polygon_from_points([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ])
        assert is_point_in_polygon((200, 200), square) is True

    def test_point_outside_polygon(self):
        """폴리곤 외부 점 판정"""
        square = polygon_from_points([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ])
        assert is_point_in_polygon((50, 50), square) is False

    def test_point_on_edge(self):
        """폴리곤 경계 점 판정 (내부로 판정)"""
        square = polygon_from_points([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ])
        assert is_point_in_polygon((100, 200), square) is True

    def test_triangle_polygon(self):
        """삼각형 폴리곤 판정"""
        triangle = polygon_from_points([
            [200, 100], [100, 300], [300, 300]
        ])
        assert is_point_in_polygon((200, 250), triangle) is True
        assert is_point_in_polygon((50, 50), triangle) is False

    def test_calculate_iou_overlap(self):
        """IoU - 겹치는 경우"""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = calculate_iou(box1, box2)
        # 교집합: 50*50=2500, 합집합: 10000+10000-2500=17500
        assert abs(iou - 2500 / 17500) < 0.01

    def test_calculate_iou_no_overlap(self):
        """IoU - 겹치지 않는 경우"""
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert calculate_iou(box1, box2) == 0.0

    def test_calculate_iou_identical(self):
        """IoU - 동일 박스"""
        box1 = (0, 0, 100, 100)
        assert calculate_iou(box1, box1) == 1.0


# =============================================================
# IntrusionDetector 테스트
# =============================================================

def _make_tracked_object(
    tracker_id: int,
    x1: float, y1: float, x2: float, y2: float,
    confidence: float = 0.9
) -> TrackedObject:
    """테스트용 TrackedObject 생성 헬퍼"""
    return TrackedObject(
        tracker_id=tracker_id,
        bbox=(x1, y1, x2, y2),
        confidence=confidence,
        class_id=0,
        class_name="person"
    )


def _make_danger_zone(
    zone_id: str,
    points: list[list[float]],
    name: str = "test-zone",
    severity: str = "high"
) -> DangerZone:
    """테스트용 DangerZone 생성 헬퍼"""
    return DangerZone(
        zone_id=zone_id,
        name=name,
        polygon=polygon_from_points(points),
        severity=severity
    )


class TestIntrusionDetector:
    """침입 감지 로직 테스트"""

    def setup_method(self):
        """각 테스트 전 초기화"""
        self.detector = IntrusionDetector(cooldown_seconds=1.0)
        # 프레임 중앙 사각형 위험구역 (100~300, 100~300)
        self.zone = _make_danger_zone(
            "zone-1",
            [[100, 100], [300, 100], [300, 300], [100, 300]],
            "테스트 위험구역"
        )

    def test_no_intrusion_outside(self):
        """위험구역 밖의 작업자 - 이벤트 없음"""
        # 바운딩박스 하단 중심: (50, 80) → 위험구역 밖
        worker = _make_tracked_object(1, 20, 30, 80, 80)

        events = self.detector.check_intrusions([worker], [self.zone])
        assert len(events) == 0

    def test_intrusion_enters_after_threshold(self):
        """
        위험구역 진입 - 연속 프레임 임계값(3) 이후 ENTERED 이벤트 발생
        """
        # 바운딩박스 하단 중심: (200, 250) → 위험구역 내부
        worker = _make_tracked_object(1, 150, 100, 250, 250)

        # 3프레임 연속 내부에 있어야 진입 확정
        events1 = self.detector.check_intrusions([worker], [self.zone])
        events2 = self.detector.check_intrusions([worker], [self.zone])
        events3 = self.detector.check_intrusions([worker], [self.zone])

        # 3번째에서 ENTERED 이벤트 발생해야 함
        assert len(events3) == 1
        assert events3[0].event_type == IntrusionState.ENTERED
        assert events3[0].tracker_id == 1
        assert events3[0].zone_id == "zone-1"

    def test_exit_after_threshold(self):
        """위험구역 이탈 - 연속 프레임 임계값(5) 이후 EXITED 이벤트 발생"""
        # 먼저 진입
        worker_inside = _make_tracked_object(1, 150, 100, 250, 250)
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            self.detector.check_intrusions([worker_inside], [self.zone])

        # 이제 밖으로 나감
        worker_outside = _make_tracked_object(1, 20, 30, 80, 80)
        exit_events = []
        for _ in range(IntrusionDetector.EXIT_THRESHOLD_FRAMES + 1):
            events = self.detector.check_intrusions([worker_outside], [self.zone])
            exit_events.extend(events)

        # EXITED 이벤트가 있어야 함
        exited = [e for e in exit_events if e.event_type == IntrusionState.EXITED]
        assert len(exited) == 1
        assert exited[0].tracker_id == 1

    def test_multiple_workers(self):
        """다수 작업자 - 각각 독립적으로 추적"""
        worker1_inside = _make_tracked_object(1, 150, 100, 250, 250)
        worker2_outside = _make_tracked_object(2, 350, 100, 450, 250)

        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events = self.detector.check_intrusions(
                [worker1_inside, worker2_outside], [self.zone]
            )

        # worker1만 침입 이벤트
        entered = [e for e in events if e.event_type == IntrusionState.ENTERED]
        assert len(entered) == 1
        assert entered[0].tracker_id == 1

    def test_multiple_zones(self):
        """다수 위험구역 동시 감지"""
        zone2 = _make_danger_zone(
            "zone-2",
            [[400, 400], [600, 400], [600, 600], [400, 600]],
            "위험구역 2"
        )

        # 두 구역 모두에 진입 (발 위치로 판정)
        # worker1 → zone1 내부, worker2 → zone2 내부
        worker1 = _make_tracked_object(1, 150, 100, 250, 250)
        worker2 = _make_tracked_object(2, 450, 350, 550, 550)

        all_events = []
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events = self.detector.check_intrusions(
                [worker1, worker2], [self.zone, zone2]
            )
            all_events.extend(events)

        entered = [e for e in all_events if e.event_type == IntrusionState.ENTERED]
        assert len(entered) == 2
        zone_ids = {e.zone_id for e in entered}
        assert "zone-1" in zone_ids
        assert "zone-2" in zone_ids

    def test_cooldown_prevents_duplicate(self):
        """쿨다운 - 같은 이벤트 반복 전송 방지"""
        worker = _make_tracked_object(1, 150, 100, 250, 250)

        # 진입 확정
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            self.detector.check_intrusions([worker], [self.zone])

        # 계속 내부에 있지만 쿨다운(1초) 안에는 STAYING 이벤트 없음
        events = self.detector.check_intrusions([worker], [self.zone])
        staying = [e for e in events if e.event_type == IntrusionState.STAYING]
        assert len(staying) == 0

    def test_active_intrusions(self):
        """현재 활성 침입 상태 조회"""
        worker = _make_tracked_object(1, 150, 100, 250, 250)

        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            self.detector.check_intrusions([worker], [self.zone])

        active = self.detector.get_active_intrusions()
        assert (1, "zone-1") in active

    def test_reset_clears_state(self):
        """리셋 시 모든 상태 초기화"""
        worker = _make_tracked_object(1, 150, 100, 250, 250)

        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            self.detector.check_intrusions([worker], [self.zone])

        self.detector.reset()
        active = self.detector.get_active_intrusions()
        assert len(active) == 0
