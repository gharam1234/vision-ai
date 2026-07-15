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
    get_multi_bottom_points,
    check_polygon_overlap_mask,
    check_segment_overlap_mask,
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

    def test_get_multi_bottom_points(self):
        """다중 하단 포인트 계산"""
        bbox = (100.0, 50.0, 200.0, 300.0)
        pts = get_multi_bottom_points(bbox)
        assert len(pts) == 3
        # pt_left (100 + 100*0.2 = 120, 300)
        assert pts[0] == (120, 300)
        # pt_center (150, 300)
        assert pts[1] == (150, 300)
        # pt_right (100 + 100*0.8 = 180, 300)
        assert pts[2] == (180, 300)

    def test_check_polygon_overlap_mask_fully_inside(self):
        """다각형 겹침 판정 - 완벽 내부"""
        square = polygon_from_points([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ])
        bbox = (150, 150, 250, 250)  # 완전히 내부
        assert check_polygon_overlap_mask(bbox, square, 640, 480, 0.2) is True

    def test_check_polygon_overlap_mask_partial(self):
        """다각형 겹침 판정 - 일부 걸침"""
        square = polygon_from_points([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ])
        # 하단 발 부분(y2=120)이 위험구역 상단(y1=100)에 걸쳐 있음
        # y_start = 120 - 100*0.2 = 100. 발 영역 y: 100~120
        # 위험구역 y: 100~300. 겹침 면적 발생!
        bbox = (150, 20, 250, 120)
        assert check_polygon_overlap_mask(bbox, square, 640, 480, 0.2) is True

    def test_check_polygon_overlap_mask_outside(self):
        """다각형 겹침 판정 - 완전히 외부"""
        square = polygon_from_points([
            [100, 100], [300, 100], [300, 300], [100, 300]
        ])
        # y2=80, y_start = 80 - 100*0.2 = 60. 발 영역 y: 60~80. 위험구역 상단(100)보다 높음
        bbox = (150, -20, 250, 80)
        assert check_polygon_overlap_mask(bbox, square, 640, 480, 0.2) is False

    def test_check_segment_overlap_mask(self):
        """세그멘테이션 실루엣 마스크 겹침 판정"""
        # 100x100 마스크 이미지
        mask = np.zeros((100, 100), dtype=np.uint8)
        # y: 40~80, x: 40~60 범위의 실루엣 모사 (세로 40픽셀)
        mask[40:80, 40:60] = 1
        # 하부 20% 영역 시작 y: 80 - 40 * 0.2 = 72. 즉 y: 72~80, x: 40~60이 발 영역

        # 1. 겹치는 다각형 (y: 75~90, x: 45~70) -> 겹침
        polygon_overlap = polygon_from_points([
            [45, 75], [70, 75], [70, 90], [45, 90]
        ])
        assert check_segment_overlap_mask(mask, polygon_overlap, 0.2) is True

        # 2. 겹치지 않는 다각형 (y: 20~30, x: 40~60) -> 실루엣에는 속하지만 상반신 영역(y < 72)이므로 발 영역 겹침 미감지
        polygon_no_overlap = polygon_from_points([
            [40, 20], [60, 20], [60, 30], [40, 30]
        ])
        assert check_segment_overlap_mask(mask, polygon_no_overlap, 0.2) is False


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

    def test_intrusion_method_multi_point(self):
        """다중 점(multi-point) 방식 침입 판정"""
        detector = IntrusionDetector(cooldown_seconds=1.0, method="multi-point")
        # zone: [100, 100] ~ [300, 300]
        # x1=80, x2=110, y2=200 -> w=30.
        # pt_left (80+6=86, 200) -> 외부
        # pt_center (95, 200) -> 외부
        # pt_right (80+24=104, 200) -> 내부 (104 >= 100)
        # 즉, 하단 정중앙(95, 200)은 바깥이지만 우측 하단(104, 200)이 구역 내부에 있음
        worker = _make_tracked_object(1, 80, 100, 110, 200)

        events = []
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events = detector.check_intrusions([worker], [self.zone])

        assert len(events) == 1
        assert events[0].event_type == IntrusionState.ENTERED

    def test_intrusion_method_overlap(self):
        """하부 영역 겹침(overlap) 방식 침입 판정"""
        detector = IntrusionDetector(
            cooldown_seconds=1.0,
            method="overlap",
            overlap_ratio=0.2,
            frame_width=640,
            frame_height=480
        )
        # zone: [100, 100] ~ [300, 300]
        # x1=50, x2=110, y2=200 -> h=100. y_start=180. y:180~200, x:50~110
        # 발 중앙점: (80, 200) -> 위험구역 외부!
        # 겹치는 영역: x[100~110], y[180~200] -> 면적 존재!
        worker = _make_tracked_object(1, 50, 100, 110, 200)

        events = []
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events = detector.check_intrusions([worker], [self.zone])

        assert len(events) == 1
        assert events[0].event_type == IntrusionState.ENTERED

    def test_intrusion_method_segment(self):
        """세그멘테이션(segment) 방식 침입 판정"""
        detector = IntrusionDetector(
            cooldown_seconds=1.0,
            method="segment",
            overlap_ratio=0.2,
            frame_width=640,
            frame_height=480
        )
        # zone: [100, 100] ~ [300, 300]
        worker = _make_tracked_object(1, 50, 100, 110, 200)

        # 640x480 마스크 이미지
        mask = np.zeros((480, 640), dtype=np.uint8)
        # y: 100~200, x: 50~110 범위의 실루엣 모사 (y:180~200 이 발 영역)
        mask[100:200, 50:110] = 1
        worker.mask = mask

        events = []
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events = detector.check_intrusions([worker], [self.zone])

        assert len(events) == 1
        assert events[0].event_type == IntrusionState.ENTERED

    def test_intrusion_method_pose_hybrid(self):
        """포즈-바박 하이브리드(pose-hybrid) 방식 침입 판정"""
        detector = IntrusionDetector(
            cooldown_seconds=1.0,
            method="pose-hybrid",
            overlap_ratio=0.2,
            frame_width=640,
            frame_height=480,
            pose_conf_threshold=0.5
        )
        # zone: [100, 100] ~ [300, 300]

        # 케이스 1: 포즈 감지 성공, 발목 중 하나가 구역 내부에 위치
        worker_pose = _make_tracked_object(1, 50, 100, 110, 200)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # 16번 index (오른쪽 발목) 가 (150, 200)에 있고 신뢰도가 0.8인 경우 -> 구역 내
        keypoints[16] = [150.0, 200.0, 0.8]
        # 15번 index (왼쪽 발목) 는 외부 (80, 200), 신뢰도 0.8
        keypoints[15] = [80.0, 200.0, 0.8]
        worker_pose.keypoints = keypoints

        events_pose = []
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events_pose = detector.check_intrusions([worker_pose], [self.zone])

        assert len(events_pose) == 1
        assert events_pose[0].event_type == IntrusionState.ENTERED

        # 케이스 2: 포즈 감지 실패 (신뢰도 낮음) -> 바박 겹침으로 폴백 작동
        # x1=50, x2=110, y2=200 -> 발 중앙점은 80(외부)이지만 x:100~110 영역이 겹침
        worker_fallback = _make_tracked_object(2, 50, 100, 110, 200)
        # keypoints 가 존재하지만 신뢰도가 0.1로 임계치(0.5) 미달
        bad_keypoints = np.zeros((17, 3), dtype=np.float32)
        bad_keypoints[16] = [80.0, 200.0, 0.1]
        bad_keypoints[15] = [80.0, 200.0, 0.1]
        worker_fallback.keypoints = bad_keypoints

        events_fallback = []
        detector.reset() # 상태 초기화
        for _ in range(IntrusionDetector.ENTER_THRESHOLD_FRAMES):
            events_fallback = detector.check_intrusions([worker_fallback], [self.zone])

        assert len(events_fallback) == 1
        assert events_fallback[0].event_type == IntrusionState.ENTERED
