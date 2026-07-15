"""
침입 및 위험 행동 감지 모듈
추적된 작업자와 위험구역을 비교하고, MediaPipe Pose/Hands 좌표를 통해 긴급 상황(제스처, 쓰러짐)을 감지하여 이벤트를 생성
"""

import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
from loguru import logger

from core.tracker import TrackedObject
from core.zone_manager import DangerZone
from utils.geometry import (
    get_bottom_center,
    is_point_in_polygon,
    get_multi_bottom_points,
    check_polygon_overlap_mask,
    check_segment_overlap_mask,
)


class IntrusionState(str, Enum):
    """위험 상태"""
    OUTSIDE = "outside"     # 안전 영역
    ENTERED = "entered"     # 방금 감지됨
    STAYING = "staying"     # 감지 상태 유지 중
    EXITED = "exited"       # 방금 해제됨


@dataclass
class IntrusionEvent:
    """위험 및 침입 이벤트 데이터"""
    event_type: IntrusionState         # 이벤트 타입
    tracker_id: int                    # 작업자 추적 ID
    zone_id: str                       # 위험구역 ID (글로벌 위협의 경우 "global")
    zone_name: str                     # 위험구역 이름
    zone_severity: str                 # 위험 등급 ("low", "medium", "high", "critical")
    timestamp: float                   # 이벤트 발생 시각
    position: tuple[int, int]          # 작업자 위치 (하단 중심점)
    bbox: tuple[float, float, float, float]  # 바운딩박스
    confidence: float                  # 감지 신뢰도
    duration: float = 0.0             # 상태 지속 시간 (초)
    threat_type: str = "intrusion"     # 위협 유형 ("intrusion", "gesture_x", "gesture_wave", "fall_down")
    keypoints: Optional[list] = None   # 직렬화용 포즈 키포인트 (33, 3) 리스트
    hand_landmarks: Optional[list] = None # 직렬화용 손 랜드마크 리스트


@dataclass
class _TrackerZoneState:
    """개별 추적자의 위험/침입 상태 추적"""
    is_inside: bool = False
    enter_time: float = 0.0
    last_event_time: float = 0.0
    consecutive_inside_frames: int = 0
    consecutive_outside_frames: int = 0

    # --- 행동/자세 감지용 상태 필드 ---
    is_gesture_x: bool = False
    gesture_x_enter_time: float = 0.0
    gesture_x_last_event_time: float = 0.0
    consecutive_gesture_x_inside_frames: int = 0
    consecutive_gesture_x_outside_frames: int = 0

    is_gesture_wave: bool = False
    gesture_wave_enter_time: float = 0.0
    gesture_wave_last_event_time: float = 0.0
    consecutive_gesture_wave_inside_frames: int = 0
    consecutive_gesture_wave_outside_frames: int = 0

    is_fall_down: bool = False
    fall_down_enter_time: float = 0.0
    fall_down_last_event_time: float = 0.0
    consecutive_fall_down_inside_frames: int = 0
    consecutive_fall_down_outside_frames: int = 0

    # 변곡점 판별용 X좌표 기록 (최대 30프레임)
    wrist_x_history: list[float] = field(default_factory=list)
    wrist_x_histories: dict[str, list[float]] = field(default_factory=lambda: {"Left": [], "Right": []})


class IntrusionDetector:
    """침입 및 위험 행동 감지기 (YOLO + MediaPipe 하이브리드 버전)"""

    # 하위 호환성 및 테스트 케이스 참조용 클래스 상수
    ENTER_THRESHOLD_FRAMES = 3
    EXIT_THRESHOLD_FRAMES = 5

    # MediaPipe Pose 관절 연결 구조 정의 (i, j)
    SKELETON_CONNECTIONS = [
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

    # MediaPipe Hands 손가락 연결 구조 정의 (i, j)
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

    def _is_skeleton_intersecting_zone(self, A: Tuple[int, int], B: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """
        두 관절 A, B를 잇는 뼈대 선분이 다각형 위험구역 경계와 교차하거나,
        혹은 두 관절 중 하나가 다각형 내부에 포함되어 있는지 검사
        """
        # 1. 두 선분의 끝점 중 하나라도 다각형 내부에 포함되어 있으면 참
        if is_point_in_polygon(A, np.array(polygon)) or is_point_in_polygon(B, np.array(polygon)):
            return True
            
        # 2. 다각형의 각 변(Edge)과 뼈대 선분이 교차하는지 검사
        n = len(polygon)
        for i in range(n):
            C = polygon[i]
            D = polygon[(i + 1) % n]
            if self._is_segment_intersect(A, B, C, D):
                return True
        return False

    def _is_segment_intersect(self, A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> bool:
        """두 선분 AB와 CD의 교차 여부를 CCW 판별을 통해 확인"""
        def ccw(p1, p2, p3):
            val = (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if val > 0:
                return 1
            elif val < 0:
                return -1
            return 0
        return ccw(A, C, D) * ccw(B, C, D) <= 0 and ccw(A, B, C) * ccw(A, B, D) <= 0

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        method: str = "point",
        overlap_ratio: float = 0.2,
        frame_width: int = 1280,
        frame_height: int = 720,
        pose_conf_threshold: float = 0.4,
        waving_amplitude_ratio: float = 0.15,
        waving_direction_changes: int = 2,
        enter_threshold_frames: int = 3,
        exit_threshold_frames: int = 5
    ):
        self._cooldown_seconds = cooldown_seconds
        self._method = method
        self._overlap_ratio = overlap_ratio
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._pose_conf_threshold = pose_conf_threshold
        
        # 감도 파라미터 연동
        self.waving_amplitude_ratio = waving_amplitude_ratio
        self.waving_direction_changes = waving_direction_changes
        self.enter_threshold_frames = enter_threshold_frames
        self.exit_threshold_frames = exit_threshold_frames

        # {(tracker_id, zone_id): _TrackerZoneState}
        self._states: dict[tuple[int, str], _TrackerZoneState] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0

        logger.info(
            f"IntrusionDetector 초기화 완료 (하이브리드) "
            f"(쿨다운: {cooldown_seconds}초, 방식: {method}, overlap_ratio: {overlap_ratio}, pose_conf: {pose_conf_threshold})"
        )

    def check_intrusions(
        self,
        tracked_objects: list[TrackedObject],
        danger_zones: list[DangerZone]
    ) -> list[IntrusionEvent]:
        """
        추적 작업자와 위험구역 매핑 및 MediaPipe 기반 제스처/쓰러짐 감지
        """
        events: list[IntrusionEvent] = []
        now = time.time()

        active_tracker_ids = {obj.tracker_id for obj in tracked_objects}

        for obj in tracked_objects:
            foot_point = get_bottom_center(obj.bbox)
            pose_kps = obj.keypoints
            hand_lms = obj.hand_landmarks

            # 차원 보정
            if pose_kps is not None and len(pose_kps.shape) == 3:
                pose_kps = pose_kps[0]

            # --------------------------------------------------------
            # 1단계: 미디어파이프 기반 글로벌 행동/자세 감지
            # --------------------------------------------------------
            global_key = (obj.tracker_id, "global")
            if global_key not in self._states:
                self._states[global_key] = _TrackerZoneState()
            g_state = self._states[global_key]

            # (1) 양팔 교차 감지 (MediaPipe Pose 기준)
            gx_detected = self._check_arm_cross(pose_kps)
            if gx_detected:
                g_state.consecutive_gesture_x_inside_frames += 1
                g_state.consecutive_gesture_x_outside_frames = 0
                if not g_state.is_gesture_x:
                    if g_state.consecutive_gesture_x_inside_frames >= self.enter_threshold_frames:
                        g_state.is_gesture_x = True
                        g_state.gesture_x_enter_time = now
                        g_state.gesture_x_last_event_time = now
                        events.append(self._create_event(obj, IntrusionState.ENTERED, "gesture_x", "global", "전체 영역", "high", now, pose_kps, hand_lms))
                        logger.warning(f"⚠️ [제스처 X 감지] 작업자#{obj.tracker_id} 양팔 교차 감지")
                else:
                    if (now - g_state.gesture_x_last_event_time) >= self._cooldown_seconds:
                        duration = now - g_state.gesture_x_enter_time
                        g_state.gesture_x_last_event_time = now
                        events.append(self._create_event(obj, IntrusionState.STAYING, "gesture_x", "global", "전체 영역", "high", now, pose_kps, hand_lms, duration))
            else:
                g_state.consecutive_gesture_x_outside_frames += 1
                g_state.consecutive_gesture_x_inside_frames = 0
                if g_state.is_gesture_x:
                    if g_state.consecutive_gesture_x_outside_frames >= self.exit_threshold_frames:
                        g_state.is_gesture_x = False
                        duration = now - g_state.gesture_x_enter_time
                        events.append(self._create_event(obj, IntrusionState.EXITED, "gesture_x", "global", "전체 영역", "high", now, pose_kps, hand_lms, duration))
                        logger.info(f"✅ [제스처 X 해제] 작업자#{obj.tracker_id} 양팔 교차 해제")

            # (2) 도움 요청 흔들기 감지 (MediaPipe Hands & Pose 기준)
            gw_detected = self._check_arm_waving(hand_lms, pose_kps, g_state)
            if gw_detected:
                g_state.consecutive_gesture_wave_inside_frames += 1
                g_state.consecutive_gesture_wave_outside_frames = 0
                if not g_state.is_gesture_wave:
                    if g_state.consecutive_gesture_wave_inside_frames >= self.enter_threshold_frames:
                        g_state.is_gesture_wave = True
                        g_state.gesture_wave_enter_time = now
                        g_state.gesture_wave_last_event_time = now
                        events.append(self._create_event(obj, IntrusionState.ENTERED, "gesture_wave", "global", "전체 영역", "high", now, pose_kps, hand_lms))
                        logger.warning(f"⚠️ [도움 요청 감지] 작업자#{obj.tracker_id} 손 흔들기 감지")
                else:
                    if (now - g_state.gesture_wave_last_event_time) >= self._cooldown_seconds:
                        duration = now - g_state.gesture_wave_enter_time
                        g_state.gesture_wave_last_event_time = now
                        events.append(self._create_event(obj, IntrusionState.STAYING, "gesture_wave", "global", "전체 영역", "high", now, pose_kps, hand_lms, duration))
            else:
                g_state.consecutive_gesture_wave_outside_frames += 1
                g_state.consecutive_gesture_wave_inside_frames = 0
                if g_state.is_gesture_wave:
                    if g_state.consecutive_gesture_wave_outside_frames >= self.exit_threshold_frames:
                        g_state.is_gesture_wave = False
                        duration = now - g_state.gesture_wave_enter_time
                        events.append(self._create_event(obj, IntrusionState.EXITED, "gesture_wave", "global", "전체 영역", "high", now, pose_kps, hand_lms, duration))
                        logger.info(f"✅ [도움 요청 해제] 작업자#{obj.tracker_id} 흔들기 해제")

            # (3) 쓰러짐 감지 (MediaPipe Pose 기준)
            fd_detected = self._check_fall_down(obj.bbox, pose_kps)
            if fd_detected:
                g_state.consecutive_fall_down_inside_frames += 1
                g_state.consecutive_fall_down_outside_frames = 0
                if not g_state.is_fall_down:
                    if g_state.consecutive_fall_down_inside_frames >= self.enter_threshold_frames:
                        g_state.is_fall_down = True
                        g_state.fall_down_enter_time = now
                        g_state.fall_down_last_event_time = now
                        events.append(self._create_event(obj, IntrusionState.ENTERED, "fall_down", "global", "전체 영역", "critical", now, pose_kps, hand_lms))
                        logger.error(f"🚨 [비상정지!!] 작업자#{obj.tracker_id} 비상정지 감지!")
                else:
                    if (now - g_state.fall_down_last_event_time) >= self._cooldown_seconds:
                        duration = now - g_state.fall_down_enter_time
                        g_state.fall_down_last_event_time = now
                        events.append(self._create_event(obj, IntrusionState.STAYING, "fall_down", "global", "전체 영역", "critical", now, pose_kps, hand_lms, duration))
            else:
                g_state.consecutive_fall_down_outside_frames += 1
                g_state.consecutive_fall_down_inside_frames = 0
                if g_state.is_fall_down:
                    if g_state.consecutive_fall_down_outside_frames >= self.exit_threshold_frames:
                        g_state.is_fall_down = False
                        duration = now - g_state.fall_down_enter_time
                        events.append(self._create_event(obj, IntrusionState.EXITED, "fall_down", "global", "전체 영역", "critical", now, pose_kps, hand_lms, duration))
                        logger.info(f"✅ [비상정지 해제] 작업자#{obj.tracker_id} 상태 정상화")

            # --------------------------------------------------------
            # 2단계: 컨베이어 벨트 위험구역 진입 검사
            # --------------------------------------------------------
            for zone in danger_zones:
                key = (obj.tracker_id, zone.zone_id)

                if key not in self._states:
                    self._states[key] = _TrackerZoneState()

                state = self._states[key]
                is_inside = False

                if self._method == "point":
                    is_inside = is_point_in_polygon(foot_point, zone.polygon)
                elif self._method == "multi-point":
                    pts = get_multi_bottom_points(obj.bbox)
                    is_inside = any(is_point_in_polygon(pt, zone.polygon) for pt in pts)
                elif self._method == "overlap":
                    is_inside = check_polygon_overlap_mask(
                        obj.bbox, zone.polygon, self._frame_width, self._frame_height, self._overlap_ratio
                    )
                elif self._method == "segment":
                    if obj.mask is not None:
                        is_inside = check_segment_overlap_mask(
                            obj.mask, zone.polygon, self._overlap_ratio
                        )
                    else:
                        is_inside = is_point_in_polygon(foot_point, zone.polygon)
                elif self._method == "pose-hybrid":
                    point_inside = False
                    step_hit = "NONE"
                    
                    # ── 단계 A: Pose 33개 관절 점 검사 ──
                    if pose_kps is not None and len(pose_kps) > 0:
                        for idx in range(len(pose_kps)):
                            x, y, conf = pose_kps[idx]
                            if conf >= self._pose_conf_threshold:
                                if is_point_in_polygon((int(x), int(y)), zone.polygon):
                                    point_inside = True
                                    step_hit = f"A-kp#{idx}"
                                    break
                                    
                        # ── 단계 B: 뼈대 선분 ↔ 위험구역 경계 교차 검사 ──
                        if not point_inside:
                            for p1, p2 in self.SKELETON_CONNECTIONS:
                                if p1 < len(pose_kps) and p2 < len(pose_kps):
                                    x1, y1, c1 = pose_kps[p1]
                                    x2, y2, c2 = pose_kps[p2]
                                    if c1 >= self._pose_conf_threshold and c2 >= self._pose_conf_threshold:
                                        A = (int(x1), int(y1))
                                        B = (int(x2), int(y2))
                                        if self._is_skeleton_intersecting_zone(A, B, zone.polygon):
                                            point_inside = True
                                            step_hit = f"B-skel({p1},{p2})"
                                            break
                                    
                    # ── 단계 B-2: Hand 관절/마디 연결선 검사 ──
                    if not point_inside and hand_lms is not None:
                        for hand in hand_lms:
                            landmarks = hand["landmarks"]
                            for idx in range(len(landmarks)):
                                h_x, h_y, h_conf = landmarks[idx]
                                if h_conf >= self._pose_conf_threshold:
                                    if is_point_in_polygon((int(h_x), int(h_y)), zone.polygon):
                                        point_inside = True
                                        step_hit = f"B2-hand#{idx}"
                                        break
                                        
                            if not point_inside:
                                for h1, h2 in self.HAND_CONNECTIONS:
                                    if h1 < len(landmarks) and h2 < len(landmarks):
                                        hx1, hy1, hc1 = landmarks[h1]
                                        hx2, hy2, hc2 = landmarks[h2]
                                        if hc1 >= self._pose_conf_threshold and hc2 >= self._pose_conf_threshold:
                                            A = (int(hx1), int(hy1))
                                            B = (int(hx2), int(hy2))
                                            if self._is_skeleton_intersecting_zone(A, B, zone.polygon):
                                                point_inside = True
                                                step_hit = f"B2-hline({h1},{h2})"
                                                break
                            if point_inside:
                                break
                    
                    # ── 단계 C: BBox 전체 ↔ 위험구역 직접 겹침 (최종 보조 판정) ──
                    if not point_inside:
                        bx1, by1, bx2, by2 = [int(v) for v in obj.bbox]
                        bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                        bbox_pts = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2), (bcx, bcy)]
                        for bp in bbox_pts:
                            if is_point_in_polygon(bp, zone.polygon):
                                point_inside = True
                                step_hit = f"C-bboxpt{bp}"
                                break
                        if not point_inside:
                            bbox_edges = [
                                ((bx1, by1), (bx2, by1)),
                                ((bx2, by1), (bx2, by2)),
                                ((bx2, by2), (bx1, by2)),
                                ((bx1, by2), (bx1, by1)),
                            ]
                            poly_pts = zone.polygon.astype(int).tolist() if hasattr(zone.polygon, 'astype') else list(zone.polygon)
                            n_poly = len(poly_pts)
                            for edge_a, edge_b in bbox_edges:
                                for pi in range(n_poly):
                                    C = tuple(poly_pts[pi])
                                    D = tuple(poly_pts[(pi + 1) % n_poly])
                                    if self._is_segment_intersect(edge_a, edge_b, C, D):
                                        point_inside = True
                                        step_hit = f"C-edge({edge_a},{edge_b})x({C},{D})"
                                        break
                                if point_inside:
                                    break
                        if not point_inside:
                            poly_pts_check = zone.polygon.astype(int).tolist() if hasattr(zone.polygon, 'astype') else list(zone.polygon)
                            if len(poly_pts_check) > 0:
                                px, py = poly_pts_check[0]
                                if bx1 <= px <= bx2 and by1 <= py <= by2:
                                    point_inside = True
                                    step_hit = "C-zoneInBbox"
                    
                    is_inside = point_inside
                    
                    # ── 디버그 출력 (매 30프레임마다 1회) ──
                    if not hasattr(self, '_debug_counter'):
                        self._debug_counter = 0
                    self._debug_counter += 1
                    if self._debug_counter % 30 == 0:
                        has_pose = pose_kps is not None
                        has_hand = hand_lms is not None
                        bbox_str = f"({int(obj.bbox[0])},{int(obj.bbox[1])},{int(obj.bbox[2])},{int(obj.bbox[3])})"
                        poly_str = str(zone.polygon[:3].tolist()) + "..." if hasattr(zone.polygon, 'tolist') and len(zone.polygon) > 3 else str(zone.polygon)
                        logger.debug(
                            f"🔍 [DEBUG] ID:{obj.tracker_id} | method={self._method} | "
                            f"bbox={bbox_str} | zone={zone.name} | "
                            f"has_pose={has_pose} | has_hand={has_hand} | "
                            f"is_inside={is_inside} | hit={step_hit} | "
                            f"poly={poly_str}"
                        )
                else:
                    is_inside = is_point_in_polygon(foot_point, zone.polygon)

                if is_inside:
                    state.consecutive_inside_frames += 1
                    state.consecutive_outside_frames = 0

                    if not state.is_inside:
                        if state.consecutive_inside_frames >= self.enter_threshold_frames:
                            state.is_inside = True
                            state.enter_time = now
                            state.last_event_time = now
                            events.append(self._create_event(obj, IntrusionState.ENTERED, "intrusion", zone.zone_id, zone.name, zone.severity, now, pose_kps, hand_lms))
                            logger.warning(f"⚠️ [침입 감지] 작업자#{obj.tracker_id} → {zone.name} 진입")
                    else:
                        if (now - state.last_event_time) >= self._cooldown_seconds:
                            duration = now - state.enter_time
                            state.last_event_time = now
                            events.append(self._create_event(obj, IntrusionState.STAYING, "intrusion", zone.zone_id, zone.name, zone.severity, now, pose_kps, hand_lms, duration))
                else:
                    state.consecutive_outside_frames += 1
                    state.consecutive_inside_frames = 0

                    if state.is_inside:
                        if state.consecutive_outside_frames >= self.exit_threshold_frames:
                            duration = now - state.enter_time
                            state.is_inside = False
                            state.last_event_time = now
                            events.append(self._create_event(obj, IntrusionState.EXITED, "intrusion", zone.zone_id, zone.name, zone.severity, now, pose_kps, hand_lms, duration))
                            logger.info(f"✅ [침입 이탈] 작업자#{obj.tracker_id} ← {zone.name} 이탈 (체류: {duration:.1f}초)")

        # 유실된 추적 ID 정리
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_stale_states(active_tracker_ids)
            self._last_cleanup = now

        return events

    def _create_event(
        self,
        obj: TrackedObject,
        state: IntrusionState,
        threat_type: str,
        zone_id: str,
        zone_name: str,
        zone_severity: str,
        timestamp: float,
        pose_kps: Optional[np.ndarray],
        hand_lms: Optional[List],
        duration: float = 0.0
    ) -> IntrusionEvent:
        """이벤트 직렬화 포맷 빌더"""
        foot_point = get_bottom_center(obj.bbox)
        pose_list = pose_kps.tolist() if pose_kps is not None else None
        
        # 손가락 랜드마크 직렬화 변환
        hand_list = None
        if hand_lms is not None:
            hand_list = []
            for hand in hand_lms:
                hand_list.append({
                    "label": hand["label"],
                    "score": hand["score"],
                    "landmarks": hand["landmarks"].tolist()
                })

        return IntrusionEvent(
            event_type=state,
            tracker_id=obj.tracker_id,
            zone_id=zone_id,
            zone_name=zone_name,
            zone_severity=zone_severity,
            timestamp=timestamp,
            position=foot_point,
            bbox=obj.bbox,
            confidence=obj.confidence,
            duration=duration,
            threat_type=threat_type,
            keypoints=pose_list,
            hand_landmarks=hand_list
        )

    def _check_arm_cross(self, keypoints: Optional[np.ndarray]) -> bool:
        """양팔 교차(X자) 판단 (MediaPipe Pose 33개 관절 기준)"""
        if keypoints is None or len(keypoints) < 25:
            return False

        # 제스처 동작 시 일시적인 모션 블러를 감안하여 신뢰도 임계치 완화
        conf_thresh = 0.3
        # MediaPipe Pose: 11=L_Shoulder, 12=R_Shoulder, 15=L_Wrist, 16=R_Wrist
        if (keypoints[11][2] < conf_thresh or keypoints[12][2] < conf_thresh or
            keypoints[15][2] < conf_thresh or keypoints[16][2] < conf_thresh):
            return False

        x_l_sh, y_l_sh = keypoints[11][0], keypoints[11][1]
        x_r_sh, y_r_sh = keypoints[12][0], keypoints[12][1]
        x_l_wr, y_l_wr = keypoints[15][0], keypoints[15][1]
        x_r_wr, y_r_wr = keypoints[16][0], keypoints[16][1]

        shoulder_width = abs(x_l_sh - x_r_sh)
        if shoulder_width == 0:
            return False

        y_shoulder_avg = (y_l_sh + y_r_sh) / 2.0

        # 양 손목이 어깨 선보다 너무 아래로 내려가지 않은 상태 (상반신 노출 상태에서 가슴/목/얼굴 앞 교차 허용)
        y_limit = y_shoulder_avg + shoulder_width * 1.5
        if (y_l_wr < y_limit) and (y_r_wr < y_limit):
            # 두 손목의 화면 상 좌우 위치가 물리적으로 역전(교차)되었는지 판정
            # 교차 각도나 거리에 상관없이, 왼손목이 오른손목의 오른편에서 왼편으로 넘어가기만 하면 감지
            if x_l_wr < x_r_wr:
                return True

        return False

    def _check_arm_waving(
        self,
        hand_landmarks: Optional[List],
        pose_keypoints: Optional[np.ndarray],
        state: _TrackerZoneState
    ) -> bool:
        """도움 요청 손 흔들기 감지 (MediaPipe Hands & Pose 기준)"""
        # 양손 개별 히스토리 저장소 초기화 보장
        if not hasattr(state, "wrist_x_histories") or not state.wrist_x_histories:
            state.wrist_x_histories = {"Left": [], "Right": []}

        if hand_landmarks is None or len(hand_landmarks) == 0:
            # 손이 검출되지 않는 동안 서서히 이전 버퍼를 비워 튐 현상 방지
            for label in ["Left", "Right"]:
                if state.wrist_x_histories.get(label):
                    state.wrist_x_histories[label].pop(0)
            return False
        
        # 어깨 좌표를 얻을 수 있는지 확인 (Pose)
        if pose_keypoints is None or len(pose_keypoints) < 13:
            return False

        conf_thresh = self._pose_conf_threshold
        if pose_keypoints[11][2] < conf_thresh or pose_keypoints[12][2] < conf_thresh:
            return False

        y_shoulder_min = min(pose_keypoints[11][1], pose_keypoints[12][1])
        shoulder_width = abs(pose_keypoints[11][0] - pose_keypoints[12][0])
        if shoulder_width == 0:
            return False

        # 얼굴/뺨 근처 높이에서 흔드는 동작도 유연하게 잡을 수 있도록 완화 (어깨 Y + 어깨너비의 35% 만큼 하향)
        y_threshold = y_shoulder_min + shoulder_width * 0.35
        
        active_labels = set()
        any_waved = False

        for hand in hand_landmarks:
            label = hand["label"]  # "Left" or "Right"
            wrist_lm = hand["landmarks"][0]  # Wrist
            
            # 손목 신뢰도가 있고 높이 기준을 만족할 경우
            if wrist_lm[2] >= conf_thresh and wrist_lm[1] < y_threshold:
                active_labels.add(label)
                
                if label not in state.wrist_x_histories:
                    state.wrist_x_histories[label] = []
                state.wrist_x_histories[label].append(wrist_lm[0])

                # 30프레임 제한 (최근 1초 기록)
                if len(state.wrist_x_histories[label]) > 30:
                    state.wrist_x_histories[label].pop(0)

                xs = state.wrist_x_histories[label]
                if len(xs) >= 8:  # 최소한의 데이터 축적 시 연산
                    # 1. 진폭(Amplitude) 검사 - 어깨 너비의 15% 이상
                    amplitude = max(xs) - min(xs)
                    if amplitude >= shoulder_width * self.waving_amplitude_ratio:
                        # 2. 이동 방향 추출
                        directions = []
                        for i in range(1, len(xs)):
                            diff = xs[i] - xs[i-1]
                            if abs(diff) > 2.0:
                                directions.append(1 if diff > 0 else -1)

                        # 3. 런렝스 압축 (중복 부호 제거)
                        compressed_dirs = []
                        for d in directions:
                            if not compressed_dirs or compressed_dirs[-1] != d:
                                compressed_dirs.append(d)

                        # 4. 방향 전환 횟수 판정
                        direction_changes = len(compressed_dirs) - 1
                        if direction_changes >= self.waving_direction_changes:
                            any_waved = True

        # 비활성 손들의 히스토리 버퍼 점진적 소거
        for label in ["Left", "Right"]:
            if label not in active_labels:
                if state.wrist_x_histories.get(label):
                    state.wrist_x_histories[label].pop(0)

        return any_waved

    def _check_fall_down(self, bbox: Tuple[float, float, float, float], keypoints: Optional[np.ndarray]) -> bool:
        """작업자 쓰러짐 판정 (MediaPipe Pose 33개 관절 기준)"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return False

        aspect_ratio = w / h

        # 단순 바박 비율(aspect_ratio >= 1.2) 단독 조건은 상반신만 잡히는 웹캠 뷰에서 심각한 오감지를 유발하므로 제거.
        # 반드시 전신 어깨와 골반이 완전 검출되어 척추 축의 수평 누운 기울기가 확인될 때만 판정하도록 제한 강화.
        if keypoints is not None and len(keypoints) >= 25:
            conf_thresh = self._pose_conf_threshold
            # MediaPipe Pose: 11=L_Shoulder, 12=R_Shoulder, 23=L_Hip, 24=R_Hip
            if (keypoints[11][2] >= conf_thresh and keypoints[12][2] >= conf_thresh and
                keypoints[23][2] >= conf_thresh and keypoints[24][2] >= conf_thresh):

                s_x = (keypoints[11][0] + keypoints[12][0]) / 2.0
                s_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
                h_x = (keypoints[23][0] + keypoints[24][0]) / 2.0
                h_y = (keypoints[23][1] + keypoints[24][1]) / 2.0

                dx = abs(h_x - s_x)
                dy = abs(h_y - s_y)

                # 척추 축의 경사각이 누워 있고(수평에 가까움), 바박 비율도 옆으로 누운 형상일 때만 검출
                if dx > 0 and (dy / dx) < 0.577:  # 각도 < 30도
                    if aspect_ratio >= 0.85:
                        return True

        return False

    def _cleanup_stale_states(self, active_tracker_ids: set[int]) -> None:
        """추적 유실된 ID 정리"""
        stale_keys = [
            key for key in self._states
            if key[0] not in active_tracker_ids
        ]
        for key in stale_keys:
            del self._states[key]

    def get_active_intrusions(self) -> dict[tuple[int, str], _TrackerZoneState]:
        """현재 위험구역 침입 대상 리턴"""
        return {
            key: state for key, state in self._states.items()
            if key[1] != "global" and state.is_inside
        }

    def get_active_threats(self) -> list[dict]:
        """현재 모든 위험 상황 요약 리턴 (시각화 연출용)"""
        active_threats = []
        for key, state in self._states.items():
            tracker_id, zone_id = key
            if zone_id == "global":
                if state.is_gesture_x:
                    active_threats.append({"tracker_id": tracker_id, "threat_type": "gesture_x", "severity": "high", "msg": "ARM CROSS"})
                if state.is_gesture_wave:
                    active_threats.append({"tracker_id": tracker_id, "threat_type": "gesture_wave", "severity": "high", "msg": "HELP WAVING"})
                if state.is_fall_down:
                    active_threats.append({"tracker_id": tracker_id, "threat_type": "fall_down", "severity": "critical", "msg": "FALL DOWN"})
            else:
                if state.is_inside:
                    active_threats.append({"tracker_id": tracker_id, "threat_type": "intrusion", "severity": "high", "msg": f"INTRUSION ({zone_id})"})
        return active_threats

    def reset(self) -> None:
        """상태 초기화"""
        self._states.clear()
        logger.info("모든 위협/침입 상태가 초기화되었습니다.")
