"""
침입 감지 모듈
추적된 작업자와 위험구역을 비교하여 침입 이벤트를 생성
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger

from core.tracker import TrackedObject
from core.zone_manager import DangerZone
from utils.geometry import get_bottom_center, is_point_in_polygon


class IntrusionState(str, Enum):
    """침입 상태"""
    OUTSIDE = "outside"     # 위험구역 외부
    ENTERED = "entered"     # 방금 진입
    STAYING = "staying"     # 체류 중
    EXITED = "exited"       # 방금 이탈


@dataclass
class IntrusionEvent:
    """침입 이벤트 데이터"""
    event_type: IntrusionState         # 이벤트 타입
    tracker_id: int                    # 작업자 추적 ID
    zone_id: str                       # 위험구역 ID
    zone_name: str                     # 위험구역 이름
    zone_severity: str                 # 위험 등급
    timestamp: float                   # 이벤트 발생 시각
    position: tuple[int, int]          # 작업자 위치 (하단 중심점)
    bbox: tuple[float, float, float, float]  # 바운딩박스
    confidence: float                  # 감지 신뢰도
    duration: float = 0.0             # 체류 시간 (초)


@dataclass
class _TrackerZoneState:
    """개별 추적자의 구역 상태 추적"""
    is_inside: bool = False
    enter_time: float = 0.0
    last_event_time: float = 0.0
    consecutive_inside_frames: int = 0
    consecutive_outside_frames: int = 0


class IntrusionDetector:
    """침입 감지기"""

    # 침입 판정에 필요한 최소 연속 프레임 (노이즈 필터링)
    ENTER_THRESHOLD_FRAMES = 3
    EXIT_THRESHOLD_FRAMES = 5

    def __init__(
        self,
        cooldown_seconds: float = 10.0
    ):
        """
        Args:
            cooldown_seconds: 같은 (추적자, 구역) 조합의 이벤트 재전송 쿨다운 (초)
        """
        self._cooldown_seconds = cooldown_seconds
        # {(tracker_id, zone_id): _TrackerZoneState}
        self._states: dict[tuple[int, str], _TrackerZoneState] = {}
        # 더 이상 추적되지 않는 ID를 정리하기 위한 타임스탬프
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # 60초마다 정리

        logger.info(f"IntrusionDetector 초기화 완료 (쿨다운: {cooldown_seconds}초)")

    def check_intrusions(
        self,
        tracked_objects: list[TrackedObject],
        danger_zones: list[DangerZone]
    ) -> list[IntrusionEvent]:
        """
        추적된 작업자들과 위험구역을 비교하여 침입 이벤트 생성

        Args:
            tracked_objects: 추적된 작업자 리스트
            danger_zones: 활성 위험구역 리스트

        Returns:
            발생한 IntrusionEvent 리스트
        """
        events: list[IntrusionEvent] = []
        now = time.time()

        # 현재 프레임에 존재하는 추적 ID 수집
        active_tracker_ids = {obj.tracker_id for obj in tracked_objects}

        for obj in tracked_objects:
            # 작업자의 하단 중심점 (발 위치 근사)
            foot_point = get_bottom_center(obj.bbox)

            for zone in danger_zones:
                key = (obj.tracker_id, zone.zone_id)

                # 상태 초기화 (처음 본 조합)
                if key not in self._states:
                    self._states[key] = _TrackerZoneState()

                state = self._states[key]

                # 폴리곤 내부 판정
                is_inside = is_point_in_polygon(foot_point, zone.polygon)

                if is_inside:
                    state.consecutive_inside_frames += 1
                    state.consecutive_outside_frames = 0

                    if not state.is_inside:
                        # 아직 "진입" 상태가 아닌데 내부에 있음
                        if state.consecutive_inside_frames >= self.ENTER_THRESHOLD_FRAMES:
                            # 충분한 프레임 동안 내부에 있었으므로 진입 확정
                            state.is_inside = True
                            state.enter_time = now

                            # 쿨다운 체크
                            if self._should_send_event(state, now):
                                event = IntrusionEvent(
                                    event_type=IntrusionState.ENTERED,
                                    tracker_id=obj.tracker_id,
                                    zone_id=zone.zone_id,
                                    zone_name=zone.name,
                                    zone_severity=zone.severity,
                                    timestamp=now,
                                    position=foot_point,
                                    bbox=obj.bbox,
                                    confidence=obj.confidence
                                )
                                events.append(event)
                                state.last_event_time = now
                                logger.warning(
                                    f"⚠️ 침입 감지! "
                                    f"작업자#{obj.tracker_id} → {zone.name} "
                                    f"(위치: {foot_point}, 위험등급: {zone.severity})"
                                )
                    else:
                        # 이미 내부에 있는 상태 = STAYING
                        duration = now - state.enter_time

                        # 주기적으로 체류 이벤트 전송
                        if self._should_send_event(state, now):
                            event = IntrusionEvent(
                                event_type=IntrusionState.STAYING,
                                tracker_id=obj.tracker_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                zone_severity=zone.severity,
                                timestamp=now,
                                position=foot_point,
                                bbox=obj.bbox,
                                confidence=obj.confidence,
                                duration=duration
                            )
                            events.append(event)
                            state.last_event_time = now

                else:
                    state.consecutive_outside_frames += 1
                    state.consecutive_inside_frames = 0

                    if state.is_inside:
                        # 이전에 내부에 있었는데 외부로 나옴
                        if state.consecutive_outside_frames >= self.EXIT_THRESHOLD_FRAMES:
                            # 충분한 프레임 동안 외부에 있었으므로 이탈 확정
                            duration = now - state.enter_time
                            state.is_inside = False

                            event = IntrusionEvent(
                                event_type=IntrusionState.EXITED,
                                tracker_id=obj.tracker_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                zone_severity=zone.severity,
                                timestamp=now,
                                position=foot_point,
                                bbox=obj.bbox,
                                confidence=obj.confidence,
                                duration=duration
                            )
                            events.append(event)
                            state.last_event_time = now
                            logger.info(
                                f"✅ 이탈 감지: "
                                f"작업자#{obj.tracker_id} ← {zone.name} "
                                f"(체류시간: {duration:.1f}초)"
                            )

        # 주기적 상태 정리 (사라진 추적자 제거)
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_stale_states(active_tracker_ids)
            self._last_cleanup = now

        return events

    def _should_send_event(self, state: _TrackerZoneState, now: float) -> bool:
        """쿨다운 기반 이벤트 전송 여부 판정"""
        if state.last_event_time == 0:
            return True
        return (now - state.last_event_time) >= self._cooldown_seconds

    def _cleanup_stale_states(self, active_tracker_ids: set[int]) -> None:
        """더 이상 추적되지 않는 상태 정리"""
        stale_keys = [
            key for key in self._states
            if key[0] not in active_tracker_ids
        ]
        for key in stale_keys:
            del self._states[key]

        if stale_keys:
            logger.debug(f"오래된 추적 상태 {len(stale_keys)}개 정리됨")

    def get_active_intrusions(self) -> dict[tuple[int, str], _TrackerZoneState]:
        """현재 위험구역 내부에 있는 추적자 상태 반환"""
        return {
            key: state for key, state in self._states.items()
            if state.is_inside
        }

    def reset(self) -> None:
        """모든 상태 초기화"""
        self._states.clear()
        logger.info("침입 감지기 상태 초기화됨")
