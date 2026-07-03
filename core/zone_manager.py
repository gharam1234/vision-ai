"""
위험구역 관리 모듈
백엔드 API에서 위험구역 폴리곤 좌표를 조회하고 관리
"""

import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from utils.geometry import polygon_from_points, normalize_polygon


@dataclass
class DangerZone:
    """위험구역 데이터 클래스"""
    zone_id: str                    # 위험구역 고유 ID
    name: str                       # 위험구역 이름
    polygon: np.ndarray             # 폴리곤 좌표 (pixel 단위)
    severity: str = "high"          # 위험 등급 (low/medium/high/critical)
    is_active: bool = True          # 활성화 상태
    color: tuple = (0, 0, 255)      # 표시 색상 (BGR)
    metadata: dict = field(default_factory=dict)


class DangerZoneManager:
    """위험구역 관리자"""

    def __init__(
        self,
        api_client=None,
        camera_id: str = "cam-001",
        poll_interval: int = 30,
        frame_width: int = 1280,
        frame_height: int = 720,
        default_zones: Optional[list[dict]] = None
    ):
        """
        Args:
            api_client: 백엔드 API 클라이언트 (None이면 기본 zone 사용)
            camera_id: 카메라 식별자
            poll_interval: 위험구역 폴링 간격 (초)
            frame_width: 프레임 너비 (좌표 변환용)
            frame_height: 프레임 높이 (좌표 변환용)
            default_zones: 기본 위험구역 데이터
        """
        self._api_client = api_client
        self._camera_id = camera_id
        self._poll_interval = poll_interval
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._zones: dict[str, DangerZone] = {}
        self._lock = threading.Lock()
        self._polling_thread: Optional[threading.Thread] = None
        self._running = False

        # 기본 위험구역 로드
        if default_zones:
            self._load_default_zones(default_zones)

        logger.info(
            f"DangerZoneManager 초기화 완료 "
            f"(카메라: {camera_id}, 폴링 간격: {poll_interval}초)"
        )

    def _load_default_zones(self, zones_data: list[dict]) -> None:
        """기본 위험구역 데이터 로드"""
        for zone_data in zones_data:
            zone = self._parse_zone_data(zone_data)
            if zone:
                self._zones[zone.zone_id] = zone
                logger.info(f"기본 위험구역 로드: {zone.name} ({zone.zone_id})")

    def _parse_zone_data(self, data: dict) -> Optional[DangerZone]:
        """API 또는 설정 데이터를 DangerZone 객체로 파싱"""
        try:
            points = data.get("points", data.get("polygon", []))
            if not points or len(points) < 3:
                logger.warning(f"유효하지 않은 위험구역 데이터: 최소 3개 좌표 필요")
                return None

            polygon = polygon_from_points(points)

            # 정규화 좌표인지 확인 (0~1 범위)
            is_normalized = data.get("is_normalized", False)
            if is_normalized:
                polygon = normalize_polygon(
                    polygon, self._frame_width, self._frame_height,
                    is_normalized=True
                )

            # 위험 등급에 따른 색상 설정
            severity = data.get("severity", "high")
            color_map = {
                "low": (0, 200, 255),        # 주황 (BGR)
                "medium": (0, 165, 255),     # 주황 (BGR)
                "high": (0, 0, 255),         # 빨강 (BGR)
                "critical": (0, 0, 200),     # 진한 빨강 (BGR)
            }

            return DangerZone(
                zone_id=data.get("zone_id", data.get("id", f"zone-{len(self._zones)}")),
                name=data.get("name", "위험구역"),
                polygon=polygon,
                severity=severity,
                is_active=data.get("is_active", True),
                color=color_map.get(severity, (0, 0, 255)),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"위험구역 데이터 파싱 실패: {e}")
            return None

    def add_zone(self, zone_data: dict) -> Optional[DangerZone]:
        """위험구역 추가"""
        zone = self._parse_zone_data(zone_data)
        if zone:
            with self._lock:
                self._zones[zone.zone_id] = zone
            logger.info(f"위험구역 추가: {zone.name} ({zone.zone_id})")
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        """위험구역 제거"""
        with self._lock:
            if zone_id in self._zones:
                del self._zones[zone_id]
                logger.info(f"위험구역 제거: {zone_id}")
                return True
        return False

    def get_zones(self) -> list[DangerZone]:
        """활성화된 위험구역 목록 반환"""
        with self._lock:
            return [z for z in self._zones.values() if z.is_active]

    def get_all_zones(self) -> list[DangerZone]:
        """모든 위험구역 목록 반환"""
        with self._lock:
            return list(self._zones.values())

    async def fetch_zones_from_api(self) -> bool:
        """백엔드 API에서 위험구역 데이터 조회"""
        if self._api_client is None:
            return False

        try:
            zones_data = await self._api_client.get_danger_zones(self._camera_id)
            if zones_data:
                with self._lock:
                    self._zones.clear()
                    for zone_data in zones_data:
                        zone = self._parse_zone_data(zone_data)
                        if zone:
                            self._zones[zone.zone_id] = zone
                logger.info(f"API에서 {len(self._zones)}개 위험구역 로드 완료")
                return True
        except Exception as e:
            logger.error(f"위험구역 API 조회 실패: {e}")
        return False

    def start_polling(self) -> None:
        """위험구역 주기적 폴링 시작"""
        if self._api_client is None:
            logger.warning("API 클라이언트 없음 - 폴링 비활성화")
            return

        self._running = True
        self._polling_thread = threading.Thread(
            target=self._polling_loop, daemon=True
        )
        self._polling_thread.start()
        logger.info(f"위험구역 폴링 시작 (간격: {self._poll_interval}초)")

    def _polling_loop(self) -> None:
        """폴링 루프 (백그라운드 스레드)"""
        import asyncio

        loop = asyncio.new_event_loop()
        while self._running:
            try:
                loop.run_until_complete(self.fetch_zones_from_api())
            except Exception as e:
                logger.error(f"폴링 오류: {e}")
            time.sleep(self._poll_interval)
        loop.close()

    def stop_polling(self) -> None:
        """폴링 중지"""
        self._running = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5)
            logger.info("위험구역 폴링 중지됨")

    def set_demo_zones(self, frame_width: int, frame_height: int) -> None:
        """
        데모/테스트용 위험구역 자동 생성
        프레임 중앙에 사각형 위험구역 생성
        """
        demo_zones = [
            {
                "zone_id": "demo-zone-1",
                "name": "위험구역 A (중앙)",
                "points": [
                    [frame_width * 0.3, frame_height * 0.3],
                    [frame_width * 0.7, frame_height * 0.3],
                    [frame_width * 0.7, frame_height * 0.8],
                    [frame_width * 0.3, frame_height * 0.8],
                ],
                "severity": "high"
            },
            {
                "zone_id": "demo-zone-2",
                "name": "위험구역 B (우측)",
                "points": [
                    [frame_width * 0.75, frame_height * 0.2],
                    [frame_width * 0.95, frame_height * 0.2],
                    [frame_width * 0.95, frame_height * 0.6],
                    [frame_width * 0.75, frame_height * 0.6],
                ],
                "severity": "critical"
            }
        ]

        for zone_data in demo_zones:
            self.add_zone(zone_data)

        logger.info(f"데모 위험구역 {len(demo_zones)}개 생성됨")
