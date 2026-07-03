"""
이벤트 전송 모듈
WebSocket 기반 실시간 이벤트 전송 + REST API fallback
"""

import json
import asyncio
import time
import threading
import queue
import base64
from typing import Optional
from loguru import logger

import cv2
import numpy as np

from processing.intrusion_detector import IntrusionEvent, IntrusionState
from communication.api_client import BackendAPIClient


class EventSender:
    """
    이벤트 전송기
    WebSocket으로 실시간 이벤트를 전송하고, 실패 시 REST API로 fallback
    별도 스레드에서 비동기 전송 수행 (메인 루프 블로킹 방지)
    """

    def __init__(
        self,
        api_client: Optional[BackendAPIClient] = None,
        ws_url: str = "ws://localhost:8000/ws/events",
        camera_id: str = "cam-001",
        save_snapshots: bool = True,
        snapshot_dir: str = "snapshots",
        max_queue_size: int = 100
    ):
        """
        Args:
            api_client: REST API 클라이언트
            ws_url: WebSocket 서버 URL
            camera_id: 카메라 식별자
            save_snapshots: 침입 감지 시 스냅샷 저장 여부
            snapshot_dir: 스냅샷 저장 경로
            max_queue_size: 이벤트 큐 최대 크기
        """
        self._api_client = api_client
        self._ws_url = ws_url
        self._camera_id = camera_id
        self._save_snapshots = save_snapshots
        self._snapshot_dir = snapshot_dir
        self._event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._sender_thread: Optional[threading.Thread] = None
        self._ws_connected = False
        self._stats = {
            "total_sent": 0,
            "total_failed": 0,
            "ws_sends": 0,
            "rest_sends": 0,
        }

        # 스냅샷 디렉토리 생성
        if save_snapshots:
            import os
            os.makedirs(snapshot_dir, exist_ok=True)

        logger.info(
            f"EventSender 초기화 완료 "
            f"(WS: {ws_url}, 스냅샷: {save_snapshots})"
        )

    def send_event(
        self,
        event: IntrusionEvent,
        frame: Optional[np.ndarray] = None
    ) -> None:
        """
        이벤트를 큐에 추가 (비블로킹)

        Args:
            event: 침입 이벤트
            frame: 현재 프레임 (스냅샷 저장용)
        """
        # 스냅샷 저장
        snapshot_path = None
        if self._save_snapshots and frame is not None and event.event_type == IntrusionState.ENTERED:
            snapshot_path = self._save_snapshot(event, frame)

        # 이벤트 데이터 직렬화
        event_data = self._serialize_event(event, snapshot_path)

        try:
            self._event_queue.put_nowait(event_data)
        except queue.Full:
            logger.warning("이벤트 큐 가득 참 - 가장 오래된 이벤트 삭제")
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event_data)
            except queue.Empty:
                pass

    def _serialize_event(
        self,
        event: IntrusionEvent,
        snapshot_path: Optional[str] = None
    ) -> dict:
        """이벤트를 전송용 딕셔너리로 직렬화"""
        return {
            "type": "intrusion_event",
            "camera_id": self._camera_id,
            "event_type": event.event_type.value,
            "tracker_id": event.tracker_id,
            "zone_id": event.zone_id,
            "zone_name": event.zone_name,
            "zone_severity": event.zone_severity,
            "timestamp": event.timestamp,
            "position": {"x": event.position[0], "y": event.position[1]},
            "bbox": {
                "x1": event.bbox[0],
                "y1": event.bbox[1],
                "x2": event.bbox[2],
                "y2": event.bbox[3]
            },
            "confidence": event.confidence,
            "duration": event.duration,
            "snapshot_path": snapshot_path
        }

    def _save_snapshot(
        self,
        event: IntrusionEvent,
        frame: np.ndarray
    ) -> Optional[str]:
        """침입 감지 스냅샷 저장"""
        try:
            import os
            from datetime import datetime

            timestamp_str = datetime.fromtimestamp(event.timestamp).strftime(
                "%Y%m%d_%H%M%S"
            )
            filename = (
                f"intrusion_{self._camera_id}_"
                f"worker{event.tracker_id}_"
                f"{event.zone_id}_"
                f"{timestamp_str}.jpg"
            )
            filepath = os.path.join(self._snapshot_dir, filename)

            # 스냅샷에 침입 정보 오버레이
            snapshot = frame.copy()
            x1, y1, x2, y2 = map(int, event.bbox)
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                snapshot,
                f"INTRUSION - Worker#{event.tracker_id} -> {event.zone_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

            cv2.imwrite(filepath, snapshot)
            logger.info(f"스냅샷 저장: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"스냅샷 저장 실패: {e}")
            return None

    def start(self) -> None:
        """이벤트 전송 스레드 시작"""
        self._running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True
        )
        self._sender_thread.start()
        logger.info("이벤트 전송기 시작됨")

    def _sender_loop(self) -> None:
        """이벤트 전송 루프 (별도 스레드)"""
        loop = asyncio.new_event_loop()

        while self._running:
            try:
                # 큐에서 이벤트 가져오기 (1초 타임아웃)
                event_data = self._event_queue.get(timeout=1.0)

                # 전송 시도
                success = loop.run_until_complete(
                    self._send_event_async(event_data)
                )

                if success:
                    self._stats["total_sent"] += 1
                else:
                    self._stats["total_failed"] += 1
                    # 로컬 로깅 (fallback)
                    logger.info(
                        f"[LOCAL] 이벤트: {event_data['event_type']} "
                        f"작업자#{event_data['tracker_id']} → "
                        f"{event_data['zone_name']}"
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"이벤트 전송 루프 오류: {e}")

        loop.close()

    async def _send_event_async(self, event_data: dict) -> bool:
        """비동기 이벤트 전송"""
        # WebSocket 전송 시도
        ws_success = await self._send_via_websocket(event_data)
        if ws_success:
            self._stats["ws_sends"] += 1
            return True

        # REST API fallback
        if self._api_client:
            rest_success = await self._api_client.send_event(event_data)
            if rest_success:
                self._stats["rest_sends"] += 1
                return True

        return False

    async def _send_via_websocket(self, event_data: dict) -> bool:
        """WebSocket으로 이벤트 전송"""
        try:
            import websockets

            async with websockets.connect(
                self._ws_url, close_timeout=3
            ) as ws:
                await ws.send(json.dumps(event_data, ensure_ascii=False))
                self._ws_connected = True
                return True

        except ImportError:
            logger.debug("websockets 라이브러리 미설치 - WebSocket 비활성화")
            return False
        except Exception:
            self._ws_connected = False
            return False

    def stop(self) -> None:
        """이벤트 전송기 중지"""
        self._running = False
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5)
        logger.info(
            f"이벤트 전송기 중지됨 "
            f"(전송: {self._stats['total_sent']}, "
            f"실패: {self._stats['total_failed']})"
        )

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    @property
    def is_ws_connected(self) -> bool:
        return self._ws_connected
