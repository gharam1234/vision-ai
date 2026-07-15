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
        self._ws = None
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
        # 스냅샷 저장 및 Base64 인코딩
        snapshot_path = None
        snapshot_b64 = None
        if self._save_snapshots and frame is not None and event.event_type == IntrusionState.ENTERED:
            snapshot_path = self._save_snapshot(event, frame)
            if snapshot_path is not None:
                try:
                    with open(snapshot_path, "rb") as f:
                        snapshot_b64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    logger.error(f"스냅샷 Base64 변환 실패: {e}")

        # 이벤트 데이터 직렬화
        event_data = self._serialize_event(event, snapshot_path, snapshot_b64)

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
        snapshot_path: Optional[str] = None,
        snapshot_b64: Optional[str] = None
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
            "snapshot_path": snapshot_path,
            "snapshot_b64": snapshot_b64,
            "threat_type": event.threat_type,
            "keypoints": event.keypoints,
            "hand_landmarks": event.hand_landmarks
        }

    def _save_snapshot(
        self,
        event: IntrusionEvent,
        frame: np.ndarray
    ) -> Optional[str]:
        """침입 및 위험 상황 감지 스냅샷 저장"""
        try:
            import os
            from datetime import datetime

            timestamp_str = datetime.fromtimestamp(event.timestamp).strftime(
                "%Y%m%d_%H%M%S"
            )
            
            prefix = event.threat_type
            filename = (
                f"{prefix}_{self._camera_id}_"
                f"worker{event.tracker_id}_"
                f"{event.zone_id}_"
                f"{timestamp_str}.jpg"
            )
            filepath = os.path.join(self._snapshot_dir, filename)

            # 스냅샷에 위협 정보 오버레이
            snapshot = frame.copy()
            x1, y1, x2, y2 = map(int, event.bbox)
            
            # 위험 수준에 따라 테두리 색상 선택 (critical: 빨간색, 그 외: 주황색)
            color = (0, 0, 255) if event.zone_severity == "critical" else (0, 140, 255)
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), color, 3)
            
            # 위협 텍스트 메시지화
            threat_msg = f"{event.threat_type.upper()}"
            if event.threat_type == "gesture_x":
                threat_msg = "GESTURE - Arm Crossed"
            elif event.threat_type == "gesture_wave":
                threat_msg = "GESTURE - Help Waving"
            elif event.threat_type == "fall_down":
                threat_msg = "EMERGENCY - Worker Fall Down"
            elif event.threat_type == "intrusion":
                threat_msg = f"INTRUSION - {event.zone_name}"

            cv2.putText(
                snapshot,
                f"{threat_msg} (Worker#{event.tracker_id})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
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
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main_async_loop())
        except Exception as e:
            logger.error(f"비동기 메인 전송 루프 예외: {e}")
        finally:
            loop.close()

    async def _main_async_loop(self) -> None:
        """비동기 메인 루프: 웹소켓 연결 유지 및 큐 처리"""
        # 웹소켓 연결 상시 유지 태스크 기동
        connect_task = asyncio.create_task(self._maintain_connection())

        while self._running:
            try:
                # 동기 큐 get_nowait()을 사용하여 비동기 대기 구현
                try:
                    event_data = self._event_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue

                # 전송 시도
                success = await self._send_event_async(event_data)
                if success:
                    self._stats["total_sent"] += 1
                else:
                    self._stats["total_failed"] += 1
                    logger.info(
                        f"[LOCAL] 이벤트: {event_data['event_type']} "
                        f"작업자#{event_data['tracker_id']} → "
                        f"{event_data['zone_name']} "
                        f"({event_data.get('threat_type', 'intrusion')})"
                    )

            except Exception as e:
                logger.error(f"비동기 이벤트 전송 처리 예외: {e}")
                await asyncio.sleep(1.0)

        # 종료 시 소켓 정리 및 백그라운드 태스크 취소
        connect_task.cancel()
        try:
            await connect_task
        except asyncio.CancelledError:
            pass
        await self._close_ws()

    async def _maintain_connection(self) -> None:
        """웹소켓 커넥션을 상시로 유지하고, 끊어지면 자동 재연결 시도"""
        import websockets

        while self._running:
            if self._ws is None or not self._ws_connected:
                try:
                    logger.debug(f"웹소켓 연결 시도 중... ({self._ws_url})")
                    self._ws = await websockets.connect(
                        self._ws_url,
                        close_timeout=3
                    )
                    self._ws_connected = True
                    logger.info(f"🟢 백엔드 웹소켓 상시 연결 수립 성공: {self._ws_url}")
                except ImportError:
                    logger.error("websockets 라이브러리 미설치 - WebSocket 비활성화")
                    await asyncio.sleep(10.0)
                except Exception as e:
                    self._ws_connected = False
                    self._ws = None
                    logger.warning(f"⚠️ 백엔드 웹소켓 연결 실패. 3초 후 재시도... ({e})")
                    await asyncio.sleep(3.0)
            else:
                if self._ws.state.name != "OPEN":
                    logger.warning("🔌 웹소켓 연결이 예기치 않게 종료되었습니다. 재접속을 시도합니다.")
                    self._ws_connected = False
                    self._ws = None
                else:
                    await asyncio.sleep(1.0)

    async def _send_event_async(self, event_data: dict) -> bool:
        """비동기 이벤트 전송"""
        # WebSocket 전송 시도 (상시 연결 채널)
        ws_success = await self._send_via_websocket(event_data)
        if ws_success:
            self._stats["ws_sends"] += 1
            return True

        # WebSocket 실패 시 REST API fallback
        if self._api_client:
            rest_success = await self._api_client.send_event(event_data)
            if rest_success:
                self._stats["rest_sends"] += 1
                return True

        return False

    async def _send_via_websocket(self, event_data: dict) -> bool:
        """상시 연결된 WebSocket 채널을 통해 이벤트 전송"""
        if not self._ws_connected or self._ws is None or self._ws.state.name != "OPEN":
            self._ws_connected = False
            return False

        try:
            await self._ws.send(json.dumps(event_data, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning(f"⚠️ 웹소켓 패킷 송신 실패 (REST Fallback 사용 예정): {e}")
            self._ws_connected = False
            self._ws = None
            return False

    async def _close_ws(self) -> None:
        """웹소켓 소켓 종료"""
        if self._ws is not None:
            try:
                await self._ws.close()
                logger.info("🔌 백엔드 웹소켓 상시 세션 안전하게 종료됨")
            except Exception:
                pass
            finally:
                self._ws = None
                self._ws_connected = False

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
