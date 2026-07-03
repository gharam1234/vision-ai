"""
백엔드 API 클라이언트 모듈
FastAPI 백엔드와 REST API 통신
"""

import httpx
from typing import Optional
from loguru import logger


class BackendAPIClient:
    """백엔드 FastAPI 서버와 통신하는 REST API 클라이언트"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        retry_count: int = 3
    ):
        """
        Args:
            base_url: 백엔드 API 베이스 URL
            api_key: API 인증 키
            timeout: 요청 타임아웃 (초)
            retry_count: 재시도 횟수
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retry_count = retry_count

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

        logger.info(f"BackendAPIClient 초기화: {self._base_url}")

    async def get_danger_zones(self, camera_id: str) -> Optional[list[dict]]:
        """
        특정 카메라의 위험구역 데이터 조회

        Args:
            camera_id: 카메라 식별자

        Returns:
            위험구역 데이터 리스트 또는 None (실패 시)
        """
        endpoint = f"/api/cameras/{camera_id}/zones"
        return await self._get(endpoint)

    async def get_camera_config(self, camera_id: str) -> Optional[dict]:
        """
        카메라 설정 조회

        Args:
            camera_id: 카메라 식별자

        Returns:
            카메라 설정 데이터 또는 None
        """
        endpoint = f"/api/cameras/{camera_id}"
        return await self._get(endpoint)

    async def send_event(self, event_data: dict) -> bool:
        """
        이벤트 데이터 전송 (REST fallback)

        Args:
            event_data: 이벤트 데이터 딕셔너리

        Returns:
            전송 성공 여부
        """
        endpoint = "/api/events"
        result = await self._post(endpoint, event_data)
        return result is not None

    async def send_incident_report(self, report_data: dict) -> Optional[dict]:
        """
        사고 보고서 전송

        Args:
            report_data: 사고 보고서 데이터

        Returns:
            서버 응답 데이터 또는 None
        """
        endpoint = "/api/incidents"
        return await self._post(endpoint, report_data)

    async def _get(self, endpoint: str) -> Optional[any]:
        """GET 요청 (재시도 포함)"""
        for attempt in range(self._retry_count):
            try:
                response = await self._client.get(endpoint)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                logger.warning(
                    f"API 요청 타임아웃: GET {endpoint} "
                    f"(시도 {attempt + 1}/{self._retry_count})"
                )
            except httpx.HTTPStatusError as e:
                logger.error(f"API HTTP 오류: {e.response.status_code} - {endpoint}")
                return None
            except httpx.ConnectError:
                logger.warning(
                    f"API 연결 실패: {endpoint} "
                    f"(시도 {attempt + 1}/{self._retry_count})"
                )
            except Exception as e:
                logger.error(f"API 요청 오류: {e}")
                return None

        logger.error(f"API 요청 최종 실패: GET {endpoint}")
        return None

    async def _post(self, endpoint: str, data: dict) -> Optional[any]:
        """POST 요청 (재시도 포함)"""
        for attempt in range(self._retry_count):
            try:
                response = await self._client.post(endpoint, json=data)
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                logger.warning(
                    f"API 요청 타임아웃: POST {endpoint} "
                    f"(시도 {attempt + 1}/{self._retry_count})"
                )
            except httpx.HTTPStatusError as e:
                logger.error(f"API HTTP 오류: {e.response.status_code} - {endpoint}")
                return None
            except httpx.ConnectError:
                logger.warning(
                    f"API 연결 실패: {endpoint} "
                    f"(시도 {attempt + 1}/{self._retry_count})"
                )
            except Exception as e:
                logger.error(f"API 요청 오류: {e}")
                return None

        logger.error(f"API 요청 최종 실패: POST {endpoint}")
        return None

    async def health_check(self) -> bool:
        """백엔드 서버 헬스 체크"""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """클라이언트 종료"""
        await self._client.aclose()
        logger.info("API 클라이언트 종료됨")
