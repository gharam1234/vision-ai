"""통신 모듈 - 백엔드 API 클라이언트, 이벤트 전송"""
from .api_client import BackendAPIClient
from .event_sender import EventSender

__all__ = ["BackendAPIClient", "EventSender"]
