"""
전체 설정 관리 모듈
환경변수 또는 .env 파일로 오버라이드 가능
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class DetectorSettings(BaseSettings):
    """YOLOv11 감지 설정"""
    model_path: str = Field(default="yolo11n.pt", description="YOLO 모델 경로")
    confidence_threshold: float = Field(default=0.5, description="감지 신뢰도 임계값")
    iou_threshold: float = Field(default=0.45, description="NMS IoU 임계값")
    device: str = Field(default="auto", description="추론 디바이스 (auto/cpu/cuda/0)")
    img_size: int = Field(default=640, description="추론 이미지 크기")
    target_classes: list[int] = Field(default=[0], description="감지 대상 클래스 ID (0=person)")

    class Config:
        env_prefix = "DETECTOR_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class TrackerSettings(BaseSettings):
    """ByteTrack 추적 설정"""
    track_thresh: float = Field(default=0.25, description="추적 신뢰도 임계값")
    track_buffer: int = Field(default=30, description="추적 버퍼 프레임 수")
    match_thresh: float = Field(default=0.8, description="매칭 임계값")

    class Config:
        env_prefix = "TRACKER_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class VideoSettings(BaseSettings):
    """영상 소스 설정"""
    source: str = Field(default="sample.mp4", description="영상 소스 경로 (파일/RTSP URL)")
    width: int = Field(default=1280, description="프레임 리사이즈 너비")
    height: int = Field(default=720, description="프레임 리사이즈 높이")
    fps_limit: int = Field(default=30, description="최대 FPS 제한")
    loop: bool = Field(default=True, description="영상 파일 반복 재생")

    class Config:
        env_prefix = "VIDEO_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class ZoneSettings(BaseSettings):
    """위험구역 설정"""
    poll_interval: int = Field(default=30, description="위험구역 폴링 간격 (초)")
    default_zones: list[dict] = Field(
        default=[],
        description="기본 위험구역 (API 연결 불가 시 사용)"
    )

    class Config:
        env_prefix = "ZONE_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class EventSettings(BaseSettings):
    """이벤트 전송 및 감도 튜닝 설정"""
    cooldown_seconds: float = Field(default=10.0, description="같은 이벤트 재전송 쿨다운 (초)")
    save_snapshots: bool = Field(default=True, description="침입 감지 시 스냅샷 저장")
    snapshot_dir: str = Field(default="snapshots", description="스냅샷 저장 경로")
    
    # 감도 튜닝용 변수 복원
    intrusion_method: str = Field(default="pose-hybrid", description="침입 감지 방법 (point/multi-point/overlap/segment/pose-hybrid)")
    overlap_ratio: float = Field(default=0.2, description="overlap 방식 사용 시 바운딩박스 하부 비율")
    pose_conf_threshold: float = Field(default=0.5, description="포즈 키포인트 검출 신뢰도 임계치")
    waving_amplitude_ratio: float = Field(default=0.15, description="손 흔들기 최소 진폭 비율 (어깨 너비 대비)")
    waving_direction_changes: int = Field(default=2, description="손 흔들기 최소 방향 전환 횟수")
    enter_threshold_frames: int = Field(default=3, description="위험 감지 판정에 필요한 최소 연속 프레임 수")
    exit_threshold_frames: int = Field(default=5, description="위험 해제 판정에 필요한 최소 연속 프레임 수")

    class Config:
        env_prefix = "EVENT_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class APISettings(BaseSettings):
    """백엔드 API 설정"""
    base_url: str = Field(default="http://localhost:8000", description="백엔드 API 베이스 URL")
    ws_url: str = Field(default="ws://localhost:8000/ws/events", description="WebSocket URL")
    api_key: Optional[str] = Field(default=None, description="API 인증 키")
    timeout: float = Field(default=5.0, description="API 요청 타임아웃 (초)")
    retry_count: int = Field(default=3, description="API 요청 재시도 횟수")

    class Config:
        env_prefix = "API_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class AppSettings(BaseSettings):
    """전체 앱 설정"""
    camera_id: str = Field(default="cam-001", description="카메라 식별자")
    show_display: bool = Field(default=True, description="시각화 디스플레이 표시")
    log_level: str = Field(default="INFO", description="로그 레벨")

    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    zone: ZoneSettings = Field(default_factory=ZoneSettings)
    event: EventSettings = Field(default_factory=EventSettings)
    api: APISettings = Field(default_factory=APISettings)

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 전역 설정 인스턴스
settings = AppSettings()
