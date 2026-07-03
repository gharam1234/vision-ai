"""
영상 소스 모듈
MP4 파일, RTSP 스트림 등 다양한 영상 소스를 추상화
"""

import cv2
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from loguru import logger


class VideoSource(ABC):
    """영상 소스 추상 클래스"""

    @abstractmethod
    def open(self) -> bool:
        """영상 소스 열기"""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """프레임 읽기"""
        pass

    @abstractmethod
    def release(self) -> None:
        """영상 소스 해제"""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """영상 소스 열림 상태"""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """원본 FPS"""
        pass

    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """(width, height)"""
        pass


class FileVideoSource(VideoSource):
    """MP4 등 영상 파일 소스"""

    def __init__(
        self,
        file_path: str,
        resize: Optional[Tuple[int, int]] = None,
        fps_limit: int = 30,
        loop: bool = True
    ):
        """
        Args:
            file_path: 영상 파일 경로
            resize: 리사이즈 크기 (width, height), None이면 원본 크기
            fps_limit: 최대 FPS 제한
            loop: 영상 반복 재생 여부
        """
        self._file_path = file_path
        self._resize = resize
        self._fps_limit = fps_limit
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._original_fps: float = 30.0
        self._original_size: Tuple[int, int] = (0, 0)
        self._last_frame_time: float = 0.0
        self._frame_count: int = 0

    def open(self) -> bool:
        """영상 파일 열기"""
        self._cap = cv2.VideoCapture(self._file_path)
        if not self._cap.isOpened():
            logger.error(f"영상 파일을 열 수 없습니다: {self._file_path}")
            return False

        self._original_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._original_size = (w, h)
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"영상 파일 열림: {self._file_path} "
            f"({w}x{h}, {self._original_fps:.1f}fps, {total}프레임)"
        )
        return True

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        """프레임 읽기 (FPS 제한 적용)"""
        if self._cap is None or not self._cap.isOpened():
            return False, None

        # FPS 제한
        min_interval = 1.0 / self._fps_limit
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        ret, frame = self._cap.read()

        # 영상 끝에 도달하면 반복 또는 종료
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return False, None
                logger.info("영상 반복 재생 시작")
                self._frame_count = 0
            else:
                logger.info("영상 파일 재생 완료")
                return False, None

        self._last_frame_time = time.time()
        self._frame_count += 1

        # 리사이즈 적용
        if self._resize is not None and frame is not None:
            frame = cv2.resize(frame, self._resize)

        return True, frame

    def release(self) -> None:
        """영상 소스 해제"""
        if self._cap is not None:
            self._cap.release()
            logger.info("영상 소스 해제됨")

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._original_fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._resize is not None:
            return self._resize
        return self._original_size

    @property
    def frame_count(self) -> int:
        return self._frame_count


class RTSPVideoSource(VideoSource):
    """RTSP 스트림 소스 (추후 확장용)"""

    def __init__(
        self,
        rtsp_url: str,
        resize: Optional[Tuple[int, int]] = None,
        fps_limit: int = 30,
        reconnect_delay: float = 5.0
    ):
        self._rtsp_url = rtsp_url
        self._resize = resize
        self._fps_limit = fps_limit
        self._reconnect_delay = reconnect_delay
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time: float = 0.0

    def open(self) -> bool:
        """RTSP 스트림 연결"""
        # GStreamer 백엔드 사용 시도 (지연시간 감소)
        self._cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            logger.error(f"RTSP 스트림 연결 실패: {self._rtsp_url}")
            return False

        # 버퍼 크기 최소화 (최신 프레임만 사용)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info(f"RTSP 스트림 연결됨: {self._rtsp_url}")
        return True

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None

        min_interval = 1.0 / self._fps_limit
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("RTSP 프레임 읽기 실패, 재연결 시도...")
            self.release()
            time.sleep(self._reconnect_delay)
            self.open()
            return False, None

        self._last_frame_time = time.time()

        if self._resize is not None and frame is not None:
            frame = cv2.resize(frame, self._resize)

        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._resize is not None:
            return self._resize
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)


def create_video_source(
    source: str,
    resize: Optional[Tuple[int, int]] = None,
    fps_limit: int = 30,
    loop: bool = True
) -> VideoSource:
    """
    소스 경로에 따라 적절한 VideoSource 인스턴스 생성

    Args:
        source: 파일 경로 또는 RTSP URL
        resize: 리사이즈 크기
        fps_limit: FPS 제한
        loop: 반복 재생 (파일만)

    Returns:
        VideoSource 인스턴스
    """
    if source.startswith("rtsp://") or source.startswith("rtsps://"):
        logger.info(f"RTSP 소스 생성: {source}")
        return RTSPVideoSource(source, resize, fps_limit)
    else:
        logger.info(f"파일 소스 생성: {source}")
        return FileVideoSource(source, resize, fps_limit, loop)
