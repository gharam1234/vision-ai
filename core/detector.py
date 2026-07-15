"""
YOLO26 기반 사람(작업자) 감지 모듈
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """감지 결과 데이터 클래스"""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    keypoints: Optional[np.ndarray] = None    # 포즈 키포인트 (17, 3) (x, y, conf)


class PersonDetector:
    """YOLO26 기반 사람 감지기"""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        img_size: int = 640,
        target_classes: Optional[list[int]] = None
    ):
        """
        Args:
            model_path: YOLO 모델 파일 경로
            confidence_threshold: 감지 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            device: 추론 디바이스 ("auto", "cpu", "cuda", "0")
            img_size: 추론 이미지 크기
            target_classes: 감지 대상 클래스 ID 리스트 (기본: [0] = person)
        """
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._img_size = img_size
        self._target_classes = target_classes or [0]  # 0 = person

        # 디바이스 결정
        if device == "auto":
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # 모델 로드
        logger.info(f"YOLO 모델 로딩: {model_path} (디바이스: {self._device})")
        self._model = YOLO(model_path)

        logger.info(
            f"PersonDetector 초기화 완료 "
            f"(confidence={confidence_threshold}, "
            f"iou={iou_threshold}, "
            f"img_size={img_size})"
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        프레임에서 사람(작업자)을 감지

        Args:
            frame: BGR 이미지 (numpy 배열)

        Returns:
            Detection 리스트
        """
        # YOLO 추론 실행
        results = self._model(
            frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._img_size,
            device=self._device,
            classes=self._target_classes,
            verbose=False
        )

        detections: list[Detection] = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self._model.names[cls_id]

                    keypoints = None
                    if hasattr(result, "keypoints") and result.keypoints is not None:
                        if len(result.keypoints.data) > i:
                            keypoints = result.keypoints.data[i].cpu().numpy()

                    detections.append(Detection(
                        bbox=(float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        keypoints=keypoints
                    ))

        return detections

    def detect_raw(self, frame: np.ndarray):
        """
        YOLO 원본 결과를 반환 (ByteTrack 연동용)

        Args:
            frame: BGR 이미지

        Returns:
            Ultralytics Results 객체
        """
        results = self._model(
            frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._img_size,
            device=self._device,
            classes=self._target_classes,
            verbose=False
        )
        return results[0] if results else None

    @property
    def model_name(self) -> str:
        return self._model.model_name if hasattr(self._model, 'model_name') else "YOLO26"

    @property
    def device(self) -> str:
        return self._device
