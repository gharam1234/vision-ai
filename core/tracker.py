"""
ByteTrack 기반 작업자 추적 모듈
supervision 라이브러리를 활용하여 프레임 간 동일 작업자를 추적
"""

import numpy as np
import supervision as sv
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from utils.geometry import calculate_iou


@dataclass
class TrackedObject:
    """추적된 객체 데이터 클래스"""
    tracker_id: int                           # 고유 추적 ID
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None         # 세그멘테이션 마스크 (사람 형태 2D 마스크)
    keypoints: Optional[np.ndarray] = None    # 포즈 키포인트 (17, 3) (x, y, conf)
    hand_landmarks: Optional[list] = None     # MediaPipe Hands 랜드마크 정보 리스트



class PersonTracker:
    """ByteTrack 기반 사람 추적기"""

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8
    ):
        """
        Args:
            track_thresh: 추적에 사용할 최소 감지 신뢰도
            track_buffer: 추적 유지 버퍼 (프레임 수, 이 프레임 동안 미감지 시 추적 종료)
            match_thresh: IoU 매칭 임계값
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=30
        )

        logger.info(
            f"PersonTracker 초기화 완료 "
            f"(track_thresh={track_thresh}, "
            f"buffer={track_buffer}, "
            f"match_thresh={match_thresh})"
        )

    def update(self, yolo_result) -> list[TrackedObject]:
        """
        YOLO 감지 결과를 받아 추적 ID를 부여

        Args:
            yolo_result: Ultralytics YOLO Results 객체

        Returns:
            TrackedObject 리스트 (추적 ID 포함)
        """
        if yolo_result is None or yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            # 감지 없음 - 빈 Detections 으로 업데이트하여 기존 추적 유지/종료
            empty_detections = sv.Detections.empty()
            self._tracker.update_with_detections(empty_detections)
            return []

        # Ultralytics Results → supervision Detections 변환
        sv_detections = sv.Detections.from_ultralytics(yolo_result)

        # ByteTrack 업데이트
        tracked_detections = self._tracker.update_with_detections(sv_detections)

        # TrackedObject 리스트 생성
        tracked_objects: list[TrackedObject] = []

        if tracked_detections.tracker_id is not None:
            names = yolo_result.names  # class_id → class_name 매핑

            # 원본 감지 bbox 수집
            original_boxes = []
            if yolo_result is not None and yolo_result.boxes is not None:
                original_boxes = yolo_result.boxes.xyxy.cpu().numpy()

            for i in range(len(tracked_detections)):
                bbox = tracked_detections.xyxy[i]
                confidence = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 0.0
                class_id = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
                tracker_id = int(tracked_detections.tracker_id[i])
                class_name = names.get(class_id, "unknown")

                # IoU 매칭을 통해 원래 YOLO 검출 결과 j 매핑
                best_j = -1
                best_iou = -1.0
                for j in range(len(original_boxes)):
                    iou = calculate_iou(
                        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        (float(original_boxes[j][0]), float(original_boxes[j][1]), float(original_boxes[j][2]), float(original_boxes[j][3]))
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                mask = None
                keypoints = None
                if best_j != -1 and best_iou >= 0.5:
                    if hasattr(yolo_result, "masks") and yolo_result.masks is not None:
                        mask = yolo_result.masks.data[best_j].cpu().numpy()
                    if hasattr(yolo_result, "keypoints") and yolo_result.keypoints is not None:
                        keypoints = yolo_result.keypoints.data[best_j].cpu().numpy()

                tracked_objects.append(TrackedObject(
                    tracker_id=tracker_id,
                    bbox=(float(bbox[0]), float(bbox[1]),
                          float(bbox[2]), float(bbox[3])),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                    keypoints=keypoints,
                    hand_landmarks=None
                ))

        return tracked_objects

    def reset(self) -> None:
        """추적기 상태 초기화"""
        self._tracker.reset()
        logger.info("추적기 상태 초기화됨")
