"""
기하학 유틸리티 함수
폴리곤 내부 판정, 좌표 변환 등
"""

import numpy as np
import cv2
from typing import Tuple


def get_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    바운딩박스의 하단 중심점 계산
    작업자의 발 위치를 근사하여 위험구역 침입 판정에 사용

    Args:
        bbox: (x1, y1, x2, y2) 바운딩박스 좌표

    Returns:
        (cx, cy) 하단 중심점 좌표
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2)  # 하단 = y2
    return (cx, cy)


def is_point_in_polygon(
    point: Tuple[int, int],
    polygon: np.ndarray
) -> bool:
    """
    점이 폴리곤 내부에 있는지 판정

    Args:
        point: (x, y) 판정할 점의 좌표
        polygon: numpy 배열 형태의 폴리곤 좌표 [(x1,y1), (x2,y2), ...]

    Returns:
        True이면 폴리곤 내부에 있음
    """
    polygon_contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
    result = cv2.pointPolygonTest(polygon_contour, point, False)
    return result >= 0  # >= 0 이면 내부 또는 경계


def polygon_from_points(points: list[list[float]]) -> np.ndarray:
    """
    좌표 리스트를 numpy 폴리곤 배열로 변환

    Args:
        points: [[x1,y1], [x2,y2], ...] 형태의 좌표 리스트

    Returns:
        numpy ndarray 형태의 폴리곤
    """
    return np.array(points, dtype=np.float32)


def normalize_polygon(
    polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
    is_normalized: bool = False
) -> np.ndarray:
    """
    정규화된 좌표(0~1)를 실제 픽셀 좌표로 변환하거나, 그대로 반환

    Args:
        polygon: 폴리곤 좌표 배열
        frame_width: 프레임 너비
        frame_height: 프레임 높이
        is_normalized: True이면 0~1 정규화 좌표, False이면 픽셀 좌표

    Returns:
        픽셀 좌표 형태의 폴리곤
    """
    if is_normalized:
        scaled = polygon.copy()
        scaled[:, 0] *= frame_width
        scaled[:, 1] *= frame_height
        return scaled.astype(np.int32)
    return polygon.astype(np.int32)


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """
    두 바운딩박스의 IoU(Intersection over Union) 계산

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union
