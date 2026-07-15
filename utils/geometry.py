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


def get_multi_bottom_points(bbox: Tuple[float, float, float, float]) -> list[Tuple[int, int]]:
    """
    바운딩박스의 하단 3개 지점(좌측 20% 지점, 하단 중앙, 우측 20% 지점) 계산

    Args:
        bbox: (x1, y1, x2, y2) 바운딩박스 좌표

    Returns:
        (x, y) 튜플의 리스트 (3개 점)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    y = int(y2)
    pt_left = (int(x1 + w * 0.2), y)
    pt_center = (int((x1 + x2) / 2), y)
    pt_right = (int(x1 + w * 0.8), y)
    return [pt_left, pt_center, pt_right]


def check_polygon_overlap_mask(
    bbox: Tuple[float, float, float, float],
    polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
    overlap_ratio: float = 0.2
) -> bool:
    """
    바운딩박스의 하부 영역과 위험구역 다각형의 비트마스크 겹침 여부 판정

    Args:
        bbox: (x1, y1, x2, y2) 바운딩박스 좌표
        polygon: 위험구역 다각형 좌표
        frame_width: 프레임 너비
        frame_height: 프레임 높이
        overlap_ratio: 바운딩박스 하부에서 발 영역으로 간주할 비율 (0.0~1.0)

    Returns:
        True이면 두 영역이 겹침
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1

    # 하부 영역 높이 계산 및 y 범위 제한
    y_start = max(y1, y2 - h * overlap_ratio)

    # 발 영역 사각형 꼭짓점
    foot_pts = np.array([
        [x1, y_start],
        [x2, y_start],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    # 1. 프레임 크기 마스크 이미지 준비
    mask_zone = np.zeros((frame_height, frame_width), dtype=np.uint8)
    mask_foot = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # 2. 다각형 그리기 (위험구역 다각형은 float32일 수 있으므로 int32 변환)
    zone_pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_zone, [zone_pts], 255)

    # 3. 발 영역 그리기
    cv2.fillPoly(mask_foot, [foot_pts], 255)

    # 4. AND 연산 수행
    overlap = cv2.bitwise_and(mask_zone, mask_foot)

    # 5. 겹치는 픽셀 수 계산 (1픽셀 이상이면 겹치는 것으로 판단)
    return cv2.countNonZero(overlap) > 0


def check_segment_overlap_mask(
    mask: np.ndarray,
    polygon: np.ndarray,
    overlap_ratio: float = 0.2
) -> bool:
    """
    사람 실루엣 마스크의 하부 영역과 위험구역 다각형의 비트마스크 겹침 여부 판정

    Args:
        mask: 사람 실루엣 2D 마스크 배열 (shape: H, W)
        polygon: 위험구역 다각형 좌표
        overlap_ratio: 실루엣 하부에서 발 영역으로 간주할 비율 (0.0~1.0)

    Returns:
        True이면 겹치는 픽셀이 존재함
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return False

    h_img, w_img = mask.shape[:2]

    # 마스크 픽셀의 y 좌표 추출
    y_indices, _ = np.where(mask > 0)
    if len(y_indices) == 0:
        return False

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    h_silhouette = y_max - y_min
    if h_silhouette <= 0:
        return False

    # 하부 영역 시작 y 계산
    y_start = y_max - h_silhouette * overlap_ratio

    # 1. 발 영역 마스크 생성
    # y좌표 그리드를 만들어서 비교
    y_grid = np.arange(h_img).reshape(-1, 1)
    # mask가 True이고 y좌표가 y_start 이상인 영역 필터링
    mask_foot = (mask > 0) & (y_grid >= y_start)

    # 2. 위험구역 다각형 마스크 생성
    mask_zone = np.zeros((h_img, w_img), dtype=np.uint8)
    zone_pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_zone, [zone_pts], 255)

    # 3. AND 연산 수행
    overlap = cv2.bitwise_and(mask_zone, mask_foot.astype(np.uint8) * 255)

    return cv2.countNonZero(overlap) > 0


