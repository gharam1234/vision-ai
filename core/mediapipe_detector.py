"""
MediaPipe를 활용한 관절 및 손가락 랜드마크 추출 모듈
(YOLO로 잡힌 사람 영역의 ROI를 크롭받아 고속 추론)
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
from loguru import logger


class MediaPipeDetector:
    """YOLO crop 이미지 기반의 MediaPipe Pose & Hands 감지기"""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        # 1. Pose 모델 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy (CPU 가동 속도와 정합성을 위해 1 채택)
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # 2. Hands 모델 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info("MediaPipe Detector (solutions.pose/hands) 초기화 완료 (하이브리드)")

    def detect_pose(
        self,
        crop_img: np.ndarray,
        crop_offset: Tuple[int, int],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """
        크롭된 이미지 영역에서 33개 Pose 관절 랜드마크를 추출하고 전역 좌표계로 변환하여 반환
        """
        if crop_img is None or crop_img.size == 0:
            return None

        h_crop, w_crop = crop_img.shape[:2]
        x_offset, y_offset = crop_offset

        # BGR -> RGB 변환
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.pose.process(img_rgb)
            if results.pose_landmarks:
                landmarks_list = results.pose_landmarks.landmark
                
                # [필터링 고도화]: 크롭 이미지 내에서 발견된 관절 중 핵심 관절(코:0, 왼어깨:11, 오른어깨:12, 왼골반:23, 오른골반:24)의
                # 신뢰도가 0.35 이상인 유효 관절이 최소 2개 이상 잡혔는지 체크하여 유령 뼈대선 차단
                valid_core_count = 0
                core_indices = [0, 11, 12, 23, 24]
                for idx in core_indices:
                    if idx < len(landmarks_list):
                        if getattr(landmarks_list[idx], 'visibility', 0.0) >= 0.35:
                            valid_core_count += 1
                            
                # 유의미한 인간 형태로 보기 어려운 유령 뼈대선인 경우 무효화
                if valid_core_count < 2:
                    return None

                pose_kps = np.zeros((33, 3), dtype=np.float32)
                for idx, lm in enumerate(landmarks_list):
                    # 크롭 비율 좌표를 픽셀 단위로 복원하고 전역 오프셋 합산
                    gx = x_offset + (lm.x * w_crop)
                    gy = y_offset + (lm.y * h_crop)
                    visibility = getattr(lm, 'visibility', 0.9)
                    pose_kps[idx] = [gx, gy, visibility]
                return pose_kps
        except Exception as e:
            logger.error(f"MediaPipe Pose 검출 중 에러: {e}")

        return None

    def detect_hands(
        self,
        crop_img: np.ndarray,
        crop_offset: Tuple[int, int]
    ) -> List[Dict]:
        """
        크롭된 이미지 영역에서 양손(최대 2개)을 검출하고 전역 좌표계로 변환하여 반환
        """
        detected_hands = []
        if crop_img is None or crop_img.size == 0:
            return detected_hands

        h_crop, w_crop = crop_img.shape[:2]
        x_offset, y_offset = crop_offset

        # BGR -> RGB 변환
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        try:
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    score = handedness.classification[0].score

                    hand_kps = np.zeros((21, 3), dtype=np.float32)
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        gx = x_offset + (lm.x * w_crop)
                        gy = y_offset + (lm.y * h_crop)
                        hand_kps[idx] = [gx, gy, score]

                    detected_hands.append({
                        "label": label,
                        "score": score,
                        "landmarks": hand_kps
                    })
        except Exception as e:
            logger.error(f"MediaPipe Hands 검출 중 에러: {e}")

        return detected_hands

    def close(self):
        """리소스 정리"""
        try:
            self.pose.close()
            self.hands.close()
            logger.info("MediaPipe Detector 리소스 정리 완료")
        except Exception:
            pass
