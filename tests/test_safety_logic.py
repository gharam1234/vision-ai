import numpy as np
import pytest
from processing.intrusion_detector import IntrusionDetector, _TrackerZoneState

# MediaPipe Pose 33개 관절 데이터 모의 생성
def create_mediapipe_skeleton(
    l_shoulder=(280, 150, 0.9),
    r_shoulder=(200, 150, 0.9),
    l_wrist=(280, 250, 0.9),
    r_wrist=(200, 250, 0.9),
    l_hip=(270, 300, 0.9),
    r_hip=(210, 300, 0.9),
    nose=(240, 80, 0.9)
):
    keypoints = np.zeros((33, 3), dtype=np.float32)
    # Nose: 0
    keypoints[0] = nose
    # L_Shoulder: 11, R_Shoulder: 12
    keypoints[11] = l_shoulder
    keypoints[12] = r_shoulder
    # L_Wrist: 15, R_Wrist: 16
    keypoints[15] = l_wrist
    keypoints[16] = r_wrist
    # L_Hip: 23, R_Hip: 24
    keypoints[23] = l_hip
    keypoints[24] = r_hip
    return keypoints

def test_check_normal_behavior():
    """정상 행동 테스트"""
    detector = IntrusionDetector(pose_conf_threshold=0.3)
    kps = create_mediapipe_skeleton()
    
    assert not detector._check_arm_cross(kps)
    assert not detector._check_fall_down((200, 80, 280, 450), kps)

def test_check_arm_cross():
    """양팔 교차(X자 제스처) 검증"""
    detector = IntrusionDetector(pose_conf_threshold=0.3)
    
    # 양 손목 X좌표 교차 (L_Wrist=230, R_Wrist=250)
    # Y좌표는 어깨(150)와 골반(300) 사이
    kps_cross = create_mediapipe_skeleton(
        l_wrist=(230, 220, 0.9),
        r_wrist=(250, 220, 0.9)
    )
    
    assert detector._check_arm_cross(kps_cross)

def test_check_arm_waving():
    """도움 요청 손 흔들기 검증"""
    detector = IntrusionDetector(pose_conf_threshold=0.3)
    state = _TrackerZoneState()
    
    kps_pose = create_mediapipe_skeleton() # 어깨 높이 150
    
    # 1. 손목 Y좌표가 어깨(150)보다 낮은 경우 (Y=200) -> 감지 안 됨
    hand_low = [{
        "label": "Left",
        "score": 0.9,
        "landmarks": np.array([[200, 200, 0.9]] * 21, dtype=np.float32)
    }]
    assert not detector._check_arm_waving(hand_low, kps_pose, state)
    
    # 2. 손목 Y좌표가 어깨보다 높고(Y=70), X좌표가 흔들리는 궤적
    x_trajectory = [100, 125, 100, 125, 100, 125, 100, 125, 100, 125]
    for x in x_trajectory:
        hand_wave = [{
            "label": "Left",
            "score": 0.9,
            "landmarks": np.array([[x, 70, 0.9]] * 21, dtype=np.float32)
        }]
        result = detector._check_arm_waving(hand_wave, kps_pose, state)
        
    assert result

def test_check_fall_down():
    """쓰러짐 검증"""
    detector = IntrusionDetector(pose_conf_threshold=0.3)
    
    # A. 종횡비가 가로로 길고 + 실제로 척추가 누워있을 때
    bbox_horizontal = (100, 300, 400, 420)
    kps_lying = create_mediapipe_skeleton(
        l_shoulder=(150, 310, 0.9),
        r_shoulder=(150, 330, 0.9),
        l_hip=(250, 330, 0.9),
        r_hip=(250, 350, 0.9)
    )
    assert detector._check_fall_down(bbox_horizontal, kps_lying)
    
    # B. 애매한 종횡비지만 척추 축이 누워 있을 때
    bbox_ambiguous = (100, 200, 220, 330)
    # 어깨 (150, 220), 골반 (250, 240)
    kps_horizontal = create_mediapipe_skeleton(
        l_shoulder=(150, 210, 0.9),
        r_shoulder=(150, 230, 0.9),
        l_hip=(250, 230, 0.9),
        r_hip=(250, 250, 0.9)
    )
    assert detector._check_fall_down(bbox_ambiguous, kps_horizontal)


def test_skeleton_line_intersection():
    """뼈대 선분이 위험구역과 교차하는 침입 시나리오 검증"""
    detector = IntrusionDetector(pose_conf_threshold=0.3)
    
    # 어깨 (150, 100) 와 골반 (150, 300) 은 둘 다 polygon 내부에 없음!
    # 하지만 뼈대선은 polygon (y: 150~250) 을 수직 관통함.
    polygon = np.array([(140, 150), (160, 150), (160, 250), (140, 250)], dtype=np.int32)
    A = (150, 100)
    B = (150, 300)
    
    assert detector._is_skeleton_intersecting_zone(A, B, polygon)
