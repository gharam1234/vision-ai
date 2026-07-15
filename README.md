# 🎯 Safety Vision AI

실시간 CCTV 영상에서 **작업자 위험구역 침입**, **쓰러짐(낙상)**, **팔짱(X자 자세)**, **손 흔들기(도움 요청)** 를 AI로 자동 감지하는 산업 안전 관제 시스템입니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| 🚧 **위험구역 침입 감지** | 마우스로 위험구역을 그리면, 작업자의 관절/뼈대선/바운딩박스가 구역에 닿는 즉시 경보 |
| 🧍 **쓰러짐(낙상) 감지** | 척추 기울기 + 바운딩박스 종횡비로 쓰러진 자세를 실시간 판별 |
| ❌ **팔짱(X자) 감지** | 양 손목이 반대쪽 어깨를 교차하는 자세를 감지 |
| 👋 **손 흔들기 감지** | 작업자의 반복적인 손 흔들기(도움 요청 제스처)를 감지 |
| 📡 **실시간 이벤트 전송** | WebSocket/REST API로 백엔드에 이벤트 및 스냅샷 전송 |

## 기술 스택

- **객체 감지**: YOLO26s (Ultralytics)
- **포즈 추정**: MediaPipe Pose/Hands
- **객체 추적**: ByteTrack (Supervision)
- **영상 처리**: OpenCV
- **설정 관리**: Pydantic Settings + dotenv

## 프로젝트 구조

```
├── main.py                  # 엔트리포인트
├── config.py                # Pydantic 기반 설정 관리
├── core/                    # 핵심 엔진
│   ├── detector.py          # YOLO 사람 감지기
│   ├── mediapipe_detector.py # MediaPipe Pose/Hands 추출기
│   ├── tracker.py           # ByteTrack 추적기
│   └── zone_manager.py      # 위험구역 관리자
├── processing/              # 처리 파이프라인
│   ├── frame_processor.py   # 프레임 처리 파이프라인
│   ├── intrusion_detector.py # 침입/제스처/쓰러짐 감지 엔진
│   └── video_source.py      # 영상 소스 추상화
├── visualization/           # 시각화
│   ├── renderer.py          # 화면 렌더러
│   └── drawer.py            # 마우스 위험구역 드로잉 도구
├── communication/           # 통신
│   ├── api_client.py        # REST API 클라이언트
│   └── event_sender.py      # 이벤트 전송기 (WS + REST)
├── utils/                   # 유틸리티
│   └── geometry.py          # 기하학 연산 (점-다각형, 선분 교차 등)
├── tests/                   # 단위 테스트
│   ├── test_intrusion.py    # 침입 감지 테스트 (26개)
│   └── test_safety_logic.py # 제스처/쓰러짐 테스트 (5개)
├── .env.example             # 환경변수 예시
├── requirements.txt         # 의존성 목록
└── api_specification.md     # 백엔드 API 사양서
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

> ⚠️ **macOS (M1/M2/M3)** 사용 시 MediaPipe는 반드시 `0.10.14` 버전을 설치하세요:
> ```bash
> pip install mediapipe==0.10.14
> ```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 값을 수정하세요
```

### 3. YOLO 모델 다운로드

첫 실행 시 `yolo26s.pt` 모델이 자동으로 다운로드됩니다. 수동 다운로드가 필요한 경우:

```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")
```

### 4. 실행

```bash
# 비디오 파일로 실행 (위험구역 마우스 드로잉)
python main.py --source sample.mp4 --draw

# 웹캠으로 실행
python main.py --source 0 --draw

# 데모 모드 (기본 위험구역 자동 생성)
python main.py --source sample.mp4 --demo
```

### 5. 테스트

```bash
pytest tests/ -v
```

## 위험구역 설정 방법

`--draw` 옵션으로 실행하면 첫 프레임에서 마우스로 위험구역을 설정할 수 있습니다:

1. **마우스 왼쪽 클릭** — 다각형 꼭짓점 찍기
2. **Enter / Space** — 다각형 확정
3. **숫자 키(1~4)** — 위험 레벨 선택 (Low / Medium / High / Critical)
4. **ESC** — 설정 완료 후 감지 시작

## 주요 설정 (.env)

```env
# YOLO 감지
DETECTOR_MODEL_PATH=yolo26s.pt
DETECTOR_CONFIDENCE_THRESHOLD=0.3

# 침입 감지 방식
EVENT_INTRUSION_METHOD=pose-hybrid

# 감도 튜닝
EVENT_ENTER_THRESHOLD_FRAMES=3     # 침입 판정 연속 프레임
EVENT_EXIT_THRESHOLD_FRAMES=5      # 이탈 판정 연속 프레임
EVENT_COOLDOWN_SECONDS=10.0        # 동일 이벤트 재알림 간격
```

## 라이선스

이 프로젝트는 내부용입니다.
