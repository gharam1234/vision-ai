# 실시간 영상 처리 및 위험구역 감지 모듈 (파트 C) - 1차 계획서

## 프로젝트 개요

산업 현장 안전 관리 시스템에서 **카메라 영상을 실시간으로 처리**하여 **작업자가 위험구역에 침입하는 것을 감지**하고, 이벤트를 백엔드 서버로 전송하는 Vision AI Server 모듈을 개발합니다.

### 기술 스택
| 구분 | 기술 |
|------|------|
| 객체 감지 | YOLOv11 (Ultralytics) |
| 객체 추적 | ByteTrack |
| 영상 처리 | OpenCV |
| GPU 가속 | CUDA (NVIDIA GPU) |
| 영상 소스 | MP4 파일 (테스트), 추후 RTSP 확장 가능 |
| 통신 | REST API + WebSocket (FastAPI 백엔드 연동) |
| 언어 | Python 3.10+ |

---

## Proposed Changes

### 프로젝트 구조

```
무제 폴더/
├── main.py                    # 엔트리포인트
├── config.py                  # 설정 관리
├── requirements.txt           # 의존성
│
├── core/
│   ├── __init__.py
│   ├── detector.py            # YOLOv11 감지기
│   ├── tracker.py             # ByteTrack 추적기
│   └── zone_manager.py        # 위험구역 관리
│
├── processing/
│   ├── __init__.py
│   ├── video_source.py        # 영상 소스 추상화
│   ├── frame_processor.py     # 프레임 처리 파이프라인
│   └── intrusion_detector.py  # 침입 감지 로직
│
├── communication/
│   ├── __init__.py
│   ├── api_client.py          # REST API 클라이언트
│   └── event_sender.py        # 이벤트 전송 (WebSocket)
│
├── visualization/
│   ├── __init__.py
│   └── renderer.py            # 시각화 렌더러
│
├── utils/
│   ├── __init__.py
│   └── geometry.py            # 기하학 유틸리티 (폴리곤 판정 등)
│
└── tests/
    └── test_intrusion.py      # 테스트
```

---

### Core 모듈

#### [NEW] [config.py](file:///Users/gharam12/antigravity/무제%20폴더/config.py)
- Pydantic BaseSettings 설정 관리

#### [NEW] [detector.py](file:///Users/gharam12/antigravity/무제%20폴더/core/detector.py)
- YOLOv11 모델 로드 및 추론 (person 클래스만 필터링)

#### [NEW] [tracker.py](file:///Users/gharam12/antigravity/무제%20폴더/core/tracker.py)
- ByteTrack 알고리즘 래핑 (supervision 라이브러리)

#### [NEW] [zone_manager.py](file:///Users/gharam12/antigravity/무제%20폴더/core/zone_manager.py)
- 위험구역 폴리곤 좌표 조회, 저장 및 주기적 API 폴링 관리

---

### Processing 모듈

#### [NEW] [video_source.py](file:///Users/gharam12/antigravity/무제%20폴더/processing/video_source.py)
- `FileVideoSource` 및 `RTSPVideoSource` 입력 추상화

#### [NEW] [frame_processor.py](file:///Users/gharam12/antigravity/무제%20폴더/processing/frame_processor.py)
- 감지 -> 추적 -> 침입 판정 -> 이벤트 전송 전체 파이프라인 조율

#### [NEW] [intrusion_detector.py](file:///Users/gharam12/antigravity/무제%20폴더/processing/intrusion_detector.py)
- 작업자 발 위치(하단 중심점) 기준 침입 판정 및 상태(ENTER/STAY/EXIT) 추적

---

### Communication 모듈

#### [NEW] [api_client.py](file:///Users/gharam12/antigravity/무제%20폴더/communication/api_client.py)
- REST API 비동기 httpx 클라이언트

#### [NEW] [event_sender.py](file:///Users/gharam12/antigravity/무제%20폴더/communication/event_sender.py)
- 별도 스레드에서 WebSocket 및 REST Fallback 전송, 스냅샷 파일 저장

---

### Visualization 모듈

#### [NEW] [renderer.py](file:///Users/gharam12/antigravity/무제%20폴더/visualization/renderer.py)
- 바운딩 박스, 위험 구역 폴리곤, 상태 정보 렌더링
