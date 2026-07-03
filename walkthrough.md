# 실시간 영상 처리 및 위험구역 감지 모듈 - 구현 워크스루

## 개요

산업 현장 안전 관리 시스템의 **파트 C (실시간 영상 처리 및 위험구역 감지)** 를 구현했습니다.
YOLOv11 + ByteTrack 기반으로 카메라 영상에서 작업자를 감지/추적하고, 위험구역 침입을 판별하여 백엔드로 이벤트를 전송합니다.

---

## 프로젝트 구조

```
무제 폴더/
├── main.py                         # 엔트리포인트 (CLI, 메인 루프)
├── config.py                       # Pydantic 기반 설정 관리
├── requirements.txt                # 의존성
├── .env.example                    # 환경변수 예시
│
├── core/                           # 핵심 모듈
│   ├── detector.py                 # YOLOv11 사람 감지
│   ├── tracker.py                  # ByteTrack 추적
│   └── zone_manager.py             # 위험구역 관리
│
├── processing/                     # 처리 모듈
│   ├── video_source.py             # 영상 소스 추상화
│   ├── frame_processor.py          # 파이프라인 조율
│   └── intrusion_detector.py       # 침입 감지 로직
│
├── communication/                  # 통신 모듈
│   ├── api_client.py               # REST API 클라이언트
│   └── event_sender.py             # 이벤트 전송 (WS + REST)
│
├── visualization/
│   └── renderer.py                 # 시각화 렌더러
│
├── utils/
│   └── geometry.py                 # 기하학 유틸리티
│
└── tests/
    └── test_intrusion.py           # 단위 테스트 (17개)
```

---

## 핵심 구현 내용

### 1. 처리 파이프라인
```
프레임 입력 → YOLOv11 감지 → ByteTrack 추적 → 침입 판정 → 이벤트 전송 → 시각화
```

### 2. 침입 감지 로직
- 작업자 **바운딩박스 하단 중심점** (발 위치 근사)으로 위험구역 침입 판정
- `cv2.pointPolygonTest()`로 폴리곤 내부 여부 판정
- **연속 프레임 기반 노이즈 필터링**: 3프레임 연속 내부 → ENTERED, 5프레임 연속 외부 → EXITED
- **쿨다운 메커니즘**: 같은 (작업자, 구역) 조합의 이벤트 중복 전송 방지

### 3. 이벤트 상태
| 상태 | 설명 |
|------|------|
| `ENTERED` | 작업자가 위험구역에 방금 진입 |
| `STAYING` | 작업자가 위험구역에 체류 중 (주기적 전송) |
| `EXITED` | 작업자가 위험구역에서 이탈 |

### 4. 통신 구조
- **WebSocket**: 침입 이벤트 실시간 전송 (1차)
- **REST API**: WebSocket 실패 시 fallback 전송
- **별도 스레드**: 이벤트 전송이 메인 루프를 블로킹하지 않음

---

## 실행 방법

### 기본 실행 (데모 모드)
```bash
python main.py --source sample.mp4 --show --demo
```

### CLI 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source`, `-s` | 영상 소스 (파일 경로/RTSP URL) | sample.mp4 |
| `--model`, `-m` | YOLO 모델 경로 | yolo11n.pt |
| `--show`, `-d` | 시각화 디스플레이 표시 | false |
| `--no-show` | 헤드리스 모드 | - |
| `--confidence`, `-c` | 감지 신뢰도 임계값 | 0.5 |
| `--demo` | 데모 위험구역 자동 생성 | false |
| `--camera-id` | 카메라 식별자 | cam-001 |
| `--api-url` | 백엔드 API URL | http://localhost:8000 |
| `--log-level` | 로그 레벨 | INFO |

### 키보드 단축키 (디스플레이 모드)
| 키 | 기능 |
|----|------|
| `q` | 프로그램 종료 |
| `r` | 추적기/침입감지기 리셋 |
| `d` | 데모 위험구역 토글 (생성/제거) |

---

## 테스트 결과

```
tests/test_intrusion.py - 17 passed ✅
```

| 카테고리 | 테스트 수 | 상태 |
|---------|---------|------|
| 기하학 유틸리티 (geometry) | 9 | ✅ 통과 |
| 침입 감지 (intrusion) | 8 | ✅ 통과 |

### 검증된 시나리오
- 폴리곤 내부/외부/경계 판정
- IoU 계산 (겹침/비겹침/동일)
- 위험구역 밖 작업자 → 이벤트 없음
- 연속 프레임 임계값 후 ENTERED 이벤트
- 이탈 후 EXITED 이벤트
- 다수 작업자 독립 추적
- 다수 위험구역 동시 감지
- 쿨다운 중복 방지
- 활성 침입 상태 조회 및 리셋

---

## 다음 단계

> [!NOTE]
> 현재 테스트 영상 파일(sample.mp4)이 없으므로, 실제 실행을 위해서는:
> 1. 사람이 포함된 테스트 MP4 영상 파일을 `sample.mp4`로 프로젝트 루트에 배치
> 2. `python main.py --source sample.mp4 --show --demo` 실행
> 3. YOLO 모델이 자동으로 다운로드됩니다 (최초 실행 시)

### 백엔드 팀(E)과 협업 사항
- FastAPI 백엔드의 위험구역 API 엔드포인트 구현 (`GET /api/cameras/{id}/zones`)
- WebSocket 이벤트 수신 엔드포인트 구현 (`ws://host/ws/events`)
- 이벤트 데이터 스키마 맞춤

### 프론트엔드 팀(E)과 협업 사항
- React 대시보드에서 위험구역 폴리곤 좌표를 API로 저장
- 실시간 침입 알림 표시
