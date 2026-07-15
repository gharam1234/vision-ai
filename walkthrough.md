# 실시간 영상 처리 및 위험구역 감지 모듈 - 구현 워크스루

## 개요

산업 현장 안전 관리 시스템의 **파트 C (실시간 영상 처리 및 위험구역 감지)** 를 구현했습니다.
YOLO26 모델 패밀리 + ByteTrack 기반으로 카메라 영상에서 작업자를 실시간으로 감지/추적하고, 다양한 고정밀 탐지 기법(Segment, Pose 등)을 통해 위험구역 침입을 판별하여 백엔드 및 실시간 UI 화면으로 이벤트를 전달합니다.

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
│   ├── detector.py                 # YOLO26 사람 감지 (Detect, Pose, Seg 지원)
│   ├── tracker.py                  # ByteTrack 추적
│   └── zone_manager.py             # 위험구역 관리
│
├── processing/                     # 처리 모듈
│   ├── video_source.py             # 영상 소스 추상화
│   ├── frame_processor.py          # 파이프라인 조율
│   └── intrusion_detector.py       # 5개 침입 감지 알고리즘 로직
│
├── communication/                  # 통신 모듈
│   ├── api_client.py               # REST API 클라이언트
│   └── event_sender.py             # 이벤트 전송 (WS + REST)
│
├── visualization/
│   ├── renderer.py                 # 시각화 렌더러 (긴급 UI 연출)
│   └── drawer.py                   # 마우스 다각형 위험구역 그리기 도구
│
├── utils/
│   └── geometry.py                 # 기하학 유틸리티 (겹침 면적 연산 등)
│
└── tests/
    └── test_intrusion.py           # 단위 테스트 (17개)
```

---

## 핵심 구현 내용

### 1. 처리 파이프라인
```
프레임 입력 → YOLO26 감지 → ByteTrack 추적 → 침입 판정 → 이벤트 전송 및 긴급 UI 렌더링
```

### 2. 고정밀 침입 감지 알고리즘 구현
* **Point**: 바운딩박스 하단 중앙점으로 빠르게 포함 여부 판정
* **Multi-Point**: 바운딩박스 하단 좌/우/중앙 3지점을 모두 비교하여 경계 감지력 강화
* **Overlap**: 바운딩박스 하단 20% 면적을 직사각형으로 만들어 위험구역 다각형과의 교차 검사 (흔들림에 가장 강함)
* **Segment**: 픽셀 수준의 실루엣 마스크 하단 20% 영역과 위험구역 다각형의 마스크 겹침 검사 (고밀도 검증)
* **Pose-Hybrid**: 양쪽 발목 관절 키포인트를 기준 판정하며, 가려짐 등으로 키포인트 획득 실패 시 `Overlap` 방식으로 자동 백업(Fallback) 적용

### 3. 노이즈 필터링 및 쿨다운
* **안정성 검증**: 3프레임 연속 구역 내 존재 시 `ENTERED`(진입), 5프레임 연속 외부 존재 시 `EXITED`(이탈)를 판정해 오경보 필터링
* **쿨다운**: 동일 대상에 대핸 경보는 기본 10초의 전송 쿨다운을 적용해 불필요한 이벤트 중복 전송 방지

### 4. 강화된 실시간 경보 UI
* 침입 이벤트 감지 즉시 전체 화면 외곽에 빨간색 경보 라인이 0.3초 주기로 점멸(Blinking)
* 화면 상단 중앙에 고대비 긴급 경보 패널(`[ ⚠️ EMERGENCY ALARM ]`)을 노출하고 침입 중인 대상 정보를 텍스트로 오버레이

---

## 실행 방법

### 드로잉 모드로 오버랩 침입 감지 실행 (추천)
```bash
python main.py --source sample.mp4 --show --draw --method overlap
```

### 포즈 기반 침입 감지 실행 (자동 yolo26m-pose.pt 로드)
```bash
python main.py --source sample.mp4 --show --draw --method pose-hybrid
```

### CLI 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source`, `-s` | 영상 소스 (파일 경로/RTSP URL) | sample.mp4 |
| `--model`, `-m` | YOLO 모델 경로 | yolo26s.pt |
| `--show`, `-d` | 시각화 디스플레이 표시 | false |
| `--no-show` | 헤드리스 모드 | - |
| `--draw` | 마우스 다각형 위험구역 그리기 활성화 | false |
| `--confidence`, `-c` | 감지 신뢰도 임계값 | 0.35 |
| `--method` | 침입 감지 방법 (`point`/`multi-point`/`overlap`/`segment`/`pose-hybrid`) | point |
| `--camera-id` | 카메라 식별자 | cam-001 |
| `--api-url` | 백엔드 API URL | http://localhost:8000 |

---

## 테스트 결과

```
tests/test_intrusion.py - 17 passed ✅
```

| 카테고리 | 테스트 수 | 상태 |
|---------|---------|------|
| 기하학 유틸리티 (geometry) | 9 | ✅ 통과 |
| 침입 감지 (intrusion) | 8 | ✅ 통과 |

### 검증 시나리오
* 마스크 겹침 비트연산 및 바운딩박스 IoU 수학적 검증
* 다중 객체 독립 ID 보존 추적 검증
* 쿨다운 내 중복 전송 차단 검증
