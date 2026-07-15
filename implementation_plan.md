# 실시간 고정밀 위험구역 침입 감지 시스템 설계 계획

## 프로젝트 개요

산업 현장의 안전 강화를 위해 기존 객체 감지 시스템의 핵심 모듈을 최신 **YOLO26** 모델군으로 업그레이드하고, 오경보율을 획득 디바이스와 환경에 맞추어 최소화할 수 있는 **5가지 침입 감지 알고리즘(Point, Multi-Point, Overlap, Segment, Pose-Hybrid)** 및 현장 위험구역의 동적 지정을 위한 **대화형 마우스 드로잉 UI**를 구현합니다.

---

## User Review Required

> [!IMPORTANT]
> **디바이스 성능별 권장 모델 매핑**:
> * 시스템 실행 시 선택된 `--method`에 따라 가장 적합한 YOLO26 모델 스펙이 자동으로 바인딩되도록 구성할 예정입니다.
>   * `point` / `multi-point` / `overlap` $\rightarrow$ **`yolo26s.pt`** (CPU 가볍고 빠른 처리 우선)
>   * `segment` $\rightarrow$ **`yolo26m-seg.pt`** (정밀 실루엣 매칭용, 메모리 약 54MB)
>   * `pose-hybrid` $\rightarrow$ **`yolo26m-pose.pt`** (정밀 관절 추적용, 메모리 약 49MB)
> * 만약 GPU 가속을 사용하지 않고 CPU만 사용하는 하드웨어 환경이라면 `segment`나 `pose-hybrid` 대신 **`overlap` 방식** 사용을 권장합니다.

---

## Proposed Changes

### 1. AI 핵심 모듈 (`core/`)

#### [MODIFY] [detector.py](file:///Users/gharam12/antigravity/무제%20폴더/core/detector.py)
* YOLO26 가중치 모델 로드 로직 구현 (`yolo26s.pt`, `yolo26m-pose.pt`, `yolo26m-seg.pt` 등 동적 바인딩)
* 디텍션 결과(바운딩 박스), 관절 키포인트(Pose Keypoints), 마스크 실루엣(Segment Mask)을 통일된 데이터 구조로 파싱하여 반환하는 인터페이스 구축

#### [MODIFY] [tracker.py](file:///Users/gharam12/antigravity/무제%20폴더/core/tracker.py)
* ByteTrack 엔진을 연동하여 프레임 간 감지된 대상에 추적 ID를 고유하게 보존

#### [MODIFY] [zone_manager.py](file:///Users/gharam12/antigravity/무제%20폴더/core/zone_manager.py)
* 메모리 기반으로 다각형(Polygon) 형태의 위험구역 데이터를 로컬에서 관리 및 백엔드 API 연동

### 2. 데이터 처리 및 판정 파이프라인 (`processing/`)

#### [MODIFY] [intrusion_detector.py](file:///Users/gharam12/antigravity/무제%20폴더/processing/intrusion_detector.py)
* 5가지 판단 메서드 추가 및 구현:
  * `point`: 작업자 바운딩박스 하단 중앙점 기준 판정
  * `multi-point`: 하단 좌/우/중앙 3지점 중 하나가 구역 내부에 속하는지 판정
  * `overlap`: 바운딩박스 하단 20% 영역의 직사각형 마스크와 위험구역의 교차 여부 판정
  * `segment`: 감지된 실루엣 마스크의 하부 20% 픽셀과 위험구역의 픽셀 단위 교차 여부 판정
  * `pose-hybrid`: 감지된 양쪽 발목 관절 좌표(15, 16번)가 구역 내부인지 판정 (포즈 누락 시 `overlap` 모드로 폴백)
* 연속 프레임 진입 검증 및 이벤트 중복 방지 쿨다운 타이머 처리

### 3. UI 및 시각화 모듈 (`visualization/`)

#### [NEW] [drawer.py](file:///Users/gharam12/antigravity/무제%20폴더/visualization/drawer.py)
* OpenCV 마우스 콜백(`cv2.setMouseCallback`)을 활용해 첫 번째 프레임 위에 사용자가 직접 점들을 찍어 다각형 위험구역을 생성할 수 있는 드로잉 모드 UI 추가

#### [MODIFY] [renderer.py](file:///Users/gharam12/antigravity/무제%20폴더/visualization/renderer.py)
* 감지 박스 정보뿐만 아니라 침입이 발생한 경우 화면 외곽에 빨간색 깜빡이 경보 테두리 및 화면 중앙 고대비 긴급 안내 배너 렌더링

---

## Verification Plan

### Automated Tests
* `utils/geometry.py` 및 `processing/intrusion_detector.py`에 관한 정밀 충돌 테스트 및 시나리오 검증 단위 테스트 작성
* **실행 명령어**:
  ```bash
  pytest tests/test_intrusion.py
  ```

### Manual Verification
* **실행 명령어**:
  ```bash
  # 오버랩 방식 기반 드로잉 모드로 동영상 분석 시작
  python main.py --source sample.mp4 --show --draw --method overlap
  ```
* **수동 체크리스트**:
  1. 프로그램 실행 직후 마우스 클릭으로 다각형을 그리고 `S`키를 누를 때 감지가 시작되는지 확인
  2. 작업자가 그린 다각형 경계에 걸쳤을 때 `overlap` 방식에 따라 즉시 테두리 경보창 및 로그가 정상 인지되는지 검증
  3. 백그라운드 스레드에서 생성된 스냅샷 폴더(`snapshots/`)에 침입 캡처본이 저장되는지 확인
