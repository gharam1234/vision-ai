# 🔌 백엔드 연동 API 명세서 (API Specification)

본 문서는 실시간 영상 처리 및 위험구역 감지 시스템(클라이언트)과 FastAPI 백엔드 서버 간의 통신 프로토콜 규격을 정의합니다.

---

## 📌 1. 통신 아키텍처 개요

시스템은 실시간성 보장과 통신 장애 대비를 위해 **이중 통신 아키텍처**를 채택하고 있습니다.

```mermaid
graph TD
    C[Vision AI 클라이언트] -->|1. 실시간 이벤트 전송| WS[WebSocket Server]
    C -->|2. 연결 해제 시 Fallback| REST[REST API Server]
    C -->|3. 구역 데이터 조회| GET_ZONES[GET /api/cameras/{id}/zones]
```

1. **REST API (HTTP)**: 시스템 초기화 시 카메라 설정 및 위험구역 다각형 좌표 정보를 동기화하는 데 사용됩니다.
2. **WebSocket (실시간)**: 침입 이벤트 발생 시 초저지연(Sub-second Latency)으로 데이터를 백엔드 대시보드로 즉각 전송합니다.
3. **REST Fallback**: 네트워크 불안정 등으로 WebSocket 연결이 끊긴 경우, 침입 경보가 누락되지 않도록 HTTP POST 요청으로 이벤트를 자동 우회 전송합니다.

---

## 🔑 2. 인증 및 헤더 규격

* **Base URL**: `http://localhost:8000` (환경 변수 `API_BASE_URL`로 오버라이드 가능)
* **WebSocket URL**: `ws://localhost:8000/ws/events` (환경 변수 `API_WS_URL`로 오버라이드 가능)
* **공통 요청 헤더 (REST)**:
  ```http
  Content-Type: application/json
  Authorization: Bearer <API_KEY_TOKEN> (설정 시 적용)
  ```

---

## 🌐 3. REST API 엔드포인트 명세

### 3.1. 특정 카메라의 위험구역 목록 조회
* **Endpoint**: `GET /api/cameras/{camera_id}/zones`
* **설명**: 해당 카메라 채널에 설정된 모든 활성 위험구역(다각형 좌표 및 등급) 목록을 조회합니다.

#### Request Example
```http
GET /api/cameras/cam-001/zones HTTP/1.1
Host: localhost:8000
```

#### Response Example (200 OK)
```json
[
  {
    "zone_id": "zone-area-01",
    "name": "A구역 로봇 가동 영역",
    "severity": "high",
    "polygon": [
      [150.0, 200.0],
      [450.0, 200.0],
      [450.0, 500.0],
      [150.0, 500.0]
    ]
  },
  {
    "zone_id": "zone-area-02",
    "name": "B구역 자재 적재 통로",
    "severity": "medium",
    "polygon": [
      [600.0, 100.0],
      [800.0, 100.0],
      [750.0, 400.0],
      [550.0, 400.0]
    ]
  }
]
```

---

### 3.2. 카메라 설정 정보 조회
* **Endpoint**: `GET /api/cameras/{camera_id}`
* **설명**: 카메라 해상도 캘리브레이션 정보 및 활성화 옵션을 조회합니다.

#### Response Example (200 OK)
```json
{
  "camera_id": "cam-001",
  "name": "메인 로봇 공정 카메라",
  "resolution": {
    "width": 1280,
    "height": 720
  },
  "fps_limit": 30,
  "is_active": true
}
```

---

### 3.3. 침입 이벤트 송신 (Fallback 용)
* **Endpoint**: `POST /api/events`
* **설명**: WebSocket 연결 오류 시 사용되는 백업 API로, 탐지된 침입 상세 데이터를 전송합니다.

#### Request Body Schema
```json
{
  "type": "intrusion_event",
  "camera_id": "cam-001",
  "event_type": "entered",
  "tracker_id": 4,
  "zone_id": "zone-area-01",
  "zone_name": "A구역 로봇 가동 영역",
  "zone_severity": "high",
  "timestamp": 1720230324.52,
  "position": { "x": 300, "y": 480 },
  "bbox": {
    "x1": 280.0,
    "y1": 310.0,
    "x2": 320.0,
    "y2": 480.0
  },
  "confidence": 0.89,
  "duration": 0.0,
  "snapshot_path": "snapshots/intrusion_cam-001_worker4_zone-area-01_20260706_113000.jpg",
  "snapshot_b64": "iVBORw0KGgoAAAANSUhEUgAAAAUA... (Base64로 인코딩된 스냅샷 이미지 바이트)"
}
```

#### Response Example (201 Created)
```json
{
  "success": true,
  "event_id": "evt_abc123xyz",
  "received_at": "2026-07-06T11:30:01Z"
}
```

---

### 3.4. 사고 보고서 전송
* **Endpoint**: `POST /api/incidents`
* **설명**: 지속적인 위험구역 침입이나 통제 거부 등 중대 안전사고 발생 시 관리자 긴급 보고서를 작성하여 전송합니다.

#### Request Body Schema
```json
{
  "incident_id": "inc_992123",
  "camera_id": "cam-001",
  "zone_id": "zone-area-01",
  "level": "critical",
  "description": "작업자#4번이 가동 중인 로봇 반경 1m 이내에 5초 이상 침입 지속함",
  "snapshot_b64": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
}
```

---

### 3.5. 헬스 체크 API
* **Endpoint**: `GET /health`
* **설명**: 클라이언트 시스템이 서버의 생존 여부 및 가동 상태를 주기적으로 감지하기 위한 API입니다.

#### Response Example (200 OK)
```json
{
  "status": "healthy",
  "version": "1.4.2"
}
```

---

## ⚡ 4. WebSocket 실시간 이벤트 명세

* **WebSocket Endpoint**: `ws://localhost:8000/ws/events`
* **전송 방향**: Client $\rightarrow$ Server (단방향 알림 패킷 송신)
* **프레임 형식**: JSON (UTF-8)

### 4.1. 침입 이벤트 패킷 구조 (IntrusionEvent)
침입 발생(ENTERED), 지속 체류(STAYING), 퇴장(EXITED) 상태 변화 시 실시간으로 발송되는 단일 패킷 포맷입니다.

```json
{
  "type": "intrusion_event",
  "camera_id": "string (카메라 식별 고유 ID)",
  "event_type": "string ('entered' | 'staying' | 'exited')",
  "tracker_id": "integer (작업자 고유 추적 ID)",
  "zone_id": "string (위험구역 고유 ID)",
  "zone_name": "string (위험구역 표시명)",
  "zone_severity": "string ('low' | 'medium' | 'high' | 'critical')",
  "timestamp": "float (이벤트 발생 Unix Epoch Time)",
  "position": {
    "x": "integer (작업자 발밑 기준 X 픽셀 좌표)",
    "y": "integer (작업자 발밑 기준 Y 픽셀 좌표)"
  },
  "bbox": {
    "x1": "float (작업자 Bounding Box 좌상단 X)",
    "y1": "float (작업자 Bounding Box 좌상단 Y)",
    "x2": "float (작업자 Bounding Box 우하단 X)",
    "y2": "float (작업자 Bounding Box 우하단 Y)"
  },
  "confidence": "float (객체 탐지 신뢰도 0.0 ~ 1.0)",
  "duration": "float (현재 구역 내 누적 체류 시간, 초 단위)",
  "snapshot_path": "string (로컬 서버에 백업된 스냅샷 파일의 상대 경로)",
  "snapshot_b64": "string (Base64로 인코딩된 실시간 스냅샷 파일 데이터)"
}
```

---

## 🛡️ 5. 예외 처리 및 장애 방어 정책

1. **요청 재시도 및 헬스체크**:
   * API 통신 지연 혹은 네트워크 패킷 드랍 발생 시 최대 **3회** 재시도를 수행합니다.
   * 1회 요청 타임아웃 제한은 **5.0초**로 기본 설정되어 있습니다.

2. **웹소켓 상시 연결 유지 및 자동 재연결**:
   * 클라이언트와 서버 간의 통신은 **상시 웹소켓 연결(Persistent Connection)** 상태를 가집니다.
   * 네트워크 장애, 방화벽 차단 또는 백엔드 재가동으로 연결이 유실되는 경우, 클라이언트는 세션을 파괴하지 않고 백그라운드 스레드에서 **3초 주기**로 자동 재연결(Auto-Reconnect)을 시도합니다.

3. **실시간 이중 채널 우회 (REST Fallback)**:
   * 웹소켓 연결이 예기치 않게 끊겼거나 자동 재접속을 시도하고 있는 과도기 상태일 때, 실시간 큐에서 처리되는 긴급 침입 알림 이벤트는 대기하지 않고 **REST Webhook API (`POST /api/events`)**로 즉시 우회 전송되어 경보 유실을 예방합니다.

4. **비동기 큐 전송 (Non-blocking Queue)**:
   * 메인 영상 분석 프로세스의 FPS 프레임 유실을 막기 위해, 모든 이벤트 전송은 별도로 분리된 백그라운드 워커 스레드(`EventSender`)에서 처리됩니다.
   * `max_queue_size=100`으로 제한된 내부 메모리 큐가 가득 찰 경우, 버퍼 오버플로우를 예방하기 위해 가장 오래된 이벤트를 삭제하고 최신 이벤트를 교체 삽입합니다.
