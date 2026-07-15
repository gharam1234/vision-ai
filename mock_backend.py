"""
E2E 테스트 및 검증용 Mock FastAPI 백엔드 서버
"""

import os
import base64
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Mock Danger Zone Detection Backend")

# 스냅샷 저장용 디렉토리 생성
SAVE_DIR = "received_snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)


@app.get("/health")
async def health_check():
    """헬스 체크 API"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 Health Check Requested")
    return {"status": "healthy", "version": "1.4.2"}


@app.get("/api/cameras/{camera_id}/zones")
async def get_danger_zones(camera_id: str):
    """특정 카메라의 위험구역 목록 조회 API"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Camera Zones Requested: {camera_id}")
    
    # API 명세서에 따른 샘플 위험구역 (테스트 용도)
    return [
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


@app.get("/api/cameras/{camera_id}")
async def get_camera_config(camera_id: str):
    """카메라 설정 정보 조회 API"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ Camera Config Requested: {camera_id}")
    return {
        "camera_id": camera_id,
        "name": f"카메라-{camera_id}",
        "resolution": {"width": 1280, "height": 720},
        "fps_limit": 30,
        "is_active": True
    }


def save_b64_image(snapshot_b64: str, camera_id: str, tracker_id: int, zone_id: str) -> str:
    """Base64 이미지를 디코딩하여 received_snapshots 폴더에 저장"""
    try:
        image_data = base64.b64decode(snapshot_b64)
        filename = f"rec_{camera_id}_tr{tracker_id}_{zone_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filepath
    except Exception as e:
        print(f"❌ 스냅샷 디코딩 및 저장 실패: {e}")
        return ""


@app.post("/api/events")
async def receive_event(request: Request):
    """침입 이벤트 수신 Webhook API (REST fallback)"""
    try:
        event_data = await request.json()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 [REST Webhook] 이벤트 수신:")
        print(f"  - 카메라: {event_data.get('camera_id')}")
        print(f"  - 이벤트 타입: {event_data.get('event_type')}")
        print(f"  - 작업자 ID: {event_data.get('tracker_id')}")
        print(f"  - 구역: {event_data.get('zone_name')} ({event_data.get('zone_id')})")
        
        # Base64 이미지 존재 시 복원 저장
        snapshot_b64 = event_data.get("snapshot_b64")
        if snapshot_b64:
            saved_path = save_b64_image(
                snapshot_b64=snapshot_b64,
                camera_id=event_data.get('camera_id', 'unknown'),
                tracker_id=event_data.get('tracker_id', 0),
                zone_id=event_data.get('zone_id', 'unknown')
            )
            if saved_path:
                print(f"  - 스냅샷 수신 및 저장 완료 -> {saved_path}")
        
        return JSONResponse(status_code=201, content={
            "success": True,
            "event_id": f"evt_rest_{int(datetime.now().timestamp())}",
            "received_at": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        print(f"❌ REST 이벤트 수신 처리 실패: {e}")
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 웹소켓 이벤트 스트리밍 수신 핸들러"""
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔌 [WebSocket] 연결 성공: {client_host}")
    
    try:
        while True:
            # 텍스트(JSON) 메시지 수신
            data = await websocket.receive_text()
            event_data = json.loads(data)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ [WebSocket] 이벤트 패킷 수신:")
            print(f"  - 카메라: {event_data.get('camera_id')}")
            print(f"  - 이벤트 타입: {event_data.get('event_type')}")
            print(f"  - 작업자 ID: {event_data.get('tracker_id')}")
            print(f"  - 구역: {event_data.get('zone_name')} ({event_data.get('zone_id')})")
            
            # Base64 이미지 존재 시 복원 저장
            snapshot_b64 = event_data.get("snapshot_b64")
            if snapshot_b64:
                saved_path = save_b64_image(
                    snapshot_b64=snapshot_b64,
                    camera_id=event_data.get('camera_id', 'unknown'),
                    tracker_id=event_data.get('tracker_id', 0),
                    zone_id=event_data.get('zone_id', 'unknown')
                )
                if saved_path:
                    print(f"  - 스냅샷 수신 및 저장 완료 -> {saved_path}")
                    
    except WebSocketDisconnect:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔌 [WebSocket] 연결 해제됨: {client_host}")
    except Exception as e:
        print(f"❌ WebSocket 데이터 처리 실패: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Mock 위험구역 백엔드 서버 구동 시작 (포트 8000)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
