# 실시간 영상 처리 및 위험구역 감지 - 태스크 트래커

## 구현 단계

- [x] 1. 프로젝트 초기 셋업 (requirements.txt, config.py, 디렉토리 구조 수립)
- [x] 2. RTSP 및 동영상 입력 소스 추상화 모듈 구축 (video_source.py)
- [x] 3. YOLO26 가중치 연동 및 감지 파이프라인 개발 (detector.py)
- [x] 4. 실시간 다중 객체 고유 ID 추적기 구현 (tracker.py)
- [x] 5. 로컬 위험구역 데이터 적재 및 API 로드 모듈 구현 (zone_manager.py, geometry.py)
- [x] 6. 5가지 고정밀 침입 감지 판정 로직 설계 (intrusion_detector.py)
  - [x] Point 방식 구현
  - [x] Multi-Point 방식 구현
  - [x] Overlap (바운딩박스 하단 20%) 방식 구현
  - [x] Segment (픽셀 마스크 하단 20%) 방식 구현
  - [x] Pose-Hybrid (발목 관절 + 폴백) 방식 구현
- [x] 7. 화면 테두리 깜빡임 및 상단 중앙 경고 배너 렌더링 시각화 모듈 구현 (renderer.py)
- [x] 8. 대화형 마우스 클릭 기반 위험구역 다각형 그리기 UI 추가 (drawer.py)
- [x] 9. 프레임 워크플로우 통합 및 CLI 제어부 구축 (frame_processor.py, main.py)
- [x] 10. 백엔드 실시간 알림 송신 웹소켓 및 API 연결 (api_client.py, event_sender.py)
- [x] 11. 17개 핵심 시나리오 단위 테스트 검증 및 버그 픽스 (tests/test_intrusion.py)
