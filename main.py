"""
실시간 영상 처리 및 위험구역 감지 시스템
엔트리포인트
"""

import argparse
import signal
import sys
import cv2
from loguru import logger

from config import AppSettings
from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.zone_manager import DangerZoneManager
from processing.video_source import create_video_source
from processing.frame_processor import FrameProcessor
from processing.intrusion_detector import IntrusionDetector
from visualization.renderer import FrameRenderer
from communication.api_client import BackendAPIClient
from communication.event_sender import EventSender


def setup_logger(log_level: str = "INFO") -> None:
    """로거 설정"""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True
    )
    logger.add(
        "logs/vision_ai_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="실시간 영상 처리 및 위험구역 감지 시스템"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="영상 소스 경로 (MP4 파일 또는 RTSP URL)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="YOLO 모델 경로 (기본: yolo11n.pt)"
    )
    parser.add_argument(
        "--show", "-d",
        action="store_true",
        default=None,
        help="시각화 디스플레이 표시"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="시각화 디스플레이 비활성화 (headless 모드)"
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="마우스로 직접 위험구역 그리기 설정 모드 활성화"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=None,
        help="감지 신뢰도 임계값 (0.0~1.0)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="백엔드 API URL"
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="카메라 식별자"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="데모 모드 (기본 위험구역 자동 생성)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨"
    )

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()

    # 설정 로드
    settings = AppSettings()

    # CLI 인자로 설정 오버라이드
    if args.source:
        settings.video.source = args.source
    if args.model:
        settings.detector.model_path = args.model
    if args.show:
        settings.show_display = True
    if args.no_show:
        settings.show_display = False
    if args.confidence:
        settings.detector.confidence_threshold = args.confidence
    if args.api_url:
        settings.api.base_url = args.api_url
    if args.camera_id:
        settings.camera_id = args.camera_id
    if args.log_level:
        settings.log_level = args.log_level

    # 로거 설정
    setup_logger(settings.log_level)

    logger.info("=" * 60)
    logger.info("🎯 실시간 영상 처리 및 위험구역 감지 시스템 시작")
    logger.info("=" * 60)
    logger.info(f"영상 소스: {settings.video.source}")
    logger.info(f"YOLO 모델: {settings.detector.model_path}")
    logger.info(f"카메라 ID: {settings.camera_id}")
    logger.info(f"디스플레이: {'ON' if settings.show_display else 'OFF'}")

    # ============================================================
    # 컴포넌트 초기화
    # ============================================================

    # 1. 영상 소스
    resize = (settings.video.width, settings.video.height)
    video_source = create_video_source(
        source=settings.video.source,
        resize=resize,
        fps_limit=settings.video.fps_limit,
        loop=settings.video.loop
    )

    if not video_source.open():
        logger.error("영상 소스 열기 실패. 프로그램을 종료합니다.")
        sys.exit(1)

    # 2. 사람 감지기 (YOLOv11)
    detector = PersonDetector(
        model_path=settings.detector.model_path,
        confidence_threshold=settings.detector.confidence_threshold,
        iou_threshold=settings.detector.iou_threshold,
        device=settings.detector.device,
        img_size=settings.detector.img_size,
        target_classes=settings.detector.target_classes
    )

    # 3. 사람 추적기 (ByteTrack)
    tracker = PersonTracker(
        track_thresh=settings.tracker.track_thresh,
        track_buffer=settings.tracker.track_buffer,
        match_thresh=settings.tracker.match_thresh
    )

    # 4. 위험구역 관리자
    frame_w, frame_h = video_source.frame_size
    zone_manager = DangerZoneManager(
        camera_id=settings.camera_id,
        poll_interval=settings.zone.poll_interval,
        frame_width=frame_w,
        frame_height=frame_h,
        default_zones=settings.zone.default_zones
    )

    # 마우스 드로잉 모드 활성화 시
    if args.draw:
        # 첫 번째 프레임 읽어오기
        ret, first_frame = video_source.read()
        if not ret or first_frame is None:
            logger.error("마우스 드로잉 설정을 위해 첫 프레임을 읽어올 수 없습니다.")
            sys.exit(1)
        
        # 드로잉 툴 생성 및 실행
        from visualization.drawer import ZoneDrawer
        drawer = ZoneDrawer()
        custom_zones = drawer.draw_zones(first_frame)

        # 설정된 위험구역 추가
        for zone_data in custom_zones:
            zone_manager.add_zone(zone_data)
        
        # 비디오 소스 리셋 (감지를 처음부터 시작하도록 파일 재생위치 0으로 초기화)
        if hasattr(video_source, '_cap') and video_source._cap is not None:
            video_source._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    # 데모 모드: 기본 위험구역 생성 (드로잉 안 했을 때만)
    elif args.demo:
        zone_manager.set_demo_zones(frame_w, frame_h)
        logger.info("🔶 데모 모드 활성화 - 기본 위험구역 생성됨")

    # 5. 침입 감지기
    intrusion_detector = IntrusionDetector(
        cooldown_seconds=settings.event.cooldown_seconds,
        method=settings.event.intrusion_method,
        overlap_ratio=settings.event.overlap_ratio,
        frame_width=frame_w,
        frame_height=frame_h,
        pose_conf_threshold=settings.event.pose_conf_threshold,
        waving_amplitude_ratio=settings.event.waving_amplitude_ratio,
        waving_direction_changes=settings.event.waving_direction_changes,
        waving_min_frames=settings.event.waving_min_frames,
        waving_conf_threshold=settings.event.waving_conf_threshold,
        waving_y_ratio=settings.event.waving_y_ratio,
        waving_history_frames=settings.event.waving_history_frames,
        waving_smooth_window=settings.event.waving_smooth_window,
        waving_pixel_threshold=settings.event.waving_pixel_threshold,
        waving_speed_threshold=settings.event.waving_speed_threshold,
        enter_threshold_frames=settings.event.enter_threshold_frames,
        exit_threshold_frames=settings.event.exit_threshold_frames,
    )

    # 6. 시각화 렌더러
    renderer = FrameRenderer()

    # 7. 이벤트 전송기
    event_sender = EventSender(
        ws_url=settings.api.ws_url,
        camera_id=settings.camera_id,
        save_snapshots=settings.event.save_snapshots,
        snapshot_dir=settings.event.snapshot_dir
    )
    event_sender.start()

    # 8. 프레임 처리 파이프라인
    frame_processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        zone_manager=zone_manager,
        intrusion_detector=intrusion_detector,
        renderer=renderer,
        event_sender=event_sender
    )

    # ============================================================
    # Graceful Shutdown 핸들러
    # ============================================================
    shutdown_requested = False

    def signal_handler(signum, frame_ref):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("강제 종료")
            sys.exit(1)
        shutdown_requested = True
        logger.info("종료 신호 수신 - 안전하게 종료 중...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ============================================================
    # 메인 루프
    # ============================================================
    logger.info("🚀 메인 루프 시작 (종료: 'q' 키 또는 Ctrl+C)")

    try:
        while not shutdown_requested:
            # 프레임 읽기
            ret, frame = video_source.read()
            if not ret or frame is None:
                if not settings.video.loop:
                    logger.info("영상 재생 완료")
                    break
                continue

            # 프레임 처리
            display_frame, events = frame_processor.process(
                frame, show_display=settings.show_display
            )

            # 디스플레이 표시
            if settings.show_display:
                cv2.imshow("Vision AI - Danger Zone Detection", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("'q' 키 입력 - 종료")
                    break
                elif key == ord('r'):
                    # 추적기 리셋
                    tracker.reset()
                    intrusion_detector.reset()
                    logger.info("추적기 및 침입 감지기 리셋")
                elif key == ord('d'):
                    # 데모 zone 토글
                    if len(zone_manager.get_zones()) == 0:
                        zone_manager.set_demo_zones(frame_w, frame_h)
                        logger.info("데모 위험구역 생성")
                    else:
                        for zone in zone_manager.get_all_zones():
                            zone_manager.remove_zone(zone.zone_id)
                        intrusion_detector.reset()
                        logger.info("위험구역 모두 제거")

    except Exception as e:
        logger.exception(f"메인 루프 오류: {e}")

    finally:
        # ============================================================
        # 정리 (Cleanup)
        # ============================================================
        logger.info("시스템 종료 중...")

        video_source.release()
        event_sender.stop()
        zone_manager.stop_polling()

        if settings.show_display:
            cv2.destroyAllWindows()

        # 최종 성능 통계
        stats = frame_processor.performance_stats
        logger.info(
            f"📊 최종 성능 통계 | "
            f"총 {stats['frame_count']}프레임 | "
            f"감지: {stats['avg_detect_ms']:.1f}ms | "
            f"추적: {stats['avg_track_ms']:.1f}ms | "
            f"침입: {stats['avg_intrusion_ms']:.1f}ms | "
            f"렌더: {stats['avg_render_ms']:.1f}ms"
        )

        event_stats = event_sender.stats
        logger.info(
            f"📤 이벤트 통계 | "
            f"전송: {event_stats['total_sent']} | "
            f"실패: {event_stats['total_failed']} | "
            f"WS: {event_stats['ws_sends']} | "
            f"REST: {event_stats['rest_sends']}"
        )

        logger.info("✅ 시스템 종료 완료")


if __name__ == "__main__":
    main()
