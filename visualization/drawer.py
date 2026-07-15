"""
대화형 위험구역 마우스 드로잉 모듈
"""

import cv2
import numpy as np
from loguru import logger
from typing import Optional


class ZoneDrawer:
    """마우스로 위험구역(다각형)을 직접 그리는 도구"""

    def __init__(self, window_name: str = "Danger Zone Setup"):
        self.window_name = window_name
        self.current_points: list[list[int]] = []
        self.completed_zones: list[list[list[int]]] = []
        self.img_width = 0
        self.img_height = 0

    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 핸들러"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 꼭짓점 추가
            self.current_points.append([x, y])
            logger.debug(f"꼭짓점 추가: ({x}, {y}) | 현재 총 {len(self.current_points)}개")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 마지막 꼭짓점 취소 (Undo)
            if self.current_points:
                removed = self.current_points.pop()
                logger.debug(f"꼭짓점 취소: {removed}")

    def draw_zones(self, frame: np.ndarray) -> list[dict]:
        """
        주어진 프레임 위에서 마우스 드로잉을 실행하여 위험구역 좌표 목록을 획득

        Args:
            frame: 영상의 첫 프레임 이미지

        Returns:
            위험구역 정보 딕셔너리 리스트
        """
        self.img_height, self.img_width = frame.shape[:2]
        self.current_points = []
        self.completed_zones = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        logger.info("=" * 60)
        logger.info("🖱️  위험구역 마우스 설정 모드 진입")
        logger.info("   - 마우스 좌클릭 : 꼭짓점 지정")
        logger.info("   - 마우스 우클릭 : 마지막 점 취소 (Undo)")
        logger.info("   - 키보드 [C]    : 현재 그리는 다각형 전체 초기화")
        logger.info("   - 키보드 [Enter]/[Space] : 현재 다각형 완성 및 저장")
        logger.info("   - 키보드 [S]    : 설정 완료 및 감지 프로그램 시작")
        logger.info("   - 키보드 [Q]    : 프로그램 종료")
        logger.info("=" * 60)

        while True:
            display = frame.copy()

            # 1. 이미 완성된 구역들 그리기
            for i, zone in enumerate(self.completed_zones):
                pts = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 0, 255), 2)
                cv2.fillPoly(display, [pts], (0, 0, 100))  # 반투명 느낌의 어두운 빨강

                # 구역 이름 표시
                centroid = np.mean(zone, axis=0).astype(int)
                cv2.putText(
                    display, f"Zone-{i+1}",
                    (centroid[0] - 25, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

            # 2. 현재 그리고 있는 꼭짓점 및 선 표시
            if self.current_points:
                # 점 찍기
                for pt in self.current_points:
                    cv2.circle(display, tuple(pt), 4, (0, 255, 255), -1)

                # 선 긋기
                if len(self.current_points) > 1:
                    pts = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], False, (0, 255, 255), 2)

            # 안내 문구 렌더링
            self._draw_guide_text(display)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            # 키보드 입력 처리
            if key == ord('q') or key == ord('Q'):
                logger.info("사용자에 의해 설정 프로그램이 종료되었습니다.")
                cv2.destroyWindow(self.window_name)
                import sys
                sys.exit(0)

            elif key == ord('c') or key == ord('C'):
                # 클리어
                self.current_points.clear()
                logger.info("현재 드로잉 포인트 초기화됨")

            elif key in [13, 32]:  # Enter (13) 또는 Space (32)
                # 구역 완성
                if len(self.current_points) >= 3:
                    self.completed_zones.append(list(self.current_points))
                    logger.info(f"위험구역 저장 완료 (Zone-{len(self.completed_zones)})")
                    self.current_points.clear()
                else:
                    logger.warning("위험구역은 최소 3개 이상의 꼭짓점이 필요합니다.")

            elif key == ord('s') or key == ord('S'):
                # 설정 완료 및 시작
                if self.completed_zones:
                    logger.info(f"총 {len(self.completed_zones)}개의 위험구역을 설정하고 감지를 시작합니다.")
                    break
                else:
                    logger.warning("감지를 시작하려면 최소 1개 이상의 위험구역을 완성 후 저장(Enter)해야 합니다.")

        cv2.destroyWindow(self.window_name)

        # zone_manager가 이해할 수 있는 딕셔너리 포맷으로 변환하여 반환
        zones_data = []
        for i, zone in enumerate(self.completed_zones):
            zones_data.append({
                "zone_id": f"drawn-zone-{i+1}",
                "name": f"Danger-Zone-{i+1}",
                "points": zone,
                "severity": "high",
                "is_normalized": False
            })

        return zones_data

    def _draw_guide_text(self, display: np.ndarray) -> None:
        """안내 설명 화면 하단 오버레이"""
        h, w = display.shape[:2]
        guide_box_height = 80
        overlay = display.copy()

        # 하단 불투명 가이드 영역 생성
        cv2.rectangle(
            overlay,
            (0, h - guide_box_height),
            (w, h),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # 텍스트 라인 출력
        lines = [
            "Left Click: Add Point | Right Click: Undo | C: Clear Points",
            f"Enter/Space: Complete Zone ({len(self.current_points)} pts) | S: Start System (Zones: {len(self.completed_zones)}) | Q: Quit"
        ]

        for idx, line in enumerate(lines):
            cv2.putText(
                display, line,
                (20, h - guide_box_height + 30 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if idx == 1 else (255, 255, 255),
                1, cv2.LINE_AA
            )
