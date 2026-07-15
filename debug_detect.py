import cv2
from ultralytics import YOLO
from loguru import logger

def main():
    # 비디오 소스 열기
    cap = cv2.VideoCapture("sample4.mp4")
    if not cap.isOpened():
        logger.error("sample4.mp4 열기 실패")
        return

    # 프레임 하나 읽기
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("프레임 읽기 실패")
        cap.release()
        return
    cap.release()

    # 모델 로드
    models = ["yolo26s-seg.pt", "yolo26m-seg.pt"]
    
    for model_name in models:
        logger.info(f"=== {model_name} 테스트 ===")
        model = YOLO(model_name)
        
        # 1. confidence 0.25 로 추론
        results = model(frame, conf=0.25, imgsz=640, verbose=False)
        if len(results) > 0:
            boxes = results[0].boxes
            logger.info(f"conf=0.25 검출 개수: {len(boxes)}")
            for idx in range(len(boxes)):
                xyxy = boxes.xyxy[idx].cpu().numpy()
                conf = float(boxes.conf[idx].cpu().numpy())
                cls = int(boxes.cls[idx].cpu().numpy())
                logger.info(f"  [{idx}] bbox: {xyxy}, conf: {conf:.3f}, class: {cls}")
        
        # 2. confidence 0.5 로 추론 (현재 기본값)
        results_high = model(frame, conf=0.5, imgsz=640, verbose=False)
        if len(results_high) > 0:
            boxes_high = results_high[0].boxes
            logger.info(f"conf=0.50 검출 개수: {len(boxes_high)}")

if __name__ == "__main__":
    main()
