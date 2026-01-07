import cv2
from ultralytics import YOLO

# Path to your YOLO model (replace with your custom toy model if you have one)
MODEL_PATH = "yolov8n.pt"  # or "runs/detect/train/weights/best.pt"

# Optional: list of toy-related class names you care about (for a custom model, use its names)
TOY_CLASSES = {"teddy bear", "sports ball", "kite", "skateboard", "baseball bat", "baseball glove"}

def main():
    # Load model
    model = YOLO(MODEL_PATH)

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        # Ultralytics returns a list; we take the first result
        result = results[0]

        # Get boxes, class IDs, and confidences
        boxes = result.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2]
        class_ids = result.boxes.cls.cpu().numpy()   # class indices
        scores = result.boxes.conf.cpu().numpy()     # confidence scores

        # Class names
        names = model.names

        for box, cls_id, score in zip(boxes, class_ids, scores):
            cls_id = int(cls_id)
            label = names.get(cls_id, str(cls_id))

            # If you only want toy-like objects, filter here
            if TOY_CLASSES and label not in TOY_CLASSES:
                continue

            x1, y1, x2, y2 = box.astype(int)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label text
            text = f"{label} {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Toy Recognition Demo - YOLO", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
