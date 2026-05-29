import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
WEIGHT_PATH = "../src/runs/detect/yolo11_vntraffic_test2/weights/best.pt"
IMAGE_PATH = "../inference/test_img/OIP1.jpg"
OUTPUT_PATH = "../image_test/output/result.jpg"
NAMES_PATH = "../inference/classes_vie.txt"

# ─────────────────────────────────────────────
# THAM SỐ INFERENCE
# ─────────────────────────────────────────────
CONF_THRESHOLD = 0.2  # Ngưỡng tin cậy (confidence)
IOU_THRESHOLD = 0.45  # Ngưỡng IoU cho NMS
IMG_SIZE = 640  # Kích thước input models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Màu sắc bounding box theo từng loại biển (BGR)
CLASS_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 200, 200), (0, 255, 0),
    (128, 0, 255), (255, 0, 0), (255, 128, 0), (255, 255, 0),
    (0, 255, 128), (128, 255, 0), (200, 0, 200), (0, 128, 128),
]


def load_class_names(names_path: str) -> list[str]:
    """Đọc danh sách tên class từ file .txt (mỗi dòng một class)."""
    path = Path(names_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tên class: {names_path}")
    with open(path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"[✓] Đã load {len(names)} classes từ {path.name}")
    return names


def draw_predictions(image: "cv2.Mat", result, class_names: list[str]) -> "cv2.Mat":
    """Vẽ bounding box và nhãn lên ảnh."""
    boxes = result.boxes
    h, w = image.shape[:2]
    overlay = image.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # Vẽ bounding box (đường viền dày hơn nếu ảnh lớn)
        thickness = max(2, int(min(h, w) / 300))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Vẽ nền nhãn
        text = f"{label}  {conf:.2f}"
        font_scale = max(0.4, min(h, w) / 1200)
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(
            overlay,
            (x1, label_y - th - baseline - 4),
            (x1 + tw + 4, label_y + baseline - 2),
            color, -1
        )
        cv2.putText(
            overlay, text, (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    # Blend overlay để box trong suốt nhẹ
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    return image


def run_inference(
        weight_path: str,
        image_path: str,
        output_path: str,
        names_path: str,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        imgsz: int = IMG_SIZE,
        device: str = DEVICE,
) -> None:
    print(f"\n{'=' * 55}")
    print("  YOLOv11 – Phát hiện biển báo giao thông Việt Nam")
    print(f"{'=' * 55}")
    print(f"  Device  : {device.upper()}")
    print(f"  Weights : {weight_path}")
    print(f"  Image   : {image_path}")
    print(f"  Conf    : {conf}  |  IoU : {iou}  |  Size : {imgsz}")
    print(f"{'=' * 55}\n")

    # Load class names & models
    class_names = load_class_names(names_path)
    model = YOLO(weight_path)
    model.to(device)

    # Load ảnh
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")

    # Predict
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]
    n_detected = len(result.boxes)
    print(f"[✓] Phát hiện được {n_detected} đối tượng\n")

    # In kết quả ra console
    if n_detected > 0:
        print(f"  {'#':<4} {'Tên biển báo':<35} {'Conf':>6}  {'Bounding Box'}")
        print(f"  {'-' * 75}")
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"  {i + 1:<4} {name:<35} {conf_v:>6.3f}  [{x1},{y1} → {x2},{y2}]")
    else:
        print("  (Không phát hiện biển báo nào với conf hiện tại)")

    # Vẽ kết quả lên ảnh
    annotated = draw_predictions(image, result, class_names)

    # Lưu ảnh output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[✓] Đã lưu kết quả → {out_path}")

    # Hiển thị ảnh (bỏ comment nếu chạy có màn hình)
    cv2.imshow("YOLOv11 – VN Traffic Signs", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference(
        weight_path=WEIGHT_PATH,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        names_path=NAMES_PATH,
    )