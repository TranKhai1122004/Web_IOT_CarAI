import cv2
import os
import time

# =========================================================
# CẤU HÌNH THÔNG SỐ
# =========================================================
VIDEO_SOURCE = "http://192.168.100.166:8080/video"
OUTPUT_DIR = "dataset_negatives"  # Thư mục gốc lưu bộ dữ liệu
INTERVAL_SEC = 0.5  # Khoảng thời gian giữa các lần chụp (0.5 giây chụp 1 tấm)

# Tạo cấu trúc thư mục chuẩn YOLO
IMG_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train", "images")
LAB_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train", "labels")

for path in [IMG_TRAIN_DIR, LAB_TRAIN_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

print("🎬 Đang kết nối tới ESP32-Cam...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("❌ Không thể kết nối tới Camera. Kiểm tra lại IP hoặc nguồn xe!")
    exit()

print(f"✅ Kết nối thành công! Bắt đầu chụp sau mỗi {INTERVAL_SEC}s.")
print("⚠️ LƯU Ý: Cho xe chạy quanh nền nhà trống (KHÔNG có biển báo). Bấm 'q' để DỪNG.")

count = 0
last_capture_time = time.time()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Mất tín hiệu Frame từ Camera, đang thử lại...")
            time.sleep(0.1)
            continue

        # Hiển thị luồng live để bạn nhìn đường đi của xe
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Captured: {count} images",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Data Collection - Press 'q' to stop", display_frame)

        # Kiểm tra khoảng thời gian để tự động chụp
        current_time = time.time()
        if current_time - last_capture_time >= INTERVAL_SEC:
            # Tạo tên file duy nhất theo timestamp để không bao giờ bị đè file
            timestamp = int(current_time * 1000)
            img_name = f"neg_{timestamp}.jpg"
            label_name = f"neg_{timestamp}.txt"

            img_path = os.path.join(IMG_TRAIN_DIR, img_name)
            label_path = os.path.join(LAB_TRAIN_DIR, label_name)

            # 1. Lưu file ảnh (.jpg)
            cv2.imwrite(img_path, frame)

            # 2. Tạo file nhãn rỗng (.txt) - Đây là mấu chốt để YOLO biết đây là nền trống
            with open(label_path, "w") as f:
                pass  # Để trống không ghi gì cả

            count += 1
            print(f"📸 Đã lưu tấm thứ {count}: {img_name} + file nhãn trống.")
            last_capture_time = current_time

        # Bấm 'q' để thoát luồng
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 Ngắt luồng bằng phím tắt.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("---")
    print(f"🎉 Hoàn thành! Đã thu thập tổng cộng: {count} ảnh âm bản.")
    print(f"📂 Toàn bộ dữ liệu nằm trong thư mục: '{OUTPUT_DIR}'")
    print(
        "💡 Bây giờ bạn chỉ cần copy đống ảnh và nhãn này ném thẳng vào tập dữ liệu cũ trên Kaggle rồi nhấn Train lại là xong!"
    )