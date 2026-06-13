import cv2
import numpy as np

# Tạo mảng lưu 4 tọa độ click chuột
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"📍 Đã chọn điểm {len(clicked_points)}: ({x}, {y})")
            # Vẽ vòng tròn đánh dấu điểm vừa click
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(
                "DONG DO TOA DO - CLICK THEO THU TU: TREN TRAI -> TREN PHAI -> DUOI TRAI -> DUOI PHAI",
                img,
            )


# Địa chỉ camera stream thực tế từ iPhone của ông
video_url = "http://192.168.100.166:8080/video"
cap = cv2.VideoCapture(video_url)

print("👉 Chờ 2 giây để bốc frame hình ảnh...")
time_sleep = 2
for _ in range(30):
    cap.read()  # Đọc bỏ qua các frame nhiễu đầu tiên
success, img = cap.read()

if not success:
    print("❌ Không thể bốc được ảnh từ iPhone! Check lại App trên điện thoại.")
    exit()

# Ép kích thước ảnh về chuẩn 640x480 đồng bộ với hệ thống AI
img = cv2.resize(img, (640, 480))
clone_img = img.copy()

cv2.namedWindow(
    "DONG DO TOA DO - CLICK THEO THU TU: TREN TRAI -> TREN PHAI -> DUOI TRAI -> DUOI PHAI"
)
cv2.setMouseCallback(
    "DONG DO TOA DO - CLICK THEO THU TU: TREN TRAI -> TREN PHAI -> DUOI TRAI -> DUOI PHAI",
    mouse_callback,
)

print("\n=== HƯỚNG DẪN ===")
print(
    "Dùng chuột click đúng 4 điểm tạo thành hình thang bao quanh LÀN ĐƯỜNG SA BÀN trước mũi xe:"
)
print("1. Click GÓC TRÊN BÊN TRÁI")
print("2. Click GÓC TRÊN BÊN PHẢI")
print("3. Click GÓC DƯỚI BÊN TRÁI")
print("4. Click GÓC DƯỚI BÊN PHẢI")
print("Ấn nút 'q' để Thoát, ấn 'r' để Reset làm lại từ đầu.\n")

while True:
    cv2.imshow(
        "DONG DO TOA DO - CLICK THEO THU TU: TREN TRAI -> TREN PHAI -> DUOI TRAI -> DUOI PHAI",
        img,
    )
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        img = clone_img.copy()
        clicked_points = []
        print("🔄 Đã xóa làm lại!")

    if len(clicked_points) == 4 or key == ord("q"):
        break

cv2.destroyAllWindows()

if len(clicked_points) == 4:
    print(
        "\n✅ THÀNH CÔNG! Copy 4 cặp tọa độ này ném vào code biến đổi Bird's Eye View:"
    )
    print(f"src = np.float32({clicked_points})")
