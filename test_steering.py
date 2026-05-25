import paho.mqtt.client as mqtt
import time

# ==========================================
# CẤU HÌNH MQTT CỤC BỘ
# ==========================================
MQTT_SERVER = "127.0.0.1"
MQTT_PORT = 1883
TOPIC_DIRECTION = "remote_car/direction"
TOPIC_SPEED = "remote_car/speed"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_SERVER, MQTT_PORT)
client.loop_start()


def test_steering(direction_cmd, test_speed, duration_seconds):
    print(
        f"\n🚀 CHUẨN BỊ TEST: Rẽ {direction_cmd} | Tốc độ: {test_speed} | Thời gian: {duration_seconds} giây"
    )
    print("Đặt xe xuống sàn phẳng, ngay hàng thẳng lối...")

    # Đếm ngược 3 giây để chuẩn bị buông tay khỏi xe
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("🔥 CHẠY!")
    # 1. Bắn tốc độ test cố định (Trùng với tốc độ chạy AI, ví dụ s100)
    client.publish(TOPIC_SPEED, f"s{test_speed}")
    time.sleep(0.1)

    # 2. Bắn lệnh rẽ (L hoặc R)
    client.publish(TOPIC_DIRECTION, direction_cmd)

    # 3. Giữ lệnh đúng số giây cần test
    time.sleep(duration_seconds)

    # 4. Phát lệnh DỪNG XE ngay lập tức (S)
    client.publish(TOPIC_DIRECTION, "S")
    print("🛑 DỪNG! Hãy đo góc lệch thực tế của xe.")


if __name__ == "__main__":
    try:
        # ----- CẤU HÌNH THAM SỐ TEST TẠI ĐÂY -----
        HUONG_RE = "L"  # L (Trái) hoặc R (Phải)
        TOC_DO_TEST = 100  # Giữ cố định tốc độ nền (trong code chính của ông là 100)
        THOI_GIAN_KICH_XUNG = (
            0.20  # Thời gian cấp điện cho motor rẽ (Thử trước với 0.2 giây)
        )

        test_steering(HUONG_RE, TOC_DO_TEST, THOI_GIAN_KICH_XUNG)

    except KeyboardInterrupt:
        client.publish(TOPIC_DIRECTION, "S")
        print("\nNgắt khẩn cấp!")
    finally:
        client.loop_stop()
        client.disconnect()
