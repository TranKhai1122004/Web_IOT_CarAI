import cv2
import numpy as np
import time
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from torch import mode
from ultralytics import YOLO
import paho.mqtt.client as mqtt

app = FastAPI()

if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Tự động tạo thư mục lưu data huấn luyện nếu chưa có
if not os.path.exists("tự_động_data"):
    os.makedirs("tự_động_data")

# =========================================================
# MQTT CONFIG
# =========================================================
MQTT_CONF = {
    # "server": "172.20.10.2",
    "server": "192.168.100.162",
    "port": 1883,
}

TOPIC_SPEED = "remote_car/speed"
TOPIC_DIRECTION = "remote_car/direction"

# =========================================================
# TUNED SERVO REAL ANGLE (Đồng bộ với ESP32 đã tuning)
# =========================================================
SERVO_LEFT = 107
SERVO_CENTER = 87
SERVO_RIGHT = 67


class GlobalState:
    mode = "reality"
    # video_source = "http://172.20.10.4:8080/video"
    video_source = "http://192.168.100.166:8080/video"
    is_running = True  # True: Chế độ tự hành (AI), False: Chế độ thủ công (Manual)
    logs = []
    current_limit_speed = 120
    last_sign_time = 0
    is_on_crosswalk = False
    original_speed_before_cross = 120
    base_motion_state = "F"
    last_speed_cmd = 0
    last_log_content = ""
    last_angle = 0.0
    last_angle_sent = SERVO_CENTER
    last_turn_time = 0.0
    last_valid_lane_width = 240.0
    camera_offset = 0


state = GlobalState()

# =========================================================
# MQTT CONNECT
# =========================================================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ MQTT Connected!")
    else:
        print(f"❌ MQTT Failed: {rc}")


mqtt_client.on_connect = on_connect

try:
    mqtt_client.connect(MQTT_CONF["server"], MQTT_CONF["port"])
    mqtt_client.loop_start()
except Exception as e:
    print(f"⚠️ MQTT ERROR: {e}")


# =========================================================
# LOGS & MQTT UTILS
# =========================================================
def add_log(msg):
    if msg != state.last_log_content:
        t = time.strftime("%H:%M:%S")
        state.logs.append(f"[{t}] {msg}")
        if len(state.logs) > 20:
            state.logs.pop(0)
        state.last_log_content = msg


def send_mqtt(topic, msg):
    try:
        mqtt_client.publish(topic, msg, qos=0)
    except Exception as e:
        add_log(f"MQTT Error: {e}")


# =========================================================
# LOAD MODEL & SPEED MAP
# =========================================================
model_sign = YOLO("models/Speed_Stop_Walk_new.pt")
model_lane = YOLO("models/LaneSeg.pt")


SPEED_MAP = {
    "speed_limit_20": 100,
    "speed_limit_30": 110,
    "speed_limit_40": 120,
    "speed_limit_50": 130,
    "speed_limit_60": 140,
    "speed_limit_70": 160,
    "speed_limit_80": 180,
    "speed_limit_90": 200,
    "speed_limit_100": 220,
    "stop": 0,
}


# =========================================================
# FRAME GENERATOR (XỬ LÝ CORE AI / STANLEY + AUTO-LOG DATA)
# =========================================================
def frame_generator():
    last_active_source = None
    cap = None

    while True:
        # Kiểm tra xem nguồn dữ liệu hệ thống có bị thay đổi từ bên ngoài (API) hay không
        if last_active_source != state.video_source:
            if cap is not None:
                cap.release()
                add_log("🔄 Đang giải phóng luồng video cũ...")

            last_active_source = state.video_source
            cap = cv2.VideoCapture(last_active_source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            add_log(f"🎬 Khởi tạo thành công nguồn mới: {last_active_source}")

        # Trường hợp thiết bị chưa sẵn sàng hoặc mất kết nối
        if cap is None or not cap.isOpened():
            add_log("❌ Không thể kết nối tới nguồn Video, thử lại sau 1 giây...")
            time.sleep(1)
            continue

        # Đọc dữ liệu từ luồng hoạt động
        success, frame = cap.read()

        if not success:
            # Nếu đang chạy giả lập video (Simulation) mà hết video -> Tự động tua lại đầu video để lặp vô hạn
            if state.mode == "simulation":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            else:
                add_log("⚠️ Luồng Reality mất tín hiệu hoặc camera ngắt kết nối.")
                if cap is not None:
                    cap.release()
                last_active_source = None  # Ép vòng lặp ngoài khởi tạo lại ở chu kỳ sau
                time.sleep(1)
                continue

        view_frame = frame.copy()
        h, w, _ = frame.shape
        mid_x = (w // 2) + state.camera_offset

        # CHỈ XỬ LÝ ĐIỀU KHIỂN AI KHI "state.is_running == True"
        if state.is_running:
            res_sign = model_sign(frame, conf=0.45, imgsz=640, verbose=False)[0]
            res_lane = model_lane(
                frame, conf=0.4, imgsz=320, verbose=False, task="segment"
            )[0]

            # --- 1. TỰ ĐỘNG TÔ MÀU LANE THEO MÔ HÌNH SEGMENTATION ---
            if res_lane.masks is not None:
                view_frame = res_lane.plot(boxes=False)

            # --- 2. NHẬN DIỆN BIỂN BÁO CẢNH BÁO & VẼ KHUNG ĐÈ LÊN ---
            found_walk_cross = False
            if res_sign.boxes is not None:
                if len(res_sign.boxes) > 0:
                    view_frame = res_sign.plot(img=view_frame)

                for box in res_sign.boxes:
                    label = model_sign.names[int(box.cls[0])]
                    x_c = float(box.xywh[0][0])
                    conf = float(box.conf[0])

                    if label == "walk_cross":
                        found_walk_cross = True
                        if not state.is_on_crosswalk:
                            state.is_on_crosswalk = True
                            state.original_speed_before_cross = (
                                state.current_limit_speed
                            )
                            target = max(
                                90, int(state.original_speed_before_cross * 0.7)
                            )
                            send_mqtt(TOPIC_SPEED, f"s{target}")
                            add_log(f"🚶 WalkCross -> Giảm tốc: {target}")

                    elif label in SPEED_MAP:
                        now = time.time()
                        if x_c > (w * 0.4) and (now - state.last_sign_time > 4):
                            limit = SPEED_MAP[label]
                            if label == "stop" and conf > 0.8:
                                send_mqtt(TOPIC_DIRECTION, "S")
                                state.base_motion_state = "S"
                                state.is_running = False
                                add_log("🛑 BIỂN BÁO STOP -> DỪNG XE & TẮT AI")
                            else:
                                state.current_limit_speed = limit
                                send_mqtt(TOPIC_SPEED, f"s{limit}")
                                add_log(f"📉 Biển báo {label} -> Tốc độ: {limit}")
                                state.last_sign_time = now

            if state.is_on_crosswalk and not found_walk_cross:
                state.is_on_crosswalk = False
                send_mqtt(TOPIC_SPEED, f"s{state.original_speed_before_cross}")
                add_log(
                    f"✅ Hết vạch qua đường -> Khôi phục: {state.original_speed_before_cross}"
                )
            # --- 3. THUẬT TOÁN ĐIỀU HƯỚNG STANLEY (ĐÃ SỬA LỖI MƯỢT MÀ) ---
            if res_lane.masks is not None:
                try:
                    # lane_mask = res_lane.masks.data[0].cpu().numpy()
                    lane_mask = np.max(res_lane.masks.data.cpu().numpy(), axis=0)
                    lane_mask = cv2.resize(lane_mask, (w, h))

                    roi_y_start, roi_y_end = int(h * 0.65), int(h * 0.95)
                    roi = lane_mask[roi_y_start:roi_y_end, :]
                    y_indices, x_indices = np.where(roi > 0.5)
                    y_indices += roi_y_start

                    if len(x_indices) > 50:
                        left_lane_mask, right_lane_mask = (
                            x_indices < mid_x,
                            x_indices >= mid_x,
                        )
                        left_x, left_y = (
                            x_indices[left_lane_mask],
                            y_indices[left_lane_mask],
                        )
                        right_x, right_y = (
                            x_indices[right_lane_mask],
                            y_indices[right_lane_mask],
                        )

                        sample_y = np.linspace(roi_y_start, roi_y_end, 6)
                        center_line_x = []

                        for y in sample_y:
                            left_points = left_x[np.abs(left_y - y) < 15]
                            right_points = right_x[np.abs(right_y - y) < 15]
                            has_left, has_right = (
                                len(left_points) > 3,
                                len(right_points) > 3,
                            )

                            if has_left and has_right:
                                measured_width = np.mean(right_points) - np.mean(
                                    left_points
                                )
                                if 140 < measured_width < 380:
                                    state.last_valid_lane_width = measured_width
                                cx = (np.mean(left_points) + np.mean(right_points)) / 2
                                center_line_x.append(cx)
                            elif has_left:
                                cx = np.mean(left_points) + (
                                    state.last_valid_lane_width / 2
                                )
                                center_line_x.append(cx)
                            elif has_right:
                                cx = np.mean(right_points) - (
                                    state.last_valid_lane_width / 2
                                )
                                center_line_x.append(cx)

                        center_line_x = np.array(center_line_x)

                        if len(center_line_x) >= 4 and state.base_motion_state != "S":

                            # =========================================================
                            # 1. TÍNH TOÁN CROSS-TRACK & HEADING
                            # =========================================================
                            target_near_x = center_line_x[-1]

                            # Sai số lệch tâm lane
                            error_e = (target_near_x - mid_x) / (w / 2.0)

                            # Góc hướng lane
                            dx = center_line_x[0] - center_line_x[-1]
                            dy = sample_y[0] - sample_y[-1]

                            heading_theta = np.arctan2(dx, -dy)
                            heading_deg = np.degrees(heading_theta)

                            # =========================================================
                            # 2. HUMAN-LIKE STABILIZER
                            # =========================================================
                            if abs(heading_deg) < 4 and abs(error_e) < 0.12:
                                error_e = 0

                            # =========================================================
                            # 3. DEAD-BAND ANTI OSCILLATION
                            # =========================================================
                            if abs(error_e) < 0.06:
                                error_e = 0

                            # =========================================================
                            # 4. STANLEY CONTROL
                            # =========================================================
                            calc_v = max(100, state.current_limit_speed)
                            k_gain = 2.2

                            steering_angle_rad = -heading_theta + np.arctan2(
                                k_gain * error_e, calc_v
                            )

                            steering_angle_deg = np.clip(
                                np.degrees(steering_angle_rad), -30, 30
                            )

                            # =========================================================
                            # 5. STRAIGHT PRIORITY MODE
                            # =========================================================
                            if abs(heading_deg) < 2.5 and abs(error_e) < 0.08:
                                steering_angle_deg = 0

                            # =========================================================
                            # 6. EMA FILTER
                            # =========================================================
                            alpha = 0.80

                            steering_angle_deg = (
                                alpha * state.last_angle
                                + (1 - alpha) * steering_angle_deg
                            )

                            state.last_angle = steering_angle_deg

                            # 4. Tính toán Tốc độ tối ưu theo góc rẽ
                            abs_angle = abs(steering_angle_deg)
                            base_speed = state.current_limit_speed
                            if abs_angle > 10:  # Vào cua gắt
                                target_speed = max(100, int(base_speed * 0.65))
                            else:
                                target_speed = base_speed

                            if target_speed != state.last_speed_cmd:
                                send_mqtt(TOPIC_SPEED, f"s{target_speed}")
                                state.last_speed_cmd = target_speed

                            # 5. ÁNH XẠ GÓC SERVO CHUẨN XÁC (Sửa lỗi lệch tâm hình học)
                            now = time.time()
                            if (
                                now - state.last_turn_time > 0.03
                            ):  # Tăng tần suất gửi lái lên ~33Hz (0.03s)
                                if steering_angle_deg >= 0:
                                    # Cua Trái: từ CENTER (87) tiến dần lên LEFT (107)
                                    servo_angle = int(
                                        SERVO_CENTER
                                        + (steering_angle_deg / 30.0)
                                        * (SERVO_LEFT - SERVO_CENTER)
                                    )
                                else:
                                    # Cua Phải: từ CENTER (87) lùi dần về RIGHT (67)
                                    servo_angle = int(
                                        SERVO_CENTER
                                        + (steering_angle_deg / 30.0)
                                        * (SERVO_CENTER - SERVO_RIGHT)
                                    )

                                servo_angle = np.clip(
                                    servo_angle, SERVO_RIGHT, SERVO_LEFT
                                )

                                # Chỉ gửi khi góc thực sự thay đổi để tránh tràn băng thông MQTT
                                if abs(servo_angle - state.last_angle_sent) >= 1:
                                    send_mqtt(TOPIC_DIRECTION, str(servo_angle))
                                    state.last_angle_sent = servo_angle
                                    add_log(
                                        f"🎯 Stanley -> Servo: {servo_angle}° | Err: {error_e:.2f}"
                                    )

                                    # -----------------------------------------------------------
                                    # HACK LOGIC: TỰ ĐỘNG CHỤP VÀ LƯU DATA CHO MACHINE/DEEP LEARNING
                                    # -----------------------------------------------------------
                                    if state.mode == "reality":  # Chỉ lưu khi chạy trên xe thực tế
                                        timestamp = int(time.time() * 1000)
                                        file_name = f"tự_động_data/frame_{timestamp}_goc_{servo_angle}.jpg"
                                        # Lưu khung hình thô gốc (chưa vẽ đè vạch lane/bbox của YOLO) để làm data sạch
                                        cv2.imwrite(file_name, frame)
                                    # -----------------------------------------------------------

                                state.last_turn_time = now

                            for i in range(len(center_line_x)):
                                cv2.circle(
                                    view_frame,
                                    (int(center_line_x[i]), int(sample_y[i])),
                                    5,
                                    (0, 0, 255),
                                    -1,
                                )
                        else:
                            if state.last_angle_sent != SERVO_CENTER:
                                send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
                                state.last_angle_sent = SERVO_CENTER

                            state.last_angle = 0
                            add_log("⚠️ MẤT LANE -> TRẢ LÁI VỀ GIỮA")

                except Exception as e:
                    print(f"⚠️ Stanley Error: {e}")
        # Hãm nhẹ tốc độ xử lý khi chạy Simulation để tránh ngốn 100% tài nguyên CPU không cần thiết
        if state.mode == "simulation":
            time.sleep(0.02)  # Khống chế ở mức ~40-50fps mượt mà
        # 1. Vẽ tâm ảnh bằng màu xanh lá (Green)
        cv2.line(view_frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)

        # 2. Vẽ tâm thực tế của xe bằng màu xanh dương (Blue)
        cv2.line(view_frame, (int(mid_x), 0), (int(mid_x), h), (255, 0, 0), 2)
        # Gửi hình ảnh luồng video về giao diện Web HTML
        _, buffer = cv2.imencode(".jpg", view_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# =========================================================
# CONTROLLER ENDPOINTS
# =========================================================
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}


@app.get("/set_mode")
def set_mode(mode: str, file: str = ""):
    state.mode = mode
    # state.video_source = (
    #     "http://172.20.10.4:8080/video" if mode == "reality" else f"uploads/{file}"
    # )
    state.video_source = (
        "http://192.168.100.166:8080/video" if mode == "reality" else f"uploads/{file}"
    )

    add_log(f"Chế độ -> {mode}")
    return {"status": "ok"}


@app.get("/set_speed")
def set_speed(val: int):
    state.current_limit_speed = val
    send_mqtt(TOPIC_SPEED, f"s{val}")
    add_log(f"Cài đặt tốc độ trần -> {val}")
    return {"status": "ok"}


@app.get("/control")
def control(cmd: str):
    if cmd == "START":
        state.is_running = True
        state.base_motion_state = "F"

        send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
        add_log("▶️ KÍCH HOẠT HỆ THỐNG TỰ HÀNH AI")

    elif cmd == "STOP_AI":
        state.is_running = False
        state.base_motion_state = "S"
        send_mqtt(TOPIC_SPEED, "S")
        send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
        add_log("⏸ KHÓA AI -> CHUYỂN SANG ĐIỀU KHIỂN THỦ CÔNG")

    else:
        if not state.is_running:
            if cmd in ["F", "B", "S"]:
                state.base_motion_state = cmd
                send_mqtt(TOPIC_DIRECTION, cmd)
                add_log(f"Thủ công -> Di chuyển: {cmd}")
            elif cmd == "L":
                send_mqtt(TOPIC_DIRECTION, str(SERVO_LEFT))
                add_log(f"Thủ công -> Bẻ lái Trái: {SERVO_LEFT}°")
            elif cmd == "R":
                send_mqtt(TOPIC_DIRECTION, str(SERVO_RIGHT))
                add_log(f"Thủ công -> Bẻ lái Phải: {SERVO_RIGHT}°")
            elif cmd == "G":
                send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
                add_log(f"Thủ công -> Thẳng lái: {SERVO_CENTER}°")
        else:
            add_log("⚠️ Vui lòng bấm 'STOP AI' trước khi muốn lái thủ công!")

    return {"status": "ok"}


@app.get("/get_status")
def get_status():
    return {
        "logs": state.logs,
        "speed": state.current_limit_speed,
        "is_running": state.is_running,
    }


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(
            app, host="127.0.0.1", port=8000, log_level="error", timeout_keep_alive=0
        )
    except KeyboardInterrupt:
        state.is_running = False
        mqtt_client.loop_stop()
        mqtt_client.disconnect()