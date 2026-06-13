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
    current_limit_speed = 100
    last_sign_time = 0
    is_on_crosswalk = False
    original_speed_before_cross = 100
    base_motion_state = "F"
    last_speed_cmd = 0
    last_log_content = ""
    last_angle = 0.0
    last_angle_sent = SERVO_CENTER
    last_turn_time = 0.0
    last_valid_lane_width = 600.0
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
model_sign = YOLO("models/Speed_Stop_Cross.pt")
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
# FRAME GENERATOR (XỬ LÝ CORE AI / STANLEY - ĐÃ FIX KẸT LUỒNG)
# =========================================================
def frame_generator():
    last_active_source = None
    cap = None

    while True:
        if last_active_source != state.video_source:
            if cap is not None:
                cap.release()
                add_log("🔄 Đang giải phóng luồng video cũ...")

            last_active_source = state.video_source
            cap = cv2.VideoCapture(last_active_source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            add_log(f"🎬 Khởi tạo thành công nguồn mới: {last_active_source}")

        if cap is None or not cap.isOpened():
            add_log("❌ Không thể kết nối tới nguồn Video, thử lại sau 1 giây...")
            time.sleep(1)
            continue

        success, frame = cap.read()

        if not success:
            if state.mode == "simulation":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            else:
                add_log("⚠️ Luồng Reality mất tín hiệu hoặc camera ngắt kết nối.")
                if cap is not None:
                    cap.release()
                last_active_source = None
                time.sleep(1)
                continue

        view_frame = frame.copy()
        h, w, _ = frame.shape
        mid_x = (w // 2) + state.camera_offset
        roi_y_start, roi_y_end = int(h * 0.55), int(
            h * 0.95
        )  # 🔧 Mở rộng ROI nhìn gần hơn để phản ứng sớm hơn ở cua
        if state.is_running:
            res_sign = model_sign(frame, conf=0.45, imgsz=640, verbose=False)[0]
            res_lane = model_lane(
                frame, conf=0.4, imgsz=640, verbose=False, task="segment"
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

            # --- 3. THUẬT TOÁN ĐIỀU HƯỚNG STANLEY (TỐI ƯU CHO SA BÀN HẸP) ---
            if res_lane.masks is not None:
                try:
                    lane_mask = np.max(res_lane.masks.data.cpu().numpy(), axis=0)
                    lane_mask = cv2.resize(lane_mask, (w, h))

                    roi_y_start, roi_y_end = int(h * 0.55), int(
                        h * 0.95
                    )  # 🔧 Đồng bộ với ROI bên trên
                    roi = lane_mask[roi_y_start:roi_y_end, :]
                    y_indices, x_indices = np.where(roi > 0.5)
                    y_indices += roi_y_start

                    # 🌟 FIX 2: Hạ từ 35 xuống 15 điểm. Lane hẹp diện tích pixel sẽ ít đi, giữ 35 xe sẽ bị mất lane liên tục.
                    if len(x_indices) > 15:
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

                        sample_y = np.linspace(
                            roi_y_start, roi_y_end, 8
                        )  # 🔧 Tăng số điểm sample để tính heading chính xác hơn ở cua
                        center_line_x = []

                        for y in sample_y:
                            # 🌟 FIX 3: Hạ từ 15 pixel xuống 6 pixel tầm quét ngang để không bị dính sang vạch đối diện khi đường hẹp
                            left_points = left_x[np.abs(left_y - y) < 6]
                            right_points = right_x[np.abs(right_y - y) < 6]
                            has_left, has_right = (
                                len(left_points) > 3,
                                len(right_points) > 3,
                            )

                            if has_left and has_right:
                                measured_width = np.mean(right_points) - np.mean(
                                    left_points
                                )
                                print(f"LaneWidth={measured_width:.0f}")

                                if 500 < measured_width < 700:
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
                            target_near_x = center_line_x[
                                -1
                            ]  # 🔧 Dùng điểm gần hơn (index -2 thay vì -3) để phản ứng cua nhanh hơn, tránh nhắm vào điểm đã ra ngoài

                            # Sai số lệch tâm lane
                            error_e = (target_near_x - mid_x) / (w / 2.0)

                            # Góc hướng lane
                            dx = center_line_x[0] - center_line_x[-1]
                            dy = sample_y[0] - sample_y[-1]

                            heading_theta = np.arctan2(dx, -dy)
                            heading_deg = np.degrees(heading_theta)

                            # =========================================================
                            # 2. HUMAN-LIKE STABILIZER & DEAD-BAND (SIẾT CHẶT CHO ĐƯỜNG HẸP)
                            # =========================================================
                            # 🌟 FIX 5: Hạ ngưỡng nịnh lái thẳng từ 0.12 xuống 0.04. Đường hẹp lệch tí là phải sửa ngay, không cho thả trôi.
                            if abs(heading_deg) < 3.0 and abs(error_e) < 0.04:
                                error_e = 0

                            if abs(error_e) < 0.03:
                                error_e = 0

                            # =========================================================
                            # 4. STANLEY CONTROL
                            # =========================================================
                            calc_v = max(100, state.current_limit_speed)
                            v_sim = calc_v / 100.0

                            # 🌟 FIX 6: Hạ Gain từ 1.5 xuống 1.0. Đường hẹp sai số nhạy hơn, để gain cao xe văng lái bay ra ngoài sa bàn.
                            k_gain = 0.8  # 🔧 Hạ gain từ 1.3 -> 1.0: xe thực tế có độ trễ cơ khí, gain cao gây overshoot & xe lao ra ngoài

                            steering_angle_rad = -heading_theta - np.arctan2(
                                k_gain * error_e, v_sim
                            )

                            steering_angle_deg = np.clip(
                                np.degrees(steering_angle_rad), -30, 30
                            )

                            # =========================================================
                            # 5. STRAIGHT PRIORITY MODE
                            # =========================================================
                            # 🌟 FIX 7: Đồng bộ ép lái thẳng dịu hơn
                            if abs(heading_deg) < 2.0 and abs(error_e) < 0.03:
                                steering_angle_deg = 0

                            # =========================================================
                            # 6. EMA FILTER (BỘ LỌC CHUẨN ĐÃ ĐƯỢC ĐỊNH NGHĨA LẠI)
                            # =========================================================
                            # 🔧 EMA thích nghi: cua gắt thì responsive hơn, đường thẳng thì smooth hơn // 0,85  0,6
                            alpha = 0.85 if abs(steering_angle_deg) > 8 else 0.6
                            steering_angle_deg = (
                                1 - alpha
                            ) * state.last_angle + alpha * steering_angle_deg

                            state.last_angle = steering_angle_deg
                            print(
                                f"E={error_e:.3f} "
                                f"H={heading_deg:.1f} "
                                f"A={steering_angle_deg:.1f} "
                                f"S={servo_angle if 'servo_angle' in locals() else 0}"
                            )

                            # 4. Tính toán Tốc độ tối ưu theo góc rẽ
                            abs_angle = abs(steering_angle_deg)
                            base_speed = state.current_limit_speed
                            if abs_angle > 15:  # 🔧 Cua rất gắt: giảm nhiều
                                target_speed = max(100, int(base_speed * 0.60))
                            elif (
                                abs_angle > 7
                            ):  # 🔧 Bắt đầu vào cua: giảm nhẹ (hạ từ 10 xuống 7 độ)
                                target_speed = max(100, int(base_speed * 0.75))
                            else:
                                target_speed = base_speed

                            if target_speed != state.last_speed_cmd:
                                send_mqtt(TOPIC_SPEED, f"s{target_speed}")
                                state.last_speed_cmd = target_speed

                            # 5. ÁNH XẠ GÓC SERVO CHUẨN XÁC
                            now = time.time()
                            if (
                                now - state.last_turn_time > 0.03
                            ):  # Tần suất gửi lái ~33Hz
                                if steering_angle_deg >= 0:
                                    servo_angle = int(
                                        SERVO_CENTER
                                        + (steering_angle_deg / 30.0)
                                        * (SERVO_LEFT - SERVO_CENTER)
                                    )
                                else:
                                    servo_angle = int(
                                        SERVO_CENTER
                                        + (steering_angle_deg / 30.0)
                                        * (SERVO_CENTER - SERVO_RIGHT)
                                    )

                                servo_angle = np.clip(
                                    servo_angle, SERVO_RIGHT, SERVO_LEFT
                                )

                                if (
                                    abs(servo_angle - state.last_angle_sent) >= 3
                                ):  # 🔧 Tăng deadband từ 2 -> 3 độ để giảm rung lái servo
                                    send_mqtt(TOPIC_DIRECTION, str(servo_angle))
                                    state.last_angle_sent = servo_angle
                                    add_log(
                                        f"🎯 Stanley -> Servo: {servo_angle}° | Err: {error_e:.2f}"
                                    )
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
                            # Nhánh khi nhận diện được điểm nhưng KHÔNG ĐỦ 4 ĐIỂM TÂM
                            if state.last_angle_sent != SERVO_CENTER:
                                send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
                                state.last_angle_sent = SERVO_CENTER
                            state.last_angle = 0
                    else:
                        # Nhánh ELSE khi TRƯỜNG HỢP MẤT LANE HOÀN TOÀN (len(x_indices) <= 15)
                        if state.last_angle_sent != SERVO_CENTER:
                            send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
                            state.last_angle_sent = SERVO_CENTER

                        state.last_angle = 0
                        add_log("⚠️ MẤT LANE -> TRẢ LÁI VỀ GIỮA")

                except Exception as e:
                    print(f"⚠️ Stanley Error: {e}")

        if state.mode == "simulation":
            time.sleep(0.02)
        cv2.line(
            view_frame,
            (0, roi_y_start),
            (w, roi_y_start),
            (255, 255, 0),
            2,
        )

        cv2.line(
            view_frame,
            (0, roi_y_end),
            (w, roi_y_end),
            (255, 255, 0),
            2,
        )
        cv2.line(view_frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(view_frame, (int(mid_x), 0), (int(mid_x), h), (255, 0, 0), 2)

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
def control_car(cmd: str):
    if cmd == "START":
        state.is_running = True
        state.base_motion_state = "F"
        send_mqtt(TOPIC_DIRECTION, "F")
        send_mqtt(TOPIC_SPEED, f"s{state.current_limit_speed}")

        add_log("🤖 AI MODE STARTED")

        return {"status": "success", "ai_running": True}

    elif cmd == "STOP_AI":
        state.is_running = False

        send_mqtt(TOPIC_DIRECTION, "S")
        send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))

        add_log("🛑 AI STOPPED -> MANUAL MODE")

        return {"status": "success", "ai_running": False}
    if getattr(state, "is_running", False):
        return {"status": "ignored", "reason": "AI mode is running"}

    max_speed = getattr(state, "current_limit_speed", 120)

    # === SỬA LẠI ĐOẠN ĐIỀU KHIỂN MOTOR (F, B, S) ===
    if cmd in ["F", "B", "S"]:
        # 1. Gửi lệnh HƯỚNG (F/B/S) vào TOPIC_DIRECTION trước
        send_mqtt(TOPIC_DIRECTION, cmd)

        # 2. Nếu là Tiến hoặc Lùi thì gửi thêm TỐC ĐỘ vào TOPIC_SPEED
        if cmd == "F":
            cmd_speed = f"s{max_speed}"
            log_msg = f"Thủ công -> Phát lệnh TIẾN với tốc độ: {max_speed}"
            send_mqtt(TOPIC_SPEED, cmd_speed)
            state.last_speed_cmd = cmd_speed
        elif cmd == "B":
            cmd_speed = f"s{max_speed}"  # ESP32 nhận chữ 's' để cắt số tốc độ, hướng đã xử lý bằng lệnh 'B' ở trên rồi
            log_msg = f"Thủ công -> Phát lệnh LÙI với tốc độ: {max_speed}"
            send_mqtt(TOPIC_SPEED, cmd_speed)
            state.last_speed_cmd = cmd_speed
        else:
            log_msg = "Thủ công -> Dừng xe"
            state.last_speed_cmd = "S"

        add_log(log_msg)

    elif cmd in ["L", "R", "G"]:
        current_steer = getattr(state, "last_steer_cmd", "G")

        if cmd == "L":
            # Nếu đang rẽ trái rồi mà bấm L tiếp -> Trả thẳng (G)
            if current_steer == "L":
                target_cmd = "G"
                servo_angle = SERVO_CENTER
                log_msg = f"Thủ công -> Bấm lại Trái: Trả thẳng lái về {servo_angle}°"
            else:
                target_cmd = "L"
                servo_angle = SERVO_LEFT
                log_msg = f"Thủ công -> Bẻ lái Trái: {servo_angle}°"

        elif cmd == "R":
            # Nếu đang rẽ phải rồi mà bấm R tiếp -> Trả thẳng (G)
            if current_steer == "R":
                target_cmd = "G"
                servo_angle = SERVO_CENTER
                log_msg = f"Thủ công -> Bấm lại Phải: Trả thẳng lái về {servo_angle}°"
            else:
                target_cmd = "R"
                servo_angle = SERVO_RIGHT
                log_msg = f"Thủ công -> Bẻ lái Phải: {servo_angle}°"
        else:
            # Nếu gọi trực tiếp lệnh "G"
            target_cmd = "G"
            servo_angle = SERVO_CENTER
            log_msg = f"Thủ công -> Lệnh trả thẳng: {servo_angle}°"

        # Gửi dữ liệu đi và cập nhật trạng thái
        send_mqtt(TOPIC_DIRECTION, target_cmd)  # Gửi chữ "L", "R" hoặc "G"
        state.last_steer_cmd = target_cmd
        state.last_angle_sent = (
            servo_angle  # Lưu lại góc servo tương ứng để đồng bộ đồng hồ hiển thị
        )
        add_log(log_msg)
    return {
        "status": "success",
        "current_speed_cmd": getattr(state, "last_speed_cmd", "S"),
        "current_servo_angle": int(getattr(state, "last_angle_sent", SERVO_CENTER)),
    }


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
