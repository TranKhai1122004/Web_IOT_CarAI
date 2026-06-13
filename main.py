import cv2
import numpy as np
import time
import os
import shutil
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import paho.mqtt.client as mqtt

app = FastAPI()

if not os.path.exists("uploads"):
    os.makedirs("uploads")

MQTT_CONF = {"server": "192.168.100.162", "port": 1883}
TOPIC_SPEED = "remote_car/speed"
TOPIC_DIRECTION = "remote_car/direction"

# Cấu hình cứng của xe: Lớn là TRÁI, Nhỏ là PHẢI
SERVO_LEFT = 115
SERVO_CENTER = 87
SERVO_RIGHT = 59


class GlobalState:
    def __init__(self):
        self.mode = "reality"
        self.video_source = "http://192.168.100.166:8080/video"
        self.is_running = True
        self.logs = []
        self.current_limit_speed = 120
        self.last_sign_time = 0
        self.is_on_crosswalk = False
        self.original_speed_before_cross = 120
        self.base_motion_state = "F"
        self.last_speed_cmd = 0
        self.last_log_content = ""
        self.last_angle = 0.0
        self.last_angle_sent = SERVO_CENTER
        self.actual_servo_angle = SERVO_CENTER
        self.last_turn_time = 0.0
        self.last_valid_lane_width = 360
        self.camera_offset = 0
        self.latest_processed_frame = None


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


def get_perspective_matrices(w, h):
    src_pts = np.float32([[162, 168], [458, 172], [2, 473], [591, 477]])
    dst_pts = np.float32(
        [
            [140, 0],      # top left
            [500, 0],      # top right
            [500, 480],    # bottom right
            [140, 480],    # bottom left
        ]
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    return M, Minv, src_pts


# =========================================================
# BACKGROUND THREAD XỬ LÝ AI + BIRD'S EYE VIEW
# =========================================================
def ai_processing_loop():
    last_active_source = None
    cap = None

    while True:
        if last_active_source != state.video_source:
            if cap is not None:
                cap.release()
            last_active_source = state.video_source
            cap = cv2.VideoCapture(last_active_source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            add_log(f"🎬 Khởi tạo nguồn Video: {last_active_source}")

        if cap is None or not cap.isOpened():
            time.sleep(1)
            continue

        success, frame = cap.read()
        if not success:
            if state.mode == "simulation":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                if cap is not None:
                    cap.release()
                last_active_source = None
                time.sleep(0.5)
                continue

        view_frame = frame.copy()
        h, w, _ = frame.shape

        M, Minv, src_pts = get_perspective_matrices(w, h)
        
        # CHUẨN HÓA TÂM TRỤC XE TRÊN HỆ BEV
        mid_x_bev = 320 + state.camera_offset

        if state.is_running:
            res_sign = model_sign(frame, conf=0.45, imgsz=640, verbose=False)[0]
            res_lane = model_lane(
                frame, conf=0.4, imgsz=320, verbose=False, task="segment"
            )[0]

            if res_lane.masks is not None:
                view_frame = res_lane.plot(boxes=False)

            # --- XỬ LÝ BIỂN BÁO ---
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
                                state.is_running = False
                                state.base_motion_state = "S"
                                send_mqtt(TOPIC_SPEED, "S")
                                send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
                                state.last_speed_cmd = "S"
                                state.last_angle_sent = SERVO_CENTER
                                add_log("🛑 BIỂN BÁO STOP -> DỪNG XE & TẮT AI TỰ ĐỘNG")
                            else:
                                state.current_limit_speed = limit
                                send_mqtt(TOPIC_SPEED, f"s{limit}")
                                add_log(f"📉 Biển báo {label} -> Tốc độ: {limit}")
                                state.last_sign_time = now

            if state.is_on_crosswalk and not found_walk_cross:
                state.is_on_crosswalk = False
                send_mqtt(TOPIC_SPEED, f"s{state.original_speed_before_cross}")
                add_log(
                    f"✅ Hết vạch -> Khôi phục: {state.original_speed_before_cross}"
                )

            # --- XỬ LÝ LANE VỚI BIRD'S EYE VIEW ---
            if res_lane.masks is not None and state.base_motion_state != "S":
                try:
                    lane_mask = np.max(res_lane.masks.data.cpu().numpy(), axis=0)
                    lane_mask = cv2.resize(lane_mask, (w, h))

                    warped_mask = cv2.warpPerspective(
                        lane_mask, M, (w, h), flags=cv2.INTER_LINEAR
                    )

                    roi_y_start, roi_y_end = 50, 450
                    roi = warped_mask[roi_y_start:roi_y_end, :]
                    y_indices, x_indices = np.where(roi > 0.5)
                    y_indices += roi_y_start

                    if len(x_indices) > 50:
                        left_lane_mask = x_indices < mid_x_bev
                        right_lane_mask = x_indices >= mid_x_bev
                        left_x, left_y = (
                            x_indices[left_lane_mask],
                            y_indices[left_lane_mask],
                        )
                        right_x, right_y = (
                            x_indices[right_lane_mask],
                            y_indices[right_lane_mask],
                        )

                        sample_y = np.linspace(roi_y_start + 30, roi_y_end - 20, 7)
                        center_line_x = []

                        for y in sample_y:
                            left_points = left_x[np.abs(left_y - y) < 20]
                            right_points = right_x[np.abs(right_y - y) < 20]
                            has_left = len(left_points) > 3
                            has_right = len(right_points) > 3

                            if has_left and has_right:
                                measured_width = np.mean(right_points) - np.mean(left_points)
                                if 280 < measured_width < 440:
                                    state.last_valid_lane_width = measured_width
                                cx = (np.mean(left_points) + np.mean(right_points)) / 2
                                center_line_x.append(cx)
                            elif has_left:
                                cx = np.mean(left_points) + (state.last_valid_lane_width / 2)
                                center_line_x.append(cx)
                            elif has_right:
                                cx = np.mean(right_points) - (state.last_valid_lane_width / 2)
                                center_line_x.append(cx)

                        center_line_x = np.array(center_line_x)

                        if len(center_line_x) >= 2:
                            target_x = center_line_x[-1]
                            error_pixels = target_x - mid_x_bev

                            deadzone_pixels = 4

                            if abs(error_pixels) < deadzone_pixels:
                                servo_angle = SERVO_CENTER
                            else:
                                if error_pixels > 0:
                                    effective_error = error_pixels - deadzone_pixels
                                else:
                                    effective_error = error_pixels + deadzone_pixels

                                max_effective_error = 150
                                effective_error = np.clip(
                                    effective_error,
                                    -max_effective_error,
                                    max_effective_error,
                                )

                                # ĐẢO LẠI THEO HỆ THỐNG ĐẶT NGƯỢC CỦA KHẢI:
                                # err > 0 (Tâm lệch PHẢI) -> Tăng góc tiến về phía SERVO_LEFT
                                # err < 0 (Tâm lệch TRÁI) -> Giảm góc tiến về phía SERVO_RIGHT
                                if effective_error > 0:  
                                    ratio = effective_error / max_effective_error
                                    servo_angle = SERVO_CENTER + ratio * (
                                        SERVO_LEFT - SERVO_CENTER
                                    )
                                else:  
                                    ratio = abs(effective_error) / max_effective_error
                                    servo_angle = SERVO_CENTER - ratio * (
                                        SERVO_CENTER - SERVO_RIGHT
                                    )

                            servo_angle = int(
                                np.clip(servo_angle, SERVO_RIGHT, SERVO_LEFT)
                            )

                            # --- Bộ lọc thích ứng Dynamic Alpha ---
                            abs_err = abs(error_pixels)
                            if abs_err > 60:    
                                alpha = 0.25 
                            elif abs_err > 25:  
                                alpha = 0.50 
                            else:               
                                alpha = 0.75 

                            state.actual_servo_angle = int(
                                alpha * state.actual_servo_angle
                                + (1 - alpha) * servo_angle
                            )
                            final_servo = np.clip(
                                state.actual_servo_angle, SERVO_RIGHT, SERVO_LEFT
                            )

                            # Điều tốc thông minh
                            abs_error_pixels = abs(error_pixels)
                            base_speed = state.current_limit_speed

                            if abs_error_pixels > 60:
                                speed_factor = max(
                                    0.65, 1.0 - (abs_error_pixels - 60) / 200
                                )
                                target_speed = max(95, int(base_speed * speed_factor))
                            else:
                                target_speed = base_speed

                            if f"s{target_speed}" != str(state.last_speed_cmd):
                                send_mqtt(TOPIC_SPEED, f"s{target_speed}")
                                state.last_speed_cmd = f"s{target_speed}"

                            # Gửi lệnh lái mượt định kỳ 0.03s
                            now = time.time()
                            if now - state.last_turn_time > 0.03:
                                print(
                                    f" [CONTROL] width={state.last_valid_lane_width:.1f} "
                                    f"err={error_pixels:.1f} "
                                    f"servo={final_servo}"
                                )
                                send_mqtt(TOPIC_DIRECTION, str(final_servo))
                                state.last_angle_sent = final_servo
                                if abs(final_servo - SERVO_CENTER) > 2:
                                    add_log(
                                        f"🚗 Lane -> Servo:{final_servo}° Err:{error_pixels:.0f}px"
                                    )
                                state.last_turn_time = now

                        else:
                            if state.last_angle_sent != SERVO_CENTER:
                                send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
                                state.last_angle_sent = SERVO_CENTER
                            state.actual_servo_angle = SERVO_CENTER

                        # Vẽ visualization chuẩn tọa độ gốc
                        for i in range(len(center_line_x)):
                            pt_bev = np.array(
                                [[[center_line_x[i], sample_y[i]]]], dtype=np.float32
                            )
                            pt_original = cv2.perspectiveTransform(pt_bev, Minv)[0][0]
                            color = (
                                (0, 255, 0)
                                if i == len(center_line_x) - 1
                                else (0, 0, 255)
                            )
                            cv2.circle(
                                view_frame,
                                (int(pt_original[0]), int(pt_original[1])),
                                8,
                                color,
                                -1,
                            )

                        if len(center_line_x) >= 2:
                            target_bev = np.array(
                                [[[target_x, sample_y[-1]]]], dtype=np.float32
                            )
                            target_original = cv2.perspectiveTransform(
                                target_bev, Minv
                            )[0][0]
                            cv2.circle(
                                view_frame,
                                (int(target_original[0]), int(target_original[1])),
                                12,
                                (0, 255, 255),
                                3,
                            )

                except Exception as e:
                    print(f"⚠️ Lane Control Error: {e}")

        # Vẽ các đường tham chiếu hình thang ROI lên giao diện nhìn
        cv2.polylines(view_frame, [src_pts.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.line(view_frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        
        # Đồng bộ đường line xanh dương chỉ thị mid_x
        mid_x_original = (w // 2) + state.camera_offset
        cv2.line(view_frame, (int(mid_x_original), 0), (int(mid_x_original), h), (255, 0, 0), 2)

        state.latest_processed_frame = view_frame.copy()

        if state.mode == "simulation":
            time.sleep(0.02)


ai_thread = threading.Thread(target=ai_processing_loop, daemon=True)
ai_thread.start()


# =========================================================
# FASTAPI FRAME GENERATOR & ROUTER
# =========================================================
def frame_generator():
    while True:
        if state.latest_processed_frame is not None:
            _, buffer = cv2.imencode(".jpg", state.latest_processed_frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


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
        send_mqtt(TOPIC_SPEED, f"s{state.current_limit_speed}")
        send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
        state.last_speed_cmd = f"s{state.current_limit_speed}"
        state.last_angle_sent = SERVO_CENTER
        add_log("🚀 KÍCH HOẠT HỆ THỐNG AI AUTOMOTOR -> XE BẮT ĐẦU CHẠY")
        return {"status": "success", "mode": "AI_STARTED"}

    elif cmd == "STOP_AI":
        state.is_running = False
        state.base_motion_state = "S"
        send_mqtt(TOPIC_SPEED, "S")
        send_mqtt(TOPIC_DIRECTION, str(SERVO_CENTER))
        state.last_speed_cmd = "S"
        state.last_angle_sent = SERVO_CENTER
        add_log("🛑 ĐÃ TẮT AI TIẾN TRÌNH -> CHUYỂN SANG ĐIỀU KHIỂN TAY")
        return {"status": "success", "mode": "AI_STOPPED"}

    if state.is_running:
        return {
            "status": "ignored",
            "reason": "AI mode is running. Please STOP AI first.",
        }

    max_speed = getattr(state, "current_limit_speed", 120)

    if cmd in ["F", "B", "S"]:
        state.base_motion_state = cmd
        if cmd == "F":
            cmd_speed = f"s{max_speed}"
        elif cmd == "B":
            cmd_speed = f"b{max_speed}"
        else:
            cmd_speed = "S"

        if cmd_speed != str(state.last_speed_cmd):
            send_mqtt(TOPIC_SPEED, cmd_speed)
            state.last_speed_cmd = cmd_speed

    elif cmd in ["L", "R", "G"]:
        if cmd == "L":
            servo_angle = SERVO_LEFT
        elif cmd == "R":
            servo_angle = SERVO_RIGHT
        else:
            servo_angle = SERVO_CENTER

        if servo_angle != state.last_angle_sent:
            send_mqtt(TOPIC_DIRECTION, str(servo_angle))
            state.last_angle_sent = servo_angle

    return {"status": "success"}


@app.get("/get_status")
def get_status():
    return {
        "logs": state.logs,
        "speed": state.current_limit_speed,
        "is_running": state.is_running,
    }


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