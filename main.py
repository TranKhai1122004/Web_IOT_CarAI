import cv2
import numpy as np
import time
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import paho.mqtt.client as mqtt

app = FastAPI()

# ==========================================
# TẠO THƯ MỤC
# ==========================================
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# ==========================================
# MQTT CONFIG
# ==========================================
MQTT_CONF = {
    "server": "0b9f6841ad1048afb1f4200aa7284c1e.s1.eu.hivemq.cloud",
    "port": 8883,
    "user": "trankhai112204",
    "pass": "Loveghita89@"
}

TOPIC_SPEED = "remote_car/speed"
TOPIC_DIRECTION = "remote_car/direction"

# ==========================================
# GLOBAL STATE
# ==========================================
class GlobalState:
    mode = "reality"

    video_source = "http://172.20.10.3:8080/video"

    is_running = True

    logs = []

    current_limit_speed = 140

    last_sign_time = 0

    is_on_crosswalk = False

    original_speed_before_cross = 140

    last_move_cmd = ""

    last_log_content = ""

    # ===== FAILSAFE =====
    prev_gray = None
    freeze_start_time = None
    emergency_stopped = False

state = GlobalState()

# ==========================================
# MQTT
# ==========================================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

mqtt_client.username_pw_set(
    MQTT_CONF["user"],
    MQTT_CONF["pass"]
)

mqtt_client.tls_set()

def on_connect(client, userdata, flags, rc, properties=None):

    if rc == 0:
        print("✅ MQTT Connected!")

    else:
        print(f"❌ MQTT Connection Failed. Code: {rc}")

mqtt_client.on_connect = on_connect

try:

    mqtt_client.connect(
        MQTT_CONF["server"],
        MQTT_CONF["port"]
    )

    mqtt_client.loop_start()

except Exception as e:

    print(f"⚠️ MQTT ERROR: {e}")

# ==========================================
# LOG
# ==========================================
def add_log(msg):

    if msg != state.last_log_content:

        t = time.strftime("%H:%M:%S")

        state.logs.append(f"[{t}] {msg}")

        if len(state.logs) > 20:
            state.logs.pop(0)

        state.last_log_content = msg

# ==========================================
# MQTT SEND
# ==========================================
def send_mqtt(topic, msg):

    try:

        mqtt_client.publish(topic, msg)

    except Exception as e:

        add_log(f"Lỗi MQTT: {e}")

# ==========================================
# EMERGENCY STOP
# ==========================================
def emergency_stop(reason):

    if not state.emergency_stopped:

        send_mqtt(TOPIC_DIRECTION, "S")

        state.is_running = False

        state.emergency_stopped = True

        add_log(f"🚨 EMERGENCY STOP: {reason}")

# ==========================================
# LOAD MODELS
# ==========================================
model_sign = YOLO("models/WalkCross_Speed_Stop_Detec.pt")

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
    "stop": 0
}

# ==========================================
# AI PROCESSING
# ==========================================
def frame_generator():

    while True:

        cap = cv2.VideoCapture(state.video_source)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while cap.isOpened():

            success, frame = cap.read()
           
            # ==========================================
            # VIDEO STOP / CAMERA LOST
            # ==========================================
            if not success:

                emergency_stop("Mất video hoặc video đã kết thúc")

                break

            view_frame = frame

            h, w, _ = frame.shape

            # ==========================================
            # DETECT FRAME FREEZE
            # ==========================================
            small = cv2.resize(frame, (64, 64))

            gray = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2GRAY
            )

            if state.prev_gray is not None:

                diff = cv2.absdiff(
                    state.prev_gray,
                    gray
                )

                motion_score = np.mean(diff)

                # Nếu frame gần như đứng yên
                if motion_score < 1.5:

                    if state.freeze_start_time is None:

                        state.freeze_start_time = time.time()

                    freeze_duration = (
                        time.time() -
                        state.freeze_start_time
                    )

                    # Đứng quá 2 giây
                    if freeze_duration > 2:

                        emergency_stop(
                            "Camera bị đứng hình"
                        )

                else:

                    # Frame chuyển động lại bình thường
                    state.freeze_start_time = None

                    state.emergency_stopped = False

            state.prev_gray = gray

            # ==========================================
            # CHẠY AI
            # ==========================================
            if state.is_running:

                res_sign = model_sign(
                    frame,
                    conf=0.45,
                    imgsz=640,
                    verbose=False
                )[0]

                res_lane = model_lane(
                    frame,
                    conf=0.45,
                    imgsz=640,
                    verbose=False,
                    task='segment'
                )[0]

                found_walk_cross = False

                # ======================================
                # SIGN DETECTION
                # ======================================
                if res_sign.boxes is not None:

                    for box in res_sign.boxes:

                        label = model_sign.names[
                            int(box.cls[0])
                        ]

                        x_c = float(box.xywh[0][0])

                        # ==============================
                        # WALK CROSS
                        # ==============================
                        if label == "walk_cross":

                            found_walk_cross = True

                            if not state.is_on_crosswalk:

                                state.is_on_crosswalk = True

                                state.original_speed_before_cross = (
                                    state.current_limit_speed
                                )

                                target = max(
                                    130,
                                    int(
                                        state.original_speed_before_cross * 0.7
                                    )
                                )

                                send_mqtt(
                                    TOPIC_SPEED,
                                    f"s{target}"
                                )

                                add_log(
                                    f"🚶 Vạch đi bộ -> Giảm tốc {target}"
                                )

                        # ==============================
                        # SPEED / STOP
                        # ==============================
                        elif label in SPEED_MAP:

                            now = time.time()

                            if (
                                x_c > (w * 0.5)
                                and
                                (now - state.last_sign_time > 3)
                            ):

                                limit = SPEED_MAP[label]

                                # STOP SIGN
                                if label == "stop":

                                    send_mqtt(
                                        TOPIC_DIRECTION,
                                        "S"
                                    )

                                    state.is_running = False

                                    add_log(
                                        "🛑 STOP SIGN -> Dừng xe"
                                    )

                                else:

                                    state.current_limit_speed = limit

                                    send_mqtt(
                                        TOPIC_SPEED,
                                        f"s{limit}"
                                    )

                                    add_log(
                                        f"📉 {label} -> {limit}"
                                    )

                                state.last_sign_time = now

                # ======================================
                # CROSS WALK END
                # ======================================
                if (
                    state.is_on_crosswalk
                    and
                    not found_walk_cross
                ):

                    state.is_on_crosswalk = False

                    send_mqtt(
                        TOPIC_SPEED,
                        f"s{state.original_speed_before_cross}"
                    )

                    add_log(
                        f"✅ Hết vạch -> Hồi tốc "
                        f"{state.original_speed_before_cross}"
                    )

                # ======================================
                # LANE FOLLOWING
                # ======================================
                if res_lane.masks is not None:

                    center_x = np.mean(
                        res_lane.masks.xyn[0][:, 0]
                    )

                    active_cmd = "F"

                    if center_x < 0.38:

                        active_cmd = "R"

                    elif center_x > 0.62:

                        active_cmd = "L"

                    if active_cmd != state.last_move_cmd:

                        send_mqtt(
                            TOPIC_DIRECTION,
                            active_cmd
                        )

                        state.last_move_cmd = active_cmd

                        dir_msg = {
                            "F": "Đi thẳng",
                            "L": "Rẽ trái",
                            "R": "Rẽ phải"
                        }

                        add_log(
                            f"Lái xe: {dir_msg[active_cmd]}"
                        )

                # ======================================
                # DRAW RESULT
                # ======================================
                view_frame = res_lane.plot(
                    boxes=False
                )

                view_frame = res_sign.plot(
                    img=view_frame
                )

            # ==========================================
            # STREAM FRAME
            # ==========================================
            _, buffer = cv2.imencode(
                '.jpg',
                view_frame
            )

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                +
                buffer.tobytes()
                +
                b'\r\n'
            )

        cap.release()

        time.sleep(0.5)

# ==========================================
# API
# ==========================================
@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return {
        "filename": file.filename
    }

# ==========================================
# SET MODE
# ==========================================
@app.get("/set_mode")
def set_mode(
    mode: str,
    file: str = ""
):

    state.mode = mode

    state.video_source = (
        "http://172.20.10.3:8080/video"
        if mode == "reality"
        else f"uploads/{file}"
    )

    # reset failsafe
    state.prev_gray = None
    state.freeze_start_time = None
    state.emergency_stopped = False

    add_log(f"Chế độ: {mode.upper()}")

    return {
        "status": "ok"
    }

# ==========================================
# SET SPEED
# ==========================================
@app.get("/set_speed")
def set_speed(val: int):

    state.current_limit_speed = val

    send_mqtt(
        TOPIC_SPEED,
        f"s{val}"
    )

    add_log(f"Cài tốc độ: {val}")

    return {
        "status": "ok"
    }

# ==========================================
# CONTROL
# ==========================================
@app.get("/control")
def control(cmd: str):

    # START AI
    if cmd == "START":

        state.is_running = True

        state.emergency_stopped = False

        state.freeze_start_time = None

        add_log("▶ BẬT AI")

    # STOP AI
    elif cmd == "STOP_AI":

        state.is_running = False

        send_mqtt(
            TOPIC_DIRECTION,
            "S"
        )

        add_log("⏸ TẮT AI - DỪNG XE")

    # MANUAL CONTROL
    else:

        send_mqtt(
            TOPIC_DIRECTION,
            cmd
        )

    return {
        "status": "ok"
    }

# ==========================================
# STATUS
# ==========================================
@app.get("/get_status")
def get_status():

    return {
        "logs": state.logs,
        "speed": state.current_limit_speed,
        "is_running": state.is_running
    }

# ==========================================
# VIDEO FEED
# ==========================================
@app.get("/video_feed")
def video_feed():

    return StreamingResponse(
        frame_generator(),
        media_type=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )

# ==========================================
# STATIC
# ==========================================
app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    import uvicorn

    try:

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="error",
            timeout_keep_alive=0
        )

    except KeyboardInterrupt:

        print("\n🛑 Đang dừng...")

        state.is_running = False

        mqtt_client.loop_stop()

        mqtt_client.disconnect()