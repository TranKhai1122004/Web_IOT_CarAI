#include <WiFi.h>
#include <PubSubClient.h>
#include <NewPing.h>
#include <ESP32Servo.h>

// =====================================================
// WIFI
// =====================================================
// const char* ssid = "Ipc";
// const char* pass = "2212202444";
const char* ssid = "Nha";
const char* pass = "Tran1204@";
// =====================================================
// MQTT
// =====================================================
// const char* mqtt_server = "172.20.10.2";
const char* mqtt_server = "192.168.100.162";
const int mqtt_port = 1883;

const char* TOPIC_DIRECTION = "remote_car/direction";
const char* TOPIC_SPEED     = "remote_car/speed";
const char* TOPIC_STATUS    = "remote_car/status";

// =====================================================
// MOTOR L298N
// =====================================================
// LEFT
#define ENA 25
#define IN1 26
#define IN2 27

// RIGHT
#define ENB 14
#define IN3 12
#define IN4 13

// =====================================================
// SERVO
// =====================================================
#define SERVO_PIN 18

#define SERVO_LEFT    102
#define SERVO_CENTER  87
#define SERVO_RIGHT   67

Servo steeringServo;

// =====================================================
// HC-SR04
// =====================================================
#define US_TRIG_PIN 5
#define US_ECHO_PIN 19

#define MAX_DISTANCE 400
#define COLLISION_THRESHOLD 15

NewPing sonar(US_TRIG_PIN, US_ECHO_PIN, MAX_DISTANCE);

// =====================================================
// GLOBAL
// =====================================================
WiFiClient espClient;
PubSubClient client(espClient);

int currentSpeed = 120;
char currentMoveState = 'S';
bool isMoving = false;
bool isOverridden = false;
unsigned long lastCollisionCheck = 0;

// Biến cờ kiểm soát trạng thái cảnh báo trên Web
bool collisionTriggered = false; 

// =====================================================
// MOTOR LOGIC
// =====================================================
void setDirectionLogic(char state) {
  if (state == 'F') {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  else if (state == 'B') {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  isMoving = false;
}

void runMotors(char state, int spd) {
  if (state == 'S') {
    stopMotors();
    return;
  }

  // Nếu xe đang dừng mà bắt đầu chạy (isMoving == false)
  if (!isMoving && spd > 0) {
    setDirectionLogic(state);
    
    // Ép xung tối đa trong 40ms để tạo mô-men xoắn phá vỡ ma sát tĩnh (Kick-start)
    analogWrite(ENA, 220); 
    analogWrite(ENB, 220);
    delay(40); 
  }

  // Sau khi xe đã có đà nhích, trả về tốc độ thực tế được yêu cầu
  setDirectionLogic(state);
  analogWrite(ENA, spd);
  analogWrite(ENB, spd);
  isMoving = true;
}

// =====================================================
// COLLISION DETECT (ĐÃ THAY ĐỔI ĐỂ TỰ ĐỘNG XÓA TRẠNG THÁI)
// =====================================================
void executeEmergencyStop() {
  isOverridden = true;
  currentMoveState = 'S';

  stopMotors();
  steeringServo.write(SERVO_CENTER);

  // Gửi cảnh báo nguy hiểm lên Node-RED
  client.publish(TOPIC_STATUS, "WARNING: COLLISION DETECTED");
  Serial.println("COLLISION -> STOP");

  collisionTriggered = true; // Đánh dấu hệ thống đang dính cảnh báo
  delay(500); // Giữ xe dừng 0.5s để đảm bảo an toàn
  isOverridden = false;
}

void checkCollision() {
  unsigned int distance = sonar.ping_cm();

  // Trường hợp 1: Đang đi tiến mà gặp vật cản -> Phanh khẩn cấp
  if (currentMoveState == 'F' && !isOverridden) {
    if (distance > 0 && distance <= COLLISION_THRESHOLD) {
      executeEmergencyStop();
      return;
    }
  }

  // Trường hợp 2: Nếu trước đó có cảnh báo, nhưng hiện tại khoảng cách đã an toàn (hoặc xe không đi tiến nữa)
  if (collisionTriggered) {
    if (distance == 0 || distance > COLLISION_THRESHOLD || currentMoveState != 'F') {
      // Gửi tin nhắn mới để xóa chữ WARNING trên giao diện Node-RED
      client.publish(TOPIC_STATUS, "SYSTEM ONLINE"); 
      Serial.println("SYSTEM -> SAFE / ONLINE");
      collisionTriggered = false; // Gỡ bỏ cờ cảnh báo
    }
  }
}
// =====================================================
// MQTT CALLBACK (ĐÃ SỬA LỖI BẺ LÁI)
// =====================================================
void callback(char* topic, byte* payload, unsigned int length) {
  char message[length + 1];
  for (int i = 0; i < length; i++) {
    message[i] = (char)payload[i];
  }
  message[length] = '\0';

  Serial.print("TOPIC: "); Serial.println(topic);
  Serial.print("MSG: "); Serial.println(message);

  // ---------------------------------------------------
  // XỬ LÝ TỐC ĐỘ
  // ---------------------------------------------------
  if (strcmp(topic, TOPIC_SPEED) == 0) {
    if (message[0] == 's') {
      int newSpeed = atoi(message + 1);
      newSpeed = constrain(newSpeed, 0, 255);
      currentSpeed = newSpeed;
  client.publish("remote_car/realtime_speed",
               String(currentSpeed).c_str());
      runMotors(currentMoveState, currentSpeed);
      Serial.print("Speed -> "); Serial.println(currentSpeed);
    }
  }

  // ---------------------------------------------------
  // XỬ LÝ HƯỚNG / GÓC QUAY SERVO
  // ---------------------------------------------------
  if (strcmp(topic, TOPIC_DIRECTION) == 0) {
    char cmd = message[0]; // Lấy ký tự đầu tiên để kiểm tra

    // TRƯỜNG HỢP 1: Nhận ký tự chữ điều khiển (Ưu tiên xử lý lệnh từ App/Nút bấm)
    if (cmd == 'F' || cmd == 'B' || cmd == 'S' || cmd == 'L' || cmd == 'R' || cmd == 'G') {
      
      if (cmd == 'F' || cmd == 'B' || cmd == 'S') {
        currentMoveState = cmd;
        runMotors(currentMoveState, currentSpeed);
        Serial.print("Button Move -> "); Serial.println(cmd);
      }
      else if (cmd == 'L') {
        steeringServo.write(SERVO_LEFT);
        Serial.println("Button Steer -> LEFT");
      }
      else if (cmd == 'R') {
        steeringServo.write(SERVO_RIGHT);
        Serial.println("Button Steer -> RIGHT");
      }
      else if (cmd == 'G') {
        steeringServo.write(SERVO_CENTER);
        Serial.println("Button Steer -> CENTER");
      }
      
      return; // Đã xử lý nút bấm xong, thoát hàm luôn để tránh trôi xuống phần đọc số
    }

    // TRƯỜNG HỢP 2: Nhận góc quay dạng số trực tiếp từ thuật toán Python gửi xuống
    if (cmd >= '0' && cmd <= '9') {
  int angle = atoi(message);
  angle = constrain(angle, 0, 180);
  
  // Chỉ bẻ lái Servo theo thuật toán, không tự tiện can thiệp vào Motor nữa!
  steeringServo.write(angle);
  Serial.print("Python Servo Angle -> "); Serial.println(angle);
}
  }
}

// =====================================================
// WIFI SETUP
// =====================================================
void setup_wifi() {
  delay(10);
  WiFi.begin(ssid, pass);
  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected");
  Serial.println(WiFi.localIP());
}

// =====================================================
// MQTT RECONNECT
// =====================================================
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting MQTT...");
    if (client.connect("ESP32_CAR")) {
      Serial.println("OK");
      client.subscribe(TOPIC_SPEED);
      client.subscribe(TOPIC_DIRECTION);
      client.publish(TOPIC_STATUS, "SYSTEM ONLINE");
    } else {
      Serial.print("FAILED: ");
      Serial.println(client.state());
      delay(3000);
    }
  }
}

// =====================================================
// ARDUINO SETUP
// =====================================================
void setup() {
  Serial.begin(115200);

  // Khởi tạo chân Motor L298N
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Khởi tạo Servo điều hướng
  ESP32PWM::allocateTimer(0); // Ép ESP32 cấp phát riêng Timer 0 để chạy PWM/Servo
  steeringServo.setPeriodHertz(50);

  // Cấu hình chân Servo và góc mặc định ban đầu
  steeringServo.attach(SERVO_PIN, 500, 2400); 
  steeringServo.write(SERVO_CENTER);

  // Kết nối mạng và MQTT
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  Serial.println("SYSTEM READY");
}

// =====================================================
// ARDUINO LOOP
// =====================================================
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Kiểm tra khoảng cách chống va chạm định kỳ (Không dùng delay trong loop)
  if (millis() - lastCollisionCheck > 200) {
    lastCollisionCheck = millis();
    checkCollision();
  }
}