#include <Arduino.h>

// ======================================================
// [프로젝트 구성]
// - 보드: Arduino Mega
// - 왼손 6축 센서  : Serial1 사용 (RX1=19, TX1=18)
// - 오른손 6축 센서: Serial2 사용 (RX2=17, TX2=16)
// - 플렉스 센서 8개:
//   왼손 검지~소지 4개 = A0, A1, A2, A3
//   오른손 검지~소지 4개 = A4, A5, A6, A7
//
// [이 버전의 목적]
// - 시리얼 모니터에서 값 확인하기 쉽게
// - 플렉스 raw 값은 빼고
// - 플렉스 normalized 값만 출력
// ======================================================


// =========================
// 플렉스 센서 설정
// =========================
const uint8_t NUM_FLEX = 8;

// 순서:
// A0 = 왼손 검지
// A1 = 왼손 중지
// A2 = 왼손 약지
// A3 = 왼손 소지
// A4 = 오른손 검지
// A5 = 오른손 중지
// A6 = 오른손 약지
// A7 = 오른손 소지
const uint8_t FLEX_PINS[NUM_FLEX] = {A0, A1, A2, A3, A4, A5, A6, A7};

// 캘리브레이션용 값
int flexStraight[NUM_FLEX];   // 손가락 폈을 때 기준값
int flexBent[NUM_FLEX];       // 손가락 굽혔을 때 기준값

// 간단한 저역통과 필터용
float flexFiltered[NUM_FLEX];
const float FLEX_ALPHA = 0.25f;


// =========================
// 시리얼 설정
// =========================
#define IMU_LEFT_SERIAL   Serial1
#define IMU_RIGHT_SERIAL  Serial2

const uint32_t USB_BAUD = 115200;   // PC와 연결되는 USB 시리얼
const uint32_t IMU_BAUD = 115200;   // 6축 센서 기본 baud rate

// 시리얼 모니터에서 보기 쉽게 10Hz로 설정
// 나중에 실제 데이터 수집할 때는 20ms(50Hz)로 낮추면 됨
const unsigned long OUTPUT_INTERVAL_MS = 20;
unsigned long lastOutputMs = 0;


// =========================
// 6축 센서 데이터 구조체
// =========================
struct IMUData {
  // 가속도
  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;

  // 각속도
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;

  // 각도
  float roll  = 0.0f;
  float pitch = 0.0f;
  float yaw   = 0.0f;

  // 온도(필요 없으면 안 써도 됨)
  float temp = 0.0f;

  // 해당 데이터가 들어왔는지 확인용
  bool hasAccel = false;
  bool hasGyro  = false;
  bool hasAngle = false;
};

// 패킷 파싱용 버퍼
struct IMUParser {
  uint8_t buf[11];
  uint8_t idx = 0;
};

IMUData imuLeft, imuRight;
IMUParser parserLeft, parserRight;


// =========================
// 유틸 함수
// =========================

// 플렉스 값을 0~1로 정규화하는 함수
// 0에 가까울수록 손가락이 펴진 상태
// 1에 가까울수록 손가락이 굽혀진 상태
float normalizeFlex(int raw, int straightVal, int bentVal) {
  if (straightVal == bentVal) return 0.0f;

  float norm;

  // 굽힘 방향에 따라 계산
  if (bentVal > straightVal) {
    norm = (float)(raw - straightVal) / (float)(bentVal - straightVal);
  } else {
    norm = (float)(straightVal - raw) / (float)(straightVal - bentVal);
  }

  // 0~1 범위로 제한
  if (norm < 0.0f) norm = 0.0f;
  if (norm > 1.0f) norm = 1.0f;

  return norm;
}

// 2바이트를 int16으로 변환
int16_t bytesToInt16(uint8_t lowByte, uint8_t highByte) {
  return (int16_t)((highByte << 8) | lowByte);
}

// 체크섬 계산
uint8_t calcChecksum(uint8_t *packet) {
  uint16_t sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += packet[i];
  }
  return (uint8_t)(sum & 0xFF);
}


// =========================
// 6축 센서 패킷 해석
// packet[0] = 0x55
// packet[1] = 0x51 -> 가속도
// packet[1] = 0x52 -> 각속도
// packet[1] = 0x53 -> 각도
// =========================
void decodePacket(uint8_t *pkt, IMUData &imu) {
  if (pkt[0] != 0x55) return;
  if (calcChecksum(pkt) != pkt[10]) return;

  int16_t x = bytesToInt16(pkt[2], pkt[3]);
  int16_t y = bytesToInt16(pkt[4], pkt[5]);
  int16_t z = bytesToInt16(pkt[6], pkt[7]);
  int16_t t = bytesToInt16(pkt[8], pkt[9]);

  // 가속도 패킷
  if (pkt[1] == 0x51) {
    imu.ax = ((float)x / 32768.0f) * 16.0f;
    imu.ay = ((float)y / 32768.0f) * 16.0f;
    imu.az = ((float)z / 32768.0f) * 16.0f;
    imu.temp = ((float)t / 340.0f) + 36.53f;
    imu.hasAccel = true;
  }

  // 각속도 패킷
  else if (pkt[1] == 0x52) {
    imu.gx = ((float)x / 32768.0f) * 2000.0f;
    imu.gy = ((float)y / 32768.0f) * 2000.0f;
    imu.gz = ((float)z / 32768.0f) * 2000.0f;
    imu.temp = ((float)t / 340.0f) + 36.53f;
    imu.hasGyro = true;
  }

  // 각도 패킷
  else if (pkt[1] == 0x53) {
    imu.roll  = ((float)x / 32768.0f) * 180.0f;
    imu.pitch = ((float)y / 32768.0f) * 180.0f;
    imu.yaw   = ((float)z / 32768.0f) * 180.0f;
    imu.temp = ((float)t / 340.0f) + 36.53f;
    imu.hasAngle = true;
  }
}


// =========================
// 6축 센서 UART 스트림 파싱
// =========================
void parseIMUSerial(HardwareSerial &imuSerial, IMUParser &parser, IMUData &imu) {
  while (imuSerial.available() > 0) {
    uint8_t b = (uint8_t)imuSerial.read();

    // 아직 패킷 시작 전이면 0x55 찾기
    if (parser.idx == 0) {
      if (b == 0x55) {
        parser.buf[0] = b;
        parser.idx = 1;
      }
      continue;
    }

    // 패킷 계속 저장
    parser.buf[parser.idx++] = b;

    // 11바이트가 모이면 하나의 패킷 해석
    if (parser.idx == 11) {
      decodePacket(parser.buf, imu);
      parser.idx = 0;
    }
  }
}


// =========================
// 플렉스 캘리브레이션
// =========================
void averageFlexSamples(int outVals[NUM_FLEX], int sampleCount, int delayMs) {
  long sums[NUM_FLEX] = {0};

  for (int s = 0; s < sampleCount; s++) {
    for (uint8_t i = 0; i < NUM_FLEX; i++) {
      sums[i] += analogRead(FLEX_PINS[i]);
    }
    delay(delayMs);
  }

  for (uint8_t i = 0; i < NUM_FLEX; i++) {
    outVals[i] = sums[i] / sampleCount;
  }
}

void calibrateFlex() {
  Serial.println();
  Serial.println("===== FLEX CALIBRATION START =====");

  // 1단계: 손가락 펴기
  Serial.println("1) 양손 검지~소지를 곧게 펴고 3초 유지하세요...");
  delay(3000);
  averageFlexSamples(flexStraight, 100, 10);

  // 2단계: 손가락 굽히기
  Serial.println("2) 양손 검지~소지를 최대한 구부리고 3초 유지하세요...");
  delay(3000);
  averageFlexSamples(flexBent, 100, 10);

  Serial.println("Flex calibration done.");

  Serial.print("Straight values: ");
  for (uint8_t i = 0; i < NUM_FLEX; i++) {
    Serial.print(flexStraight[i]);
    if (i < NUM_FLEX - 1) Serial.print(", ");
  }
  Serial.println();

  Serial.print("Bent values: ");
  for (uint8_t i = 0; i < NUM_FLEX; i++) {
    Serial.print(flexBent[i]);
    if (i < NUM_FLEX - 1) Serial.print(", ");
  }
  Serial.println();

  Serial.println("===== FLEX CALIBRATION END =====");
  Serial.println();
}


// =========================
// setup
// =========================
void setup() {
  // PC와 연결된 USB 시리얼
  Serial.begin(USB_BAUD);

  // 왼손 / 오른손 6축 센서 UART
  IMU_LEFT_SERIAL.begin(IMU_BAUD);
  IMU_RIGHT_SERIAL.begin(IMU_BAUD);

  delay(1000);

  // 플렉스 핀 초기화
  for (uint8_t i = 0; i < NUM_FLEX; i++) {
    pinMode(FLEX_PINS[i], INPUT);
    flexFiltered[i] = analogRead(FLEX_PINS[i]);
  }

  Serial.println("System start.");
  Serial.println("6축 센서 자동 안정화를 위해 장갑을 잠시 가만히 두세요...");
  delay(3500);

  // 플렉스 캘리브레이션
  calibrateFlex();

  // 출력 헤더
  Serial.println("===== SERIAL OUTPUT HEADER =====");
  Serial.println(
    "timestamp_ms,"
    "lf0_norm,lf1_norm,lf2_norm,lf3_norm,"
    "rf0_norm,rf1_norm,rf2_norm,rf3_norm,"
    "l_ax_g,l_ay_g,l_az_g,l_gx_dps,l_gy_dps,l_gz_dps,l_roll_deg,l_pitch_deg,l_yaw_deg,"
    "r_ax_g,r_ay_g,r_az_g,r_gx_dps,r_gy_dps,r_gz_dps,r_roll_deg,r_pitch_deg,r_yaw_deg"
  );
  Serial.println("===== START DATA =====");
}


// =========================
// loop
// =========================
void loop() {
  // 6축 센서 데이터는 계속 파싱
  parseIMUSerial(IMU_LEFT_SERIAL, parserLeft, imuLeft);
  parseIMUSerial(IMU_RIGHT_SERIAL, parserRight, imuRight);

  unsigned long now = millis();

  // 출력 주기 도달 전이면 종료
  if (now - lastOutputMs < OUTPUT_INTERVAL_MS) {
    return;
  }
  lastOutputMs = now;

  // -------------------------
  // 플렉스 읽기 + 정규화
  // -------------------------
  float flexNorm[NUM_FLEX];

  for (uint8_t i = 0; i < NUM_FLEX; i++) {
    int raw = analogRead(FLEX_PINS[i]);

    // 간단한 필터
    flexFiltered[i] = FLEX_ALPHA * raw + (1.0f - FLEX_ALPHA) * flexFiltered[i];

    // 정규화
    int filteredInt = (int)(flexFiltered[i] + 0.5f);
    flexNorm[i] = normalizeFlex(filteredInt, flexStraight[i], flexBent[i]);
  }

  // -------------------------
  // CSV 출력
  // -------------------------
  Serial.print(now);

  // 왼손 플렉스 정규화 4개 + 오른손 플렉스 정규화 4개
  for (uint8_t i = 0; i < NUM_FLEX; i++) {
    Serial.print(",");
    Serial.print(flexNorm[i], 4);
  }

  // 왼손 6축 센서
  Serial.print(",");
  Serial.print(imuLeft.ax, 4);
  Serial.print(",");
  Serial.print(imuLeft.ay, 4);
  Serial.print(",");
  Serial.print(imuLeft.az, 4);
  Serial.print(",");
  Serial.print(imuLeft.gx, 4);
  Serial.print(",");
  Serial.print(imuLeft.gy, 4);
  Serial.print(",");
  Serial.print(imuLeft.gz, 4);
  Serial.print(",");
  Serial.print(imuLeft.roll, 4);
  Serial.print(",");
  Serial.print(imuLeft.pitch, 4);
  Serial.print(",");
  Serial.print(imuLeft.yaw, 4);

  // 오른손 6축 센서
  Serial.print(",");
  Serial.print(imuRight.ax, 4);
  Serial.print(",");
  Serial.print(imuRight.ay, 4);
  Serial.print(",");
  Serial.print(imuRight.az, 4);
  Serial.print(",");
  Serial.print(imuRight.gx, 4);
  Serial.print(",");
  Serial.print(imuRight.gy, 4);
  Serial.print(",");
  Serial.print(imuRight.gz, 4);
  Serial.print(",");
  Serial.print(imuRight.roll, 4);
  Serial.print(",");
  Serial.print(imuRight.pitch, 4);
  Serial.print(",");
  Serial.println(imuRight.yaw, 4);
}