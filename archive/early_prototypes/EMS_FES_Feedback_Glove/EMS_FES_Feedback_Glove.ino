/*
  ==========================================================================
  EMS/FES 피드백 제어 장갑 - ESP32 DevKit V1
  --------------------------------------------------------------------------
  - 양쪽 장갑에 부착된 플렉스 센서 10개(장갑A: 목표, 장갑B: 실제)를 측정
  - 장갑A(센서1~5)의 굽힘률을 "목표값", 장갑B(센서6~10)의 굽힘률을 "실제값"으로
    사용하여 오차를 계산하고, 오차에 따라 자극 세기 명령값(stimulationLevel)을
    증가/감소/유지시키는 폐루프(closed-loop) 제어를 수행한다.
  - 이 코드는 실제 전기 자극(EMS/FES) 파형을 만들지 않는다.
    sendStimulationCommand() 함수에서 stimulationLevel 값을 Serial Monitor에
    출력하는 것까지만 담당하며, 추후 Arduino Nano + openEMSstim 등 외부
    제어부와 UART로 연결해서 실제 자극을 발생시킬 예정이다.
  ==========================================================================
*/

// ================================ 핀 정의 ================================

// ---- 장갑A (목표 움직임) : 센서 1~5 ----
#define SENSOR1_PIN 25
#define SENSOR2_PIN 33
#define SENSOR3_PIN 32
#define SENSOR4_PIN 35
#define SENSOR5_PIN 34

// ---- 장갑B (실제 움직임 / 피드백) : 센서 6~10 ----
#define SENSOR6_PIN  15
#define SENSOR7_PIN  2
#define SENSOR8_PIN  4
#define SENSOR9_PIN  26
#define SENSOR10_PIN 27

// ---- 버튼 ----
#define BUTTON1_PIN 21   // 초기값(기준값) 측정 버튼
#define BUTTON2_PIN 19   // 시스템 작동(RUNNING) 버튼

// ================================ 상수/설정 ================================

const int NUM_SENSORS  = 10;   // 전체 센서 개수
const int NUM_CHANNELS = 5;    // 손가락(자극 채널) 개수

// 센서 배열: index 0~4 = 센서1~5(목표), index 5~9 = 센서6~10(실제)
// 센서(i) <-> 센서(i+5) 가 서로 대응 관계 (예: index0=센서1 <-> index5=센서6)
int sensorPins[NUM_SENSORS] = {
  SENSOR1_PIN, SENSOR2_PIN, SENSOR3_PIN, SENSOR4_PIN, SENSOR5_PIN,   // 목표(장갑A)
  SENSOR6_PIN, SENSOR7_PIN, SENSOR8_PIN, SENSOR9_PIN, SENSOR10_PIN   // 실제(장갑B)
};

// ---- 센서별 보정값 (나중에 실측 후 자유롭게 수정) ----
// rangeValue[i] = "완전히 굽혔을 때 ADC 값이 초기값(sensorZero)에서 얼마나
//                  변하는지"를 나타내는 값이다.
//   - 굽히면 ADC 값이 증가하는 센서 -> 양수로 설정
//   - 굽히면 ADC 값이 감소하는 센서 -> 음수로 설정
// 아직 각 센서의 실제 굽힘 특성(완전 굽힘 기준값)이 확정되지 않았으므로
// 임시값(1500)으로 채워두고, 실제 테스트 후 센서별로 각각 수정해서 쓴다.
float rangeValue[NUM_SENSORS] = {
  1500, 1500, 1500, 1500, 1500,   // 센서 1~5 (목표, 장갑A)
  1500, 1500, 1500, 1500, 1500    // 센서 6~10 (실제, 장갑B)
};

// ---- 자극 세기(stimulationLevel) 설정 ----
const int STIM_MIN = 0;    // 자극 없음
const int STIM_MAX = 10;   // 최대 허용 명령값 (필요시 이 값만 수정)
const int DEADBAND = 5;    // 오차가 이 값(%) 이내면 자극 세기를 그대로 유지

// ---- 측정 설정 ----
const int CALIB_SAMPLES  = 30;  // 초기값 측정 시 평균 낼 횟수 (20~50 권장)
const int NORMAL_SAMPLES = 5;   // 평상시(RUNNING) 측정 시 평균 낼 횟수(노이즈 감소)
const unsigned long LOOP_PERIOD_MS = 400; // 기본 제어 주기(ms)

// ---- 센서 이상값(단선/합선 등) 판정 기준 ----
// ESP32 ADC는 0~4095 범위이며, 값이 양 끝단에 붙어서 고정되면 비정상으로 간주
const int ADC_FAULT_LOW  = 20;
const int ADC_FAULT_HIGH = 4075;

// ---- 버튼 디바운스 ----
const unsigned long DEBOUNCE_DELAY = 50; // ms

// ================================ 시스템 상태 ================================

enum SystemState { WAITING, CALIBRATING, READY, RUNNING };
SystemState currentState = WAITING;

bool calibrated = false;   // 초기값 측정 완료 여부
bool commError  = false;   // 통신 오류 플래그 (추후 Nano/openEMSstim 연결 시 사용)

// ================================ 전역 변수 ================================

float sensorZero[NUM_SENSORS];        // 초기값(기준 raw 값)
int   sensorRaw[NUM_SENSORS];         // 현재 raw 측정값
float sensorPercent[NUM_SENSORS];     // 0~100% 굽힘률
bool  sensorFault[NUM_SENSORS];       // 센서 이상 여부(true = 이상)

float errorValue[NUM_CHANNELS];         // 오차(%) = 목표 - 실제
int   stimulationLevel[NUM_CHANNELS];   // 자극 세기 명령값(0~STIM_MAX)

// ---- 버튼 디바운스 상태 변수 ----
int button1LastRaw = HIGH;
int button1Stable  = HIGH;
unsigned long button1LastChange = 0;

int button2LastRaw = HIGH;
int button2Stable  = HIGH;
unsigned long button2LastChange = 0;


// ==========================================================================
// setup()
// ==========================================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  // 센서 핀 입력 설정 (analogRead 전용 핀이지만 명시적으로 지정)
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(sensorPins[i], INPUT);
  }

  // 버튼: 누르면 LOW가 되도록 내부 풀업 사용
  pinMode(BUTTON1_PIN, INPUT_PULLUP);
  pinMode(BUTTON2_PIN, INPUT_PULLUP);

  // 시스템 시작 시 모든 자극 세기는 0
  for (int i = 0; i < NUM_CHANNELS; i++) {
    stimulationLevel[i] = 0;
  }

  currentState = WAITING;

  Serial.println(F("=================================================="));
  Serial.println(F("EMS/FES 피드백 제어 시스템 시작"));
  Serial.println(F("STATE : WAITING"));
  Serial.println(F("버튼1(GPIO21)을 눌러 양쪽 장갑의 초기값을 측정하세요."));
  Serial.println(F("=================================================="));
}


// ==========================================================================
// loop()
// ==========================================================================
void loop() {
  handleButtons();

  if (currentState == RUNNING) {
    runControlLoop();
  }

  // 기본 제어 주기 (요구사항: 400ms)
  delay(LOOP_PERIOD_MS);
}


// ==========================================================================
// 버튼 입력 처리 (디바운스 + HIGH->LOW 순간에 1회만 동작)
// ==========================================================================
void handleButtons() {

  // ---- 버튼1 : 초기값 측정 ----
  int raw1 = digitalRead(BUTTON1_PIN);
  if (raw1 != button1LastRaw) {
    button1LastChange = millis();
    button1LastRaw = raw1;
  }
  if ((millis() - button1LastChange) > DEBOUNCE_DELAY && raw1 != button1Stable) {
    button1Stable = raw1;
    if (button1Stable == LOW) {   // 눌리는 순간 1회만 실행
      calibrateSensors();
    }
  }

  // ---- 버튼2 : 시스템 작동(RUNNING) 시작 ----
  int raw2 = digitalRead(BUTTON2_PIN);
  if (raw2 != button2LastRaw) {
    button2LastChange = millis();
    button2LastRaw = raw2;
  }
  if ((millis() - button2LastChange) > DEBOUNCE_DELAY && raw2 != button2Stable) {
    button2Stable = raw2;
    if (button2Stable == LOW) {   // 눌리는 순간 1회만 실행
      startRunning();
    }
  }
}


// ==========================================================================
// 버튼2 처리: 초기값 측정이 끝난 경우에만 RUNNING 상태로 진입
// ==========================================================================
void startRunning() {
  if (!calibrated) {
    // 안전조건: 초기값 측정 전에는 RUNNING 진입 금지
    Serial.println(F("초기값 측정이 완료되지 않았습니다. 먼저 버튼1을 눌러주세요."));
    return;
  }

  if (currentState == READY) {
    currentState = RUNNING;

    // RUNNING 진입 시 목표 굽힘률을 한 번 측정해서 초기 자극 세기를 예측한다.
    readAllSensors();
    calculatePercent();
    predictInitialStimulation();

    Serial.println(F("STATE : RUNNING"));
    Serial.println(F("폐루프 제어 시작"));
  }
}


// ==========================================================================
// 버튼1 처리: 양쪽 장갑 센서 1~10의 초기값(기준값)을 동시에 측정
// ==========================================================================
void calibrateSensors() {
  currentState = CALIBRATING;
  calibrated = false;

  // 초기값 측정 중에는 자극 세기를 항상 0으로 유지 (안전조건)
  for (int i = 0; i < NUM_CHANNELS; i++) {
    stimulationLevel[i] = 0;
  }

  Serial.println(F("STATE : CALIBRATING"));
  Serial.println(F("CALIBRATING..."));

  // 센서 1~10 모두 동시에(순차적으로 빠르게) 초기값 측정
  for (int i = 0; i < NUM_SENSORS; i++) {
    sensorZero[i] = readSensorAverage(sensorPins[i], CALIB_SAMPLES);
  }

  calibrated = true;
  currentState = READY;

  Serial.println(F("초기값 측정 완료"));
  Serial.println(F("STATE : READY"));
  Serial.println(F("버튼2(GPIO19)를 눌러 폐루프 제어를 시작하세요."));
}


// ==========================================================================
// 지정한 핀을 samples 회 analogRead()해서 평균값을 반환 (노이즈 감소용)
// ==========================================================================
float readSensorAverage(int pin, int samples) {
  long sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delay(2); // ADC 안정화를 위한 짧은 지연
  }
  return (float)sum / samples;
}


// ==========================================================================
// 센서 10개를 모두 측정하여 sensorRaw[]에 저장하고, 이상값 여부를 판정
// ==========================================================================
void readAllSensors() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    float avg = readSensorAverage(sensorPins[i], NORMAL_SAMPLES);
    sensorRaw[i] = (int)avg;

    // 값이 ADC 범위 양 끝단에 붙어있으면 단선/합선 등 이상으로 판정
    if (sensorRaw[i] <= ADC_FAULT_LOW || sensorRaw[i] >= ADC_FAULT_HIGH) {
      sensorFault[i] = true;
    } else {
      sensorFault[i] = false;
    }
  }
}


// ==========================================================================
// 초기값(sensorZero)과 현재값(sensorRaw)의 차이를 이용해 0~100% 굽힘률 계산
// rangeValue의 부호가 센서 방향(증가형/감소형)을 자동으로 처리해준다.
// ==========================================================================
void calculatePercent() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    float diff = sensorRaw[i] - sensorZero[i];
    float percent = (diff / rangeValue[i]) * 100.0;

    if (percent < 0)   percent = 0;
    if (percent > 100) percent = 100;

    sensorPercent[i] = percent;
  }
}


// ==========================================================================
// 목표(센서1~5)와 실제(센서6~10) 굽힘률 차이(오차)를 계산
// error > 0 : 실제가 목표보다 덜 굽혀짐 (자극 더 필요)
// error < 0 : 실제가 목표보다 더 굽혀짐 (자극 줄여야 함)
// ==========================================================================
void calculateErrors() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    errorValue[i] = sensorPercent[i] - sensorPercent[i + NUM_CHANNELS];
  }
}


// ==========================================================================
// RUNNING 진입 시 목표 굽힘률을 바탕으로 최초 자극 세기를 대략적으로 예측
// (목표가 낮으면 낮은 단계, 목표가 높으면 높은 단계)
// ==========================================================================
void predictInitialStimulation() {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    int predicted = (int)((sensorPercent[i] / 100.0) * STIM_MAX + 0.5);
    stimulationLevel[i] = constrain(predicted, STIM_MIN, STIM_MAX);
  }
}


// ==========================================================================
// 오차(errorValue)를 바탕으로 자극 세기를 한 단계씩 증가/감소/유지
// ==========================================================================
void updateStimulationLevel() {
  for (int i = 0; i < NUM_CHANNELS; i++) {

    // 안전조건: 목표센서(i) 또는 실제센서(i+5) 중 하나라도 이상이면 증가 금지
    bool faultDetected = sensorFault[i] || sensorFault[i + NUM_CHANNELS];

    if (errorValue[i] > DEADBAND) {
      // 실제가 목표보다 덜 굽혀짐 -> 자극 세기 1단계 증가 (이상 없을 때만)
      if (!faultDetected) {
        stimulationLevel[i] += 1;
      }
    } else if (errorValue[i] < -DEADBAND) {
      // 실제가 목표보다 더 굽혀짐 -> 자극 세기 1단계 감소 (감소는 항상 허용)
      stimulationLevel[i] -= 1;
    }
    // -DEADBAND ~ +DEADBAND 사이면 아무 것도 하지 않고 유지

    // 안전조건: 항상 최소/최대 범위 안에서만 값 유지
    stimulationLevel[i] = constrain(stimulationLevel[i], STIM_MIN, STIM_MAX);

    // 추후 통신 오류가 감지되면 안전을 위해 모든 채널을 0으로 만든다
    if (commError) {
      stimulationLevel[i] = 0;
    }
  }
}


// ==========================================================================
// RUNNING 상태에서 한 사이클(400ms)마다 실행되는 폐루프 제어 본체
// 순서: 센서 측정 -> 굽힘률 계산 -> 오차 계산 -> 자극 세기 갱신 -> 명령 전송/출력
// ==========================================================================
void runControlLoop() {
  readAllSensors();          // 센서 1~10 측정 (목표+실제 동시)
  calculatePercent();        // 0~100% 굽힘률 변환
  calculateErrors();         // 목표-실제 오차 계산
  updateStimulationLevel();  // 오차 기반 자극 세기 보정
  sendStimulationCommand();  // 외부 제어부로 보낼 명령값 처리(현재는 출력만)
  printSensorData();         // Serial Monitor 출력
}


// ==========================================================================
// 계산된 stimulationLevel[5]을 외부 EMS/FES 제어부로 전달하는 함수
// --------------------------------------------------------------------------
// [현재 단계]
//   실제 전기 자극 신호는 만들지 않는다. stimulationLevel 값을
//   Serial Monitor에 출력하는 역할만 수행한다.
//
// [추후 확장 예정]
//   Arduino Nano(또는 openEMSstim 제어부)와 UART(예: Serial2)로 연결하여
//   아래와 같은 형태로 실제 명령을 전송할 수 있다.
//     Serial2.begin(...);
//     Serial2.write(stimulationLevel, NUM_CHANNELS);
//   통신 응답이 없거나 오류가 발생하면 commError = true 로 설정하면
//   updateStimulationLevel()에서 자동으로 모든 채널을 0으로 만든다.
// ==========================================================================
void sendStimulationCommand() {
  Serial.print(F(">> [to external EMS/FES unit, not yet wired] "));
  for (int i = 0; i < NUM_CHANNELS; i++) {
    Serial.print("CH");
    Serial.print(i + 1);
    Serial.print("=");
    Serial.print(stimulationLevel[i]);
    if (i < NUM_CHANNELS - 1) Serial.print(", ");
  }
  Serial.println();
}


// ==========================================================================
// Serial Monitor에 목표/실제/오차/자극세기를 보기 쉽게 출력
// ==========================================================================
void printSensorData() {
  Serial.println(F("TARGET:"));
  Serial.print("1="); Serial.print((int)sensorPercent[0]); Serial.print("% | ");
  Serial.print("2="); Serial.print((int)sensorPercent[1]); Serial.print("% | ");
  Serial.print("3="); Serial.print((int)sensorPercent[2]); Serial.print("% | ");
  Serial.print("4="); Serial.print((int)sensorPercent[3]); Serial.print("% | ");
  Serial.print("5="); Serial.print((int)sensorPercent[4]); Serial.println("%");

  Serial.println(F("ACTUAL:"));
  Serial.print("6=");  Serial.print((int)sensorPercent[5]); Serial.print("% | ");
  Serial.print("7=");  Serial.print((int)sensorPercent[6]); Serial.print("% | ");
  Serial.print("8=");  Serial.print((int)sensorPercent[7]); Serial.print("% | ");
  Serial.print("9=");  Serial.print((int)sensorPercent[8]); Serial.print("% | ");
  Serial.print("10="); Serial.print((int)sensorPercent[9]); Serial.println("%");

  Serial.println(F("ERROR:"));
  Serial.print("1-6=");  Serial.print((int)errorValue[0]); Serial.print(" | ");
  Serial.print("2-7=");  Serial.print((int)errorValue[1]); Serial.print(" | ");
  Serial.print("3-8=");  Serial.print((int)errorValue[2]); Serial.print(" | ");
  Serial.print("4-9=");  Serial.print((int)errorValue[3]); Serial.print(" | ");
  Serial.print("5-10="); Serial.println((int)errorValue[4]);

  Serial.println(F("STIM:"));
  Serial.print("CH1="); Serial.print(stimulationLevel[0]); Serial.print(" | ");
  Serial.print("CH2="); Serial.print(stimulationLevel[1]); Serial.print(" | ");
  Serial.print("CH3="); Serial.print(stimulationLevel[2]); Serial.print(" | ");
  Serial.print("CH4="); Serial.print(stimulationLevel[3]); Serial.print(" | ");
  Serial.print("CH5="); Serial.println(stimulationLevel[4]);

  Serial.println(F("----------------------------------"));
}
