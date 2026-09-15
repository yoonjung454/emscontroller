#include "Arduino.h"
#include "Wire.h"
#include "AD5252.h"
#include "EMSChannel.h"

#define SERIAL_BAUD_RATE 19200
#define STIM_INTENSITY_PERCENT 10
#define HOLD_MS 3000UL

AD5252 digitalPot(0);

// 앞에서 확인한 실제 채널 순서
EMSChannel channel1(6, 7, A3, &digitalPot, 3);
EMSChannel channel2(5, 4, A2, &digitalPot, 1);

bool isRunning = false;
int activeCommand = 0;
unsigned long stimStartedAt = 0;

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  EMSChannel::start();

  stopAllChannels();

  delay(100);

  Serial.println("=== EMS 채널 제어 시작 ===");
  Serial.println("1 입력 : 1채널 3초");
  Serial.println("2 입력 : 2채널 3초");
  Serial.println("3 입력 : 모든 채널 3초");
  Serial.println("0 입력 : 즉시 정지");
}

void loop() {
  // EMSChannel::deactivate()는 호출 즉시 완전히 꺼지지 않는다 -- 세기(포텐셔미터)만
  // 바로 낮추고 "끄는 중(DEACTIVATING)" 상태로 넘어간 뒤, 50ms 후 check()가 호출되어야
  // 비로소 릴레이 핀(switch_1/switch_2)이 LOW로 내려가며 진짜로 끊긴다. 원래 메인
  // 펌웨어는 loop()에서 emsSystem.check()를 계속 돌려 이걸 처리해주는데, 이 스케치엔
  // 그 호출이 없어서 릴레이가 영원히 "연결된" 채로 남아있었다 -- 매 loop마다 확인해준다.
  channel1.check();
  channel2.check();

  // 시리얼 명령 확인
  char command = readCommand();

  if (command == '1' || command == '2' || command == '3') {
    stopAllChannels();
    startStimulation(command - '0');
  } 
  else if (command == '0') {
    stopAllChannels();
    Serial.println("모든 채널 즉시 정지");
  }

  // 3초가 지나면 자동 정지
  if (isRunning && millis() - stimStartedAt >= HOLD_MS) {
    stopAllChannels();
    Serial.println("3초 경과 - 모든 채널 종료");
  }
}

char readCommand() {
  char command = 0;

  while (Serial.available() > 0) {
    char received = Serial.read();

    // 줄바꿈 문자는 무시
    if (received == '\r' || received == '\n') {
      continue;
    }

    if (received >= '0' && received <= '3') {
      command = received;
    }
  }

  return command;
}

void startStimulation(int command) {
  activeCommand = command;
  stimStartedAt = millis();
  isRunning = true;

  if (command == 1) {
    channel1.setIntensity(STIM_INTENSITY_PERCENT);
    channel1.activate();

    Serial.println("1채널 자극 중 - 3초");
  }
  else if (command == 2) {
    channel2.setIntensity(STIM_INTENSITY_PERCENT);
    channel2.activate();

    Serial.println("2채널 자극 중 - 3초");
  }
  else if (command == 3) {
    channel1.setIntensity(STIM_INTENSITY_PERCENT);
    channel2.setIntensity(STIM_INTENSITY_PERCENT);

    channel1.activate();
    channel2.activate();

    Serial.println("1채널 + 2채널 동시 자극 중 - 3초");
  }
}

void stopAllChannels() {
  channel1.deactivate();
  channel2.deactivate();

  isRunning = false;
  activeCommand = 0;
}