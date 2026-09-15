# AI 비전 피드백 기반 폐루프 EMS 손동작 제어 시스템

카메라(MediaPipe)로 손가락 굽힘 정도를 실시간으로 측정하고, 자연어 명령에서 뽑아낸 목표
굽힘값과 비교해 OpenEMSstim(Arduino Nano)에 보내는 EMS 제어값(0~100, **mA 아님**)을
자동으로 올리고 내리는 폐루프 제어 프로토타입입니다.

> ⚠️ **먼저 읽어주세요**: 이 프로그램은 사람 몸에 전기 자극을 가하는 하드웨어를 다룹니다.
> 기본 실행 모드는 항상 **시뮬레이션**이고, 실제 EMS 출력은 `--live-serial`(또는 `--live`)
> 플래그 + `config.json`의 `safety.safety_max_intensity`를 담당자가 직접 0보다 크게 설정해야만
> 나갑니다. 이 값을 임의로 정해주지 않습니다 — 아래 "안전 경고" 섹션을 반드시 읽으세요.

---

## 1. 전체 시스템 구조 요약

```
[노트북]
  자연어 명령 입력 (GUI)
        │
        ▼
  command_interpreter.py  ── 규칙 기반으로 "목표 굽힘 %" 추출 (LLM 없이 동작, 나중에 교체 가능)
        │
        ▼
  adaptive_model.py       ── 과거 세션 데이터로 "이 사람은 이 정도 자극값이면 대략 이만큼 구부러짐"을 추정
        │ (초기 자극값 제안, 항상 safety로 재검증됨)
        ▼
  controller.py            ── target% vs current% 오차 기반 폐루프 제어 (P 또는 PID, 단계 제한)
        │                          ▲
        │ EMS 제어값(0~100)         │ current% (필터링된 굽힘값)
        ▼                          │
  safety_manager.py  ◄─────────────┤   (모든 실제 출력은 여기를 반드시 통과)
        │                          │
        ▼                          │
  serial_link.py  ──USB(19200bps)──▶  Arduino Nano (openemsstim_serial_controller.ino)
        │                                   │
        │                                   ▼
        │                          EMSChannel.setIntensity() → AD5252 디지털 가변저항
        │                                   │
        │                                   ▼
        │                     외부 EMS 기기 → 패드 → 손가락 굽힘근 자극
        │
        ▼
  vision_tracker.py  ◄── OpenCV 카메라 + MediaPipe Hand Landmarker (21개 랜드마크)
        │  (검지/중지/약지/새끼 4개 손가락의 MCP/PIP/DIP 각도 → 0~100% 굽힘값)
        ▼
  data_logger.py  ── 매 tick을 CSV로 기록 (adaptive_model 학습 데이터 겸 실험 로그)
        │
        ▼
  gui.py  ── 카메라 영상, 관절, 명령창, 그래프, 비상정지 버튼 등 표시 (Tkinter)
```

시뮬레이션 모드(`simulator.py`)는 위 그림에서 `vision_tracker.py`와 `serial_link.py`를 각각
`HandDynamicsSimulator`, `SimulatedArduino`로 바꿔 끼우는 것뿐입니다. `controller.py`,
`command_interpreter.py`, `safety_manager.py`, `data_logger.py`, `gui.py`는 실제/시뮬레이션
어느 쪽이든 완전히 동일한 코드로 동작합니다.

---

## 2. 핵심 가정과 실제 장치에서 확인해야 하는 항목

이 코드는 아래 항목을 **실제로 확인**했거나, 확인이 필요하다고 명시합니다 (임의로 지어낸
핀 번호/클래스는 없습니다):

| 항목 | 상태 | 확인한 방법 / 확인해야 할 것 |
|---|---|---|
| `EMSChannel::setIntensity(int)`가 0~100을 받는다 | ✅ 확인함 | 공식 `EMSChannel.cpp` 원문 확인 (`arduino-openEMSstim.ino`도 동일) |
| 채널1 핀 = `(5, 4, A2, wiperIndex=1)`, 채널2 핀 = `(6, 7, A3, wiperIndex=3)` | ✅ 공식 예제 값 확인함 | **하지만 팀 보드가 이 배선과 같은지는 당신이 직접 확인해야 합니다** — 아래 참고 |
| `AD5252 digitalPot(0)` (I2C 주소 오프셋 0) | ✅ 공식 예제 값 확인함 | 팀 보드의 실제 I2C 주소가 다르면 `openemsstim_serial_controller.ino`의 이 줄을 수정 |
| `Serial.begin(19200)` (USB 통신 속도) | ✅ 공식 펌웨어 기본값 확인함 | `config.json`의 `serial.baud_rate`와 반드시 일치시킬 것 |
| `applySignal()`/`check()`가 millis() 기반으로 자동 정지한다 | ✅ 확인함 | 그대로 재사용 (우리가 새로 만들지 않음) |
| CH340 드라이버로 Arduino Nano 호환보드가 보인다 | ⚠️ 확인 필요 | 처음 연결 시 Windows 장치 관리자에서 "USB-SERIAL CH340" 포트가 보이는지 확인 |
| 채널 1만 정상 작동, 채널 2는 하드웨어 고장 | ✅ 팀이 알려준 정보를 그대로 반영 | `arduino/hardware_config.h`의 `CHANNEL_2_ENABLED false` |
| 사람마다 안전한 최대 자극 세기 | ❌ **이 코드는 정하지 않음** | 전문가 감독 하에 직접 결정 후 `config.json`에 입력 |
| 웹캠이 약 30~45도 방향(손등+손가락 관절이 보이는 각도)에서 촬영된다 | 권장 사항 | 카메라를 손 위쪽 대각선에서 손등을 향하도록 배치 |

**핀 배선 재확인 방법**: `arduino/openemsstim_serial_controller.ino`를 열어
`EMSChannel emsChannel1(...)`, `EMSChannel emsChannel2(...)` 두 줄을 찾으세요. 괄호 안 숫자가
팀이 실제로 납땜/배선한 핀 번호와 일치하는지 멀티미터나 회로도로 확인한 다음 진행하세요.
다르면 숫자만 실제 배선에 맞게 고치면 됩니다 (클래스 구조는 그대로 사용).

---

## 3. 프로젝트 폴더 구조

```
ems_closed_loop/
├── main.py                                  # 진입점 (CLI 인자 처리, 전체 조립, 워커 스레드 시작)
├── gui.py                                    # Tkinter GUI
├── vision_tracker.py                         # 카메라 + MediaPipe Hand Landmarker
├── command_interpreter.py                    # 자연어 → 목표 굽힘 % (규칙 기반, LLM 교체 가능 구조)
├── controller.py                             # 폐루프 제어 (P / PID)
├── adaptive_model.py                         # 개인화 모델 (선형/2차 회귀로 초기 자극값 추정)
├── serial_link.py                            # PC ↔ Arduino 시리얼 통신
├── safety_manager.py                         # 모든 안전 규칙의 단일 창구
├── simulator.py                              # 카메라/Arduino 없이 테스트하는 시뮬레이션
├── data_logger.py                            # CSV 로깅 + 과거 세션 불러오기
├── config.py                                 # 설정 데이터클래스 + config.json 로드/저장
├── config.json                               # (최초 실행 시 자동 생성됨. 안전 상한값은 0으로 시작)
├── requirements.txt
├── README.md                                 # 이 파일
├── hand_landmarker.task                      # MediaPipe 모델 파일
├── arduino/
│   ├── openemsstim_serial_controller.ino     # 공식 펌웨어 + 시리얼 명령 패치
│   └── hardware_config.h                     # 채널 활성화/BLE/타임아웃 등 우리 쪽 설정
├── tests/
│   ├── test_command_interpreter.py
│   ├── test_controller.py
│   ├── test_safety_manager.py
│   ├── test_adaptive_model.py
│   └── test_config.py
├── logs/                                      # 세션별 CSV (실행 시 자동 생성)
└── models/
    └── adaptive_model.json                    # 학습된 개인화 모델 (있으면 자동 로드)
```

각 파일은 이미 이 폴더에 전체 코드로 작성되어 있고, `python -m unittest discover -s tests`로
38개 단위 테스트가 모두 통과하는 상태입니다. (아래 "9. 시뮬레이션 실행 방법"에서 직접 확인할 수
있습니다.) 이 문서에서는 코드를 다시 옮겨 적지 않고, 무엇이 어디 있는지와 어떻게 실행하는지에
집중합니다.

---

## 4. 각 파일의 역할 (요약)

- **main.py** — CLI 플래그(`--live`, `--live-camera`, `--live-serial`)를 해석해 실제/시뮬레이션
  구성요소를 선택하고, `ControlLoopWorker`를 백그라운드 스레드로 돌리며 GUI를 메인 스레드에서 띄웁니다.
- **gui.py** — 카메라 영상, 관절, 명령 입력창, 30/60/90% 버튼, 캘리브레이션 버튼, 포트 선택,
  실시간 수치, 그래프, CSV 상태, 큰 빨간 비상정지 버튼을 그립니다. 워커의 `Snapshot`을
  주기적으로 읽어와 화면만 갱신합니다 (직접 하드웨어를 만지지 않음).
- **vision_tracker.py** — OpenCV로 프레임을 읽고 MediaPipe Hand Landmarker로 21개 랜드마크를
  뽑은 뒤, 검지/중지/약지/새끼 4개 손가락의 MCP-PIP-DIP 각도를 3D 월드 좌표로 계산해 평균 굽힘값을
  만들고, 캘리브레이션(펴짐/구부림) 기준으로 0~100%로 정규화합니다. 중앙값 필터 + EMA로 흔들림을
  줄이고, 연속 미검출 프레임 수로 "잠깐 놓침"과 "완전히 놓침"을 구분합니다.
- **command_interpreter.py** — "살짝/반쯤/꽉 쥐어", "NN% 쥐어", "정지", "종료" 등을 정규식과
  키워드 매칭으로 해석합니다. `llm_backend` 콜백을 넣으면 나중에 실제 LLM으로 교체 가능합니다.
- **controller.py** — 목표-현재 오차로 P(비례) 또는 PID 제어를 계산하고, 한 번에 바꿀 수 있는
  최대 단계(`max_step_up`/`max_step_down`)와 안정화 시간(`control_period_s`)을 강제합니다.
  허용오차 안에 `success_hold_seconds` 이상 머물면 성공으로 판정하고 즉시 0으로 내립니다.
- **adaptive_model.py** — 과거 세션에서 "목표에 안정적으로 도달했던(HOLDING/SUCCESS)" 순간의
  (자극값, 실제 굽힘값) 쌍만 골라 1차/2차 회귀를 적합시키고, 다음 실행의 시작 자극값을 추정합니다.
  데이터가 부족하면 항상 안전한 낮은 기본값을 돌려줍니다.
- **serial_link.py** — 포트 자동/수동 탐색, PING/PONG 핸드셰이크, ARM/SET/STOP/STOP_ALL/STATUS
  전송과 응답 대기, 하트비트 송신, 연결 끊김 감지를 담당합니다.
- **safety_manager.py** — `safety_max_intensity`(기본 0) 게이트, 캘리브레이션/손 인식/연결 여부
  전제조건 검사, 연속 자극 시간·전체 실험 시간·쿨다운 타이머, 비상정지 콜백을 관리하는 **유일한**
  안전 판단 창구입니다.
- **simulator.py** — 카메라 없이 자극값→굽힘값 반응(1차 지연 + 비선형 포화 + 노이즈)을 흉내내는
  `HandDynamicsSimulator`와, Arduino 없이 같은 프로토콜에 응답하는 `SimulatedArduino`.
- **data_logger.py** — 세션별 CSV 기록과, `adaptive_model.py`가 학습에 쓸 과거 세션 전체 로드.
- **config.py** — 모든 튜닝 값(제어 게인, 안전 상한, 시리얼 설정, 프리셋 %, ...)의 단일 출처.

---

## 5. Windows 설치 방법 / 6. Python 권장 버전

- **Python 3.10 또는 3.11 (64비트)** 을 권장합니다. (mediapipe/opencv의 Windows용 사전빌드
  wheel이 이 버전대에서 가장 안정적으로 제공됩니다. 이 리포지토리 자체는 3.13에서도 단위
  테스트가 통과했지만, 실제 대회 환경에서는 3.10/3.11처럼 더 널리 검증된 버전을 권장합니다.)
- [python.org](https://www.python.org/downloads/) 에서 설치할 때 **"Add python.exe to PATH"**
  체크박스를 반드시 켜세요.
- Arduino Nano(호환보드, CH340) USB 드라이버가 필요합니다. 대부분 Windows 10/11은 자동으로
  잡지만, 장치관리자에 포트가 안 보이면 "CH340 driver windows"로 검색해 제조사 드라이버를 설치하세요.

## 7. 가상환경 생성과 라이브러리 설치 명령

PowerShell 기준:

```powershell
cd ems_closed_loop
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치 확인:

```powershell
python -m unittest discover -s tests -v
```

`OK`가 뜨고 실패한 테스트가 없으면 준비된 것입니다 (카메라/Arduino 없이도 전부 통과합니다).

---

## 8. Arduino IDE 설정 및 업로드 방법

1. Arduino IDE 설치, 보드 매니저에서 **Arduino AVR Boards** 설치.
2. 툴 > 보드 > **Arduino Nano** 선택, 프로세서는 보드에 맞게 (대부분 ATmega328P, 구형 부트로더면
   "ATmega328P (Old Bootloader)") 선택.
3. `arduino/` 폴더에 아래 라이브러리 파일들이 **이미 들어있습니다** (팀이 실제로 쓰던
   letyourbodymove/openEMSstim 원본에서 그대로 복사해둔 것 -- 수정하지 않았습니다):
   - `AD5252.cpp/.h`, `EMSChannel.cpp/.h`, `EMSSystem.cpp/.h`, `Debug.cpp/.h`
   - `AltSoftSerial.cpp/.h`, `AltSoftSerial_Boards.h`, `AltSoftSerial_Timers.h`,
     `Rn4020BTLe.cpp/.h`, `known_boards.h`, `known_timers.h` (BLE용 -- 기본 설정에서는
     `hardware_config.h`의 `ENABLE_BLUETOOTH_MODULE 0`이라 실제로는 안 쓰이지만, 다른 .cpp가
     참조하지 않으므로 폴더에 있어도 무방합니다)

   ⚠️ 팀의 실제 보드가 이 파일들과 다른 버전(예: `Serial.begin(115200)`을 쓰는 원본 등)으로
   이미 세팅되어 있었다면, 이 프로젝트의 `openemsstim_serial_controller.ino`를 업로드하는 순간
   `hardware_config.h`의 `SERIAL_BAUD_RATE`(19200)로 통일되니 신경 쓰지 않아도 됩니다.
4. 이 프로젝트가 이미 제공하는 `openemsstim_serial_controller.ino` + `hardware_config.h`는
   그대로 두세요 (핀 번호를 새로 지어내지 않고 공식 예제 값을 그대로 재사용합니다 — 2번 섹션 참고).
5. Arduino IDE에서 `arduino/openemsstim_serial_controller.ino`를 엽니다 (같은 폴더의 .h/.cpp가
   자동으로 함께 컴파일됩니다).
6. 보드가 실제로 이 핀 배선과 같은지 다시 한번 확인한 뒤 (2번 섹션), **업로드**.

   ※ 코드 업로드는 항상 USB로만 가능합니다. 블루투스(RN4020)는 프로그램이 이미 올라간 뒤
   명령을 주고받는 용도일 뿐, 코드 자체를 무선으로 넣는 기능은 이 보드에 없습니다.
7. 시리얼 모니터를 19200bps로 열면 부팅 시 다음과 같은 로그가 보여야 합니다:
   ```
   SETUP:
       BT: DISABLED (ENABLE_BLUETOOTH_MODULE=0 in hardware_config.h)
       EMS: INITIALIZING CHANNELS
       EMS: INITIALIZED
       EMS: STARTED
   SETUP DONE (LED 13 WILL BE ON)
       SIMPLE PROTOCOL READY (PING/ARM/SET/STOP/STOP_ALL/STATUS)
   ```
   시리얼 모니터에 `PING`을 입력해 보내면 `PONG`이 와야 합니다 — 이게 되면 통신 준비 완료입니다.

---

## 9. 시뮬레이션 실행 방법

**항상 이 순서로 시작하세요.** 카메라/Arduino가 전혀 없어도 전체 프로그램이 돌아갑니다.

```powershell
python main.py
```

- GUI가 뜨고, 왼쪽 상단 "모드 / 연결"에 `카메라: 시뮬레이션 / Arduino: 시뮬레이션`이 보입니다.
- 자연어 명령창에 `손을 반쯤 쥐어`를 입력하고 **실행**을 누르거나, 60% 빠른 버튼을 누르세요.
- "현재 굽힘값"이 서서히 60%로 올라가고, "EMS 제어값"도 함께 올라가다가 목표 근처에서 멈추고,
  "판단 상태"가 `목표 유지 중` → `성공`으로 바뀌는 걸 그래프로 확인할 수 있습니다.
- 30%, 60%, 90% 각각 시험해서 매번 잘 수렴하는지 확인하세요. 이게 **"이 모드로 30/60/90% 목표
  추종을 검증"** 요구사항입니다.
- `정지`/`종료` 명령, 비상정지 버튼(스페이스/ESC/화면의 빨간 버튼)을 눌러 즉시 0으로 떨어지는지도
  확인하세요.

제어기만 따로, GUI 없이 빠르게 확인하려면:

```powershell
python -m unittest discover -s tests -v
```

---

## 10. LED 또는 측정 장비를 이용한 비인체 테스트 방법

사람 몸에 붙이기 전에 반드시 이 단계를 거치세요.

1. **Arduino 단독 통신 검증** (카메라 불필요):
   ```powershell
   python main.py --live-serial
   ```
   GUI에서 포트를 선택해 연결하고, 명령창에 `손을 살짝 쥐어`를 입력해보세요. 이때 현재 굽힘값은
   시뮬레이션(HandDynamicsSimulator)이 만들어내지만, `EMS 제어값`은 **진짜로 Arduino에 SET
   명령을 보내고 있습니다**. 시리얼 모니터(또는 별도 터미널)로 Arduino가 `OK,1`을 응답하는지
   확인하세요.
2. **LED/측정 장비로 채널 1 출력 확인**: EMS 패드 대신 채널 1의 출력단에 (a) 직렬로 저항을 낀
   LED, 또는 (b) 멀티미터/오실로스코프를 연결해, `EMS 제어값`이 올라갈 때 실제로 출력(LED
   밝기, 전압/저항값)이 따라 올라가는지 확인하세요. `channel1.getIntensity()`는 STATUS 명령으로
   조회할 수 있습니다.
3. 이 단계에서 이상하면 (LED가 안 켜짐, 밝기가 이상함) — 아직 사람에게 연결하지 마세요. 배선,
   AD5252 I2C 주소, 채널 핀 번호(2번 섹션)를 다시 확인하세요.

---

## 11. 실제 장치 연결 전 체크리스트

- [ ] 위 9, 10번 단계를 모두 통과했다.
- [ ] `config.json`의 `safety.safety_max_intensity`를 **전문가 감독 하에** 사람에게 안전한
      값으로 직접 설정했다 (이 리포지토리는 절대 이 숫자를 대신 정해주지 않습니다).
- [ ] `safety.safety_min_intensity` ≤ `safety.safety_max_intensity` 인지 확인했다.
- [ ] 채널 1 전용 패드가 손가락 굽힘근(전완 굴근) 위치에 올바르게 부착되어 있다.
- [ ] 채널 2는 하드웨어 고장 상태이므로 사용하지 않는다 (코드도 이를 거부합니다).
- [ ] **하드웨어 비상정지(E-stop)가 소프트웨어와 별개로 준비되어 있다** — 이 프로그램의 비상정지
      버튼/스페이스바/ESC는 소프트웨어 레벨입니다. USB 케이블을 뽑거나 EMS 기기 자체의 전원
      스위치처럼 프로그램과 무관하게 즉시 자극을 끌 수 있는 물리적 수단을 반드시 별도로 준비하세요.
- [ ] 참가자에게 안전 주의사항을 안내했다 (아래 "안전 경고" 참고).
- [ ] 카메라가 손등/손가락 관절이 보이는 30~45도 대각선 각도로 고정되어 있다.

---

## 12. 실제 실행 방법

체크리스트를 모두 통과했다면:

```powershell
python main.py --live
```

1. GUI에서 포트를 선택하고 **연결**을 누릅니다 (PONG 응답까지 확인됩니다).
2. **① 펴짐 캘리브레이션**을 누르고 손을 쫙 편 상태를 2~3초 유지합니다.
3. **② 구부림 캘리브레이션**을 누르고 주먹을 최대한 쥔 상태를 2~3초 유지합니다.
4. 캘리브레이션 상태가 "완료"로 바뀌면 명령을 입력합니다 (예: `손을 반쯤 쥐어`).
5. 오직 이 시점부터 `safety_manager.can_arm_live()`의 모든 조건(연결/핸드셰이크/캘리브레이션/
   손 인식/시간 제한)이 통과해야 실제 ARM/SET이 나갑니다. 하나라도 실패하면 명령은 그냥
   실행되지 않고, GUI에 그 이유가 표시됩니다.

`--live-camera`만 쓰면 카메라는 실제인데 Arduino는 시뮬레이션이라 EMS 제어값 튜닝을 사람 없이도
카메라 노이즈/조명까지 반영해서 확인할 수 있습니다.

---

## 13. 자연어 명령 사용 예시

| 입력 | 결과 |
|---|---|
| `손을 살짝 쥐어` / `조금만 쥐어` | 목표 30% (설정 파일에서 조정 가능) |
| `손을 반쯤 쥐어` / `반쯤 구부려` | 목표 60% |
| `손을 꽉 쥐어` / `최대한 쥐어` | 목표 90% |
| `40% 쥐어` / `손가락을 75퍼센트 구부려` | 목표 40% / 75% (0~100 어떤 값도 가능) |
| `120% 쥐어` | 오류: "목표 굽힘 값은 0~100% 사이여야 합니다" |
| `정지` / `멈춰` / `손 펴` | 즉시 자극 정지 |
| `종료` | 프로그램 종료 (정지 명령까지 포함) |
| `쥐어` (정도 없음) | 오류: 얼마나 쥘지 명확히 말해달라는 안내 |

30/60/90 프리셋 값은 `config.json`의 `presets.light_percent` / `half_percent` / `strong_percent`
에서 바꿀 수 있습니다.

---

## 14. 예상되는 문제와 해결 방법

| 문제 | 원인/해결 |
|---|---|
| `카메라(index=0)를 열 수 없습니다` | 다른 프로그램(Zoom, 다른 파이썬 창)이 카메라를 쓰고 있는지 확인. `config.json`의 `camera.camera_index`를 0→1로 바꿔보기. |
| `PONG 응답이 없습니다 (핸드셰이크 실패)` | 포트/보율(19200) 확인, Arduino가 실제로 업로드된 상태인지, 다른 프로그램(시리얼 모니터)이 포트를 이미 잡고 있지 않은지 확인. |
| `SET` 이 계속 `ERROR,NOT_ARMED` | GUI에서 아직 연결/ARM 시퀀스가 안 끝난 상태. 연결 버튼을 먼저 누르고 손이 인식/캘리브레이션된 뒤에 명령을 실행하세요. |
| `ERROR,UNSUPPORTED_CHANNEL,2` | 의도된 동작입니다 — 채널 2는 하드웨어 고장으로 비활성화되어 있습니다 (15번 섹션 참고). |
| 손가락 %가 계속 흔들린다 | `config.json`의 `hand_tracking.smoothing_window`, `ema_alpha`를 조정 (창을 키우거나 alpha를 낮추면 더 부드러워지되 반응이 느려집니다). |
| 자극값이 목표를 지나쳐 계속 올라간다 | `control.max_step_up`을 줄이거나 `control.kp`를 낮추세요. PID 모드라면 `ki`를 낮추고 `integral_limit`을 줄이세요. |
| 시뮬레이션에서 자극값이 안 움직인다 | `safety.safety_max_intensity`를 실수로 0인 채로 두고 하드웨어 클램프를 시뮬레이션에도 적용하는 옛 버전이면 이 증상이 납니다 — 이 코드는 시뮬레이션 값에는 하드웨어 안전 상한을 적용하지 않도록 수정되어 있습니다 (그래도 이상하면 `controller.py`/`main.py`에서 `clamp_working_intensity` 사용 여부를 확인). |
| 캘리브레이션이 안 끝난다 | 펴짐/구부림 상태를 계속 카메라에 유지하세요 (기본 30프레임). 손이 화면 밖으로 나가면 그 프레임은 카운트되지 않습니다. |
| GUI가 멈춘 것처럼 느려진다 | 그래프/영상 갱신 주기(`gui.py`의 `POLL_MS`, `GRAPH_POLL_MS`)를 늘려보세요. |

---

## 15. 2채널 복구 후 확장 방법

1. `arduino/hardware_config.h`에서 `#define CHANNEL_2_ENABLED false` → `true`로 변경 후 재업로드.
   (핀 번호 `(6, 7, A3, wiperIndex=3)`은 이미 코드에 있으니 새로 추가할 것이 없습니다.)
2. `serial_link.py`/`simulator.py`는 이미 `channel` 매개변수를 받는 구조라 코드 변경이 필요 없고,
   `main.py`의 `_drive_hardware`/`_tick`에서 두 번째 컨트롤러 인스턴스를 만들어 채널 2에 연결하면
   됩니다 (예: `controller_ch2 = ClosedLoopController(config.control, safety)` 를 추가하고,
   손가락별로 다른 목표를 주고 싶다면 `vision_tracker.py`의 4손가락 평균 대신 손가락별 값을
   그대로 노출하도록 `FrameResult`를 확장하세요 — `_average_finger_bend`를 손가락별 딕셔너리로
   바꾸는 정도의 작은 수정입니다).
3. `command_interpreter.py`에 "검지만 굽혀"처럼 손가락별 명령을 추가하려면 `_PRESET_PHRASES`
   옆에 손가락 이름 키워드 테이블을 추가하고 `CommandResult`에 대상 채널/손가락 필드를 넣으세요.
4. 2채널이 되면 GUI에도 채널별 목표/현재/제어값/오차 행을 하나 더 추가하면 됩니다 (`gui.py`의
   `_build_readout_panel`이 그 패턴을 그대로 보여줍니다).

---

## 안전 경고 (반드시 읽기)

- 이 소프트웨어는 사람에게 안전한 자극 세기를 **정해주지 않습니다**. `safety_max_intensity`는
  반드시 전문가 감독 하에, 실제 참가자와 실제 장비로 별도 검증한 뒤 사람이 직접 입력해야 합니다.
- **심장박동기(페이스메이커) 등 삽입형 전자 의료기기를 사용하는 사람에게는 절대 사용하지 마세요.**
- 상처, 발진, 피부 자극이 있는 부위에는 사용하지 마세요.
- **목, 머리, 가슴을 가로지르는 위치**에는 절대 패드를 부착하지 마세요.
- **승인된(regulation-approved) 외부 EMS/TENS 기기와 그 전용 패드만** 사용하세요. OpenEMSstim은
  자체적으로 자극 신호를 만들지 않는 "진폭 조절기"일 뿐이며, 신호 자체는 외부 EMS 기기가 만듭니다.
- 이 프로그램의 비상정지(스페이스바/ESC/빨간 버튼)는 **소프트웨어 레벨**입니다. USB를 뽑거나
  EMS 기기 전원을 직접 끌 수 있는 **하드웨어 비상정지 수단을 프로그램과 무관하게 별도로** 준비하세요.
- 군사적 용도로 사용을 금지하는 openEMSstim 원 라이선스 조건이 이 프로젝트에도 그대로 적용됩니다.
