# EMS Controller

EMS(전기근육자극) 기반 뇌졸중·마비 환자 재활 보조 시스템 — 한성공학경진대회 16번 팀.

카메라(MediaPipe)로 손/팔의 움직임을 실시간으로 인식해 목표 동작으로 삼고, 그 목표에
가까워지도록 EMS 자극 세기를 자동으로 조절하는 폐루프 제어 시스템입니다.

```
[카메라] --MediaPipe 인식--> [PC: 웹앱]
                                  |  목표 vs 실제 오차 계산 → 폐루프 제어값(0~100) 산출
                                  v
                       [Web Serial, 19200bps]
                                  v
                 [Arduino Nano + openEMSstim 펌웨어]
                                  v
                     [TENS 7000 저주파 자극기] → 전극 패드 → 손/팔 근육
```

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `app.js`, `index.html`, `hand_camera.py`, `*.task`, `run_server.bat` | **웹앱** (실제 시연에 쓰는 최신 버전). `run_server.bat` 실행 후 Chrome/Edge로 `http://localhost:8000` 접속. 거울/행동 보조/개인화/팔 인식 4가지 모드 지원. |
| `python-ems_closed_loop/` | 웹앱 이전에 먼저 만들어진 **파이썬(Tkinter) 프로토타입**. 같은 폐루프 제어 로직을 파이썬으로 구현한 버전 (참고/백업용, 시연에는 웹앱을 사용). |
| `firmware/openemsstim_serial_controller/` | **현재 실제 보드에 올라가 있는 최신 Arduino 펌웨어** (openEMSstim 기반, PING/ARM/SET/STOP 시리얼 프로토콜). |
| `firmware/channel_alternate_test/` | 채널1/채널2가 실제로 어느 핀에서 나오는지 확인할 때 쓴 진단용 스케치. |
| `mediapipehand/` | 2026-08-23 시점 구버전 스냅샷 (참고용, 시연에 쓰지 말 것). |
| `ems_demo_standalone.html` | 위 구버전을 모델까지 통째로 한 파일에 번들링한 공유용 데모 (역시 구버전). |
| `*.pdf` | 공식 예비심사 보고서 및 프로젝트 정리 문서. |

## 안전 관련 필수 사항

- `app.js`의 `config.safety.maxIntensity`(안전 최대 제어값)는 기본값 0으로, 사람이 화면에서
  직접 값을 입력하기 전까지 실제 EMS 출력이 나가지 않습니다.
- 연속 자극 시간 제한(기본 10초), 전체 실험 제한시간(기본 300초), 비상정지(스페이스바/ESC/
  화면 버튼)가 구현되어 있습니다.
- 실제 사람 몸에 전기자극을 가하는 시스템이므로, 안전 상한값은 반드시 전문가 감독 하에
  결정해야 합니다.

## 실행 방법

1. `run_server.bat` 더블클릭 (Python 필요)
2. Chrome 또는 Edge로 `http://localhost:8000` 접속
3. 모드 선택 → 카메라 실행 → 초기값(캘리브레이션) 측정 → Arduino 연결 → 안전값 설정 → 제어 시작
