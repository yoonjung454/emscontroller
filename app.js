import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// 음성 명령(마이크)용 -- initVoiceCommand()가 페이지 초기화 시점(파일 아래쪽의
// init 호출 구간)에 바로 참조하므로 맨 위 상수 선언부에 둔다 (아래쪽 함수
// 정의부에 두면 TDZ로 인해 "Cannot access before initialization" 오류가 난다).
const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
let voiceRecognition = null;
let voiceListening = false;

// ============================================================================
// Supabase (참가자별 캘리브레이션/개인화 프로필 클라우드 저장)
// ============================================================================
// anon key는 클라이언트에 노출돼도 되는 공개 키입니다 (RLS 정책으로 접근 범위를
// 제한). 로그인 없이 참가자 ID 문자열 하나로만 구분하는 프로토타입 구조라,
// 같은 ID를 아는 사람은 그 데이터를 읽고 덮어쓸 수 있다는 점은 감안하세요.
const SUPABASE_URL = "https://boywicwpvicvgfhvfsjn.supabase.co";
const SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJveXdpY3dwdmljdmdmaHZmc2puIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODc3ODI4NzIsImV4cCI6MjEwMzM1ODg3Mn0.Mnzeqh0SmS4iSX3QlT9VsmeDb_XInEzhV9j9bjjop-w";
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// ============================================================================
// 상수 / 설정
// ============================================================================

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

// 엄지 제외 (ems_closed_loop/vision_tracker.py 와 동일한 기준) -- 폐루프 제어(목표
// 평균/채널 출력)는 이 4손가락 기준을 그대로 유지한다. 엄지는 실시간 표시 전용으로만
// 아래 THUMB_JOINTS/DISPLAY_FINGER_KEYS를 통해 별도로 추가한다 (제어에는 관여 안 함).
const FINGERS = {
  Index: [[0, 5, 6], [5, 6, 7], [6, 7, 8]],
  Middle: [[0, 9, 10], [9, 10, 11], [10, 11, 12]],
  Ring: [[0, 13, 14], [13, 14, 15], [14, 15, 16]],
  Little: [[0, 17, 18], [17, 18, 19], [18, 19, 20]]
};
const FINGER_LABELS = { Index: "검지", Middle: "중지", Ring: "약지", Little: "소지", Thumb: "엄지" };
const FINGER_KEYS = Object.keys(FINGERS); // 제어(목표 평균/채널 출력)에 쓰이는 4손가락

// 엄지는 CMC 관절이 다른 손가락과 다른(대립/벌림) 구조라 정확한 굽힘각은 아니지만,
// hand_camera.py와 동일한 3관절 각도합 방식으로 근사해서 표시용으로만 사용한다.
const THUMB_JOINTS = [[0, 1, 2], [1, 2, 3], [2, 3, 4]];
// 실시간 표(target/actual)에 표시할 손가락 전체 -- 제어용 FINGER_KEYS(4) + 엄지(표시 전용).
const DISPLAY_FINGER_KEYS = [...FINGER_KEYS, "Thumb"];

// 왼손 = 목표 동작을 캡처하는 손(SOURCE), 오른손 = EMS를 연결할 손(ACTUAL)
const ROLES = ["source", "actual"];
const ROLE_LABELS = { source: "왼손(TARGET)", actual: "오른손(ACTUAL)" };
const ROLE_COLORS = { source: "#4da6ff", actual: "#ffb454" };

// ============================================================================
// 팔 인식 모드 (MediaPipe Pose Landmarker) -- 손가락 대신 팔꿈치 굽힘을 재는
// 4번째 모드 전용 상수. 손 인식(HandLandmarker)과 완전히 별도의 모델이며,
// "팔 인식" 모드일 때만 동작(perf 절약 -- ensurePoseLandmarker/renderLoop 참고).
// ============================================================================
// Pose Landmarker는 한 사람의 몸 전체를 한 번에 인식하므로(손처럼 손마다 따로
// 나오지 않음), 아래 두 팔(어깨/팔꿈치/손목 인덱스)이 항상 같이 결과에 들어있다.
// 모델 자체의 "왼쪽/오른쪽" 라벨은 신뢰하지 않고(손 인식과 같은 이유 -- 미러링된
// 캔버스를 인식시키므로), 화면에 실제로 보이는 x좌표로 역할을 정한다.
const POSE_ARM_A = { shoulder: 11, elbow: 13, wrist: 15 };
const POSE_ARM_B = { shoulder: 12, elbow: 14, wrist: 16 };
const POSE_VISIBILITY_THRESHOLD = 0.5; // 이 미만이면 가려짐/화면 밖으로 취급

const SMOOTH_WINDOW = 5;
const EMA_ALPHA = 0.35;
const SAMPLE_TARGET = 30; // 캘리브레이션(펴짐/구부림) 프레임 수
const SAMPLE_TIMEOUT_MS = 10000;
const CAPTURE_TARGET = 15; // 왼손 스냅샷 캡처 프레임 수 (약 0.5초)
const CAPTURE_TIMEOUT_MS = 8000;
const MIN_CAL_GAP_DEG = 5;
const HAND_LOSS_FRAMES_THRESHOLD = 10;

const STATE_LABELS = {
  STANDBY: "대기 (제어 시작 전)",
  IDLE: "대기",
  WAITING_FOR_HAND: "오른손 인식 안됨 (대기 중)",
  INCREASING: "구부리는 중 (채널1)",
  DECREASING: "펴는 중 (채널2)",
  HOLDING: "목표 유지 중",
  SUCCESS: "성공",
  LOCKED: "✅ 도달 완료 (자세 유지 중 -- 해제 버튼으로 종료)",
  SAFETY_STOP: "안전 정지",
  SWEEP_HOLD: "개인화 측정 중 (자극 유지)",
  SWEEP_REST: "개인화 측정 중 (휴식)"
};

// 개인화 캘리브레이션(자극값 스윕) 타이밍 -- 폐루프 제어와는 별개의 "열린 루프"
// 테스트다. 정해진 자극값을 순서대로 하나씩 줘보고 결과만 측정한다.
const SWEEP_HOLD_MS = 4000;   // 각 자극값을 유지하는 시간 (근육이 안정될 시간)
const SWEEP_SAMPLE_MS = 1200; // 유지 구간의 마지막 이만큼만 평균내서 기록 (앞부분은 과도기라 버림)
const SWEEP_REST_MS = 2500;   // 다음 단계로 넘어가기 전 0으로 쉬는 시간
const SWEEP_RESEND_MS = 900;  // SET 명령 재전송 주기 (TTL 만료로 자극이 끊기지 않게)

// ============================================================================
// 행동 버튼 (미리 정의된 목표 동작)
// ============================================================================
// "행동"을 고르면 그 행동에 미리 정의된 목표 동작을 그대로 불러와 폐루프
// 제어에 사용한다 (왼손 캡처 대신). 카메라는 오른손 측정(피드백) 용도로 쓰이고,
// 왼손은 아래 "실시간 왼손 연동"을 켰을 때만 쓰인다.
//
// targetPercent는 지금은 전부 PLACEHOLDER 값이다 -- 실제 관절각/손가락 굽힘률을
// 측정해서 넣은 게 아니라, 구조와 UI(행동 선택 → 목표 데이터 생성 → 피드백 비교)를
// 먼저 완성해두기 위한 임시 수치다. targetPerFinger를 나중에 채우면(4손가락 개별
// 목표) 손가락별 제어로 확장할 수 있게 필드를 미리 열어뒀다 (지금은 전부 null --
// 채워지면 targetPercent 대신 이쪽이 우선 사용되도록 selectAction()에서 처리).
const ACTIONS = [
  {
    id: "grip_cup",
    label: "컵 잡기",
    description: "컵이나 음료 용기를 잡는 동작",
    targetPercent: 55,   // placeholder
    targetPerFinger: null // 나중에 { Index: .., Middle: .., Ring: .., Little: .., Thumb: .. } 로 교체
  },
  {
    id: "pour",
    label: "따르기 동작",
    description: "잡은 용기를 기울여 음료를 따르는 동작",
    targetPercent: 45,   // placeholder
    targetPerFinger: null
  },
  {
    id: "shake",
    label: "흔들기 동작",
    description: "용기를 잡고 흔드는 동작",
    targetPercent: 60,   // placeholder
    targetPerFinger: null
  },
  {
    id: "elbow_flex",
    label: "팔 굽히기",
    description: "팔꿈치를 굽히는 동작",
    targetPercent: 70,   // placeholder -- 현재 시스템은 손가락 굽힘 축만 제어하므로 임시로 같은 축을 사용
    targetPerFinger: null
  },
  {
    id: "release",
    label: "물건 놓기",
    description: "잡고 있던 물체를 놓는 동작",
    targetPercent: 5,    // placeholder
    targetPerFinger: null
  }
];

// 제어값 갱신 주기(config.control.controlPeriodMs, 기본 6초)를 이 배수로 나눠서
// 더 빠르게 반응하게 한다.
const CONTROL_SPEED_MULTIPLIER = 2.5;
function getControlPeriodMs() {
  return config.control.controlPeriodMs / CONTROL_SPEED_MULTIPLIER;
}

let selectedAction = null;
// 행동 버튼 대신 쓸 수 있는 실시간 연동 -- 켜져 있으면 매 프레임 왼손(SOURCE)의
// 현재 굽힘값을 그대로 목표로 흘려보낸다 (한 번 얼리는 스냅샷이 아니라 계속
// 갱신됨). 왼손엔 EMS 패치가 없고 카메라로만 측정되고, 실제 전기자극은
// 오른손(ACTUAL)에만 나간다.
let liveMirrorActive = false;

// 앱 모드 -- "mirror"(거울 모드) | "action"(행동 보조 모드, 오픈루프 -- 회원
// 개인화/개인화 측정(수동 램프업)도 이 안에 포함됨, 어차피 개인화 측정 결과가
// 행동 보조의 채널 입력칸으로 들어가는 용도라 같이 묶었다) | "arm"(팔 인식
// 모드, 팔꿈치 버전 거울 모드) | "test"(테스트 모드, 방향키로 채널 직접 테스트).
// 네 모드는 동시에 활성화되지 않는다 -- setAppMode()가 모드를 바꿀 때마다
// 이전 모드에서 돌고 있던 걸 전부 정지시킨다 (아래 설명 참고).
let appMode = "mirror";

// 테스트 모드 -- 지금 방향키(←/→)로 누르고 있는 채널(1|2) | null. 카메라·
// 캘리브레이션·목표비교 없이 순수하게 "이 채널에 지금 전기가 나가는지"만
// 빠르게 확인하는 용도 (channel_alternate_test.ino와 같은 목적을 웹 UI에서).
let testKeyChannel = null;

// 행동 보조 모드 상태 -- "물따르기"/"이두운동" 중 하나만 동시에 실행 가능.
// 거울 모드의 activeChannel/controllerIntensity(채널 1개만 표현 가능한 구조)와
// 달리, 여기는 채널1/채널2를 동시에 서로 다른 값으로 켤 수 있어야 해서
// (예: 물따르기가 두 근육을 동시에 써야 할 수 있음) 별도의 독립적인 상태로 관리한다.
let runningActionKey = null; // null | "spoonLift" | "bicep"
let actionModeInterval = null;

// 개인화 모드(수동 램프업) 상태
let personalizationRampActive = false;
let personalizationRampChannel = 1;
let personalizationRampIntensity = 0;
let personalizationRampInterval = null;
const PERSONALIZATION_RAMP_TICK_MS = 500; // 이 주기마다 세기를 올리고 하드웨어로 재전송

const DEFAULT_CONFIG = {
  presets: { light: 30, half: 60, strong: 90 },
  control: {
    kpUp: 0.02,   // 덜 구부러졌을 때(부족) -- 더 작게: 아주 조심스럽게 증가
    kpDown: 0.3,  // 더 구부러졌을 때(과함) -- 크게: 신전 기능이 없으니 오차에 비례해 빠르게 회수
    tolerancePercent: 5,
    successHoldSeconds: 1.5,
    maxStepUp: 1,       // 한 번에 최대 +1까지만 (더 낮춤)
    maxStepDown: 8,
    controlPeriodMs: 6000 // 근육이 반응할 시간을 더 주기 위해 4초 -> 6초
  },
  safety: {
    maxIntensity: 0,
    minIntensity: 0,
    heartbeatIntervalMs: 400,
    commandTtlMs: 1500,
    maxCommandTtlMs: 5000,
    maxContinuousStimSeconds: 10,
    totalExperimentSeconds: 300,
    cooldownSeconds: 5
  },
  serial: { baudRate: 19200 }
};

// ============================================================================
// 손가락 각도 계산 (hand_camera.py / ems_closed_loop 와 동일한 공식)
// ============================================================================

function calculateJointAngle(a, b, c) {
  const v1 = [a.x - b.x, a.y - b.y, a.z - b.z];
  const v2 = [c.x - b.x, c.y - b.y, c.z - b.z];
  const norm1 = Math.hypot(v1[0], v1[1], v1[2]);
  const norm2 = Math.hypot(v2[0], v2[1], v2[2]);
  const denom = norm1 * norm2;
  if (denom === 0) return 180.0;
  let cosine = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / denom;
  cosine = Math.min(1, Math.max(-1, cosine));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function calculateFingerBend(worldLandmarks, joints) {
  let total = 0;
  for (const [a, b, c] of joints) {
    const angle = calculateJointAngle(worldLandmarks[a], worldLandmarks[b], worldLandmarks[c]);
    total += Math.max(0, 180 - angle);
  }
  return Math.min(total, 270);
}

// 손가락(관절 3개 합)과 달리 팔꿈치는 관절 1개뿐이라 calculateFingerBend를 그대로
// 쓰지 않고 calculateJointAngle만 재사용한다 -- 계산 공식(세 점 사이 각도, "편
// 상태=180도"→"굽힘=0"으로 뒤집기)은 손가락과 완전히 동일하다.
function calculateArmBend(worldLandmarks, shoulderIdx, elbowIdx, wristIdx) {
  const angle = calculateJointAngle(worldLandmarks[shoulderIdx], worldLandmarks[elbowIdx], worldLandmarks[wristIdx]);
  return Math.max(0, 180 - angle);
}

class MedianEmaFilter {
  constructor(windowSize, alpha) {
    this.windowSize = windowSize;
    this.alpha = alpha;
    this.buffer = [];
    this.ema = null;
  }
  push(value) {
    this.buffer.push(value);
    if (this.buffer.length > this.windowSize) this.buffer.shift();
    const sorted = [...this.buffer].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    this.ema = this.ema === null ? median : this.alpha * median + (1 - this.alpha) * this.ema;
    return this.ema;
  }
  reset() {
    this.buffer = [];
    this.ema = null;
  }
}

// ============================================================================
// 자연어 명령 해석 (command_interpreter.py 를 JS로 포팅, 규칙 기반, LLM 없음)
// ============================================================================

const STOP_WORDS = ["정지", "멈춰", "멈춤", "스탑", "stop", "손 펴", "손펴"];
const QUIT_WORDS = ["종료", "끝내", "quit", "exit"];
const PRESET_PHRASES = [
  [["살짝", "조금만", "약하게"], "light"],
  [["최대한", "꽉", "세게", "강하게"], "strong"],
  [["반쯤", "반만", "반 정도", "절반"], "half"]
];
const PERCENT_PATTERN = /(\d{1,3}(?:\.\d+)?)\s*(?:%|퍼센트|프로)/;

function interpretCommand(text) {
  text = (text || "").trim();
  if (!text) return { action: "error", message: "명령이 비어 있습니다." };
  const lowered = text.toLowerCase();

  for (const w of QUIT_WORDS) {
    if (lowered.includes(w)) return { action: "stop", message: "웹 데모에서는 '종료'도 자극 정지로 처리됩니다." };
  }
  for (const w of STOP_WORDS) {
    if (lowered.includes(w)) return { action: "stop", message: "EMS 자극을 정지합니다." };
  }

  const match = text.match(PERCENT_PATTERN);
  if (match) {
    const value = parseFloat(match[1]);
    if (value < 0 || value > 100) {
      return { action: "error", message: `목표 굽힘 값은 0~100% 사이여야 합니다 (입력값: ${value}%).` };
    }
    return { action: "set_target", targetPercent: value, message: `목표 굽힘 ${value.toFixed(0)}% 설정` };
  }

  for (const [keywords, name] of PRESET_PHRASES) {
    if (keywords.some((k) => text.includes(k))) {
      const percent = { light: config.presets.light, half: config.presets.half, strong: config.presets.strong }[name];
      return { action: "set_target", targetPercent: percent, message: `목표 굽힘 ${percent.toFixed(0)}% 설정 (${name})` };
    }
  }

  if (text.includes("쥐어") || text.includes("구부려") || text.includes("쥐기")) {
    return {
      action: "error",
      message: "얼마나 쥘지 명확하지 않습니다. '살짝/반쯤/꽉 쥐어' 또는 '45% 쥐어'처럼 정도를 포함해 말해주세요."
    };
  }

  return {
    action: "error",
    message: "이해하지 못한 명령입니다. 예: '손을 살짝 쥐어', '반쯤 쥐어', '꽉 쥐어', '45% 쥐어', '정지'"
  };
}

// ============================================================================
// 상태
// ============================================================================

let config = loadConfig();
let calibration = loadCalibration(); // { flat: {finger:deg}, bent: {finger:deg} } -- 두 손 공용
let adaptiveModel = loadAdaptiveModel(); // { coeffs: [slope, intercept] | null, sampleCount }
let personalization = loadPersonalization(); // { points: [{intensity,percent}], coeffs: [slope,intercept]|null, fittedAt }
let sweep = null; // 개인화 캘리브레이션 진행 중 상태, 없으면 null

let isRunning = false;
let handLandmarker = null;
let poseLandmarker = null; // "팔 인식" 모드에서만 생성/사용 (ensurePoseLandmarker 참고)
let lastVideoTime = -1;
let invertHandedness = false;

// 손별(source/actual) 필터 · 최신값
const filters = { source: {}, actual: {} };
for (const role of ROLES) for (const finger of DISPLAY_FINGER_KEYS) filters[role][finger] = new MedianEmaFilter(SMOOTH_WINDOW, EMA_ALPHA);

const detectedThisFrame = { source: false, actual: false };
let consecutiveMissActual = 0;
let handLostSustained = true; // ACTUAL(오른손) 기준, 안전 판단에 사용

// 팔별(source/actual) 필터 · 최신값 -- 손가락과 구조는 같지만 팔은 값이 하나뿐이라
// (손가락처럼 4개가 아님) finger 키로 순회하는 객체 대신 단순 source/actual 필드로 둔다.
const armFilters = { source: new MedianEmaFilter(SMOOTH_WINDOW, EMA_ALPHA), actual: new MedianEmaFilter(SMOOTH_WINDOW, EMA_ALPHA) };
const armLatestSmoothed = { source: null, actual: null }; // role -> degrees
const armLatestPercent = { source: null, actual: null }; // role -> 0-100 | null
const armDetectedThisFrame = { source: false, actual: false };
let consecutiveMissArmActual = 0;
let armLostSustained = true; // ACTUAL(오른팔) 기준, "팔 인식" 모드의 안전 판단에 사용
let armSampling = null; // 팔 초기값(펴짐/구부림) 측정 -- { mode, sum, count, startedAt }

const latestSmoothed = { source: {}, actual: {} }; // role -> finger -> degrees
const latestPercent = { source: {}, actual: {} }; // role -> finger -> 0-100 | null

let targetPerFinger = { Index: null, Middle: null, Ring: null, Little: null, Thumb: null };
let currentAverage = null; // ACTUAL(오른손) 4손가락 평균

let sampling = null; // 초기값(펴짐/구부림) 측정, ACTUAL(오른손) 기준
let capturing = null; // 왼손 스냅샷 캡처

// 폐루프 제어기 상태
let targetPercent = null; // targetPerFinger 의 평균값
let controllerIntensity = 0; // "작업용" 0~100, 하드웨어 안전상한과는 별개 -- activeChannel이 가리키는
                              // 채널에 지금 이 세기가 적용된다 (두 채널 값을 따로 들고 있지 않고,
                              // "지금 활성 채널 + 그 세기" 한 쌍으로만 표현 -- 이러면 두 채널이 동시에
                              // 켜지는 상황 자체가 구조적으로 생길 수 없다).
let activeChannel = 1; // 1 = 채널1(구부림/굴근), 2 = 채널2(폄/신근)
let successSince = null;
let lastStepTime = 0;
let controlState = "IDLE";
let lastError = null;

// 도달 완료 후 "자세 유지(LOCKED)" 상태 -- 해제 버튼(또는 ■ 제어 정지)을 누르기
// 전까지 target을 지우지 않고 지금 세기를 그대로 계속 흘린다 (연속 출력).
let holdLocked = false;

// 안전 관리자 상태
// 예전엔 별도 "라이브 모드" 체크박스로 껐다 켰다 했는데, ▶ 제어 시작 버튼
// (controlEnabled)이 이미 같은 역할을 해서 없앴다. Arduino 연결 여부와
// 안전 최대값이 여전히 실제 출력을 막아주는 별개의 게이트로 남아있다.
const liveModeRequested = true;
let safetyTripped = false;
let safetyTripReason = "";
let experimentStart = null;
let continuousStimStart = null;
let cooldownUntil = null;
let armedCh1 = false;
let armedCh2 = false;
let lastDrivenChannel = null; // 직전에 실제로 SET을 보낸 채널 -- 채널이 바뀌면 이전 채널부터 확실히 끈다
let controlEnabled = false; // "▶ 제어 시작" 버튼을 눌러야만 true -- 이게 false면 무슨 일이 있어도 하드웨어로 안 나감
let lastHardwareSendTime = 0; // 하드웨어 전송을 control_period_ms 주기로만 제한 (카메라 프레임마다 보내면 명령이 밀려서 큐가 쌓임)

// 시리얼
let serialLink = null;

// 로깅
let logRows = []; // { t, target, current, intensity, error, state, success, handDetected }

// 실제 AI(Claude) 명령 해석 -- 꺼져있거나 키가 없거나 호출이 실패하면
// 항상 규칙 기반 interpretCommand()로 대체된다 (아래 submitCommand 참고).
let aiConfig = loadAiConfig();

// ============================================================================
// DOM
// ============================================================================

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusBadgeSource = document.getElementById("statusBadgeSource");
const statusBadgeActual = document.getElementById("statusBadgeActual");
const runBtn = document.getElementById("runBtn");
const invertHandsCheckbox = document.getElementById("invertHandsCheckbox");
const calFlatBtn = document.getElementById("calFlatBtn");
const calBentBtn = document.getElementById("calBentBtn");
const resetBtn = document.getElementById("resetBtn");
const tableBody = document.getElementById("fingerTableBody");

// 예전 "거울 모드"(왼손 캡처/명령/프리셋/AI 해석) UI가 제거되면서, 그 카드에
// 있던 엘리먼트를 가리키던 참조들도 같이 비활성화했다 (없는 엘리먼트를
// getElementById 하면 null이 되고, 그 뒤 .addEventListener 등을 부르는 순간
// 스크립트 전체가 죽기 때문). 관련 함수(startCapture/submitCommand/
// interpretCommandWithAI 등)는 그대로 남아있으니, index.html의 주석 처리된
// mirrorTargetCard와 함께 이 줄들만 다시 살리면 복구된다.
// const sourceLiveTableBody = document.getElementById("sourceLiveTableBody");
// const mirrorTargetCard = document.getElementById("mirrorTargetCard");

// 모드 선택 (거울 / 행동 보조 / 개인화 / 팔 인식) -- 4개
const modeMirrorBtn = document.getElementById("modeMirrorBtn");
const modeActionBtn = document.getElementById("modeActionBtn");
const modeArmBtn = document.getElementById("modeArmBtn");
const modeTestBtn = document.getElementById("modeTestBtn");
const modeDescriptionText = document.getElementById("modeDescriptionText");
const mirrorModeCard = document.getElementById("mirrorModeCard");
const actionModeCard = document.getElementById("actionModeCard");
const armModeCard = document.getElementById("armModeCard");
const testModeCard = document.getElementById("testModeCard");

const actionButtonsRow = document.getElementById("actionButtonsRow");
const selectedActionText = document.getElementById("selectedActionText");
const actionDescriptionText = document.getElementById("actionDescriptionText");
const targetCompareLabel = document.getElementById("targetCompareLabel");
const actualCompareLabel = document.getElementById("actualCompareLabel");
const liveMirrorBtn = document.getElementById("liveMirrorBtn");

// 팔 인식 모드
const armCalFlatBtn = document.getElementById("armCalFlatBtn");
const armCalBentBtn = document.getElementById("armCalBentBtn");
const armResetBtn = document.getElementById("armResetBtn");
const armHandTargetDisplay = document.getElementById("armHandTargetDisplay");
const armHandActualDisplay = document.getElementById("armHandActualDisplay");
const armHandErrorDisplay = document.getElementById("armHandErrorDisplay");

// 테스트 모드
const testIntensityInput = document.getElementById("testIntensityInput");
const testModeStatusText = document.getElementById("testModeStatusText");

// 행동 보조 모드 (오픈루프 고정 전류)
const spoonLiftCh1Input = document.getElementById("spoonLiftCh1Input");
const spoonLiftCh2Input = document.getElementById("spoonLiftCh2Input");
const spoonLiftBtn = document.getElementById("spoonLiftBtn");
const bicepCh1Input = document.getElementById("bicepCh1Input");
const bicepCh2Input = document.getElementById("bicepCh2Input");
const bicepRepsInput = document.getElementById("bicepRepsInput");
const bicepBtn = document.getElementById("bicepBtn");
const actionModeStatusText = document.getElementById("actionModeStatusText");

// 회원 개인화 (행동 보조 모드)
const memberIdInput = document.getElementById("memberIdInput");
const memberNameInput = document.getElementById("memberNameInput");
const memberLoadBtn = document.getElementById("memberLoadBtn");
const memberRegisterBtn = document.getElementById("memberRegisterBtn");
const memberWelcomeText = document.getElementById("memberWelcomeText");
const voiceCommandBtn = document.getElementById("voiceCommandBtn");
const voiceCommandStatusText = document.getElementById("voiceCommandStatusText");

// 개인화 모드 (수동 램프업)
const personalizationCh1Radio = document.getElementById("personalizationCh1Radio");
const personalizationCh2Radio = document.getElementById("personalizationCh2Radio");
const personalizationRampStepInput = document.getElementById("personalizationRampStepInput");
const personalizationRampBtn = document.getElementById("personalizationRampBtn");
const personalizationRampValue = document.getElementById("personalizationRampValue");
const personalizationRampStatusText = document.getElementById("personalizationRampStatusText");
const personalizationRampResultText = document.getElementById("personalizationRampResultText");
const personalizationTargetSpoon = document.getElementById("personalizationTargetSpoon");
const personalizationTargetBicep = document.getElementById("personalizationTargetBicep");
const personalizationSaveBtn = document.getElementById("personalizationSaveBtn");
const maxContinuousInput = document.getElementById("maxContinuousInput");
const totalExperimentInput = document.getElementById("totalExperimentInput");
const resetExperimentTimerBtn = document.getElementById("resetExperimentTimerBtn");
const experimentTimerPill = document.getElementById("experimentTimerPill");
const holdBehaviorHint = document.getElementById("holdBehaviorHint");

// const captureBtn = document.getElementById("captureBtn");
// const captureOverlay = document.getElementById("captureOverlay");
// const captureTitle = document.getElementById("captureTitle");
// const captureProgress = document.getElementById("captureProgress");

// const commandInput = document.getElementById("commandInput");
// const commandRunBtn = document.getElementById("commandRunBtn");
// const commandMessage = document.getElementById("commandMessage");
// const presetLightBtn = document.getElementById("presetLight");
// const presetHalfBtn = document.getElementById("presetHalf");
// const presetStrongBtn = document.getElementById("presetStrong");

// const aiEnabledCheckbox = document.getElementById("aiEnabledCheckbox");
// const aiConfigRow = document.getElementById("aiConfigRow");
// const aiApiKeyInput = document.getElementById("aiApiKeyInput");
// const aiModelInput = document.getElementById("aiModelInput");
// const aiStatusBadge = document.getElementById("aiStatusBadge");
// const ruleBasedHint = document.getElementById("ruleBasedHint");

const safetyMaxInput = document.getElementById("safetyMaxInput");
const safetyMaxPill = document.getElementById("safetyMaxPill");
const safetyTripRow = document.getElementById("safetyTripRow");
const safetyTripReasonText = document.getElementById("safetyTripReasonText");
const safetyTripHint = document.getElementById("safetyTripHint");
const clearSafetyTripBtn = document.getElementById("clearSafetyTripBtn");
const serialConnectBtn = document.getElementById("serialConnectBtn");
const serialDisconnectBtn = document.getElementById("serialDisconnectBtn");
const serialStatusPill = document.getElementById("serialStatusPill");
const serialSupportHint = document.getElementById("serialSupportHint");
const startControlBtn = document.getElementById("startControlBtn");
const releaseHoldBtn = document.getElementById("releaseHoldBtn");

const personalizationBtn = document.getElementById("personalizationBtn");
const personalizationStatusLine = document.getElementById("personalizationStatusLine");
const personalizationSummary = document.getElementById("personalizationSummary");
const personalizationBadge = document.getElementById("personalizationBadge");
const graphPersonalizationCanvas = document.getElementById("graphPersonalization");
const graphPersonalizationCtx = graphPersonalizationCanvas.getContext("2d");

const targetValueDisplay = document.getElementById("targetValueDisplay");
const actualValueDisplay = document.getElementById("actualValueDisplay");
const errorValueDisplay = document.getElementById("errorValueDisplay");
const stimValueDisplay = document.getElementById("stimValueDisplay");
const activeChannelPill = document.getElementById("activeChannelPill");
const hardwareSentHint = document.getElementById("hardwareSentHint");
const elapsedDisplay = document.getElementById("elapsedDisplay");
const controlStateText = document.getElementById("controlStateText");
const controlLog = document.getElementById("controlLog");

const graphPercentCanvas = document.getElementById("graphPercent");
const graphIntensityCanvas = document.getElementById("graphIntensity");
const graphPercentCtx = graphPercentCanvas.getContext("2d");
const graphIntensityCtx = graphIntensityCanvas.getContext("2d");

const logCountText = document.getElementById("logCountText");
const downloadCsvBtn = document.getElementById("downloadCsvBtn");
const clearLogBtn = document.getElementById("clearLogBtn");
const estopBtn = document.getElementById("estopBtn");
const toastEl = document.getElementById("toast");

// 참가자 프로필(Supabase) UI -- 나중에 다시 추가할 예정이라 지금은 index.html에서
// 카드 자체를 뺐다. 그래서 이 DOM 참조들도 같이 비활성화 (없는 엘리먼트를 참조하면
// null이 되고, 그 뒤 .addEventListener 등을 부르는 순간 스크립트 전체가 죽는다).
// 아래쪽의 checkSupabaseConnection/loadProfileFromSupabase/saveProfileToSupabase
// 함수 자체는 그대로 남겨뒀으니, 카드를 다시 추가할 때 이 다섯 줄과 관련
// 이벤트 바인딩/초기화 호출만 다시 살리면 된다.
// const supabaseStatusBadge = document.getElementById("supabaseStatusBadge");
// const participantIdInput = document.getElementById("participantIdInput");
// const loadProfileBtn = document.getElementById("loadProfileBtn");
// const saveProfileBtn = document.getElementById("saveProfileBtn");
// const profileStatusText = document.getElementById("profileStatusText");

// ============================================================================
// 토스트 알림 (목표 동작 설정 완료/실패 등을 눈에 잘 띄게 알려줌)
// ============================================================================

let toastHideTimer = null;
function showToast(message, kind = "ok", durationMs = 2800) {
  toastEl.textContent = message;
  toastEl.className = "toast show" + (kind === "warn" ? " warn" : kind === "bad" ? " bad" : "");
  if (toastHideTimer) clearTimeout(toastHideTimer);
  toastHideTimer = setTimeout(() => toastEl.classList.remove("show"), durationMs);
}

// 토스트는 금방 사라져서 놓치기 쉬우니까, 버튼 자체에도 잠깐 "처리됐다" 표시를
// 남긴다 (색이 바뀌고, 원하면 텍스트도 잠깐 바뀜 -- 지정한 시간 뒤 원래대로 복구).
function flashButtonPress(btn, tempLabel, durationMs = 1200) {
  const original = btn.textContent;
  if (tempLabel) btn.textContent = tempLabel;
  btn.classList.add("flash-confirm");
  setTimeout(() => {
    btn.classList.remove("flash-confirm");
    if (tempLabel) btn.textContent = original;
  }, durationMs);
}

// ============================================================================
// 초기화
// ============================================================================

buildTable();
// buildSourceLiveTable(); -- 예전 거울 모드 카드 제거로 비활성화 (위 DOM 참조 주석 참고)
buildActionButtons();
initVoiceCommand();
initSafetyUi();
initPresetLabels();
// AI 명령 해석(Gemini) UI는 예전 거울 모드 카드와 함께 제거됨 -- 아래 3줄과
// updateAiUi() 호출 비활성화 (관련 함수/설정 로드-저장 로직은 그대로 남아있음)
// aiEnabledCheckbox.checked = aiConfig.enabled;
// aiApiKeyInput.value = aiConfig.apiKey;
// aiModelInput.value = aiConfig.model;
// updateAiUi();
resizeGraphs();
window.addEventListener("resize", resizeGraphs);
setInterval(drawGraphs, 250);
setInterval(updateElapsedDisplay, 500);
updatePersonalizationSummary();
drawPersonalizationGraph();
// participantIdInput.value = loadStoredParticipantId();
// checkSupabaseConnection();

if (!("serial" in navigator)) {
  serialSupportHint.textContent = "⚠ 이 브라우저는 Web Serial API를 지원하지 않습니다. Chrome 또는 Edge에서 http://localhost 로 열어주세요.";
  serialConnectBtn.disabled = true;
}

// ============================================================================
// 이벤트 바인딩
// ============================================================================

runBtn.addEventListener("click", () => (isRunning ? stopRun() : startRun()));
invertHandsCheckbox.addEventListener("change", () => { invertHandedness = invertHandsCheckbox.checked; });
calFlatBtn.addEventListener("click", () => startSampling("flat"));
calBentBtn.addEventListener("click", () => startSampling("bent"));
resetBtn.addEventListener("click", () => {
  calibration = { flat: {}, bent: {} };
  saveCalibration();
  buildTable();
  showToast("🔄 초기값이 초기화되었습니다", "warn");
});
armCalFlatBtn.addEventListener("click", () => startArmSampling("flat"));
armCalBentBtn.addEventListener("click", () => startArmSampling("bent"));
armResetBtn.addEventListener("click", () => {
  delete calibration.flat.Arm;
  delete calibration.bent.Arm;
  saveCalibration();
  showToast("🔄 팔 초기값이 초기화되었습니다", "warn");
});
// 예전 거울 모드 카드(캡처/명령/프리셋)의 이벤트 바인딩 -- 카드가 제거되어
// 비활성화 (관련 함수 자체는 아래에 그대로 남아있음, 복구 가능)
// captureBtn.addEventListener("click", startCapture);
liveMirrorBtn.addEventListener("click", toggleLiveMirror);

modeMirrorBtn.addEventListener("click", () => setAppMode("mirror"));
modeActionBtn.addEventListener("click", () => setAppMode("action"));
modeArmBtn.addEventListener("click", () => setAppMode("arm"));
modeTestBtn.addEventListener("click", () => setAppMode("test"));

spoonLiftBtn.addEventListener("click", () => startSequentialRamp("spoonLift", 2000, 3000));
bicepBtn.addEventListener("click", startBicepRoutine);

memberLoadBtn.addEventListener("click", loadMember);
memberIdInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadMember();
});
memberRegisterBtn.addEventListener("click", registerMember);
voiceCommandBtn.addEventListener("click", toggleVoiceCommand);

personalizationRampBtn.addEventListener("click", () => {
  if (personalizationRampActive) stopPersonalizationRamp("사용자가 정지 버튼을 눌렀습니다");
  else startPersonalizationRamp();
});
personalizationSaveBtn.addEventListener("click", savePersonalizationValue);

// commandRunBtn.addEventListener("click", submitCommand);
// commandInput.addEventListener("keydown", (e) => {
//   if (e.key === "Enter") submitCommand();
// });
// presetLightBtn.addEventListener("click", () => applyPresetTarget(config.presets.light, "light"));
// presetHalfBtn.addEventListener("click", () => applyPresetTarget(config.presets.half, "half"));
// presetStrongBtn.addEventListener("click", () => applyPresetTarget(config.presets.strong, "strong"));

safetyMaxInput.addEventListener("change", () => {
  config.safety.maxIntensity = Math.max(0, Math.min(100, Number(safetyMaxInput.value) || 0));
  safetyMaxInput.value = config.safety.maxIntensity;
  saveConfig();
  updateSafetyPill();
});

maxContinuousInput.addEventListener("change", () => {
  config.safety.maxContinuousStimSeconds = Math.max(1, Math.min(600, Number(maxContinuousInput.value) || 10));
  maxContinuousInput.value = config.safety.maxContinuousStimSeconds;
  saveConfig();
});

totalExperimentInput.addEventListener("change", () => {
  config.safety.totalExperimentSeconds = Math.max(10, Math.min(36000, Number(totalExperimentInput.value) || 300));
  totalExperimentInput.value = config.safety.totalExperimentSeconds;
  saveConfig();
});
resetExperimentTimerBtn.addEventListener("click", () => {
  // "켜자마자 바로 꺼짐" 증상의 흔한 원인 -- 카메라를 맨 처음 켠 시점부터 세는
  // 전체 실험 타이머가 이미 다 찬 경우. 지금부터 다시 세도록 리셋한다.
  experimentStart = performance.now();
  logControl("⏱ 전체 실험 타이머를 초기화했습니다");
  showToast("⏱ 실험 타이머 초기화됨", "ok");
});

clearSafetyTripBtn.addEventListener("click", () => {
  const hadReason = safetyTripReason;
  resetAfterTrip();
  logControl(`✅ 안전 정지 해제됨 (직전 원인: ${hadReason})`);
  showToast("✅ 안전 정지 해제됨 -- 다시 사용할 수 있습니다", "ok");
});

// AI 명령 해석 UI도 예전 거울 모드 카드와 함께 제거되어 비활성화
// aiEnabledCheckbox.addEventListener("change", () => {
//   aiConfig.enabled = aiEnabledCheckbox.checked;
//   saveAiConfig();
//   updateAiUi();
// });
// aiApiKeyInput.addEventListener("change", () => {
//   aiConfig.apiKey = aiApiKeyInput.value.trim();
//   saveAiConfig();
// });
// aiModelInput.addEventListener("change", () => {
//   aiConfig.model = aiModelInput.value.trim() || "gemini-3.7-flash";
//   aiModelInput.value = aiConfig.model;
//   saveAiConfig();
//   updateAiUi();
// });

startControlBtn.addEventListener("click", () => {
  controlEnabled = !controlEnabled;
  if (controlEnabled) {
    startControlBtn.textContent = "■ 제어 정지";
    startControlBtn.classList.add("running");
    logControl("▶ 제어 시작 -- 이제부터 목표와 비교해서 자극값이 조절됩니다");
    showToast("▶ 제어 시작", "ok");
  } else {
    startControlBtn.textContent = "▶ 제어 시작";
    startControlBtn.classList.remove("running");
    controllerIntensity = 0;
    if (liveModeRequested && serialLink) serialLink.stopAll().catch(() => {});
    armedCh1 = false;
    armedCh2 = false;
    lastDrivenChannel = null;
    logControl("■ 제어 정지 -- 자극을 멈췄습니다 (목표는 유지됨, 다시 시작하려면 버튼을 다시 누르세요)");
    showToast("■ 제어 정지", "warn");
  }
});

releaseHoldBtn.addEventListener("click", () => releaseHold("사용자가 해제 버튼을 눌렀습니다"));

serialConnectBtn.addEventListener("click", connectSerial);
serialDisconnectBtn.addEventListener("click", disconnectSerial);

personalizationBtn.addEventListener("click", () => {
  if (sweep) abortSweep("사용자가 중지 버튼을 눌렀습니다");
  else startPersonalizationSweep();
});

downloadCsvBtn.addEventListener("click", downloadCsv);
clearLogBtn.addEventListener("click", () => {
  logRows = [];
  updateLogCount();
});

estopBtn.addEventListener("click", () => triggerEmergencyStop("사용자 비상정지 버튼"));
window.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  const isTyping = tag === "INPUT" || tag === "TEXTAREA";
  if (e.code === "Escape") {
    e.preventDefault();
    triggerEmergencyStop("키보드 비상정지 (ESC)");
  } else if (e.code === "Space" && !isTyping) {
    e.preventDefault();
    triggerEmergencyStop("키보드 비상정지 (SPACE)");
  } else if ((e.key === "a" || e.key === "A") && !isTyping && personalizationRampActive) {
    e.preventDefault();
    stopPersonalizationRamp("사용자가 A 키를 눌러 정지");
  } else if (appMode === "test" && !isTyping && (e.code === "ArrowLeft" || e.code === "ArrowRight")) {
    e.preventDefault();
    testModeKeyDown(e.code === "ArrowLeft" ? 1 : 2);
  }
});
window.addEventListener("keyup", (e) => {
  if (appMode === "test" && (e.code === "ArrowLeft" || e.code === "ArrowRight")) {
    testModeKeyUp(e.code === "ArrowLeft" ? 1 : 2);
  }
});
window.addEventListener("blur", () => {
  // 방향키를 누른 채로 알트탭 등으로 창 포커스가 빠지면 keyup이 아예 안 올 수
  // 있다 -- 자극이 계속 나가는 채로 창만 바뀌는 걸 막기 위한 안전망.
  if (appMode === "test" && testKeyChannel !== null) {
    testModeKeyUp(testKeyChannel);
  }
});
document.addEventListener("visibilitychange", () => {
  // 브라우저 탭이 백그라운드로 가면 requestAnimationFrame/타이머가 느려지거나
  // 멈출 수 있어, 자극이 계속되는데 화면 갱신/워치독은 멈추는 위험한 상황이
  // 생길 수 있다. 그래서 탭이 보이지 않게 되는 순간 즉시 안전 정지한다.
  // 행동 보조/개인화 모드는 카메라(isRunning)와 무관하게 setInterval로 계속
  // 돌 수 있으므로, 이 두 상태도 같이 확인한다. 테스트 모드는 방향키를 누르고
  // 있는 동안에만 나가므로 keyup 대신 즉시 채널을 끈다(전체 비상정지까지는 안 함).
  if (document.hidden && (isRunning || runningActionKey || personalizationRampActive)) {
    triggerEmergencyStop("브라우저 탭이 백그라운드로 전환됨");
  }
  if (document.hidden && appMode === "test" && testKeyChannel !== null) {
    testModeKeyUp(testKeyChannel);
  }
});
window.addEventListener("beforeunload", () => {
  if (serialLink) serialLink.stopAll().catch(() => {});
});

// 참가자 프로필 카드가 지금 UI에 없어서 이 바인딩들도 같이 비활성화 (위 DOM 참조 주석 참고)
// loadProfileBtn.addEventListener("click", loadProfileFromSupabase);
// saveProfileBtn.addEventListener("click", saveProfileToSupabase);
// participantIdInput.addEventListener("keydown", (e) => {
//   if (e.key === "Enter") loadProfileFromSupabase();
// });

// ============================================================================
// 설정 / 캘리브레이션 / 개인화 모델 저장·불러오기 (localStorage)
// ============================================================================

function loadConfig() {
  try {
    const raw = localStorage.getItem("emsWebConfig");
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        presets: { ...DEFAULT_CONFIG.presets, ...(parsed.presets || {}) },
        control: { ...DEFAULT_CONFIG.control, ...(parsed.control || {}) },
        safety: { ...DEFAULT_CONFIG.safety, ...(parsed.safety || {}) },
        serial: { ...DEFAULT_CONFIG.serial, ...(parsed.serial || {}) }
      };
    }
  } catch (e) { /* ignore, fall back to defaults */ }
  return JSON.parse(JSON.stringify(DEFAULT_CONFIG));
}
function saveConfig() {
  localStorage.setItem("emsWebConfig", JSON.stringify(config));
}

function loadCalibration() {
  try {
    const raw = localStorage.getItem("emsWebCalibration");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && parsed.flat && parsed.bent) return parsed;
    }
  } catch (e) { /* ignore */ }
  return { flat: {}, bent: {} };
}
function saveCalibration() {
  localStorage.setItem("emsWebCalibration", JSON.stringify(calibration));
}

function loadAdaptiveModel() {
  try {
    const raw = localStorage.getItem("emsWebAdaptiveModel");
    if (raw) return JSON.parse(raw);
  } catch (e) { /* ignore */ }
  return { coeffs: null, sampleCount: 0 };
}
function saveAdaptiveModel() {
  localStorage.setItem("emsWebAdaptiveModel", JSON.stringify(adaptiveModel));
}

function loadPersonalization() {
  try {
    const raw = localStorage.getItem("emsWebPersonalization");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.points)) return parsed;
    }
  } catch (e) { /* ignore */ }
  return { points: [], coeffs: null, fittedAt: null };
}
function savePersonalization() {
  localStorage.setItem("emsWebPersonalization", JSON.stringify(personalization));
}

function loadAiConfig() {
  try {
    const raw = localStorage.getItem("emsWebAiConfig");
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        enabled: !!parsed.enabled,
        apiKey: parsed.apiKey || "",
        model: parsed.model || "gemini-3.7-flash"
      };
    }
  } catch (e) { /* ignore */ }
  return { enabled: false, apiKey: "", model: "gemini-3.7-flash" };
}
function saveAiConfig() {
  localStorage.setItem("emsWebAiConfig", JSON.stringify(aiConfig));
}
function updateAiUi() {
  aiConfigRow.style.display = aiConfig.enabled ? "block" : "none";
  if (aiConfig.enabled) {
    aiStatusBadge.textContent = `🤖 AI(${aiConfig.model}) 사용 중`;
    aiStatusBadge.style.color = "var(--accent-2)";
    aiStatusBadge.style.background = "rgba(110,231,183,0.12)";
    ruleBasedHint.style.display = "none";
  } else {
    aiStatusBadge.textContent = "실제 AI 아님 · 정해진 표현/패턴만 인식";
    aiStatusBadge.style.color = "";
    aiStatusBadge.style.background = "";
    ruleBasedHint.style.display = "";
  }
}

function initSafetyUi() {
  safetyMaxInput.value = config.safety.maxIntensity;
  maxContinuousInput.value = config.safety.maxContinuousStimSeconds;
  totalExperimentInput.value = config.safety.totalExperimentSeconds;
  updateSafetyPill();
  updateSafetyTripUi();
}
function updateSafetyPill() {
  if (config.safety.maxIntensity <= 0) {
    safetyMaxPill.textContent = "0 = 실제 출력 차단됨";
    safetyMaxPill.className = "pill warn";
  } else {
    safetyMaxPill.textContent = `실제 출력 허용 (최대 ${config.safety.maxIntensity})`;
    safetyMaxPill.className = "pill ok";
  }
}
function initPresetLabels() {
  // presetLightVal/presetHalfVal/presetStrongVal은 예전 거울 모드 카드(프리셋
  // 버튼)에 있었는데 그 카드가 제거되어 더 이상 없음 -- successHoldSecText만 남음.
  document.getElementById("successHoldSecText").textContent = config.control.successHoldSeconds;
}

// ============================================================================
// Supabase 참가자 프로필 (캘리브레이션 + 개인화, 참가자 ID로 구분)
// ============================================================================
// 테이블: calibration_profiles(participant_id text pk, calibration jsonb,
// personalization jsonb, updated_at timestamptz). RLS로 anon 읽기/쓰기 허용.
// localStorage는 그대로 "이 브라우저의 캐시"로 계속 쓰고, Supabase는 참가자가
// 다른 컴퓨터/브라우저로 옮겨도 이어서 쓸 수 있게 하는 별도 저장소로 둔다.

function loadStoredParticipantId() {
  try {
    return localStorage.getItem("emsWebParticipantId") || "";
  } catch (e) {
    return "";
  }
}
function saveStoredParticipantId(id) {
  try {
    localStorage.setItem("emsWebParticipantId", id);
  } catch (e) { /* ignore */ }
}

async function checkSupabaseConnection() {
  try {
    const { error } = await supabase.from("calibration_profiles").select("participant_id").limit(1);
    if (error) throw error;
    supabaseStatusBadge.textContent = "✅ 연결됨";
    supabaseStatusBadge.style.color = "var(--accent-2)";
    supabaseStatusBadge.style.background = "rgba(110,231,183,0.12)";
  } catch (err) {
    supabaseStatusBadge.textContent = "⚠ 연결 실패 (테이블/정책 확인 필요)";
    supabaseStatusBadge.style.color = "var(--danger)";
    supabaseStatusBadge.style.background = "rgba(255,107,107,0.12)";
    logControl("⚠ Supabase 연결 실패: " + (err.message || err));
  }
}

async function loadProfileFromSupabase() {
  const id = participantIdInput.value.trim();
  if (!id) {
    showToast("참가자 ID를 입력하세요", "warn");
    return;
  }
  profileStatusText.textContent = "불러오는 중...";
  try {
    const { data, error } = await supabase
      .from("calibration_profiles")
      .select("calibration, personalization, updated_at")
      .eq("participant_id", id)
      .maybeSingle();
    if (error) throw error;

    saveStoredParticipantId(id);

    if (!data) {
      profileStatusText.textContent = `"${id}"로 저장된 데이터가 없습니다 (새 참가자 -- 캘리브레이션 후 "저장"을 눌러주세요).`;
      showToast(`"${id}"는 새 참가자입니다`, "warn");
      return;
    }

    if (data.calibration && data.calibration.flat && data.calibration.bent) {
      calibration = data.calibration;
      saveCalibration();
    }
    if (data.personalization && Array.isArray(data.personalization.points)) {
      personalization = data.personalization;
      savePersonalization();
    }

    buildTable();
    updateFingerTable();
    drawPersonalizationGraph();
    updatePersonalizationSummary();

    profileStatusText.textContent = `"${id}" 프로필을 불러왔습니다 (마지막 저장: ${new Date(data.updated_at).toLocaleString()})`;
    logControl(`☁ Supabase에서 "${id}" 프로필 불러옴`);
    showToast(`✅ "${id}" 프로필 불러옴`, "ok");
  } catch (err) {
    profileStatusText.textContent = "불러오기 실패: " + (err.message || err);
    showToast("❌ 프로필 불러오기 실패: " + (err.message || err), "bad", 4000);
  }
}

async function saveProfileToSupabase() {
  const id = participantIdInput.value.trim();
  if (!id) {
    showToast("참가자 ID를 입력하세요", "warn");
    return;
  }
  profileStatusText.textContent = "저장하는 중...";
  try {
    const { error } = await supabase.from("calibration_profiles").upsert({
      participant_id: id,
      calibration,
      personalization,
      updated_at: new Date().toISOString()
    });
    if (error) throw error;

    saveStoredParticipantId(id);
    profileStatusText.textContent = `"${id}"로 저장했습니다.`;
    logControl(`☁ Supabase에 "${id}" 프로필 저장함`);
    showToast(`✅ "${id}"로 저장됨`, "ok");
  } catch (err) {
    profileStatusText.textContent = "저장 실패: " + (err.message || err);
    showToast("❌ 프로필 저장 실패: " + (err.message || err), "bad", 4000);
  }
}

// ============================================================================
// 카메라 실행 / 정지
// ============================================================================

// "팔 인식" 모드일 때만 필요한 Pose Landmarker -- 손 모델과 별개 모델(약 5.7MB)이라
// 처음부터 같이 불러오지 않고, 실제로 그 모드에 들어갈 때(setAppMode)만 불러온다
// (한 번 불러오면 poseLandmarker에 캐시되어 재사용됨). setAppMode와 startRun 양쪽에서
// 거의 동시에 호출될 수 있어서, 진행 중인 로딩 Promise를 poseLandmarkerPromise에
// 캐시해 중복으로 두 번 불러오는 걸 막는다.
let poseLandmarkerPromise = null;
async function ensurePoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;
  if (!poseLandmarkerPromise) {
    poseLandmarkerPromise = (async () => {
      const filesetResolverPose = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolverPose, {
        baseOptions: { modelAssetPath: "./pose_landmarker_lite.task" },
        runningMode: "VIDEO",
        numPoses: 1
      });
      return poseLandmarker;
    })().catch((err) => {
      poseLandmarkerPromise = null; // 실패하면 다음 시도 때 다시 불러올 수 있게 캐시 해제
      throw err;
    });
  }
  return poseLandmarkerPromise;
}

async function startRun() {
  runBtn.disabled = true;
  runBtn.textContent = "카메라 준비 중...";
  try {
    if (!handLandmarker) {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
        baseOptions: { modelAssetPath: "./hand_landmarker.task" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.7,
        minTrackingConfidence: 0.7
      });
    }
    if (appMode === "arm") await ensurePoseLandmarker().catch(() => {}); // 실패해도 카메라 자체는 켜지게 (아래 renderLoop가 poseLandmarker null이면 알아서 건너뜀)

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    isRunning = true;
    experimentStart = experimentStart ?? performance.now();
    runBtn.textContent = "정지";
    runBtn.classList.add("running");
    calFlatBtn.disabled = false;
    calBentBtn.disabled = false;
    armCalFlatBtn.disabled = false;
    armCalBentBtn.disabled = false;

    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error(err);
    alert(
      "카메라 또는 모델을 불러오지 못했습니다.\n\n" + err.message +
      "\n\n※ 이 페이지는 로컬 웹서버(예: python -m http.server)로 열어야 합니다."
    );
    runBtn.textContent = "실행 시작";
  } finally {
    runBtn.disabled = false;
  }
}

function stopRun() {
  isRunning = false;
  sampling = null;
  capturing = null;
  resetStartControlButton(); // 카메라를 정지하면 제어도 확실히 같이 정지 상태로
  if (liveMirrorActive) stopLiveMirror(); // 카메라가 꺼지면 실시간 연동도 같이 정지

  const stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setStatusBadge(statusBadgeSource, false, "source");
  setStatusBadge(statusBadgeActual, false, "actual");

  runBtn.textContent = "실행 시작";
  runBtn.classList.remove("running");
  calFlatBtn.disabled = true;
  calBentBtn.disabled = true;
  armCalFlatBtn.disabled = true;
  armCalBentBtn.disabled = true;
}

// ============================================================================
// 메인 루프
// ============================================================================
// MediaPipe의 handedness 판정은 "미러링된(셀카) 입력"을 가정한다. 그래서
// 원본 영상을 그대로 인식시키지 않고, 매 프레임 캔버스에 좌우로 뒤집어
// 그린 뒤 그 캔버스를 그대로 인식시킨다 (hand_camera.py의 cv2.flip(frame,1)
// 과 동일한 효과). 이렇게 하면 랜드마크 좌표와 화면에 그려지는 좌표가
// 항상 같은 공간에 있어서, 뱃지/텍스트가 반전되어 보이는 종류의 버그가
// 구조적으로 생기지 않는다.

function renderLoop() {
  if (!isRunning) return;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    const result = handLandmarker.detectForVideo(canvas, performance.now());
    // 팔 인식 모드일 때만 Pose 모델도 같이 돌린다 (다른 모드에서는 손 인식 하나로
    // 충분하고, 매 프레임 모델을 하나 더 돌리는 연산 비용을 아끼기 위함).
    const poseResult = appMode === "arm" && poseLandmarker
      ? poseLandmarker.detectForVideo(canvas, performance.now())
      : null;
    processResult(result, poseResult);
  }
  requestAnimationFrame(renderLoop);
}

function getRoleForHandedness(categoryName) {
  // 실측 결과 이 환경에서는 기본적으로 좌우가 뒤집혀 인식되어, 기본값
  // 자체를 반전시켨다 (체크박스를 매번 누르지 않아도 되도록). 체크박스는
  // 이제 "한 번 더 뒤집기" 용도로 남겨둔다 -- 나중에 카메라/설정이 바뀌어
  // 다시 반대로 나오면 체크박스로 되돌릴 수 있다.
  let trueSide = categoryName === "Left" ? "Right" : "Left";
  if (invertHandedness) trueSide = trueSide === "Left" ? "Right" : "Left";
  return trueSide === "Left" ? "source" : "actual";
}

function processResult(result, poseResult) {
  const now = performance.now();
  detectedThisFrame.source = false;
  detectedThisFrame.actual = false;

  const numHands = result.landmarks ? result.landmarks.length : 0;
  for (let i = 0; i < numHands; i++) {
    const categoryName = result.handedness?.[i]?.[0]?.categoryName || "Left";
    const role = getRoleForHandedness(categoryName);
    if (detectedThisFrame[role]) continue; // 같은 역할이 이미 처리됨 (오분류 방지)
    detectedThisFrame[role] = true;

    const imageLandmarks = result.landmarks[i];
    const worldLandmarks = result.worldLandmarks[i];
    drawSkeleton(imageLandmarks, role);
    drawRoleLabel(imageLandmarks[0], role);

    const rawBends = {};
    for (const finger of FINGER_KEYS) {
      rawBends[finger] = calculateFingerBend(worldLandmarks, FINGERS[finger]);
      latestSmoothed[role][finger] = filters[role][finger].push(rawBends[finger]);
    }
    // 엄지 -- 표시 전용 (제어에는 관여하지 않음, 위 THUMB_JOINTS 주석 참고)
    rawBends.Thumb = calculateFingerBend(worldLandmarks, THUMB_JOINTS);
    latestSmoothed[role].Thumb = filters[role].Thumb.push(rawBends.Thumb);

    if (sampling && role === "actual") {
      for (const finger of DISPLAY_FINGER_KEYS) sampling.sums[finger] += rawBends[finger];
      sampling.count += 1;
      updateSamplingUI();
      if (sampling.count >= SAMPLE_TARGET) finishSampling();
    }
    if (capturing && role === "source") {
      for (const finger of DISPLAY_FINGER_KEYS) capturing.sums[finger] += rawBends[finger];
      capturing.count += 1;
      updateCaptureUI();
      if (capturing.count >= CAPTURE_TARGET) finishCapture();
    }
  }

  if (!detectedThisFrame.actual) {
    consecutiveMissActual += 1;
    if (consecutiveMissActual >= HAND_LOSS_FRAMES_THRESHOLD) handLostSustained = true;
  } else {
    consecutiveMissActual = 0;
    handLostSustained = false;
  }

  if (sampling && now - sampling.startedAt > SAMPLE_TIMEOUT_MS && sampling.count < SAMPLE_TARGET) {
    abortSampling("측정 시간이 초과되었습니다. 오른손이 잘 보이도록 하고 다시 시도해주세요.");
  }
  if (capturing && now - capturing.startedAt > CAPTURE_TIMEOUT_MS && capturing.count < CAPTURE_TARGET) {
    abortCapture("캡처 시간이 초과되었습니다. 왼손이 잘 보이도록 하고 다시 시도해주세요.");
  }

  // ---- 팔 인식 (Pose Landmarker) -- "팔 인식" 모드일 때만 poseResult가 들어온다 ----
  armDetectedThisFrame.source = false;
  armDetectedThisFrame.actual = false;
  if (poseResult && poseResult.landmarks && poseResult.landmarks.length > 0) {
    processArmResult(poseResult);
  }
  if (!armDetectedThisFrame.actual) {
    consecutiveMissArmActual += 1;
    if (consecutiveMissArmActual >= HAND_LOSS_FRAMES_THRESHOLD) armLostSustained = true;
  } else {
    consecutiveMissArmActual = 0;
    armLostSustained = false;
  }
  if (armSampling && now - armSampling.startedAt > SAMPLE_TIMEOUT_MS && armSampling.count < SAMPLE_TARGET) {
    abortArmSampling("측정 시간이 초과되었습니다. 오른팔이 잘 보이도록 하고 다시 시도해주세요.");
  }

  // 팔 인식 모드에서는 손가락 표/평균 대신 팔 값을 쓴다 (아래 폐루프 제어부터는
  // "팔 인식" 모드도 손가락 모드와 완전히 동일한 코드 경로 -- targetPercent/
  // currentAverage 두 전역값만 팔 기준으로 채워주면 updateController() 이하
  // 로직이 그대로 재사용된다).
  // 손가락 %는 모드와 무관하게 항상 갱신한다 -- 팔 인식 모드에서도 "참고용
  // 손 굽힘" 표시가 필요해졌고, 원래도 실시간 왼손 연동 계산이 여기 의존한다.
  updateFingerTable();
  updateSourcePercent();
  const handAverage = computeAverage(latestPercent.actual);

  if (appMode === "arm") {
    setStatusBadge(statusBadgeSource, armDetectedThisFrame.source, "source");
    setStatusBadge(statusBadgeActual, armDetectedThisFrame.actual, "actual");
    currentAverage = armLatestPercent.actual;
    // 팔 인식 모드는 별도 "시작/정지" 버튼 없이 모드에 들어와 있는 동안 항상
    // 왼팔을 실시간으로 목표에 반영한다 (거울 모드의 "실시간 왼손 연동"과 동일한
    // 개념을, 이 모드에서는 토글 없이 기본 동작으로 둔 것).
    targetPercent = armLatestPercent.source;

    // 손 굽힘은 참고용으로 같이 보여준다 -- 아직 채널을 구동하는 폐루프 목표는
    // 팔꿈치 기준 그대로다 (손+팔꿈치를 동시에 채널로 제어하려면 2채널로는
    // 부족해서 순차 단계식 설계가 별도로 필요함 -- 다음 작업에서 진행 예정).
    const handTarget = latestPercent.source ? computeAverage(latestPercent.source) : null;
    armHandTargetDisplay.textContent = handTarget === null ? "-" : `${handTarget.toFixed(0)}%`;
    armHandActualDisplay.textContent = handAverage === null ? "-" : `${handAverage.toFixed(0)}%`;
    if (handTarget === null || handAverage === null) {
      armHandErrorDisplay.textContent = "-";
      armHandErrorDisplay.style.color = "var(--text-dim)";
    } else {
      const handError = handTarget - handAverage;
      armHandErrorDisplay.textContent = `${handError > 0 ? "+" : ""}${handError.toFixed(0)}%`;
      armHandErrorDisplay.style.color = Math.abs(handError) <= config.control.tolerancePercent ? "var(--accent-2)" : "var(--warn)";
    }
  } else {
    setStatusBadge(statusBadgeSource, detectedThisFrame.source, "source");
    setStatusBadge(statusBadgeActual, detectedThisFrame.actual, "actual");
    currentAverage = handAverage;
  }

  // "▶ 제어 시작"을 누르기 전까지는 캘리브레이션/캡처가 다 끝나 있어도 절대
  // 하드웨어로 아무것도 나가지 않는다. 예전에는 캘리브레이션이 완료되는
  // 순간 이미 잡혀있던 목표와 바로 비교가 시작돼서 사용자가 누른 것도 없는데
  // 바로 자극이 나가는 문제가 있었다.
  if (sweep) {
    // 개인화 캘리브레이션(자극값 스윕) 진행 중 -- 목표 추종 폐루프와는 완전히
    // 별개의 열린 루프라, 이 tick 동안은 그 로직을 건너뛰고 스윕만 진행한다.
    // (스윕을 시작할 때 controlEnabled를 이미 강제로 꺼뒀으므로 아래 하드웨어
    // 전송 스로틀 블록과 충돌하지 않는다.)
    const runtime = runtimeCheck(handLostSustained);
    if (!runtime.ok && !safetyTripped) {
      triggerEmergencyStop(runtime.reason);
      abortSweep(runtime.reason);
    } else {
      tickSweep(now);
    }
  } else if (!controlEnabled) {
    controlState = "STANDBY";
    lastError = null;
    controllerIntensity = 0;
    successSince = null;
  } else {
    // 팔 인식 모드는 위에서 이미 armLatestPercent 기준으로 armLostSustained를
    // 갱신해뒀으므로, 여기서는 어느 쪽 "놓침" 플래그를 볼지만 모드에 따라 고른다.
    const lostSustained = appMode === "arm" ? armLostSustained : handLostSustained;

    // ---- 안전 확인 (컨트롤러 계산보다 먼저: 트립되면 이번 tick에서 즉시 0으로) ----
    const runtime = runtimeCheck(lostSustained);
    if (!runtime.ok && !safetyTripped) {
      triggerEmergencyStop(runtime.reason);
    }

    // 행동 보조 모드의 실시간 왼손 연동 -- 매 프레임 목표를 왼손의 지금 값으로
    // 갱신한다 (거울 모드 캡처처럼 한 번 얼리지 않음). 컨트롤러 상태는 안 건드리고
    // targetPercent/targetPerFinger만 바꾸므로, 아래 updateController()가 평소처럼
    // 새 목표에 대해 오차/허용범위/유지 로직을 그대로 적용한다. (팔 인식 모드는
    // targetPercent를 위에서 이미 매 프레임 직접 채웠으므로 여기선 손 전용 로직만.)
    if (liveMirrorActive) {
      updateLiveMirrorTarget();
    }

    // ---- 폐루프 제어 ----
    const prevState = controlState;
    updateController(lostSustained ? null : currentAverage, now);
    if (controlState !== prevState) logControl(`상태 변경: ${STATE_LABELS[prevState] || prevState} → ${STATE_LABELS[controlState] || controlState}`);

    if (controllerIntensity > 0) notifyStimStarted();
    else notifyStimStopped();
  }

  // 카메라는 초당 수십 프레임이지만, 하드웨어로는 control_period_ms(기본 6초)에
  // 한 번씩만 보낸다. 매 프레임 보내면 시리얼 명령이 계속 쌓여서(큐 적체) 실제
  // 전송이 몇 초씩 밀리는 심각한 문제가 있었다 (실측으로 발견됨).
  //
  // LOCKED(자세 유지) 상태일 때는 예외 -- HOLD_PULSE_ON_MS(800ms)짜리 짧은
  // 펄스를 HOLD_PULSE_PERIOD_MS(4초)마다 주는데, control_period_ms(6초)
  // 간격으로만 보내면 펄스의 켜짐/꺼짐 구간을 자주 놓쳐서 실제로는 전기가
  // 거의 안 나가게 된다. 그래서 LOCKED일 때는 훨씬 촘촘한 간격으로 보낸다.
  const hardwareSendIntervalMs = holdLocked ? 400 : getControlPeriodMs();
  if (controlEnabled && liveModeRequested && now - lastHardwareSendTime >= hardwareSendIntervalMs) {
    lastHardwareSendTime = now;
    driveHardwareIfNeeded(activeChannel, controllerIntensity).catch((err) => logControl("하드웨어 전송 오류: " + err.message));
  }

  // ---- 기록 ----
  logRows.push({
    t: now,
    target: targetPercent,
    current: currentAverage,
    intensity: controllerIntensity,
    error: lastError,
    state: controlState,
    success: controlState === "SUCCESS",
    handDetected: detectedThisFrame.actual
  });
  if (logRows.length > 5000) logRows = logRows.slice(-3000);
  updateLogCount();

  updateCompareDisplay();
  stimValueDisplay.textContent = Math.round(controllerIntensity);
  updateActiveChannelPill();
  updateHardwareSentHint();
  controlStateText.textContent = STATE_LABELS[controlState] || controlState;
}

function updateActiveChannelPill() {
  if (activeChannel === 1) {
    activeChannelPill.textContent = "채널1 (구부림)";
    activeChannelPill.className = "pill";
  } else {
    activeChannelPill.textContent = "채널2 (폄)";
    activeChannelPill.className = "pill warn";
  }
}

// "EMS 제어값"은 안전 상한과 무관한 내부 계산값이라, 실제로 하드웨어에 나가는
// (안전 최대값으로 잘린) 값과 다를 수 있다. 그걸 혼동해서 "화면엔 100인데 왜
// 전기가 안 오지?"가 되는 걸 막기 위해 실제 전송값을 바로 옆에 같이 보여준다.
function updateHardwareSentHint() {
  const sent = clampHardware(controllerIntensity);
  const channelLabel = activeChannel === 1 ? "채널1" : "채널2";
  if (!liveModeRequested) {
    hardwareSentHint.textContent = "실제 전송값: - (라이브 모드 아님, 시뮬레이션만)";
    hardwareSentHint.style.color = "var(--text-dim)";
  } else if (sent < controllerIntensity) {
    hardwareSentHint.textContent = `실제 전송값(${channelLabel}): ${sent} (안전 최대값 ${config.safety.maxIntensity}로 잘림!)`;
    hardwareSentHint.style.color = "var(--warn)";
  } else {
    hardwareSentHint.textContent = `실제 전송값(${channelLabel}): ${sent}`;
    hardwareSentHint.style.color = "var(--accent-2)";
  }
}

function drawSkeleton(landmarks, role) {
  const w = canvas.width, h = canvas.height;
  const color = ROLE_COLORS[role];
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  for (const [s, e] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[s].x * w, landmarks[s].y * h);
    ctx.lineTo(landmarks[e].x * w, landmarks[e].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = color;
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawArmSkeleton(shoulderPt, elbowPt, wristPt, role) {
  const w = canvas.width, h = canvas.height;
  const color = ROLE_COLORS[role];
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(shoulderPt.x * w, shoulderPt.y * h);
  ctx.lineTo(elbowPt.x * w, elbowPt.y * h);
  ctx.lineTo(wristPt.x * w, wristPt.y * h);
  ctx.stroke();
  ctx.fillStyle = color;
  for (const pt of [shoulderPt, elbowPt, wristPt]) {
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ============================================================================
// 팔 인식 (Pose Landmarker 결과 처리) -- "팔 인식" 모드 전용
// ============================================================================
// Pose Landmarker는 손처럼 손별로 배열이 나뉘어 나오지 않고, 한 사람의 몸 전체
// (양팔 포함) 랜드마크가 한 번에 나온다. 그래서 손 인식의 "categoryName으로
// source/actual 판정" 대신, 화면(미러링된 캔버스)에 더 왼쪽에 보이는 팔을
// source(TARGET), 더 오른쪽에 보이는 팔을 actual(EMS)로 판정한다 -- 실제 거울을
// 보는 것과 같은 방향 감각이라 더 직관적이기도 하다.
function processArmResult(poseResult) {
  const imageLandmarks = poseResult.landmarks[0];
  const worldLandmarks = poseResult.worldLandmarks[0];
  if (!imageLandmarks || !worldLandmarks) return;

  const candidates = [POSE_ARM_A, POSE_ARM_B]
    .map((idx) => ({
      idx,
      shoulderPt: imageLandmarks[idx.shoulder],
      elbowPt: imageLandmarks[idx.elbow],
      wristPt: imageLandmarks[idx.wrist]
    }))
    .filter((a) =>
      a.shoulderPt && a.elbowPt && a.wristPt &&
      (a.shoulderPt.visibility ?? 1) >= POSE_VISIBILITY_THRESHOLD &&
      (a.elbowPt.visibility ?? 1) >= POSE_VISIBILITY_THRESHOLD &&
      (a.wristPt.visibility ?? 1) >= POSE_VISIBILITY_THRESHOLD
    );

  if (candidates.length === 0) return;

  // 두 팔이 다 보이면 x좌표로 왼쪽/오른쪽을 정렬해서 나눠주고, 한쪽만 보이면
  // 화면 가운데(0.5) 기준으로 어느 쪽인지만 판단한다.
  candidates.sort((a, b) => a.elbowPt.x - b.elbowPt.x);
  const roles = candidates.length === 2
    ? ["source", "actual"]
    : [candidates[0].elbowPt.x < 0.5 ? "source" : "actual"];

  candidates.forEach((arm, i) => {
    const role = roles[i];
    if (!role || armDetectedThisFrame[role]) return; // 이미 이번 프레임에 그 역할이 처리됨
    armDetectedThisFrame[role] = true;

    drawArmSkeleton(arm.shoulderPt, arm.elbowPt, arm.wristPt, role);

    const rawBend = calculateArmBend(worldLandmarks, arm.idx.shoulder, arm.idx.elbow, arm.idx.wrist);
    armLatestSmoothed[role] = armFilters[role].push(rawBend);
    armLatestPercent[role] = percentFor("Arm", armLatestSmoothed[role]);

    if (armSampling && role === "actual") {
      armSampling.sum += rawBend;
      armSampling.count += 1;
      updateArmSamplingUI();
      if (armSampling.count >= SAMPLE_TARGET) finishArmSampling();
    }
  });
}

// ---- 팔 초기값(펴짐/구부림) 측정 -- 손가락 캘리브레이션과 같은 방식, 값 하나뿐 ----
function startArmSampling(mode) {
  if (!isRunning || armSampling) return;
  armSampling = { mode, sum: 0, count: 0, startedAt: performance.now() };
  const label = mode === "flat" ? "펴짐" : "구부림";
  logControl(`오른팔 ${label} 초기값 측정 시작 -- 그대로 유지하세요`);
  showToast(`📏 오른팔 ${label} 초기값 측정 중... 팔을 그대로 유지하세요`, "ok", 2000);
}
function updateArmSamplingUI() {
  if (!armSampling) return;
  logControlProgress(`팔 초기값 측정 중... ${armSampling.count}/${SAMPLE_TARGET}`);
}
function finishArmSampling() {
  const { mode, sum, count } = armSampling;
  calibration[mode].Arm = sum / count;
  saveCalibration();
  armSampling = null;
  const label = mode === "flat" ? "펴짐" : "구부림";
  logControl(`팔 ${label} 초기값 측정 완료`);
  showToast(`✅ 오른팔 ${label} 초기값 측정 완료`, "ok");
}
function abortArmSampling(message) {
  armSampling = null;
  showToast(`❌ 팔 초기값 측정 실패: ${message}`, "bad", 4000);
  alert(message);
}

function drawRoleLabel(wrist, role) {
  const x = wrist.x * canvas.width;
  const y = wrist.y * canvas.height + 20;
  ctx.fillStyle = ROLE_COLORS[role];
  ctx.font = "bold 14px 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(ROLE_LABELS[role], x, y);
}

function setStatusBadge(el, ok, role) {
  el.textContent = `${ROLE_LABELS[role]}: ${ok ? "인식됨" : "인식 안됨"}`;
  el.classList.toggle("ok", ok);
  el.classList.toggle("bad", !ok);
}

// ============================================================================
// 캘리브레이션 (오른손 · ACTUAL 기준, 두 손 공용으로 사용)
// ============================================================================

function startSampling(mode) {
  if (!isRunning || sampling) return;
  const sums = {};
  for (const finger of DISPLAY_FINGER_KEYS) sums[finger] = 0;
  sampling = { mode, sums, count: 0, startedAt: performance.now() };
  const label = mode === "flat" ? "펴짐" : "구부림";
  logControl(`오른손 ${label} 초기값 측정 시작 -- 그대로 유지하세요`);
  showToast(`📏 오른손 ${label} 초기값 측정 중... 손을 그대로 유지하세요`, "ok", 2000);
}
function updateSamplingUI() {
  if (!sampling) return;
  logControlProgress(`초기값 측정 중... ${sampling.count}/${SAMPLE_TARGET}`);
}
function finishSampling() {
  const { mode, sums, count } = sampling;
  for (const finger of DISPLAY_FINGER_KEYS) calibration[mode][finger] = sums[finger] / count;
  saveCalibration();
  sampling = null;
  const label = mode === "flat" ? "펴짐" : "구부림";
  logControl(`${label} 초기값 측정 완료`);
  showToast(`✅ 오른손 ${label} 초기값 측정 완료`, "ok");
}
function abortSampling(message) {
  sampling = null;
  showToast(`❌ 초기값 측정 실패: ${message}`, "bad", 4000);
  alert(message);
}

// ============================================================================
// 왼손 목표 동작 캡처 (스냅샷)
// ============================================================================

function startCapture() {
  if (!isRunning || capturing) return;
  const sums = {};
  for (const finger of DISPLAY_FINGER_KEYS) sums[finger] = 0;
  capturing = { sums, count: 0, startedAt: performance.now() };
  captureTitle.textContent = "왼손 동작을 그대로 유지하세요...";
  captureOverlay.classList.add("active");
  updateCaptureUI();
}
function updateCaptureUI() {
  if (!capturing) return;
  captureProgress.textContent = `${capturing.count} / ${CAPTURE_TARGET}`;
}
function finishCapture() {
  const { sums, count } = capturing;
  const rawAverages = {};
  for (const finger of DISPLAY_FINGER_KEYS) rawAverages[finger] = sums[finger] / count;

  for (const finger of DISPLAY_FINGER_KEYS) {
    targetPerFinger[finger] = percentFor(finger, rawAverages[finger]);
  }
  capturing = null;
  captureOverlay.classList.remove("active");

  const avg = computeAverage(targetPerFinger);
  if (avg === null) {
    logControl("⚠ 왼손 캡처는 완료했지만, 오른손 초기값(펴짐/구부림) 보정이 안 되어 %로 환산할 수 없습니다. 초기값 측정을 먼저 해주세요.");
    showToast("⚠ 캡처는 됐지만 초기값(펴짐/구부림) 보정이 먼저 필요합니다", "warn", 4000);
  } else if (isInCooldown()) {
    showToast(`⏳ 쿨다운 중입니다 (${cooldownRemainingSeconds().toFixed(1)}s 남음) — 잠시 후 다시 캡처해주세요`, "warn", 3500);
  } else {
    setTarget(avg, suggestInitialIntensity(avg));
    commandMessage.textContent = `왼손 캡처 완료 → 목표 평균 ${avg.toFixed(0)}%`;
    logControl(`📸 왼손 캡처 완료 → 목표 평균 ${avg.toFixed(0)}%`);
    showToast(`✅ 목표 동작 설정 완료 (왼손 캡처): 평균 ${avg.toFixed(0)}%`, "ok");
  }
}
function abortCapture(message) {
  capturing = null;
  captureOverlay.classList.remove("active");
  alert(message);
}

// ============================================================================
// 손가락별 표 / 퍼센트 계산
// ============================================================================

function buildTable() {
  tableBody.innerHTML = "";
  for (const finger of DISPLAY_FINGER_KEYS) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${FINGER_LABELS[finger]}</td>
      <td class="num" id="target-${finger}">-</td>
      <td class="num" id="actual-${finger}">-</td>
      <td class="num err-cell" id="err-${finger}">-</td>
    `;
    tableBody.appendChild(tr);
  }
}

// 왼손(SOURCE) 실시간 미리보기 표 -- 캡처하기 전에도 지금 왼손이 손가락별로
// 몇 %인지 계속 보여준다 (캡처된 target과는 별개, 그냥 지금 이 순간의 값).
// (예전엔 여기서 왼손 실시간 미리보기 표(DOM)도 같이 그렸는데, 그 표가 있던
// 카드가 제거되면서 표 그리는 부분은 뺐다. 하지만 이 계산 자체(latestPercent.source
// 채우기)는 "실시간 왼손 연동"(updateLiveMirrorTarget)이 매 프레임 그대로
// 의존하고 있어서 반드시 계속 돌아야 한다 -- 표시만 없어진 것이지 계산까지
// 없어지면 안 된다.)
function updateSourcePercent() {
  for (const finger of DISPLAY_FINGER_KEYS) {
    latestPercent.source[finger] = percentFor(finger, latestSmoothed.source[finger]);
  }
}

// 보정값은 손가락별로 저장되어 있으므로, 손가락별로 환산한다.
function percentFor(finger, smoothDeg) {
  const flat = calibration.flat[finger];
  const bent = calibration.bent[finger];
  if (smoothDeg === undefined || flat === undefined || bent === undefined || Math.abs(bent - flat) < MIN_CAL_GAP_DEG) {
    return null;
  }
  return Math.min(100, Math.max(0, ((smoothDeg - flat) / (bent - flat)) * 100));
}

function updateFingerTable() {
  for (const finger of DISPLAY_FINGER_KEYS) {
    latestPercent.actual[finger] = percentFor(finger, latestSmoothed.actual[finger]);
  }

  for (const finger of DISPLAY_FINGER_KEYS) {
    const t = targetPerFinger[finger];
    const a = latestPercent.actual[finger];
    document.getElementById(`target-${finger}`).textContent = t === null || t === undefined ? "-" : `${t.toFixed(0)}%`;
    document.getElementById(`actual-${finger}`).textContent = a === null || a === undefined ? "-" : `${a.toFixed(0)}%`;
    const errCell = document.getElementById(`err-${finger}`);
    if (t === null || t === undefined || a === null || a === undefined) {
      errCell.textContent = "-";
      errCell.style.color = "var(--text-dim)";
    } else {
      const e = t - a;
      errCell.textContent = `${e > 0 ? "+" : ""}${e.toFixed(0)}%`;
      errCell.style.color = Math.abs(e) <= config.control.tolerancePercent ? "var(--accent-2)" : "var(--warn)";
    }
  }

  const targetAvg = computeAverage(targetPerFinger);
  const actualAvg = computeAverage(latestPercent.actual);
  document.getElementById("avg-target").textContent = targetAvg === null ? "-" : `${targetAvg.toFixed(0)}%`;
  document.getElementById("avg-actual").textContent = actualAvg === null ? "-" : `${actualAvg.toFixed(0)}%`;
  const avgErrCell = document.getElementById("avg-error");
  if (targetAvg === null || actualAvg === null) {
    avgErrCell.textContent = "-";
    avgErrCell.style.color = "var(--text-dim)";
  } else {
    const e = targetAvg - actualAvg;
    avgErrCell.textContent = `${e > 0 ? "+" : ""}${e.toFixed(0)}%`;
    avgErrCell.style.color = Math.abs(e) <= config.control.tolerancePercent ? "var(--accent-2)" : "var(--warn)";
  }
}

function computeAverage(perFingerObj) {
  const values = FINGER_KEYS.map((f) => perFingerObj[f]);
  if (values.some((v) => v === null || v === undefined)) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function isCalibrated() {
  return FINGER_KEYS.every((f) => {
    const flat = calibration.flat[f], bent = calibration.bent[f];
    return flat !== undefined && bent !== undefined && Math.abs(bent - flat) >= MIN_CAL_GAP_DEG;
  });
}

// ============================================================================
// 명령 처리 (자연어 명령 / 프리셋은 4손가락에 동일한 목표를 지정)
// ============================================================================

async function submitCommand() {
  const text = commandInput.value.trim();
  if (!text) return;
  commandInput.value = "";
  commandRunBtn.disabled = true;

  let result = null;
  if (aiConfig.enabled && aiConfig.apiKey) {
    commandMessage.textContent = "🤖 AI가 명령을 해석하는 중...";
    result = await interpretCommandWithAI(text);
    // null이면 (호출 실패/키 없음/파싱 실패 등) 조용히 규칙 기반으로 대체 --
    // AI가 꺼져있거나 문제가 생겼다고 해서 명령 자체가 무시되면 안 된다.
  }
  if (result === null) {
    result = interpretCommand(text);
  }

  commandRunBtn.disabled = false;
  commandMessage.textContent = result.message;
  logControl(`명령: "${text}" → ${result.message}`);

  if (result.action === "set_target") {
    applyUniformTarget(result.targetPercent);
    showToast(`✅ 목표 동작 설정 완료: ${result.targetPercent.toFixed(0)}%`, "ok");
  } else if (result.action === "stop") {
    forceZeroController("명령: 정지");
    if (liveModeRequested && serialLink) serialLink.stopAll().catch(() => {});
    armedCh1 = false;
    armedCh2 = false;
    lastDrivenChannel = null;
    showToast("⏹ 정지 명령이 적용되었습니다", "warn");
  } else if (result.action === "error") {
    // 규칙 기반이든 AI든 이해하지 못한 경우 -- 조용히 넘기지 않고 눈에 띄게 알려준다.
    showToast(`❓ 명령을 인식하지 못했습니다: ${result.message}`, "bad", 4000);
  }
}

// 실제 Claude API를 브라우저에서 직접 호출한다 (백엔드 서버 없음). 실패하면
// 반드시 null을 반환해서 호출자가 규칙 기반으로 안전하게 대체하도록 한다 --
// 이 함수가 던지는 예외 때문에 명령 자체가 먹통이 되는 일은 없어야 한다.
async function interpretCommandWithAI(text) {
  const contextLine =
    targetPercent !== null
      ? `현재 목표값은 ${targetPercent.toFixed(0)}% 입니다 (상대적인 표현, 예: "조금 더 세게"의 기준이 됩니다).`
      : "현재 설정된 목표값은 없습니다.";

  const systemPrompt = `당신은 손가락 굽힘 재활 기기의 자연어 명령을 해석하는 파서입니다.
이 기기는 목표 굽힘 정도(0=완전히 편 상태, 100=완전히 주먹 쥔 상태)를 정하고, EMS 자극으로 손가락을 그 목표까지 구부립니다.
사용자의 한국어 문장을 보고 반드시 아래 셋 중 하나의 JSON 객체만 출력하세요. 설명, 코드블록, 다른 텍스트는 절대 포함하지 마세요.

1) 목표 굽힘 정도를 지정/변경하는 문장: {"action":"set_target","target_percent":<0-100 사이 정수>,"message":"<한국어로 무엇을 했는지 한 줄 설명>"}
2) 정지/멈춤을 요청하는 문장: {"action":"stop","message":"<한국어 설명>"}
3) 손동작 목표와 전혀 관련 없는 문장(날씨, 잡담 등): {"action":"error","message":"<왜 처리할 수 없는지 한국어로 설명>"}

**중요**: 숫자가 직접 안 나와도, 어떤 물건을 쥐거나 잡는 비유("OO를 쥘 정도로", "OO 잡을 정도로 구부려")가
나오면 그 물건의 대략적인 크기/모양을 상식적으로 떠올려서, 손가락이 그걸 감싸 쥐려면 어느 정도
구부러져야 하는지 **스스로 추론**해서 반드시 target_percent 숫자를 채워 set_target으로 응답하세요.
예시 목록에 없는 물건(마우스, 햄스터, 사과, 볼펜 등 무엇이든)이 나와도 절대 예외로 두지 말고,
크기가 작을수록/둥글수록 더 많이 구부러져야 하고, 크기가 크거나 평평할수록 덜 구부러진다는
식으로 스스로 판단하세요. **사용자에게 숫자를 알려달라고 절대 되묻지 마세요** (error로 응답하지
마세요) -- 확신이 없어도 최선의 추정치를 내는 것이 당신의 핵심 역할입니다.
오직 손동작 목표와 전혀 무관한 문장(날씨, 잡담 등)일 때만 error를 쓰세요.

${contextLine}

예시:
"손가락을 절반만 굽혀줘" → {"action":"set_target","target_percent":50,"message":"목표 굽힘 50% 설정"}
"아까보다 조금 더 세게" → 현재 목표에서 약 10~15%p 올린 값으로 set_target
"탁구공을 쥘 정도로만 구부려줘" → 작고 둥근 물체를 살짝 감싸는 정도, target_percent 40 근처
"계란 하나를 살짝 쥐는 정도" → target_percent 35 근처
"컴퓨터 마우스를 쥘 정도로 쥐어" → 손바닥에 맞는 중간 크기 물체를 감싸 쥐는 정도, target_percent 45 근처
"햄스터를 잡을 정도로 구부려" → 작고 부드러운 걸 다치지 않게 살짝 감싸는 정도, target_percent 30 근처
(위 숫자는 예시일 뿐, 실제로는 매번 그 물건의 크기를 스스로 떠올려서 판단할 것 -- 절대 되묻지 말 것)
"그만해", "멈춰줘" → stop
"오늘 날씨 어때" → error`;

  // Gemini의 "responseSchema"로 JSON 형식을 강제한다 -- 모델이 설명을 덧붙이거나
  // 코드블록으로 감싸는 것 자체를 막아줘서, 파싱이 훨씬 안정적이다.
  const responseSchema = {
    type: "object",
    properties: {
      action: { type: "string", enum: ["set_target", "stop", "error"] },
      target_percent: { type: "number" },
      message: { type: "string" }
    },
    // target_percent도 필수로 만든다 -- 안 그러면 모델이 action:"set_target"으로
    // 판단하고도 숫자 자체를 빼먹고 응답할 수 있다 (stop/error일 땐 그냥 0을
    // 채우면 되므로 항상 필수로 둬도 무해하다).
    required: ["action", "message", "target_percent"]
  };

  try {
    const model = aiConfig.model || "gemini-3.7-flash";
    const url =
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent` +
      `?key=${encodeURIComponent(aiConfig.apiKey)}`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text }] }],
        systemInstruction: { parts: [{ text: systemPrompt }] },
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema,
          maxOutputTokens: 300
        }
      })
    });

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      logControl(`⚠ AI 호출 실패 (HTTP ${response.status}) → 규칙 기반으로 대체: ${errBody.slice(0, 200)}`);
      return null;
    }

    const data = await response.json();
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
    if (!raw) {
      // 안전 필터에 걸리면 candidates[0].finishReason이 "SAFETY" 등으로 오고
      // parts가 없을 수 있다 -- 이 경우도 조용히 규칙 기반으로 대체한다.
      const finishReason = data.candidates?.[0]?.finishReason;
      logControl(`⚠ AI 응답에 텍스트가 없음(${finishReason || "unknown"}) → 규칙 기반으로 대체`);
      return null;
    }

    // responseSchema를 강제했으니 보통 이미 순수 JSON이지만, 혹시 모를 코드블록도 방어적으로 제거
    const jsonText = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
    const parsed = JSON.parse(jsonText);
    logControl(`🤖 AI 원본 응답: ${jsonText}`); // 뭐가 왔는지 항상 로그에 남겨서 디버깅 쉽게

    if (parsed.action === "set_target") {
      const pct = Number(parsed.target_percent);
      if (!Number.isFinite(pct) || pct < 0 || pct > 100) {
        return { action: "error", message: "AI가 0~100 범위 밖의 값을 반환했습니다." };
      }
      return { action: "set_target", targetPercent: pct, message: parsed.message || `목표 굽힘 ${pct.toFixed(0)}% 설정 (AI)` };
    }
    if (parsed.action === "stop") {
      return { action: "stop", message: parsed.message || "AI가 정지로 해석했습니다." };
    }
    return { action: "error", message: parsed.message || "AI가 명령을 이해하지 못했습니다." };
  } catch (err) {
    logControl(`⚠ AI 명령 해석 오류: ${err.message} → 규칙 기반으로 대체`);
    return null;
  }
}

function applyPresetTarget(percent, name) {
  commandMessage.textContent = `목표 굽힘 ${percent}% 설정 (${name})`;
  logControl(`빠른 실행: ${percent}%`);
  applyUniformTarget(percent);
  showToast(`✅ 목표 동작 설정 완료: ${percent}%`, "ok");
}

function applyUniformTarget(percent) {
  if (isInCooldown()) {
    showToast(`⏳ 쿨다운 중입니다 (${cooldownRemainingSeconds().toFixed(1)}s 남음) — 잠시 후 다시 시도하세요`, "warn", 3500);
    return;
  }
  for (const finger of DISPLAY_FINGER_KEYS) targetPerFinger[finger] = percent;
  setTarget(percent, suggestInitialIntensity(percent));
  // updateCompareDisplay/updateFingerTable은 원래 카메라 루프(processResult)
  // 안에서만 갱신됐다. 그러면 카메라가 안 돌고 있을 때 명령만 테스트하면
  // targetPercent는 내부적으로 바뀌었는데 화면(TARGET 칸)엔 반영이 안 되는
  // 것처럼 보인다. 명령을 적용한 즉시 화면도 갱신한다.
  updateCompareDisplay();
  updateFingerTable();
}

// ============================================================================
// 행동 보조 모드 (Action Assist Mode) -- 거울 모드의 setTarget/targetPerFinger를
// 그대로 재사용한다. 컨트롤러/안전 로직은 전혀 새로 만들지 않았다.
// ============================================================================

function buildActionButtons() {
  actionButtonsRow.innerHTML = "";
  for (const action of ACTIONS) {
    const btn = document.createElement("button");
    btn.className = "preset";
    btn.id = `actionBtn-${action.id}`;
    btn.textContent = action.label;
    btn.addEventListener("click", () => selectAction(action));
    actionButtonsRow.appendChild(btn);
  }
}

function selectAction(action) {
  // 프리셋 행동과 실시간 왼손 연동은 동시에 쓰지 않는다 -- 행동을 고르면 실시간
  // 연동은 자동으로 꺼진다.
  if (liveMirrorActive) stopLiveMirror();

  selectedAction = action;
  selectedActionText.textContent = action.label;

  for (const btn of actionButtonsRow.querySelectorAll("button")) {
    btn.classList.toggle("selected", btn.id === `actionBtn-${action.id}`);
  }

  const hasTarget = action.targetPerFinger || typeof action.targetPercent === "number";
  if (!hasTarget) {
    actionDescriptionText.textContent = `${action.description} -- ⚠ 아직 목표 동작 데이터가 입력되지 않았습니다.`;
    logControl(`행동 선택: ${action.label} -- 목표 데이터 미설정`);
    showToast(`⚠ "${action.label}"의 목표 동작 데이터가 아직 없습니다`, "warn");
    return;
  }
  if (isInCooldown()) {
    showToast(`⏳ 쿨다운 중입니다 (${cooldownRemainingSeconds().toFixed(1)}s 남음) — 잠시 후 다시 시도하세요`, "warn", 3500);
    return;
  }

  if (action.targetPerFinger) {
    // 손가락별 목표 데이터가 채워지면 이쪽을 우선 사용 (현재는 항상 null이라
    // 아래 targetPercent 분기로 감).
    for (const finger of DISPLAY_FINGER_KEYS) targetPerFinger[finger] = action.targetPerFinger[finger] ?? null;
    const avg = computeAverage(targetPerFinger);
    setTarget(avg ?? action.targetPercent, suggestInitialIntensity(avg ?? action.targetPercent));
  } else {
    for (const finger of DISPLAY_FINGER_KEYS) targetPerFinger[finger] = action.targetPercent;
    setTarget(action.targetPercent, suggestInitialIntensity(action.targetPercent));
  }
  updateCompareDisplay();
  updateFingerTable();

  actionDescriptionText.textContent = `${action.description} (목표 ${action.targetPercent}% -- placeholder 값)`;
  logControl(`🎯 행동 선택: ${action.label} → 목표 ${action.targetPercent}% (placeholder)`);
  showToast(`✅ "${action.label}" 목표 설정 (placeholder ${action.targetPercent}%)`, "ok");
}

// ---- 실시간 왼손 연동 (Live Mirror) ------------------------------------------

// 매 프레임 호출됨 (liveMirrorActive일 때만). setTarget()과 달리 successSince/
// holdLocked/controlState 등 컨트롤러 상태를 건드리지 않는다 -- 목표값만 계속
// 갱신하고, 오차/허용범위/유지 판정은 아래 updateController()가 매 틱마다 새로
// 계산하는 값을 그대로 쓰게 둔다. 왼손이 안 보이거나(latestPercent.source가
// null) 초기값 측정이 안 되어 있으면, 이전 목표를 그대로 유지한다 (갑자기
// 0/사라짐 처리하지 않음 -- 손을 잠깐 놓친 것과 캘리브레이션 안 된 것을
// 구분할 수 없어서 안전한 쪽인 "유지"를 택함).
function updateLiveMirrorTarget() {
  const avg = computeAverage(latestPercent.source);
  if (avg === null) return;
  targetPercent = Math.max(0, Math.min(100, avg));
  for (const finger of DISPLAY_FINGER_KEYS) {
    const v = latestPercent.source[finger];
    targetPerFinger[finger] = v === null || v === undefined ? null : v;
  }
}

function startLiveMirror() {
  if (!isRunning) {
    showToast("⚠ 먼저 카메라를 실행하세요", "warn");
    return;
  }
  liveMirrorActive = true;
  selectedAction = null;
  selectedActionText.textContent = "없음 (실시간 왼손 연동 중)";
  actionDescriptionText.textContent = "왼손의 지금 굽힘 정도가 계속 오른손 목표로 흘러갑니다.";
  for (const btn of actionButtonsRow.querySelectorAll("button")) btn.classList.remove("selected");
  liveMirrorBtn.textContent = "⏹ 실시간 연동 중지";
  liveMirrorBtn.classList.add("running");
  logControl("🔴 실시간 왼손 연동 시작");
  showToast("🔴 실시간 왼손 연동 시작 -- 왼손을 움직이면 오른손 목표가 계속 갱신됩니다", "ok");
}

function stopLiveMirror() {
  liveMirrorActive = false;
  liveMirrorBtn.textContent = "🔴 실시간 왼손 연동 시작";
  liveMirrorBtn.classList.remove("running");
  targetPercent = null;
  for (const finger of DISPLAY_FINGER_KEYS) targetPerFinger[finger] = null;
  successSince = null;
  holdLocked = false;
  controllerIntensity = 0;
  activeChannel = 1;
  controlState = "STANDBY";
  selectedActionText.textContent = "없음";
  actionDescriptionText.textContent = "행동을 선택하면 여기에 설명과 목표값이 표시됩니다.";
  updateCompareDisplay();
  updateFingerTable();
  logControl("⏹ 실시간 왼손 연동 중지");
  showToast("⏹ 실시간 왼손 연동 중지", "warn");
}

function toggleLiveMirror() {
  if (liveMirrorActive) stopLiveMirror();
  else startLiveMirror();
}

// ============================================================================
// 모드 전환 (거울 / 행동 보조 / 개인화)
// ============================================================================

function setAppMode(mode) {
  if (mode === appMode) return;

  // 이전 모드에서 돌고 있던 걸 전부 안전하게 정지 -- 특히 거울 모드의 폐루프
  // 컨트롤러(activeChannel/controllerIntensity)와 행동 보조/개인화 모드의 독립
  // 채널 제어가 동시에 하드웨어를 건드리면 두 시스템이 같은 채널을 놓고 서로
  // 다른 값을 계속 덮어쓰는 충돌이 생긴다 -- 그래서 모드가 바뀌면 무조건 다 끈다.
  if (liveMirrorActive) stopLiveMirror();
  if (runningActionKey) stopActionMode("모드 전환");
  if (personalizationRampActive) stopPersonalizationRamp("모드 전환");
  if (armSampling) armSampling = null; // 팔 초기값 측정 중이었으면 중단 (알림 없이 조용히 취소)
  testKeyChannel = null; // 테스트 모드에서 나가면서 방향키가 눌린 채로 남아있지 않게 확실히 정리
  if (controlEnabled) {
    controlEnabled = false;
    resetStartControlButton();
  }
  forceZeroController("모드 전환");
  if (serialLink) serialLink.stopAll().catch(() => {});
  armedCh1 = false;
  armedCh2 = false;
  lastDrivenChannel = null;

  appMode = mode;
  mirrorModeCard.style.display = mode === "mirror" ? "" : "none";
  actionModeCard.style.display = mode === "action" ? "" : "none";
  armModeCard.style.display = mode === "arm" ? "" : "none";
  testModeCard.style.display = mode === "test" ? "" : "none";
  modeMirrorBtn.classList.toggle("selected", mode === "mirror");
  modeActionBtn.classList.toggle("selected", mode === "action");
  modeArmBtn.classList.toggle("selected", mode === "arm");
  modeTestBtn.classList.toggle("selected", mode === "test");

  modeDescriptionText.innerHTML =
    mode === "mirror"
      ? "<b>거울 모드</b>: 행동 버튼(카메라로 오차를 계속 보정) 또는 실시간 왼손 연동으로 오른손을 목표에 맞춥니다."
      : mode === "action"
      ? "<b>행동 보조 모드</b>: 채널1/채널2에 직접 입력한 고정 전류를 그대로 내보냅니다 (카메라 오차 보정 없음). 개인화 측정(수동 램프업)도 이 안에 함께 있습니다."
      : mode === "arm"
      ? "<b>팔 인식 모드</b>: 왼팔의 팔꿈치 굽힘 정도를 실시간으로 오른팔 목표로 흘려보내고, 카메라로 측정한 오른팔의 실제 굽힘에 맞춰 자극 세기를 자동 조절합니다 (팔 버전 거울 모드)."
      : "<b>테스트 모드</b>: 카메라/캘리브레이션 없이, ←(채널1)/→(채널2) 방향키를 누르고 있는 동안만 그 채널에 고정 세기로 자극을 내보냅니다. 하드웨어가 실제로 잘 연결됐는지 빠르게 확인할 때만 쓰세요.";

  targetCompareLabel.textContent = mode === "arm" ? "TARGET (왼팔 · 실시간 연동)" : "TARGET (행동 선택 / 실시간 왼손 연동)";
  actualCompareLabel.textContent = mode === "arm" ? "ACTUAL (오른팔 · 팔꿈치 굽힘)" : "ACTUAL (오른손 · 4손가락 평균)";

  // 팔 인식 모드는 아직 모델을 안 불러왔을 수 있으니(perf를 위해 지연 로딩) 이
  // 시점에 미리 불러오기 시작 -- 카메라 실행 버튼을 누르기 전에 미리 받아둔다.
  if (mode === "arm" && !poseLandmarker) {
    showToast("🦾 팔 인식 모델을 불러오는 중입니다...", "ok", 3000);
    ensurePoseLandmarker()
      .then(() => showToast("✅ 팔 인식 모델 준비 완료", "ok"))
      .catch((err) => showToast("❌ 팔 인식 모델 로딩 실패: " + (err.message || err), "bad", 5000));
  }

  logControl(`모드 전환: ${mode === "mirror" ? "거울 모드" : mode === "action" ? "행동 보조 모드" : mode === "arm" ? "팔 인식 모드" : "테스트 모드"}`);
}

// ============================================================================
// 회원 개인화 -- 같은 세기라도 사람마다 반응이 달라서, 회원번호별로 행동 보조
// 모드 입력값(채널1/채널2/반복 횟수)을 저장해뒀다가 불러온다. localStorage에
// 저장 (참가자 프로필/Supabase 버전과는 별개 -- 그건 나중에 다시 붙일 예정이라
// 지금은 이 간단한 로컬 버전으로 감).
// ============================================================================

function loadMembers() {
  try {
    const raw = localStorage.getItem("emsWebMembers");
    if (raw) return JSON.parse(raw);
  } catch (e) { /* ignore */ }
  return {};
}
function saveMembers(members) {
  localStorage.setItem("emsWebMembers", JSON.stringify(members));
}

function loadMember() {
  const id = memberIdInput.value.trim();
  if (!id) {
    showToast("회원번호를 입력하세요", "warn");
    return;
  }
  const members = loadMembers();
  const member = members[id];
  if (!member) {
    memberWelcomeText.textContent = `⚠ "${id}"는 등록되지 않은 회원번호입니다. 아래 값을 입력하고 "회원 등록"을 눌러 새로 등록하세요.`;
    memberWelcomeText.style.color = "var(--warn)";
    showToast(`⚠ "${id}"는 등록되지 않은 회원입니다`, "warn");
    return;
  }

  memberNameInput.value = member.name || "";
  const a = member.actions || {};
  if (a.spoonLift) {
    spoonLiftCh1Input.value = a.spoonLift.ch1 ?? 0;
    spoonLiftCh2Input.value = a.spoonLift.ch2 ?? 0;
  }
  if (a.bicep) {
    bicepCh1Input.value = a.bicep.ch1 ?? 0;
    bicepCh2Input.value = a.bicep.ch2 ?? 0;
    bicepRepsInput.value = a.bicep.reps ?? 5;
  }

  memberWelcomeText.textContent = `✅ ${member.name}님 환영합니다 -- 저장된 채널값을 불러왔습니다.`;
  memberWelcomeText.style.color = "var(--accent-2)";
  logControl(`👤 회원 불러오기: ${id} (${member.name})`);
  showToast(`✅ ${member.name}님 환영합니다`, "ok");
  flashButtonPress(memberLoadBtn, "✅ 불러옴!");
}

function registerMember() {
  const id = memberIdInput.value.trim();
  const name = memberNameInput.value.trim();
  if (!id || !name) {
    showToast("회원번호와 이름을 모두 입력하세요", "warn");
    return;
  }

  const members = loadMembers();
  members[id] = {
    name,
    actions: {
      spoonLift: {
        ch1: Math.max(0, Math.min(100, Number(spoonLiftCh1Input.value) || 0)),
        ch2: Math.max(0, Math.min(100, Number(spoonLiftCh2Input.value) || 0))
      },
      bicep: {
        ch1: Math.max(0, Math.min(100, Number(bicepCh1Input.value) || 0)),
        ch2: Math.max(0, Math.min(100, Number(bicepCh2Input.value) || 0)),
        reps: Math.max(1, Math.min(100, Number(bicepRepsInput.value) || 5))
      }
    }
  };
  saveMembers(members);

  memberWelcomeText.textContent = `✅ ${name}님(${id}) 등록/저장 완료 -- 지금 입력칸의 값으로 저장했습니다.`;
  memberWelcomeText.style.color = "var(--accent-2)";
  logControl(`👤 회원 등록/갱신: ${id} (${name})`);
  showToast(`✅ ${name}님 등록 완료`, "ok");
  flashButtonPress(memberRegisterBtn, "✅ 등록됨!");
}

// ============================================================================
// 음성 명령 (마이크 -- "수저들기보조" / "이두운동" 딱 두 단어만 인식)
// ============================================================================
// 브라우저 내장 Web Speech API만 쓴다 (SpeechRecognition으로 듣고,
// SpeechSynthesis로 대답). 별도 서버/키 없음 -- Chrome/Edge 계열만 지원 (Web
// Serial 요구사항이랑 동일한 브라우저라 이미 맞음).
// (SpeechRecognitionCtor 자체는 initVoiceCommand()가 초기화 시점에 이미 참조하므로
// 파일 맨 위쪽 상수 선언부로 옮겨뒀다 -- 여기 남겨두면 TDZ로 인해
// "Cannot access before initialization" 오류가 난다.)


function speak(text) {
  try {
    if (!("speechSynthesis" in window)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "ko-KR";
    window.speechSynthesis.cancel(); // 이전에 말하던 게 남아있으면 끊고 새로 말함
    window.speechSynthesis.speak(utterance);
  } catch (e) { /* 음성 출력 실패는 치명적이지 않으니 조용히 무시 */ }
}

function initVoiceCommand() {
  if (!SpeechRecognitionCtor) {
    voiceCommandBtn.disabled = true;
    voiceCommandBtn.title = "이 브라우저는 음성 인식을 지원하지 않습니다 (Chrome/Edge 필요)";
    voiceCommandStatusText.textContent = "⚠ 이 브라우저는 음성 인식을 지원하지 않습니다 (Chrome 또는 Edge 필요).";
    return;
  }
  voiceRecognition = new SpeechRecognitionCtor();
  voiceRecognition.lang = "ko-KR";
  voiceRecognition.continuous = false;
  voiceRecognition.interimResults = false;
  voiceRecognition.maxAlternatives = 1;

  voiceRecognition.onstart = () => {
    voiceListening = true;
    voiceCommandBtn.textContent = "🎤 듣는 중...";
    voiceCommandBtn.classList.add("running");
    voiceCommandStatusText.textContent = "🎤 듣고 있습니다 -- \"수저들기보조\" 또는 \"이두운동\"이라고 말해주세요.";
  };

  voiceRecognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript.trim();
    logControl(`🎤 음성 인식 결과: "${transcript}"`);
    handleVoiceCommand(transcript);
  };

  voiceRecognition.onerror = (event) => {
    voiceCommandStatusText.textContent = `⚠ 음성 인식 오류: ${event.error}`;
    showToast(`⚠ 음성 인식 오류: ${event.error}`, "warn");
  };

  voiceRecognition.onend = () => {
    voiceListening = false;
    voiceCommandBtn.textContent = "🎤 음성 명령";
    voiceCommandBtn.classList.remove("running");
  };
}

function handleVoiceCommand(transcript) {
  const text = transcript.replace(/\s+/g, ""); // "수저 들기 보조"처럼 띄어써도 인식되게 공백 제거하고 비교
  if (text.includes("수저") || text.includes("숟가락")) {
    voiceCommandStatusText.textContent = `✅ 인식됨: "수저 들기 보조" → 시작합니다`;
    speak("수저 들기 보조를 시작합니다");
    startSequentialRamp("spoonLift", 2000, 3000);
  } else if (text.includes("이두")) {
    voiceCommandStatusText.textContent = `✅ 인식됨: "이두운동" → 시작합니다`;
    speak("이두운동을 시작합니다");
    startBicepRoutine();
  } else {
    voiceCommandStatusText.textContent = `❓ "${transcript}" -- "수저들기보조" 또는 "이두운동"만 인식합니다.`;
    speak("명령을 이해하지 못했습니다");
    showToast(`❓ 음성 명령을 이해하지 못했습니다: "${transcript}"`, "warn");
  }
}

function toggleVoiceCommand() {
  if (!voiceRecognition) return;
  if (voiceListening) {
    voiceRecognition.stop();
  } else {
    try {
      voiceRecognition.start();
    } catch (e) {
      // 이미 듣고 있는 상태에서 다시 start()를 부르면 예외가 나는 브라우저가 있어서 방어
    }
  }
}

// ============================================================================
// 행동 보조 모드 (오픈루프 -- 채널1/채널2에 입력한 고정 전류를 그대로 출력)
// ============================================================================

const ACTION_MODE_DEFS = {
  spoonLift: { label: "수저 들기 보조", ch1Input: spoonLiftCh1Input, ch2Input: spoonLiftCh2Input, btn: spoonLiftBtn },
  bicep: { label: "이두운동", ch1Input: bicepCh1Input, ch2Input: bicepCh2Input, btn: bicepBtn }
};

// 거울 모드의 activeChannel(채널 1개만 표현 가능)과 달리, 여기는 채널1/채널2를
// 동시에 서로 다른 값으로 켤 수 있다 (한 행동이 두 근육을 동시에 쓸 수 있으므로).
async function driveActionChannels(ch1Intensity, ch2Intensity) {
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) return;
  const ttl = clampTtl(config.safety.commandTtlMs);

  if (ch1Intensity > 0) {
    if (!armedCh1) {
      const ok = await serialLink.arm(1);
      if (ok) armedCh1 = true;
    }
    if (armedCh1) await serialLink.setIntensity(1, clampHardware(ch1Intensity), ttl);
  } else if (armedCh1) {
    await serialLink.setIntensity(1, 0, 500);
  }

  if (ch2Intensity > 0) {
    if (!armedCh2) {
      const ok = await serialLink.arm(2);
      if (ok) armedCh2 = true;
    }
    if (armedCh2) await serialLink.setIntensity(2, clampHardware(ch2Intensity), ttl);
  } else if (armedCh2) {
    await serialLink.setIntensity(2, 0, 500);
  }
}

// 순차 램프업 -- 두 채널을 동시에 목표까지 올리는 게 아니라 채널1 먼저, 채널2
// 나중에 순서대로 올린다 (행동마다 채널별로 걸리는 시간이 다를 수 있어서
// ch1RampMs/ch2RampMs를 인자로 받는다):
//   0 ~ ch1RampMs: 채널1이 0에서 목표까지 서서히 올라감 (채널2는 0)
//   ch1RampMs ~ ch1RampMs+ch2RampMs: 채널1은 목표값 유지, 채널2가 0에서 목표까지 서서히
//   그 이후: 둘 다 목표값 유지, 정지 버튼(또는 다시 누르기)을 누르기 전까지 계속
async function startSequentialRamp(key, ch1RampMs, ch2RampMs) {
  const def = ACTION_MODE_DEFS[key];

  // 개인화 측정이 켜진 채로 행동을 시작하면 같은 채널을 두고 서로 다른 값을
  // 계속 덮어쓰는 충돌이 생긴다 -- 행동을 시작하기 전에 먼저 확실히 끈다.
  if (personalizationRampActive) stopPersonalizationRamp("행동 실행으로 전환");

  if (runningActionKey && runningActionKey !== key) {
    stopActionMode(`"${ACTION_MODE_DEFS[runningActionKey].label}"에서 "${def.label}"로 전환`);
  } else if (runningActionKey === key) {
    stopActionMode("사용자가 다시 눌러 정지");
    return;
  }

  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    showToast(`⚠ ${allowed.reason}`, "warn", 4000);
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }

  const ch1Target = Math.max(0, Math.min(100, Number(def.ch1Input.value) || 0));
  const ch2Target = Math.max(0, Math.min(100, Number(def.ch2Input.value) || 0));
  if (ch1Target === 0 && ch2Target === 0) {
    showToast("⚠ 채널1/채널2 전류를 0보다 크게 입력하세요", "warn");
    return;
  }
  const totalRampS = (ch1RampMs + ch2RampMs) / 1000;
  if (config.safety.maxContinuousStimSeconds < totalRampS) {
    showToast(
      `⚠ 연속 자극 시간 제한이 ${config.safety.maxContinuousStimSeconds}초라 램프업(최소 ${totalRampS}초) 도중 자동 정지될 수 있어요 -- ③ 카드에서 늘려주세요`,
      "warn",
      5000
    );
  }

  runningActionKey = key;
  def.btn.textContent = `■ ${def.label} 정지`;
  def.btn.classList.add("running");
  logControl(
    `🎬 행동 보조 모드: ${def.label} 시작 (채널1 목표=${ch1Target} · ${ch1RampMs / 1000}초, 채널2 목표=${ch2Target} · ${ch2RampMs / 1000}초, 순차 램프업)`
  );
  showToast(`▶ ${def.label} 실행 (채널1→채널2 순서로 서서히)`, "ok");

  const rampState = { ch1Target, ch2Target, ch1RampMs, ch2RampMs, startedAt: performance.now() };
  sequentialRampTick(rampState); // 즉시 한 번 전송
  actionModeInterval = setInterval(() => sequentialRampTick(rampState), 400);
}

function sequentialRampTick(rampState) {
  if (safetyTripped) {
    stopActionMode("안전 정지 상태");
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    stopActionMode("시리얼 연결 끊김");
    return;
  }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    stopActionMode(allowed.reason);
    return;
  }
  if (continuousStimExceeded()) {
    stopActionMode("최대 연속 자극 시간을 초과했습니다");
    return;
  }
  if (totalTimeExceeded()) {
    stopActionMode("전체 실험 제한시간을 초과했습니다");
    return;
  }

  const { ch1Target, ch2Target, ch1RampMs, ch2RampMs } = rampState;
  const elapsed = performance.now() - rampState.startedAt;
  let ch1, ch2, phaseLabel;
  if (elapsed < ch1RampMs) {
    // 1단계: 채널1만 0 -> 목표로 서서히
    ch1 = ch1Target * (elapsed / ch1RampMs);
    ch2 = 0;
    phaseLabel = `채널1 올리는 중 (${Math.round(ch1)}/${ch1Target})`;
  } else if (elapsed < ch1RampMs + ch2RampMs) {
    // 2단계: 채널1은 목표 유지, 채널2가 0 -> 목표로 서서히
    ch1 = ch1Target;
    ch2 = ch2Target * ((elapsed - ch1RampMs) / ch2RampMs);
    phaseLabel = `채널1 유지(${ch1Target}) + 채널2 올리는 중 (${Math.round(ch2)}/${ch2Target})`;
  } else {
    // 3단계: 둘 다 목표 유지, 정지 누를 때까지 계속
    ch1 = ch1Target;
    ch2 = ch2Target;
    phaseLabel = `채널1(${ch1Target}) + 채널2(${ch2Target}) 유지 중 -- 정지 버튼을 누르기 전까지 계속`;
  }

  actionModeStatusText.textContent = `${ACTION_MODE_DEFS[runningActionKey].label}: ${phaseLabel}`;
  driveActionChannels(ch1, ch2).catch((err) => logControl("행동 보조 모드 전송 오류: " + err.message));
  if (ch1 > 0 || ch2 > 0) notifyStimStarted();
  else notifyStimStopped();
}

// 이두운동 전용 -- 채널1은 한 번(2초)만 올라가서 전체 반복 내내 목표값을 그대로
// 유지하고, 채널2가 "올라감(3초) → 유지(1초) → 내려감 → 짧은 휴식"을 입력한
// 반복 횟수만큼 되풀이한다. 내려가는 시간(BICEP_CH2_DOWN_MS)과 반복 사이 휴식
// (BICEP_REST_BETWEEN_REPS_MS)은 요청에 명시되지 않아서, 올라가는 시간과
// 대칭(3초)/짧은 값(0.5초)으로 우선 정했다 -- 필요하면 바꿀 수 있다.
const BICEP_CH1_RAMP_MS = 2000;
const BICEP_CH2_UP_MS = 3000;
const BICEP_CH2_HOLD_MS = 1000;
const BICEP_CH2_DOWN_MS = 3000;
const BICEP_REST_BETWEEN_REPS_MS = 500;

async function startBicepRoutine() {
  const key = "bicep";
  const def = ACTION_MODE_DEFS[key];

  if (personalizationRampActive) stopPersonalizationRamp("행동 실행으로 전환");

  if (runningActionKey && runningActionKey !== key) {
    stopActionMode(`"${ACTION_MODE_DEFS[runningActionKey].label}"에서 "${def.label}"로 전환`);
  } else if (runningActionKey === key) {
    stopActionMode("사용자가 다시 눌러 정지");
    return;
  }

  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    showToast(`⚠ ${allowed.reason}`, "warn", 4000);
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }

  const ch1Target = Math.max(0, Math.min(100, Number(def.ch1Input.value) || 0));
  const ch2Target = Math.max(0, Math.min(100, Number(def.ch2Input.value) || 0));
  if (ch1Target === 0 && ch2Target === 0) {
    showToast("⚠ 채널1/채널2 전류를 0보다 크게 입력하세요", "warn");
    return;
  }
  const reps = Math.max(1, Math.min(100, Number(bicepRepsInput.value) || 1));

  const oneRepMs = BICEP_CH2_UP_MS + BICEP_CH2_HOLD_MS + BICEP_CH2_DOWN_MS + BICEP_REST_BETWEEN_REPS_MS;
  const totalS = (BICEP_CH1_RAMP_MS + reps * oneRepMs) / 1000;
  if (config.safety.maxContinuousStimSeconds < totalS) {
    showToast(
      `⚠ 연속 자극 시간 제한이 ${config.safety.maxContinuousStimSeconds}초라 전체 세트(약 ${Math.ceil(totalS)}초) 도중 자동 정지될 수 있어요 -- ③ 카드에서 늘려주세요`,
      "warn",
      5000
    );
  }

  runningActionKey = key;
  def.btn.textContent = "■ 이두운동 정지";
  def.btn.classList.add("running");
  logControl(`🎬 행동 보조 모드: 이두운동 시작 (채널1=${ch1Target}, 채널2=${ch2Target}, ${reps}회 반복)`);
  showToast(`▶ 이두운동 실행 (${reps}회 반복)`, "ok");

  const state = {
    ch1Target,
    ch2Target,
    reps,
    repIndex: 0,
    phase: "ch1Ramp", // ch1Ramp -> ch2Up -> ch2Hold -> ch2Down -> rest -> (ch2Up ...반복) -> 완료
    phaseStartedAt: performance.now()
  };
  bicepRoutineTick(state); // 즉시 한 번 전송
  actionModeInterval = setInterval(() => bicepRoutineTick(state), 400);
}

function bicepRoutineTick(state) {
  if (safetyTripped) {
    stopActionMode("안전 정지 상태");
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    stopActionMode("시리얼 연결 끊김");
    return;
  }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    stopActionMode(allowed.reason);
    return;
  }
  if (continuousStimExceeded()) {
    stopActionMode("최대 연속 자극 시간을 초과했습니다");
    return;
  }
  if (totalTimeExceeded()) {
    stopActionMode("전체 실험 제한시간을 초과했습니다");
    return;
  }

  const now = performance.now();
  const elapsed = now - state.phaseStartedAt;
  let ch1 = state.ch1Target; // ch1Ramp 단계 이후엔 항상 목표값 그대로 유지
  let ch2 = 0;
  let phaseLabel = "";

  if (state.phase === "ch1Ramp") {
    ch1 = state.ch1Target * Math.min(1, elapsed / BICEP_CH1_RAMP_MS);
    phaseLabel = `채널1 올리는 중 (${Math.round(ch1)}/${state.ch1Target})`;
    if (elapsed >= BICEP_CH1_RAMP_MS) {
      state.phase = "ch2Up";
      state.phaseStartedAt = now;
      state.repIndex = 1;
    }
  } else if (state.phase === "ch2Up") {
    ch2 = state.ch2Target * Math.min(1, elapsed / BICEP_CH2_UP_MS);
    phaseLabel = `${state.repIndex}/${state.reps}회차 -- 채널2 올리는 중 (${Math.round(ch2)}/${state.ch2Target}), 채널1 유지(${state.ch1Target})`;
    if (elapsed >= BICEP_CH2_UP_MS) {
      state.phase = "ch2Hold";
      state.phaseStartedAt = now;
    }
  } else if (state.phase === "ch2Hold") {
    ch2 = state.ch2Target;
    phaseLabel = `${state.repIndex}/${state.reps}회차 -- 채널2 유지(${state.ch2Target}), 채널1 유지(${state.ch1Target})`;
    if (elapsed >= BICEP_CH2_HOLD_MS) {
      state.phase = "ch2Down";
      state.phaseStartedAt = now;
    }
  } else if (state.phase === "ch2Down") {
    ch2 = state.ch2Target * (1 - Math.min(1, elapsed / BICEP_CH2_DOWN_MS));
    phaseLabel = `${state.repIndex}/${state.reps}회차 -- 채널2 내리는 중 (${Math.round(ch2)}/${state.ch2Target}), 채널1 유지(${state.ch1Target})`;
    if (elapsed >= BICEP_CH2_DOWN_MS) {
      if (state.repIndex >= state.reps) {
        stopActionMode(`이두운동 ${state.reps}회 완료`);
        return;
      }
      state.phase = "rest";
      state.phaseStartedAt = now;
    }
  } else if (state.phase === "rest") {
    ch2 = 0;
    phaseLabel = `${state.repIndex}/${state.reps}회차 완료 -- 다음 반복 준비 중, 채널1 유지(${state.ch1Target})`;
    if (elapsed >= BICEP_REST_BETWEEN_REPS_MS) {
      state.repIndex += 1;
      state.phase = "ch2Up";
      state.phaseStartedAt = now;
    }
  }

  actionModeStatusText.textContent = `이두운동: ${phaseLabel}`;
  driveActionChannels(ch1, ch2).catch((err) => logControl("이두운동 전송 오류: " + err.message));
  if (ch1 > 0 || ch2 > 0) notifyStimStarted();
  else notifyStimStopped();
}

async function startAction(key) {
  const def = ACTION_MODE_DEFS[key];
  if (!def) return;

  if (runningActionKey && runningActionKey !== key) {
    stopActionMode(`"${ACTION_MODE_DEFS[runningActionKey].label}"에서 "${def.label}"로 전환`);
  } else if (runningActionKey === key) {
    stopActionMode("사용자가 다시 눌러 정지");
    return;
  }

  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    showToast(`⚠ ${allowed.reason}`, "warn", 4000);
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }

  const ch1Val = Math.max(0, Math.min(100, Number(def.ch1Input.value) || 0));
  const ch2Val = Math.max(0, Math.min(100, Number(def.ch2Input.value) || 0));
  if (ch1Val === 0 && ch2Val === 0) {
    showToast("⚠ 채널1/채널2 전류를 0보다 크게 입력하세요", "warn");
    return;
  }

  // 행동 보조 모드도 도중에 한 번도 0으로 안 끊기고 계속 흐르는 방식이라, 연속
  // 자극 시간 제한(기본 10초)에 걸려 짧게짧게 계속 자동 정지될 수 있다. 실행
  // 버튼을 누를 때마다 미리 알려준다 (막지는 않음).
  if (config.safety.maxContinuousStimSeconds < 30) {
    showToast(
      `⚠ 연속 자극 시간 제한이 ${config.safety.maxContinuousStimSeconds}초라 곧 자동 정지될 수 있어요 -- 계속 유지하려면 ③ 카드에서 늘려주세요`,
      "warn",
      5000
    );
  }

  runningActionKey = key;
  def.btn.textContent = `■ ${def.label} 정지`;
  def.btn.classList.add("running");
  actionModeStatusText.textContent = `${def.label} 실행 중 (채널1: ${ch1Val}, 채널2: ${ch2Val})`;
  logControl(`🎬 행동 보조 모드: ${def.label} 시작 (채널1=${ch1Val}, 채널2=${ch2Val})`);
  showToast(`▶ ${def.label} 실행`, "ok");

  actionModeTick(ch1Val, ch2Val); // 즉시 한 번 전송
  actionModeInterval = setInterval(() => actionModeTick(ch1Val, ch2Val), 400);
}

function actionModeTick(ch1Val, ch2Val) {
  if (safetyTripped) {
    stopActionMode("안전 정지 상태");
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    stopActionMode("시리얼 연결 끊김");
    return;
  }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    stopActionMode(allowed.reason);
    return;
  }
  if (continuousStimExceeded()) {
    stopActionMode("최대 연속 자극 시간을 초과했습니다");
    return;
  }
  if (totalTimeExceeded()) {
    stopActionMode("전체 실험 제한시간을 초과했습니다");
    return;
  }
  driveActionChannels(ch1Val, ch2Val).catch((err) => logControl("행동 보조 모드 전송 오류: " + err.message));
  if (ch1Val > 0 || ch2Val > 0) notifyStimStarted();
  else notifyStimStopped();
}

function stopActionMode(reason) {
  if (actionModeInterval) {
    clearInterval(actionModeInterval);
    actionModeInterval = null;
  }
  if (runningActionKey) {
    const def = ACTION_MODE_DEFS[runningActionKey];
    def.btn.textContent = `▶ ${def.label} 실행`;
    def.btn.classList.remove("running");
    logControl(`⏹ 행동 보조 모드 정지: ${reason}`);
  }
  runningActionKey = null;
  actionModeStatusText.textContent = "대기 중";
  notifyStimStopped();
  if (serialLink) {
    serialLink.setIntensity(1, 0, 500).catch(() => {});
    serialLink.setIntensity(2, 0, 500).catch(() => {});
  }
}

// ============================================================================
// 개인화 모드 (수동 램프업 -- 키보드 A로 정지)
// ============================================================================

function startPersonalizationRamp() {
  if (personalizationRampActive) return;

  // 개인화 측정과 행동(수저/이두)이 이제 같은 "행동 보조 모드" 안에 같이 있어서,
  // 서로 모르고 동시에 하드웨어를 건드리면 같은 채널을 두고 값이 계속 덮어써지는
  // 충돌이 생긴다 -- 시작하기 전에 실행 중인 행동을 먼저 확실히 끈다.
  if (runningActionKey) stopActionMode("개인화 측정 시작으로 전환");

  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    showToast(`⚠ ${allowed.reason}`, "warn", 4000);
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }

  const step = Math.max(1, Math.min(20, Number(personalizationRampStepInput.value) || 2));

  // 개인화 램프는 도중에 한 번도 0으로 안 끊기고 계속 흐르기 때문에, 연속 자극
  // 시간 제한(maxContinuousStimSeconds)이 너무 낮으면 100에 채 도달하기도
  // 전에 매번 자동으로 끊길 수 있다. 미리 계산해서 그럴 것 같으면 경고한다
  // (그래도 시작은 그대로 진행 -- 사람이 판단할 문제라 막지는 않음).
  const secondsToReach100 = (100 / step) * (PERSONALIZATION_RAMP_TICK_MS / 1000);
  if (secondsToReach100 > config.safety.maxContinuousStimSeconds) {
    showToast(
      `⚠ 연속 자극 시간 제한이 ${config.safety.maxContinuousStimSeconds}초라 100에 닿기 전(약 ${Math.ceil(secondsToReach100)}초 소요)에 자동 정지될 수 있어요 -- ③ 카드에서 늘려주세요`,
      "warn",
      5000
    );
  }

  personalizationRampChannel = personalizationCh2Radio.checked ? 2 : 1;
  personalizationRampIntensity = 0;
  personalizationRampActive = true;
  personalizationRampBtn.textContent = "■ 개인화 모드 정지";
  personalizationRampBtn.classList.add("running");
  personalizationRampStatusText.textContent = `채널${personalizationRampChannel} 증가 중 -- 키보드 A로 정지`;
  personalizationRampValue.textContent = "0";
  logControl(`📈 개인화 모드 시작 (채널${personalizationRampChannel})`);
  showToast("📈 개인화 모드 시작 -- 원하는 지점에서 키보드 A를 누르세요", "ok");

  personalizationRampInterval = setInterval(() => personalizationRampTick(step), PERSONALIZATION_RAMP_TICK_MS);
}

function personalizationRampTick(step) {
  if (safetyTripped) {
    stopPersonalizationRamp("안전 정지 상태");
    return;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    stopPersonalizationRamp("시리얼 연결 끊김");
    return;
  }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    stopPersonalizationRamp(allowed.reason);
    return;
  }
  if (continuousStimExceeded()) {
    stopPersonalizationRamp("최대 연속 자극 시간을 초과했습니다");
    return;
  }
  if (totalTimeExceeded()) {
    stopPersonalizationRamp("전체 실험 제한시간을 초과했습니다");
    return;
  }

  personalizationRampIntensity = Math.min(100, personalizationRampIntensity + step);
  personalizationRampValue.textContent = Math.round(personalizationRampIntensity);
  const ch1 = personalizationRampChannel === 1 ? personalizationRampIntensity : 0;
  const ch2 = personalizationRampChannel === 2 ? personalizationRampIntensity : 0;
  driveActionChannels(ch1, ch2).catch((err) => logControl("개인화 모드 전송 오류: " + err.message));
  notifyStimStarted();

  if (personalizationRampIntensity >= 100) {
    stopPersonalizationRamp("최대값(100)에 도달");
  }
}

function stopPersonalizationRamp(reason) {
  if (personalizationRampInterval) {
    clearInterval(personalizationRampInterval);
    personalizationRampInterval = null;
  }
  if (personalizationRampActive) {
    const reached = Math.round(personalizationRampIntensity);
    logControl(`⏹ 개인화 모드 정지: ${reason} (마지막 세기: ${reached})`);
    personalizationRampResultText.innerHTML = `마지막 측정: 채널${personalizationRampChannel} · 세기 <b>${reached}</b> (${reason})`;
    showToast(`⏹ 개인화 모드 정지 (세기 ${reached})`, "warn");
  }
  personalizationRampActive = false;
  personalizationRampBtn.textContent = "▶ 개인화 모드 시작";
  personalizationRampBtn.classList.remove("running");
  personalizationRampStatusText.textContent = "대기 중";
  notifyStimStopped();
  if (serialLink) {
    serialLink.setIntensity(1, 0, 500).catch(() => {});
    serialLink.setIntensity(2, 0, 500).catch(() => {});
  }
}

// 개인화 모드에서 방금 잰 세기(personalizationRampIntensity)를, 위에서 고른
// 채널(personalizationRampChannel)에 맞춰 선택한 행동의 채널 입력칸에 채워넣는다.
// 여기서 끝나는 게 아니라, 그 뒤 "회원 등록"을 눌러야 실제로 그 사람 값으로
// 저장된다 (이 버튼은 입력칸까지만 채워줌).
function savePersonalizationValue() {
  if (personalizationRampIntensity <= 0) {
    showToast("⚠ 아직 측정된 세기가 없습니다 -- 개인화 모드를 먼저 실행하세요", "warn");
    return;
  }
  const targetAction = personalizationTargetBicep.checked ? "bicep" : "spoonLift";
  const value = Math.round(personalizationRampIntensity);
  const input =
    targetAction === "spoonLift"
      ? (personalizationRampChannel === 1 ? spoonLiftCh1Input : spoonLiftCh2Input)
      : (personalizationRampChannel === 1 ? bicepCh1Input : bicepCh2Input);

  input.value = value;
  const actionLabel = targetAction === "spoonLift" ? "수저 들기 보조" : "이두운동";
  logControl(`💾 개인화 값 저장: ${actionLabel} 채널${personalizationRampChannel} = ${value} (입력칸에 채움)`);
  showToast(`✅ ${actionLabel} 채널${personalizationRampChannel}에 ${value} 채워짐 -- 회원 등록으로 마저 저장하세요`, "ok");
  flashButtonPress(personalizationSaveBtn, "✅ 저장됨!");
}

// ============================================================================
// 폐루프 제어기 (controller.py 를 JS로 포팅)
// ============================================================================

function setTarget(percent, initialIntensity) {
  targetPercent = Math.max(0, Math.min(100, percent));
  successSince = null;
  lastStepTime = 0;
  holdLocked = false; // 새 목표가 들어오면 이전 자세 유지는 자동으로 해제
  activeChannel = 1; // 새 목표는 일단 채널1(구부림)부터 시작 -- 실제 오차 방향 보고 자동 전환됨
  if (initialIntensity !== undefined && initialIntensity !== null) {
    controllerIntensity = clampWorking(initialIntensity);
  }
  controlState = "IDLE";
}

function forceZeroController(reason) {
  controllerIntensity = 0;
  activeChannel = 1;
  targetPercent = null;
  for (const finger of DISPLAY_FINGER_KEYS) targetPerFinger[finger] = null;
  successSince = null;
  holdLocked = false;
  controlState = "SAFETY_STOP";
  logControl(reason);
}

function updateController(currentPercent, now) {
  if (targetPercent === null) {
    controlState = "IDLE";
    lastError = null;
    return;
  }
  if (currentPercent === null) {
    successSince = null;
    controlState = "WAITING_FOR_HAND";
    lastError = null;
    return;
  }

  const error = targetPercent - currentPercent;
  lastError = error;

  if (Math.abs(error) <= config.control.tolerancePercent) {
    // 오차가 줄어도 즉시 끄지 않고, "해제"나 "■ 제어 정지"를 누르기 전까지
    // 지금 세기를 그대로 계속 흘린다 (펄스가 아니라 진짜 연속 출력).
    // controllerIntensity를 여기서 건드리지 않으므로, 목표에 접근하며
    // 마지막으로 계산됐던 값이 그대로 유지된다.
    //
    // ⚠ 이러면 config.safety.maxContinuousStimSeconds(연속 자극 시간 안전
    // 상한, 기본 10초)에 그대로 걸린다 -- runtimeCheck()가 이 시간을 넘기면
    // 자동으로 비상정지시킨다. 코드가 이 값을 알아서 늘리지 않으니, 더 오래
    // 유지하려면 ③ 안전 설정 카드에서 사람이 직접 늘려야 한다.
    if (successSince === null) successSince = now;
    const heldForS = (now - successSince) / 1000;
    if (heldForS >= config.control.successHoldSeconds) {
      controlState = "LOCKED";
      if (!holdLocked) {
        // holdLocked는 "해제" 버튼이 먹히게 하는 표시로 쓰인다 (releaseHold()가
        // 이 값을 확인함).
        holdLocked = true;
        logControl("✅ 도달 완료 -- 자세를 유지합니다 (계속 자극, 해제 버튼을 누르기 전까지)");
        showToast("✅ 도달 완료 -- 계속 자극 중 (해제 버튼으로 종료)", "ok");
      }
    } else {
      controlState = "HOLDING";
    }
    return; // controllerIntensity는 건드리지 않음 -- 마지막 값 그대로 유지 (연속 출력)
  }
  successSince = null;
  // 유지(LOCKED) 중에 자세가 흐트러져 오차가 다시 벌어지면, 해제 버튼을 누르지
  // 않아도 아래 일반 비례제어 로직으로 자동으로 다시 목표를 향해 조정한다.
  holdLocked = false;

  if (now - lastStepTime < getControlPeriodMs()) {
    controlState = error > 0 ? "INCREASING" : "DECREASING";
    return;
  }
  lastStepTime = now;

  // 2채널 방향 라우팅: error>0(덜 구부러짐) -> 채널1(구부림/굴근), error<0
  // (더 구부러짐, 펴야 함) -> 채널2(폄/신근). 채널이 바뀌는 순간엔 이전 채널
  // 세기를 그대로 이어받지 않고 0부터 다시 램프업한다 -- 서로 다른 근육으로
  // 갑자기 큰 세기가 그대로 넘어가지 않게 하기 위함. (두 채널이 동시에 켜지는
  // 일은 activeChannel 하나로만 controllerIntensity를 표현하는 구조상 애초에
  // 불가능하다 -- driveHardwareIfNeeded()도 이 activeChannel 하나만 하드웨어로 보낸다.)
  const desiredChannel = error > 0 ? 1 : 2;
  if (desiredChannel !== activeChannel) {
    activeChannel = desiredChannel;
    controllerIntensity = 0;
  }

  // 오차가 클수록 크게, 목표에 가까워질수록 작게 조절되도록 진짜 비례(P) 계산.
  // kpUp/maxStepUp은 채널1(구부림), kpDown/maxStepDown은 채널2(폄) 쪽 게인이다
  // (원래는 "신전 기능이 없어서 넘친 쪽을 빠르게 회수"하려고 kpDown을 크게 뒀던
  // 건데, 이제 채널2가 실제로 펴는 근육을 담당하므로 "빠르게 펴는 속도"가 됐다
  // -- 실제 반응 보고 kpDown 값을 다시 튜닝해야 할 수 있다).
  const gain = activeChannel === 1 ? config.control.kpUp : config.control.kpDown;
  const maxStep = activeChannel === 1 ? config.control.maxStepUp : config.control.maxStepDown;
  const rawStep = gain * Math.abs(error);
  const step = Math.min(maxStep, rawStep);
  const before = controllerIntensity;
  controllerIntensity = clampWorkingFloat(controllerIntensity + step);
  controlState = step > 0.01 ? (activeChannel === 1 ? "INCREASING" : "DECREASING") : "HOLDING";

  // control_period_ms에 한 번만 실행되는 구간이라 스팸 걱정 없이 매번 사람이
  // 읽을 수 있는 진행상황 문장을 남긴다.
  if (Math.abs(controllerIntensity - before) > 0.001) {
    const reason = activeChannel === 1 ? "덜 구부러짐 → 채널1(구부림)" : "더 구부러짐 → 채널2(폄)";
    logControl(`${reason} (오차 ${error > 0 ? "+" : ""}${error.toFixed(0)}%) → 자극값 ${before.toFixed(1)}→${controllerIntensity.toFixed(1)}`);
  }
}

// (예전엔 여기 tickHoldPulse()가 있었다 -- LOCKED 상태에서 계속 흘리는 대신
// 짧게 펄스만 주는 방식. 이제 "거울 모드" 하나로 합쳐지면서 연속 출력 방식만
// 남았고, 이 펄스 로직은 updateController()의 LOCKED 분기에서 더 이상 쓰이지
// 않아 제거했다. 필요해지면 이전 대화 기록에서 복구 가능.)

// "해제" 버튼 -- 자세 유지(LOCKED) 상태를 끝내고 완전히 정지한다.
function releaseHold(reason) {
  if (!holdLocked) {
    showToast("지금은 자세 유지 중이 아닙니다", "warn");
    return;
  }
  holdLocked = false;
  controllerIntensity = 0;
  activeChannel = 1;
  targetPercent = null;
  for (const finger of DISPLAY_FINGER_KEYS) targetPerFinger[finger] = null;
  successSince = null;
  controlState = "STANDBY";
  notifyStimStopped();
  startCooldown();
  armedCh1 = false;
  armedCh2 = false;
  lastDrivenChannel = null;
  logControl(`⏹ 자세 유지 해제: ${reason}`);
  showToast("⏹ 자세 유지를 해제했습니다", "warn");
}

// ============================================================================
// 안전 관리 (safety_manager.py 를 JS로 포팅)
// ============================================================================

function clampWorking(v) {
  return Math.round(Math.min(100, Math.max(0, v)));
}
// clampWorking과 달리 반올림하지 않는다. controllerIntensity를 매 스텝마다
// 정수로 반올림해서 저장하면, kpUp이 작을 때 나오는 0.1~0.3 같은 미세한
// 증가분이 반올림에 먹혀서 사라지고 목표 근처에서 값이 영원히 멈춰버리는
// 버그가 있었다. 그래서 내부 누적값은 float로 유지하고, 화면 표시/하드웨어
// 전송 시점에만 반올림한다.
function clampWorkingFloat(v) {
  return Math.min(100, Math.max(0, v));
}
function clampHardware(v) {
  const lo = Math.max(0, config.safety.minIntensity);
  const hi = Math.min(100, config.safety.maxIntensity);
  if (hi <= 0) return 0;
  return Math.round(Math.min(Math.max(v, lo), hi));
}
function clampTtl(ms) {
  return Math.min(Math.max(ms, 0), config.safety.maxCommandTtlMs);
}

function liveOutputAllowedByConfig() {
  if (config.safety.maxIntensity <= 0) {
    return { ok: false, reason: "안전 최대 제어값이 0입니다. 직접 0보다 크게 설정해야 실제 출력이 허용됩니다." };
  }
  if (config.safety.maxIntensity > 100) return { ok: false, reason: "안전 최대 제어값은 0~100 사이여야 합니다." };
  if (config.safety.minIntensity < 0 || config.safety.minIntensity > config.safety.maxIntensity) {
    return { ok: false, reason: "안전 최소값 설정이 올바르지 않습니다." };
  }
  return { ok: true };
}

function runtimeCheck(handLost) {
  if (!liveModeRequested) return { ok: true };
  if (!serialLink || !serialLink.isConnected()) return { ok: false, reason: "시리얼 연결이 끊어졌습니다." };
  if (handLost) return { ok: false, reason: appMode === "arm" ? "오른팔(ACTUAL)을 카메라가 놓쳤습니다." : "오른손(ACTUAL)을 카메라가 놓쳤습니다." };
  if (continuousStimExceeded()) return { ok: false, reason: "최대 연속 자극 시간을 초과했습니다." };
  if (totalTimeExceeded()) return { ok: false, reason: "전체 실험 제한시간을 초과했습니다." };
  return { ok: true };
}

function triggerEmergencyStop(reason) {
  const wasTripped = safetyTripped;
  safetyTripped = true;
  safetyTripReason = reason;
  forceZeroController(`🛑 안전 정지: ${reason}`);
  if (runningActionKey) stopActionMode(`🛑 안전 정지: ${reason}`);
  if (personalizationRampActive) stopPersonalizationRamp(`🛑 안전 정지: ${reason}`);
  if (serialLink) serialLink.stopAll().catch(() => {});
  armedCh1 = false;
  armedCh2 = false;
  lastDrivenChannel = null;
  resetStartControlButton(); // 비상정지 후엔 반드시 다시 "▶ 제어 시작"을 눌러야만 재개됨
  updateSafetyTripUi();
  if (!wasTripped) logControl(`🛑 비상정지: ${reason}`);
}
function resetStartControlButton() {
  controlEnabled = false;
  startControlBtn.textContent = "▶ 제어 시작";
  startControlBtn.classList.remove("running");
}
function resetAfterTrip() {
  safetyTripped = false;
  safetyTripReason = "";
  updateSafetyTripUi();
}

// safetyTripped는 한 번 true가 되면 이걸 풀기 전까지 모든 모드가 첫 틱에서
// 바로 다시 꺼진다 (actionModeTick/personalizationRampTick의 safetyTripped
// 체크 참고). 예전엔 이 상태가 화면 어디에도 안 보여서 "왜 자꾸 꺼지지?"의
// 원인을 알 방법이 없었다 -- 지금은 ③ 카드 맨 위에 이유와 해제 버튼을 보여준다.
function updateSafetyTripUi() {
  if (safetyTripped) {
    safetyTripRow.style.display = "";
    safetyTripHint.style.display = "";
    safetyTripReasonText.textContent = `원인: ${safetyTripReason}`;
  } else {
    safetyTripRow.style.display = "none";
    safetyTripHint.style.display = "none";
    safetyTripReasonText.textContent = "";
  }
}

function notifyStimStarted() {
  if (continuousStimStart === null) continuousStimStart = performance.now();
}
function notifyStimStopped() {
  continuousStimStart = null;
}
function continuousStimExceeded() {
  if (continuousStimStart === null) return false;
  return (performance.now() - continuousStimStart) / 1000 > config.safety.maxContinuousStimSeconds;
}
function totalTimeExceeded() {
  if (experimentStart === null) return false;
  return (performance.now() - experimentStart) / 1000 > config.safety.totalExperimentSeconds;
}
function startCooldown() {
  cooldownUntil = performance.now() + config.safety.cooldownSeconds * 1000;
}
function cooldownRemainingSeconds() {
  return cooldownUntil === null ? 0 : Math.max(0, (cooldownUntil - performance.now()) / 1000);
}
function isInCooldown() {
  if (cooldownUntil === null) return false;
  if (performance.now() >= cooldownUntil) {
    cooldownUntil = null;
    return false;
  }
  return true;
}

async function driveHardwareIfNeeded(channel, intensity) {
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) return;

  // 채널이 바뀌면(구부리다가 펴는 쪽으로, 또는 그 반대로) 이전 채널을 먼저
  // 확실히 0으로 끈다 -- 두 채널이 동시에 켜지는 일이 절대 없게 하기 위함
  // (길항근 동시수축 방지). activeChannel/controllerIntensity 하나로만 표현하는
  // 구조상 우리 쪽 상태는 이미 안전하지만, 하드웨어(Arduino)는 마지막으로 받은
  // SET을 TTL 끝날 때까지 계속 유지하므로 명시적으로 꺼줘야 한다.
  if (lastDrivenChannel !== null && lastDrivenChannel !== channel) {
    try {
      await serialLink.setIntensity(lastDrivenChannel, 0, 500);
    } catch (e) { /* ignore -- 아래에서 새 채널 전송이 실패하면 그건 잡아서 로그로 알림 */ }
  }

  const alreadyArmed = channel === 1 ? armedCh1 : armedCh2;
  if (!alreadyArmed) {
    const ok = await serialLink.arm(channel);
    if (!ok) return;
    if (channel === 1) armedCh1 = true;
    else armedCh2 = true;
  }
  const ttl = clampTtl(config.safety.commandTtlMs);
  await serialLink.setIntensity(channel, clampHardware(intensity), ttl);
  lastDrivenChannel = channel;
}

// ============================================================================
// 테스트 모드 -- ←(채널1)/→(채널2) 방향키를 누르고 있는 동안만 그 채널에
// 고정 세기를 내보낸다 (떼면 즉시 정지). 카메라·캘리브레이션·목표비교·연속
// 자극시간/전체 실험시간 제한 같은 폐루프 쪽 로직은 전혀 거치지 않는다 --
// 순전히 "이 채널에 지금 전기가 나가는지"만 빠르게 확인하려는 용도.
//
// 그래도 절대 건너뛰지 않는 것 두 가지:
//   1) config.safety.maxIntensity(기본 0) 하드웨어 클램프 -- clampHardware()를
//      그대로 통과시키므로, ③ 카드에서 안전 최대값을 0보다 크게 설정하지
//      않으면 여기서도 실제로는 0만 나간다.
//   2) safetyTripped(비상정지) 상태 -- 걸려있으면 테스트 모드도 막는다.
// 이 둘은 "번거로운 절차"가 아니라 마지막 하드웨어 안전판이라 그대로 둔다.
async function testModeKeyDown(channel) {
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }

  testKeyChannel = channel;

  // 다른 채널이 눌려있던 상태였다면 먼저 확실히 끔 (두 채널 동시 자극 방지 --
  // driveHardwareIfNeeded와 같은 이유).
  if (lastDrivenChannel !== null && lastDrivenChannel !== channel) {
    try { await serialLink.setIntensity(lastDrivenChannel, 0, 500); } catch (e) { /* ignore */ }
  }

  const alreadyArmed = channel === 1 ? armedCh1 : armedCh2;
  if (!alreadyArmed) {
    const ok = await serialLink.arm(channel);
    if (!ok) {
      showToast(`⚠ 채널${channel} ARM 실패`, "bad");
      return;
    }
    if (channel === 1) armedCh1 = true;
    else armedCh2 = true;
  }

  // ③ 카드의 안전 최대값(clampHardware)을 거치지 않고, 입력한 세기를 0~100
  // 범위만 맞춰서 그대로 내보낸다 (테스트 모드는 전기 주기 + 세기 설정, 그
  // 두 가지만 하도록 요청받아 나머지 안전 게이트는 여기서는 뺐다).
  const intensity = Math.round(Math.min(100, Math.max(0, Number(testIntensityInput.value) || 0)));
  // TTL을 짧게 잡아서(600ms), 키를 계속 누르고 있으면 브라우저 키 반복 입력이
  // 이 함수를 계속 다시 불러 SET을 반복 전송하며 TTL을 계속 갱신한다. 혹시
  // 반복 입력이 하필 늦게 와도 600ms 안에는 자동으로 꺼지니, keyup을 못 받는
  // 상황(창 포커스 이탈 등)에서도 오래 켜진 채로 남지 않는다.
  await serialLink.setIntensity(channel, intensity, 600);
  lastDrivenChannel = channel;
  testModeStatusText.textContent = `채널${channel} 자극 중 (세기 ${intensity})`;
}

async function testModeKeyUp(channel) {
  if (testKeyChannel !== channel) return; // 이미 다른 키로 넘어갔거나 이미 꺼진 상태
  testKeyChannel = null;
  if (!serialLink || !serialLink.isConnected()) return;
  try { await serialLink.setIntensity(channel, 0, 500); } catch (e) { /* ignore */ }
  testModeStatusText.textContent = "대기 중 (←/→ 방향키를 누르고 있으면 자극)";
}

// ============================================================================
// 개인화 모델 (간단 선형회귀, adaptive_model.py 를 JS로 포팅)
// ============================================================================

// 최소자승 선형회귀 -- fitAdaptiveModel(로그 기반)과 fitPersonalizationModel
// (명시적 스윕 기반) 둘 다 이 함수를 공유한다.
function linearRegression(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const sumX = xs.reduce((a, b) => a + b, 0);
  const sumY = ys.reduce((a, b) => a + b, 0);
  const sumXY = xs.reduce((s, x, i) => s + x * ys[i], 0);
  const sumXX = xs.reduce((s, x) => s + x * x, 0);
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-6) return null;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

function fitAdaptiveModel() {
  const rows = logRows.filter((r) => (r.state === "HOLDING" || r.state === "SUCCESS") && r.current !== null);
  if (rows.length < 8) {
    adaptiveModel = { coeffs: null, sampleCount: rows.length };
    saveAdaptiveModel();
    return;
  }
  const fit = linearRegression(rows.map((r) => r.intensity), rows.map((r) => r.current));
  adaptiveModel = { coeffs: fit ? [fit.slope, fit.intercept] : null, sampleCount: rows.length };
  saveAdaptiveModel();
}

// 개인화 캘리브레이션(자극값 스윕)에서 모은, 훨씬 깨끗한 (자극값, 실제 굽힘%)
// 데이터로 회귀를 만든다. 아래 suggestInitialIntensity()는 이게 있으면 이걸
// 우선 쓰고, 없을 때만 fitAdaptiveModel()의 로그 기반 추정으로 대체한다.
function fitPersonalizationModel() {
  if (personalization.points.length < 2) {
    personalization.coeffs = null;
    savePersonalization();
    return;
  }
  const fit = linearRegression(personalization.points.map((p) => p.intensity), personalization.points.map((p) => p.percent));
  personalization.coeffs = fit ? [fit.slope, fit.intercept] : null;
  personalization.fittedAt = performance.now();
  savePersonalization();
}

function suggestInitialIntensity(target) {
  if (personalization.coeffs) {
    const [slope, intercept] = personalization.coeffs;
    if (Math.abs(slope) > 1e-6) {
      const guess = (target - intercept) / slope;
      if (Number.isFinite(guess)) return clampWorking(guess);
    }
  }
  fitAdaptiveModel();
  if (!adaptiveModel || !adaptiveModel.coeffs) return clampWorking(config.safety.minIntensity);
  const [slope, intercept] = adaptiveModel.coeffs;
  if (Math.abs(slope) < 1e-6) return clampWorking(config.safety.minIntensity);
  const guess = (target - intercept) / slope;
  if (!Number.isFinite(guess)) return clampWorking(config.safety.minIntensity);
  return clampWorking(guess); // 작업용 범위로만 clamp -- 실제 기기 전송 시에는 별도로 clampHardware() 적용
}

// ============================================================================
// 개인화 캘리브레이션 루틴 (열린 루프 자극값 스윕 + 시각화)
// ============================================================================
// 폐루프 제어(목표를 향해 조금씩 조절)와는 다르게, 여기서는 정해진 자극값을
// 순서대로 하나씩 그대로 줘보고 그 결과(실제 굽힘%)만 측정한다. 이렇게 모은
// 깨끗한 (자극값, 굽힘%) 점들로 "이 사람은 자극이 얼마면 얼마나 반응하는지"
// 개인별 곡선을 만들어서, 다음에 목표를 정할 때 시작 자극값을 더 정확하게
// 추천하는 데 쓴다 (교수님이 말씀하신 "사용자별 근육 반응 학습").

function buildSweepLevels() {
  const max = config.safety.maxIntensity;
  const candidates = [20, 40, 60, 80, 100].filter((v) => v <= max);
  if (candidates.length >= 2) return candidates;
  // 안전 최대값이 낮게 잡혀 있어도(예: 30) 그 안에서 최소 2단계는 만든다.
  if (max >= 10) return [Math.round(max / 2), max];
  return [];
}

async function startPersonalizationSweep() {
  if (sweep) return;
  if (!isRunning) { showToast("⚠ 먼저 카메라를 실행하세요", "warn"); return; }
  if (!isCalibrated()) { showToast("⚠ 오른손 초기값(펴짐/구부림) 측정을 먼저 해주세요", "warn"); return; }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    showToast(`⚠ ${allowed.reason}`, "warn", 4000);
    return;
  }
  const levels = buildSweepLevels();
  if (levels.length < 2) {
    showToast("⚠ 안전 최대 제어값이 너무 낮아 두 단계 이상 만들 수 없습니다", "warn", 4000);
    return;
  }
  const confirmed = confirm(
    `지금부터 자극값 ${levels.join(" → ")}을 순서대로, 각각 ${(SWEEP_HOLD_MS / 1000).toFixed(1)}초씩 자동으로 줍니다.\n` +
    "중간에 불편하면 언제든 비상정지(SPACE/ESC) 또는 아래 '중지' 버튼으로 즉시 멈출 수 있습니다.\n계속하시겠습니까?"
  );
  if (!confirmed) return;

  // 이 루틴이 도는 동안은 목표 추종 폐루프와 하드웨어를 동시에 건드리면
  // 안 되므로, 이미 켜져 있었다면 확실히 꺼둔다.
  if (controlEnabled) {
    controlEnabled = false;
    resetStartControlButton();
  }
  controllerIntensity = 0;

  if (!armedCh1) {
    const ok = await serialLink.arm(1);
    armedCh1 = ok;
    if (!ok) {
      showToast("⚠ Arduino ARM 실패 -- 다시 시도해주세요", "bad", 4000);
      return;
    }
  }

  sweep = {
    levels,
    levelIndex: 0,
    phase: "hold", // "hold" -> "rest" -> (다음 레벨의) "hold" ... -> 완료
    phaseStartedAt: performance.now(),
    lastSendTime: 0,
    samples: [],
    collected: []
  };
  startControlBtn.disabled = true; // 스윕 중에는 폐루프 제어를 별도로 시작할 수 없게
  personalizationBtn.textContent = "■ 중지";
  personalizationBtn.classList.add("running");
  logControl(`🧪 개인화 캘리브레이션 시작: 자극값 ${levels.join(", ")} 순서로 측정합니다`);
  showToast("🧪 개인화 캘리브레이션 시작", "ok");
}

function tickSweep(now) {
  if (!sweep) return;
  const level = sweep.levels[sweep.levelIndex];
  const elapsed = now - sweep.phaseStartedAt;

  if (sweep.phase === "hold") {
    controlState = "SWEEP_HOLD";
    controllerIntensity = level; // 화면(EMS 제어값)에 지금 실제로 뭘 주고 있는지 그대로 보여줌

    // TTL이 중간에 끊겨서 자극이 뚝뚝 끊기지 않도록 주기적으로 재전송한다
    // (기존 폐루프 제어의 하드웨어 전송 스로틀과 같은 이유).
    if (now - sweep.lastSendTime >= SWEEP_RESEND_MS) {
      sweep.lastSendTime = now;
      const ttl = clampTtl(SWEEP_RESEND_MS + 600);
      serialLink.setIntensity(1, clampHardware(level), ttl).catch(() => {});
    }

    // 유지 구간의 마지막 부분만 표본으로 채택 (앞부분은 근육이 아직 반응
    // 중인 과도기라 버린다).
    if (elapsed >= SWEEP_HOLD_MS - SWEEP_SAMPLE_MS && currentAverage !== null) {
      sweep.samples.push(currentAverage);
    }

    const remain = Math.max(0, (SWEEP_HOLD_MS - elapsed) / 1000);
    personalizationStatusLine.textContent =
      `측정 중: 자극값 ${level} (${sweep.levelIndex + 1}/${sweep.levels.length}) — ${remain.toFixed(1)}s 남음`;

    if (elapsed >= SWEEP_HOLD_MS) {
      const avg = sweep.samples.length ? sweep.samples.reduce((a, b) => a + b, 0) / sweep.samples.length : null;
      if (avg !== null) {
        sweep.collected.push({ intensity: level, percent: avg });
        logControl(`🧪 자극값 ${level} → 실제 굽힘 ${avg.toFixed(0)}% (측정 완료)`);
      } else {
        logControl(`⚠ 자극값 ${level} 측정 중 오른손이 감지되지 않아 이 단계는 버립니다`);
      }
      sweep.phase = "rest";
      sweep.phaseStartedAt = now;
      sweep.samples = [];
      controllerIntensity = 0;
      serialLink.setIntensity(1, 0, 500).catch(() => {});
    }
  } else {
    controlState = "SWEEP_REST";
    controllerIntensity = 0;
    const remain = Math.max(0, (SWEEP_REST_MS - elapsed) / 1000);
    personalizationStatusLine.textContent = `휴식 중... (${remain.toFixed(1)}s)`;

    if (elapsed >= SWEEP_REST_MS) {
      sweep.levelIndex += 1;
      if (sweep.levelIndex >= sweep.levels.length) {
        finishSweep();
        return;
      }
      sweep.phase = "hold";
      sweep.phaseStartedAt = now;
      sweep.lastSendTime = 0;
    }
  }
}

function finishSweep() {
  const collected = sweep.collected;
  sweep = null;
  startControlBtn.disabled = false;
  personalizationBtn.textContent = "개인화 측정 시작";
  personalizationBtn.classList.remove("running");
  controlState = "STANDBY";
  if (serialLink) serialLink.setIntensity(1, 0, 500).catch(() => {});

  if (collected.length < 2) {
    personalizationStatusLine.textContent = "측정 실패: 유효한 데이터가 2개 미만입니다. 다시 시도해주세요.";
    logControl("🧪 개인화 캘리브레이션 실패: 유효 데이터 부족 (오른손이 자주 감지되지 않았을 수 있습니다)");
    showToast("❌ 개인화 캘리브레이션 실패 (데이터 부족)", "bad", 4000);
    return;
  }

  personalization.points = collected;
  fitPersonalizationModel();
  drawPersonalizationGraph();
  updatePersonalizationSummary();
  personalizationStatusLine.textContent = "측정 완료";
  logControl(`🧪 개인화 캘리브레이션 완료: ${collected.map((p) => `${p.intensity}→${p.percent.toFixed(0)}%`).join(", ")}`);
  showToast("✅ 개인화 캘리브레이션 완료", "ok");
}

function abortSweep(reason) {
  if (!sweep) return;
  sweep = null;
  controllerIntensity = 0;
  controlState = "STANDBY";
  if (serialLink) serialLink.setIntensity(1, 0, 500).catch(() => {});
  startControlBtn.disabled = false;
  personalizationBtn.textContent = "개인화 측정 시작";
  personalizationBtn.classList.remove("running");
  personalizationStatusLine.textContent = `중지됨: ${reason}`;
  logControl(`🧪 개인화 캘리브레이션 중지: ${reason}`);
  showToast(`⏹ 개인화 캘리브레이션 중지: ${reason}`, "warn");
}

function drawPersonalizationGraph() {
  const w = graphPersonalizationCanvas.width, h = graphPersonalizationCanvas.height;
  graphPersonalizationCtx.clearRect(0, 0, w, h);
  graphPersonalizationCtx.strokeStyle = "rgba(255,255,255,0.08)";
  graphPersonalizationCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h * i) / 4;
    graphPersonalizationCtx.beginPath();
    graphPersonalizationCtx.moveTo(0, y);
    graphPersonalizationCtx.lineTo(w, y);
    graphPersonalizationCtx.stroke();
  }
  if (w === 0 || h === 0) return;

  const xScale = (v) => (Math.min(100, Math.max(0, v)) / 100) * w;
  const yScale = (v) => h - (Math.min(100, Math.max(0, v)) / 100) * h;

  if (personalization.coeffs) {
    const [slope, intercept] = personalization.coeffs;
    graphPersonalizationCtx.beginPath();
    graphPersonalizationCtx.moveTo(xScale(0), yScale(intercept));
    graphPersonalizationCtx.lineTo(xScale(100), yScale(slope * 100 + intercept));
    graphPersonalizationCtx.strokeStyle = "#6ee7b7";
    graphPersonalizationCtx.lineWidth = 2;
    graphPersonalizationCtx.stroke();
  }

  graphPersonalizationCtx.fillStyle = "#ffb454";
  for (const p of personalization.points) {
    graphPersonalizationCtx.beginPath();
    graphPersonalizationCtx.arc(xScale(p.intensity), yScale(p.percent), 4, 0, Math.PI * 2);
    graphPersonalizationCtx.fill();
  }
}

function updatePersonalizationSummary() {
  if (!personalization.coeffs || personalization.points.length < 2) {
    personalizationSummary.textContent = "아직 측정된 데이터가 없습니다. 위 버튼으로 개인화 측정을 시작해보세요.";
    personalizationBadge.textContent = "측정 안 됨";
    personalizationBadge.style.color = "";
    personalizationBadge.style.background = "";
    return;
  }
  const [slope, intercept] = personalization.coeffs;
  const at100 = Math.max(0, Math.min(100, slope * 100 + intercept));
  personalizationSummary.innerHTML =
    `자극 10당 약 <b>${(slope * 10).toFixed(1)}%p</b> 반응 · 자극 100 기준 예상 굽힘 약 <b>${at100.toFixed(0)}%</b> ` +
    `(측정 ${personalization.points.length}개 지점 기반). 이제 목표를 설정하면 이 곡선을 기반으로 시작 자극값을 추천합니다.`;
  personalizationBadge.textContent = `측정됨 (${personalization.points.length}개 지점)`;
  personalizationBadge.style.color = "var(--accent-2)";
  personalizationBadge.style.background = "rgba(110,231,183,0.12)";
}

// ============================================================================
// Web Serial 링크 (serial_link.py 와 동일한 텍스트 프로토콜)
// ============================================================================

class WebSerialLink {
  constructor(baudRate) {
    this.baudRate = baudRate;
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.keepReading = false;
    this.handshakeOk = false;
    this.lastError = "";
    this.heartbeatTimer = null;
    this.onMessage = null;
    this._buffer = "";
    this._waiter = null;
    this._chain = Promise.resolve(); // 명령이 겹치지 않도록 순서대로 실행
  }

  isConnected() {
    return this.port !== null;
  }

  async requestAndConnect(connectTimeoutMs = 2000) {
    if (!("serial" in navigator)) {
      this.lastError = "이 브라우저는 Web Serial API를 지원하지 않습니다 (Chrome/Edge 필요).";
      return false;
    }
    try {
      this.port = await navigator.serial.requestPort();
      await this.port.open({ baudRate: this.baudRate });
    } catch (err) {
      this.lastError = "포트를 열지 못했습니다: " + err.message;
      this.port = null;
      return false;
    }

    this.writer = this.port.writable.getWriter();
    this.keepReading = true;
    this._readLoop();

    // 포트를 열면 보드가 자동으로 리셋되고 재부팅하는 경우가 있는데, 그
    // 시간이 클론 보드/드라이버에 따라 꽤 걸릴 수 있다. 한 번만 PING을
    // 보내고 포기하지 않고, 부팅이 끝날 때까지 몇 번 재시도한다.
    await new Promise((r) => setTimeout(r, 2000)); // 자동 리셋 안정화 대기 (여유있게)

    let reply = null;
    for (let attempt = 0; attempt < 3 && reply === null; attempt++) {
      reply = await this.sendAndWait("PING", ["PONG"], connectTimeoutMs);
    }
    if (reply === null) {
      this.lastError = "PONG 응답이 없습니다 (핸드셰이크 실패).";
      await this.disconnect();
      return false;
    }
    this.handshakeOk = true;
    this.heartbeatTimer = setInterval(() => {
      this._writeLine("PING").catch(() => {});
    }, config.safety.heartbeatIntervalMs);
    return true;
  }

  async disconnect() {
    this.handshakeOk = false;
    this.keepReading = false;
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    try {
      if (this.reader) await this.reader.cancel();
    } catch (e) { /* ignore */ }
    try {
      if (this.writer) {
        await this.writer.close();
      }
    } catch (e) { /* ignore */ }
    try {
      if (this.port) await this.port.close();
    } catch (e) { /* ignore */ }
    this.port = null;
    this.reader = null;
    this.writer = null;
  }

  async _writeLine(text) {
    if (!this.writer) throw new Error("연결되어 있지 않음");
    await this.writer.write(new TextEncoder().encode(text + "\n"));
  }

  async _readLoop() {
    const textDecoder = new TextDecoderStream();
    this.port.readable.pipeTo(textDecoder.writable).catch(() => {});
    this.reader = textDecoder.readable.getReader();
    try {
      while (this.keepReading) {
        const { value, done } = await this.reader.read();
        if (done) break;
        if (!value) continue;
        this._buffer += value;
        let idx;
        while ((idx = this._buffer.indexOf("\n")) >= 0) {
          const line = this._buffer.slice(0, idx).trim();
          this._buffer = this._buffer.slice(idx + 1);
          if (line) this._handleLine(line);
        }
      }
    } catch (err) {
      this.lastError = "읽기 오류: " + err.message;
      triggerEmergencyStop("시리얼 읽기 오류: " + err.message);
    }
  }

  _handleLine(line) {
    if (this._waiter && this._waiter.prefixes.some((p) => line.startsWith(p))) {
      const resolve = this._waiter.resolve;
      this._waiter = null;
      resolve(line);
      return;
    }
    // 하트비트 PONG은 0.4초마다 오가는 순수 배경 신호라, 로그창에 그대로
    // 찍으면 몇 초 안에 진짜 중요한 메시지(에러 등)를 다 밀어내버린다.
    if (line.startsWith("PONG")) return;
    if (this.onMessage) this.onMessage(line);
  }

  sendAndWait(command, prefixes, timeoutMs) {
    const run = () =>
      new Promise((resolve) => {
        let settled = false;
        const timer = setTimeout(() => {
          if (!settled) {
            settled = true;
            this._waiter = null;
            resolve(null);
          }
        }, timeoutMs);
        this._waiter = {
          prefixes,
          resolve: (line) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            resolve(line);
          }
        };
        this._writeLine(command).catch(() => {
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            this._waiter = null;
            resolve(null);
          }
        });
      });
    const result = this._chain.then(run, run);
    this._chain = result.catch(() => {});
    return result;
  }

  async arm(channel) {
    const reply = await this.sendAndWait(`ARM,${channel}`, ["ARMED", "ERROR"], 1000);
    return reply !== null && reply.startsWith("ARMED");
  }
  async setIntensity(channel, intensity, durationMs) {
    intensity = Math.max(0, Math.min(100, Math.round(intensity)));
    durationMs = Math.max(0, Math.round(durationMs));
    const reply = await this.sendAndWait(`SET,${channel},${intensity},${durationMs}`, ["OK", "ERROR"], 1000);
    if (reply === null) {
      this.lastError = "SET 응답 없음";
      return false;
    }
    if (reply.startsWith("ERROR")) {
      this.lastError = reply;
      return false;
    }
    return true;
  }
  async stop(channel) {
    const reply = await this.sendAndWait(`STOP,${channel}`, ["STOPPED", "ERROR"], 1000);
    return reply !== null && reply.startsWith("STOPPED");
  }
  async stopAll() {
    const reply = await this.sendAndWait("STOP_ALL", ["STOPPED", "ERROR"], 1000);
    return reply !== null && reply.startsWith("STOPPED");
  }
}

async function connectSerial() {
  if (!serialLink) {
    serialLink = new WebSerialLink(config.serial.baudRate);
    serialLink.onMessage = (line) => logControl("Arduino: " + line);
  }
  serialStatusPill.textContent = "연결 시도 중...";
  serialStatusPill.className = "pill";
  const ok = await serialLink.requestAndConnect();
  if (ok) {
    resetAfterTrip();
    serialStatusPill.textContent = "연결됨 · PONG 확인됨";
    serialStatusPill.className = "pill ok";
    logControl("Arduino 연결 및 핸드셰이크 완료");
  } else {
    serialStatusPill.textContent = "연결 실패: " + serialLink.lastError;
    serialStatusPill.className = "pill bad";
  }
}

async function disconnectSerial() {
  if (serialLink) {
    await serialLink.stopAll().catch(() => {});
    await serialLink.disconnect();
  }
  armedCh1 = false;
  armedCh2 = false;
  lastDrivenChannel = null;
  serialStatusPill.textContent = "연결 안 됨";
  serialStatusPill.className = "pill";
}

// ============================================================================
// 화면 표시 / 로그 / 그래프 / CSV
// ============================================================================

function updateCompareDisplay() {
  targetValueDisplay.textContent = targetPercent === null ? "-" : `${targetPercent.toFixed(0)}%`;
  actualValueDisplay.textContent = currentAverage === null ? "-" : `${currentAverage.toFixed(0)}%`;

  // 표시용 오차는 target/actual(둘 다 손을 잠깐 놓쳐도 마지막 값을 유지함)로
  // 직접 계산한다. controller의 lastError는 "이번 프레임에 실제로 판단했는지"
  // 기준이라 손을 놓친 순간엔 null이 되는데, 그렇다고 위의 두 숫자가 "-"로
  // 바뀌는 것도 아니라서 그것만 따르면 "73% / 30%인데 오차는 -"처럼 보인다.
  const displayError = targetPercent !== null && currentAverage !== null ? targetPercent - currentAverage : null;

  if (displayError === null) {
    errorValueDisplay.textContent = "-";
    errorValueDisplay.style.color = "var(--text-dim)";
  } else {
    const sign = displayError > 0 ? "+" : "";
    errorValueDisplay.textContent = `${sign}${displayError.toFixed(0)}%`;
    errorValueDisplay.style.color = Math.abs(displayError) <= config.control.tolerancePercent ? "var(--accent-2)" : "var(--warn)";
  }
}

function updateElapsedDisplay() {
  if (experimentStart === null) {
    elapsedDisplay.textContent = "0s";
    experimentTimerPill.textContent = "타이머 시작 전";
    experimentTimerPill.className = "pill";
    return;
  }
  const elapsedS = (performance.now() - experimentStart) / 1000;
  elapsedDisplay.textContent = `${Math.floor(elapsedS)}s`;
  const remainingS = Math.max(0, config.safety.totalExperimentSeconds - elapsedS);
  if (remainingS <= 0) {
    experimentTimerPill.textContent = "0s 남음 -- 초과됨! 모든 모드 즉시 정지됨";
    experimentTimerPill.className = "pill bad";
  } else if (remainingS <= 30) {
    experimentTimerPill.textContent = `${Math.ceil(remainingS)}s 남음`;
    experimentTimerPill.className = "pill warn";
  } else {
    experimentTimerPill.textContent = `${Math.ceil(remainingS)}s 남음`;
    experimentTimerPill.className = "pill";
  }
}

function logControl(message) {
  const line = document.createElement("div");
  line.textContent = message;
  controlLog.appendChild(line);
  controlLog.scrollTop = controlLog.scrollHeight;
  while (controlLog.childElementCount > 200) controlLog.removeChild(controlLog.firstChild);
}
// 캘리브레이션/캡처 진행률처럼 자주 갱신되는 줄은 로그를 도배하지 않도록
// 마지막 줄을 덮어쓴다.
let progressLineEl = null;
function logControlProgress(message) {
  if (!progressLineEl || progressLineEl.parentElement !== controlLog) {
    progressLineEl = document.createElement("div");
    progressLineEl.style.color = "var(--accent)";
    controlLog.appendChild(progressLineEl);
  }
  progressLineEl.textContent = message;
  controlLog.scrollTop = controlLog.scrollHeight;
}

function updateLogCount() {
  logCountText.textContent = logRows.length;
}

function resizeGraphs() {
  for (const c of [graphPercentCanvas, graphIntensityCanvas, graphPersonalizationCanvas]) {
    c.width = c.clientWidth;
    c.height = c.clientHeight;
  }
}

function drawGraphs() {
  const recent = logRows.slice(-400);
  drawLineGraph(graphPercentCtx, recent, [
    { key: "target", color: "#4da6ff" },
    { key: "current", color: "#ff6b6b" }
  ]);
  drawLineGraph(graphIntensityCtx, recent, [{ key: "intensity", color: "#8e44ad" }]);
  drawPersonalizationGraph();
}

function drawLineGraph(gfxCtx, rows, series) {
  const w = gfxCtx.canvas.width, h = gfxCtx.canvas.height;
  gfxCtx.clearRect(0, 0, w, h);
  gfxCtx.strokeStyle = "rgba(255,255,255,0.08)";
  gfxCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h * i) / 4;
    gfxCtx.beginPath();
    gfxCtx.moveTo(0, y);
    gfxCtx.lineTo(w, y);
    gfxCtx.stroke();
  }
  if (rows.length < 2) return;

  const t0 = rows[0].t, t1 = rows[rows.length - 1].t;
  const span = Math.max(1, t1 - t0);
  const xScale = (t) => ((t - t0) / span) * w;
  const yScale = (v) => h - (Math.min(100, Math.max(0, v)) / 100) * h;

  for (const s of series) {
    gfxCtx.beginPath();
    let started = false;
    for (const row of rows) {
      const v = row[s.key];
      if (v === null || v === undefined) {
        started = false;
        continue;
      }
      const x = xScale(row.t), y = yScale(v);
      if (!started) {
        gfxCtx.moveTo(x, y);
        started = true;
      } else {
        gfxCtx.lineTo(x, y);
      }
    }
    gfxCtx.strokeStyle = s.color;
    gfxCtx.lineWidth = 2;
    gfxCtx.stroke();
  }
}

function downloadCsv() {
  const header = "timestamp_ms,target_percent,current_percent,ems_intensity,error,control_state,success,hand_detected\n";
  const body = logRows
    .map((r) =>
      [
        r.t.toFixed(1),
        r.target === null ? "" : r.target.toFixed(2),
        r.current === null ? "" : r.current.toFixed(2),
        r.intensity,
        r.error === null ? "" : r.error.toFixed(2),
        r.state,
        r.success ? 1 : 0,
        r.handDetected ? 1 : 0
      ].join(",")
    )
    .join("\n");
  const blob = new Blob([header + body], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ems_web_session_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
