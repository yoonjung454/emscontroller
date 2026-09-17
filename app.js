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
// 같은 이유(페이지 초기화 시점에 바로 켜는 코드가 아래쪽 함수보다 먼저 이
// 값들을 참조함)로 이 넷도 여기 맨 위에 둔다 -- 탱글이 음성 비서 상태.
const VOICE_AWAKE_WINDOW_MS = 10000;
let voiceEnabled = false;          // 마이크 버튼/자동 시작으로 켠 상태 -- 켜져 있으면 onend에서 계속 재시작해서 "항상 듣는 중"이 됨
let voiceAwakeUntil = 0;           // performance.now() 기준, 이 시각까지는 이름을 부른 것으로 치고 명령을 받아들임
let voicePendingBicepReps = false; // "이두운동해줘" 듣고 몇 회인지 되묻는 중인지
let voicePendingSpoonLiftConfirm = false; // 밥/수저 관련 말 듣고 "실행할까요?" 되물은 뒤 대답을 기다리는 중인지
let voiceSessionActive = false;    // 지금 브라우저 인식 세션이 실제로 살아있는지(onstart~onend 사이) -- watchdog이 이걸로 죽었는지 판단

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
  REDUCING: "세기를 낮추는 중 (목표보다 더 구부러짐)",
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

// (예전엔 여기 "행동 버튼" ACTIONS 배열이 있었다 -- 컵 잡기/따르기 동작/흔들기
// 동작/팔 굽히기/물건 놓기 등 미리 정의된 목표 동작 버튼. 전부 실제 측정
// 데이터가 아닌 PLACEHOLDER 수치였고, 거울 모드는 실시간 왼손 연동 하나로
// 충분하다고 판단해 제거했다. 필요해지면 이전 대화 기록에서 복구 가능.)

// 제어값 갱신 주기(config.control.controlPeriodMs, 기본 6초)를 이 배수로 나눠서
// 더 빠르게 반응하게 한다.
const CONTROL_SPEED_MULTIPLIER = 2.5;
function getControlPeriodMs() {
  return config.control.controlPeriodMs / CONTROL_SPEED_MULTIPLIER;
}

// 켜져 있으면 매 프레임 왼손(SOURCE)의 현재 굽힘값을 그대로 목표로 흘려보낸다
// (한 번 얼리는 스냅샷이 아니라 계속 갱신됨). 왼손엔 EMS 패치가 없고 카메라로만
// 측정되고, 실제 전기자극은 오른손(ACTUAL)에만 나간다.
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
let testResendInterval = null; // 방향키를 누르고 있는 동안 재전송을 계속 돌리는 타이머 -- 아래 testModeKeyDown 참고

// 행동 보조 모드 상태 -- "물따르기"/"이두운동" 중 하나만 동시에 실행 가능.
// 거울 모드의 activeChannel/controllerIntensity(채널 1개만 표현 가능한 구조)와
// 달리, 여기는 채널1/채널2를 동시에 서로 다른 값으로 켤 수 있어야 해서
// (예: 물따르기가 두 근육을 동시에 써야 할 수 있음) 별도의 독립적인 상태로 관리한다.
let runningActionKey = null; // null | "spoonLift" | "bicep"
let actionModeInterval = null;

// (예전엔 여기 개인화 측정(목표 도달 세기 자동 탐색) 상태가 있었다 --
// 손/팔꿈치 목표%를 자동으로 찾아 세기를 기록해주던 기능. 필요 없다는
// 요청으로 완전히 제거함. 실행(spoonLift/bicep)은 그냥 매번 0부터
// 자동으로 세기를 찾는다.)

const DEFAULT_CONFIG = {
  presets: { light: 30, half: 60, strong: 90 },
  control: {
    // 오르는 속도(kpUp/maxStepUp)와 내리는 속도(kpDown/maxStepDown)를 동일하게
    // 맞췄다. 중간값(0.017/0.85) -> 3배(0.051/2.55) -> 거기서 다시 2배(0.102/5.1)
    // -> 팔 인식 모드에서 "큰 폭으로 곱해서 뛰는 것처럼 보인다"는 피드백을 받고,
    // maxStep을 1로 낮춰서 진짜 "1씩 점진적으로" 올라가게 바꾸고, 그만큼 줄어든
    // 한 스텝당 양을 보충하려고 주기(controlPeriodMs)도 6000 -> 2250(900ms마다
    // 1씩)으로 줄였다 -- 이때는 "숫자가 부드럽게 1씩 오르되 목표 도달 시간은
    // 대략 2배로 느려짐"이 목적이었음.
    // -> 수저 들기 보조에서 "채널2(팔꿈치) 올라가는 속도가 너무 느리다"는
    // 피드백을 받고, maxStep(1씩 점진적)은 그대로 두고 주기만 3배 더 줄여서
    // (900ms -> 300ms) 부드러운 느낌은 유지한 채 전체 속도만 3배로 올림.
    kpUp: 0.102,
    kpDown: 0.102,
    tolerancePercent: 5,
    successHoldSeconds: 1.5,
    maxStepUp: 1,
    maxStepDown: 1,
    controlPeriodMs: 750, // getControlPeriodMs() = 750/2.5 = 300ms마다 최대 1씩 (900ms의 1/3)
    // ⚠ 버그 수정: "타겟이 액추얼보다 낮아졌는데 전기가 세진다"는 피드백.
    // stepFlexOnlyAxis(팔 인식/행동 보조 모드, 단일 채널 flex-only 축)가 내려갈
    // 때도 올라갈 때와 똑같이 kpDown/maxStepDown(=updateController 거울모드의
    // "채널2 램프업 게인"과 공용)을 300ms 주기로만 써서, 실시간으로 계속
    // 바뀌는 목표가 훅 내려가도 한동안 "액추얼이 목표보다 훨씬 높은" 채로
    // 남아 전기가 계속/더 세지는 것처럼 보였다. kpDown/maxStepDown은
    // updateController의 신전 채널 게인이라 건드리면 거울모드까지 영향받으므로
    // 손대지 않고, stepFlexOnlyAxis 전용으로 내려가는 속도만 따로 크게 뒀다 --
    // 세기를 낮추는 건 안전 문제가 없으니 올릴 때처럼 조심스러울 필요가 없다.
    reduceKp: 0.6,
    reduceMaxStep: 6
  },
  safety: {
    maxIntensity: 0,
    minIntensity: 0,
    heartbeatIntervalMs: 400,
    commandTtlMs: 1500,
    maxCommandTtlMs: 5000,
    // 목표에 도달해 LOCKED로 계속 자극을 유지하는 상태(팔 인식 모드 등)가 이
    // 제한에 그대로 걸려 "목표 도달하고 몇 초 뒤 혼자 꺼짐"으로 보였다 --
    // 가정용 저전류 기기라는 전제로 10초 -> 60초로 늘림 (③ 안전 설정 카드에서
    // 언제든 사람이 더 조정 가능).
    maxContinuousStimSeconds: 60,
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
let invertArmSides = false; // POSE_ARM_A가 반대로(actual로) 인식되면 체크박스로 뒤집기 (invertHandedness와 동일한 개념)

// 팔 인식 모드 전용 -- 손(채널1)/팔꿈치(채널2)를 동시에, 각각 독립적으로
// 제어한다. 채널이 2개뿐이라 손/팔꿈치 각각 "펴는" 채널까지는 못 만들고,
// 둘 다 "목표만큼 부족하면 조심스럽게 증가, 도달했으면 그 세기로 유지"만
// 한다 (펴는 건 본인이 힘을 빼거나 중력에 맡김 -- 일부 EMS 보조기기가
// 실제로 이렇게 그립/보조 전용으로만 동작한다).
let armHandCtrl = { intensity: 0, state: "STANDBY", successSince: null, lastStepTime: 0 };
let armElbowCtrl = { intensity: 0, state: "STANDBY", successSince: null, lastStepTime: 0 };
// 세기 "계산"은 getControlPeriodMs()(2400ms) 주기로 하지만, 그 값을 하드웨어로
// "재전송"하는 건 훨씬 자주 해야 한다 -- SET의 TTL(기본 1500ms)보다 재전송
// 간격이 길면 다음 전송이 오기 전에 아두이노가 자동으로 꺼버려서 자극이
// 끊겼다 이어졌다 한다. TTL보다 충분히 짧게 잡는다.
const ARM_HARDWARE_RESEND_MS = 600;

// 행동 보조 모드(수저 들기 보조/이두운동)도 같은 stepFlexOnlyAxis 엔진을
// 재사용하므로, 팔 인식 모드와 똑같은 모양의 축 상태를 따로 둔다 (동시에
// 두 모드가 같이 돌 일은 없지만, 상태가 서로 섞이지 않도록 분리).
let actionHandCtrl = { intensity: 0, state: "STANDBY", successSince: null, lastStepTime: 0 };
let actionElbowCtrl = { intensity: 0, state: "STANDBY", successSince: null, lastStepTime: 0 };

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
let sessionStartedAt = null; // 카메라를 처음 켠 시각 -- "경과 시간" 표시용일 뿐, 강제 정지 안 함
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

// ---- 도달 시행(trial) 기록 -- 거울 모드/팔 인식 모드가 공유하는 controlState를
// 보고 "목표에 도달했다"(LOCKED)를 한 번씩 셀 때마다 기록한다. 12초 안에
// 도달 못하면 그 시도는 실패로 기록하고 바로 다음 시도를 재는 걸 시작한다 --
// CSV로 내려받으면 "시행 번호/성공·실패/도달 시간"이 바로 나와서, 매 tick
// 원본 로그(logRows)보다 사람이 보기 훨씬 편하다.
const TRIAL_TIMEOUT_MS = 12000;
let trialAttemptStart = null; // 지금 시도를 재기 시작한 시각 (performance.now()) -- null이면 관찰 안 하는 중
let trialLog = []; // { n, result: "성공"|"실패", seconds: number|null, target, actual, time }

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
const armCalCard = document.getElementById("armCalCard");
const mirrorArmControlPanel = document.getElementById("mirrorArmControlPanel");
const testModeCard = document.getElementById("testModeCard");

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
const armHandIntensityDisplay = document.getElementById("armHandIntensityDisplay");
const armHandStateText = document.getElementById("armHandStateText");
const invertArmSidesCheckbox = document.getElementById("invertArmSidesCheckbox");

// 테스트 모드
const testIntensityCh1Input = document.getElementById("testIntensityCh1Input");
const testIntensityCh2Input = document.getElementById("testIntensityCh2Input");
const testModeStatusText = document.getElementById("testModeStatusText");

// 행동 보조 모드 (목표 % 자동 세팅 폐루프)
const spoonLiftHandTargetInput = document.getElementById("spoonLiftHandTargetInput");
const spoonLiftElbowTargetInput = document.getElementById("spoonLiftElbowTargetInput");
const spoonLiftBtn = document.getElementById("spoonLiftBtn");
const bicepHandTargetInput = document.getElementById("bicepHandTargetInput");
const bicepElbowCurlTargetInput = document.getElementById("bicepElbowCurlTargetInput");
const bicepElbowReleaseTargetInput = document.getElementById("bicepElbowReleaseTargetInput");
const bicepRepsInput = document.getElementById("bicepRepsInput");
const bicepBtn = document.getElementById("bicepBtn");
// 수저 들기 보조/이두운동이 동시에 돌 수 없어서(runningActionKey 하나뿐),
// 예전엔 각자 따로 TARGET/ACTUAL/오차/세기 칸을 4벌 갖고 있었는데 화면에
// 항상 4개가 같이 보여서 헷갈렸다 -- "손"/"팔꿈치" 딱 2벌만 공용으로 두고
// 지금 실행 중인 쪽(spoonLiftTick 또는 bicepClosedLoopTick)이 그때그때
// 채워쓰도록 통합.
const actionHandTargetDisplay = document.getElementById("actionHandTargetDisplay");
const actionHandActualDisplay = document.getElementById("actionHandActualDisplay");
const actionHandErrorDisplay = document.getElementById("actionHandErrorDisplay");
const actionHandIntensityDisplay = document.getElementById("actionHandIntensityDisplay");
const actionHandStateText = document.getElementById("actionHandStateText");
const actionElbowTargetDisplay = document.getElementById("actionElbowTargetDisplay");
const actionElbowActualDisplay = document.getElementById("actionElbowActualDisplay");
const actionElbowErrorDisplay = document.getElementById("actionElbowErrorDisplay");
const actionElbowIntensityDisplay = document.getElementById("actionElbowIntensityDisplay");
const actionElbowStateText = document.getElementById("actionElbowStateText");
const actionModeStatusText = document.getElementById("actionModeStatusText");

// 행동 보조 모드 설정 저장 버튼 (수저 들기 보조/이두운동 각자)
const spoonLiftSaveBtn = document.getElementById("spoonLiftSaveBtn");
const bicepSaveBtn = document.getElementById("bicepSaveBtn");
const voiceCommandBtn = document.getElementById("voiceCommandBtn");
const voiceCommandStatusText = document.getElementById("voiceCommandStatusText");
const voiceListenIndicator = document.getElementById("voiceListenIndicator");
const voiceListenIndicatorText = document.getElementById("voiceListenIndicatorText");
const voiceAiApiKeyInput = document.getElementById("voiceAiApiKeyInput");

const maxContinuousInput = document.getElementById("maxContinuousInput");
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
const trialCountText = document.getElementById("trialCountText");
const trialSuccessCountText = document.getElementById("trialSuccessCountText");
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
function showToast(message, kind = "ok", durationMs = 2800, big = false) {
  toastEl.textContent = message;
  toastEl.className = "toast show" + (kind === "warn" ? " warn" : kind === "bad" ? " bad" : "") + (big ? " big" : "");
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
// buildActionButtons(); -- 거울 모드의 미리 정의된 행동 버튼(placeholder) 제거로 비활성화
initVoiceCommand();
initSafetyUi();
initPresetLabels();
applyStoredActionSettings(); // 저장된 수저 들기 보조/이두운동 설정을 페이지 열자마자 채움

// 음성 비서 "탱글이" -- 마이크 버튼을 누르지 않아도 페이지를 열면 바로 항상
// 듣는 상태가 되게 자동으로 켠다(버튼은 끄고 싶을 때 쓰는 용도로 남겨둠).
// 브라우저가 마이크 권한을 아직 안 물어봤으면 여기서 권한 팝업이 뜬다.
if (SpeechRecognitionCtor) toggleVoiceCommand();
// "탱글아"로 깨어난 뒤 10초가 지나면 오른쪽 위 표시가 저절로 "듣고 있음"으로
// 돌아와야 하는데, 새로 말을 걸지 않으면 그 갱신을 트리거할 이벤트가 없어서
// 주기적으로(0.5초마다) 다시 그려준다.
setInterval(() => { watchVoiceSession(); updateVoiceIndicator(); }, 500);
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
// 카메라가 좌우로 뒤집혀 보이는 원인은 손/팔이 따로가 아니라 카메라 자체 하나라,
// 손 체크박스를 누르면 팔 쪽도 같이 뒤집히고(반대도 마찬가지) 두 체크박스가 항상
// 같은 상태를 보여주게 동기화한다 -- 매번 따로 체크 안 해도 됨.
invertHandsCheckbox.addEventListener("change", () => {
  invertHandedness = invertHandsCheckbox.checked;
  invertArmSides = invertHandsCheckbox.checked;
  invertArmSidesCheckbox.checked = invertHandsCheckbox.checked;
});
invertArmSidesCheckbox.addEventListener("change", () => {
  invertArmSides = invertArmSidesCheckbox.checked;
  invertHandedness = invertArmSidesCheckbox.checked;
  invertHandsCheckbox.checked = invertArmSidesCheckbox.checked;
});
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
  delete calibration.flat.ArmSource;
  delete calibration.bent.ArmSource;
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

spoonLiftBtn.addEventListener("click", () => startSpoonLiftClosedLoop());
bicepBtn.addEventListener("click", () => startBicepClosedLoop());

spoonLiftSaveBtn.addEventListener("click", () => saveActionInputs("spoonLift", spoonLiftSaveBtn));
bicepSaveBtn.addEventListener("click", () => saveActionInputs("bicep", bicepSaveBtn));
voiceCommandBtn.addEventListener("click", toggleVoiceCommand);
voiceAiApiKeyInput.value = aiConfig.apiKey;
voiceAiApiKeyInput.addEventListener("change", () => {
  aiConfig.apiKey = voiceAiApiKeyInput.value.trim();
  saveAiConfig();
  logControl(aiConfig.apiKey ? "🤖 음성 비서 AI API 키 저장됨" : "🤖 음성 비서 AI API 키 비움 -- 정해진 문구만 인식");
});

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
  trialLog = [];
  trialAttemptStart = null;
  updateTrialCountUI();
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
  // 행동 보조 모드는 카메라(isRunning)와 무관하게 setInterval로 계속 돌 수
  // 있으므로 이 상태도 같이 확인한다. 테스트 모드는 방향키를 누르고 있는
  // 동안에만 나가므로 keyup 대신 즉시 채널을 끈다(전체 비상정지까지는 안 함).
  if (document.hidden && (isRunning || runningActionKey)) {
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
        // control(kpUp/kpDown/maxStepUp/maxStepDown/tolerancePercent/...)은
        // 화면에 이 값을 바꾸는 입력칸이 하나도 없다 -- 전부 코드에서만 정하는
        // 값이다. 그런데도 예전엔 저장된 localStorage 값을 그대로 덮어써서,
        // 코드에서 속도(kpUp/kpDown 등) 기본값을 나중에 고쳐도 이미 한 번
        // saveConfig()가 호출된 브라우저에서는 옛날 값이 계속 남아있는 문제가
        // 있었다 (실제로 이 문제로 속도 조정이 안 먹혔음). 그래서 control은
        // 저장된 값을 무시하고 항상 최신 DEFAULT_CONFIG.control을 그대로 쓴다.
        control: { ...DEFAULT_CONFIG.control },
        safety: (() => {
          const merged = { ...DEFAULT_CONFIG.safety, ...(parsed.safety || {}) };
          // 예전 기본값(10초)이 이미 저장돼 있으면, 코드에서 기본값을 60초로
          // 올려도 저장된 값이 우선(병합)돼서 "목표 도달 후 몇 초 뒤 혼자 꺼짐"이
          // 그대로 재현된다. 10초 이하로 저장돼 있으면 예전 기본값을 그대로
          // 들고 있던 것으로 보고 새 기본값(60초)으로 올려준다 -- 사람이 직접
          // 10초보다 더 길게 늘려둔 값은 그대로 유지된다.
          if (merged.maxContinuousStimSeconds <= 10) merged.maxContinuousStimSeconds = 60;
          return merged;
        })(),
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
      // ⚠ 정확도 개선: "lite" 모델은 빠른 대신 관절 각도가 좀 부정확했다 --
      // 팔 인식이 급하다는 요청으로 더 정확한 "full" 모델로 교체(속도는 lite
      // 보다 느리지만 단일 인물·VIDEO 모드에서는 대부분 하드웨어에서 충분히
      // 실시간으로 돌아간다). minPoseDetectionConfidence 등도 손 인식(0.7)과
      // 맞춰서 약한 신뢰도의 흔들리는 관절을 덜 받아들이게 했다.
      poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolverPose, {
        baseOptions: { modelAssetPath: "./pose_landmarker_full.task" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.7,
        minPosePresenceConfidence: 0.7,
        minTrackingConfidence: 0.7
      });
      // ⚠ 버그 수정: "행동 보조 모드에서 이두운동 하면 초반에 팔 엑추얼 값이
      // 측정 안 됨" 피드백. createFromOptions()가 끝났다고 바로 실시간 속도로
      // 추론되는 게 아니다 -- WASM 백엔드는 첫 detectForVideo() 호출에서
      // 실제 추론 커널을 그제서야 준비(JIT)해서, "full" 모델+720p로 올린 뒤로
      // 그 첫 호출 하나가 몇 초씩 걸렸다. 이 워밍업 비용이 하필 운동을 막
      // 시작한 첫 몇 프레임에 그대로 나가서 "초반엔 안 잡히다가 나중에서야
      // 잡히는" 것처럼 보였다. 준비 단계(로딩 토스트가 떠 있는 동안)에 아무
      // 화면도 없는 빈 캔버스로 한 번 미리 호출해 그 비용을 여기서 대신
      // 치르게 하면, 실제 운동이 시작될 땐 이미 워밍업이 끝나 있다.
      try {
        const warmupCanvas = document.createElement("canvas");
        warmupCanvas.width = 1280;
        warmupCanvas.height = 720;
        poseLandmarker.detectForVideo(warmupCanvas, performance.now());
      } catch (warmupErr) {
        // 워밍업은 최선을 다하는 것일 뿐 -- 실패해도 모델 자체는 정상이므로 무시.
      }
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
    // 팔 인식(Pose)이 실제로 필요한 모드(팔 인식/행동 보조)만 미리 불러온다 --
    // 테스트 모드는 위 needsPose와 같은 이유로 더 이상 Pose를 안 쓴다.
    if (appMode === "arm" || appMode === "action") await ensurePoseLandmarker().catch(() => {}); // 실패해도 카메라 자체는 켜지게 (아래 renderLoop가 poseLandmarker null이면 알아서 건너뜀)

    // ⚠ 정확도 개선: 640x480은 손엔 충분했지만 팔(어깨~손목 전체)까지 잡으려면
    // 화면에서 관절 하나하나가 차지하는 픽셀 수가 너무 적어서 각도 오차가 컸다.
    // 1280x720(HD)로 올려서 더 선명하게 잡히게 함 -- "ideal"이라 이 해상도를
    // 지원 안 하는 웹캠이면 가능한 가장 가까운 해상도로 자동으로 낮아진다.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    isRunning = true;
    sessionStartedAt = sessionStartedAt ?? performance.now();
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
    // 거울 모드만 손 인식 하나로 충분해서 Pose 모델을 끄고, 나머지(팔 인식/
    // 행동 보조) 모드는 Pose도 같이 돌린다.
    // ⚠ 버그 수정: 행동 보조 모드는 원래 "runningActionKey !== null"(실행
    // 버튼을 눌러 루틴이 실제로 돌고 있을 때)만 조건으로 걸어뒀었다. 그래서
    // 행동 보조 모드 화면에 들어가기만 하고 아직 실행을 안 누른 상태에서는
    // 팔 인식이 전혀 안 되는 것처럼 보였다 -- appMode === "action"도 무조건
    // 켜지게 바꿔서 화면에 들어가는 즉시 팔 인식이 되게 함.
    //
    // ⚠ 버그 수정(2차): 한때 "거울 모드 제외 전부 팔 인식"으로 테스트 모드도
    // 여기 포함시켰었는데, Pose 모델(특히 정확도 개선 후의 "full" 모델+고해상도)
    // 이 메인 스레드를 프레임마다 붙잡고 있는 시간이 길어져서, 카메라를 켜둔
    // 채 테스트 모드 방향키를 뗐을 때 그 "정지(세기 0)" 시리얼 명령이 밀려서
    // 늦게 나가는 바람에 "손 뗐는데도 전기가 계속 옴"이라는 안전 문제가
    // 생겼다. 테스트 모드는 "카메라 없이도 즉시 정지"가 핵심 요구사항이라
    // 팔 인식(참고용 표시일 뿐이었음)을 다시 뺐다 -- 손 스켈레톤은 원래도
    // 모드와 무관하게 항상 그려지므로 테스트 모드에서도 그대로 보인다.
    const needsPose = (appMode === "arm" || appMode === "action") && poseLandmarker;
    const poseResult = needsPose
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
  if (
    armSampling &&
    now - armSampling.startedAt > SAMPLE_TIMEOUT_MS &&
    (armSampling.counts.source < SAMPLE_TARGET || armSampling.counts.actual < SAMPLE_TARGET)
  ) {
    abortArmSampling("측정 시간이 초과되었습니다. 양팔이 잘 보이도록 하고 다시 시도해주세요.");
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

  const handTarget = latestPercent.source ? computeAverage(latestPercent.source) : null;

  if (appMode === "arm") {
    setStatusBadge(statusBadgeSource, armDetectedThisFrame.source, "source");
    setStatusBadge(statusBadgeActual, armDetectedThisFrame.actual, "actual");
    // ⑤ 카드(공용 표시)는 팔꿈치(채널2)를 대표값으로 계속 보여준다. 손은
    // 채널1 값으로 아래에서 따로(암모드카드 안) 표시한다.
    currentAverage = armLatestPercent.actual;
    targetPercent = armLatestPercent.source;

    updateCompareBoxes(armHandTargetDisplay, armHandActualDisplay, armHandErrorDisplay, handTarget, handAverage);
  } else {
    setStatusBadge(statusBadgeSource, detectedThisFrame.source, "source");
    setStatusBadge(statusBadgeActual, detectedThisFrame.actual, "actual");
    currentAverage = handAverage;
  }

  // "▶ 제어 시작"을 누르기 전까지는 캘리브레이션/캡처가 다 끝나 있어도 절대
  // 하드웨어로 아무것도 나가지 않는다. 예전에는 캘리브레이션이 완료되는
  // 순간 이미 잡혀있던 목표와 바로 비교가 시작돼서 사용자가 누른 것도 없는데
  // 바로 자극이 나가는 문제가 있었다.
  if (appMode === "arm") {
    // ---- 팔 인식 모드 전용: 손(채널1)·팔꿈치(채널2) 독립 폐루프 ----
    if (!controlEnabled) {
      armHandCtrl.intensity = 0; armHandCtrl.state = "STANDBY"; armHandCtrl.successSince = null;
      armElbowCtrl.intensity = 0; armElbowCtrl.state = "STANDBY"; armElbowCtrl.successSince = null;
    } else {
      const runtime = runtimeCheck(armLostSustained);
      if (!runtime.ok && !safetyTripped) {
        triggerEmergencyStop(runtime.reason);
      }
      // ⚠ safetyTripped는 트립되는 순간 말고 매 tick 계속 확인해야 한다 (위
      // updateController()에 남긴 것과 같은 버그 -- 여기 없으면 트립된 다음에도
      // 이 블록이 계속 stepFlexOnlyAxis를 불러 세기를 올리고 하드웨어로 내보낸다).
      if (safetyTripped) {
        armHandCtrl.intensity = 0; armHandCtrl.state = "SAFETY_STOP"; armHandCtrl.successSince = null;
        armElbowCtrl.intensity = 0; armElbowCtrl.state = "SAFETY_STOP"; armElbowCtrl.successSince = null;
      } else if (armLostSustained) {
        armHandCtrl.state = "WAITING_FOR_HAND";
        armElbowCtrl.state = "WAITING_FOR_HAND";
      } else {
        stepFlexOnlyAxis(armHandCtrl, handTarget, handAverage, now);
        stepFlexOnlyAxis(armElbowCtrl, armLatestPercent.source, armLatestPercent.actual, now);
      }
      if (armHandCtrl.intensity > 0 || armElbowCtrl.intensity > 0) notifyStimStarted();
      else notifyStimStopped();
    }

    // ⑤ 카드가 읽는 공용 변수들을 팔꿈치(채널2) 기준으로 채워서 그대로 재사용.
    activeChannel = 2;
    controllerIntensity = armElbowCtrl.intensity;
    controlState = armElbowCtrl.state;
    lastError = (armLatestPercent.source !== null && armLatestPercent.actual !== null)
      ? armLatestPercent.source - armLatestPercent.actual
      : null;
    armHandIntensityDisplay.textContent = Math.round(armHandCtrl.intensity);
    armHandStateText.textContent = STATE_LABELS[armHandCtrl.state] || armHandCtrl.state;

    // ⚠ 버그 수정: 여기서도 getControlPeriodMs()(2400ms) 간격으로만 보내고
    // 있었는데, SET의 TTL(config.safety.commandTtlMs, 기본 1500ms)이 그보다
    // 짧아서 다음 재전송이 오기 전에 아두이노가 먼저 자동으로 꺼버렸다 --
    // "1초쯤 오다가 끊기고" 하던 증상의 원인. stepFlexOnlyAxis가 세기를
    // "계산"하는 주기(getControlPeriodMs)와, 그 값을 하드웨어로 "재전송"해서
    // TTL을 계속 갱신하는 주기(ARM_HARDWARE_RESEND_MS, TTL보다 훨씬 짧음)를
    // 분리한다 -- LOCKED 상태일 때 400ms로 더 촘촘히 보내던 것과 같은 이유.
    if (controlEnabled && liveModeRequested && now - lastHardwareSendTime >= ARM_HARDWARE_RESEND_MS) {
      lastHardwareSendTime = now;
      driveArmHardwareIfNeeded(armHandCtrl.intensity, armElbowCtrl.intensity)
        .catch((err) => logControl("하드웨어 전송 오류: " + err.message));
    }
  } else if (sweep) {
    // 개인화 캘리브레이션(자극값 스윕) 진행 중 -- 목표 추종 폐루프와는 완전히
    // 별개의 열린 루프라, 이 tick 동안은 그 로직을 건너뛰고 스윕만 진행한다.
    // (스윕을 시작할 때 controlEnabled를 이미 강제로 꺼뒀으므로 아래 하드웨어
    // 전송 스로틀 블록과 충돌하지 않는다.)
    // ⚠ 버그 수정: 예전엔 "!runtime.ok && !safetyTripped"만 봐서, 이미 트립된
    // 상태인데 그 순간의 runtimeCheck 자체는 통과하면(예: 손이 다시 잡히거나
    // 시간 제한을 아직 안 넘겼으면) else로 빠져서 tickSweep()이 계속 실제
    // 자극(serialLink.setIntensity)을 내보냈다. safetyTripped를 별도로 먼저 확인.
    if (safetyTripped) {
      abortSweep("안전 정지 상태");
    } else {
      const runtime = runtimeCheck(handLostSustained);
      if (!runtime.ok) {
        triggerEmergencyStop(runtime.reason);
        abortSweep(runtime.reason);
      } else {
        tickSweep(now);
      }
    }
  } else if (!controlEnabled) {
    // ⚠ 버그 수정: 예전엔 이 분기(제어 시작 전)에서 실시간 왼손 연동 타겟을
    // 아예 갱신하지 않아서, "실시간 왼손 연동 시작"을 눌러도 "▶ 제어 시작"까지
    // 눌러야만 TARGET 값이 나왔다. 실제 전기자극(controllerIntensity)은 여전히
    // 제어 시작을 눌러야 나가는 게 맞지만, 목표값 표시 자체는 연동을 시작하는
    // 즉시 보여야 자연스러워서 여기서도 갱신한다.
    if (liveMirrorActive) {
      updateLiveMirrorTarget();
    }
    controlState = "STANDBY";
    lastError = null;
    controllerIntensity = 0;
    successSince = null;
  } else {
    // ---- 안전 확인 (컨트롤러 계산보다 먼저: 트립되면 이번 tick에서 즉시 0으로) ----
    const runtime = runtimeCheck(handLostSustained);
    if (!runtime.ok && !safetyTripped) {
      triggerEmergencyStop(runtime.reason);
    }

    // 행동 보조 모드의 실시간 왼손 연동 -- 매 프레임 목표를 왼손의 지금 값으로
    // 갱신한다 (거울 모드 캡처처럼 한 번 얼리지 않음). 컨트롤러 상태는 안 건드리고
    // targetPercent/targetPerFinger만 바꾸므로, 아래 updateController()가 평소처럼
    // 새 목표에 대해 오차/허용범위/유지 로직을 그대로 적용한다.
    if (liveMirrorActive) {
      updateLiveMirrorTarget();
    }

    // ---- 폐루프 제어 ----
    const prevState = controlState;
    updateController(handLostSustained ? null : currentAverage, now);
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
  // (팔 인식 모드는 위에서 이미 자체적으로 하드웨어 전송을 끝냈으므로 건너뛴다.)
  if (appMode !== "arm") {
    const hardwareSendIntervalMs = holdLocked ? 400 : getControlPeriodMs();
    if (controlEnabled && liveModeRequested && now - lastHardwareSendTime >= hardwareSendIntervalMs) {
      lastHardwareSendTime = now;
      driveHardwareIfNeeded(activeChannel, controllerIntensity).catch((err) => logControl("하드웨어 전송 오류: " + err.message));
    }
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
  updateTrialTracking(now);

  updateCompareDisplay();
  stimValueDisplay.textContent = Math.round(controllerIntensity);
  updateActiveChannelPill();
  updateHardwareSentHint();
  controlStateText.textContent = STATE_LABELS[controlState] || controlState;
}

function updateActiveChannelPill() {
  if (appMode === "arm") {
    // 팔 인식 모드는 손(채널1)/팔꿈치(채널2)가 항상 동시에 켜져 있고 둘 다
    // "구부림 전용"이라, 거울모드의 구부림/폄 라벨이 안 맞는다. ⑤ 카드는
    // 팔꿈치(채널2)를 대표로 보여주는 중이라는 걸 명시한다 (손은 위쪽
    // "🖐 손 굽힘" 섹션에 따로 있음).
    activeChannelPill.textContent = "채널2 (팔꿈치, 구부림 전용)";
    activeChannelPill.className = "pill";
    return;
  }
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

  // 예전엔 "화면에서 x좌표가 더 작은 쪽 = TARGET"으로 매 프레임 다시 판정했는데,
  // 팔을 굽혔다 펴면 팔꿈치 x좌표가 계속 움직여서 두 팔이 화면 가운데 근처에서
  // 겹치거나 지나갈 때마다 TARGET/ACTUAL이 순간적으로 뒤바뀌는 문제가 있었다
  // (목표가 갑자기 튀면서 채널이 계속 전환되고, 그 와중에 한쪽 신뢰도가 잠깐
  // 떨어지면 "놓침"으로 판정돼 비상정지까지 걸림). 손처럼 "이 팔(A)은 항상
  // source, 저 팔(B)은 항상 actual"로 고정하고, 반대로 인식되면
  // invertArmSides 체크박스로 뒤집는 방식으로 바꿨다 -- 움직여도 역할이 안 바뀐다.
  candidates.forEach((arm) => {
    const isArmA = arm.idx === POSE_ARM_A;
    // 손과 마찬가지로 미러링된 화면에서 Pose도 좌우가 기본적으로 뒤집혀 나와서
    // (실측 확인됨), 기본값 자체를 반전시켜뒀다 -- invertArmSidesCheckbox는
    // "한 번 더 뒤집기" 용도로 남겨둠 (카메라/환경이 바뀌어 다시 반대로 나오면 사용).
    const role = invertArmSides ? (isArmA ? "source" : "actual") : (isArmA ? "actual" : "source");
    if (armDetectedThisFrame[role]) return; // 이미 이번 프레임에 그 역할이 처리됨 (정상적으론 안 생김)
    armDetectedThisFrame[role] = true;

    drawArmSkeleton(arm.shoulderPt, arm.elbowPt, arm.wristPt, role);

    const rawBend = calculateArmBend(worldLandmarks, arm.idx.shoulder, arm.idx.elbow, arm.idx.wrist);
    armLatestSmoothed[role] = armFilters[role].push(rawBend);
    // ⚠ 버그 수정: 예전엔 오른팔(actual)만 보정하고 그 하나의 펴짐/구부림
    // 각도를 왼팔(source)에도 그대로 재사용했다 -- "두 팔의 유연성이 비슷할
    // 것"이라는 가정인데, 실제로는 카메라 각도/팔 길이 차이 등으로 같은
    // 각도만큼 구부려도 왼팔 %가 더 낮게 나오는 문제가 있었다(그리고 이 값이
    // 팔 인식 모드에서 TARGET으로 그대로 쓰이므로 단순 표시 오차가 아니라
    // 실제 목표가 틀어지는 문제였음). 이제 왼팔(source)은 별도의 "ArmSource"
    // 보정값을 쓴다 -- 아래 팔 초기값 측정에서 양팔을 동시에 재서 각자
    // 보정값을 만든다.
    armLatestPercent[role] = percentFor(role === "source" ? "ArmSource" : "Arm", armLatestSmoothed[role]);

    if (armSampling) {
      armSampling.sums[role] += rawBend;
      armSampling.counts[role] += 1;
      updateArmSamplingUI();
      if (armSampling.counts.source >= SAMPLE_TARGET && armSampling.counts.actual >= SAMPLE_TARGET) {
        finishArmSampling();
      }
    }
  });
}

// ---- 팔 초기값(펴짐/구부림) 측정 -- 왼팔/오른팔을 동시에 재서 각자 보정값을
// 따로 만든다 (위 버그 설명 참고). 캘리브레이션 자세(쫙 펴짐/최대한 구부림)는
// 양팔을 대칭으로 유지하기 자연스러워서, 사용자가 추가로 할 일은 없다 --
// 버튼 2개(①②)는 그대로다.
function startArmSampling(mode) {
  if (!isRunning || armSampling) return;
  armSampling = { mode, sums: { source: 0, actual: 0 }, counts: { source: 0, actual: 0 }, startedAt: performance.now() };
  const label = mode === "flat" ? "펴짐" : "구부림";
  logControl(`양팔 ${label} 초기값 측정 시작 -- 양팔을 그대로 유지하세요`);
  showToast(`📏 양팔 ${label} 초기값 측정 중... 양팔을 그대로 유지하세요`, "ok", 2000);
}
function updateArmSamplingUI() {
  if (!armSampling) return;
  const done = Math.min(armSampling.counts.source, armSampling.counts.actual);
  logControlProgress(`팔 초기값 측정 중... ${done}/${SAMPLE_TARGET}`);
}
function finishArmSampling() {
  const { mode, sums, counts } = armSampling;
  calibration[mode].Arm = sums.actual / counts.actual;
  calibration[mode].ArmSource = sums.source / counts.source;
  saveCalibration();
  armSampling = null;
  const label = mode === "flat" ? "펴짐" : "구부림";
  logControl(`양팔 ${label} 초기값 측정 완료`);
  showToast(`✅ 양팔 ${label} 초기값 측정 완료`, "ok");
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

// 발표용 큰 TARGET/ACTUAL/오차 3칸(compare-grid) 값을 채우는 공용 함수.
// 팔 인식 모드의 손(채널1) 섹션에서 원래 이 계산을 그대로 손코딩해뒀던 걸
// 추출한 것 -- 행동 보조 모드의 수저 들기 보조/이두운동에도 똑같이 큰 칸을
// 붙이면서 중복을 없앴다.
function updateCompareBoxes(targetEl, actualEl, errorEl, target, actual) {
  targetEl.textContent = target === null || target === undefined ? "-" : `${target.toFixed(0)}%`;
  actualEl.textContent = actual === null || actual === undefined ? "-" : `${actual.toFixed(0)}%`;
  if (target === null || target === undefined || actual === null || actual === undefined) {
    errorEl.textContent = "-";
    errorEl.style.color = "var(--text-dim)";
  } else {
    const error = target - actual;
    errorEl.textContent = `${error > 0 ? "+" : ""}${error.toFixed(0)}%`;
    errorEl.style.color = Math.abs(error) <= config.control.tolerancePercent ? "var(--accent-2)" : "var(--warn)";
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

// (예전엔 여기 buildActionButtons()/selectAction()이 있었다 -- "컵 잡기/따르기
// 동작/흔들기 동작/팔 굽히기/물건 놓기" 같은 미리 정의된(placeholder 목표값)
// 행동 버튼들. 실제 측정 데이터가 아니라 전부 임시 수치였고 거울 모드는
// 실시간 왼손 연동만 쓰기로 하면서 제거했다. 필요해지면 이전 대화 기록의
// ACTIONS 배열/이 두 함수로 복구 가능.)

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
  selectedActionText.textContent = "실시간 왼손 연동 중";
  actionDescriptionText.textContent = "왼손의 지금 굽힘 정도가 계속 오른손 목표로 흘러갑니다.";
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
  selectedActionText.textContent = "대기 중";
  actionDescriptionText.textContent = "실시간 왼손 연동을 시작하면 여기에 표시됩니다.";
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
  if (armSampling) armSampling = null; // 팔 초기값 측정 중이었으면 중단 (알림 없이 조용히 취소)
  testKeyChannel = null; // 테스트 모드에서 나가면서 방향키가 눌린 채로 남아있지 않게 확실히 정리
  if (testResendInterval) { clearInterval(testResendInterval); testResendInterval = null; }
  if (controlEnabled) {
    controlEnabled = false;
    resetStartControlButton();
  }
  forceZeroController("모드 전환");
  // ⚠ 버그 수정: forceZeroController가 정리 목적으로 controlState를 항상
  // "SAFETY_STOP"("안전 정지 상태")로 남겨두는데, 거울/팔인식/행동보조 모드는
  // 다음 tick에서 자기 상태로 바로 덮어써서 문제가 안 되지만, 테스트 모드는
  // controlState를 아예 안 건드리는 모드라 이 값이 그대로 남아 "안전 정지
  // 상태"가 계속 표시됐다(실제로 안전장치가 없는 모드인데도). 테스트 모드로
  // 들어갈 땐 이 표시를 정상(대기중)으로 바로 되돌린다.
  if (mode === "test") controlState = "STANDBY";
  // 카메라가 꺼져 있으면(테스트 모드는 카메라 없이도 쓰는 게 정상) processResult()가
  // 안 돌아서 위에서 바꾼 controlState가 화면 글자에 반영이 안 되므로 직접 갱신.
  controlStateText.textContent = STATE_LABELS[controlState] || controlState;
  if (serialLink) serialLink.stopAll().catch(() => {});
  armedCh1 = false;
  armedCh2 = false;
  lastDrivenChannel = null;

  appMode = mode;
  mirrorModeCard.style.display = mode === "mirror" ? "" : "none";
  actionModeCard.style.display = mode === "action" ? "" : "none";
  armModeCard.style.display = mode === "arm" ? "" : "none";
  testModeCard.style.display = mode === "test" ? "" : "none";
  // 팔 초기값 측정 카드는 ② 카드 바로 다음(③ 안전 설정보다 앞)으로 옮겨졌다 --
  // 팔 인식 모드뿐 아니라 행동 보조 모드도 이 값이 필요하므로(actionCalibrationReady)
  // 두 모드 모두에서 보이게 한다.
  armCalCard.style.display = (mode === "arm" || mode === "action") ? "" : "none";
  // 행동 보조 모드는 ⑤ 카드의 공용 제어(controlEnabled) 플로우를 안 써서 이
  // 패널이 항상 대기중/0으로 고정돼 있다 -- 위에 따로 생긴 손/팔꿈치 칸과
  // 헷갈리지 않게 행동 보조 모드일 땐 숨긴다(기록 로그는 계속 보여줌).
  mirrorArmControlPanel.style.display = mode === "action" ? "none" : "";
  modeMirrorBtn.classList.toggle("selected", mode === "mirror");
  modeActionBtn.classList.toggle("selected", mode === "action");
  modeArmBtn.classList.toggle("selected", mode === "arm");
  modeTestBtn.classList.toggle("selected", mode === "test");

  modeDescriptionText.innerHTML =
    mode === "mirror"
      ? "<b>거울 모드</b>: 행동 버튼(카메라로 오차를 계속 보정) 또는 실시간 왼손 연동으로 오른손을 목표에 맞춥니다."
      : mode === "action"
      ? "<b>행동 보조 모드</b>: 손(채널1)·팔꿈치(채널2) 목표 %만 입력하면, 카메라로 실제 굽힘을 측정해가며 자극 세기를 자동으로 찾아 목표에 맞춥니다 (자동 세팅 폐루프, 사람마다 다른 반응에 자동으로 맞춰짐)."
      : mode === "arm"
      ? "<b>팔 인식 모드</b>: 왼팔의 팔꿈치 굽힘 정도를 실시간으로 오른팔 목표로 흘려보내고, 카메라로 측정한 오른팔의 실제 굽힘에 맞춰 자극 세기를 자동 조절합니다 (팔 버전 거울 모드)."
      : "<b>테스트 모드</b>: 캘리브레이션 없이, ←(채널1)/→(채널2) 방향키를 누르고 있는 동안만 그 채널에 고정 세기로 자극을 내보냅니다 (하드웨어 연결을 빠르게 확인할 때 사용). 카메라를 켜두면 손 인식 결과가 참고용으로 함께 표시되지만, 방향키 자극 자체는 카메라 없이도 동작합니다. (팔 인식은 이 모드에서는 안 돌립니다 -- 방향키를 뗐을 때 전류가 바로 안 끊기는 문제가 있어서 뺐습니다.)";

  targetCompareLabel.textContent = mode === "arm" ? "TARGET (왼팔 · 실시간 연동)" : "TARGET (행동 선택 / 실시간 왼손 연동)";
  actualCompareLabel.textContent = mode === "arm" ? "ACTUAL (오른팔 · 팔꿈치 굽힘)" : "ACTUAL (오른손 · 4손가락 평균)";

  // 팔 인식이 필요한 모드(팔 인식/행동 보조)는 아직 모델을 안 불러왔을 수
  // 있으니(perf를 위해 지연 로딩) 이 시점에 미리 불러오기 시작 -- 카메라
  // 실행 버튼을 누르기 전에 미리 받아둔다. 테스트 모드는 이제 안 씀(위
  // needsPose 주석 참고 -- 방향키 뗐을 때 전류가 바로 안 끊기는 문제가 있었음).
  if ((mode === "arm" || mode === "action") && !poseLandmarker) {
    showToast("🦾 팔 인식 모델을 불러오는 중입니다...", "ok", 3000);
    ensurePoseLandmarker()
      .then(() => showToast("✅ 팔 인식 모델 준비 완료", "ok"))
      .catch((err) => showToast("❌ 팔 인식 모델 로딩 실패: " + (err.message || err), "bad", 5000));
  }

  logControl(`모드 전환: ${mode === "mirror" ? "거울 모드" : mode === "action" ? "행동 보조 모드" : mode === "arm" ? "팔 인식 모드" : "테스트 모드"}`);
  updateSafetyTripUi(); // 테스트 모드 진입/이탈에 따라 안전 정지 배너 표시 여부를 다시 맞춤
}

// ============================================================================
// 행동 보조 모드 설정 저장 -- 예전엔 회원번호별로 따로 저장했는데(참가자마다
// 목표%가 다를 수 있다는 가정), 실제로는 그렇게 나눠 쓰지 않아서 회원번호
// 자체를 없앴다. 이제는 "이 컴퓨터에서 마지막으로 저장한 값 하나"만
// localStorage에 기억한다 -- 수저 들기 보조/이두운동 각자의 "저장" 버튼으로
// 저장하고, 페이지를 새로 열면 자동으로 다시 채워진다.
// ============================================================================

function loadActionSettings() {
  try {
    const raw = localStorage.getItem("emsWebActionSettings");
    if (raw) return JSON.parse(raw);
  } catch (e) { /* ignore */ }
  return {};
}
function saveActionSettings(settings) {
  localStorage.setItem("emsWebActionSettings", JSON.stringify(settings));
}

// 페이지를 열 때(또는 저장 직후) 저장된 값으로 입력칸을 채운다.
function applyStoredActionSettings() {
  const s = loadActionSettings();
  if (s.spoonLift) {
    spoonLiftHandTargetInput.value = s.spoonLift.handTarget ?? 70;
    spoonLiftElbowTargetInput.value = s.spoonLift.elbowTarget ?? 50;
  }
  if (s.bicep) {
    bicepHandTargetInput.value = s.bicep.handTarget ?? 70;
    bicepElbowCurlTargetInput.value = s.bicep.elbowCurlTarget ?? 85;
    bicepElbowReleaseTargetInput.value = s.bicep.elbowReleaseTarget ?? 20;
    bicepRepsInput.value = s.bicep.reps ?? 5;
  }
}

// key: "spoonLift" | "bicep" -- 지금 그 섹션 입력칸의 목표%(/반복 횟수)를 저장한다.
function saveActionInputs(key, btn) {
  const settings = loadActionSettings();
  if (key === "spoonLift") {
    settings.spoonLift = {
      handTarget: Math.max(0, Math.min(100, Number(spoonLiftHandTargetInput.value) || 0)),
      elbowTarget: Math.max(0, Math.min(100, Number(spoonLiftElbowTargetInput.value) || 0))
    };
  } else {
    settings.bicep = {
      handTarget: Math.max(0, Math.min(100, Number(bicepHandTargetInput.value) || 0)),
      elbowCurlTarget: Math.max(0, Math.min(100, Number(bicepElbowCurlTargetInput.value) || 0)),
      elbowReleaseTarget: Math.max(0, Math.min(100, Number(bicepElbowReleaseTargetInput.value) || 0)),
      reps: Math.max(1, Math.min(100, Number(bicepRepsInput.value) || 5))
    };
  }
  saveActionSettings(settings);
  logControl(`💾 ${ACTION_MODE_DEFS[key].label} 설정 저장`);
  showToast(`✅ ${ACTION_MODE_DEFS[key].label} 설정 저장 완료`, "ok");
  if (btn) flashButtonPress(btn, "✅ 저장됨!");
}

// ============================================================================
// 음성 비서 "탱글이" -- 마이크 버튼을 한 번 누르면 계속 듣고 있는 상태가
// 되고(매번 다시 누를 필요 없음), "탱글아"/"탱그라"라고 부르면 "네"라고
// 대답한다. 그 다음(또는 같은 문장 안에서 바로) 아래 명령 문구가 들리면
// 지금 앱이 어떤 모드에 있든 행동 보조 모드로 바꾸고 해당 동작을 실행한다.
//
// ⚠ 안전 설계: 이름을 부르지 않은 채 흘러가는 대화(예: 옆에서 그냥 "나 밥
// 먹었어"라고 한 말)에는 반응하지 않는다 -- "탱글아/탱그라"를 부른 뒤
// VOICE_AWAKE_WINDOW_MS(10초) 동안만 명령을 받아들인다("이름 불러야 반응"이
// 라는 흔한 음성비서 방식과 동일). 실제 사람 몸에 전기자극을 내보내는
// 기능이라, 아무 말에나 반응하면 안전하지 않다고 판단했다.
//
// 브라우저 내장 Web Speech API만 쓴다 (SpeechRecognition으로 듣고,
// SpeechSynthesis로 대답). 별도 서버/키 없음 -- Chrome/Edge 계열만 지원 (Web
// Serial 요구사항이랑 동일한 브라우저라 이미 맞음).
// (SpeechRecognitionCtor 자체는 initVoiceCommand()가 초기화 시점에 이미 참조하므로
// 파일 맨 위쪽 상수 선언부로 옮겨뒀다 -- 여기 남겨두면 TDZ로 인해
// "Cannot access before initialization" 오류가 난다.)

// (VOICE_AWAKE_WINDOW_MS/voiceEnabled/voiceAwakeUntil/voicePendingBicepReps는
// 페이지 로딩 시 자동으로 마이크를 켜는 코드가 이 함수들보다 먼저 실행되는
// 중에 이미 참조하므로, 파일 맨 위쪽 상수 선언부로 옮겨뒀다 -- 바로 위 문단
// 설명과 같은 TDZ 문제라 여기 남겨두면 안 됨.)

// "3회"처럼 숫자+단위를 그대로 speak()에 넘기면 브라우저 한국어 TTS가
// 숫자와 단위 사이를 어색하게 끊어 읽는 경우가 있어서, "회"(한자어 계수사)
// 앞에 오는 숫자는 한글로 직접 풀어써서 훨씬 자연스럽게 읽히게 한다
// (일회/이회/삼회처럼). 반복 횟수 범위(1~100)만 다루면 되므로 두 자리까지만
// 지원하고, 그 이상은 그냥 숫자 그대로 반환한다(안전한 폴백).
const SINO_KOREAN_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"];
function sinoKoreanNumber(n) {
  n = Math.round(n);
  if (n === 0) return "영";
  if (n < 0 || n > 99) return String(n);
  const tens = Math.floor(n / 10);
  const ones = n % 10;
  let result = "";
  if (tens > 0) result += tens === 1 ? "십" : SINO_KOREAN_DIGITS[tens] + "십";
  if (ones > 0) result += SINO_KOREAN_DIGITS[ones];
  return result;
}

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
  voiceRecognition.continuous = true; // "항상 켜진" 느낌을 위해 -- 한 번 start()하면 여러 문장을 계속 듣는다
  voiceRecognition.interimResults = false;
  voiceRecognition.maxAlternatives = 1;

  voiceRecognition.onstart = () => {
    voiceSessionActive = true;
  };

  voiceRecognition.onresult = (event) => {
    // continuous 모드에서는 event.results에 이전 것들이 계속 쌓일 수 있어서,
    // 이번에 새로 들어온 마지막 결과만 본다.
    const transcript = event.results[event.results.length - 1][0].transcript.trim();
    if (!transcript) return;
    logControl(`🎤 음성 인식: "${transcript}"`);
    handleVoiceTranscript(transcript);
  };

  voiceRecognition.onerror = (event) => {
    // "no-speech"/"aborted" 등은 계속 듣는 중에 흔히 발생하는 정상적인
    // 상황이라 무시하고 onend에서 알아서 재시작된다. 마이크 권한 문제만
    // 실제로 멈춘다(재시작해봤자 계속 같은 오류만 나므로).
    if (event.error === "not-allowed" || event.error === "audio-capture" || event.error === "service-not-allowed") {
      voiceEnabled = false;
      voiceCommandBtn.textContent = "🎤 마이크 켜기";
      voiceCommandBtn.classList.remove("running");
      voiceCommandStatusText.textContent = `⚠ 마이크 오류(${event.error}) -- 브라우저의 마이크 권한을 확인해주세요.`;
      showToast(`⚠ 마이크 오류: ${event.error}`, "warn");
    }
  };

  voiceRecognition.onend = () => {
    // ⚠ 버그 수정: 한 번 명령을 실행하고 나면(특히 speak()로 소리를 낸 뒤)
    // 그 다음부턴 "탱글아"를 불러도 반응이 없고 마이크를 껐다 켜야만 다시
    // 되던 문제 -- 브라우저가 이 세션을 조용히 끊었는데, 그 직후 바로 부르는
    // start()가 (이전 세션이 완전히 안 정리된 타이밍 등으로) 실패하면 그냥
    // 조용히 포기하고 있었다. voiceSessionActive를 꺼두고, 아래 watchdog이
    // 몇 초 안에 이걸 보고 계속 재시도하게 해서 스스로 복구되게 한다.
    voiceSessionActive = false;
    if (voiceEnabled) {
      try { voiceRecognition.start(); } catch (e) { /* 실패해도 watchdog이 계속 재시도함 */ }
    }
  };
}

// onend/onerror만으로는 완전히 못 잡는 경우(이벤트 자체가 안 오거나, onend의
// 즉시 재시작이 타이밍 문제로 실패하는 경우)에 대비한 안전망. voiceEnabled인데
// 세션이 살아있지 않은 상태가 잠깐이라도 관측되면 계속 start()를 다시
// 시도한다 -- 이미 살아있으면(voiceSessionActive) 아무것도 안 하므로, 정상
// 작동 중일 땐 그냥 조용하다.
function watchVoiceSession() {
  if (!voiceRecognition || !voiceEnabled || voiceSessionActive) return;
  try { voiceRecognition.start(); } catch (e) { /* 아직 이전 세션 정리 중일 수 있음 -- 다음 tick에 다시 시도 */ }
}

function isWakeWord(text) {
  return text.includes("탱글") || text.includes("탱그라");
}

// 숫자 표현 인식 -- Web Speech API가 보통 "3회"/"세 번" 같은 걸 숫자로 바로
// 옮겨주지만(예: "3회"), 혹시 한글로 그대로 나오는 경우("세 번")를 대비해
// 1~10 한글 표현도 같이 봐준다.
const KOR_NUMBER_WORDS = {
  한: 1, 하나: 1, 일: 1, 두: 2, 둘: 2, 이: 2, 세: 3, 셋: 3, 삼: 3, 네: 4, 넷: 4, 사: 4,
  다섯: 5, 오: 5, 여섯: 6, 육: 6, 일곱: 7, 칠: 7, 여덟: 8, 팔: 8, 아홉: 9, 구: 9, 열: 10, 십: 10
};
function parseSpokenNumber(text) {
  const digitMatch = text.match(/\d+/);
  if (digitMatch) return Math.max(1, Math.min(100, parseInt(digitMatch[0], 10)));
  for (const [word, n] of Object.entries(KOR_NUMBER_WORDS)) {
    if (text.includes(word)) return n;
  }
  return null;
}

function matchesSpoonLiftPhrase(text) {
  return ["수저", "숟가락", "밥먹고싶", "밥먹는거", "밥먹여줘"].some((kw) => text.includes(kw));
}

// "응", "네", "해줘"처럼 긍정으로 대답했는지 판단 -- 수저 들기 보조를 실제
// 실행하기 전에 한 번 더 확인받을 때 씀(밥/수저 얘기만 하고 실제로는
// 원하지 않았을 수도 있어서, 바로 실행하지 않고 되물어본다).
function isAffirmative(text) {
  return ["응", "네", "예", "그래", "좋아", "해줘", "해주세요", "실행", "시작", "오케이", "okay", "ok"].some((kw) =>
    text.includes(kw)
  );
}

// 밥/수저 관련 말이 들리면 바로 실행하지 않고 한 번 되물어본다.
function askSpoonLiftConfirmation() {
  voicePendingSpoonLiftConfirm = true;
  voiceAwakeUntil = performance.now() + VOICE_AWAKE_WINDOW_MS;
  speak("수저 들기 보조를 실행할까요?");
  voiceCommandStatusText.textContent = "🎤 수저 들기 보조 -- 실행할까요? (\"응\", \"해줘\" 등으로 대답해주세요)";
}

// 위 키워드 매칭으로 못 잡은(정해진 문구가 아닌) 발화를 AI(Gemini)에게 보내
// "수저 들기 보조/이두운동/둘 다 아님" 중 뭘 원하는지 판단시킨다. API 키가
// 없으면 아예 호출하지 않고 null을 반환 -- 호출자는 이 경우 그냥 "이해 못함"
// 으로 처리한다. interpretCommandWithAI()(옛 거울 모드 명령 해석)와 같은
// Gemini 호출 패턴(REST 직접 호출 + responseSchema로 JSON 강제)을 재사용.
async function classifyVoiceIntentWithAI(transcript) {
  if (!aiConfig.apiKey) return null;

  const systemPrompt = `당신은 재활 보조 기기의 음성 비서 "탱글이"입니다. 이 기기는 딱 두 가지
전기자극 동작을 실행할 수 있습니다.
1) 수저 들기 보조(spoon_lift) -- 손과 팔꿈치를 굽혀서 숟가락을 들어올려 밥/음식을 먹는 걸
   돕는 동작. "배고프다", "뭔가 먹고 싶다", "밥 먹여달라" 같은 표현은 전부 여기 해당합니다.
2) 이두운동(bicep) -- 팔꿈치를 굽혔다 펴는 걸 반복하는 운동. "운동하고 싶다", "팔 좀
   움직이고 싶다", "이두 운동" 같은 표현이 해당합니다.

이 둘 중 하나가 아닌 발화는, 전기자극과 무관하게 그냥 평소 대화형 AI 비서처럼 자연스럽게
대답해주면 됩니다(저녁 메뉴 추천, 잡담, 일반 상식 질문 등 뭐든 좋습니다) -- 실제로 전기
자극을 실행하지는 않고, 말로만 대답합니다. 이 reply는 그대로 음성으로 읽히니 아주 짧고
간결하게 -- 딱 핵심만 담은 한 문장으로 끝내고, "필요하신 게 있으면 말씀해주세요" 같은
뒤에 덧붙이는 인사치레/추가 제안 문장은 절대 붙이지 마세요.

사용자의 한국어 발화를 보고 반드시 아래 JSON 중 하나만 출력하세요 (설명/코드블록 금지):
{"action":"spoon_lift","message":"<한 줄 설명>","reply":""}
{"action":"bicep","message":"<한 줄 설명>","reply":""}
{"action":"chat","message":"일반 대화로 처리","reply":"<사용자에게 그대로 소리 내어 읽어줄 한국어 대답, 짧은 한 문장으로만>"}

reply는 action이 "chat"일 때만 채우고, spoon_lift/bicep일 때는 빈 문자열로 두세요.`;

  const responseSchema = {
    type: "object",
    properties: {
      action: { type: "string", enum: ["spoon_lift", "bicep", "chat"] },
      message: { type: "string" },
      reply: { type: "string" }
    },
    required: ["action", "message", "reply"]
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
        contents: [{ role: "user", parts: [{ text: transcript }] }],
        systemInstruction: { parts: [{ text: systemPrompt }] },
        generationConfig: { responseMimeType: "application/json", responseSchema, maxOutputTokens: 400 }
      })
    });

    if (!response.ok) {
      const errBody = await response.text().catch(() => "");
      logControl(`⚠ 음성 비서 AI 호출 실패 (HTTP ${response.status}): ${errBody.slice(0, 200)}`);
      return null;
    }

    const data = await response.json();
    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
    if (!raw) {
      logControl(`⚠ 음성 비서 AI 응답에 텍스트가 없음(${data.candidates?.[0]?.finishReason || "unknown"})`);
      return null;
    }
    const jsonText = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
    const parsed = JSON.parse(jsonText);
    logControl(`🤖 음성 비서 AI 응답: ${jsonText}`);
    return parsed;
  } catch (err) {
    logControl(`⚠ 음성 비서 AI 해석 오류: ${err.message}`);
    return null;
  }
}

async function startSpoonLiftFromVoice() {
  voiceCommandStatusText.textContent = "✅ \"수저 들기 보조\" 인식 -- 시작합니다";
  speak("수저 들기 보조를 시작합니다");
  if (appMode !== "action") setAppMode("action");
  await startSpoonLiftClosedLoop();
}

async function startBicepFromVoice(reps) {
  const clamped = Math.max(1, Math.min(100, reps));
  bicepRepsInput.value = clamped;
  voiceCommandStatusText.textContent = `✅ "이두운동 ${clamped}회" 인식 -- 시작합니다`;
  speak(`이두운동 ${sinoKoreanNumber(clamped)}회를 시작합니다`);
  if (appMode !== "action") setAppMode("action");
  await startBicepClosedLoop();
}

async function handleVoiceTranscript(transcript) {
  // try/finally로 감싸서, 아래 어느 return 경로로 빠지든 오른쪽 위 표시(듣고
  // 있음/깨어있음)가 이번에 바뀐 상태를 바로 반영하게 한다.
  try {
    const text = transcript.replace(/\s+/g, ""); // 띄어써도 인식되게 공백 제거하고 비교
    const now = performance.now();
    const heardWake = isWakeWord(text);
    if (heardWake) voiceAwakeUntil = now + VOICE_AWAKE_WINDOW_MS;
    const awake = now < voiceAwakeUntil;

    // ---- 수저 들기 보조 실행 여부를 되물은 직후라면, 이번 발화는 그 대답으로 취급 ----
    if (voicePendingSpoonLiftConfirm) {
      if (!awake) { voicePendingSpoonLiftConfirm = false; return; } // 너무 오래 걸림 -- 처음부터 다시
      voicePendingSpoonLiftConfirm = false;
      if (isAffirmative(text)) {
        startSpoonLiftFromVoice();
      } else {
        speak("알겠습니다");
        voiceCommandStatusText.textContent = "수저 들기 보조를 실행하지 않았습니다.";
      }
      return;
    }

    // ---- 이두운동 반복 횟수를 되묻은 직후라면, 이번 발화는 그 대답으로 취급 ----
    if (voicePendingBicepReps) {
      if (!awake) { voicePendingBicepReps = false; return; } // 너무 오래 걸림 -- 처음부터(이름부터) 다시
      const reps = parseSpokenNumber(text);
      if (reps === null) {
        voiceAwakeUntil = now + VOICE_AWAKE_WINDOW_MS; // 계속 들을 시간을 연장
        speak("몇 회인지 다시 말씀해주세요");
        voiceCommandStatusText.textContent = `❓ "${transcript}"에서 횟수를 못 찾았습니다 -- 숫자로 다시 말씀해주세요.`;
        return;
      }
      voicePendingBicepReps = false;
      startBicepFromVoice(reps);
      return;
    }

    if (!awake) return; // 이름도 안 불렀고, 부른 지 오래 지남 -- 그냥 지나가는 대화로 보고 무시

    if (matchesSpoonLiftPhrase(text)) {
      askSpoonLiftConfirmation();
      return;
    }
    if (text.includes("이두")) {
      voicePendingBicepReps = true;
      voiceAwakeUntil = now + VOICE_AWAKE_WINDOW_MS;
      speak("몇 회 하시겠어요?");
      voiceCommandStatusText.textContent = "🎤 이두운동 -- 몇 회 할지 말씀해주세요 (예: \"3회\")";
      return;
    }

    // ---- 정해진 문구로 못 잡았을 때만 AI에게 넘긴다 (API 키가 있을 때만) ----
    // 이름만 부르고 아무 내용도 없는 경우("탱글아"만)는 AI를 부를 필요가
    // 없어서, 이름을 뗀 나머지 글자가 어느 정도 있을 때만 호출한다.
    const withoutWake = text.replace(/탱그라|탱글아|탱글이|탱글/g, "");
    if (aiConfig.apiKey && withoutWake.length >= 2) {
      voiceCommandStatusText.textContent = `🤖 "${transcript}" 이해하는 중...`;
      const result = await classifyVoiceIntentWithAI(transcript);
      if (result?.action === "spoon_lift") {
        askSpoonLiftConfirmation();
        return;
      }
      if (result?.action === "bicep") {
        voicePendingBicepReps = true;
        voiceAwakeUntil = performance.now() + VOICE_AWAKE_WINDOW_MS;
        speak("몇 회 하시겠어요?");
        voiceCommandStatusText.textContent = "🎤 이두운동 -- 몇 회 할지 말씀해주세요 (예: \"3회\")";
        return;
      }
      if (result?.action === "chat" && result.reply) {
        // 전기자극 동작(수저/이두)이 아닌 그 외 모든 말은 그냥 평소 대화형
        // AI처럼 대답만 해준다 -- 아무것도 설정/실행하지 않음.
        speak(result.reply);
        voiceCommandStatusText.textContent = `💬 "${transcript}" → ${result.reply}`;
        voiceAwakeUntil = performance.now() + VOICE_AWAKE_WINDOW_MS; // 대화가 이어질 수 있게 깨어있는 시간 연장
        return;
      }
      // result === null이면(호출 자체가 실패한 것 -- Gemini 쪽 503/네트워크
      // 오류 등, classifyVoiceIntentWithAI가 이미 logControl에 자세한 원인을
      // 남겨둠) 화면 글자만으로는 사용자가 왜 반응이 없는지 알기 어려워서
      // 여기서 소리로도 알려준다.
      if (result === null) {
        speak("지금 잠깐 응답이 안 돼요. 다시 한 번 말씀해주세요.");
        voiceCommandStatusText.textContent = `⚠ AI 응답 실패 -- "${transcript}"는 정해진 문구로만 다시 말씀해주세요.`;
        return;
      }
    }

    if (heardWake) {
      // 이름만 부르고 아직 다른 명령은 없었음 -- 인사만 하고 다음 말을 기다림
      speak("네");
      voiceCommandStatusText.textContent = "✅ 네! 명령을 말씀해주세요 (예: \"수저들기 보조해줘\", \"이두운동해줘\")";
      return;
    }
    voiceCommandStatusText.textContent = `❓ "${transcript}" -- 이해하지 못했습니다.`;
  } finally {
    updateVoiceIndicator();
  }
}

function toggleVoiceCommand() {
  if (!voiceRecognition) return;
  voiceEnabled = !voiceEnabled;
  if (voiceEnabled) {
    voiceCommandBtn.textContent = "⏹ 마이크 끄기";
    voiceCommandBtn.classList.add("running");
    voiceCommandStatusText.textContent = "🎤 듣고 있습니다 -- \"탱글아\" 또는 \"탱그라\"라고 불러주세요.";
    try { voiceRecognition.start(); } catch (e) { /* 이미 시작된 상태 등 방어 */ }
  } else {
    voiceCommandBtn.textContent = "🎤 마이크 켜기";
    voiceCommandBtn.classList.remove("running");
    voiceCommandStatusText.textContent = "마이크가 꺼져 있습니다.";
    voiceAwakeUntil = 0;
    voicePendingBicepReps = false;
    try { voiceRecognition.stop(); } catch (e) { /* 이미 멈춘 상태 등 방어 */ }
  }
  updateVoiceIndicator();
}

// 화면 오른쪽 위에 항상 떠 있는 작은 표시 -- 지금 마이크가 듣고 있는지,
// "탱글아"를 불러서 깨어있는(명령을 받는) 상태인지 한눈에 보여준다. 어느
// 모드/카드에 있든 항상 같은 자리에 보이게 body 바로 아래 고정 배치했다.
// awake 상태는 10초 뒤 저절로 풀리는데, 그 사이 새로 말을 안 해도 표시가
// 제때 "듣고 있음"으로 돌아오도록 아래 setInterval에서 주기적으로도 갱신한다.
function updateVoiceIndicator() {
  if (!voiceEnabled) {
    voiceListenIndicator.style.display = "none";
    return;
  }
  voiceListenIndicator.style.display = "flex";
  const awake = performance.now() < voiceAwakeUntil;
  if (voicePendingBicepReps) {
    voiceListenIndicator.className = "voice-listen-indicator awake";
    voiceListenIndicatorText.textContent = "🎤 몇 회인지 대답해주세요";
  } else if (awake) {
    voiceListenIndicator.className = "voice-listen-indicator awake";
    voiceListenIndicatorText.textContent = "🎤 탱글이가 듣고 있어요 (명령하세요)";
  } else {
    voiceListenIndicator.className = "voice-listen-indicator listening";
    voiceListenIndicatorText.textContent = "🎤 듣고 있음 (\"탱글아\" 불러보세요)";
  }
}

// ============================================================================
// 행동 보조 모드 (목표 % 자동 세팅 폐루프 -- 채널1/채널2에 고정 전류를 입력하지
// 않고, 목표%만 정하면 카메라로 실시간 측정하며 세기를 자동으로 찾는다)
// ============================================================================

const ACTION_MODE_DEFS = {
  spoonLift: { label: "수저 들기 보조", btn: spoonLiftBtn },
  bicep: { label: "이두운동", btn: bicepBtn }
};

// 손가락 캘리브레이션(isCalibrated())과 팔 캘리브레이션(calibration.*.Arm)이
// 둘 다 끝나야 손/팔꿈치 % 변환이 다 가능해서 폐루프를 시작할 수 있다.
function actionCalibrationReady() {
  const armFlat = calibration.flat.Arm, armBent = calibration.bent.Arm;
  const armOk = armFlat !== undefined && armBent !== undefined && Math.abs(armBent - armFlat) >= MIN_CAL_GAP_DEG;
  return isCalibrated() && armOk;
}

function resetActionAxisCtrls() {
  actionHandCtrl.intensity = 0; actionHandCtrl.state = "STANDBY"; actionHandCtrl.successSince = null; actionHandCtrl.lastStepTime = 0;
  actionElbowCtrl.intensity = 0; actionElbowCtrl.state = "STANDBY"; actionElbowCtrl.successSince = null; actionElbowCtrl.lastStepTime = 0;
}

// 시작 전 공통 확인(연결/안전값/캘리브레이션) + 팔 인식 모델 로딩까지 기다림.
// 실패하면 이유를 토스트로 띄우고 false를 반환한다.
async function ensureActionModeReady() {
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) {
    showToast(`⚠ ${allowed.reason}`, "warn", 4000);
    return false;
  }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return false;
  }
  if (!actionCalibrationReady()) {
    showToast("⚠ 먼저 ② 초기값 측정(손)과 팔 인식 모드의 '팔 초기값 측정'을 끝내주세요", "warn", 5000);
    return false;
  }
  try {
    await ensurePoseLandmarker();
  } catch (err) {
    showToast("❌ 팔 인식 모델 로딩 실패: " + (err.message || err), "bad", 5000);
    return false;
  }
  return true;
}

// ============================================================================
// 🥄 수저 들기 보조 -- 손 목표 %와 팔꿈치 목표 %를 동시에 폐루프로 도달·유지
// ============================================================================
async function startSpoonLiftClosedLoop() {
  const key = "spoonLift";
  const def = ACTION_MODE_DEFS[key];


  if (runningActionKey && runningActionKey !== key) {
    stopActionMode(`"${ACTION_MODE_DEFS[runningActionKey].label}"에서 "${def.label}"로 전환`);
  } else if (runningActionKey === key) {
    stopActionMode("사용자가 다시 눌러 정지");
    return;
  }

  showToast("🎬 수저 들기 보조 준비 중...", "ok", 2000);
  if (!(await ensureActionModeReady())) return;

  const handTarget = Math.max(0, Math.min(100, Number(spoonLiftHandTargetInput.value) || 0));
  const elbowTarget = Math.max(0, Math.min(100, Number(spoonLiftElbowTargetInput.value) || 0));

  resetActionAxisCtrls();
  runningActionKey = key;
  def.btn.textContent = `■ ${def.label} 정지`;
  def.btn.classList.add("running");
  logControl(`🎬 행동 보조 모드: ${def.label} 시작 (손 목표=${handTarget}%, 팔꿈치 목표=${elbowTarget}%, 자동 세팅)`);
  showToast(`▶ ${def.label} 실행 (자동으로 세기를 찾는 중)`, "ok");

  const state = { handTarget, elbowTarget }; // 손/팔꿈치 동시 진행 -- 둘 다 LOCKED되면 완료 처리 후 정지
  spoonLiftTick(state); // 즉시 한 번 전송
  actionModeInterval = setInterval(() => spoonLiftTick(state), 400);
}

// 수저 들기 보조는 손(채널1)과 팔꿈치(채널2)를 동시에 켜두지 않는다 -- 먼저
// 손이 목표(그립)에 도달하면 손 쪽 전류를 멈추고, 그다음 팔꿈치(채널2)로
// 넘어가라고 안내한 뒤 팔꿈치만 목표를 향해 폐루프로 움직인다.
// (이두운동은 손을 운동 내내 계속 쥐고 있어야 해서 동시 유지 방식 그대로 둠.)
function spoonLiftTick(state) {
  if (safetyTripped) { stopActionMode("안전 정지 상태"); return; }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) { stopActionMode("시리얼 연결 끊김"); return; }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) { stopActionMode(allowed.reason); return; }
  if (continuousStimExceeded()) { stopActionMode("최대 연속 자극 시간을 초과했습니다"); return; }

  const now = performance.now();
  const handAverage = computeAverage(latestPercent.actual);

  // ⚠ 설계 변경: 예전엔 손이 먼저 목표에 도달하면 멈추고 그 다음 팔꿈치로
  // "순차" 진행했는데, 실제로 수저를 드는 동작은 손을 쥐는 것과 팔꿈치를
  // 드는 것이 동시에 일어나야 자연스럽다는 피드백을 받고 다시 동시 진행으로
  // 되돌렸다. 손/팔꿈치 둘 다 매 tick 그대로 목표를 향해 움직이고, 각자
  // 목표에 도달하면 그 축만 유지(LOCKED)하다가 둘 다 도달하면 완료 처리한다.
  if (handLostSustained || armLostSustained) {
    actionHandCtrl.state = "WAITING_FOR_HAND";
    actionElbowCtrl.state = "WAITING_FOR_HAND";
  } else {
    stepFlexOnlyAxis(actionHandCtrl, state.handTarget, handAverage, now);
    stepFlexOnlyAxis(actionElbowCtrl, state.elbowTarget, armLatestPercent.actual, now);
    if (actionHandCtrl.state === "LOCKED" && actionElbowCtrl.state === "LOCKED") {
      // ⚠ 버그 수정(이전과 동일한 종류): 도달해도 계속 유지만 하면 마지막
      // intensity가 "그대로 얼어붙은 채" 계속 하드웨어로 재전송된다 --
      // 목표에 도달하면 실제로 전류를 멈춰야 하므로 완료 처리하고 정지한다.
      stopActionMode("✅ 목표 도달 -- 수저 들기 보조 완료");
      return;
    }
  }

  actionHandStateText.textContent = `${STATE_LABELS[actionHandCtrl.state] || actionHandCtrl.state} (${Math.round(actionHandCtrl.intensity)})`;
  actionElbowStateText.textContent = `${STATE_LABELS[actionElbowCtrl.state] || actionElbowCtrl.state} (${Math.round(actionElbowCtrl.intensity)})`;
  actionModeStatusText.textContent =
    `수저 들기 보조: 손/팔꿈치 동시 진행 -- 손 ${Math.round(handAverage ?? 0)}%/${state.handTarget}% · 팔꿈치 ${Math.round(armLatestPercent.actual ?? 0)}%/${state.elbowTarget}%`;

  // 발표용 큰 TARGET/ACTUAL/오차/세기 칸(팔 인식 모드와 같은 스타일)
  updateCompareBoxes(actionHandTargetDisplay, actionHandActualDisplay, actionHandErrorDisplay, state.handTarget, handAverage);
  actionHandIntensityDisplay.textContent = Math.round(actionHandCtrl.intensity);
  updateCompareBoxes(actionElbowTargetDisplay, actionElbowActualDisplay, actionElbowErrorDisplay, state.elbowTarget, armLatestPercent.actual);
  actionElbowIntensityDisplay.textContent = Math.round(actionElbowCtrl.intensity);

  driveArmHardwareIfNeeded(actionHandCtrl.intensity, actionElbowCtrl.intensity)
    .catch((err) => logControl("행동 보조 모드 전송 오류: " + err.message));
  if (actionHandCtrl.intensity > 0 || actionElbowCtrl.intensity > 0) notifyStimStarted();
  else notifyStimStopped();
}

// ============================================================================
// 💪 이두운동 -- 손 목표로 쥔 채 유지하고, 팔꿈치가 "굽힘 목표 ↔ 이완 목표"를
// 반복 횟수만큼 오간다. 각 단계는 정해진 시간이 아니라 카메라가 실제로 그
// %에 도달(LOCKED)하면 바로 다음 단계로 넘어간다 -- 고정 시간표 자체가 없다
// (굽힘 도달 후 잠깐 유지하는 단계가 있었는데, "도착하면 바로 천천히
// 내려가자"는 요청으로 없앰 -- 도달 즉시 이완으로 넘어감).
// ============================================================================

async function startBicepClosedLoop() {
  const key = "bicep";
  const def = ACTION_MODE_DEFS[key];

  if (runningActionKey && runningActionKey !== key) {
    stopActionMode(`"${ACTION_MODE_DEFS[runningActionKey].label}"에서 "${def.label}"로 전환`);
  } else if (runningActionKey === key) {
    stopActionMode("사용자가 다시 눌러 정지");
    return;
  }

  showToast("🎬 이두운동 준비 중...", "ok", 2000);
  if (!(await ensureActionModeReady())) return;

  const handTarget = Math.max(0, Math.min(100, Number(bicepHandTargetInput.value) || 0));
  const elbowCurlTarget = Math.max(0, Math.min(100, Number(bicepElbowCurlTargetInput.value) || 0));
  const elbowReleaseTarget = Math.max(0, Math.min(100, Number(bicepElbowReleaseTargetInput.value) || 0));
  const reps = Math.max(1, Math.min(100, Number(bicepRepsInput.value) || 1));

  resetActionAxisCtrls();
  runningActionKey = key;
  def.btn.textContent = "■ 이두운동 정지";
  def.btn.classList.add("running");
  logControl(
    `🎬 행동 보조 모드: 이두운동 시작 (손=${handTarget}%, 팔꿈치 굽힘=${elbowCurlTarget}%/이완=${elbowReleaseTarget}%, ${reps}회 반복, 자동 세팅)`
  );
  showToast(`▶ 이두운동 실행 (${reps}회 반복, 자동으로 세기를 찾는 중)`, "ok");

  const state = {
    handTarget, elbowCurlTarget, elbowReleaseTarget, reps,
    repIndex: 0,
    phase: "grip" // grip -> curl -> release -> (curl... 반복) -> 완료
  };
  bicepClosedLoopTick(state); // 즉시 한 번 전송
  actionModeInterval = setInterval(() => bicepClosedLoopTick(state), 400);
}

function bicepClosedLoopTick(state) {
  if (safetyTripped) { stopActionMode("안전 정지 상태"); return; }
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) { stopActionMode("시리얼 연결 끊김"); return; }
  const allowed = liveOutputAllowedByConfig();
  if (!allowed.ok) { stopActionMode(allowed.reason); return; }
  if (continuousStimExceeded()) { stopActionMode("최대 연속 자극 시간을 초과했습니다"); return; }

  const now = performance.now();
  const handAverage = computeAverage(latestPercent.actual);

  if (handLostSustained || armLostSustained) {
    actionHandCtrl.state = "WAITING_FOR_HAND";
    actionElbowCtrl.state = "WAITING_FOR_HAND";
    actionModeStatusText.textContent = "이두운동: 손/팔 인식 대기 중";
    actionHandStateText.textContent = STATE_LABELS.WAITING_FOR_HAND;
    actionElbowStateText.textContent = STATE_LABELS.WAITING_FOR_HAND;
    updateCompareBoxes(actionHandTargetDisplay, actionHandActualDisplay, actionHandErrorDisplay, state.handTarget, null);
    actionHandIntensityDisplay.textContent = Math.round(actionHandCtrl.intensity);
    updateCompareBoxes(actionElbowTargetDisplay, actionElbowActualDisplay, actionElbowErrorDisplay, null, null);
    actionElbowIntensityDisplay.textContent = Math.round(actionElbowCtrl.intensity);
    driveArmHardwareIfNeeded(actionHandCtrl.intensity, actionElbowCtrl.intensity).catch(() => {});
    return;
  }

  // 손은 처음부터 끝까지 계속 목표를 유지하도록 매 tick 그대로 재적용한다
  // (쥔 손이 느슨해지면 폐루프가 알아서 다시 조여줌).
  stepFlexOnlyAxis(actionHandCtrl, state.handTarget, handAverage, now);

  // 지금 단계에서 팔꿈치가 향하고 있는 목표(굽힘/이완 중 어느 쪽인지) -- 큰
  // 칸에 "지금 단계" 목표를 보여주기 위함. 손 쥐는 중(grip)일 땐 아직 팔꿈치
  // 차례가 아니라, 다음에 향할 굽힘 목표를 미리 보여준다.
  let elbowTargetNow = state.elbowCurlTarget;

  let phaseLabel = "";
  if (state.phase === "grip") {
    phaseLabel = `손 쥐는 중 (목표 ${state.handTarget}%)`;
    if (actionHandCtrl.state === "LOCKED") {
      state.phase = "curl";
      state.repIndex = 1;
    }
  } else if (state.phase === "curl") {
    stepFlexOnlyAxis(actionElbowCtrl, state.elbowCurlTarget, armLatestPercent.actual, now);
    phaseLabel = `${state.repIndex}/${state.reps}회차 -- 팔꿈치 굽히는 중 (목표 ${state.elbowCurlTarget}%)`;
    // ⚠ LOCKED(허용범위 안에서 successHoldSeconds만큼 버텨야 함)를 기다리지
    // 않고, 실제 굽힘이 목표치에 도달하거나 넘어서는 그 즉시 이완으로
    // 넘어간다 -- "수축이 목표 이상 되면 바로 이완"이라는 요청대로.
    if (armLatestPercent.actual !== null && armLatestPercent.actual >= state.elbowCurlTarget) {
      speak(`${sinoKoreanNumber(state.repIndex)}회`);
      logControl(`💪 수축 완료 -- ${state.repIndex}/${state.reps}회`);
      showToast(`💪 ${state.repIndex}회 완료`, "ok", 1800, true);
      state.phase = "release";
    }
  } else if (state.phase === "release") {
    stepFlexOnlyAxis(actionElbowCtrl, state.elbowReleaseTarget, armLatestPercent.actual, now);
    elbowTargetNow = state.elbowReleaseTarget;
    phaseLabel = `${state.repIndex}/${state.reps}회차 -- 팔꿈치 이완 중 (목표 ${state.elbowReleaseTarget}%)`;
    // 굽힘 쪽과 똑같이, LOCKED를 기다리지 않고 실제 이완 %가 목표치에
    // 도달하거나 그 이하가 되는 즉시 다음 단계로 넘어간다.
    if (armLatestPercent.actual !== null && armLatestPercent.actual <= state.elbowReleaseTarget) {
      if (state.repIndex >= state.reps) {
        speak("완료되었습니다");
        stopActionMode(`이두운동 ${state.reps}회 완료`);
        return;
      }
      state.repIndex += 1;
      state.phase = "curl";
    }
  }

  actionHandStateText.textContent = `${STATE_LABELS[actionHandCtrl.state] || actionHandCtrl.state} (${Math.round(actionHandCtrl.intensity)})`;
  actionElbowStateText.textContent = `${STATE_LABELS[actionElbowCtrl.state] || actionElbowCtrl.state} (${Math.round(actionElbowCtrl.intensity)})`;
  actionModeStatusText.textContent = `이두운동: ${phaseLabel}`;

  // 발표용 큰 TARGET/ACTUAL/오차/세기 칸(팔 인식 모드와 같은 스타일)
  updateCompareBoxes(actionHandTargetDisplay, actionHandActualDisplay, actionHandErrorDisplay, state.handTarget, handAverage);
  actionHandIntensityDisplay.textContent = Math.round(actionHandCtrl.intensity);
  updateCompareBoxes(actionElbowTargetDisplay, actionElbowActualDisplay, actionElbowErrorDisplay, elbowTargetNow, armLatestPercent.actual);
  actionElbowIntensityDisplay.textContent = Math.round(actionElbowCtrl.intensity);

  driveArmHardwareIfNeeded(actionHandCtrl.intensity, actionElbowCtrl.intensity)
    .catch((err) => logControl("이두운동 전송 오류: " + err.message));
  if (actionHandCtrl.intensity > 0 || actionElbowCtrl.intensity > 0) notifyStimStarted();
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
  resetActionAxisCtrls();
  notifyStimStopped();
  if (serialLink) {
    serialLink.setIntensity(1, 0, 500).catch(() => {});
    serialLink.setIntensity(2, 0, 500).catch(() => {});
  }
}

// (예전엔 여기 개인화 측정(목표 도달 세기 자동 탐색) 기능 전체가 있었다 --
// 손 -> 팔꿈치 순서로 자동으로 세기를 찾아 기록해주고, 실행 시 그 세기로
// 웜스타트하던 것. 필요 없다는 요청으로 selectPersonalizationCalAction/
// startPersonalizationCal/personalizationCalTick/finishPersonalizationCal/
// stopPersonalizationCal/seedActionAxisCtrls/updateActionSeedTexts를 전부
// 제거했다. 이제 spoonLift/bicep 실행은 항상 0부터 자동으로 세기를 찾는다.)

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
  // ⚠ 버그 수정: safetyTripped는 그동안 "트립되는 그 순간" forceZeroController()로
  // 한 번 0으로 만드는 것 말고는, 그 뒤로 매 tick 계속 확인하는 코드가 없었다.
  // 그래서 실시간 왼손 연동(liveMirrorActive)이나 팔 인식 모드처럼 매 프레임
  // targetPercent를 다시 채워주는 기능이 켜져 있으면, 트립된 다음에도 바로
  // 다음 tick에 목표가 다시 채워지면서 제어가 조용히 재개돼버렸다 (행동 보조/
  // 개인화 모드는 각자의 tick 함수에 safetyTripped 체크가 있어서 이 문제가
  // 없었는데, 거울모드/팔인식모드가 쓰는 여기(updateController)엔 빠져있었음).
  if (safetyTripped) {
    controllerIntensity = 0;
    successSince = null;
    holdLocked = false;
    controlState = "SAFETY_STOP";
    lastError = null;
    return;
  }
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
    // ⚠ 버그 수정: 단, 목표(targetPercent) 자체가 거의 0(실시간 왼손 연동 등에서
    // "구부리지 말라"는 뜻)인데 실제 굽힘도 마침 그 근처면, 이 순간까지 남아있던
    // 세기를 그대로 "유지"해버려서 목표가 0이 됐는데도 전기가 계속 나가는
    // 문제가 있었다 -- 목표가 거의 0일 땐 유지할 자세가 없으니 세기도 0으로.
    if (targetPercent <= config.control.tolerancePercent) {
      controllerIntensity = 0;
    }
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

// ⚠ 요청에 따라 "오른손/오른팔을 카메라가 놓치면 비상정지"를 없앴다 --
// handLost 인자는 더 이상 안 쓰지만(호출부를 다 바꾸지 않으려고) 그대로
// 받기만 하고 무시한다. 손을 놓친 동안은(handLostSustained/armLostSustained)
// 각 모드의 컨트롤러가 WAITING_FOR_HAND로 표시하고 목표 추적만 잠깐
// 멈추며, 이 함수가 강제로 완전 정지시키지는 않는다.
function runtimeCheck(handLost) {
  if (!liveModeRequested) return { ok: true };
  if (!serialLink || !serialLink.isConnected()) return { ok: false, reason: "시리얼 연결이 끊어졌습니다." };
  if (continuousStimExceeded()) return { ok: false, reason: "최대 연속 자극 시간을 초과했습니다." };
  return { ok: true };
}

function triggerEmergencyStop(reason) {
  const wasTripped = safetyTripped;
  safetyTripped = true;
  safetyTripReason = reason;
  forceZeroController(`🛑 안전 정지: ${reason}`);
  if (runningActionKey) stopActionMode(`🛑 안전 정지: ${reason}`);
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
// 바로 다시 꺼진다 (spoonLiftTick/bicepClosedLoopTick의 safetyTripped 체크
// 참고). 예전엔 이 상태가 화면 어디에도 안 보여서 "왜 자꾸 꺼지지?"의
// 원인을 알 방법이 없었다 -- 지금은 ③ 카드 맨 위에 이유와 해제 버튼을 보여준다.
function updateSafetyTripUi() {
  // 테스트 모드는 안전장치 자체가 없는 모드라(요청에 따라 의도적으로 뺌),
  // 다른 모드에서 트립된 게 남아있어도 이 배너는 테스트 모드에서만큼은
  // 띄우지 않는다 -- 실제로 testModeKeyDown/sendTestPulse는 safetyTripped를
  // 전혀 참조하지 않으므로 표시만 안 맞았던 것.
  if (safetyTripped && appMode !== "test") {
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
// 팔 인식 모드 전용 폐루프 -- 손(채널1)과 팔꿈치(채널2)를 동시에 독립적으로
// "부족하면 조심스럽게 증가(kpUp/maxStepUp), 넘치면 빠르게 감소(reduceKp/
// reduceMaxStep), 도달했으면 유지"하는 단순 P제어 -- updateController()처럼
// 펴는 채널로 전환하는 로직은 없어서 그만큼 훨씬 단순하다.
// ============================================================================
function stepFlexOnlyAxis(ctrl, targetPercent, currentPercent, now) {
  if (targetPercent === null || currentPercent === null) {
    ctrl.state = "WAITING_FOR_HAND";
    ctrl.successSince = null;
    return; // intensity는 건드리지 않음 -- 마지막 값 유지 (잠깐 놓친 것일 수 있어서)
  }
  const error = targetPercent - currentPercent;
  if (Math.abs(error) <= config.control.tolerancePercent) {
    // ⚠ 버그 수정: 목표(targetPercent) 자체가 거의 0(사실상 "구부리지 말라"는
    // 뜻)인데 실제 굽힘도 마침 그 근처라서 허용 오차 안에 들어오면, 예전엔
    // 그 순간까지 남아있던 intensity를 그대로 "유지"해버렸다 -- 목표가
    // 낮아져서 0이 됐는데도 전기가 계속 나가던 원인. 목표가 거의 0일 땐
    // 유지할 자세가 없으니 그냥 세기도 0으로 내린다(그 외의 정상적인
    // "목표에 도달해서 유지" 상황은 기존처럼 마지막 세기를 그대로 유지).
    if (targetPercent <= config.control.tolerancePercent) {
      ctrl.intensity = 0;
    }
    if (ctrl.successSince === null) ctrl.successSince = now;
    const heldForS = (now - ctrl.successSince) / 1000;
    ctrl.state = heldForS >= config.control.successHoldSeconds ? "LOCKED" : "HOLDING";
    return;
  }
  ctrl.successSince = null;

  if (error < 0) {
    // 목표보다 더 구부러짐(또는 실시간 연동 목표가 방금 낮아짐) -- 세기를
    // 낮추는 쪽은 안전 문제가 없으므로 주기 제한 없이 매 프레임 바로 반응해서
    // 빠르게 따라 내려간다 (reduceKp/reduceMaxStep, 올릴 때보다 훨씬 큼). 예전엔
    // 여기도 올릴 때와 똑같이 300ms에 1씩만 내려가서, 실시간으로 계속 바뀌는
    // 목표가 훅 낮아져도 한동안 "액추얼이 목표보다 훨씬 높은" 채로 남아 전기가
    // 계속/더 세지는 것처럼 보이는 문제가 있었다.
    ctrl.lastStepTime = now;
    const step = Math.min(config.control.reduceMaxStep, config.control.reduceKp * Math.abs(error));
    ctrl.intensity = clampWorkingFloat(ctrl.intensity - step);
    ctrl.state = "REDUCING";
    return;
  }

  // ⚠ 버그 수정: 이 함수는 카메라 프레임마다(초당 수십 번) 불렸는데, 거울모드
  // updateController()는 원래 control_period_ms 주기로만 세기를 바꾼다.
  // 여기 그 제한이 빠져있어서 체감 속도가 실제보다 수십 배 빠르게 느껴졌다
  // ("너무 빠르다"는 문제의 핵심 원인). 동일하게 주기 제한을 건다 -- 단, 올릴
  // 때만: 안전을 위해 세기를 올릴 땐 여전히 조심스럽게 한 주기에 조금씩만.
  if (now - ctrl.lastStepTime < getControlPeriodMs()) {
    ctrl.state = "INCREASING";
    return;
  }
  ctrl.lastStepTime = now;

  // 목표보다 덜 구부러짐 -- 조심스럽게 증가 (kpUp/maxStepUp)
  const step = Math.min(config.control.maxStepUp, config.control.kpUp * error);
  ctrl.intensity = clampWorkingFloat(ctrl.intensity + step);
  ctrl.state = "INCREASING";
}

// 손(채널1)·팔꿈치(채널2)를 매번 같이 보낸다 -- 거울모드의 driveHardwareIfNeeded와
// 달리 "다른 채널을 먼저 끄는" 로직이 없다: 여기선 두 채널이 항상 동시에
// 켜져 있는 게 정상이기 때문 (한쪽이 다른 쪽을 밀어내지 않음).
async function driveArmHardwareIfNeeded(handIntensity, elbowIntensity) {
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) return;
  const ttl = clampTtl(config.safety.commandTtlMs);
  if (!armedCh1) {
    const ok = await serialLink.arm(1);
    if (ok) armedCh1 = true;
  }
  if (!armedCh2) {
    const ok = await serialLink.arm(2);
    if (ok) armedCh2 = true;
  }
  if (armedCh1) await serialLink.setIntensity(1, clampHardware(handIntensity), ttl);
  if (armedCh2) await serialLink.setIntensity(2, clampHardware(elbowIntensity), ttl);
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
const TEST_RESEND_MS = 200; // TTL(600ms)보다 충분히 짧게 재전송 -- 아래 testModeKeyDown 참고

async function testModeKeyDown(channel) {
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) {
    showToast("⚠ Arduino가 연결되어 있지 않습니다", "warn");
    return;
  }

  // ⚠ 버그 수정: 예전엔 브라우저의 keydown "자동 반복" 이벤트가 올 때마다
  // 매번 새로 SET을 보내는 식이었다. 근데 이 자동 반복 간격은 OS/브라우저
  // 설정에 따라 들쭉날쭉해서(키보드 반복 속도가 느리게 설정된 PC 등), 다음
  // 반복 이벤트가 오기 전에 TTL(600ms)이 먼저 끝나버리면 그 사이 잠깐씩
  // 전류가 끊기는 "주기적으로 끊김" 증상이 생겼다. 지금은 keydown 한 번(눌린
  // 첫 순간)에만 반응하고, 그때부터는 이 함수가 아니라 별도 setInterval
  // (testResendInterval)이 TEST_RESEND_MS(200ms)마다 알아서 계속 재전송한다
  // -- 브라우저/OS의 키 반복 타이밍과 완전히 무관해져서, 키를 누르고 있는 한
  // 절대 안 끊긴다.
  if (testKeyChannel === channel) return; // 이미 이 채널로 눌려 있는 중 -- 반복 keydown은 무시(재전송은 인터벌이 담당)

  if (testResendInterval) { clearInterval(testResendInterval); testResendInterval = null; }

  // 다른 채널이 눌려있던 상태였다면 먼저 확실히 끔 (두 채널 동시 자극 방지 --
  // driveHardwareIfNeeded와 같은 이유).
  if (lastDrivenChannel !== null && lastDrivenChannel !== channel) {
    try { await serialLink.setIntensity(lastDrivenChannel, 0, 500); } catch (e) { /* ignore */ }
  }

  testKeyChannel = channel;
  const ok = await sendTestPulse(channel);
  if (!ok) return;
  testResendInterval = setInterval(() => {
    // ⚠ 버그 수정(2차): TEST_RESEND_MS마다 무조건 새로 sendTestPulse를 부르면,
    // SET 하나가 실제로 응답(OK/ERROR)받는 데 200ms보다 오래 걸릴 때마다(직렬
    // 통신 왕복 지연, 아두이노 쪽 처리 지연 등) WebSerialLink의 명령 큐(_chain)에
    // 다음 SET이 계속 쌓인다. 큐가 밀리기 시작하면 실제로 하드웨어에 도착하는
    // 시점이 점점 늦어져서, 앞서 보낸 SET의 TTL(600ms)이 먼저 만료돼 채널이
    // 꺼졌다가 밀려있던 다음 SET이 뒤늦게 도착해서 다시 켜지는 식으로 "왔다
    // 안 왔다"가 반복된다. 이전 전송이 아직 응답을 못 받은 상태면 이번 틱은
    // 그냥 건너뛰어서(새로 쌓지 않음) 큐가 절대 밀리지 않게 한다 -- 실제 재전송
    // 간격이 통신 왕복 속도에 맞춰 자연스럽게 늘어날 뿐, 밀려서 끊기지는 않는다.
    if (testPulseInFlight) return;
    sendTestPulse(channel);
  }, TEST_RESEND_MS);
}

let testPulseInFlight = false; // 위 setInterval 콜백의 큐 적체 방지 가드

// 실제로 한 번 SET을 내보내는 부분 -- testModeKeyDown(최초 1회)과
// testResendInterval(그 뒤로 계속)이 공유해서 부른다.
async function sendTestPulse(channel) {
  testPulseInFlight = true;
  try {
    const alreadyArmed = channel === 1 ? armedCh1 : armedCh2;
    if (!alreadyArmed) {
      const ok = await serialLink.arm(channel);
      if (!ok) {
        showToast(`⚠ 채널${channel} ARM 실패`, "bad");
        return false;
      }
      if (channel === 1) armedCh1 = true;
      else armedCh2 = true;
    }

    // ③ 카드의 안전 최대값(clampHardware)을 거치지 않고, 입력한 세기를 0~100
    // 범위만 맞춰서 그대로 내보낸다 (테스트 모드는 전기 주기 + 세기 설정, 그
    // 두 가지만 하도록 요청받아 나머지 안전 게이트는 여기서는 뺐다). 채널1/채널2
    // 세기를 따로 입력받아 각자 다른 값으로 테스트할 수 있게 한다.
    const intensityInput = channel === 1 ? testIntensityCh1Input : testIntensityCh2Input;
    const intensity = Math.round(Math.min(100, Math.max(0, Number(intensityInput.value) || 0)));
    // TTL을 짧게 잡아서(1000ms), TEST_RESEND_MS(200ms)마다(위 가드로 밀리지 않게)
    // 재전송하며 TTL을 계속 갱신한다. 혹시 재전송이 하필 늦어도 1초 안에는
    // 자동으로 꺼지니, keyup을 못 받는 상황(창 포커스 이탈 등)에서도 오래 켜진
    // 채로 남지 않는다.
    await serialLink.setIntensity(channel, intensity, 1000);
    lastDrivenChannel = channel;
    testModeStatusText.textContent = `채널${channel} 자극 중 (세기 ${intensity})`;
    return true;
  } finally {
    testPulseInFlight = false;
  }
}

async function testModeKeyUp(channel) {
  if (testKeyChannel !== channel) return; // 이미 다른 키로 넘어갔거나 이미 꺼진 상태
  testKeyChannel = null;
  if (testResendInterval) { clearInterval(testResendInterval); testResendInterval = null; }
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

// ⚠ 요청에 따라 "전체 실험 제한시간"(초과하면 모든 모드 강제 정지) 기능은
// 없앴다 -- 이 표시는 그냥 카메라를 처음 켠 뒤 얼마나 지났는지 보여주기만
// 하는 참고용 경과 시간이고, 아무것도 강제로 멈추지 않는다.
function updateElapsedDisplay() {
  if (sessionStartedAt === null) {
    elapsedDisplay.textContent = "0s";
    return;
  }
  const elapsedS = (performance.now() - sessionStartedAt) / 1000;
  elapsedDisplay.textContent = `${Math.floor(elapsedS)}s`;
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

// controlEnabled(▶ 제어 시작)로 목표를 실제로 쫓고 있는 동안만 관찰한다 --
// 대기 중/비상정지/목표 없음일 때는 시행으로 안 치고 관찰을 멈춘다.
function updateTrialTracking(now) {
  if (!controlEnabled || safetyTripped || targetPercent === null) {
    trialAttemptStart = null;
    return;
  }
  if (controlState === "LOCKED" || controlState === "SUCCESS") {
    if (trialAttemptStart !== null) {
      recordTrial("성공", (now - trialAttemptStart) / 1000);
      trialAttemptStart = null; // 도달해서 유지 중인 동안은 다음 시도를 세지 않음 -- 목표를 벗어나야(또는 새 목표) 다시 시작
    }
    return;
  }
  if (trialAttemptStart === null) {
    trialAttemptStart = now; // 목표를 향해 다시 움직이기 시작 -- 새 시도 개시
    return;
  }
  if (now - trialAttemptStart > TRIAL_TIMEOUT_MS) {
    recordTrial("실패", null);
    trialAttemptStart = now; // 바로 다음 시도 관찰 시작
  }
}

function recordTrial(result, seconds) {
  trialLog.push({
    n: trialLog.length + 1,
    result,
    seconds,
    target: targetPercent === null ? null : Math.round(targetPercent),
    actual: currentAverage === null ? null : Math.round(currentAverage),
    time: new Date().toLocaleTimeString("ko-KR")
  });
  updateTrialCountUI();
  const successCount = trialLog.filter((t) => t.result === "성공").length;
  logControl(
    result === "성공"
      ? `🎯 도달 성공 (${seconds.toFixed(1)}초) -- 누적 ${successCount}/${trialLog.length}회`
      : `⏱ 12초 안에 도달 못함(실패) -- 누적 ${successCount}/${trialLog.length}회`
  );
}

function updateTrialCountUI() {
  const successCount = trialLog.filter((t) => t.result === "성공").length;
  trialCountText.textContent = trialLog.length;
  trialSuccessCountText.textContent = successCount;
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

// ⚠ 예전엔 카메라 프레임마다(매 tick) 한 줄씩 그대로 CSV로 내보내서, 몇 분만
// 써도 수천 줄짜리 원본 로그가 됐다 -- 사람이 보고 "몇 번 시도해서 몇 번
// 성공했는지" 바로 알기 어려웠다. 이제 위 trialLog(도달 시행 기록)를 기준으로
// "시행 번호/성공·실패/도달 시간" 한 줄 = 시도 1회로 뽑고, 맨 위에 요약(총
// 시행/성공/성공률/평균 도달 시간)을 붙인다. 원본 tick 로그가 필요하면 아래
// "원본 tick 기록" 표에서 그대로 가져간다(참고용으로 남겨둠).
function downloadCsv() {
  const successRows = trialLog.filter((t) => t.result === "성공");
  const avgSeconds =
    successRows.length > 0
      ? successRows.reduce((sum, t) => sum + t.seconds, 0) / successRows.length
      : null;
  const successRate = trialLog.length > 0 ? (successRows.length / trialLog.length) * 100 : 0;

  const summaryLines = [
    "요약",
    `총 시행 횟수,${trialLog.length}`,
    `성공 횟수,${successRows.length}`,
    `실패 횟수,${trialLog.length - successRows.length}`,
    `성공률(%),${successRate.toFixed(1)}`,
    `평균 도달 시간(성공만·초),${avgSeconds === null ? "-" : avgSeconds.toFixed(1)}`,
    "",
    "시행별 결과",
    "시행 번호,결과,도달 시간(초),목표 %,실제 %,시각"
  ];
  const trialLines = trialLog.map((t) =>
    [t.n, t.result, t.seconds === null ? "-" : t.seconds.toFixed(1), t.target ?? "", t.actual ?? "", t.time].join(",")
  );

  // 원본 tick 로그도 참고용으로 아래에 이어 붙인다(필요 없으면 그냥 무시하면 됨).
  const rawHeader = ["", "원본 tick 기록(참고용)", "timestamp_ms,target_percent,current_percent,ems_intensity,error,control_state,success,hand_detected"];
  const rawLines = logRows.map((r) =>
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
  );

  const csv = [...summaryLines, ...trialLines, ...rawHeader, ...rawLines].join("\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" }); // BOM -- 엑셀에서 한글 안 깨지게
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ems_web_session_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
