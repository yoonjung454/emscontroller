import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

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

// 엄지 제외 (ems_closed_loop/vision_tracker.py 와 동일한 기준)
const FINGERS = {
  Index: [[0, 5, 6], [5, 6, 7], [6, 7, 8]],
  Middle: [[0, 9, 10], [9, 10, 11], [10, 11, 12]],
  Ring: [[0, 13, 14], [13, 14, 15], [14, 15, 16]],
  Little: [[0, 17, 18], [17, 18, 19], [18, 19, 20]]
};
const FINGER_LABELS = { Index: "검지", Middle: "중지", Ring: "약지", Little: "소지" };
const FINGER_KEYS = Object.keys(FINGERS);

// 왼손 = 목표 동작을 캡처하는 손(SOURCE), 오른손 = EMS를 연결할 손(ACTUAL)
const ROLES = ["source", "actual"];
const ROLE_LABELS = { source: "왼손(TARGET)", actual: "오른손(ACTUAL)" };
const ROLE_COLORS = { source: "#4da6ff", actual: "#ffb454" };

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
  INCREASING: "자극 증가",
  DECREASING: "자극 감소",
  HOLDING: "목표 유지 중",
  SUCCESS: "성공",
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
let lastVideoTime = -1;
let invertHandedness = false;

// 손별(source/actual) 필터 · 최신값
const filters = { source: {}, actual: {} };
for (const role of ROLES) for (const finger of FINGER_KEYS) filters[role][finger] = new MedianEmaFilter(SMOOTH_WINDOW, EMA_ALPHA);

const detectedThisFrame = { source: false, actual: false };
let consecutiveMissActual = 0;
let handLostSustained = true; // ACTUAL(오른손) 기준, 안전 판단에 사용

const latestSmoothed = { source: {}, actual: {} }; // role -> finger -> degrees
const latestPercent = { source: {}, actual: {} }; // role -> finger -> 0-100 | null

let targetPerFinger = { Index: null, Middle: null, Ring: null, Little: null };
let currentAverage = null; // ACTUAL(오른손) 4손가락 평균

let sampling = null; // 초기값(펴짐/구부림) 측정, ACTUAL(오른손) 기준
let capturing = null; // 왼손 스냅샷 캡처

// 폐루프 제어기 상태
let targetPercent = null; // targetPerFinger 의 평균값
let controllerIntensity = 0; // "작업용" 0~100, 하드웨어 안전상한과는 별개
let successSince = null;
let lastStepTime = 0;
let controlState = "IDLE";
let lastError = null;

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

const captureBtn = document.getElementById("captureBtn");
const captureOverlay = document.getElementById("captureOverlay");
const captureTitle = document.getElementById("captureTitle");
const captureProgress = document.getElementById("captureProgress");

const commandInput = document.getElementById("commandInput");
const commandRunBtn = document.getElementById("commandRunBtn");
const commandMessage = document.getElementById("commandMessage");
const presetLightBtn = document.getElementById("presetLight");
const presetHalfBtn = document.getElementById("presetHalf");
const presetStrongBtn = document.getElementById("presetStrong");

const aiEnabledCheckbox = document.getElementById("aiEnabledCheckbox");
const aiConfigRow = document.getElementById("aiConfigRow");
const aiApiKeyInput = document.getElementById("aiApiKeyInput");
const aiModelInput = document.getElementById("aiModelInput");
const aiStatusBadge = document.getElementById("aiStatusBadge");
const ruleBasedHint = document.getElementById("ruleBasedHint");

const safetyMaxInput = document.getElementById("safetyMaxInput");
const safetyMaxPill = document.getElementById("safetyMaxPill");
const serialConnectBtn = document.getElementById("serialConnectBtn");
const serialDisconnectBtn = document.getElementById("serialDisconnectBtn");
const serialStatusPill = document.getElementById("serialStatusPill");
const serialSupportHint = document.getElementById("serialSupportHint");
const startControlBtn = document.getElementById("startControlBtn");

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

// ============================================================================
// 초기화
// ============================================================================

buildTable();
initSafetyUi();
initPresetLabels();
aiEnabledCheckbox.checked = aiConfig.enabled;
aiApiKeyInput.value = aiConfig.apiKey;
aiModelInput.value = aiConfig.model;
updateAiUi();
resizeGraphs();
window.addEventListener("resize", resizeGraphs);
setInterval(drawGraphs, 250);
setInterval(updateElapsedDisplay, 500);
updatePersonalizationSummary();
drawPersonalizationGraph();

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
captureBtn.addEventListener("click", startCapture);

commandRunBtn.addEventListener("click", submitCommand);
commandInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitCommand();
});
presetLightBtn.addEventListener("click", () => applyPresetTarget(config.presets.light, "light"));
presetHalfBtn.addEventListener("click", () => applyPresetTarget(config.presets.half, "half"));
presetStrongBtn.addEventListener("click", () => applyPresetTarget(config.presets.strong, "strong"));

safetyMaxInput.addEventListener("change", () => {
  config.safety.maxIntensity = Math.max(0, Math.min(100, Number(safetyMaxInput.value) || 0));
  safetyMaxInput.value = config.safety.maxIntensity;
  saveConfig();
  updateSafetyPill();
});

aiEnabledCheckbox.addEventListener("change", () => {
  aiConfig.enabled = aiEnabledCheckbox.checked;
  saveAiConfig();
  updateAiUi();
});
aiApiKeyInput.addEventListener("change", () => {
  aiConfig.apiKey = aiApiKeyInput.value.trim();
  saveAiConfig();
});
aiModelInput.addEventListener("change", () => {
  aiConfig.model = aiModelInput.value.trim() || "gemini-3.7-flash";
  aiModelInput.value = aiConfig.model;
  saveAiConfig();
  updateAiUi();
});

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
    logControl("■ 제어 정지 -- 자극을 멈췄습니다 (목표는 유지됨, 다시 시작하려면 버튼을 다시 누르세요)");
    showToast("■ 제어 정지", "warn");
  }
});

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
  }
});
document.addEventListener("visibilitychange", () => {
  // 브라우저 탭이 백그라운드로 가면 requestAnimationFrame/타이머가 느려지거나
  // 멈출 수 있어, 자극이 계속되는데 화면 갱신/워치독은 멈추는 위험한 상황이
  // 생길 수 있다. 그래서 탭이 보이지 않게 되는 순간 즉시 안전 정지한다.
  if (document.hidden && isRunning) {
    triggerEmergencyStop("브라우저 탭이 백그라운드로 전환됨");
  }
});
window.addEventListener("beforeunload", () => {
  if (serialLink) serialLink.stopAll().catch(() => {});
});

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
  updateSafetyPill();
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
  document.getElementById("presetLightVal").textContent = config.presets.light;
  document.getElementById("presetHalfVal").textContent = config.presets.half;
  document.getElementById("presetStrongVal").textContent = config.presets.strong;
}

// ============================================================================
// 카메라 실행 / 정지
// ============================================================================

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
    captureBtn.disabled = false;

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
  captureOverlay.classList.remove("active");
  resetStartControlButton(); // 카메라를 정지하면 제어도 확실히 같이 정지 상태로

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
  captureBtn.disabled = true;
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
    processResult(result);
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

function processResult(result) {
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

    if (sampling && role === "actual") {
      for (const finger of FINGER_KEYS) sampling.sums[finger] += rawBends[finger];
      sampling.count += 1;
      updateSamplingUI();
      if (sampling.count >= SAMPLE_TARGET) finishSampling();
    }
    if (capturing && role === "source") {
      for (const finger of FINGER_KEYS) capturing.sums[finger] += rawBends[finger];
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

  setStatusBadge(statusBadgeSource, detectedThisFrame.source, "source");
  setStatusBadge(statusBadgeActual, detectedThisFrame.actual, "actual");

  updateFingerTable();
  currentAverage = computeAverage(latestPercent.actual);

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
    // ---- 안전 확인 (컨트롤러 계산보다 먼저: 트립되면 이번 tick에서 즉시 0으로) ----
    const runtime = runtimeCheck(handLostSustained);
    if (!runtime.ok && !safetyTripped) {
      triggerEmergencyStop(runtime.reason);
    }

    // ---- 폐루프 제어 ----
    const prevState = controlState;
    updateController(handLostSustained ? null : currentAverage, now);
    if (controlState !== prevState) logControl(`상태 변경: ${STATE_LABELS[prevState] || prevState} → ${STATE_LABELS[controlState] || controlState}`);

    if (controllerIntensity > 0) notifyStimStarted();
    else notifyStimStopped();
  }

  // 카메라는 초당 수십 프레임이지만, 하드웨어로는 control_period_ms(기본 0.5초)에
  // 한 번씩만 보낸다. 매 프레임 보내면 시리얼 명령이 계속 쌓여서(큐 적체) 실제
  // 전송이 몇 초씩 밀리는 심각한 문제가 있었다 (실측으로 발견됨).
  if (controlEnabled && liveModeRequested && now - lastHardwareSendTime >= config.control.controlPeriodMs) {
    lastHardwareSendTime = now;
    driveHardwareIfNeeded(controllerIntensity).catch((err) => logControl("하드웨어 전송 오류: " + err.message));
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
  updateHardwareSentHint();
  controlStateText.textContent = STATE_LABELS[controlState] || controlState;
}

// "EMS 제어값"은 안전 상한과 무관한 내부 계산값이라, 실제로 하드웨어에 나가는
// (안전 최대값으로 잘린) 값과 다를 수 있다. 그걸 혼동해서 "화면엔 100인데 왜
// 전기가 안 오지?"가 되는 걸 막기 위해 실제 전송값을 바로 옆에 같이 보여준다.
function updateHardwareSentHint() {
  const sent = clampHardware(controllerIntensity);
  if (!liveModeRequested) {
    hardwareSentHint.textContent = "실제 전송값: - (라이브 모드 아님, 시뮬레이션만)";
    hardwareSentHint.style.color = "var(--text-dim)";
  } else if (sent < controllerIntensity) {
    hardwareSentHint.textContent = `실제 전송값: ${sent} (안전 최대값 ${config.safety.maxIntensity}로 잘림!)`;
    hardwareSentHint.style.color = "var(--warn)";
  } else {
    hardwareSentHint.textContent = `실제 전송값: ${sent}`;
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
  for (const finger of FINGER_KEYS) sums[finger] = 0;
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
  for (const finger of FINGER_KEYS) calibration[mode][finger] = sums[finger] / count;
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
  for (const finger of FINGER_KEYS) sums[finger] = 0;
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
  for (const finger of FINGER_KEYS) rawAverages[finger] = sums[finger] / count;

  for (const finger of FINGER_KEYS) {
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
  for (const finger of FINGER_KEYS) {
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
  for (const finger of FINGER_KEYS) {
    latestPercent.actual[finger] = percentFor(finger, latestSmoothed.actual[finger]);
  }

  for (const finger of FINGER_KEYS) {
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
  for (const finger of FINGER_KEYS) targetPerFinger[finger] = percent;
  setTarget(percent, suggestInitialIntensity(percent));
  // updateCompareDisplay/updateFingerTable은 원래 카메라 루프(processResult)
  // 안에서만 갱신됐다. 그러면 카메라가 안 돌고 있을 때 명령만 테스트하면
  // targetPercent는 내부적으로 바뀌었는데 화면(TARGET 칸)엔 반영이 안 되는
  // 것처럼 보인다. 명령을 적용한 즉시 화면도 갱신한다.
  updateCompareDisplay();
  updateFingerTable();
}

// ============================================================================
// 폐루프 제어기 (controller.py 를 JS로 포팅)
// ============================================================================

function setTarget(percent, initialIntensity) {
  targetPercent = Math.max(0, Math.min(100, percent));
  successSince = null;
  lastStepTime = 0;
  if (initialIntensity !== undefined && initialIntensity !== null) {
    controllerIntensity = clampWorking(initialIntensity);
  }
  controlState = "IDLE";
}

function forceZeroController(reason) {
  controllerIntensity = 0;
  targetPercent = null;
  for (const finger of FINGER_KEYS) targetPerFinger[finger] = null;
  successSince = null;
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
    // 오차가 허용범위(±5% 등) 안에 들어오는 순간 신전 기능이 없으니 더
    // 구부러지지 않도록 그 즉시 전기를 끊는다. "성공"으로 확정해서 목표를
    // 지우는 것과는 별개 -- 그건 아래처럼 잠깐 더 유지되는지 지켜본 뒤 결정.
    controllerIntensity = 0;
    if (successSince === null) successSince = now;
    const heldForS = (now - successSince) / 1000;
    if (heldForS >= config.control.successHoldSeconds) {
      controllerIntensity = 0;
      controlState = "SUCCESS";
      // SUCCESS는 종료 상태: target을 지워야 자극이 꺼진 뒤 손이 풀리면서
      // 오차가 다시 벌어져 증가↔성공을 무한 반복하는 걸 막을 수 있다.
      targetPercent = null;
      for (const finger of FINGER_KEYS) targetPerFinger[finger] = null;
      notifyStimStopped();
      startCooldown();
      armedCh1 = false;
      return;
    }
    controlState = "HOLDING";
    return;
  }
  successSince = null;

  if (now - lastStepTime < config.control.controlPeriodMs) {
    controlState = error > 0 ? "INCREASING" : "DECREASING";
    return;
  }
  lastStepTime = now;

  // 오차가 클수록 크게, 목표에 가까워질수록 작게 조절되도록 진짜 비례(P) 계산.
  // 신전 기능이 없어서 "부족한 쪽"은 조심스럽게(kpUp 작게), "넘친 쪽"은 빠르게
  // 회수하도록(kpDown 크게) 방향별로 게인을 다르게 둔다. maxStepUp/maxStepDown은
  // 그래도 남아있는 안전상 최종 한도.
  const gain = error > 0 ? config.control.kpUp : config.control.kpDown;
  const rawStep = gain * error;
  const step = Math.max(-config.control.maxStepDown, Math.min(config.control.maxStepUp, rawStep));
  const before = controllerIntensity;
  controllerIntensity = clampWorkingFloat(controllerIntensity + step);
  controlState = step > 0.01 ? "INCREASING" : step < -0.01 ? "DECREASING" : "HOLDING";

  // control_period_ms(기본 0.5초)에 한 번만 실행되는 구간이라 스팸 걱정 없이
  // 매번 사람이 읽을 수 있는 진행상황 문장을 남긴다. (소수점까지 보여줘야
  // "정수로는 안 움직이는 것처럼 보이지만 실제로는 조금씩 오르고 있다"가
  // 눈에 보인다 -- 위에서 고친 반올림 버그를 다시 놓치지 않기 위함)
  if (Math.abs(controllerIntensity - before) > 0.001) {
    const reason = error > 0 ? "덜 구부러짐" : "더 구부러짐";
    const sign = error > 0 ? "+" : "";
    logControl(`${reason} (오차 ${sign}${error.toFixed(0)}%) → 자극값 ${before.toFixed(1)}→${controllerIntensity.toFixed(1)}`);
  }
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
  if (handLost) return { ok: false, reason: "오른손(ACTUAL)을 카메라가 놓쳤습니다." };
  if (continuousStimExceeded()) return { ok: false, reason: "최대 연속 자극 시간을 초과했습니다." };
  if (totalTimeExceeded()) return { ok: false, reason: "전체 실험 제한시간을 초과했습니다." };
  return { ok: true };
}

function triggerEmergencyStop(reason) {
  const wasTripped = safetyTripped;
  safetyTripped = true;
  safetyTripReason = reason;
  forceZeroController(`🛑 안전 정지: ${reason}`);
  if (serialLink) serialLink.stopAll().catch(() => {});
  armedCh1 = false;
  resetStartControlButton(); // 비상정지 후엔 반드시 다시 "▶ 제어 시작"을 눌러야만 재개됨
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

async function driveHardwareIfNeeded(intensity) {
  if (!serialLink || !serialLink.isConnected() || !serialLink.handshakeOk) return;
  if (!armedCh1) {
    const ok = await serialLink.arm(1);
    if (!ok) return;
    armedCh1 = true;
  }
  const ttl = clampTtl(config.safety.commandTtlMs);
  await serialLink.setIntensity(1, clampHardware(intensity), ttl);
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
    return;
  }
  elapsedDisplay.textContent = `${Math.floor((performance.now() - experimentStart) / 1000)}s`;
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
