const orbit = document.getElementById("orbit");
const ribbon = document.getElementById("ribbon");
const mixer = document.getElementById("mixer");
const listenBtn = document.getElementById("listenBtn");
const listenLabel = document.getElementById("listenLabel");
const micSelect = document.getElementById("micSelect");
const emotionEn = document.getElementById("emotionEn");
const emotionKo = document.getElementById("emotionKo");
const confidence = document.getElementById("confidence");
const hint = document.getElementById("hint");
const liveDot = document.getElementById("liveDot");
const statusLabel = document.getElementById("statusLabel");
const modelLabel = document.getElementById("modelLabel");
const latencyEl = document.getElementById("latency");
const metricContext = document.getElementById("metricContext");
const mInfer = document.getElementById("mInfer");
const mFeat = document.getElementById("mFeat");
const mTotal = document.getElementById("mTotal");
const mAvg = document.getElementById("mAvg");
const mP95 = document.getElementById("mP95");
const mRate = document.getElementById("mRate");
const metricInfer = document.querySelector('.metric[data-key="infer"]');
const metricTotal = document.querySelector('.metric[data-key="total"]');

let config = null;
let audioCtx = null;
let captureNode = null;
let source = null;
let analyser = null;
let mediaStream = null;
let socket = null;
let listening = false;
let selectedMicId = "";
let listingMics = false;
let raf = 0;
let timeData = new Float32Array(1024);
let freqData = new Uint8Array(512);
let emotionColor = "#cbb489";
let rmsLevel = 0;
let keys = {};
let ribbonColors = [];
const METRIC_WINDOW = 50;
let metricHist = [];
let lastRecvAt = 0;
let intervalHist = [];

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  return dpr;
}

function downsample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.round(input.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const t = x - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

async function loadConfig() {
  const res = await fetch("/api/config");
  config = await res.json();
  const loaded = config.loaded ? "ready" : "not loaded";
  modelLabel.textContent = `${config.model_name || "model"} · ${loaded}`;
  const hopMs = Math.round((config.hop_ms || config.hop_sec * 1000) || 180);
  const device = (config.device || "cpu").replace("cuda:0", "cuda");
  if (metricContext) {
    metricContext.textContent = `${device} · hop ${hopMs} ms · win ${config.window_sec}s`;
  }
  mixer.innerHTML = "";
  keys = {};
  config.labels.forEach((label) => {
    const meta = config.emotions[label];
    const el = document.createElement("div");
    el.className = "key";
    el.style.setProperty("--tone", meta.color);
    el.innerHTML = `<div class="fill"></div><div class="name">${label}</div><div class="pct">0%</div>`;
    mixer.appendChild(el);
    keys[label] = el;
  });
}

function setChrome(state) {
  liveDot.classList.toggle("live", state === "live");
  liveDot.classList.toggle("silent", state === "silent");
  statusLabel.textContent = state;
}

function fmtMs(value) {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 10) return `${value.toFixed(1)} ms`;
  return `${Math.round(value)} ms`;
}

function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

function mean(values) {
  if (!values.length) return null;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function resetMetrics() {
  metricHist = [];
  intervalHist = [];
  lastRecvAt = 0;
  if (mInfer) mInfer.textContent = "—";
  if (mFeat) mFeat.textContent = "—";
  if (mTotal) mTotal.textContent = "—";
  if (mAvg) mAvg.textContent = "—";
  if (mP95) mP95.textContent = "—";
  if (mRate) mRate.textContent = "—";
  if (latencyEl) latencyEl.textContent = "— ms";
  metricInfer?.classList.remove("warn");
  metricTotal?.classList.remove("warn");
}

function updateTelemetry(msg) {
  const hopMs = msg.hop_ms || (config && config.hop_ms) || 180;
  const infer = Number(msg.infer_ms) || 0;
  const feat = Number(msg.feat_ms) || 0;
  const total = Number(msg.latency_ms) || 0;
  const now = performance.now();
  if (lastRecvAt) {
    intervalHist.push(now - lastRecvAt);
    if (intervalHist.length > METRIC_WINDOW) intervalHist.shift();
  }
  lastRecvAt = now;

  if (!msg.silent) {
    metricHist.push(total);
    if (metricHist.length > METRIC_WINDOW) metricHist.shift();
  }

  if (mInfer) mInfer.textContent = msg.silent ? "skip" : fmtMs(infer);
  if (mFeat) mFeat.textContent = msg.silent ? "—" : fmtMs(feat);
  if (mTotal) mTotal.textContent = fmtMs(total);
  if (latencyEl) latencyEl.textContent = fmtMs(total);

  const avg = mean(metricHist);
  const p95 = percentile(metricHist, 0.95);
  if (mAvg) mAvg.textContent = avg == null ? "—" : fmtMs(avg);
  if (mP95) mP95.textContent = p95 == null ? "—" : fmtMs(p95);

  const interval = mean(intervalHist);
  if (mRate) {
    mRate.textContent = interval ? `${(1000 / interval).toFixed(1)} Hz` : "—";
  }

  const overrun = Boolean(msg.overrun) || (!msg.silent && total > hopMs);
  metricInfer?.classList.toggle("warn", !msg.silent && infer > hopMs);
  metricTotal?.classList.toggle("warn", overrun);
}

function applyPrediction(msg) {
  const silent = msg.silent;
  const noise = msg.source === "noise";
  emotionEn.textContent = noise ? "Not speech" : silent ? "Silence" : capitalize(msg.emotion);
  emotionKo.textContent = noise ? "목소리 아님" : silent ? "고요" : msg.ko;
  const pct = Math.round(msg.confidence * 100);
  const rmsTxt = `rms ${msg.rms.toFixed(4)}`;
  confidence.textContent = noise
    ? `not a voice · ${rmsTxt}`
    : silent
    ? `below speech threshold · ${rmsTxt}`
    : `${pct}%  confidence · ${msg.source || "model"} · ${rmsTxt}`;
  emotionColor = silent ? "#8f867b" : msg.color;
  document.documentElement.style.setProperty("--emotion", emotionColor);
  rmsLevel = msg.rms;
  setChrome(silent ? "silent" : "live");
  updateTelemetry(msg);

  config.labels.forEach((name) => {
    const el = keys[name];
    const p = msg.probs[name] || 0;
    el.classList.toggle("active", !silent && name === msg.emotion);
    el.querySelector(".fill").style.width = `${Math.max(2, p * 100)}%`;
    el.querySelector(".pct").textContent = `${Math.round(p * 100)}%`;
  });

  ribbonColors.push(silent ? "#1b1916" : msg.color);
  if (ribbonColors.length > 240) ribbonColors.shift();
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function drawOrbit() {
  const ctx = orbit.getContext("2d");
  const dpr = fitCanvas(orbit);
  const w = orbit.width;
  const h = orbit.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.34;
  ctx.clearRect(0, 0, w, h);

  if (analyser) {
    analyser.getFloatTimeDomainData(timeData);
    analyser.getByteFrequencyData(freqData);
  }

  ctx.strokeStyle = "rgba(244,238,230,0.08)";
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 1.28, 0, Math.PI * 2);
  ctx.stroke();

  const bars = 96;
  for (let i = 0; i < bars; i += 1) {
    const mag = (freqData[i + 4] || 0) / 255;
    const inner = radius * 1.18;
    const outer = inner + mag * radius * 0.42;
    const a = (i / bars) * Math.PI * 2 - Math.PI / 2;
    ctx.strokeStyle = `rgba(203,180,137,${0.15 + mag * 0.65})`;
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
    ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer);
    ctx.stroke();
  }

  ctx.beginPath();
  const n = timeData.length;
  for (let i = 0; i < n; i += 1) {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    const r = radius * (0.72 + timeData[i] * 0.55);
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = emotionColor;
  ctx.globalAlpha = 0.85;
  ctx.lineWidth = 1.8 * dpr;
  ctx.stroke();
  ctx.globalAlpha = 1;

  const glow = radius * (0.42 + Math.min(rmsLevel * 8, 0.18));
  const g = ctx.createRadialGradient(cx, cy, 8, cx, cy, glow);
  g.addColorStop(0, hexToRgba(emotionColor, 0.22));
  g.addColorStop(1, hexToRgba(emotionColor, 0));
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cy, glow, 0, Math.PI * 2);
  ctx.fill();
}

function drawRibbon() {
  const ctx = ribbon.getContext("2d");
  fitCanvas(ribbon);
  const w = ribbon.width;
  const h = ribbon.height;
  ctx.clearRect(0, 0, w, h);
  const n = ribbonColors.length;
  if (!n) return;
  const bw = w / 240;
  ribbonColors.forEach((color, i) => {
    ctx.fillStyle = color;
    ctx.fillRect(i * bw, 0, bw + 1, h);
  });
}

function hexToRgba(hex, a) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function loop() {
  drawOrbit();
  drawRibbon();
  raf = requestAnimationFrame(loop);
}

function workletSource() {
  return `
    class CaptureProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const channel = inputs[0] && inputs[0][0];
        if (channel) {
          this.port.postMessage(channel.slice());
        }
        return true;
      }
    }
    registerProcessor("capture-processor", CaptureProcessor);
  `;
}

function isPlaybackDevice(label) {
  return /monitor|loopback|output|stereo mix|what u hear|wave out/i.test(label);
}

function isLikelyHeadsetMic(label) {
  return /headset|headphone|earbud|airpod|usb|boom|mic/i.test(label) && !isPlaybackDevice(label);
}

async function refreshMicList() {
  if (!micSelect || !navigator.mediaDevices?.enumerateDevices) return;
  listingMics = true;
  try {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices.filter((d) => d.kind === "audioinput");
  const current = selectedMicId || micSelect.value;
  micSelect.innerHTML = "";
  const fallback = document.createElement("option");
  fallback.value = "";
  fallback.textContent = "Default microphone";
  micSelect.appendChild(fallback);
  inputs.forEach((device, i) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    const label = device.label || `Microphone ${i + 1}`;
    option.textContent = isPlaybackDevice(label) ? `${label} (playback — skip)` : label;
    micSelect.appendChild(option);
  });
  const preferred = inputs.find((d) => isLikelyHeadsetMic(d.label));
  if (current && inputs.some((d) => d.deviceId === current)) {
    micSelect.value = current;
  } else if (preferred) {
    micSelect.value = preferred.deviceId;
  }
  selectedMicId = micSelect.value;
  } finally {
    listingMics = false;
  }
}

function micConstraints() {
  const audio = {
    channelCount: 1,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: true,
  };
  if (selectedMicId) audio.deviceId = { exact: selectedMicId };
  return { audio };
}

function tapGraph(node) {
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  node.connect(mute);
  mute.connect(audioCtx.destination);
}

async function startMic() {
  selectedMicId = micSelect ? micSelect.value : selectedMicId;
  const requestedId = selectedMicId;
  mediaStream = await navigator.mediaDevices.getUserMedia(micConstraints());
  await refreshMicList();
  if (micSelect && micSelect.value && micSelect.value !== requestedId) {
    mediaStream.getTracks().forEach((track) => track.stop());
    selectedMicId = micSelect.value;
    mediaStream = await navigator.mediaDevices.getUserMedia(micConstraints());
  }

  const trackLabel = mediaStream.getAudioTracks()[0]?.label || "";
  if (isPlaybackDevice(trackLabel)) {
    hint.textContent =
      "This input is headphone playback, not a mic. Choose a device named Microphone or Headset.";
  }

  audioCtx = new AudioContext();
  if (audioCtx.state === "suspended") await audioCtx.resume();

  source = audioCtx.createMediaStreamSource(mediaStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.72;
  source.connect(analyser);

  const sendPcm = (samples) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const pcm = downsample(samples, audioCtx.sampleRate, config.sample_rate);
    socket.send(pcm.buffer);
  };

  let workletReady = false;
  try {
    const blob = new Blob([workletSource()], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    await audioCtx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    captureNode = new AudioWorkletNode(audioCtx, "capture-processor");
    captureNode.port.onmessage = (event) => sendPcm(event.data);
    source.connect(captureNode);
    tapGraph(captureNode);
    workletReady = true;
  } catch (err) {
    workletReady = false;
  }
  if (!workletReady) {
    const processor = audioCtx.createScriptProcessor(2048, 1, 1);
    processor.onaudioprocess = (event) => {
      sendPcm(event.inputBuffer.getChannelData(0).slice());
    };
    source.connect(processor);
    tapGraph(processor);
    captureNode = processor;
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws/stream`);
  socket.binaryType = "arraybuffer";
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = () => reject(new Error("Could not open the live stream."));
  });
  socket.onmessage = (event) => applyPrediction(JSON.parse(event.data));
  socket.onclose = () => {
    if (listening) hint.textContent = "Stream closed. Tap listen to reconnect.";
  };
}

async function stopMic() {
  listening = false;
  listenBtn.classList.remove("on");
  listenLabel.textContent = "Listen";
  setChrome("idle");
  resetMetrics();
  if (socket) {
    socket.close();
    socket = null;
  }
  if (captureNode) captureNode.disconnect();
  if (source) source.disconnect();
  if (audioCtx) await audioCtx.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  audioCtx = null;
  captureNode = null;
  source = null;
  analyser = null;
  mediaStream = null;
}

listenBtn.addEventListener("click", async () => {
  if (listening) {
    await stopMic();
    hint.textContent = `The model hears a ${config.window_sec}s window ending now.`;
    return;
  }
  try {
    hint.textContent = "Allow the microphone, then speak.";
    await startMic();
    listening = true;
    listenBtn.classList.add("on");
    listenLabel.textContent = "Stop";
    setChrome("live");
    resetMetrics();
    hint.textContent = "Speak whenever you like. Each update uses the audio gathered so far.";
  } catch (err) {
    hint.textContent = err.message || "Microphone permission was denied.";
  }
});

if (micSelect) {
  micSelect.addEventListener("change", async () => {
    if (listingMics) return;
    selectedMicId = micSelect.value;
    if (!listening) return;
    await stopMic();
    try {
      await startMic();
      listening = true;
      listenBtn.classList.add("on");
      listenLabel.textContent = "Stop";
      setChrome("live");
      resetMetrics();
      hint.textContent = "Switched input. Speak to check the level.";
    } catch (err) {
      hint.textContent = err.message || "Could not open that microphone.";
    }
  });
}

window.addEventListener("resize", () => {
  fitCanvas(orbit);
  fitCanvas(ribbon);
});
window.visualViewport?.addEventListener("resize", () => {
  fitCanvas(orbit);
  fitCanvas(ribbon);
});

loadConfig().then(() => {
  loop();
});
