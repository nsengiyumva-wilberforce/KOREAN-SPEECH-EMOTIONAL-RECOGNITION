"""Live KITE emotion server.

Uses the class-weighted XLS-R checkpoint with logit adjustment (tau 0.75).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from hyperpyyaml import load_hyperpyyaml

RECIPE = Path(__file__).resolve().parents[1]
if str(RECIPE) not in sys.path:
    sys.path.insert(0, str(RECIPE))
os.chdir(RECIPE)

from train_with_wav2vec2 import EmoIdBrain, dataio_prep  # noqa: E402
from combine_margin import TAU, WEIGHTED, train_log_prior  # noqa: E402

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0
INFER_HOP_SEC = 0.2
MAX_WAV_SEC = 4.0
SILENCE_RMS = 0.005
MODEL_NAME = "XLS-R + logit adjustment"

EMOTION_META = {
    "anger": {"ko": "분노", "color": "#e25b3a"},
    "happiness": {"ko": "행복", "color": "#e4b84a"},
    "neutral": {"ko": "평온", "color": "#c5beb6"},
    "sadness": {"ko": "슬픔", "color": "#4a7fa3"},
    "surprise": {"ko": "놀람", "color": "#e07a8a"},
}

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="KITE Speech Emotion Recognition")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

RUNTIME = {
    "model": None,
    "log_prior": None,
    "labels": [],
    "device": None,
    "on_gpu": None,
    "loaded": False,
}


def pick_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def load_brain(hparams_path, device):
    """Load one finished run and keep it on the inference device."""
    with open(hparams_path, encoding="utf-8") as handle:
        hparams = load_hyperpyyaml(handle)
    hparams["skip_prep"] = True
    datasets = dataio_prep(hparams)
    hparams["wav2vec2"] = hparams["wav2vec2"].to(device)
    brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": str(device)},
        checkpointer=hparams["checkpointer"],
    )
    brain.on_evaluate_start(max_key=hparams["select_metric"])
    brain.modules.eval()
    return brain, datasets, hparams


def load_models(device: torch.device):
    print("[hser] loading class-weighted XLS-R", flush=True)
    model, datasets, hparams = load_brain(WEIGHTED, device)
    if device.type == "cuda":
        model.modules.to(device)
    RUNTIME.update(
        {
            "model": model,
            "log_prior": train_log_prior(datasets, hparams),
            "labels": list(hparams["class_names"]),
            "device": device,
            "loaded": True,
        }
    )
    silence = np.zeros(int(SAMPLE_RATE * WINDOW_SEC), dtype=np.float32)
    _log_probs(model, silence)


def _log_probs(brain, wave: np.ndarray) -> torch.Tensor:
    device = RUNTIME["device"]
    wavs = torch.from_numpy(np.ascontiguousarray(wave)).float().unsqueeze(0).to(device)
    lens = torch.ones(1, device=device)
    use_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_amp
    ):
        outputs = brain.modules.wav2vec2(wavs, lens)
        outputs = brain.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = brain.modules.output_mlp(outputs)
        return brain.hparams.log_softmax(outputs).squeeze(0).float().cpu()


def _sync_device():
    device = RUNTIME["device"]
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def speech_span(wave: np.ndarray) -> np.ndarray:
    """Keep the spoken samples. Training clips contain speech, not long silence."""
    frame = int(0.02 * SAMPLE_RATE)
    if wave.size < frame:
        return wave[:0]
    usable = wave.size - (wave.size % frame)
    frames = wave[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(rms.max()) if rms.size else 0.0
    if peak < 0.008:
        return wave[:0]
    voiced = np.flatnonzero(rms >= max(0.008, 0.2 * peak))
    # A tap or shake is only a couple of frames. Speech keeps going.
    if voiced.size < 10:
        return wave[:0]
    pad = int(0.1 * SAMPLE_RATE)
    start = max(0, int(voiced[0]) * frame - pad)
    end = min(wave.size, int(voiced[-1] + 1) * frame + pad)
    clip = wave[start:end]
    max_len = int(MAX_WAV_SEC * SAMPLE_RATE)
    if clip.size > max_len:
        clip = clip[-max_len:]
    if clip.size < int(0.25 * SAMPLE_RATE):
        return wave[:0]
    return np.ascontiguousarray(clip)


def sounds_like_speech(clip: np.ndarray) -> bool:
    """Reject thumps, shakes, and hiss. The emotion model only knows speech."""
    usable = clip.size - (clip.size % 2)
    if usable < int(0.2 * SAMPLE_RATE):
        return False
    frame = int(0.02 * SAMPLE_RATE)
    frames = clip[: clip.size - (clip.size % frame)].reshape(-1, frame)
    energy = np.mean(np.square(frames), axis=1)
    total = float(energy.sum()) + 1e-12
    loudest = float(np.sort(energy)[-2:].sum())
    if loudest / total > 0.85:
        return False
    windowed = clip[:usable] * np.hanning(usable)
    spec = np.abs(np.fft.rfft(windowed))[1:]
    if spec.size == 0:
        return False
    flatness = float(
        np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12)
    )
    freqs = np.fft.rfftfreq(usable, 1.0 / SAMPLE_RATE)[1:]
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    return flatness < 0.5 and 150.0 <= centroid <= 4500.0


def predict_wave(wave: np.ndarray, ema: np.ndarray | None):
    t0 = time.perf_counter()
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(wave)))) if wave.size else 0.0
    labels = RUNTIME["labels"]
    n = len(labels)
    clip = speech_span(wave)
    clip_rms = (
        float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
    )
    if clip.size == 0 or peak < 4e-4 or clip_rms < SILENCE_RMS:
        kind = "silence"
    elif not sounds_like_speech(clip):
        kind = "noise"
    else:
        kind = "speech"

    if kind != "speech":
        probs = ema if ema is not None else np.full(n, 1.0 / n, dtype=np.float32)
        total_ms = (time.perf_counter() - t0) * 1000
        return probs, True, rms, total_ms, 0.0, kind

    wave = clip

    t_infer = time.perf_counter()
    chosen_log = _log_probs(RUNTIME["model"], wave) - TAU * RUNTIME["log_prior"]
    source = "logit adjustment"
    _sync_device()
    raw = torch.softmax(chosen_log, dim=-1).numpy().astype(np.float32)
    infer_ms = (time.perf_counter() - t_infer) * 1000
    # Keep a light blend across speech decisions only. Silence must not
    # pull the next phrase back to neutral.
    if ema is not None and ema.shape == raw.shape:
        raw = (0.25 * ema + 0.75 * raw).astype(np.float32)
    total_ms = (time.perf_counter() - t0) * 1000
    return raw, False, rms, total_ms, infer_ms, source


def pack_result(probs, silent, rms, latency_ms, infer_ms, source):
    labels = RUNTIME["labels"]
    dist = {label: float(probs[i]) for i, label in enumerate(labels)}
    top = max(dist, key=dist.get)
    hop_ms = INFER_HOP_SEC * 1000
    return {
        "emotion": top,
        "ko": EMOTION_META.get(top, {}).get("ko", top),
        "color": EMOTION_META.get(top, {}).get("color", "#c5beb6"),
        "confidence": float(dist[top]),
        "probs": dist,
        "silent": silent,
        "rms": rms,
        "latency_ms": round(latency_ms, 1),
        "feat_ms": 0.0,
        "infer_ms": round(infer_ms, 1),
        "hop_ms": round(hop_ms, 1),
        "overrun": (not silent) and latency_ms > hop_ms,
        "source": source,
        "ts": time.time(),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config():
    labels = RUNTIME["labels"]
    return {
        "sample_rate": SAMPLE_RATE,
        "window_sec": WINDOW_SEC,
        "hop_sec": INFER_HOP_SEC,
        "loaded": RUNTIME["loaded"],
        "device": str(RUNTIME["device"]),
        "labels": labels,
        "silence_rms": SILENCE_RMS,
        "hop_ms": round(INFER_HOP_SEC * 1000, 1),
        "window_ms": round(WINDOW_SEC * 1000, 1),
        "model_name": MODEL_NAME,
        "tau": TAU,
        "emotions": {
            label: EMOTION_META.get(label, {"ko": label, "color": "#c5beb6"})
            for label in labels
        },
    }


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    window = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(INFER_HOP_SEC * SAMPLE_RATE)
    # Grow with whatever has arrived. Cap at the most recent few seconds.
    buf = np.zeros(0, dtype=np.float32)
    pending = 0
    ema = None
    stop = asyncio.Event()

    async def reader():
        nonlocal buf, pending
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if not data:
                    continue
                chunk = np.frombuffer(data, dtype=np.float32)
                if chunk.size == 0:
                    continue
                buf = np.concatenate([buf, chunk])
                if buf.size > window:
                    buf = buf[-window:].copy()
                pending += chunk.size
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()

    async def inferer():
        nonlocal pending, ema
        try:
            while not stop.is_set():
                await asyncio.sleep(0.05)
                if pending < hop:
                    continue
                pending = 0
                wave = buf.copy()
                probs, silent, rms, latency, infer_ms, source = await asyncio.to_thread(
                    predict_wave, wave, ema
                )
                if not silent:
                    ema = probs
                await ws.send_json(
                    pack_result(probs, silent, rms, latency, infer_ms, source)
                )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            stop.set()

    tasks = [asyncio.create_task(reader()), asyncio.create_task(inferer())]
    await stop.wait()
    for task in tasks:
        task.cancel()


def local_ipv4s():
    """IPv4 addresses other devices on the LAN can use to reach this machine."""
    addresses = set()
    skip = ("docker", "br-", "veth", "virbr")
    try:
        listed = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True)
    except (OSError, subprocess.CalledProcessError):
        listed = ""
    for line in listed.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2] != "inet":
            continue
        if parts[1].startswith(skip):
            continue
        ip = parts[3].split("/")[0]
        if not ip.startswith("127."):
            addresses.add(ip)
    if addresses:
        return sorted(addresses)
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        addresses.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    return sorted(addresses)


def ensure_tls(ips):
    """Write a local certificate that covers this machine's LAN addresses.

    Phone browsers only allow the microphone on HTTPS (or on localhost).
    A self-signed certificate is enough after the warning is accepted.
    """
    folder = Path(__file__).resolve().parent / "certs"
    folder.mkdir(exist_ok=True)
    cert = folder / "cert.pem"
    key = folder / "key.pem"
    stamp = folder / "addresses.txt"
    wanted = "\n".join(["localhost", "127.0.0.1", *ips])
    if cert.exists() and key.exists() and stamp.is_file() and stamp.read_text() == wanted:
        return cert, key
    names = ["DNS:localhost", "IP:127.0.0.1", *[f"IP:{ip}" for ip in ips]]
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "825",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=kite-emotion",
            "-addext",
            "subjectAltName=" + ",".join(names),
        ],
        check=True,
        capture_output=True,
    )
    stamp.write_text(wanted)
    return cert, key


def parse_args():
    parser = argparse.ArgumentParser(description="KITE live emotion server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7861, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve plain HTTP. The microphone then works only on this computer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.cpu)
    load_models(device)
    print(
        f"[hser] ready · {MODEL_NAME} · tau {TAU} · {device}",
        flush=True,
    )
    ips = local_ipv4s()
    scheme = "http" if args.http else "https"
    ssl_kwargs = {}
    if not args.http:
        cert, key = ensure_tls(ips)
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    print(f"[hser] this computer: {scheme}://127.0.0.1:{args.port}", flush=True)
    for ip in ips:
        print(f"[hser] another device: {scheme}://{ip}:{args.port}", flush=True)
    if not args.http:
        print(
            "[hser] accept the certificate warning, then allow the microphone",
            flush=True,
        )
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
