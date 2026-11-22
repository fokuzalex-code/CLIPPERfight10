import json, os, re, subprocess
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
WHISPER = os.getenv("WHISPER_MODEL", "tiny")
HOSTS = ("kick.com", "twitch.tv")
FONT = next((f for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/system/fonts/Roboto-Bold.ttf",
                         "/system/fonts/Roboto-Regular.ttf", "/system/fonts/DroidSans.ttf") if os.path.exists(f)), "")
PROFILES = {"rapido": (720, 1280, "ultrafast", 28), "equilibrado": (1080, 1920, "veryfast", 23), "alta": (1080, 1920, "medium", 20)}
HOT = re.compile(r"no puede ser|vamos|increíble|jaja|wow|omg|clutch|insane|\bgg\b|what the|noo+", re.I)

def sh(c):
    return subprocess.run(c, capture_output=True, check=True).stdout

def probe(p):
    j = json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height:format=duration", "-of", "json", str(p)]))
    s = j["streams"][0]
    return s["width"], s["height"], float(j["format"]["duration"])

def download(url, d):
    import yt_dlp
    o = {"outtmpl": str(d / "src.%(ext)s"), "format": "best[height<=720]/best", "quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(o) as y:
        return Path(y.prepare_filename(y.extract_info(url, download=True)))

def energy(p):
    raw = sh(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"])
    a = np.frombuffer(raw, np.int16).astype(np.float32)
    n = len(a) // 8000
    return np.sqrt((a[: n * 8000].reshape(n, 8000) ** 2).mean(1))

def transcribe(p):  # solo para analizar; NO se queman subtítulos
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []  # sin Whisper: se detecta solo por audio
    segs, _ = WhisperModel(WHISPER, compute_type="int8").transcribe(str(p), vad_filter=True)
    return [(s.start, s.end, s.text.strip()) for s in segs]

def pick(en, segs, o):
    e = (en - en.min()) / (np.ptp(en) + 1e-9)
    w, kw = np.zeros(len(e)), np.zeros(len(e))
    for a, b, x in segs:
        r = slice(int(a), min(int(b) + 1, len(e)))
        w[r] += len(x.split()) / max(b - a, 1)
        if HOT.search(x): kw[r] += 1
    w /= w.max() + 1e-9
    f = .55 * e + .25 * w + .2 * kw
    L = min((o["min_s"] + o["max_s"]) // 2, len(e))
    c = np.concatenate([[0], np.cumsum(f)])
    cand = sorted(((c[i + L] - c[i]) / L, i) for i in range(0, max(len(e) - L, 1), 5))[::-1]
    out = []
    for sc, i in cand:
        if all(abs(i - s) >= L for s, _, _ in out):
            out.append((i, L, int(min(99, sc / (cand[0][0] + 1e-9) * 95))))
        if len(out) == o["n"]: break
    return out

def face_x(p, s, L, W):
    try:
        import cv2
    except ImportError:
        return .5
    cap = cv2.VideoCapture(str(p))
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    xs = []
    for k in range(6):
        cap.set(cv2.CAP_PROP_POS_MSEC, (s + L * k / 6) * 1000)
        ok, f = cap.read()
        if not ok: continue
        fs = det.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1.2, 5)
        if len(fs):
            x, _, w, _ = max(fs, key=lambda r: r[2] * r[3]); xs.append((x + w / 2) / W)
    return float(np.median(xs)) if xs else .5

STOP = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos todos uno les ni contra otros ese eso ante esto antes algunos qué unos yo otro otra él tanto esa estos mucho nada cual poco ella estar estas algo the and you that this was for are with have just like what its not but".split())

def topics(text, k=2):
    from collections import Counter
    w = [x for x in re.findall(r"[a-záéíóúñ]{5,}", text.lower()) if x not in STOP]
    return " ".join("#" + x for x, _ in Counter(w).most_common(k))

def meta(text, tags):
    t = (text[:60] or "Momento épico")
    try:
        import requests
        r = requests.post(OLLAMA + "/api/generate", timeout=60, json={"model": LLM, "stream": False,
            "prompt": "Escribe un título gancho de máximo 8 palabras para este clip. Solo el título:\n" + text[:1500]})
        t = r.json()["response"].strip().strip('"') or t
    except Exception:
        pass
    return t, (tags + " " + topics(text)).strip()

def render(src, s, L, out, mode, cx, wm, W, H, q="equilibrado", bar=False):
    ow, oh, preset, crf = PROFILES.get(q, PROFILES["equilibrado"])
    cw = min(int(H * 9 / 16) // 2 * 2, W // 2 * 2)
    if mode == "fill" and cw < W:
        x = int(max(0, min(W - cw, cx * W - cw / 2)))
        g = f"[0:v]crop={cw}:{H}:{x}:0,scale={ow}:{oh}[v0]"
    else:
        g = (f"[0:v]split[a][b];[a]scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},boxblur=20[bg];"
             f"[b]scale={ow}:-2[fg];[bg][fg]overlay=0:(H-h)/2[v0]")
    # barra de progreso animada (retención): una barra roja que avanza durante el clip
    g += (f";[v0][1:v]overlay=x='-w+w*t/{L}':y=H-h:shortest=1[v1]" if bar else ";[v0]null[v1]")
    t = re.sub(r"[^\w@.#\- ]", "", wm)
    ff = f"fontfile={FONT}:" if FONT else ""
    g += (f";[v1]drawtext={ff}text='{t}':fontcolor=white@0.75:fontsize={int(oh * .023)}:x=(w-tw)/2:y=h-{int(oh * .1)}[v]"
          if t else ";[v1]null[v]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(s), "-i", str(src)]
    if bar: cmd += ["-f", "lavfi", "-i", f"color=c=0xD8322B:s={ow}x{max(8, oh // 160)}:r=30:d={L}"]
    cmd += ["-t", str(L), "-filter_complex", g, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", preset,
            "-crf", str(crf)]
    if q != "rapido": cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]  # audio a volumen parejo
    sh(cmd + ["-c:a", "aac", "-movflags", "+faststart", str(out)])

def process(src, d, o, cb):
    if src.startswith("http"):
        if not (urlparse(src).hostname or "").endswith(HOSTS):
            raise ValueError("Solo se admiten enlaces de Kick o Twitch")
        cb(.05, "Descargando"); src = download(src, d)
    W, H, _ = probe(src)
    (d / "meta.json").write_text(json.dumps(dict(src=str(src), o=o, W=W, H=H)))
    cb(.2, "Analizando audio"); en = energy(src)
    cb(.3, "Transcribiendo para detectar momentos"); segs = transcribe(src)
    cb(.6, "Eligiendo los mejores momentos"); picks = pick(en, segs, o)
    out = []
    for k, (s, L, sc) in enumerate(picks):
        cb(.6 + .4 * k / max(len(picks), 1), f"Renderizando clip {k + 1} de {len(picks)}")
        text = " ".join(x for a, b, x in segs if a < s + L and b > s)
        title, tags = meta(text, o["hashtags"])
        name = f"clip{k + 1}.mp4"
        render(src, s, L, d / name, o["mode"], face_x(src, s, L, W) if o["mode"] == "fill" else .5,
               o["watermark"], W, H, o["quality"], o["bar"])
        out.append(dict(file=name, start=int(s), dur=int(L), score=sc, title=title, tags=tags))
    return out

def rerender(d, k, start, dur):
    m = json.loads((d / "meta.json").read_text()); o = m["o"]
    cx = face_x(m["src"], start, dur, m["W"]) if o["mode"] == "fill" else .5
    render(m["src"], start, dur, d / f"clip{k + 1}.mp4", o["mode"], cx, o["watermark"], m["W"], m["H"], o["quality"], o["bar"])
