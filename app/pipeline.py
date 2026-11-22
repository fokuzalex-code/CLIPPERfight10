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
    r = subprocess.run(c, capture_output=True)
    if r.returncode:
        raise RuntimeError((r.stderr.decode(errors="ignore").strip().splitlines() or ["Error en FFmpeg"])[-1])
    return r.stdout

def probe(p):
    j = json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height:format=duration", "-of", "json", str(p)]))
    s = j["streams"][0]
    return s["width"], s["height"], float(j["format"]["duration"])

def valid_url(u):
    h = (urlparse(u).hostname or "").lower()
    return u.startswith("http") and any(h == x or h.endswith("." + x) for x in HOSTS)

def download(url, d, a=0, b=0):
    import yt_dlp
    o = {"outtmpl": str(d / "src.%(ext)s"), "format": "best[height<=720]/best", "quiet": True, "noplaylist": True}
    if a or b:  # solo baja el tramo pedido, no todo el stream
        o["download_ranges"] = yt_dlp.utils.download_range_func(None, [(a * 60, b * 60 if b else float("inf"))])
        o["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(o) as y:
        return Path(y.prepare_filename(y.extract_info(url, download=True)))

def cut(src, d, a, b):
    out = d / "cut.mp4"
    sh(["ffmpeg", "-y", "-v", "error", "-ss", str(a * 60), "-i", str(src)] + (["-t", str((b - a) * 60)] if b else []) + ["-c", "copy", str(out)])
    return out

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

def track(p, s, L, W, H, n=8):
    """Detecta caras en n instantes del clip: [(t, [(cx, cy, w, h), ...] de mayor a menor)] o None si no hay OpenCV."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened(): return None
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if det.empty(): return None  # sin modelo de caras: se usa fondo difuminado en vez de fallar
    out = []
    for k in range(n):
        t = L * k / (n - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, (s + t) * 1000)
        ok, f = cap.read()
        fs = []
        if ok:
            sc = 640 / f.shape[1]
            g = cv2.cvtColor(cv2.resize(f, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
            fs = [((x + w / 2) / sc, (y + h / 2) / sc, w / sc, h / sc) for x, y, w, h in det.detectMultiScale(g, 1.2, 5, minSize=(24, 24))]
        out.append((t, sorted(fs, key=lambda f: -f[2] * f[3])))
    return out

def clamp(v, lo, hi): return max(lo, min(hi, v))
def med(v): v = sorted(v); return v[len(v) // 2]

def pw(ts, xs):  # posición x que se interpola linealmente entre muestras (seguimiento suave)
    e = f"{xs[-1]:.0f}"
    for i in range(len(ts) - 2, -1, -1):
        e = f"if(lt(t,{ts[i + 1]:.2f}),{xs[i]:.1f}+({xs[i + 1] - xs[i]:.1f})*(t-{ts[i]:.2f})/{max(ts[i + 1] - ts[i], .01):.2f},{e})"
    return e

def auto_mode(tr, W, H):
    big = [fs[0] for _, fs in tr if fs]
    if not big: return "fit"
    both = sum(1 for _, fs in tr if any(f[0] < W * .45 for f in fs) and any(f[0] > W * .55 for f in fs))
    if both >= max(2, len(tr) // 3): return "podcast"
    return "camara" if med([f[2] * f[3] for f in big]) / (W * H) > .02 else "gameplay"

def layout(mode, tr, W, H, ow, oh, L):
    """Devuelve (grafo de filtros terminado en [v0], modo realmente usado)."""
    fit = (f"[0:v]split[a][b];[a]scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},boxblur=20[bg];"
           f"[b]scale={ow}:-2[fg];[bg][fg]overlay=0:(H-h)/2[v0]")
    if not tr or mode == "fit": return fit, "fit"
    if mode == "auto": mode = auto_mode(tr, W, H)
    if mode == "fill": mode = "camara"
    fl = [f for _, fs in tr for f in fs]
    big = [fs[0] for _, fs in tr if fs]
    if mode == "podcast":  # dos personas: una arriba y otra abajo
        left, right = [f for f in fl if f[0] < W / 2], [f for f in fl if f[0] >= W / 2]
        if len(left) >= 2 and len(right) >= 2:
            cw = min(int(H * 9 / 8) // 2 * 2, W // 2 * 2); top = oh // 4 * 2
            xl = int(clamp(med([f[0] for f in left]) - cw / 2, 0, W - cw))
            xr = int(clamp(med([f[0] for f in right]) - cw / 2, 0, W - cw))
            return (f"[0:v]split[a][b];[a]crop={cw}:{H}:{xl}:0,scale={ow}:{top}[t];"
                    f"[b]crop={cw}:{H}:{xr}:0,scale={ow}:{oh - top}[u];[t][u]vstack[v0]"), "podcast"
        mode = "camara"
    if mode == "camara" and big:  # cámara sola: el recorte sigue a la persona durante todo el clip
        cw = int(H * 9 / 16) // 2 * 2
        if cw < W:
            last, xs = big[0][0], []
            for _, fs in tr:
                last = fs[0][0] if fs else last
                xs.append(last)
            sm = [sum(xs[max(0, i - 1):i + 2]) / len(xs[max(0, i - 1):i + 2]) for i in range(len(xs))]
            xs = [clamp(x - cw / 2, 0, W - cw) for x in sm]
            return f"[0:v]crop=w={cw}:h={H}:x='{pw([t for t, _ in tr], xs)}':y=0,scale={ow}:{oh}[v0]", "camara"
    if mode == "gameplay" and big:  # juego arriba, cámara del streamer abajo
        top = (oh * 2 // 3) // 2 * 2; bot = oh - top
        fw = int(clamp(med([f[2] for f in big]) * 3.5, W * .12, W * .5)) // 2 * 2
        fh = min(int(fw * bot / ow) // 2 * 2, H // 2 * 2)
        fx = int(clamp(med([f[0] for f in big]) - fw / 2, 0, W - fw)); fy = int(clamp(med([f[1] for f in big]) - fh / 2, 0, H - fh))
        return (f"[0:v]split=3[a][b][c];[a]scale={ow}:{top}:force_original_aspect_ratio=increase,crop={ow}:{top},boxblur=20[bg];"
                f"[b]scale={ow}:-2[fg];[bg][fg]overlay=0:(H-h)/2[g];[c]crop={fw}:{fh}:{fx}:{fy},scale={ow}:{bot}[cam];[g][cam]vstack[v0]"), "gameplay"
    return fit, "fit"

STOP = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos todos uno les ni contra otros ese eso ante esto antes algunos qué unos yo otro otra él tanto esa estos mucho nada cual poco ella estar estas algo the and you that this was for are with have just like what its not but".split())

def topics(text, k=2):
    from collections import Counter
    w = [x for x in re.findall(r"[a-záéíóúñ]{5,}", text.lower()) if x not in STOP]
    return " ".join("#" + x for x, _ in Counter(w).most_common(k))

def meta(text, tags):
    t = re.split(r"[.!?¡¿]", text)[0][:70].strip()
    try:
        import requests
        r = requests.post(OLLAMA + "/api/generate", timeout=60, json={"model": LLM, "stream": False,
            "prompt": "Escribe un título gancho de máximo 8 palabras para este clip. Solo el título:\n" + text[:1500]})
        t = r.json()["response"].strip().strip('"') or t
    except Exception:
        pass
    return t, (tags + " " + topics(text)).strip()

def render(src, s, L, out, mode, tr, wm, W, H, q="equilibrado", bar=False):
    ow, oh, preset, crf = PROFILES.get(q, PROFILES["equilibrado"])
    g, used = layout(mode, tr, W, H, ow, oh, L)
    g += (f";[v0][1:v]overlay=x='-w+w*t/{L}':y=H-h:shortest=1[v1]" if bar else ";[v0]null[v1]")
    t = re.sub(r"[^\w@.#\- ]", "", wm)
    ff = f"fontfile={FONT}:" if FONT else ""
    g += (f";[v1]drawtext={ff}text='{t}':fontcolor=white@0.75:fontsize={int(oh * .023)}:x=(w-tw)/2:y=h-{int(oh * .1)}[v]"
          if t else ";[v1]null[v]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(s), "-i", str(src)]
    if bar: cmd += ["-f", "lavfi", "-i", f"color=c=0xD8322B:s={ow}x{max(8, oh // 160)}:r=30:d={L}"]
    cmd += ["-t", str(L), "-filter_complex", g, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    if q != "rapido": cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    sh(cmd + ["-c:a", "aac", "-movflags", "+faststart", str(out)])
    return used

def process(src, d, o, cb):
    a, b = o["from_min"], o["to_min"]
    if src.startswith("http"):
        if not valid_url(src): raise ValueError("Solo se admiten enlaces de Kick o Twitch")
        cb(.05, "Descargando"); src = download(src, d, a, b)
    elif a or b:
        cb(.1, "Recortando el tramo"); src = cut(src, d, a, b)
    W, H, _ = probe(src)
    (d / "meta.json").write_text(json.dumps(dict(src=str(src), o=o, W=W, H=H)))
    cb(.2, "Analizando audio"); en = energy(src)
    cb(.3, "Transcribiendo para detectar momentos"); segs = transcribe(src)
    cb(.6, "Eligiendo los mejores momentos"); picks = pick(en, segs, o)
    out = []
    for k, (s, L, sc) in enumerate(picks):
        cb(.61 + .39 * k / max(len(picks), 1), f"Renderizando clip {k + 1} de {len(picks)}")
        text = " ".join(x for a2, b2, x in segs if a2 < s + L and b2 > s)
        title, tags = meta(text, o["hashtags"])
        name = f"clip{k + 1}.mp4"
        tr = track(src, s, L, W, H) if o["mode"] != "fit" else None
        used = render(src, s, L, d / name, o["mode"], tr, o["watermark"], W, H, o["quality"], o["bar"])
        out.append(dict(file=name, start=int(s), dur=int(L), score=sc, title=title or f"Clip {k + 1}", tags=tags, mode=used, offset=a * 60))
    return out

def rerender(d, k, start, dur):
    m = json.loads((d / "meta.json").read_text()); o = m["o"]
    tr = track(m["src"], start, dur, m["W"], m["H"]) if o["mode"] != "fit" else None
    render(m["src"], start, dur, d / f"clip{k + 1}.mp4", o["mode"], tr, o["watermark"], m["W"], m["H"], o["quality"], o["bar"])
