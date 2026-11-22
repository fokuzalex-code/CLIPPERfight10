import json, os, re, shutil, sqlite3, threading, uuid, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import pipeline

DATA = Path(os.getenv("DATA_DIR", "data")); (DATA / "jobs").mkdir(parents=True, exist_ok=True)
db = sqlite3.connect(DATA / "jobs.db", check_same_thread=False); lock = threading.Lock()
db.execute("create table if not exists jobs(id text primary key,status text,progress real,msg text,clips text)")
pool = ThreadPoolExecutor(1)  # cola: un trabajo a la vez
app = FastAPI(title="Clipper.fight")

def upd(i, **k):
    with lock:
        for c, v in k.items():
            db.execute(f"update jobs set {c}=? where id=?", (json.dumps(v) if c == "clips" else v, i))
        db.commit()

def work(i, src, o):
    try:
        upd(i, status="running")
        clips = pipeline.process(src, DATA / "jobs" / i, o, lambda p, m: upd(i, progress=p, msg=m))
        upd(i, status="done", progress=1, msg="Listo", clips=clips)
    except Exception as e:
        upd(i, status="error", msg=str(e)[:300])

@app.post("/api/jobs")
async def create(url: str = Form(""), file: UploadFile | None = File(None), n: int = Form(5),
                 min_s: int = Form(20), max_s: int = Form(60), mode: str = Form("fill"),
                 watermark: str = Form(""), hashtags: str = Form(""), quality: str = Form("equilibrado"), bar: str = Form("")):
    i = uuid.uuid4().hex[:10]; d = DATA / "jobs" / i; d.mkdir(parents=True)
    if file and file.filename:
        src = d / ("src" + Path(file.filename).suffix)
        with open(src, "wb") as f: shutil.copyfileobj(file.file, f)
        src = str(src)
    elif url.startswith("http"):
        src = url
    else:
        raise HTTPException(400, "Pega un enlace de Kick o Twitch, o sube un video")
    with lock:
        db.execute("insert into jobs values(?,?,?,?,?)", (i, "queued", 0, "En cola", "[]")); db.commit()
    pool.submit(work, i, src, dict(n=max(1, min(n, 10)), min_s=min_s, max_s=max(max_s, min_s),
                                   mode=mode, quality=quality, bar=bar == "1", watermark=watermark, hashtags=hashtags))
    return {"id": i}

@app.get("/api/jobs/{i}")
def job(i: str):
    with lock:
        r = db.execute("select status,progress,msg,clips from jobs where id=?", (i,)).fetchone()
    if not r: raise HTTPException(404, "Trabajo no encontrado")
    return dict(status=r[0], progress=r[1], msg=r[2], clips=json.loads(r[3]))

def jid(i):
    if not re.fullmatch(r"[0-9a-f]{10}", i): raise HTTPException(404, "Trabajo no encontrado")
    return i

def clips_of(i):
    with lock:
        r = db.execute("select clips from jobs where id=?", (jid(i),)).fetchone()
    if not r: raise HTTPException(404, "Trabajo no encontrado")
    return json.loads(r[0])

@app.get("/api/jobs")
def recent():
    with lock:
        rows = db.execute("select id,status,clips from jobs order by rowid desc limit 20").fetchall()
    return [dict(id=i, status=s, n=len(json.loads(c)), title=(json.loads(c) or [{"title": ""}])[0]["title"]) for i, s, c in rows]

@app.delete("/api/jobs/{i}")
def remove(i: str):
    clips_of(i)
    with lock:
        db.execute("delete from jobs where id=?", (i,)); db.commit()
    shutil.rmtree(DATA / "jobs" / i, ignore_errors=True)
    return {"ok": True}

@app.get("/api/jobs/{i}/zip")
def zip_all(i: str):
    clips_of(i); d = DATA / "jobs" / i; p = d / "clips.zip"
    with zipfile.ZipFile(p, "w") as z:
        for f in sorted(d.glob("clip*.mp4")): z.write(f, f.name)
    return FileResponse(p, filename="clips.zip")

def redo_work(i, k, s, L):
    try:
        upd(i, status="running", msg="Recortando clip")
        pipeline.rerender(DATA / "jobs" / i, k, s, L)
        upd(i, status="done", progress=1, msg="Clip actualizado")
    except Exception as e:
        upd(i, status="error", msg=str(e)[:300])

@app.post("/api/jobs/{i}/clips/{k}")
def redo(i: str, k: int, start: int = Form(...), dur: int = Form(...)):
    cl = clips_of(i)
    if not 0 <= k < len(cl): raise HTTPException(404, "Clip no encontrado")
    cl[k]["start"], cl[k]["dur"] = max(0, start), max(5, min(dur, 180))
    upd(i, status="queued", progress=0, msg="En cola", clips=cl)
    pool.submit(redo_work, i, k, cl[k]["start"], cl[k]["dur"])
    return {"ok": True}

app.mount("/files", StaticFiles(directory=DATA / "jobs"), name="files")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
