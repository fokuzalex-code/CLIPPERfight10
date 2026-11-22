import json, os, re, shutil, sqlite3, threading, time, uuid, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import pipeline

DATA = Path(os.getenv("DATA_DIR", "data")); JOBS = DATA / "jobs"; JOBS.mkdir(parents=True, exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "1500")); MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4")); TTL = int(os.getenv("TTL_HOURS", "24")) * 3600
db = sqlite3.connect(DATA / "jobs.db", check_same_thread=False); lock = threading.Lock()
db.execute("create table if not exists jobs(id text primary key,status text,progress real,msg text,clips text,owner text,created real)")
for c in ("owner text", "created real"):  # migración de bases antiguas
    try: db.execute("alter table jobs add column " + c)
    except sqlite3.OperationalError: pass
db.execute("update jobs set status='error',msg='El servidor se reinició; vuelve a generar' where status in ('queued','running')")
db.commit()
pool = ThreadPoolExecutor(1)  # cola: un trabajo a la vez
app = FastAPI(title="Clipper.fight")

def upd(i, **k):
    with lock:
        for c, v in k.items():
            db.execute(f"update jobs set {c}=? where id=?", (json.dumps(v) if c == "clips" else v, i))
        db.commit()

def cleanup():  # borra trabajos viejos para no llenar el disco
    with lock:
        old = [r[0] for r in db.execute("select id from jobs where created<?", (time.time() - TTL,))]
        for i in old: db.execute("delete from jobs where id=?", (i,))
        db.commit()
    for i in old: shutil.rmtree(JOBS / i, ignore_errors=True)

def work(i, src, o):
    try:
        upd(i, status="running")
        clips = pipeline.process(src, JOBS / i, o, lambda p, m: upd(i, progress=p, msg=m))
        upd(i, status="done", progress=1, msg="Listo", clips=clips)
    except Exception as e:
        upd(i, status="error", msg=str(e)[:300])

def jid(i):
    if not re.fullmatch(r"[0-9a-f]{16}", i): raise HTTPException(404, "Trabajo no encontrado")
    return i

def row(i):
    with lock:
        r = db.execute("select clips,owner from jobs where id=?", (jid(i),)).fetchone()
    if not r: raise HTTPException(404, "Trabajo no encontrado")
    return json.loads(r[0]), r[1]

@app.get("/health")
def health(): return {"ok": True}

@app.post("/api/jobs")
def create(url: str = Form(""), file: UploadFile | None = File(None), n: int = Form(5), min_s: int = Form(20),
           max_s: int = Form(60), from_min: int = Form(0), to_min: int = Form(0), mode: str = Form("auto"),
           watermark: str = Form(""), hashtags: str = Form(""), quality: str = Form("equilibrado"),
           bar: str = Form(""), x_client: str = Header("")):
    cleanup()
    with lock:
        busy = db.execute("select count(*) from jobs where status in ('queued','running')").fetchone()[0]
    if busy >= MAX_QUEUE: raise HTTPException(429, "Hay muchos trabajos en cola. Intenta en unos minutos")
    if not (file and file.filename) and not pipeline.valid_url(url):
        raise HTTPException(400, "Pega un enlace de Kick o Twitch, o sube un video")
    i = uuid.uuid4().hex[:16]; d = JOBS / i; d.mkdir(parents=True)
    if file and file.filename:
        src = d / ("src" + re.sub(r"[^\w.]", "", Path(file.filename).suffix)[:6]); size = 0; big = False
        with open(src, "wb") as f:
            while chunk := file.file.read(1 << 20):
                size += len(chunk)
                if size > MAX_MB << 20: big = True; break
                f.write(chunk)
        if big:
            shutil.rmtree(d, ignore_errors=True); raise HTTPException(413, f"El video supera {MAX_MB} MB")
        src = str(src)
    else:
        src = url
    mn = max(5, min(min_s, 120)); fm = max(0, from_min); tm = max(0, to_min)
    o = dict(n=max(1, min(n, 10)), min_s=mn, max_s=max(mn, min(max_s, 180)), from_min=fm, to_min=tm if tm > fm else 0,
             mode=mode, quality=quality, bar=bar == "1", watermark=watermark[:40], hashtags=hashtags[:200])
    with lock:
        db.execute("insert into jobs values(?,?,?,?,?,?,?)", (i, "queued", 0, "En cola", "[]", x_client, time.time())); db.commit()
    pool.submit(work, i, src, o)
    return {"id": i}

@app.get("/api/jobs")
def recent(x_client: str = Header("")):
    if not x_client: return []
    with lock:
        rows = db.execute("select id,status,clips from jobs where owner=? order by rowid desc limit 20", (x_client,)).fetchall()
    return [dict(id=i, status=s, n=len(json.loads(c)), title=(json.loads(c) or [{"title": ""}])[0]["title"]) for i, s, c in rows]

@app.get("/api/jobs/{i}")
def job(i: str):
    with lock:
        r = db.execute("select status,progress,msg,clips from jobs where id=?", (jid(i),)).fetchone()
    if not r: raise HTTPException(404, "Trabajo no encontrado")
    return dict(status=r[0], progress=r[1], msg=r[2], clips=json.loads(r[3]))

@app.delete("/api/jobs/{i}")
def remove(i: str, x_client: str = Header("")):
    _, owner = row(i)
    if owner != x_client: raise HTTPException(403, "No es tu trabajo")
    with lock:
        db.execute("delete from jobs where id=?", (i,)); db.commit()
    shutil.rmtree(JOBS / i, ignore_errors=True)
    return {"ok": True}

@app.get("/api/jobs/{i}/zip")
def zip_all(i: str):
    row(i); d = JOBS / i; p = d / "clips.zip"
    with zipfile.ZipFile(p, "w") as z:
        for f in sorted(d.glob("clip*.mp4")): z.write(f, f.name)
    return FileResponse(p, filename="clips.zip")

def redo_work(i, k, s, L):
    try:
        upd(i, status="running", msg="Recortando clip")
        pipeline.rerender(JOBS / i, k, s, L)
        upd(i, status="done", progress=1, msg="Clip actualizado")
    except Exception as e:
        upd(i, status="error", msg=str(e)[:300])

@app.post("/api/jobs/{i}/clips/{k}")
def redo(i: str, k: int, start: int = Form(...), dur: int = Form(...), x_client: str = Header("")):
    cl, owner = row(i)
    if owner != x_client: raise HTTPException(403, "No es tu trabajo")
    if not 0 <= k < len(cl): raise HTTPException(404, "Clip no encontrado")
    cl[k]["start"], cl[k]["dur"] = max(0, start), max(5, min(dur, 180))
    upd(i, status="queued", progress=0, msg="En cola", clips=cl)
    pool.submit(redo_work, i, k, cl[k]["start"], cl[k]["dur"])
    return {"ok": True}

app.mount("/files", StaticFiles(directory=JOBS), name="files")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
