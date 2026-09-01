#!/usr/bin/env python3
"""
server.py - Flask backend for the prompt->video web UI.

Wraps the trained model + geovid renderer behind a small REST API and serves a
single-page frontend. State (chats, messages, videos) lives in a local SQLite
DB; rendered mp4s live in webapp/videos/. Rendering runs in a background thread
(one at a time, guarded by a lock) so the UI stays responsive on modest hardware.

    python webapp/server.py            # http://127.0.0.1:5000
    python webapp/server.py --port 8000 --rules-only
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from flask import (Flask, g, jsonify, request, send_from_directory,
                   send_file, abort)

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp"
VIDEOS = WEBAPP / "videos"
STATIC = WEBAPP / "static"
DB_PATH = WEBAPP / "data.db"
VIDEOS.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
import geovid          # noqa: E402
import grammar as G    # noqa: E402
import render as R      # noqa: E402
from infer import PromptModel  # noqa: E402

app = Flask(__name__, static_folder=None)

MODEL: PromptModel | None = None
RENDER_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}       # in-memory job status
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# DB helpers
# --------------------------------------------------------------------------- #

def db():
    conn = getattr(g, "_db", None)
    if conn is None:
        conn = g._db = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
    return conn


@app.teardown_appcontext
def _close_db(exc):
    conn = getattr(g, "_db", None)
    if conn is not None:
        conn.close()


def worker_db():
    """Fresh connection for background threads (no flask g)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, title TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, chat_id TEXT, role TEXT, content TEXT,
            spec TEXT, video_id TEXT, created REAL,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY, chat_id TEXT, prompt TEXT, spec TEXT,
            file TEXT, created REAL);
        """)
    # eqfile column (equation-list download for stickman); added if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(videos)")]
    if "eqfile" not in cols:
        conn.execute("ALTER TABLE videos ADD COLUMN eqfile TEXT")
    conn.commit()
    conn.close()


def now():
    return time.time()


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return send_file(STATIC / "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/videos/<path:name>")
def video_files(name):
    return send_from_directory(VIDEOS, name, conditional=True)


# --------------------------------------------------------------------------- #
# Chats
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    return jsonify({"backend": "t5-small" if (MODEL and MODEL.ok) else "rules",
                    "archetypes": sorted(G.ARCH_NAMES)})


@app.get("/api/chats")
def list_chats():
    rows = db().execute(
        "SELECT c.id, c.title, c.created, "
        "(SELECT COUNT(*) FROM videos v WHERE v.chat_id=c.id) AS videos "
        "FROM chats c ORDER BY c.created DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/chats")
def create_chat():
    title = (request.get_json(silent=True) or {}).get("title") or "New chat"
    cid = uuid.uuid4().hex
    db().execute("INSERT INTO chats VALUES (?,?,?)", (cid, title, now()))
    db().commit()
    return jsonify({"id": cid, "title": title, "created": now(), "videos": 0})


@app.patch("/api/chats/<cid>")
def rename_chat(cid):
    title = (request.get_json(silent=True) or {}).get("title", "").strip()
    if not title:
        abort(400)
    db().execute("UPDATE chats SET title=? WHERE id=?", (title, cid))
    db().commit()
    return jsonify({"ok": True})


@app.delete("/api/chats/<cid>")
def delete_chat(cid):
    # remove video files belonging to this chat
    for r in db().execute("SELECT file, eqfile FROM videos WHERE chat_id=?", (cid,)):
        _unlink_video(r["file"], r["eqfile"])
    db().execute("DELETE FROM chats WHERE id=?", (cid,))
    db().commit()
    return jsonify({"ok": True})


@app.get("/api/chats/<cid>/messages")
def chat_messages(cid):
    rows = db().execute(
        "SELECT m.*, v.file AS video_file, v.eqfile AS video_eqfile "
        "FROM messages m LEFT JOIN videos v ON m.video_id=v.id "
        "WHERE m.chat_id=? ORDER BY m.created ASC", (cid,)).fetchall()
    return jsonify([_msg_json(r) for r in rows])


def _msg_json(r):
    keys = r.keys()
    return {
        "id": r["id"], "role": r["role"], "content": r["content"],
        "spec": r["spec"], "video_id": r["video_id"],
        "video_file": r["video_file"] if "video_file" in keys else None,
        "eqfile": r["video_eqfile"] if "video_eqfile" in keys else None,
        "created": r["created"],
    }


# --------------------------------------------------------------------------- #
# Generation (background job)
# --------------------------------------------------------------------------- #

@app.post("/api/chats/<cid>/generate")
def generate(cid):
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        abort(400, "empty prompt")
    duration = float(data.get("duration", 6))
    duration = max(2.0, min(20.0, duration))

    chat = db().execute("SELECT id, title FROM chats WHERE id=?", (cid,)).fetchone()
    if not chat:
        abort(404)

    # persist the user message
    umid = uuid.uuid4().hex
    db().execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                 (umid, cid, "user", prompt, None, None, now()))
    # first prompt becomes the chat title
    if chat["title"] == "New chat":
        db().execute("UPDATE chats SET title=? WHERE id=?",
                     (prompt[:48], cid))
    db().commit()

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "chat_id": cid, "prompt": prompt}
    threading.Thread(target=_render_job, args=(job_id, cid, prompt, duration),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "user_message_id": umid})


def _render_job(job_id, cid, prompt, duration):
    def setj(**kw):
        with JOBS_LOCK:
            JOBS[job_id].update(kw)
    try:
        setj(status="understanding")
        spec = MODEL.spec(prompt)
        dsl = G.spec_to_dsl(spec)

        setj(status="rendering", spec=dsl)
        vid_id = uuid.uuid4().hex
        fname = f"{vid_id}.mp4"
        with RENDER_LOCK:                      # one render at a time
            extras = R.render_spec(spec, VIDEOS / fname, duration=duration,
                                   verbose=False)

        # stickman writes its equation list alongside the mp4 (same dir)
        eqfile = None
        if extras.get("equations_jsonl"):
            eqfile = Path(extras["equations_jsonl"]).name

        conn = worker_db()
        conn.execute("INSERT INTO videos (id,chat_id,prompt,spec,file,created,eqfile)"
                     " VALUES (?,?,?,?,?,?,?)",
                     (vid_id, cid, prompt, dsl, fname, now(), eqfile))
        amid = uuid.uuid4().hex
        reply = f"Generated **{spec.get('arch')}** ({dsl})."
        if eqfile:
            reply += f" {extras['frames']} frames, each a distinct equation set."
        conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                     (amid, cid, "assistant", reply, dsl, vid_id, now()))
        conn.commit()
        conn.close()

        setj(status="done", video_id=vid_id, video_file=fname,
             spec=dsl, message=reply, assistant_message_id=amid, eqfile=eqfile)
    except Exception as e:                     # pragma: no cover
        setj(status="error", error=str(e))


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


@app.post("/api/chats/<cid>/vectorize")
def vectorize_upload(cid):
    if "file" not in request.files:
        abort(400, "no file")
    f = request.files["file"]
    name = f.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext not in IMG_EXT | VID_EXT:
        abort(400, "unsupported file type")
    if not db().execute("SELECT id FROM chats WHERE id=?", (cid,)).fetchone():
        abort(404)

    tmp = VIDEOS / f"upload_{uuid.uuid4().hex}{ext}"
    f.save(tmp)
    umid = uuid.uuid4().hex
    db().execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                 (umid, cid, "user", f"Uploaded {name} to vectorize", None,
                  None, now()))
    db().commit()

    job_id = uuid.uuid4().hex
    is_video = ext in VID_EXT
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "chat_id": cid, "prompt": name}
    threading.Thread(target=_vectorize_job,
                     args=(job_id, cid, str(tmp), is_video, name),
                     daemon=True).start()
    return jsonify({"job_id": job_id, "user_message_id": umid})


def _vectorize_job(job_id, cid, tmp_path, is_video, name):
    def setj(**kw):
        with JOBS_LOCK:
            JOBS[job_id].update(kw)
    try:
        setj(status="detecting pose")
        import vectorize as V
        vid_id = uuid.uuid4().hex
        fname = f"{vid_id}.mp4"
        with RENDER_LOCK:
            if is_video:
                V.vectorize_video(tmp_path, VIDEOS / fname)
                extras = {"equations_jsonl":
                          str((VIDEOS / fname).with_suffix(".equations.jsonl"))}
            else:
                extras = V.vectorize_image_to_video(tmp_path, VIDEOS / fname)
        eqfile = Path(extras["equations_jsonl"]).name \
            if extras.get("equations_jsonl") else None

        conn = worker_db()
        dsl = "source=real-video" if is_video else "source=real-image"
        conn.execute("INSERT INTO videos (id,chat_id,prompt,spec,file,created,eqfile)"
                     " VALUES (?,?,?,?,?,?,?)",
                     (vid_id, cid, f"vectorized: {name}", dsl, fname, now(), eqfile))
        amid = uuid.uuid4().hex
        reply = (f"Vectorized **{name}** with YOLO-pose → stick-figure equations."
                 " Every frame is an equation set.")
        conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                     (amid, cid, "assistant", reply, dsl, vid_id, now()))
        conn.commit(); conn.close()
        setj(status="done", video_id=vid_id, video_file=fname, spec=dsl,
             message=reply, assistant_message_id=amid, eqfile=eqfile)
    except Exception as e:                     # pragma: no cover
        setj(status="error", error=str(e))
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    return jsonify(job)


# --------------------------------------------------------------------------- #
# Videos (gallery + management)
# --------------------------------------------------------------------------- #

@app.get("/api/videos")
def list_videos():
    rows = db().execute(
        "SELECT v.*, c.title AS chat_title FROM videos v "
        "LEFT JOIN chats c ON v.chat_id=c.id ORDER BY v.created DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.delete("/api/videos/<vid>")
def delete_video(vid):
    row = db().execute("SELECT file, eqfile FROM videos WHERE id=?",
                       (vid,)).fetchone()
    if not row:
        abort(404)
    _unlink_video(row["file"], row["eqfile"])
    db().execute("UPDATE messages SET video_id=NULL WHERE video_id=?", (vid,))
    db().execute("DELETE FROM videos WHERE id=?", (vid,))
    db().commit()
    return jsonify({"ok": True})


def _unlink_video(fname, eqfile=None):
    for f in (fname, eqfile):
        if not f:
            continue
        try:
            (VIDEOS / f).unlink(missing_ok=True)
        except OSError:
            pass
    if eqfile:  # also the readable .txt companion
        try:
            (VIDEOS / eqfile).with_suffix(".txt").unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rules-only", action="store_true",
                    help="skip the neural model (faster start, less flexible)")
    args = ap.parse_args()

    init_db()
    ckpt = None if args.rules_only else str(ROOT / "model" / "checkpoint")
    print("loading model..." if ckpt else "using rule parser only")
    MODEL = PromptModel(ckpt)
    print(f"model backend: {'t5-small' if MODEL.ok else 'rules'}")
    print(f"serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
