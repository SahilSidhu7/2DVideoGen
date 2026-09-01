# EquationVideo web UI

A chat-style web app over the prompt→video pipeline. Type a prompt, get an
animated equation video; manage your chats and a gallery of all generated videos.

```
python webapp/server.py            # -> http://127.0.0.1:5000
python webapp/server.py --port 8000
python webapp/server.py --rules-only   # skip the neural model (faster start)
```

> Windows: prefix with `USE_TF=0` (or set it in the environment) so transformers
> doesn't try to load an incompatible Keras 3 backend.

## What it does

- **Chat area** — send a prompt, watch a live status (understanding → rendering),
  then the video appears inline with its parsed spec shown as tags.
- **Chat management** — sidebar lists all chats (newest first) with a video count;
  new chat button; double-click to rename; × to delete (removes its videos too).
  The first prompt auto-titles the chat.
- **Gallery** — every generated video in a grid, each with its prompt + spec,
  download and delete buttons.
- **Persistence** — chats, messages, and videos are stored in `webapp/data.db`
  (SQLite); mp4s in `webapp/videos/`. State survives restarts.

## Architecture

```
 browser (static/index.html, vanilla JS)
        │  REST + poll
        ▼
 Flask server.py ── PromptModel (t5-small) ── grammar.build_scene ── geovid ── mp4
        │
        └── SQLite (chats / messages / videos)
```

Rendering runs in a **background thread**, one at a time (guarded by a lock), so
the UI stays responsive and a modest CPU isn't overloaded. The frontend creates a
job and polls `/api/jobs/<id>` until the video is ready.

## API

| method | route | purpose |
|--------|-------|---------|
| GET  | `/api/health` | model backend + archetype list |
| GET/POST | `/api/chats` | list / create chat |
| PATCH/DELETE | `/api/chats/<id>` | rename / delete chat |
| GET | `/api/chats/<id>/messages` | messages in a chat |
| POST | `/api/chats/<id>/generate` | start a render job `{prompt, duration}` |
| GET | `/api/jobs/<id>` | job status / result |
| GET | `/api/videos` | all videos |
| DELETE | `/api/videos/<id>` | delete a video (row + file) |
| GET | `/videos/<file>` | serve an mp4 |

## Stickman archetype

Prompts like *"a green stickman walking across the screen"*, *"a stickman
playing football"*, *"stick figure jumping"*, *"a guy running"*, *"someone
waving hello"*, or *"stickman dancing"* generate the per-frame figure
(`stickman.py`) doing that action. Supported actions: **walk, run, jump, kick
(football, with a ball), wave, dance**. The action is detected from the prompt;
"football"/"soccer"/"kick" → the kicking animation with a ball that flies off. Because
it is drawn from a fresh equation set every frame (not a single time-animated
scene), generation is dispatched through `render.py` → `stickman.render_from_spec`
instead of the normal geovid path. The assistant reply notes the frame count and
the message/gallery card gain a **∑ equations** download (the full per-frame
equation list as `.equations.jsonl`).

Detection is keyword-based in `grammar.parse_prompt_rules`; `infer.PromptModel`
forces `arch=stickman` when the rule parser matches, since the t5 model was not
trained on this archetype (retrain with stickman examples to make it native).

## Notes

- No build step, no frontend framework, no CDN — one static HTML file.
- Single-user/local by design. For multi-user, put it behind a real WSGI server
  (gunicorn/waitress) and move the in-memory `JOBS` dict to the DB or a queue.
