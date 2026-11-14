"""Capture station: a LAN web page with capture buttons and a shared shot checklist.

Each button press runs one tagged capture on the machine's capture-control sensor, waits for the frames
to sync, and adds them to the training dataset. Checklist state is shared by everyone on the page.

    TEAM_CODE=<code> uv run python tools/capture_station.py        # then open http://<this-ip>:8899/?code=<code>

Uses the Viam CLI login of the machine it runs on. Refuses to capture if teleop is recording or the arm
is not at inspect-pose.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ORG = os.environ.get("VIAM_ORG", "6d6c7293-bc67-43e4-9c8b-31c76aac27d3")
PART = os.environ.get("VIAM_PART", "f08ae88f-5fa8-4775-a3bf-0f4fda96940f")
DATASET = os.environ.get("VIAM_DATASET", "6af772edf64790cdeaa8ee06")
CAM = os.environ.get("CAM", "cam")
SENSOR = "capture-control"
RUN_TAG = os.environ.get("RUN_TAG", "sortit-v1")
TEAM_CODE = os.environ.get("TEAM_CODE", "")
PORT = int(os.environ.get("PORT", "8899"))
POSE_TOL_DEG = 2.0
STATE_FILE = Path(os.environ.get("STATE_FILE", "runs/capture_state.json"))

DEFAULT_SHOTS = [
    ("D1", "anomalous", "Screw with blue thread-lock patch", "shot-03", 5),
    ("D2", "anomalous", "Same screw, rotated / moved", "shot-04", 5),
    ("D3", "anomalous", "Another defect type, if you have one (any damaged/odd part)", None, 0),
    ("D4", "anomalous", "Defective screw next to a good hex bolt", None, 0),
    ("G1", "normal", "Hex bolt, lying flat, centre of view", None, 0),
    ("G2", "normal", "Hex bolt, lying flat, another spot + rotation", None, 0),
    ("G3", "normal", "Hex bolt standing on its head", None, 0),
    ("G4", "normal", "Nut, lying flat", None, 0),
    ("G5", "normal", "Good long socket-head screw (no blue patch), flat", None, 0),
    ("G6", "normal", "Any good part near the edge of the view", None, 0),
    ("G7", "normal", "Two or three good parts together", None, 0),
    ("G8", "normal", "Empty surface, nothing in view", None, 0),
]

lock = threading.Lock()  # one capture at a time


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    shots = [{"id": i, "label": lab, "desc": d, "tag": t, "frames": n, "status": "done" if t else "todo", "by": ""}
             for i, lab, d, t, n in DEFAULT_SHOTS]
    return {"shots": shots, "next_shot": 5, "log": [], "busy": None}


state = load_state()


def save() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))


def log(msg: str) -> None:
    state["log"].insert(0, f"{datetime.now().strftime('%H:%M:%S')}  {msg}")
    del state["log"][40:]
    save()


def viam_run(method: str, data: dict, timeout: int = 60, tries: int = 3) -> dict:
    """Call a machine API through the Viam CLI, retrying the cloud's transient 'host appears to be offline'."""
    for attempt in range(tries):
        out = subprocess.run(["viam", "machines", "part", "run", f"--part={PART}", f"--method={method}", "-d", json.dumps(data)],
                             capture_output=True, text=True, timeout=timeout)
        if not out.returncode:
            return json.loads(out.stdout)
        msg = (out.stderr or out.stdout).strip()[-300:]
        if "appears to be offline" not in msg or attempt == tries - 1:
            raise RuntimeError(msg)
        time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def inspect_pose_deg() -> list[float] | None:
    out = subprocess.run(["viam", "machines", "part", "config", f"--part={PART}"], capture_output=True, text=True, timeout=60)
    cfg = json.loads(out.stdout)
    for c in cfg.get("components", []):
        if c["name"] == "inspect-pose" and c.get("attributes", {}).get("joints"):
            import math
            return [math.degrees(j) for j in c["attributes"]["joints"]]
    return None


INSPECT = None


def preflight() -> str | None:
    """Return a reason to refuse capturing, or None if it is safe."""
    global INSPECT
    r = viam_run("viam.component.sensor.v1.SensorService.GetReadings", {"name": SENSOR})["readings"]
    if r.get("sequences") or any(o.get("capture_frequency_hz", 0) > 0 for o in r.get("overrides", [])):
        return "capture-control is busy (teleop/demo recording?) — try again when it's idle"
    if INSPECT is None:
        INSPECT = inspect_pose_deg()
    joints = viam_run("viam.component.arm.v1.ArmService.GetJointPositions", {"name": "arm"})["positions"]["values"]
    if INSPECT and max(abs(a - b) for a, b in zip(joints, INSPECT)) > POSE_TOL_DEG:
        return "arm is not at inspect-pose — send it there first"
    return None


def capture(shot: dict, who: str) -> None:
    dc = "viam.component.sensor.v1.SensorService.DoCommand"
    try:
        reason = preflight()
        if reason:
            shot["status"] = "todo"
            log(f"{shot['id']} refused: {reason}")
            return
        tag = f"shot-{state['next_shot']:02d}"
        state["next_shot"] += 1
        shot.update(status="capturing", tag=tag, by=who)
        save()
        start = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        try:
            viam_run(dc, {"name": SENSOR, "command": {"start_capture": True, "frequency_hz": 1.0, "tags": [shot["label"], RUN_TAG, tag]}})
            time.sleep(3)
        finally:  # always try to switch capture off, even if start errored after reaching the machine
            for attempt in range(3):
                try:
                    viam_run(dc, {"name": SENSOR, "command": {"stop_capture": True}})
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        log(f"{shot['id']}: WARNING could not stop capture — check capture-control")
                    time.sleep(2)
        shot["status"] = "syncing"
        save()
        added = 0
        for _ in range(12):  # frames sync every ~6 s; wait up to ~60 s
            time.sleep(5)
            out = subprocess.run(["viam", "dataset", "data", "add", "filter", f"--dataset-id={DATASET}", f"--org-ids={ORG}",
                                  f"--part-id={PART}", f"--component-name={CAM}", "--mime-types=image/jpeg", f"--tags={tag}",
                                  f"--start={start}"], capture_output=True, text=True, timeout=60)
            words = out.stdout.split()
            added = int(words[1]) if len(words) > 1 and words[0] == "Added" and words[1].isdigit() else 0
            if added:
                break
        shot.update(status="done" if added else "check", frames=added)
        log(f"{shot['id']} ({shot['label']}) by {who or 'someone'}: {tag}, {added} frames added")
    except Exception as e:  # noqa: BLE001 — surface any failure on the page
        shot["status"] = "check"
        log(f"{shot['id']} failed: {e}")
    finally:
        state["busy"] = None
        save()


CAMERAS = (CAM, "overview-cam")
SNAP_MIN_AGE = 6.0  # seconds; every CLI call opens a new WebRTC connection to the machine, so share and throttle frames
_snap_cache: dict[str, tuple[float, bytes]] = {}
_snap_lock = threading.Lock()


def snapshot(name: str = CAM) -> bytes:
    """One shared frame per camera for all viewers, refreshed at most every SNAP_MIN_AGE s and never mid-capture."""
    with _snap_lock:
        cached = _snap_cache.get(name)
        if cached and (time.time() - cached[0] < SNAP_MIN_AGE or state["busy"]):
            return cached[1]
        r = viam_run("viam.component.camera.v1.CameraService.GetImages", {"name": name})
        for im in r.get("images", []):
            if "jpeg" in (im.get("mimeType") or ""):
                img = base64.b64decode(im["image"])
                _snap_cache[name] = (time.time(), img)
                return img
        raise RuntimeError("no jpeg from camera")


PAGE = Path(__file__).with_name("capture_station.html")


class Handler(BaseHTTPRequestHandler):
    def _ok(self, q: dict) -> bool:
        if TEAM_CODE and q.get("code", [""])[0] != TEAM_CODE:
            self.send_error(403, "add ?code=<team code> to the URL")
            return False
        return True

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._ok(q):
            return
        if u.path == "/":
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/state":
            totals = {"normal": 0, "anomalous": 0}
            for s in state["shots"]:
                if s["status"] == "done":
                    totals[s["label"]] += s["frames"]
            self._send(json.dumps({**state, "totals": totals, "run_tag": RUN_TAG}).encode(), "application/json")
        elif u.path == "/snapshot.jpg":
            name = q.get("cam", [CAM])[0]
            if name not in CAMERAS:
                return self.send_error(404)
            try:
                self._send(snapshot(name), "image/jpeg")
            except Exception as e:  # noqa: BLE001
                self._send(str(e).encode(), "text/plain", 502)
        else:
            self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._ok(q):
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        who = str(body.get("who", ""))[:24]
        if u.path == "/capture":
            with lock:
                if state["busy"]:
                    return self._send(json.dumps({"error": f"busy with {state['busy']}"}).encode(), "application/json", 409)
                shot = next((s for s in state["shots"] if s["id"] == body.get("id")), None)
                if not shot:
                    return self._send(b'{"error":"unknown shot"}', "application/json", 404)
                if shot["status"] in ("capturing", "syncing"):
                    return self._send(b'{"error":"already capturing"}', "application/json", 409)
                if shot["status"] == "done":  # capturing a done shot again adds a new row with the same instruction
                    shot = {**shot, "id": f"{shot['id']}+{sum(1 for s in state['shots'] if s['id'].startswith(shot['id']))}",
                            "tag": None, "frames": 0}
                    state["shots"].append(shot)
                state["busy"] = shot["id"]
                shot["status"] = "capturing"
                save()
            threading.Thread(target=capture, args=(shot, who), daemon=True).start()
            self._send(b'{"ok":true}', "application/json")
        elif u.path == "/edit":  # fix a row's wording (and optionally mark it skipped) without a restart
            with lock:
                shot = next((s for s in state["shots"] if s["id"] == body.get("id")), None)
                if not shot:
                    return self._send(b'{"error":"unknown shot"}', "application/json", 404)
                if body.get("desc"):
                    shot["desc"] = str(body["desc"])[:120]
                if body.get("status") == "skipped" and shot["status"] == "todo":
                    shot["status"] = "skipped"
                log(f"{shot['id']} edited by {who or 'someone'}")
            self._send(b'{"ok":true}', "application/json")
        elif u.path == "/reset":  # un-stick a row left in capturing/syncing/check (e.g. after a server restart)
            with lock:
                shot = next((s for s in state["shots"] if s["id"] == body.get("id")), None)
                if not shot or state["busy"] == shot["id"]:
                    return self._send(b'{"error":"cannot reset this shot now"}', "application/json", 409)
                shot.update(status="todo", tag=None, frames=0, by="")
                log(f"{shot['id']} reset to todo by {who or 'someone'}")
            self._send(b'{"ok":true}', "application/json")
        elif u.path == "/add":
            lab = body.get("label") if body.get("label") in ("normal", "anomalous") else "normal"
            n = sum(1 for s in state["shots"] if s["label"] == lab) + 1
            state["shots"].append({"id": ("G" if lab == "normal" else "D") + str(n), "label": lab,
                                   "desc": str(body.get("desc", ""))[:80] or "extra shot", "tag": None, "frames": 0,
                                   "status": "todo", "by": ""})
            save()
            self._send(b'{"ok":true}', "application/json")
        else:
            self.send_error(404)

    def log_message(self, *a):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    # a restart kills any in-flight capture thread: mark rows it left mid-way so they can be re-shot
    for s in state["shots"]:
        if s["status"] in ("capturing", "syncing"):
            s["status"] = "check"
    state["busy"] = None
    save()
    print(f"capture station on http://0.0.0.0:{PORT}/" + (f"?code={TEAM_CODE}" if TEAM_CODE else "  (no TEAM_CODE set!)"))
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
