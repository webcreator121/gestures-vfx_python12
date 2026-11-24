"""Threaded webcam capture + MediaPipe Hands.

The capture/detection loop runs in a background thread so the render loop never
blocks on the camera. The frame is mirrored *before* detection, so every
coordinate is already in "selfie" space.
"""
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config


@dataclass
class TrackedHand:
    id: int
    lm: np.ndarray            # (21, 3) normalised x, y (0..1, mirrored), z
    handedness: str = "?"
    score: float = 0.0


@dataclass
class TrackerResult:
    seq: int = 0
    t: float = 0.0
    frame: object = None      # mirrored BGR frame (full capture resolution)
    hands: list = field(default_factory=list)
    status: str = "starting"
    camera_ok: bool = False
    detect_ms: float = 0.0


class HandTracker:
    def __init__(self, camera_index=None):
        self.index = config.CAMERA_INDEX if camera_index is None else camera_index
        self._lock = threading.Lock()
        self._result = TrackerResult()
        self._stop = threading.Event()
        self._thread = None
        self._tracks = {}      # id -> (wrist xy, last seen time)
        self._prev = {}        # id -> smoothed landmarks
        self._next_id = 0

    # -- public ---------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def latest(self):
        with self._lock:
            return self._result

    # -- internals ------------------------------------------------------
    def _publish(self, **kw):
        with self._lock:
            r = self._result
            self._result = TrackerResult(
                seq=kw.get("seq", r.seq), t=kw.get("t", r.t),
                frame=kw.get("frame", r.frame), hands=kw.get("hands", r.hands),
                status=kw.get("status", r.status),
                camera_ok=kw.get("camera_ok", r.camera_ok),
                detect_ms=kw.get("detect_ms", r.detect_ms))

    def _open_camera(self):
        backends = [cv2.CAP_ANY]
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_SIZE[1])
            cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        return None

    def _make_hands(self):
        try:
            import mediapipe as mp
            return mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                model_complexity=config.MODEL_COMPLEXITY,
                min_detection_confidence=config.DETECT_CONF,
                min_tracking_confidence=config.TRACK_CONF)
        except Exception as exc:  # noqa: BLE001 - report any import/model error
            self._publish(status=f"MediaPipe error: {exc}")
            return None

    def _assign_ids(self, wrists, now):
        """Greedy nearest-wrist matching so hand ids stay stable across frames."""
        self._tracks = {i: v for i, v in self._tracks.items()
                        if now - v[1] < config.HAND_GHOST_TIME}
        pairs = sorted((float(np.hypot(*(w - tr[0]))), wi, tid)
                       for wi, w in enumerate(wrists)
                       for tid, tr in self._tracks.items())
        ids, used = [None] * len(wrists), set()
        for dist, wi, tid in pairs:
            if dist < 0.3 and ids[wi] is None and tid not in used:
                ids[wi] = tid
                used.add(tid)
        for wi in range(len(wrists)):
            if ids[wi] is None:
                ids[wi] = self._next_id
                self._next_id += 1
        for wi, tid in enumerate(ids):
            self._tracks[tid] = (wrists[wi], now)
        return ids

    def _smooth(self, hid, lm):
        prev = self._prev.get(hid)
        if prev is not None:
            motion = float(np.mean(np.linalg.norm(lm[:, :2] - prev[:, :2], axis=1)))
            a = min(0.95, config.LM_SMOOTH + motion * 8)   # fast moves lag less
            lm = prev + (lm - prev) * a
        self._prev[hid] = lm
        return lm

    def _run(self):
        hands_model = self._make_hands()
        cap, fails = None, 0
        while not self._stop.is_set():
            if cap is None:
                cap = self._open_camera()
                if cap is None:
                    self._publish(status="No webcam found - retrying", camera_ok=False,
                                  hands=[])
                    time.sleep(1.5)
                    continue
                fails = 0
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    cap.release()
                    cap = None
                    self._publish(status="Webcam lost - reconnecting", camera_ok=False,
                                  hands=[])
                time.sleep(0.01)
                continue
            fails = 0
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            hands, ms = [], 0.0
            if hands_model is not None:
                h, w = frame.shape[:2]
                small = frame if w <= config.DETECT_WIDTH else cv2.resize(
                    frame, (config.DETECT_WIDTH, int(h * config.DETECT_WIDTH / w)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                t0 = time.perf_counter()
                res = hands_model.process(rgb)
                ms = (time.perf_counter() - t0) * 1000
                if res.multi_hand_landmarks:
                    lms = [np.array([[p.x, p.y, p.z] for p in hl.landmark])
                           for hl in res.multi_hand_landmarks]
                    ids = self._assign_ids([lm[0, :2] for lm in lms], now)
                    for i, lm in enumerate(lms):
                        label, score = "?", 0.0
                        if res.multi_handedness:
                            c = res.multi_handedness[i].classification[0]
                            label, score = c.label, c.score
                        hands.append(TrackedHand(ids[i], self._smooth(ids[i], lm),
                                                 label, score))
                status = f"{len(hands)} hand(s)" if hands else "No hands"
            else:
                status = self._result.status
            self._publish(seq=self._result.seq + 1, t=now, frame=frame, hands=hands,
                          status=status, camera_ok=True, detect_ms=ms)
        if cap is not None:
            cap.release()
        if hands_model is not None:
            hands_model.close()
