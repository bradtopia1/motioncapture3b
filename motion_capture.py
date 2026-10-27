#!/usr/bin/env python3
"""motion_capture.py - save a clip whenever the Pi camera sees movement.

Small enough for a Raspberry Pi 3B. The H.264 video comes straight out of the
Pi's hardware encoder, and only a small "lores" copy of each frame is inspected
for motion, so almost no CPU is used while the scene is still.

Each clip starts PRE seconds before the movement that triggered it - the encoded
frames are held in a circular buffer, so the run-up is already in memory - and
keeps going until POST seconds after the last movement. Nothing walks out of
frame.

Run it on the Pi with:

    python3 motion_capture.py

See README.md for the install step and the tuning knobs.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import TimeoutError as CaptureTimeout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

LOG = logging.getLogger("motioncapture")

# picamera2's own "medium" quality suggestion is 6 Mbit/s for 1080p30 (the
# BITRATE_TABLE in picamera2/encoders/h264_encoder.py), so scale that for
# whatever size and rate we are actually using.
_REFERENCE_BITRATE = 6_000_000
_REFERENCE_COMPLEXITY = 1920 * 1080 * 30

# If no frame arrives in this long the camera has stopped; give up after a few
# in a row so systemd (or the user) can restart us rather than hang silently.
_CAPTURE_TIMEOUT = 5.0
_MAX_CAPTURE_FAILURES = 6

_HEARTBEAT_SECONDS = 300.0


@dataclass(frozen=True)
class Settings:
    """Everything the run needs, with the derived values in one place."""

    width: int
    height: int
    fps: int
    motion_width: int
    motion_height: int
    pre_seconds: float
    post_seconds: float
    max_clip_seconds: float
    threshold: float
    min_motion_percent: float
    min_motion_frames: int
    warmup_frames: int
    keyframe_seconds: float
    output_dir: Path
    log_file: Path

    @property
    def keyframe_interval(self) -> int:
        """Encoder I-period, in frames."""
        return max(1, round(self.fps * self.keyframe_seconds))

    @property
    def buffer_frames(self) -> int:
        """Frames held in the circular buffer.

        A clip opens at the first key frame found in the buffer, which can be up
        to one key frame interval old, so buffer PRE plus that much again to
        still hand over a full PRE seconds of run-up.
        """
        return max(1, round((self.pre_seconds + self.keyframe_seconds) * self.fps))

    @property
    def bitrate(self) -> int:
        return int(_REFERENCE_BITRATE * self.width * self.height * self.fps / _REFERENCE_COMPLEXITY)


class MotionDetector:
    """Tiny background-model motion detector, run on a downscaled grey image.

    Each frame is averaged down by ``pool`` in both directions (which also takes
    the edge off sensor noise), compared with a slowly adapting background, and
    the changed pixels are counted. Motion is reported once enough of them have
    changed for ``min_frames`` frames in a row.
    """

    # Changed pixels still creep towards the new value, but at a fraction of the
    # still rate. Without this a change to the whole frame - a light switching
    # on, a cloud passing, the camera's own exposure settling at dusk - could
    # never be learned, and the detector would call motion forever. Slow enough
    # that someone walking past is not swallowed by the background.
    _CHANGED_ADAPT_RATIO = 0.25

    def __init__(
        self,
        *,
        threshold: float,
        min_area_percent: float,
        pool: int = 2,
        alpha: float = 0.05,
        warmup_frames: int = 30,
        min_frames: int = 2,
    ) -> None:
        self.threshold = float(threshold)
        self.min_area_fraction = max(0.0, float(min_area_percent)) / 100.0
        self.pool = max(1, int(pool))
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.warmup_frames = max(0, int(warmup_frames))
        self.min_frames = max(1, int(min_frames))

        self._bg: np.ndarray | None = None
        self._seen = 0
        self._streak = 0

    def update(self, gray: np.ndarray) -> tuple[bool, float]:
        """Feed one greyscale frame; return (motion, fraction of pixels changed)."""
        small = self._pooled(gray)

        if self._bg is None or self._bg.shape != small.shape:
            self._bg = small.copy()
            self._seen += 1
            return False, 0.0

        changed = np.abs(small - self._bg) > self.threshold
        fraction = float(np.count_nonzero(changed)) / changed.size

        # Follow the scene quickly where it is still and slowly where it is
        # changing, so a stationary object is not absorbed while it is moving
        # but a change to the whole frame still gets learned.
        step = small - self._bg
        self._bg = self._bg + step * np.where(changed, self.alpha * self._CHANGED_ADAPT_RATIO, self.alpha)

        self._seen += 1
        self._streak = self._streak + 1 if fraction >= self.min_area_fraction else 0
        motion = self._streak >= self.min_frames and self._seen > self.warmup_frames
        return motion, fraction

    def _pooled(self, gray: np.ndarray) -> np.ndarray:
        """Average-pool the image by ``pool`` in each direction, as float32."""
        if self.pool > 1:
            h, w = gray.shape[:2]
            ph, pw = h // self.pool, w // self.pool
            if ph and pw:
                view = gray[: ph * self.pool, : pw * self.pool]
                return view.reshape(ph, self.pool, pw, self.pool).mean(axis=(1, 3), dtype=np.float32)
        return gray.astype(np.float32, copy=False)


class ClipRecorder:
    """Turns motion into files.

    The encoder runs continuously into a circular buffer, so the footage from
    just before a trigger is still available. On the first motion the buffered
    run-up is flushed into a new file and the file keeps growing until
    ``post_seconds`` after the last motion (or ``max_clip_seconds``, whichever
    comes first).
    """

    def __init__(
        self,
        output,
        *,
        directory: Path,
        post_seconds: float,
        max_clip_seconds: float = 0.0,
    ) -> None:
        self._output = output
        self._directory = Path(directory)
        self._post_seconds = float(post_seconds)
        self._max_clip_seconds = max(0.0, float(max_clip_seconds))
        self._path: Path | None = None
        self._last_motion = 0.0
        self._started = 0.0

    @property
    def active(self) -> bool:
        return self._path is not None

    def tick(self, motion: bool, now: float) -> Path | None:
        """Advance the state machine; returns a clip path once one is finished."""
        if motion:
            self._last_motion = now
            if self._path is None:
                self._start(now)
                return None

        if self._path is not None:
            quiet = now - self._last_motion >= self._post_seconds
            too_long = self._max_clip_seconds > 0 and now - self._started >= self._max_clip_seconds
            if quiet or too_long:
                return self._stop()
        return None

    def finish(self) -> Path | None:
        """Stop an in-progress clip (used on shutdown). Returns its path."""
        if self._path is None:
            return None
        return self._stop()

    def _start(self, now: float) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._path = self._free_name()
        # Order matters: setting fileoutput opens the file and resets the
        # first-frame flag, then start() lets the buffer flush into it.
        self._output.fileoutput = str(self._path)
        self._output.start()
        self._started = now
        LOG.info("motion - recording %s", self._path.name)

    def _stop(self) -> Path:
        path = self._path
        self._output.stop()  # drains the tail of the buffer and closes the file
        self._path = None
        return path

    def _free_name(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._directory / f"clip_{stamp}.h264"
        suffix = 1
        while path.exists():
            path = self._directory / f"clip_{stamp}_{suffix}.h264"
            suffix += 1
        return path


def lores_to_gray(frame: np.ndarray, height: int) -> np.ndarray:
    """Reduce a lores capture to a single 8-bit image.

    picamera2 hands back either a packed planar YUV420 buffer (the luma plane
    first) or an interleaved RGB/BGR array depending on what the camera settled
    on, so cope with both rather than trusting one.
    """
    if frame.ndim == 3:
        return frame[:, :, 0]
    if frame.shape[0] > height:
        return frame[:height]
    return frame


def remux_to_mp4(h264_path: Path, fps: int) -> Path | None:
    """Rewrap a raw H.264 stream as an .mp4 without re-encoding.

    Returns ``None`` when ffmpeg is not installed or refused the file, in which
    case the caller keeps the raw stream instead.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    mp4_path = h264_path.with_suffix(".mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(h264_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, OSError) as exc:
        LOG.warning("could not turn %s into an .mp4: %s", h264_path.name, exc)
        return None
    return mp4_path


def _megabytes(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1e6:.1f} MB"
    except OSError:
        return "unknown size"


def save_clip(h264_path: Path, fps: int) -> None:
    """Finish a clip off: remux to .mp4 and drop the raw stream."""
    mp4_path = remux_to_mp4(h264_path, fps)
    if mp4_path is None:
        LOG.info("saved %s (%s)", h264_path.name, _megabytes(h264_path))
        return
    h264_path.unlink(missing_ok=True)
    LOG.info("saved %s (%s)", mp4_path.name, _megabytes(mp4_path))


def watch(picam2, detector: MotionDetector, recorder: ClipRecorder, settings: Settings, stop_event) -> None:
    """Poll the lores stream until asked to stop, recording clips on motion."""
    failures = 0
    frames = 0
    clips = 0
    primed = False
    last_heartbeat = time.monotonic()

    while not stop_event.is_set():
        try:
            frame = picam2.capture_array("lores", wait=_CAPTURE_TIMEOUT)
        except CaptureTimeout:
            failures += 1
            LOG.warning("no frame from the camera (%d/%d)", failures, _MAX_CAPTURE_FAILURES)
            if failures >= _MAX_CAPTURE_FAILURES:
                raise RuntimeError("camera stopped delivering frames - check the ribbon cable")
            continue
        except Exception as exc:  # libcamera reports failures as plain RuntimeError
            raise RuntimeError(f"capture failed: {exc}") from exc
        failures = 0
        frames += 1

        gray = lores_to_gray(frame, settings.motion_height)
        if frames == 1:
            LOG.info("camera running - video %dx%d @ %g fps, motion watch %dx%d",
                     settings.width, settings.height, settings.fps,
                     gray.shape[1], gray.shape[0])

        motion, fraction = detector.update(gray)
        finished = recorder.tick(motion, time.monotonic())
        if finished is not None:
            save_clip(finished, settings.fps)
            clips += 1
        elif motion and recorder.active:
            LOG.debug("motion continues (%.2f%% of the frame changed)", fraction * 100)

        if not primed and frames >= settings.buffer_frames:
            primed = True
            LOG.info("buffer full - clips will now include the full %g s of run-up", settings.pre_seconds)

        now = time.monotonic()
        if now - last_heartbeat >= _HEARTBEAT_SECONDS:
            last_heartbeat = now
            LOG.info("watching - %d frames, %d clip(s) saved, %.2f%% of frame moving",
                     frames, clips, fraction * 100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a clip when the Pi camera sees motion, with the seconds "
                    "leading up to it included.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    video = parser.add_argument_group("video")
    video.add_argument("--width", type=int, default=1280, help="recorded frame width")
    video.add_argument("--height", type=int, default=720, help="recorded frame height")
    video.add_argument("--fps", type=int, default=15, help="frames per second")
    video.add_argument("--keyframe-seconds", type=float, default=1.0,
                       help="seconds between key frames; also how much extra video is buffered")

    clip = parser.add_argument_group("clip")
    clip.add_argument("--pre", type=float, default=5.0, help="seconds to keep before the motion")
    clip.add_argument("--post", type=float, default=5.0, help="seconds to keep after the last motion")
    clip.add_argument("--max-clip", type=float, default=300.0,
                      help="cut a clip that never stops moving this many seconds in (0 = no limit)")
    clip.add_argument("--output-dir", type=Path, default=Path("clips"), help="where clips are written")

    motion = parser.add_argument_group("motion")
    motion.add_argument("--motion-width", type=int, default=320,
                       help="width of the small image motion is measured on, in pixels")
    motion.add_argument("--threshold", type=float, default=18.0,
                       help="how much brighter or darker a pixel must get to count as changed (0-255)")
    motion.add_argument("--min-motion-percent", type=float, default=0.4,
                       help="percent of the frame that must change to count as motion")
    motion.add_argument("--motion-frames", type=int, default=2,
                       help="consecutive frames of motion needed before recording starts")
    motion.add_argument("--warmup", type=int, default=30,
                       help="frames used to learn the background before motion counts")

    parser.add_argument("--log-file", type=Path, default=Path("motioncapture.log"),
                        help="where the run log is written")
    parser.add_argument("--verbose", action="store_true", help="also log per-frame detail")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    # Motion is measured on a smaller copy of the frame; keep the shape of the
    # video and round to an even number of lines.
    motion_height = max(2, int(round(args.motion_width * args.height / args.width / 2)) * 2)
    return Settings(
        width=args.width,
        height=args.height,
        fps=args.fps,
        motion_width=args.motion_width,
        motion_height=motion_height,
        pre_seconds=args.pre,
        post_seconds=args.post,
        max_clip_seconds=args.max_clip,
        threshold=args.threshold,
        min_motion_percent=args.min_motion_percent,
        min_motion_frames=args.motion_frames,
        warmup_frames=args.warmup,
        keyframe_seconds=args.keyframe_seconds,
        output_dir=args.output_dir,
        log_file=args.log_file,
    )


def setup_logging(verbose: bool, log_file: Path) -> None:
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOG.addHandler(console)

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError as exc:
        LOG.warning("could not open %s for logging: %s", log_file, exc)
        return
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)


def run(settings: Settings, stop_event: threading.Event) -> None:
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import CircularOutput
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "picamera2 is not installed. On Raspberry Pi OS: "
            "sudo apt install -y python3-picamera2 python3-numpy"
        ) from exc

    LOG.info("video %dx%d @ %g fps, bitrate %.1f Mbit/s, key frame every %d frames",
             settings.width, settings.height, settings.fps,
             settings.bitrate / 1e6, settings.keyframe_interval)
    LOG.info("clips of %g s before and %g s after motion, written to %s",
             settings.pre_seconds, settings.post_seconds, settings.output_dir.resolve())

    picam2 = Picamera2()
    try:
        picam2.configure(
            picam2.create_video_configuration(
                main={"size": (settings.width, settings.height), "format": "YUV420"},
                lores={"size": (settings.motion_width, settings.motion_height), "format": "YUV420"},
                controls={"FrameRate": settings.fps},
            )
        )

        encoder = H264Encoder(
            bitrate=settings.bitrate,
            framerate=settings.fps,
            iperiod=settings.keyframe_interval,
            repeat=True,
        )
        output = CircularOutput(buffersize=settings.buffer_frames)
        picam2.start_recording(encoder, output)

        detector = MotionDetector(
            threshold=settings.threshold,
            min_area_percent=settings.min_motion_percent,
            warmup_frames=settings.warmup_frames,
            min_frames=settings.min_motion_frames,
        )
        recorder = ClipRecorder(
            output,
            directory=settings.output_dir,
            post_seconds=settings.post_seconds,
            max_clip_seconds=settings.max_clip_seconds,
        )

        try:
            watch(picam2, detector, recorder, settings, stop_event)
        finally:
            # Flush whatever clip is in flight before the encoder goes away.
            pending = recorder.finish()
            if pending is not None:
                save_clip(pending, settings.fps)
    finally:
        try:
            picam2.stop_recording()
        except Exception:  # pragma: no cover - nothing useful to do about it
            pass
        picam2.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = build_settings(args)
    setup_logging(args.verbose, settings.log_file)

    stop_event = threading.Event()

    def request_stop(signum, _frame):
        LOG.info("signal %d - shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    LOG.info("motion_capture starting (pid %d)", __import__("os").getpid())
    try:
        run(settings, stop_event)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - signal handler usually wins
        pass
    LOG.info("motion_capture stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
