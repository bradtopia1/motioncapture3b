# motioncapture3b

Watches a Raspberry Pi camera and saves a video clip whenever something moves in
the frame. Each clip starts **5 seconds before** the movement and runs on for
**5 seconds after** the last movement, so nothing is missed at either end.

Built for a **Raspberry Pi 3B**: the video is compressed by the Pi's hardware
H.264 encoder and motion is measured on a small copy of each frame, so almost no
CPU is used while the scene is still. On the Pi there is **nothing to pip
install** — both packages come with Raspberry Pi OS.

## Install

Everything it needs is already in Raspberry Pi OS, so this is one command:

```bash
sudo apt install -y python3-picamera2 python3-numpy
```

`python3-picamera2` is the library that talks to the camera, and it must come
from `apt` rather than `pip` — it is tied to the Pi's camera firmware.

## Run

From the folder this file is in, on the Pi:

```bash
python3 motion_capture.py
```

Leave it running; **Ctrl-C** stops it (and finishes off any clip in progress).
On a *dev machine* there is no camera, so instead run the tests:

```bash
python3 -m unittest -v
```

That needs `numpy` only (see `requirements.txt`).

### Where things are written

Relative to the folder you start it in:

| Path | What it is |
| --- | --- |
| `clips/clip_YYYYMMDD_HHMMSS.mp4` | one video per motion event |
| `motioncapture.log` | the run log, ~1 MB, keeping 3 older files |

Clips are written as raw H.264 and then rewrapped (not re-encoded — it is fast)
into `.mp4`, so they play anywhere. If `ffmpeg` is not installed the raw `.h264`
is kept instead and the log says so; `ffmpeg` is a bonus, not a requirement.

Both `clips/` and the log are in `.gitignore`, so generated video never ends up
in a commit.

## Tuning

Run `python3 motion_capture.py --help` for the full list. The ones worth knowing:

| Option | Default | Use it for |
| --- | --- | --- |
| `--pre` / `--post` | `5` / `5` | seconds kept either side of the motion |
| `--min-motion-percent` | `0.4` | raise it if harmless things keep triggering, lower it to catch distant movement |
| `--threshold` | `18` | raise it in a noisy or flickering scene |
| `--motion-frames` | `2` | frames of motion in a row before recording starts; raise to ignore brief flickers |
| `--width` / `--height` / `--fps` | `1280` `720` `15` | drop to `640 480 10` if the Pi struggles |
| `--max-clip` | `300` | caps a clip that never stops moving (`0` for no limit) |
| `--output-dir` | `clips` | put clips on a USB stick or a bigger card |

The log prints the percentage of the frame that was moving, so it is easy to see
how much headroom a setting leaves.

## How it works

Frames go straight from the camera into the hardware H.264 encoder and into a
circular buffer holding the last few seconds. In parallel, a small `lores` copy
of each frame is compared against a slowly adapting background; when enough
pixels change, the buffered seconds are flushed to a new file and it keeps
recording until the movement has been over for `--post` seconds.

Because the buffer is always filling, that "5 seconds before" is genuinely
already recorded — it is not reconstructed afterwards. One caveat worth knowing:
a clip has to begin on a key frame, so a little under `--pre` seconds may be
dropped, depending on where the previous key frame fell. Extra video is buffered
to cover this.

## Status

The motion detection and clip-timing logic is covered by tests. The camera path
itself has **not** been run on a real Pi yet — this machine has no camera — so
the first run on the Pi is what will confirm it.
