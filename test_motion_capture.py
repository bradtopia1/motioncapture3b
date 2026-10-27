"""Tests for the parts of motion_capture.py that need no camera.

Run with:  .venv/bin/python -m unittest -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from motion_capture import ClipRecorder, MotionDetector, lores_to_gray


class FakeOutput:
    """Stands in for picamera2's CircularOutput.

    Like the real one it opens a file when ``fileoutput`` is set and closes it
    on ``stop``, so the recorder can be exercised without an encoder.
    """

    def __init__(self) -> None:
        self._file = None
        self.path: str | None = None
        self.started = 0
        self.stopped = 0

    @property
    def fileoutput(self):
        return self._file

    @fileoutput.setter
    def fileoutput(self, path):
        if self._file is not None:
            self._file.close()
        self.path = path
        self._file = open(path, "wb") if path is not None else None

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self.stopped += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class MotionDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = MotionDetector(
            threshold=18.0,
            min_area_percent=0.4,
            pool=2,
            warmup_frames=5,
            min_frames=2,
        )
        self.still = np.full((90, 160), 100, dtype=np.uint8)

    def test_a_still_scene_never_triggers(self) -> None:
        for _ in range(20):
            motion, fraction = self.detector.update(self.still)
            self.assertFalse(motion)
            self.assertEqual(fraction, 0.0)

    def test_motion_is_ignored_while_the_background_is_learning(self) -> None:
        for _ in range(3):
            self.detector.update(self.still)

        frame = self.still.copy()
        frame[20:40, 20:60] = 255
        motion, fraction = self.detector.update(frame)

        self.assertFalse(motion)
        self.assertGreater(fraction, 0.0)

    def test_movement_is_reported_after_warmup(self) -> None:
        for _ in range(8):
            self.detector.update(self.still)

        frame = self.still.copy()
        frame[20:40, 20:60] = 255
        reported = [self.detector.update(frame)[0] for _ in range(4)]

        self.assertTrue(any(reported), "a lit-up block should register as motion")

    def test_a_single_odd_frame_is_not_enough(self) -> None:
        detector = MotionDetector(
            threshold=18.0, min_area_percent=0.4, pool=2, warmup_frames=0, min_frames=2
        )
        for _ in range(3):
            detector.update(self.still)

        odd = self.still.copy()
        odd[20:40, 20:60] = 255
        self.assertFalse(detector.update(odd)[0])  # one frame only
        self.assertTrue(detector.update(odd)[0])  # second frame in a row

    def test_a_bright_frame_is_not_motion_once_it_settles(self) -> None:
        # A change that stays put should be absorbed into the background.
        for _ in range(8):
            self.detector.update(self.still)

        brighter = np.full((90, 160), 140, dtype=np.uint8)
        for _ in range(200):
            motion, _ = self.detector.update(brighter)
        self.assertFalse(motion, "a steady new scene should stop reading as motion")


class LoresToGrayTests(unittest.TestCase):
    def test_planar_yuv420_keeps_only_the_luma_plane(self) -> None:
        planar = np.zeros((90 * 3 // 2, 160), dtype=np.uint8)
        planar[:90] = 7

        gray = lores_to_gray(planar, 90)

        self.assertEqual(gray.shape, (90, 160))
        self.assertTrue((gray == 7).all())

    def test_three_channel_frame_keeps_one_channel(self) -> None:
        rgb = np.zeros((90, 160, 3), dtype=np.uint8)
        rgb[:, :, 0] = 9

        gray = lores_to_gray(rgb, 90)

        self.assertEqual(gray.shape, (90, 160))
        self.assertTrue((gray == 9).all())

    def test_a_plain_greyscale_frame_is_passed_through(self) -> None:
        frame = np.full((90, 160), 3, dtype=np.uint8)
        self.assertEqual(lores_to_gray(frame, 90).shape, (90, 160))


class ClipRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.output = FakeOutput()
        self.recorder = ClipRecorder(
            self.output, directory=self.directory, post_seconds=5.0, max_clip_seconds=300.0
        )

    def tearDown(self) -> None:
        self.output.close()
        self._tmp.cleanup()

    def started_clip(self) -> Path:
        self.recorder.tick(True, 0.0)
        assert self.output.path is not None
        self.assertTrue(self.recorder.active)
        path = Path(self.output.path)
        self.assertTrue(path.exists())
        return path

    def test_a_still_scene_records_nothing(self) -> None:
        self.assertIsNone(self.recorder.tick(False, 0.0))
        self.assertFalse(self.recorder.active)
        self.assertEqual(self.output.started, 0)

    def test_motion_opens_a_file(self) -> None:
        self.started_clip()
        self.assertEqual(self.output.started, 1)
        self.assertEqual(self.output.stopped, 0)

    def test_the_clip_runs_on_for_the_post_seconds(self) -> None:
        self.started_clip()
        self.assertIsNone(self.recorder.tick(False, 4.0))
        self.assertTrue(self.recorder.active)

        clip = self.recorder.tick(False, 5.0)

        self.assertIsNotNone(clip)
        self.assertFalse(self.recorder.active)
        self.assertEqual(self.output.stopped, 1)
        self.assertTrue(clip.exists())

    def test_further_motion_keeps_the_clip_open(self) -> None:
        self.recorder.tick(True, 0.0)
        self.recorder.tick(True, 4.0)
        self.assertIsNone(self.recorder.tick(False, 8.0))  # only 4 s of quiet
        self.assertTrue(self.recorder.active)

    def test_continuous_motion_is_cut_off_at_the_limit(self) -> None:
        recorder = ClipRecorder(
            self.output, directory=self.directory, post_seconds=5.0, max_clip_seconds=10.0
        )
        recorder.tick(True, 0.0)
        self.assertIsNone(recorder.tick(True, 9.0))
        self.assertIsNotNone(recorder.tick(True, 10.0))

    def test_finish_flushes_a_clip_in_flight(self) -> None:
        self.started_clip()
        clip = self.recorder.finish()
        self.assertIsNotNone(clip)
        self.assertEqual(self.output.stopped, 1)
        self.assertIsNone(self.recorder.finish())

    def test_two_clips_in_the_same_second_do_not_collide(self) -> None:
        first = self.started_clip()
        self.recorder.tick(False, 5.0)
        second = self.started_clip()
        self.assertNotEqual(first, second)


class SettingsTests(unittest.TestCase):
    def test_the_buffer_covers_the_run_up_at_the_going_rate(self) -> None:
        from motion_capture import build_parser, build_settings

        settings = build_settings(build_parser().parse_args([]))
        # 5 s of run-up plus one key frame interval of slack, at 15 fps.
        self.assertEqual(settings.buffer_frames, 90)
        self.assertEqual(settings.keyframe_interval, 15)

    def test_motion_size_follows_the_video_shape(self) -> None:
        from motion_capture import build_parser, build_settings

        settings = build_settings(build_parser().parse_args(["--width", "640", "--height", "480",
                                                             "--motion-width", "320"]))
        self.assertEqual(settings.motion_height, 240)


if __name__ == "__main__":
    unittest.main()
