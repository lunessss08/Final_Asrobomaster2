"""Failure-path checks; no robot connection or runtime map writes."""
import json
import threading
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robo_app.aim.control import FireController
from robo_app.aim import settings as aim_settings
from robo_app.maze.camera import FrameGrabber
from robo_app.maze.errors import Aborted, HardwareFault
from robo_app.maze.hardware import RealRobot
from robo_app.maze.mapping import MazeMap
from robo_app.maze.mission import MazeMission
from robo_app.maze.simulation import SimRobot
from robo_app import aim as gs


class ReliabilityTests(unittest.TestCase):
    def mission(self):
        io = SimRobot((0, 0), 0, 3, 3)
        return MazeMission(io, FrameGrabber(io), MazeMap(3, 3), (0, 0), 0)

    def hardware(self):
        io = RealRobot.__new__(RealRobot)
        io._stop_event = threading.Event()
        io.stop_all = Mock()
        io.chassis = Mock()
        io.stop_wheels = Mock()
        io.ensure_stopped = Mock()
        return io

    def test_stop_disarms_and_sends_stop_immediately(self):
        m = self.mission()
        m.io.stop_all = Mock()
        m.armed = m.fire_ctrl.armed = True
        m.stop()
        self.assertTrue(m.stop_event.is_set())
        self.assertFalse(m.armed)
        self.assertFalse(m.fire_ctrl.armed)
        m.io.stop_all.assert_called_once()

    def test_fault_reason_remains_visible_after_job_stops(self):
        m = self.mission()
        m.io.stop_all = Mock()
        m.io.set_status_led = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch('robo_app.maze.mission.LOOK_LOG_ROOT', Path(directory)):
            self.assertTrue(m.start_job('EXPLORE', Mock(side_effect=HardwareFault('ToF blocked'))))
            deadline = time.monotonic() + 1
            while m.busy and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn('ToF blocked', (Path(directory) / 'faults.jsonl').read_text())
        self.assertFalse(m.busy)
        self.assertEqual(m.last_fault, 'ToF blocked')
        self.assertEqual(m.status, 'ToF blocked')

    def test_sdk_wait_interrupts_running_action(self):
        io = self.hardware()
        action = SimpleNamespace(is_completed=False, has_succeeded=False)
        io._pause = Mock(side_effect=lambda seconds: io._stop_event.set())
        with self.assertRaises(Aborted):
            io._wait_action('move', action, 4)
        io.stop_all.assert_called_once()

    def test_sdk_failed_and_missing_actions_are_faults(self):
        for action in (None, SimpleNamespace(is_completed=True, has_succeeded=False, state='failed')):
            io = self.hardware()
            with self.subTest(action=action), self.assertRaises(HardwareFault):
                io._wait_action('move', action, 4)
            io.stop_all.assert_called_once()

    def test_sdk_action_timeout_stops_motion(self):
        for action in (SimpleNamespace(is_completed=False),
                       SimpleNamespace(is_completed=True, has_succeeded=False,
                                       state='action_running')):
            io = self.hardware()
            with self.subTest(action=action), \
                    patch('robo_app.maze.hardware.time.monotonic', side_effect=[0, 5]):
                with self.assertRaises(HardwareFault):
                    io._wait_action('move', action, 4)
            io.stop_all.assert_called_once()

    def test_successful_action_uses_status_without_sdk_blocking_wait(self):
        io = self.hardware()
        action = Mock(is_completed=True, has_succeeded=True)
        io._wait_action('move', action, 4)
        action.wait_for_completed.assert_not_called()
        io.stop_all.assert_not_called()

    def test_sdk_100_percent_running_waits_for_final_success_state(self):
        io = self.hardware()
        action = SimpleNamespace(is_completed=True, has_succeeded=False, state='action_running')
        def finish(_seconds):
            action.state = 'action_succeeded'
            action.has_succeeded = True
        io._pause = Mock(side_effect=finish)
        io._wait_action('lateral correction', action, 5)
        io._pause.assert_called_once()
        io.stop_all.assert_not_called()

    def test_gimbal_timeout_scales_with_travel_and_reports_command(self):
        io = self.hardware()
        io.gimbal = Mock()
        io._wait_action = Mock()
        io._gimbal_pose = (0, -90)
        io.gimbal_to(-20, 180, speed=30)
        self.assertAlmostEqual(io._wait_action.call_args.args[2], 12.0)
        self.assertEqual(io._gimbal_pose, (-20, 180))

        io._wait_action.side_effect = HardwareFault('gimbal move: timed out')
        with self.assertRaisesRegex(HardwareFault, r'pitch -15°.*yaw \+180°.*limit 10\.0s'):
            io.gimbal_to(-15, 180, speed=60)
        self.assertEqual(io._gimbal_pose, (-20, 180))

    def test_water_burst_uses_one_sdk_command_for_three_shots(self):
        io = self.hardware()
        io.blaster = Mock()
        io.blaster.fire.return_value = True
        io.leds = Mock()
        self.assertTrue(io.fire(times=3))
        self.assertEqual(io.blaster.fire.call_count, 1)
        self.assertEqual(io.blaster.fire.call_args.kwargs['times'], 3)

    def test_chassis_led_is_red_during_motion_and_white_for_gimbal(self):
        io = self.hardware()
        io.leds = Mock()
        io.gimbal = Mock()
        io._wait_action = Mock()
        io._pause = Mock()
        io.set_status_led(False, True)

        def during_turn(label, action, timeout):
            self.assertEqual(io.leds.show_status.call_args.args, ('red', False, False))
        io._wait_action.side_effect = during_turn
        io.rotate(1)
        self.assertEqual(io.leds.show_status.call_args.args, ('white', False, True))

        def during_gimbal(label, action, timeout):
            self.assertEqual(io.leds.show_status.call_args.args, ('white', False, True))
        io._wait_action.side_effect = during_gimbal
        io.gimbal_to(0, 90)
        self.assertEqual(io.leds.show_status.call_args.args, ('white', False, True))

    def test_stopped_robot_does_not_submit_short_move(self):
        io = self.hardware()
        io._stop_event.set()
        with self.assertRaises(Aborted):
            io.move_straight(-0.1)
        io.chassis.move.assert_not_called()

    def test_short_move_failure_does_not_restore_known_pose(self):
        m = self.mission()
        m.io.move_straight = Mock(side_effect=HardwareFault('failed'))
        with self.assertRaises(HardwareFault):
            m.move_in_cell(-0.1)
        self.assertTrue(m.pose_uncertain)

    def test_short_move_success_restores_previous_pose_state(self):
        m = self.mission()
        for previous in (True, False):
            m.pose_uncertain = previous
            m.move_in_cell(-0.01)
            self.assertEqual(m.pose_uncertain, previous)

    def test_start_check_rejects_all_or_partial_missing_readings(self):
        for readings in ([None] * 4, [100, None, 100, 100]):
            m = self.mission()
            m.maze.set_edge((0, 0), (1, 0), 1)
            m.io.read_tof_mm = Mock(side_effect=readings)
            with self.subTest(readings=readings):
                self.assertFalse(m.verify_start())
                self.assertTrue(m.pose_uncertain)

    def test_start_check_accepts_matching_readings(self):
        m = self.mission()
        m.maze.set_edge((0, 0), (1, 0), 1)
        m.io.read_tof_mm = Mock(side_effect=[1000, 1000, 100, 100])
        self.assertTrue(m.verify_start())

    def test_level_tof_requires_three_fresh_samples(self):
        io = self.hardware()
        io._tof_log = deque()
        io._pause = Mock()
        with patch('robo_app.maze.hardware.time.monotonic', return_value=100):
            io._tof_log.extend([(99.9, 250), (100, 250), (100.01, 251)])
            self.assertIsNone(io.read_tof_mm())
            io._tof_log.append((100.02, 249))
            self.assertEqual(io.read_tof_mm(), 250)

    def test_connection_rejects_unconfirmed_free_mode(self):
        for accepted, reported in ((False, 'free'), (True, 'gimbal_lead')):
            ep = Mock()
            ep.initialize.return_value = True
            ep.set_robot_mode.return_value = accepted
            ep.get_robot_mode.return_value = reported
            with self.subTest(accepted=accepted, reported=reported), \
                    patch('robo_app.maze.hardware.robot.Robot', return_value=ep):
                with self.assertRaisesRegex(ConnectionError, 'FREE robot mode'):
                    RealRobot()
            ep.close.assert_called_once()

    def test_connection_rejects_missing_camera_stream(self):
        ep = Mock()
        ep.initialize.return_value = True
        ep.set_robot_mode.return_value = True
        ep.get_robot_mode.return_value = 'free'
        ep.camera.start_video_stream.return_value = False
        with patch('robo_app.maze.hardware.robot.Robot', return_value=ep), \
                patch('robo_app.maze.hardware.gs.LedManager'), \
                patch.object(RealRobot, 'gimbal_to'):
            with self.assertRaisesRegex(ConnectionError, 'Camera stream did not start'):
                RealRobot()
        ep.close.assert_called_once()

    def test_map_save_overwrites_one_file_without_backups(self):
        m = self.mission()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'maze_map.json'
            path.write_text('old map')
            with patch('robo_app.maze.mission.MAP_FILE', path):
                m.save_map(quiet=True)
                m.maze.visited.add((0, 0))
                m.save_map(quiet=True)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['visited'], [[0, 0]])
            self.assertEqual([p.name for p in path.parent.iterdir()], ['maze_map.json'])

    def test_clear_map_does_not_reset_when_file_delete_fails(self):
        m = self.mission()
        original = m.maze
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'maze_map.json'
            path.write_text('keep this map')
            with patch('robo_app.maze.mission.MAP_FILE', path), \
                    patch('pathlib.Path.unlink', side_effect=OSError('disk error')):
                m.clear_map()
            self.assertIs(m.maze, original)
            self.assertEqual(path.read_text(), 'keep this map')

    def test_clear_map_deletes_file_without_creating_archive(self):
        m = self.mission()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'maze_map.json'
            path.write_text('old map')
            with patch('robo_app.maze.mission.MAP_FILE', path):
                m.clear_map()
            self.assertEqual(list(path.parent.iterdir()), [])
            self.assertTrue(m.pose_uncertain)
            self.assertFalse(m.maze.visited)

    def test_reset_pose_stays_uncertain_when_verification_aborts(self):
        m = self.mission()
        m.verify_start = Mock(side_effect=Aborted())
        with self.assertRaises(Aborted):
            m.reset_pose()
        self.assertTrue(m.pose_uncertain)

    def test_frame_wait_can_be_interrupted(self):
        grabber = FrameGrabber(Mock())
        with self.assertRaises(Aborted):
            grabber.wait_new(0, check=Mock(side_effect=Aborted()))

    def test_shape_filter_rejects_same_color_wrong_plate(self):
        m = self.mission()
        target = (1, 0)
        m.maze.add_target(target, 'red', 'circle', 1, (0.87, 0))
        # Real color masks/contours, with two red signs of different shapes.
        import cv2
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.circle(frame, (250, 200), 45, (0, 0, 255), -1)
        cv2.rectangle(frame, (550, 170), (660, 230), (0, 0, 255), -1)
        fd = gs.prepare_frame(frame)
        cands = gs.find_targets(fd, gs.build_mask(fd.hsv, m.color_ranges['red'], m.blind_zones), 'red')
        self.assertGreaterEqual(len(cands), 2)
        chosen = m.aim_candidates(fd, target, cands)
        self.assertEqual(len(chosen), 1)
        self.assertLess(chosen[0].cx, fd.size[0] / 2)

    def test_size_change_restarts_lock_before_evaluation(self):
        m = self.mission()
        target = m.cell
        m.maze.add_target(target, 'red', '?', 1, (0.95, 0), near_seen=(0.47, 0))
        m.confirm_target_wall = Mock(return_value=True)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        fd = gs.prepare_frame(frame)
        w, h = fd.size
        first = gs.Target('red', w / 2, h / 2, 100, (0, 0, 10, 10))
        replacement = gs.Target('red', w / 2, h / 2, 200, (0, 0, 20, 10))
        # The second plate occupies the same aiming position. Geometry alone
        # cannot identify a switch, but its size must reset the old lock.
        lock_events = []
        m.fire_ctrl = Mock(settled=False)
        m.fire_ctrl.reset_lock.side_effect = lambda: lock_events.append('reset')
        m.fire_ctrl.evaluate.side_effect = lambda *args: (lock_events.append('evaluate') or ('TRACKING', False))
        m.grabber.wait_new = Mock(side_effect=[(frame, 1), (frame, 2), Aborted()])
        m.io.fire = Mock()
        with patch('robo_app.maze.mission.gs.find_targets', side_effect=[[first], [replacement]]):
            with self.assertRaises(Aborted):
                m.aim_and_fire(target)
        self.assertEqual(lock_events, ['reset', 'evaluate', 'reset', 'evaluate'])
        m.io.fire.assert_not_called()

    def test_short_move_rejects_completed_but_failed_sdk_action(self):
        io = self.hardware()
        io.chassis.move.return_value = SimpleNamespace(is_completed=True, has_succeeded=False, state='failed')
        with self.assertRaises(HardwareFault):
            io.move_straight(-0.1)
        io.stop_all.assert_called_once()
        io.ensure_stopped.assert_not_called()

    def test_unknown_map_shape_keeps_color_candidates(self):
        m = self.mission()
        candidates = [object()]
        self.assertEqual(m.aim_candidates(None, (1, 0), candidates), candidates)

    def test_fire_lock_requires_consecutive_aligned_frames(self):
        ctrl = FireController()
        ctrl.armed = True
        with patch('robo_app.aim.control.time.monotonic', return_value=0):
            for _ in range(aim_settings.MIN_LOCK_HITS):
                ctrl.evaluate(0, 0, False)
            ctrl.evaluate(10, 0, False)
            for _ in range(aim_settings.MIN_LOCK_HITS - 1):
                self.assertFalse(ctrl.evaluate(0, 0, False)[1])
            self.assertIsNone(ctrl._lock_start)

    def test_stale_frame_gap_resets_fire_dwell(self):
        ctrl = FireController()
        ctrl.armed = True
        with patch('robo_app.aim.control.time.monotonic', side_effect=[0, 0.1, 0.2, 1.0]):
            for _ in range(3):
                ctrl.evaluate(0, 0, False)
            self.assertFalse(ctrl.evaluate(0, 0, False)[1])
        self.assertFalse(ctrl.settled)
        self.assertEqual(ctrl._hits, 1)

    def test_continuous_lock_can_still_fire(self):
        ctrl = FireController()
        ctrl.armed = True
        times = [i * 0.1 for i in range(12)]
        with patch('robo_app.aim.control.time.monotonic', side_effect=times):
            decisions = [ctrl.evaluate(0, 0, False)[1] for _ in times]
        self.assertTrue(any(decisions))


if __name__ == '__main__':
    unittest.main()
