"""Stationary ToF centering and lateral chassis command."""

import threading
import unittest
from unittest.mock import Mock, patch

from robo_app.maze.camera import FrameGrabber
from robo_app.maze.centering import center_shift_mm
from robo_app.maze.errors import HardwareFault
from robo_app.maze.hardware import RealRobot
from robo_app.maze.mapping import MazeMap
from robo_app.maze.mission import MazeMission
from robo_app.maze.simulation import SimRobot


class CenteringTests(unittest.TestCase):
    def mission(self):
        io = SimRobot((0, 0), 0, 4, 4)
        io._pause = Mock()
        maze = MazeMap(4, 4)
        maze.set_edge((0, 0), (1, 0), -1)
        return MazeMission(io, FrameGrabber(io), maze, (0, 0), 0, use_slam=True)

    def test_bounded_centering_math(self):
        self.assertEqual(center_shift_mm(240, 160, 200), 40)
        self.assertEqual(center_shift_mm(400, 160, None), 60)
        self.assertEqual(center_shift_mm(300, None, None), 0)
        self.assertEqual(center_shift_mm(100, 300, 200), -60)
        self.assertEqual(center_shift_mm(260, 260, 180), 0)

    @patch('robo_app.maze.mission.time.sleep')
    def test_two_mapped_walls_scan_and_shift_before_drive(self, sleep):
        m = self.mission()
        m.io.read_tof_mm = Mock(side_effect=[240, 160])
        m.io.move_sideways = Mock()
        m.io.gimbal_to = Mock()
        m.center_before_move(0)
        self.assertEqual([c.args for c in m.io.gimbal_to.call_args_list], [(0, -90), (0, 90), (0, 0)])
        m.io.move_sideways.assert_called_once_with(0.04)
        self.assertFalse(m.pose_uncertain)

    @patch('robo_app.maze.mission.time.sleep')
    def test_one_wall_requires_calibrated_offset(self, sleep):
        m = self.mission()
        m.maze.set_edge((0, 0), (1, 0), 0)
        m.io.read_tof_mm = Mock(return_value=300)
        m.io.move_sideways = Mock()
        m.center_before_move(0)
        m.io.read_tof_mm.assert_not_called()
        m._tof_offset_known = True
        m.center_before_move(0)
        m.io.move_sideways.assert_called_once_with(0.05)

    @patch('robo_app.maze.mission.time.sleep')
    def test_body_drift_faults_without_lateral_move(self, sleep):
        m = self.mission()
        def drift():
            m.io.yaw = 9
            return 240
        m.io.read_tof_mm = Mock(side_effect=drift)
        m.io.move_sideways = Mock()
        with self.assertRaisesRegex(HardwareFault, 'body moved'):
            m.center_before_move(0)
        self.assertTrue(m.pose_uncertain)
        m.io.move_sideways.assert_not_called()

    def test_real_lateral_command_uses_sdk_sign_and_checks_completion(self):
        io = RealRobot.__new__(RealRobot)
        io._stop_event = threading.Event()
        io.chassis = Mock()
        io._wait_action = Mock()
        io.stop_wheels = Mock()
        io.ensure_stopped = Mock()
        io.move_sideways(0.04)
        io.chassis.move.assert_called_once_with(x=0, y=-0.04, z=0, xy_speed=0.15)
        self.assertEqual(io._wait_action.call_args.args[2], 5)
        io.ensure_stopped.assert_called_once()
        with self.assertRaises(HardwareFault):
            io.move_sideways(0.07)

    def test_large_heading_error_stops_forward_drive(self):
        io = RealRobot.__new__(RealRobot)
        io._stop_event = threading.Event()
        io.pos, io.pos_t = (0.0, 0.0), 100.0
        io.tof_t, io.tof_seq, io.tof_mm = 100.0, 1, 900
        io.attitude_t, io.yaw = 100.0, 20.0
        io.wall_center_mm = 250
        io.chassis = Mock()
        io.stop_wheels = Mock()
        with patch('robo_app.maze.hardware.time.monotonic', return_value=100.0), \
                patch('robo_app.maze.hardware.time.sleep'):
            with self.assertRaisesRegex(HardwareFault, 'heading drifted'):
                io.forward_one_cell(0, lambda: None)
        io.chassis.drive_speed.assert_not_called()
        io.stop_wheels.assert_called_once()


if __name__ == '__main__':
    unittest.main()
