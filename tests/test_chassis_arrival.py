"""Arrival braking and fresh-odometry stop checks, without robot hardware."""
import threading
import unittest
from unittest.mock import Mock, patch

from robo_app.maze.errors import HardwareFault
from robo_app.maze.hardware import RealRobot
from robo_app.maze.settings import APPROACH_MAX_SPEED, APPROACH_ZONE_M, CELL_M, STOP_STABLE_SAMPLES


class ArrivalTests(unittest.TestCase):
    def robot(self):
        io = RealRobot.__new__(RealRobot)
        io.pos = (0.0, 0.0)
        io.pos_t = 100.0
        io.stop_wheels = Mock()
        return io

    def verify(self, io, step, fresh=True):
        clock = [100.0]
        def sleep(seconds):
            clock[0] += seconds
            io.pos = (io.pos[0] + step, 0.0)
            if fresh:
                io.pos_t = clock[0]
        with patch('robo_app.maze.hardware.time.monotonic', side_effect=lambda: clock[0]), patch('robo_app.maze.hardware.time.sleep', side_effect=sleep):
            io.ensure_stopped()
        return clock[0]

    def test_stop_requires_multiple_fresh_stationary_samples(self):
        io = self.robot()
        elapsed = self.verify(io, 0) - 100
        self.assertGreaterEqual(elapsed, 0.15 * STOP_STABLE_SAMPLES - 1e-6)
        io.stop_wheels.assert_not_called()

    def test_stale_unchanged_position_is_not_a_confirmed_stop(self):
        io = self.robot()
        with self.assertRaisesRegex(HardwareFault, 'odometry stale'):
            self.verify(io, 0, fresh=False)
        io.stop_wheels.assert_called()

    def test_continued_motion_resends_stop_and_faults(self):
        io = self.robot()
        with self.assertRaisesRegex(HardwareFault, 'did not settle'):
            self.verify(io, 0.01)
        self.assertGreater(io.stop_wheels.call_count, 1)

    def test_slow_creep_cannot_pass_as_stationary(self):
        io = self.robot()
        with self.assertRaisesRegex(HardwareFault, 'did not settle'):
            self.verify(io, 0.002)

    def test_resend_then_settle_is_accepted(self):
        io = self.robot()
        clock = [100.0]
        def sleep(seconds):
            clock[0] += seconds
            if not io.stop_wheels.called:
                io.pos = (io.pos[0] + 0.01, 0)
            io.pos_t = clock[0]
        with patch('robo_app.maze.hardware.time.monotonic', side_effect=lambda: clock[0]), patch('robo_app.maze.hardware.time.sleep', side_effect=sleep):
            io.ensure_stopped()
        io.stop_wheels.assert_called_once()

    def drive(self, coast):
        io = self.robot()
        io._stop_event = threading.Event()
        io.tof_t, io.tof_mm, io.tof_seq = 100.0, 2000, 1
        io.wall_center_mm, io.yaw = 250, 0
        io.side_speed = Mock(return_value=0.1)
        io.chassis = Mock()
        io.ensure_stopped = Mock()
        clock, speed = [100.0], [0.0]
        commands = []
        def drive_speed(**kwargs):
            commands.append((io.pos[0], kwargs))
            speed[0] = kwargs['x']
        def sleep(seconds):
            clock[0] += seconds
            io.pos = (io.pos[0] + speed[0] * seconds, 0)
            io.pos_t = io.tof_t = clock[0]
            io.tof_seq += 1
        def stop():
            speed[0] = 0
            io.pos = (io.pos[0] + coast, 0)
        io.chassis.drive_speed.side_effect = drive_speed
        io.stop_wheels.side_effect = stop
        with patch('robo_app.maze.hardware.time.monotonic', side_effect=lambda: clock[0]), patch('robo_app.maze.hardware.time.sleep', side_effect=sleep):
            fraction = io.forward_one_cell(0, lambda: None)
        return fraction, commands

    def test_final_approach_is_slow_and_moderate_coast_is_accepted(self):
        fraction, commands = self.drive(0.012)
        approach = [kw for pos, kw in commands if CELL_M - pos <= APPROACH_ZONE_M]
        self.assertTrue(approach)
        self.assertTrue(all(kw['x'] <= APPROACH_MAX_SPEED for kw in approach))
        self.assertTrue(all(abs(kw['y']) <= kw['x'] * 0.5 for kw in approach))
        self.assertLessEqual(abs(fraction * CELL_M - CELL_M), 0.02)

    def test_large_coast_still_rejects_arrival(self):
        with self.assertRaisesRegex(HardwareFault, 'not at cell centre'):
            self.drive(0.04)

    def test_stationary_tof_recheck_continues_only_when_clear(self):
        for clears in (True, False):
            with self.subTest(clears=clears):
                io = self.robot()
                io._stop_event = threading.Event()
                io.tof_t, io.tof_mm, io.tof_seq = 100.0, 900, 1
                io.wall_center_mm, io.yaw = 201, 0
                io.side_speed = Mock(return_value=0)
                io.ensure_stopped = Mock()
                io.chassis = Mock()
                clock, speed, obstructed, recovered = [100.0], [0.0], [False], [False]
                def drive_speed(**kwargs):
                    speed[0] = kwargs['x']
                def stop():
                    speed[0] = 0.0
                    if obstructed[0] and clears:
                        obstructed[0] = False
                        recovered[0] = True
                def sleep(seconds):
                    clock[0] += seconds
                    io.pos = (io.pos[0] + speed[0] * seconds, 0)
                    if io.pos[0] >= 0.54 and not recovered[0]:
                        obstructed[0] = True
                    io.tof_mm = 201 if obstructed[0] else 201 + (0.60 - io.pos[0]) * 1000
                    io.pos_t = io.tof_t = clock[0]
                    io.tof_seq += 1
                io.chassis.drive_speed.side_effect = drive_speed
                io.stop_wheels.side_effect = stop
                io.read_tof_mm = Mock(side_effect=lambda: 201 if obstructed[0] else
                                      201 + (0.60 - io.pos[0]) * 1000)
                with patch('robo_app.maze.hardware.time.monotonic', side_effect=lambda: clock[0]), \
                        patch('robo_app.maze.hardware.time.sleep', side_effect=sleep):
                    if clears:
                        self.assertGreaterEqual(io.forward_one_cell(0, lambda: None), 0.98)
                        self.assertTrue(recovered[0])
                    else:
                        with self.assertRaisesRegex(HardwareFault, 'front ToF still blocked'):
                            io.forward_one_cell(0, lambda: None)
                        self.assertLess(io.pos[0], 0.58)
                io.read_tof_mm.assert_called_once()

    def test_mapped_end_wall_can_confirm_near_arrival_without_driving_closer(self):
        cases = ((False, 0.562, 213.5, 216.5, False),
                 (True, 0.562, 213.5, 216.5, True),
                 (True, 0.486, 226.0, 222.0, True),
                 (True, 0.450, 213.5, 216.5, False),
                 (True, 0.562, 170.0, 216.5, False))
        for known_end_wall, distance, reading, wall_center, confirmed in cases:
            with self.subTest(known_end_wall=known_end_wall, distance=distance, reading=reading):
                io = self.robot()
                io._stop_event = threading.Event()
                io.tof_t, io.tof_mm, io.tof_seq = 100.0, reading, 1
                io.wall_center_mm, io.yaw = wall_center, 0
                io.side_speed = Mock(return_value=0)
                io.ensure_stopped = Mock()
                io.chassis = Mock()
                io.read_tof_mm = Mock(return_value=reading)
                clock = [100.0]
                def drive_speed(**kwargs):
                    io.pos = (distance, 0.0)
                def sleep(seconds):
                    clock[0] += seconds
                    io.pos_t = io.tof_t = clock[0]
                    io.tof_seq += 1
                io.chassis.drive_speed.side_effect = drive_speed
                with patch('robo_app.maze.hardware.time.monotonic', side_effect=lambda: clock[0]), \
                        patch('robo_app.maze.hardware.time.sleep', side_effect=sleep):
                    if confirmed:
                        fraction = io.forward_one_cell(0, lambda: None, known_end_wall=True)
                        self.assertAlmostEqual(fraction * CELL_M, distance)
                        self.assertTrue(io.tof_arrival_confirmed)
                    else:
                        with self.assertRaisesRegex(HardwareFault, 'front ToF still blocked'):
                            io.forward_one_cell(0, lambda: None, known_end_wall=known_end_wall)
                        self.assertFalse(io.tof_arrival_confirmed)
                io.read_tof_mm.assert_called_once()

    def test_post_stop_tof_confirms_579_mm_arrival_only_for_mapped_wall(self):
        for known_end_wall, reading, accepted in ((True, 228.0, True),
                                                  (False, 228.0, False),
                                                  (True, 270.0, False)):
            with self.subTest(known_end_wall=known_end_wall, reading=reading):
                io = self.robot()
                io._stop_event = threading.Event()
                io.tof_t, io.tof_mm, io.tof_seq = 100.0, 228.0, 1
                io.wall_center_mm, io.yaw = 221.0, 0
                io.side_speed = Mock(return_value=0)
                io.ensure_stopped = Mock()
                io.chassis = Mock()
                io.read_tof_mm = Mock(return_value=reading)
                clock = [100.0]

                def drive_speed(**kwargs):
                    io.pos = (0.581, 0.0)

                def stop():
                    io.pos = (0.579, 0.0)

                def sleep(seconds):
                    clock[0] += seconds
                    io.pos_t = io.tof_t = clock[0]
                    io.tof_seq += 1

                io.chassis.drive_speed.side_effect = drive_speed
                io.stop_wheels.side_effect = stop
                with patch('robo_app.maze.hardware.time.monotonic', side_effect=lambda: clock[0]), \
                        patch('robo_app.maze.hardware.time.sleep', side_effect=sleep):
                    if accepted:
                        fraction = io.forward_one_cell(0, lambda: None, known_end_wall=known_end_wall)
                        self.assertAlmostEqual(fraction * CELL_M, 0.579)
                        self.assertTrue(io.tof_arrival_confirmed)
                    else:
                        with self.assertRaisesRegex(HardwareFault, 'not at cell centre'):
                            io.forward_one_cell(0, lambda: None, known_end_wall=known_end_wall)
                        self.assertFalse(io.tof_arrival_confirmed)


if __name__ == '__main__':
    unittest.main()
