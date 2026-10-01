"""Regression checks for side plates, multiple plates and actual cell arrival."""
import json
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from robo_app import aim as gs
from robo_app.maze.camera import FrameGrabber
from robo_app.maze.errors import HardwareFault
from robo_app.maze.hardware import RealRobot
from robo_app.maze.mapping import MazeMap
from robo_app.maze.mission import MazeMission
from robo_app.maze.simulation import SimRobot


class MazeTargetTests(unittest.TestCase):
    def mission(self):
        io = SimRobot((0, 0), 0, 4, 4)
        return MazeMission(io, FrameGrabber(io), MazeMap(4, 4), (0, 0), 0)

    def scene(self):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.rectangle(frame, (350, 185), (390, 235), (0, 255, 0), -1)
        cv2.rectangle(frame, (570, 185), (610, 235), (0, 255, 0), -1)
        return frame

    def test_two_green_plates_tracked_independently(self):
        m = self.mission()
        frame = self.scene()
        m.grabber.wait_new = Mock(side_effect=[(frame, i) for i in range(1, 6)])
        m.io.read_tof_mm = Mock(return_value=850)
        hits = m.look_for_target(0, 0)
        self.assertEqual(len(hits), 2)
        self.assertEqual([h[0] for h in hits], ['green', 'green'])
        self.assertGreater(abs(hits[0][3] - hits[1][3]), 0.1)
        self.assertEqual(m.maze.edge((0, 1), (0, 2)), -1)

    def test_near_wall_plate_is_detected_as_current_cell_target(self):
        io = SimRobot((3, 0), 1, 4, 4)
        io.TARGETS = {(3, 0): ('red', 'square', 1)}
        io.backoff = 0.12  # keeps the full nearby plate in camera view
        grabber = FrameGrabber(io)
        m = MazeMission(io, grabber, MazeMap(4, 4), (3, 0), 1)
        grabber.start()
        try:
            io.gimbal_to(-20, 0)
            hits = m.look_for_target(1, grabber.seq, -20, near=True)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0][:2], ('red', 'square'))
            self.assertLess(hits[0][2], 0.5)
            self.assertTrue(m._last_wall_check['confirmed'])
            self.assertTrue(m._last_wall_check['near'])
        finally:
            grabber.stop()
            grabber.join(timeout=2)

    def test_side_plate_does_not_create_front_wall(self):
        m = self.mission()
        frame = self.scene()
        m.grabber.wait_new = Mock(side_effect=[(frame, i) for i in range(1, 6)])
        # A genuine colored plate is visible, but the front wall is farther
        # than the adjacent cell: the image cannot manufacture a nearer wall.
        m.io.read_tof_mm = Mock(return_value=1450)
        self.assertEqual(m.look_for_target(0, 0), [])
        self.assertEqual(m.maze.edge((0, 1), (0, 2)), 0)
        self.assertEqual(m.maze.plate_count, 0)

    def test_missing_wall_reading_is_a_fault(self):
        m = self.mission()
        m.io.read_tof_mm = Mock(return_value=None)
        with self.assertRaises(HardwareFault):
            m.confirm_target_wall(0)
        self.assertEqual(m.maze.edge((0, 1), (0, 2)), 0)

    def test_color_votes_from_different_plates_are_not_pooled(self):
        m = self.mission()
        frame = self.scene()
        fd = gs.prepare_frame(frame)
        candidates = gs.detect_all_colors(fd, m.color_ranges, m.blind_zones)
        self.assertEqual(len(candidates), 2)
        m.grabber.wait_new = Mock(side_effect=[(frame, i) for i in range(1, 6)])
        m.confirm_target_wall = Mock(return_value=True)
        with patch('robo_app.maze.mission.gs.detect_all_colors', side_effect=[
                [candidates[0]], [candidates[1]], [candidates[0]], [candidates[1]], []]):
            self.assertEqual(m.look_for_target(0, 0), [])
        m.confirm_target_wall.assert_not_called()

    def test_map_preserves_same_color_plates_and_deduplicates_repeat_sightings(self):
        m = MazeMap(4, 4)
        for lateral in (-0.1, 0.1, -0.095):
            self.assertTrue(m.add_target((1, 1), 'green', 'rect', 0, (0.87, lateral), wall_verified=True))
        self.assertEqual(m.plate_count, 2)
        self.assertEqual(m.edge((1, 1), (1, 2)), 0)  # image alone cannot set a wall
        saved = json.loads(json.dumps(m.to_json()))
        restored = MazeMap.from_json(saved)
        self.assertEqual(restored.plate_count, 2)
        self.assertEqual(restored.to_json(), m.to_json())

    def test_same_plate_at_two_scan_pitches_is_one_target(self):
        maze = MazeMap(6, 6)
        maze.add_target((5, 0), 'red', 'square', 1, (0.893, 0.133), True)
        maze.add_target((5, 0), 'red', 'square', 1, (1.129, 0.168), True)
        self.assertEqual(len(maze.plates_at((5, 0))), 1)
        restored = MazeMap.from_json(maze.to_json())
        self.assertEqual(len(restored.plates_at((5, 0))), 1)
        maze.add_target((5, 0), 'red', 'square', 1, (1.026, 0.05), True,
                        near_seen=(0.426, 0.05))
        self.assertEqual(len(maze.plates_at((5, 0))), 1)
        self.assertEqual(maze.plates_at((5, 0))[0]['near_seen'], (0.426, 0.05))

    def test_explore_fire_routes_to_matching_target_before_empty_cell(self):
        maze = MazeMap(6, 6)
        here, target, empty = (4, 0), (5, 0), (4, 1)
        maze.visited.add(here)
        maze.set_edge(here, target, 1)
        maze.set_edge(here, empty, 1)
        maze.add_target(target, 'red', 'square', 1, (1.01, 0.05), True)
        self.assertEqual(maze.path_to_unvisited(here, 1), [here, empty])
        self.assertEqual(maze.path_to_unvisited(here, 1, {target}), [here, target])

    def test_target_cell_is_visited_even_when_all_its_edges_are_known(self):
        maze = MazeMap(2, 1)
        start, target = (0, 0), (1, 0)
        maze.visited.add(start)
        maze.set_edge(start, target, 1)
        maze.add_target(target, 'red', 'square', 1, (0.9, 0.0), True)
        self.assertEqual(maze.unknown_dirs(target), [])
        self.assertEqual(maze.path_to_unvisited(start, 1), [start, target])
        maze.visited.add(target)
        self.assertIsNone(maze.path_to_unvisited(target, 1))

    def test_round_one_dijkstra_and_round_two_keeps_astar(self):
        maze = MazeMap(3, 2)
        for y in range(2):
            for x in range(2):
                maze.set_edge((x, y), (x + 1, y), 1)
        for x in range(3):
            maze.set_edge((x, 0), (x, 1), 1)
        start, target, goal = (0, 0), (1, 0), (2, 0)
        maze.add_target(target, 'red', 'square', 2, (0.35, 0), True)
        maze.visited.update({start, target, (0, 1), (1, 1), (2, 1)})

        self.assertEqual(maze.path_to_unvisited(start, 0), [start, target, goal])
        self.assertEqual(maze.path_to(start, 0, goal),
                         [start, (0, 1), (1, 1), (2, 1), goal])

    def test_round_one_dijkstra_can_choose_two_straight_cells_over_a_uturn(self):
        maze = MazeMap(4, 1)
        start = (1, 0)
        maze.visited.update({start, (2, 0)})
        for a, b in (((0, 0), start), (start, (2, 0)), ((2, 0), (3, 0))):
            maze.set_edge(a, b, 1)
        self.assertEqual(maze.path_to_unvisited(start, 1),
                         [start, (2, 0), (3, 0)])

    def test_round_one_dijkstra_does_not_detour_to_a_far_target(self):
        maze = MazeMap(4, 1)
        start = (0, 0)
        maze.visited.add(start)
        for x in range(3):
            maze.set_edge((x, 0), (x + 1, 0), 1)
        maze.add_target((3, 0), 'red', 'square', 1, (0.95, 0), True)
        self.assertEqual(maze.path_to_unvisited(start, 1, {(3, 0)}),
                         [start, (1, 0)])

    def test_multiple_wall_directions_keep_separate_shooting_spots(self):
        m = MazeMap(3, 3)
        m.visited.update(((1, 0), (0, 1)))
        m.set_edge((1, 0), (1, 1), 1)
        m.set_edge((0, 1), (1, 1), 1)
        m.add_target((1, 1), 'red', 'rect', 0, (0.87, 0), True)
        m.add_target((1, 1), 'blue', 'rect', 1, (0.87, 0), True)
        self.assertEqual(m.shooting_spots((1, 1), 0), [((1, 1), 0)])
        self.assertEqual(m.shooting_spots((1, 1), 1), [((1, 1), 1)])

    def test_shoot_run_processes_every_plate_in_picked_cell(self):
        m = self.mission()
        cell = (0, 1)
        m.maze.visited.add((0, 0))
        m.maze.set_edge((0, 0), cell, 1)
        for lateral in (-0.1, 0.1):
            m.maze.add_target(cell, 'green', 'rect', 0, (0.87, lateral), True)
        m.maze.explored = True
        m.face = Mock()
        m.go_to = Mock(return_value=True)
        m.aim_and_fire = Mock(return_value='AIMED')
        m.save_map = Mock()
        m.shoot_run(False, 'FROM HERE', 'STOP', 'PICK ORDER', [cell], None)
        self.assertEqual([c.args for c in m.aim_and_fire.call_args_list], [(cell, 0), (cell, 1)])
        self.assertEqual(m.results[cell], '2/2')
        self.assertEqual(list(m.plate_results.values()), ['AIMED', 'AIMED'])

    def test_unverified_legacy_target_is_not_engaged(self):
        m = self.mission()
        cell = (0, 1)
        m.maze.visited.add((0, 0))
        m.maze.set_edge((0, 0), cell, 1)
        m.maze.add_target(cell, 'red', 'rect', 0, (0.87, 0))
        m.maze.explored = True
        m.aim_and_fire = Mock()
        m.save_map = Mock()
        m.shoot_run(False, 'FROM HERE', 'STOP', 'NEAREST', [], None)
        m.aim_and_fire.assert_not_called()
        self.assertEqual(m.plate_results[(cell, 0)], 'UNVERIFIED')

    def test_target_wall_marker_does_not_shift_map_wall_coordinates(self):
        from robo_app.maze.ui import App, WALL_COLOR
        m = self.mission()
        m.maze = MazeMap(6, 6)
        m.maze.add_target((2, 2), 'red', 'rect', 1, (0.87, 0), True)
        m.maze.add_target((2, 2), 'green', 'square', 1, (0.87, 0.15), True)
        m.maze.set_edge((4, 4), (5, 4), -1)
        app = App(m, m.grabber)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        app.draw_map(canvas)
        x0, y0, size, wx, wy, width, height = app._grid
        x = x0 + (5 - wx) * size
        y = y0 + (wy + height - 1 - 4) * size + size // 2
        self.assertEqual(tuple(canvas[y, x]), WALL_COLOR)

    def test_simulation_hits_two_distinct_same_color_plates(self):
        import tempfile
        from pathlib import Path
        io = SimRobot((2, 0), 1, 4, 4)
        io.backoff = 0.20
        io.EXTRA_TARGETS = [((3, 0), 'green', 'square', 1, 0.14)]
        grabber = FrameGrabber(io)
        m = MazeMission(io, grabber, MazeMap(4, 4), (2, 0), 1)
        m.maze.visited.add((2, 0))
        m.maze.set_edge((2, 0), (3, 0), 1)
        grabber.start()
        try:
            io.gimbal_to(-15, 0)
            hits = m.look_for_target(1, grabber.seq)
            self.assertEqual(len(hits), 2)
            for color, shape, distance, lateral in hits:
                m.maze.add_target((3, 0), color, shape, 1, (distance, lateral), True)
            m.maze.explored = True
            with tempfile.TemporaryDirectory() as folder, patch('robo_app.maze.mission.MAP_FILE', Path(folder) / 'map.json'):
                m.shoot_run(True, 'FROM HERE', 'STOP', 'NEAREST', [], None)
            self.assertEqual(len(m.plate_results), 2)
            self.assertEqual(set(m.plate_results.values()), {'SHOT'})
            self.assertEqual(set(io.shot_plates), {((3, 0), 0.0), ((3, 0), 0.14)})
            self.assertEqual(len(io.shot_plates), 6)
        finally:
            m.stop()
            grabber.stop()
            grabber.join(timeout=2)

    def test_short_drive_does_not_advance_map_position(self):
        m = self.mission()
        m.face = Mock()
        m.io.forward_one_cell = Mock(return_value=0.85)
        m.io.move_straight = Mock()
        m.io.read_tof_mm = Mock(return_value=1000)
        self.assertFalse(m.drive_path([(0, 0), (0, 1)]))
        self.assertEqual(m.cell, (0, 0))
        self.assertNotIn((0, 1), m.maze.visited)

    def test_overshoot_invalidates_pose(self):
        m = self.mission()
        m.face = Mock()
        m.io.forward_one_cell = Mock(return_value=1.1)
        with self.assertRaises(HardwareFault):
            m.drive_path([(0, 0), (0, 1)])
        self.assertEqual(m.cell, (0, 0))
        self.assertTrue(m.pose_uncertain)

    def driving_robot(self, front=1000):
        io = RealRobot.__new__(RealRobot)
        io._stop_event = threading.Event()
        io.pos = (0.0, 0.0)
        io.pos_t = io.tof_t = 100.0
        io.tof_mm, io.tof_seq = front, 1
        io.wall_center_mm = 250
        io.yaw = 0
        io.stop_wheels = Mock()
        io.ensure_stopped = Mock()
        io.side_speed = Mock(return_value=0)
        io.chassis = Mock()
        def drive(**kwargs):
            io.pos = (io.pos[0] + 0.03, 0)
            io.tof_seq += 1
        io.chassis.drive_speed.side_effect = drive
        return io

    def test_hardware_requires_fresh_odometry(self):
        io = self.driving_robot()
        io.pos_t = 0
        with patch('robo_app.maze.hardware.time.monotonic', return_value=100):
            with self.assertRaises(HardwareFault):
                io.forward_one_cell(0, lambda: None)
        io.chassis.drive_speed.assert_not_called()
        io.stop_wheels.assert_called()

    def test_front_tof_cannot_claim_arrival_after_only_nine_cm(self):
        io = self.driving_robot(front=250)
        io.read_tof_mm = Mock(return_value=250)
        with patch('robo_app.maze.hardware.time.monotonic', return_value=100), patch('robo_app.maze.hardware.time.sleep'):
            with self.assertRaises(HardwareFault):
                io.forward_one_cell(0, lambda: None)
        self.assertLess(io.pos[0], 0.2)
        io.stop_wheels.assert_called()

    def test_hardware_accepts_confirmed_full_cell(self):
        io = self.driving_robot()
        with patch('robo_app.maze.hardware.time.monotonic', return_value=100), patch('robo_app.maze.hardware.time.sleep'):
            fraction = io.forward_one_cell(0, lambda: None)
        self.assertAlmostEqual(fraction, 1.0)
        io.stop_wheels.assert_called()


if __name__ == '__main__':
    unittest.main()
