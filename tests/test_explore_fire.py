"""Selected-target firing during exploration, with shot persistence."""
import json
import math
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robo_app.maze.camera import FrameGrabber
from robo_app.maze.errors import Aborted
from robo_app.maze.mapping import MazeMap
from robo_app.maze.hardware import RealRobot
from robo_app.maze.mission import MazeMission
from robo_app.maze.simulation import SimRobot
from robo_app import aim as gs
from robo_app.maze.ui import App


class ExploreFireTests(unittest.TestCase):
    def mission(self):
        io = SimRobot((0, 0), 0, 4, 4)
        m = MazeMission(io, FrameGrabber(io), MazeMap(4, 4), (0, 0), 0)
        m.save_map = Mock()
        return m

    def scene(self):
        m = self.mission()
        target = (0, 0)
        m.maze.visited.add(target)
        m.maze.set_edge(target, (0, 1), -1)
        for color, shape, offset in [('red', 'circle', -0.15), ('red', 'rect', 0), ('blue', 'circle', 0.15)]:
            m.maze.add_target(target, color, shape, 0, (0.95, offset), True,
                              near_seen=(0.35, offset))
        m._explore_fire = True
        m._explore_colors = frozenset({'red'})
        m._explore_shapes = frozenset({'circle'})
        m.face = Mock()
        m.aim_and_fire = Mock(return_value='SHOT')
        return m, target

    def test_live_exploration_selects_color_and_shape_together(self):
        m, target = self.scene()
        m.shoot_visible_during_explore()
        m.aim_and_fire.assert_called_once_with(target, 0, finish_on_request=True)
        self.assertEqual([p['shot'] for p in m.maze.plates_at(target)], [True, False, False])
        self.assertFalse(m.armed)
        self.assertFalse(m.fire_ctrl.armed)
        m.save_map.assert_called_once_with(quiet=True)

    @patch('robo_app.maze.mission.time.sleep')
    def test_near_plate_is_aimed_before_scanning_next_direction(self, sleep):
        m = self.mission()
        m._explore_fire = True
        m._explore_colors = frozenset({'red'})
        m._explore_shapes = frozenset({'circle'})
        m.io.read_tof_mm = Mock(return_value=300)
        m.face = Mock()
        order = []
        def look(direction, seq, pitch, near=False):
            order.append(f'look {direction} {pitch:+.0f}')
            if direction == 3 and near and pitch == -20:
                return [('red', 'circle', 0.35, 0.0)]
            return []
        def aim(target, index, finish_on_request=False):
            order.append('aim')
            return 'SHOT'
        m.look_for_target = Mock(side_effect=look)
        m.aim_and_fire = Mock(side_effect=aim)
        m.scan_cell()
        self.assertLess(order.index('look 3 +0'), order.index('aim'))
        self.assertLess(order.index('aim'), order.index('look 0 -20'))
        self.assertTrue(m.maze.plates_at((0, 0))[0]['shot'])
        m.aim_and_fire.assert_called_once_with((0, 0), 0, finish_on_request=True)

    def test_adjacent_plate_is_mapped_but_never_fired_from_previous_cell(self):
        m = self.mission()
        m.maze.visited.add((0, 0))
        m.maze.set_edge((0, 0), (0, 1), 1)
        m.maze.add_target((0, 1), 'red', 'circle', 0, (0.87, 0), True)
        m._explore_colors = frozenset({'red'})
        m._explore_shapes = frozenset({'circle'})
        m.aim_and_fire = Mock(return_value='SHOT')
        m.shoot_visible_during_explore()
        m.aim_and_fire.assert_not_called()
        self.assertFalse(m.maze.plates_at((0, 1))[0]['shot'])

    def test_aim_refuses_cross_cell_shot_even_if_called_directly(self):
        m = self.mission()
        m.maze.add_target((0, 1), 'red', 'circle', 0, (0.87, 0), True)
        m.io.fire = Mock()
        self.assertEqual(m.aim_and_fire((0, 1)), 'SKIPPED')
        m.io.fire.assert_not_called()

    @patch('robo_app.maze.mission.time.sleep')
    def test_map_only_skips_open_directions_without_firing(self, sleep):
        m = self.mission()
        m.io.read_tof_mm = Mock(return_value=900)
        m.io.gimbal_to = Mock()
        m.look_for_target = Mock(return_value=[])
        m.shoot_visible_during_explore = Mock()
        m.scan_cell()
        poses = [(call.args[0], call.args[2]) for call in m.look_for_target.call_args_list]
        self.assertEqual(poses, [(3, pitch)
                                 for pitch in (-20, -15, -10, -5, 0)])
        commands = list(m.io.gimbal_to.call_args_list)
        self.assertEqual([c.args for c in commands[:5]],
                         [(0, -90), (-20, -90), (-15, -90), (-10, -90), (-5, -90)])
        self.assertEqual(commands[1].kwargs.get('speed'), 90)
        self.assertFalse(any(c.args[0] > 0 for c in commands))
        self.assertEqual(m.maze.edge((0, 0), (0, 1)), 1)
        self.assertEqual(m.maze.edge((0, 0), (1, 0)), 1)
        m.shoot_visible_during_explore.assert_not_called()

    @patch('robo_app.maze.mission.time.sleep')
    def test_cell_surrounded_by_walls_skips_rear_target_sweep(self, sleep):
        io = SimRobot((0, 0), 0, 1, 1)
        m = MazeMission(io, FrameGrabber(io), MazeMap(1, 1), (0, 0), 0)
        m.save_map = Mock()
        m.look_for_target = Mock(return_value=[])
        m.scan_cell()
        self.assertEqual([(c.args[0], c.args[2], c.kwargs.get('near'))
                          for c in m.look_for_target.call_args_list],
                         [(d, pitch, True) for d in (3, 0, 1)
                          for pitch in (-20, -15, -10, -5, 0)])

    def test_plate_distance_uses_the_pitch_of_its_scan(self):
        fd = Mock(size=(960, 540))
        for pitch in (-20, 0):
            elevation = -math.degrees(math.atan2(0.22 - 0.10, 0.87 - 0.10))
            x, y = gs.degrees_to_pixel(0, elevation - pitch, 960, 540)
            distance, lateral = MazeMission.target_ground(fd, Mock(cx=x, cy=y), pitch)
            self.assertAlmostEqual(distance, 0.87, delta=0.02)
            self.assertAlmostEqual(lateral, 0, delta=0.01)

    def test_real_gimbal_uses_configured_scan_speed(self):
        io = RealRobot.__new__(RealRobot)
        io._stop_event = threading.Event()
        io.gimbal = Mock()
        io._wait_action = Mock()
        io.gimbal_to(-5, 0, speed=90)
        io.gimbal.moveto.assert_called_once_with(
            pitch=-5, yaw=0, pitch_speed=90, yaw_speed=90)

    @patch('robo_app.maze.mission.time.sleep')
    def test_wall_plate_is_fired_from_current_cell_after_near_scan(self, sleep):
        io = SimRobot((3, 0), 1, 4, 4)
        m = MazeMission(io, FrameGrabber(io), MazeMap(4, 4), (3, 0), 1)
        m.save_map = Mock()
        m.maze.visited.add((2, 0))
        m.maze.set_edge((2, 0), (3, 0), 1)
        m.io.read_tof_mm = Mock(return_value=300)
        m.look_for_target = Mock(side_effect=lambda d, seq, pitch, near=False:
                                 [('red', 'square' if pitch in (-20, -15) else 'rect', 0.35, 0.02)]
                                 if d == 1 and near and pitch in (-20, -15, -10) else [])
        m._explore_fire = True
        m._explore_colors = frozenset({'red'})
        m._explore_shapes = frozenset({'square'})
        m.face = Mock()
        m.aim_and_fire = Mock(return_value='SHOT')
        m.scan_cell()
        self.assertEqual(m.maze.plates_at((3, 0))[0]['seen'], (0.95, 0.02))
        self.assertEqual(m.maze.plates_at((3, 0))[0]['near_seen'], (0.35, 0.02))
        self.assertEqual(MazeMap.from_json(m.maze.to_json()).plates_at((3, 0))[0]['near_seen'],
                         (0.35, 0.02))
        self.assertEqual(m.maze.plates_at((3, 0))[0]['wall'], 1)
        self.assertEqual(m.maze.plates_at((3, 0))[0]['shape'], 'square')
        self.assertTrue(m.maze.plates_at((3, 0))[0]['shot'])
        self.assertEqual(m.cell, (3, 0))
        self.assertEqual(io.cell, (3, 0))
        m.aim_and_fire.assert_called_once_with((3, 0), 0, finish_on_request=True)
        self.assertEqual(m.cell, (3, 0))
        self.assertEqual(m.aim_and_fire.call_count, 1)

    @patch('robo_app.maze.mission.time.sleep')
    def test_cropped_shallow_view_does_not_override_deep_plate_range(self, sleep):
        m = self.mission()
        m.io.read_tof_mm = Mock(return_value=300)
        m.look_for_target = Mock(side_effect=lambda d, seq, pitch, near=False:
            ([('red', 'rect', 0.342, 0.05), ('red', 'rect', 0.359, 0.05)] if pitch == -20 else
             [('red', 'square', 0.399, 0.05)] if pitch == -15 else
             [('red', 'rect', 0.434, 0.05)] if pitch == -10 else [])
            if d == 0 and near else [])
        m.scan_cell()
        plates = m.maze.plates_at((0, 0))
        self.assertEqual(len(plates), 1)
        self.assertEqual(plates[0]['shape'], 'rect')
        self.assertAlmostEqual(plates[0]['near_seen'][0], 0.3505)

    def test_aim_backs_up_only_with_confirmed_open_rear(self):
        target = (1, 0)
        m = MazeMission(SimRobot(target, 1, 4, 4), None, MazeMap(4, 4), target, 1)
        m.grabber = Mock(seq=0)
        m.save_map = Mock()
        m.maze.add_target(target, 'red', 'rect', 1, (0.942, 0.0), True,
                          near_seen=(0.342, 0.0))
        m.confirm_target_wall = Mock(return_value=True)
        m.move_in_cell = Mock()
        m.io.fire = Mock()
        m.grabber.wait_new = Mock(side_effect=Aborted())
        self.assertEqual(m.aim_and_fire(target), 'SKIPPED')
        m.move_in_cell.assert_not_called()
        m.maze.set_edge(target, (0, 0), 1)
        with self.assertRaises(Aborted):
            m.aim_and_fire(target)
        m.move_in_cell.assert_called_once()
        self.assertLess(m.move_in_cell.call_args.args[0], 0)
        self.assertLessEqual(-m.move_in_cell.call_args.args[0], 0.14)
        self.assertTrue(m.pose_uncertain)
        m.io.fire.assert_not_called()

    def test_aim_backoff_skips_when_rear_tof_is_blocked(self):
        target = (1, 0)
        m = MazeMission(SimRobot(target, 1, 4, 4), None, MazeMap(4, 4), target, 1)
        m.maze.add_target(target, 'red', 'rect', 1, (0.942, 0.0), True,
                          near_seen=(0.342, 0.0))
        m.maze.set_edge(target, (0, 0), 1)
        m.confirm_target_wall = Mock(return_value=True)
        m.io.read_tof_mm = Mock(return_value=300)
        m.move_in_cell = Mock()
        self.assertEqual(m.aim_and_fire(target), 'SKIPPED')
        m.move_in_cell.assert_not_called()
        self.assertFalse(m.pose_uncertain)

    def test_explore_rescans_saved_unshot_plate(self):
        m = self.mission()
        m.maze.add_target((0, 0), 'red', 'square', 0, (0.956, 0), True,
                          near_seen=(0.356, 0))
        m.looked.add(((0, 0), 0))
        m.pose_uncertain = False
        was_looked = []
        m.scan_cell = Mock(side_effect=lambda: was_looked.append(((0, 0), 0) in m.looked))
        m.explore(fire=True, colors={'red'}, shapes={'square'})
        self.assertEqual(was_looked, [False])

    def test_round_one_dijkstra_revisits_saved_unshot_target(self):
        maze = MazeMap(3, 1)
        maze.visited.update({(1, 0), (2, 0)})
        maze.set_edge((2, 0), (1, 0), 1)
        maze.set_edge((1, 0), (0, 0), 1)
        self.assertEqual(maze.path_to_unvisited((2, 0), 3, {(1, 0)}),
                         [(2, 0), (1, 0)])

    def test_near_wall_must_still_be_confirmed_before_firing(self):
        m = self.mission()
        m.maze.add_target((0, 0), 'red', 'square', 0, (0.95, 0.0), True,
                          near_seen=(0.35, 0.0))
        m.confirm_target_wall = Mock(return_value=False)
        m.io.fire = Mock()
        self.assertEqual(m.aim_and_fire((0, 0), 0), 'SKIPPED')
        m.confirm_target_wall.assert_called_once_with(0, near=True)
        m.io.fire.assert_not_called()

    def test_too_close_target_skips_when_required_backoff_exceeds_limit(self):
        m = self.mission()
        m.maze.add_target((0, 0), 'red', 'square', 0, (0.87, 0.0), True,
                          near_seen=(0.27, 0.0))
        m.confirm_target_wall = Mock(return_value=True)
        m.move_in_cell = Mock()
        m.io.fire = Mock()
        self.assertEqual(m.aim_and_fire((0, 0), 0), 'SKIPPED')
        m.move_in_cell.assert_not_called()
        m.io.fire.assert_not_called()

    def test_close_plate_is_shot_after_safe_backoff_and_returns_to_start_pose(self):
        io = SimRobot((3, 0), 1, 4, 4)
        io.backoff = 0.08
        grabber = FrameGrabber(io)
        m = MazeMission(io, grabber, MazeMap(4, 4), (3, 0), 1)
        m.save_map = Mock()
        m.maze.set_edge((3, 0), (2, 0), 1)
        m.maze.add_target((3, 0), 'green', 'square', 1, (0.95, 0.0), True,
                          near_seen=(0.35, 0.0))
        m.armed = m.fire_ctrl.armed = True
        grabber.start()
        try:
            self.assertEqual(m.aim_and_fire((3, 0), 0), 'SHOT')
            self.assertEqual(len(io.shots), 3)
            self.assertAlmostEqual(io.backoff, 0.08)
            self.assertFalse(m.pose_uncertain)
        finally:
            grabber.stop()
            grabber.join(timeout=2)

    def test_simulated_near_plate_is_hit_from_its_own_cell(self):
        io = SimRobot((3, 0), 1, 4, 4)
        io.backoff = 0.20
        grabber = FrameGrabber(io)
        m = MazeMission(io, grabber, MazeMap(4, 4), (3, 0), 1)
        m.save_map = Mock()
        m.maze.visited.add((3, 0))
        m.maze.add_target((3, 0), 'green', 'square', 1, (0.99, 0.0), True,
                          near_seen=(0.47, 0.0))
        m.armed = m.fire_ctrl.armed = True
        grabber.start()
        try:
            self.assertEqual(m.aim_and_fire((3, 0), 0), 'SHOT')
            self.assertEqual([shot[0] for shot in io.shots], [(3, 0)] * 3)
            self.assertEqual(m.maze.plates_at((3, 0))[0]['shots_fired'], 3)
            self.assertEqual(io.cell, (3, 0))
            self.assertAlmostEqual(io.backoff, 0.20)
        finally:
            grabber.stop()
            grabber.join(timeout=2)

    def test_resuming_after_first_shot_fires_only_two_more(self):
        io = SimRobot((3, 0), 1, 4, 4)
        io.backoff = 0.20
        grabber = FrameGrabber(io)
        m = MazeMission(io, grabber, MazeMap(4, 4), (3, 0), 1)
        m.save_map = Mock()
        m.maze.add_target((3, 0), 'green', 'square', 1, (0.99, 0), True,
                          near_seen=(0.47, 0), shots_fired=1)
        m.armed = m.fire_ctrl.armed = True
        grabber.start()
        try:
            self.assertEqual(m.aim_and_fire((3, 0), 0), 'SHOT')
            self.assertEqual(len(io.shots), 2)
            self.assertEqual(m.maze.plates_at((3, 0))[0]['shots_fired'], 3)
            self.assertEqual(MazeMap.from_json(m.maze.to_json()).plates_at((3, 0))[0]['shots_fired'], 3)
        finally:
            grabber.stop()
            grabber.join(timeout=2)

    def test_loaded_two_shot_target_needs_one_more_shot(self):
        m = self.mission()
        m.maze.add_target((0, 0), 'red', 'circle', 0, (0.95, 0), True,
                          near_seen=(0.35, 0), shot=True, shots_fired=2)
        m.restore_state({})
        plate = m.maze.plates_at((0, 0))[0]
        self.assertFalse(plate['shot'])
        self.assertEqual(plate['shots_fired'], 2)

    def test_measured_shot_offset_changes_the_lock_point(self):
        m = self.mission()
        m.maze.add_target((0, 0), 'red', '?', 0, (0.99, 0.0), True,
                          near_seen=(0.47, 0.0))
        m.confirm_target_wall = Mock(return_value=True)
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        candidate = gs.Target('red', 480, 270, 100, (475, 265, 10, 10))
        m.grabber.wait_new = Mock(side_effect=[(frame, 1), Aborted()])
        m.fire_ctrl.evaluate = Mock(return_value=('TRACKING', False))
        with patch('robo_app.maze.mission.shot_offset', return_value=(1.0, -2.0)), \
                patch('robo_app.maze.mission.gs.find_targets', return_value=[candidate]):
            with self.assertRaises(Aborted):
                m.aim_and_fire((0, 0), 0)
        m.fire_ctrl.evaluate.assert_called_once_with(-1.0, 2.0, False)

    def test_already_shot_plate_is_not_repeated_on_second_visit(self):
        m, _ = self.scene()
        m.shoot_visible_during_explore()
        m._explore_attempted.clear()
        m.shoot_visible_during_explore()
        self.assertEqual(m.aim_and_fire.call_count, 1)

    def test_shape_only_filter_and_unknown_shape(self):
        m = self.mission()
        m._explore_shapes = frozenset({'circle'})
        self.assertTrue(m.matches_explore_filter({'color': 'blue', 'shape': 'circle'}))
        self.assertFalse(m.matches_explore_filter({'color': 'red', 'shape': '?'}))
        self.assertFalse(m.matches_explore_filter({'color': 'red', 'shape': 'square'}))

    def test_color_only_filter(self):
        m = self.mission()
        m._explore_colors = frozenset({'red'})
        self.assertTrue(m.matches_explore_filter({'color': 'red', 'shape': 'rect'}))
        self.assertFalse(m.matches_explore_filter({'color': 'blue', 'shape': 'circle'}))

    def test_no_filter_does_not_start_live_exploration(self):
        m = self.mission()
        m.scan_cell = Mock()
        m.explore(True)
        m.scan_cell.assert_not_called()
        self.assertFalse(m.armed)

    def test_round_one_continues_past_time_notice_to_visit_target_cell(self):
        m = self.mission()
        target = (0, 1)
        m.maze.set_edge((0, 0), target, 1)
        m.maze.add_target(target, 'red', 'square', 0, (0.9, 0), True)
        m.scan_cell = Mock(side_effect=lambda: m.maze.visited.add(m.cell))

        def arrive(path, on_arrive=None):
            m.cell = path[-1]
            on_arrive()
            return True

        m.drive_path = Mock(side_effect=arrive)
        with patch('robo_app.maze.mission.time.monotonic', side_effect=[0, 600, 601, 602, 603]):
            m.explore(False)
        self.assertIn(target, m.maze.visited)
        self.assertTrue(m.maze.explored)
        m.drive_path.assert_called_once()

    def test_finish_request_prevents_new_engagement(self):
        m, _ = self.scene()
        m.finish_requested = True
        m.shoot_visible_during_explore()
        m.aim_and_fire.assert_not_called()

    def test_abort_disarms_and_does_not_mark_a_shot(self):
        m, target = self.scene()
        m.aim_and_fire.side_effect = Aborted()
        with self.assertRaises(Aborted):
            m.shoot_visible_during_explore()
        self.assertFalse(m.armed)
        self.assertFalse(m.fire_ctrl.armed)
        self.assertFalse(m.maze.plates_at(target)[0]['shot'])

    def test_shot_survives_resighting_and_json_roundtrip(self):
        m, target = self.scene()
        m.shoot_visible_during_explore()
        m.maze.add_target(target, 'red', 'circle', 0, (0.95, -0.145), True)
        restored = MazeMap.from_json(json.loads(json.dumps(m.maze.to_json())))
        self.assertTrue(restored.plates_at(target)[0]['shot'])
        self.assertFalse(restored.plates_at(target)[1]['shot'])

    def test_gui_defaults_to_red_circle_and_passes_live_filter(self):
        m = self.mission()
        app = App(m, m.grabber)
        m.start_job = Mock()
        self.assertEqual(app.key_action(ord('e')), 'explore')
        app.do('explore')
        m.start_job.assert_called_once_with('EXPLORE', m.explore, True, {'red'}, {'circle'})

    def test_gui_filters_are_changeable_and_map_only_is_available(self):
        m = self.mission()
        app = App(m, m.grabber)
        app.do('filter_color')
        app.do('filter_shape')
        self.assertEqual(app.color_choices[app.color_filter], 'green')
        self.assertEqual(app.shape_choices[app.shape_filter], 'square')
        m.start_job = Mock()
        app.do('map_only')
        m.start_job.assert_called_once_with('EXPLORE', m.explore, False)

    def test_filtered_plan_uses_only_matching_plates_shooting_spots(self):
        m = self.mission()
        target = (1, 1)
        m.maze.visited.add((0, 1))
        m.maze.set_edge((0, 0), (0, 1), 1)
        m.maze.set_edge((0, 1), target, 1)
        m.maze.add_target(target, 'red', 'circle', 0, (0.87, 0), True)
        m.maze.add_target(target, 'green', 'rect', 1, (0.87, 0), True)
        self.assertEqual(m.plan_targets('NEAREST', [], {'red'}, {'circle'}), [target])
        self.assertNotIn((target, 0), m.plate_results)
        self.assertNotIn((target, 1), m.plate_results)

    def test_second_round_obeys_filter_and_skips_previous_shot(self):
        m, target = self.scene()
        m.shoot_visible_during_explore()
        m.maze.explored = True
        m.aim_and_fire.reset_mock()
        m.shoot_run(True, 'FROM HERE', 'STOP', 'NEAREST', [], None, {'red'}, {'circle'})
        m.aim_and_fire.assert_not_called()
        self.assertEqual(m.plate_results[(target, 0)], 'ALREADY SHOT')
        self.assertEqual(len(m.plate_results), 1)


if __name__ == '__main__':
    unittest.main()
