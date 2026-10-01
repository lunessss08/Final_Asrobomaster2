"""Grid SLAM integration checks; no hardware or production map writes."""
import unittest
from unittest.mock import Mock, patch

from robo_app.maze.camera import FrameGrabber
from robo_app.maze.errors import HardwareFault
from robo_app.maze.grid_slam import GridSLAM
from robo_app.maze.mapping import MazeMap
from robo_app.maze.mission import MazeMission
from robo_app.maze.simulation import SimRobot


class SlamIntegrationTests(unittest.TestCase):
    def mission(self):
        robot = SimRobot((0, 0), 0, 4, 4)
        robot._pause = Mock()
        m = MazeMission(robot, FrameGrabber(robot), MazeMap(4, 4), (0, 0), 0, use_slam=True)
        m.slam.rng.seed(3)
        m.save_map = Mock()
        m.look_for_target = Mock(return_value=[])
        return m

    def corridor(self):
        maze = MazeMap(3, 1)
        maze.set_edge((0, 0), (1, 0), 1)
        maze.set_edge((1, 0), (2, 0), 1)
        slam = GridSLAM(maze, (0, 0), 1, seed=3)
        slam.observe({0: 250, 1: 1450, 2: 250, 3: 250}, 1)
        return slam

    def test_slipped_and_overshot_motion_are_corrected(self):
        for actual, east, west in (((0, 0), 1450, 250), ((2, 0), 250, 1450)):
            with self.subTest(actual=actual):
                slam = self.corridor()
                slam.predict_move(1)
                confidence = slam.observe({0: 250, 1: east, 2: 250, 3: west}, 1)
                self.assertEqual(slam.cell, actual)
                self.assertGreaterEqual(confidence, 0.70)

    @patch('robo_app.maze.mission.time.sleep')
    def test_complete_exploration_preserves_target_metadata(self, sleep):
        m = self.mission()
        authoritative = m.maze
        m.maze.add_target((1, 0), 'red', 'circle', 1, (0.87, 0), wall_verified=True, shot=True)
        # A target recorded after the particle maps were created must survive.
        m.explore()
        self.assertIs(m.maze, authoritative)
        self.assertTrue(m.maze.explored)
        self.assertEqual(m.cell, m.io.cell)
        self.assertTrue(m.maze.plates_at((1, 0))[0]['shot'])
        self.assertGreater(len(m.maze.visited), 8)

    @patch('robo_app.maze.mission.time.sleep')
    def test_localization_uses_only_level_rays(self, sleep):
        m = self.mission()
        original = m.io.gimbal_to
        m.io.gimbal_to = Mock(side_effect=original)
        m.localize_slam()
        self.assertTrue(all(call.args[0] == 0 for call in m.io.gimbal_to.call_args_list))
        self.assertIn((0, 180), [call.args for call in m.io.gimbal_to.call_args_list])

    @patch('robo_app.maze.mission.time.sleep')
    def test_missing_tof_stops_before_camera_observations(self, sleep):
        m = self.mission()
        m.io.read_tof_mm = Mock(return_value=None)
        with self.assertRaises(HardwareFault):
            m.scan_cell()
        self.assertTrue(m.pose_uncertain)
        m.look_for_target.assert_not_called()
        m.save_map.assert_not_called()

    @patch('robo_app.maze.mission.time.sleep')
    def test_low_confidence_retry_requires_same_pose(self, sleep):
        m = self.mission()
        m.slam.observe = Mock(side_effect=[0.69, 0.85])
        m.localize_slam()
        self.assertEqual(m.slam.observe.call_count, 2)
        m = self.mission()
        poses = iter(((0, 0), (1, 0)))
        def observe(scan, heading):
            m.slam.cell = next(poses)
            return 0.69 if m.slam.cell == (0, 0) else 0.85
        m.slam.observe = Mock(side_effect=observe)
        with self.assertRaises(HardwareFault):
            m.localize_slam()
        self.assertTrue(m.pose_uncertain)

    def test_partial_motion_does_not_invent_wall_or_reverse(self):
        m = self.mission()
        m.maze.set_edge((0, 0), (1, 0), 1)
        m.face = Mock()
        m.io.forward_one_cell = Mock(return_value=0.76)
        m.io.move_straight = Mock()
        with self.assertRaises(HardwareFault):
            m.drive_path([(0, 0), (1, 0)])
        self.assertTrue(m.pose_uncertain)
        self.assertEqual(m.maze.edge((0, 0), (1, 0)), 1)
        m.io.move_straight.assert_not_called()

    def test_slam_accepts_tof_confirmed_arrival_at_mapped_end_wall(self):
        m = self.mission()
        start, target, beyond = (0, 0), (1, 0), (2, 0)
        m.maze.set_edge(start, target, 1)
        m.maze.set_edge(target, beyond, -1)
        m.face = Mock()
        m.center_before_move = Mock()
        m.io.forward_one_cell = Mock(return_value=0.562 / 0.6)
        m.io.tof_arrival_confirmed = True
        m.slam.predict_move = Mock()
        m.localize_slam = Mock()
        self.assertTrue(m.drive_path([start, target], on_arrive=m.localize_slam))
        self.assertEqual(m.cell, target)
        self.assertTrue(m.io.forward_one_cell.call_args.kwargs['known_end_wall'])
        m.localize_slam.assert_called_once()

    def test_corrected_pose_discards_old_route(self):
        m = self.mission()
        m.face = Mock()
        m.io.forward_one_cell = Mock(return_value=1.0)
        m.slam.predict_move = Mock()
        def corrected():
            m.cell = (0, 0)
        m.localize_slam = Mock(side_effect=corrected)
        self.assertFalse(m.drive_path([(0, 0), (1, 0), (2, 0)]))
        m.io.forward_one_cell.assert_called_once()
        self.assertEqual(m.route, [])

    @patch('robo_app.maze.mission.time.sleep')
    def test_start_reset_reinitializes_particles(self, sleep):
        m = self.mission()
        m.verify_start = Mock(return_value=True)
        m.cell = (2, 0)
        m.slam.predict_move(1)
        self.assertTrue(m.reset_pose())
        self.assertTrue(all(p.cell == m.start_cell for p in m.slam.particles))

    def test_unknown_start_grid_is_rejected(self):
        maze = MazeMap(7, 7)
        maze.field = (4, 4)
        io = SimRobot((3, 3), 0, 7, 7)
        with self.assertRaises(ValueError):
            MazeMission(io, FrameGrabber(io), maze, (3, 3), 0, use_slam=True)


if __name__ == '__main__':
    unittest.main()
