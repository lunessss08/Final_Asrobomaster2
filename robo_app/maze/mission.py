"""maze: mission."""

from .camera import FrameGrabber
from .centering import center_shift_mm
from .errors import Aborted, HardwareFault
from .mapping import MazeMap
from .shot_calibration import load_shot_calibration, shot_offset
from .grid_slam import GridSLAM
from .settings import (
    AIM_ACQUIRE_DEG,
    AIM_PITCH_MARGIN_DEG,
    AIM_TIMEOUT_S,
    AIM_TRACK_JUMP_DEG,
    AIM_TRACK_AREA_RATIO,
    CAMERA_FORWARD_M,
    CAMERA_HEIGHT_M,
    CELL_M,
    Cell,
    DIRS,
    DIR_NAMES,
    DRY_HOLD_S,
    GIMBAL_SETTLE_S,
    GIMBAL_SCAN_SPEED,
    LOOK_LOG_ROOT,
    MAP_FILE,
    MAX_AIM_BACKOFF_M,
    MIN_TRAVEL_FRACTION,
    PITCH_MIN_DEG,
    ROUND1_BUDGET_S,
    SHOTS_PER_TARGET,
    TARGET_CELL_RANGE,
    TARGET_DEPTH_DRIFT_M,
    TARGET_HEIGHT_M,
    TARGET_MAX_LATERAL_M,
    TARGET_MIN_HITS,
    TARGET_NEAR_RANGE,
    TARGET_SCAN_FRAMES,
    TARGET_SCAN_PITCH,
    TARGET_SCAN_PITCHES,
    TARGET_WALL_GAP_M,
    TARGET_WALL_TOL_MM,
    TARGET_TRACK_M,
    CELL_ARRIVAL_TOL_M,
    TURN_FINE_DEG,
    TURN_TOL_DEG,
    WALL_CENTER_RANGE_MM,
    WALL_TOF_MM,
    YAW_TARGETS,
)
from .vision import normalize_angle, plate_shape
from collections import Counter
from collections import deque
from pathlib import Path
from robo_app import aim as gs
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import cv2
import json
import math
import numpy as np
import os
import threading
import time
import traceback

# ==========================================
# 5. ตัวคุมภารกิจ (ทำงานใน thread แยก ส่วน GUI แค่อ่านสถานะไปแสดง)
# ==========================================
class MazeMission:
    def __init__(self, io, grabber: FrameGrabber, maze: MazeMap, start: Cell, heading: int, use_slam: bool = False):
        self.io, self.grabber, self.maze = io, grabber, maze
        self.start_cell = start
        self.start_heading = heading
        self.cell, self.heading = start, heading
        if use_slam and maze.field is not None:
            raise ValueError("Grid SLAM requires a known start cell in the actual arena")
        self.slam = GridSLAM(maze, start, heading) if use_slam else None
        self._slam_calibrated = False
        self._tof_offset_known = False
        self.yaw_offset = io.yaw - YAW_TARGETS[heading]  # มุม IMU ตอนเริ่ม = ทิศเริ่มต้น
        self.color_ranges, zones, _, _ = gs.load_config(gs.CONFIG_PATH)
        self.blind_zones = gs.BlindZones(gs.AUTO_BLIND_ZONE, zones)

        self.stop_event = threading.Event()
        if hasattr(io, "set_stop_event"):
            io.set_stop_event(self.stop_event)
        self.busy = False
        self.phase = "IDLE"
        self.status = "Ready"
        self.last_fault: Optional[str] = None
        self.logs: deque = deque(maxlen=9)
        self.look_dir: Optional[int] = None       # ทิศที่หัวกำลังมอง (วาดบนแผนที่)
        self.overlay: List[gs.Target] = []        # เป้าที่เห็นในภาพล่าสุด (วาดบนภาพกล้อง)
        self.overlay_w = gs.PROCESS_W
        self.route: List[Cell] = []
        self.results: Dict[Cell, str] = {}
        self.plate_results: Dict[Tuple[Cell, int], str] = {}
        self._last_wall_check = None
        self._last_track_hits = {}
        self.armed = False
        self.pose_uncertain = False                # หยุดกลางการเคลื่อนที่: ตำแหน่งในแผนที่ไม่ตรงของจริงแล้ว
        self.look_dir_path: Optional[Path] = None  # โฟลเดอร์เก็บภาพ/ค่าตอนก้มหาเป้า (สร้างตอนมองครั้งแรก)
        self.bumper_hits: Dict[Tuple[Cell, Cell], int] = {}
        self.looked: Set[Tuple[Cell, int]] = set()   # (ช่อง, ทิศ) ที่กล้องมองหาเป้าไปแล้ว
        self._wall_center_learned = False
        self._wall_center_tried = False               # ลองเรียนระยะกลางช่องที่ช่อง start แล้ว (ลองครั้งเดียว)
        self._explore_fire = False
        self._explore_colors = frozenset()
        self._explore_shapes = frozenset()
        self._explore_attempted = set()
        self.finish_requested = False                 # กด f ระหว่างสำรวจ = จบรอบ 1 ที่ช่องถัดไปที่ถึง
        self.pid_yaw = gs.PIDController(**gs.YAW_PID)
        self.pid_pitch = gs.PIDController(**gs.PITCH_PID)
        self.fire_ctrl = gs.FireController()
        self.shot_calibration = load_shot_calibration()

    # ---------- งานเบื้องหลัง ----------
    def log(self, msg: str, gui: bool = True):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if gui:
            self.logs.append(msg)

    def check(self):
        if self.stop_event.is_set():
            raise Aborted()

    def start_job(self, name: str, fn, *args) -> bool:
        if self.busy:
            return False
        self.stop_event.clear()
        self.last_fault = None
        self.busy, self.phase = True, name

        def runner():
            try:
                fn(*args)
            except Aborted:
                self.log("STOPPED by user")
            except HardwareFault as e:
                self.last_fault = str(e)
                self.log(f"FAULT: {e}")
                self.record_fault(name, e)
            except Exception as e:  # แสดงบนจอด้วย ไม่ให้ thread ตายเงียบ
                self.last_fault = str(e)
                traceback.print_exc()
                self.log(f"ERROR: {e}")
                self.record_fault(name, e)
            finally:
                self.armed = self.fire_ctrl.armed = False
                self.io.stop_all()
                self.io.set_status_led(False, False)
                self.look_dir, self.overlay, self.route = None, [], []
                self.busy, self.phase = False, "IDLE"
                self.status = self.last_fault or ("Position unconfirmed" if self.pose_uncertain else "Ready")

        threading.Thread(target=runner, name=name, daemon=True).start()
        return True

    def record_fault(self, job: str, error: Exception):
        """Keep the fault and last sensor values for diagnosis after the GUI closes."""
        event = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "job": job,
                 "type": type(error).__name__, "message": str(error),
                 "cell": list(self.cell), "heading": self.heading,
                 "pose_uncertain": self.pose_uncertain,
                 "tof_mm": getattr(self.io, "tof_mm", None),
                 "odometry_m": getattr(self.io, "pos", None)}
        try:
            LOOK_LOG_ROOT.mkdir(parents=True, exist_ok=True)
            with (LOOK_LOG_ROOT / "faults.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.log(f"Could not save fault log: {exc}")

    def stop(self):
        self.stop_event.set()
        self.armed = self.fire_ctrl.armed = False
        self.io.stop_all()

    # ---------- การเคลื่อนที่ ----------
    def imu_heading(self) -> int:
        """ ทิศ (0-3) ที่ IMU บอกว่าตัวรถหันอยู่จริง ปัดเข้าทิศที่ใกล้สุด """
        return int(round(normalize_angle(self.io.yaw - self.yaw_offset) / 90.0)) % 4

    def move_in_cell(self, distance: float):
        """Keep the pose uncertain until a short move is confirmed successful."""
        self.check()
        was_uncertain = self.pose_uncertain
        self.pose_uncertain = True
        self.io.move_straight(distance)
        self.check()
        self.pose_uncertain = was_uncertain

    def face(self, d: int):
        """ หมุนตัวรถไปทิศ d แล้วหันหัวตรงตัวรถ (ToF ด้านหน้าเป็นกันชนตอนเดิน)
        เช็คกับ IMU ทุกครั้ง: คำสั่งเลี้ยวหลุดหรือหมุนไม่ถึง = ลองใหม่ 1 ครั้ง ยังไม่ได้ก็หยุด (แผนที่จะได้ไม่หมุนเพี้ยน) """
        err = 0.0
        for _ in range(2):
            steps = (d - self.imu_heading()) % 4
            if steps:
                self.io.rotate(steps)
            else:
                err = normalize_angle(self.io.yaw - (self.yaw_offset + YAW_TARGETS[d]))
                if abs(err) > TURN_TOL_DEG:
                    self.io.turn_by(-err)
            err = normalize_angle(self.io.yaw - (self.yaw_offset + YAW_TARGETS[d]))
            if abs(err) <= TURN_TOL_DEG:
                if abs(err) > TURN_FINE_DEG:  # เลี้ยวถึงแต่ยังเบี้ยว: แก้ก่อนเดิน ไม่งั้นรถไถลข้างระหว่างที่ IMU ค่อยๆ ดึงกลับ
                    self.io.turn_by(-err)
                self.heading = d
                self.io.gimbal_to(0, 0)
                return
            self.log(f"Turn to {DIR_NAMES[d]} missed: IMU {err:+.0f} deg off, retry")
        self.heading = self.imu_heading()
        raise HardwareFault(f"cannot turn to {DIR_NAMES[d]}: IMU still {err:+.0f} deg off")

    def center_before_move(self, d: int):
        """Measure mapped side walls while stationary, then correct at most 60 mm."""
        left_wall = self.maze.edge(self.cell, MazeMap.step(self.cell, (d + 3) % 4)) == -1
        right_wall = self.maze.edge(self.cell, MazeMap.step(self.cell, (d + 1) % 4)) == -1
        if not (left_wall or right_wall):
            return
        # One wall alone needs a measured ToF mounting offset. With two walls,
        # their distance difference gives a correction even without calibration.
        offset = self.slam.wall_offset_m * 1000 if self.slam is not None and self._tof_offset_known else None
        if not (left_wall and right_wall) and offset is None:
            return
        yaw_before = self.io.yaw
        pos_before = getattr(self.io, "pos", None)
        readings = {}
        try:
            for side, wall, angle in (("left", left_wall, -90), ("right", right_wall, 90)):
                if not wall:
                    continue
                self.check()
                self.io.gimbal_to(0, angle)
                time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
                distance = self.io.read_tof_mm()
                if distance is None:
                    distance = self.io.read_tof_mm()
                readings[side] = distance
        finally:
            self.io.gimbal_to(0, 0)
        drift = normalize_angle(self.io.yaw - yaw_before)
        pos_after = getattr(self.io, "pos", None)
        translation = (math.hypot(pos_after[0] - pos_before[0], pos_after[1] - pos_before[1])
                       if pos_before is not None and pos_after is not None else 0.0)
        if abs(drift) > 8 or translation > 0.04:
            self.pose_uncertain = True
            raise HardwareFault("body moved during side-wall centering scan")
        shift_mm = center_shift_mm(readings.get("left"), readings.get("right"), offset)
        if shift_mm:
            self.check()
            was_uncertain = self.pose_uncertain
            self.pose_uncertain = True
            self.io.move_sideways(shift_mm / 1000)
            self.check()
            self.pose_uncertain = was_uncertain
            self.log(f"Centered {shift_mm:+.0f} mm left (ToF L {readings.get('left')}, R {readings.get('right')})")
        # The gimbal scan must not leave the chassis aimed off the cell axis.
        err = normalize_angle(self.io.yaw - (self.yaw_offset + YAW_TARGETS[d]))
        if abs(err) > TURN_FINE_DEG:
            self.io.turn_by(-err)
            err = normalize_angle(self.io.yaw - (self.yaw_offset + YAW_TARGETS[d]))
        if abs(err) > TURN_FINE_DEG:
            self.pose_uncertain = True
            raise HardwareFault(f"body heading {err:+.1f} deg off after centering")

    def drive_path(self, path: List[Cell], on_arrive=None) -> bool:
        """ เดินตาม path ทีละช่อง คืน False เมื่อต้องวางเส้นทางใหม่
        กันชนเบรกกลางทาง: ถอยกลับกลางช่องแล้ววัด ToF นิ่งๆ ซ้ำ ยืนยันแล้ว (หรือเบรกช่องเดิมซ้ำ) ถึงบันทึกเป็นกำแพง
        """
        self.route = list(path)
        for nxt in path[1:]:
            self.check()
            if self.finish_requested:
                return False
            d = DIRS.index((nxt[0] - self.cell[0], nxt[1] - self.cell[1]))
            try:
                self.face(d)
                self.center_before_move(d)
                self.check()
            except (Aborted, HardwareFault):
                self.pose_uncertain = True
                raise
            self.status = f"Drive {self.cell} -> {nxt}  {getattr(self.io, 'side_status', '')}"
            try:
                end_wall = (self.slam is not None
                            and self.maze.edge(nxt, MazeMap.step(nxt, d)) == -1)
                fraction = self.io.forward_one_cell(self.yaw_offset + YAW_TARGETS[d], self.check,
                                                    known_end_wall=end_wall)
            except (Aborted, HardwareFault):
                self.pose_uncertain = True  # หยุดกลางทาง: รถไม่อยู่กลางช่องไหนแล้ว
                raise
            if fraction > 1 + CELL_ARRIVAL_TOL_M / CELL_M:
                self.pose_uncertain = True
                raise HardwareFault("drive overshot the next cell centre")
            if getattr(self.io, 'tof_arrival_confirmed', False):
                self.log(f"ToF confirms {self.cell}->{nxt} centre despite odometry "
                         f"{fraction * CELL_M:.3f} m; checking with SLAM")
            if fraction < MIN_TRAVEL_FRACTION and not getattr(self.io, 'tof_arrival_confirmed', False):
                if self.slam is not None:
                    self.pose_uncertain = True
                    self.io.stop_all()
                    raise HardwareFault("Stopped between cells: no wall recorded; place robot at START")
                if fraction > 0.05:
                    self.move_in_cell(-fraction * CELL_M)  # กลับไปกลางช่องเดิม
                edge = MazeMap._key(self.cell, nxt)
                self.bumper_hits[edge] = self.bumper_hits.get(edge, 0) + 1
                tof = self.io.read_tof_mm()  # หัวยังหันทิศ d อยู่
                if (tof is not None and tof < WALL_TOF_MM) or self.bumper_hits[edge] >= 2:
                    self.log(f"Bumper stop {self.cell}->{nxt}: wall (ToF {tof} mm), replan")
                    self.maze.set_edge(self.cell, nxt, -1)
                    if self.slam is not None:
                        self.slam.record_wall(d)
                    self.save_map(quiet=True)
                else:
                    self.log(f"Bumper stop {self.cell}->{nxt} but ToF now "
                             f"{'no reading' if tof is None else f'{tof:.0f} mm'}: false alarm, retry")
                return False
            if self.slam is not None:
                self.slam.predict_move(d)
            self.cell = nxt
            self.route = self.route[1:]
            if on_arrive is not None:
                on_arrive()
            elif self.slam is not None:
                self.localize_slam()
                self.save_map(quiet=True)
            if self.cell != nxt:
                self.route = []
                return False  # The rest of this path belongs to the old pose.
        return True

    def go_to(self, goal: Cell) -> bool:
        for _ in range(6):  # วางเส้นทางใหม่ได้หลายรอบถ้าเจอกำแพงไม่คาดคิด
            self.check()
            if self.cell == goal:
                return True
            path = self.maze.path_to(self.cell, self.heading, goal)
            if path is None:
                self.log(f"No path to {goal}")
                return False
            if self.drive_path(path):
                return True
        return self.cell == goal

    # ---------- รอบ 1: สำรวจ ----------
    def explore(self, fire: bool = False, colors=None, shapes=None):
        if self.pose_uncertain:
            self.log("Position unknown: put robot at START, press h, then e")
            return
        if fire and not colors and not shapes:
            self.log("Choose a COLOR or SHAPE before EXPLORE + FIRE")
            return
        self._explore_fire = fire
        self._explore_colors = frozenset(colors or ())
        self._explore_shapes = frozenset(shapes or ())
        self._explore_attempted = set()
        if fire:
            # A saved SKIPPED plate needs a fresh range estimate on resume.
            for cell, plates in self.maze.target_plates.items():
                for plate in plates:
                    if (not plate.get('shot', False) and plate['wall'] is not None
                            and plate.get('near_seen') is not None
                            and self.matches_explore_filter(plate)):
                        self.looked.discard((cell, plate['wall']))
        self.armed = self.fire_ctrl.armed = False
        self.log("ROUND 1: " + ("EXPLORE + FIRE" if fire else "MAP ONLY"))
        if fire:
            self.log("Fire filter: " + (", ".join(sorted(self._explore_colors)) or "any color")
                     + " AND " + (", ".join(sorted(self._explore_shapes)) or "any shape"))
        self.io.set_status_led(False, True)
        self.maze.explored = False
        self.finish_requested = False
        t0 = time.monotonic()
        budget_warned = False
        try:
            self.scan_cell()
            while True:
                self.check()
                if self.finish_requested:
                    self.log("Round 1 finished early by the operator (f)")
                    break
                if not budget_warned and time.monotonic() - t0 > ROUND1_BUDGET_S:
                    self.log(f"Round 1 passed {ROUND1_BUDGET_S} s: continuing to visit reachable cells")
                    budget_warned = True
                preferred = ({c for c, plates in self.maze.target_plates.items()
                              if any(p['wall_verified'] and not p.get('shot', False)
                                     and self.matches_explore_filter(p)
                                     and (c, i) not in self._explore_attempted
                                     for i, p in enumerate(plates))} if fire else None)
                path = self.maze.path_to_unvisited(self.cell, self.heading, preferred)
                if path is None:
                    break
                # Scan and replan after each cell so newly discovered walls,
                # targets, and SLAM corrections affect the next decision.
                self.drive_path(path[:2], on_arrive=self.scan_cell)
            self.finish_requested = False
            self.maze.explored = True
            self.save_map()
            shots = sum(p.get('shot', False) for plates in self.maze.target_plates.values() for p in plates)
            self.log(f"ROUND 1 done in {time.monotonic() - t0:.0f} s: {len(self.maze.visited)} cells, "
                     f"{self.maze.plate_count} plates, {shots} shot")
            if not self.maze.targets:
                self.log("No target found: see the Look lines and the photos in logs/looks_*")
        finally:
            self._explore_fire = False
            self.armed = self.fire_ctrl.armed = False
            self.io.stop_all()
            self.io.set_status_led(False, False)

    @staticmethod
    def matches_target_filter(plate, colors=None, shapes=None) -> bool:
        return ((not colors or plate['color'] in colors)
                and (not shapes or plate['shape'] in shapes))

    def matches_explore_filter(self, plate) -> bool:
        return (bool(self._explore_colors or self._explore_shapes)
                and self.matches_target_filter(plate, self._explore_colors, self._explore_shapes))

    def shoot_visible_during_explore(self, targets: Optional[Set[Cell]] = None):
        """Lock and fire only at matching plates on this cell's walls."""
        original_heading = self.heading
        for target in sorted(self.maze.target_plates):
            if target != self.cell or (targets is not None and target not in targets):
                continue
            for index, plate in enumerate(self.maze.plates_at(target)):
                self.check()
                if self.finish_requested:
                    return
                key = (target, index)
                if (key in self._explore_attempted or plate.get('shot', False)
                        or not plate['wall_verified'] or not self.matches_explore_filter(plate)):
                    continue
                spot = next((s for s in self.maze.shooting_spots(target, index) if s[0] == self.cell), None)
                # During scan_cell this cell is marked visited after all
                # directions finish; near_seen still establishes the current spot.
                wall = plate['wall']
                if (spot is None and target == self.cell and wall is not None
                        and plate.get('near_seen') is not None
                        and self.maze.edge(self.cell, MazeMap.step(self.cell, wall)) == -1):
                    spot = (self.cell, wall)
                if spot is None:
                    continue
                self._explore_attempted.add(key)
                self.log(f"Explore: engage {self.maze.target_label(target, index)} {target} #{index + 1}")
                try:
                    try:
                        self.face(spot[1])
                    except (Aborted, HardwareFault):
                        self.pose_uncertain = True
                        raise
                    self.check()
                    self.armed = self.fire_ctrl.armed = True
                    self.io.set_status_led(True, True)
                    result = self.aim_and_fire(target, index, finish_on_request=True)
                    self.plate_results[key] = result
                    if result == 'SHOT':
                        plate['shot'] = True
                    states = [p.get('shot', False) for p in self.maze.plates_at(target)]
                    self.results[target] = result if len(states) == 1 else f"{sum(states)}/{len(states)}"
                    self.save_map(quiet=True)
                    self.log(f"Explore {target} #{index + 1}: {result}")
                finally:
                    self.armed = self.fire_ctrl.armed = False
                    self.io.set_status_led(False, True)
        if self.heading != original_heading:
            try:
                self.check()
                self.face(original_heading)
            except (Aborted, HardwareFault):
                self.pose_uncertain = True
                raise

    def finish_round1(self):
        """ จบรอบ 1 ด้วยแผนที่เท่าที่มี (หุ่นไม่ขยับ) ให้ยิงรอบ 2 ได้ เช่นหลังกด STOP หรือเวลาใกล้หมด """
        self.maze.explored = True
        self.save_map()
        self.log(f"Round 1 marked done: {len(self.maze.visited)} cells, {self.maze.plate_count} plates")

    def worth_looking(self, c: Cell, d: int) -> bool:
        """ มองหาเป้าจาก c ไปทิศ d ไหม: ยังไม่เคยมอง, ช่องถัดไปยังไม่มีเป้า และผนังไกลของช่องนั้นยังไม่รู้ว่าเปิด
        (เป้าต้องพิงผนัง ถ้าฝั่งไกลเป็นทางเปิดไม่มีทางมีเป้า -> ประหยัดเวลาหันหัว) """
        nb = MazeMap.step(c, d)
        return ((c, d) not in self.looked
                and self.maze.edge(nb, MazeMap.step(nb, d)) != 1)

    def localize_slam(self):
        """Use level ToF rays before assigning camera observations to a cell."""
        previous = self.cell
        rel_of = {0: 0, 1: 90, 2: 180, 3: -90}
        try:
            for attempt in range(2):
                scan = {}
                yaw_before = self.io.yaw
                pos_before = getattr(self.io, "pos", None)
                for d in sorted(range(4), key=lambda d: rel_of[(d - self.heading) % 4]):
                    self.check()
                    self.look_dir = d
                    self.status = f"SLAM scan {self.cell} {DIR_NAMES[d]}"
                    self.io.gimbal_to(0, rel_of[(d - self.heading) % 4])
                    time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
                    value = self.io.read_tof_mm()
                    scan[d] = value if value is not None and math.isfinite(value) and value >= 20 else None
                self.io.gimbal_to(0, 0)
                drift = normalize_angle(self.io.yaw - yaw_before)
                pos_after = getattr(self.io, "pos", None)
                shift = (math.hypot(pos_after[0] - pos_before[0], pos_after[1] - pos_before[1])
                         if pos_before is not None and pos_after is not None else 0.0)
                if abs(drift) > 8 or shift > 0.04:
                    raise HardwareFault("Body moved during SLAM scan")
                if sum(value is not None for value in scan.values()) < 3:
                    if attempt == 0:
                        self.log("SLAM: missing ToF readings, stationary recheck")
                        continue
                    raise HardwareFault("SLAM: fewer than three valid ToF directions")
                if not self._slam_calibrated:
                    self._tof_offset_known = any(
                        value is not None and 80 <= value <= 450
                        and not self.maze.inside(MazeMap.step(self.cell, d))
                        for d, value in scan.items())
                    self.slam.calibrate_from_boundary(scan)
                    self._slam_calibrated = True
                    walls = {rel_of[(d - self.heading) % 4]: value for d, value in scan.items()
                             if value is not None and value < WALL_TOF_MM}
                    if self.cell == self.start_cell and walls:
                        self.learn_wall_center(walls)
                confidence = self.slam.observe(scan, self.heading)
                self.cell = self.slam.cell
                if confidence >= 0.70 and (not attempt or self.cell == previous):
                    # Keep the authoritative map object and its target/shot metadata.
                    self.maze.edges = self.slam.maze.edges.copy()
                    self.maze.visited = self.slam.maze.visited.copy()
                    self.log(f"SLAM {self.cell}: {confidence:.0%}"
                             + (f" corrected from {previous}" if self.cell != previous else ""))
                    self.face(self.heading)  # Correct scan drift before driving or looking.
                    return
                if attempt == 0 and self.cell == previous:
                    self.log("SLAM confidence low: stationary recheck")
                    continue
                raise HardwareFault(f"SLAM pose ambiguous ({confidence:.0%}); reset at START")
            raise HardwareFault("SLAM could not confirm the pose")
        except (Aborted, HardwareFault):
            self.pose_uncertain = True
            self.io.stop_all()
            raise
        finally:
            self.look_dir = None

    def scan_cell(self):
        """Map unknown edges with level ToF; look for plates only on this cell's walls."""
        if self.slam is not None:
            self.localize_slam()
        c = self.cell
        rel_of = {0: 0, 1: 90, 2: 180, 3: -90}
        rear = (self.heading + 2) % 4
        todo = []
        for d in range(4):
            nb = MazeMap.step(c, d)
            if self.maze.edge(c, nb) == 0 and self.maze.outside_field(nb):
                self.maze.set_edge(c, nb, -1)  # นอกสนามจริง (ตารางเผื่อ) = กำแพงรอบสนาม ไม่ต้องวัด
            status = self.maze.edge(c, nb)
            if (c, d) in self.looked and status != 0:
                continue
            if d == rear and status != 0:
                continue  # No target sweep behind the chassis.
            if status == 1:
                continue  # No wall in this cell: inspect the adjacent cell after entering it.
            if self.slam is not None and status == 0:
                continue  # A missing SLAM ray is unknown, not evidence of an opening.
            todo.append(d)
        todo.sort(key=lambda d: rel_of[(d - self.heading) % 4])
        wall_mm: Dict[int, float] = {}   # มุมหัวเทียบตัวรถ -> ระยะแนวราบถึงกำแพง
        for d in todo:
            self.check()
            rel, nb = rel_of[(d - self.heading) % 4], MazeMap.step(c, d)
            self.look_dir = d
            yaw_before = self.io.yaw
            pos_before = getattr(self.io, "pos", None)
            down_pitch = TARGET_SCAN_PITCHES[0]
            self.status = f"Scan {c} {DIR_NAMES[d]} level"
            self.io.gimbal_to(0, rel)
            if self.maze.edge(c, nb) == 0:
                time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
                tof = self.io.read_tof_mm()
                if tof is None:
                    tof = self.io.read_tof_mm()  # อ่านซ้ำอีกรอบ
                if tof is None:
                    self.log(f"ToF no reading at {c} {DIR_NAMES[d]}: treat as wall")
                    self.maze.set_edge(c, nb, -1)
                elif tof < WALL_TOF_MM:
                    self.maze.set_edge(c, nb, -1)
                    wall_mm[rel] = tof
                else:
                    self.maze.set_edge(c, nb, 1)
            if self.maze.edge(c, nb) != -1:
                continue
            if d == rear:
                continue  # Unknown rear edges still get level ToF mapping without a plate sweep.
            self.status = f"Scan {c} {DIR_NAMES[d]} down"
            # Turn at level first, then descend in one slow, uninterrupted
            # movement. Collect fresh images on the upward inspection poses.
            self.io.gimbal_to(down_pitch, rel, speed=GIMBAL_SCAN_SPEED)
            time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
            near_hits = []
            for pitch in TARGET_SCAN_PITCHES:
                self.check()
                self.look_dir = d
                self.status = f"Scan {c} {DIR_NAMES[d]} pitch {pitch:+.0f}°"
                if pitch != down_pitch:
                    self.io.gimbal_to(pitch, rel, speed=GIMBAL_SCAN_SPEED)
                    time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
                pos_after = getattr(self.io, "pos", None)
                translation = (math.hypot(pos_after[0] - pos_before[0], pos_after[1] - pos_before[1])
                               if pos_before is not None and pos_after is not None else 0.0)
                if abs(normalize_angle(self.io.yaw - yaw_before)) > 8 or translation > 0.04:
                    self.pose_uncertain = True
                    raise HardwareFault("body moved during gimbal target sweep")
                seq = self.grabber.seq
                near_hits.extend((pitch, color, shape, dist, lateral)
                                 for color, shape, dist, lateral
                                 in self.look_for_target(d, seq, pitch, near=True))
            if near_hits:
                # A nearby plate may be cropped at one pitch. Merge sightings
                # across the sweep and choose the most common shape before
                # storing it; do not let a late partial view overwrite it.
                tracks = []
                for pitch, color, shape, dist, lateral in near_hits:
                    shooting_geo = (dist + CELL_M, lateral)
                    track = next((t for t in tracks if t['color'] == color and
                                  abs(shooting_geo[0] - np.median(t['geos'], axis=0)[0]) < TARGET_DEPTH_DRIFT_M and
                                  abs(lateral - np.median(t['geos'], axis=0)[1]) < TARGET_TRACK_M), None)
                    if track is None:
                        track = {'color': color, 'geos': [], 'by_pitch': {}, 'shapes': Counter()}
                        tracks.append(track)
                    track['geos'].append(shooting_geo)
                    track['by_pitch'].setdefault(pitch, []).append(shooting_geo)
                    track['shapes'][shape] += 1
                for track in tracks:
                    shape = track['shapes'].most_common(1)[0][0]
                    # A shallow view can crop the plate and bias its range high.
                    # Use the deepest pitch that actually saw this plate.
                    distance, lateral = np.median(track['by_pitch'][min(track['by_pitch'])], axis=0)
                    if self.maze.add_target(c, track['color'], shape, d,
                                            (float(distance), float(lateral)), wall_verified=True,
                                            near_seen=(float(distance - CELL_M), float(lateral))):
                        self.log(f"Target {track['color'].upper()} {shape} at {c} "
                                 f"({DIR_NAMES[d]} wall, {distance - CELL_M:.2f} m from this cell, "
                                 f"lateral {lateral:+.2f} m)")
                self.save_map(quiet=True)
                if self._explore_fire and not self.finish_requested:
                    self.look_dir = None
                    self.shoot_visible_during_explore({c})
            self.looked.add((c, d))  # นับว่ามองแล้วเมื่อมองเสร็จจริง (หยุด/กล้องค้างกลางทาง = มองใหม่ครั้งหน้า)
        if self.slam is None and c == self.start_cell and not self._wall_center_learned and not self._wall_center_tried:
            self._wall_center_tried = True
            if not wall_mm:  # ขอบช่อง start รู้อยู่แล้ว (เช่นขอบสนาม): วัดกำแพงที่รู้หนึ่งด้านไว้เรียนระยะกลางช่อง
                walls = [d for d in range(4) if self.maze.edge(c, MazeMap.step(c, d)) == -1]
                if walls:
                    d = min(walls, key=lambda d: abs(rel_of[(d - self.heading) % 4]))
                    self.io.gimbal_to(TARGET_SCAN_PITCH, rel_of[(d - self.heading) % 4])
                    time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
                    tof = self.io.read_tof_mm()
                    if tof is not None and tof < WALL_TOF_MM:
                        wall_mm[rel_of[(d - self.heading) % 4]] = tof * math.cos(math.radians(TARGET_SCAN_PITCH))
                    todo = todo or [d]
            if wall_mm:
                self.learn_wall_center(wall_mm)
            else:
                self.log(f"No wall at the start cell to learn from: keep {self.io.wall_center_mm:.0f} mm")
        if todo:
            self.look_dir = None
            self.io.gimbal_to(0, 0)
        self.maze.visited.add(c)   # ครบทุกทิศแล้วค่อยนับว่าไปแล้ว (หยุดกลางการสแกน = กลับมาสแกนต่อ)
        self.save_map(quiet=True)  # เซฟทุกช่อง: ล่ม/กดหยุดกลางรอบ 1 ก็สำรวจต่อจากไฟล์ได้
        if self._explore_fire and not self.finish_requested:
            self.shoot_visible_during_explore()

    def learn_wall_center(self, wall_mm: Dict[int, float]):
        """ ช่อง start: หุ่นถูกวางกลางช่องพอดี ระยะ ToF ถึงกำแพง = ระยะที่ใช้หยุดกลางช่องตอนเดิน
        ใช้กำแพงด้านหน้าถ้ามี (ตอนเดิน ToF หันหน้า) ไม่งั้นมัธยฐานของกำแพงรอบตัว """
        value = wall_mm[0] if 0 in wall_mm else float(np.median(list(wall_mm.values())))
        low, high = WALL_CENTER_RANGE_MM
        if low <= value <= high:
            self.io.wall_center_mm = value
            self._wall_center_learned = True
            self.log(f"Wall distance at cell centre: {value:.0f} mm (stops each drive at this ToF reading)")
        else:
            self.log(f"Start-cell wall distance {value:.0f} mm looks wrong: keep {self.io.wall_center_mm:.0f} mm")

    def confirm_target_wall(self, d: int, near: bool = False) -> bool:
        """Confirm the mapped wall with a fresh, level ToF reading."""
        relative = {0: 0, 1: 90, 2: 180, 3: -90}[(d - self.heading) % 4]
        self.check()
        self.io.gimbal_to(0, relative)
        time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
        reading = self.io.read_tof_mm()
        if reading is None:
            reading = self.io.read_tof_mm()
        if reading is None:
            raise HardwareFault("no ToF reading while confirming a target wall")
        expected = self.io.wall_center_mm + (0 if near else CELL_M * 1000)
        confirmed = (100 <= reading <= 450 and
                     self.maze.edge(self.cell, MazeMap.step(self.cell, d)) == -1) if near else (
                     abs(reading - expected) <= TARGET_WALL_TOL_MM)
        self._last_wall_check = {'tof_mm': reading, 'expected_mm': expected,
                                 'confirmed': confirmed, 'near': near}
        if not self._last_wall_check['confirmed']:
            self.log(f"Reject targets toward {DIR_NAMES[d]}: far wall not confirmed (ToF {reading}, expected {expected:.0f} mm)")
            return False
        if not near:
            neighbour = MazeMap.step(self.cell, d)
            self.maze.set_edge(neighbour, MazeMap.step(neighbour, d), -1)
        return True

    def look_for_target(self, d: int, seq: int, pitch: float = TARGET_SCAN_PITCH,
                        near: bool = False) -> List[Tuple[str, str, float, float]]:
        """Track plates across fresh frames at the commanded gimbal pitch."""
        votes: Counter = Counter()
        seen: Dict[str, dict] = {}
        tracks = []
        last = None
        frames = misses = 0
        self._last_wall_check = None
        while frames < TARGET_SCAN_FRAMES:
            self.check()
            frame, seq = self.grabber.wait_new(seq, check=self.check)
            if frame is None:
                misses += 1
                if misses >= 2:
                    raise HardwareFault("camera stream stalled: no new frame while looking for targets")
                continue
            misses = 0
            frames += 1
            fd = gs.prepare_frame(frame)
            found = gs.detect_all_colors(fd, self.color_ranges, self.blind_zones)
            self.overlay, self.overlay_w = found, fd.size[0]
            last = (frame, fd, found)
            used = set()
            for t in found:
                info = seen.setdefault(t.color, {"sightings": 0, "above_horizon": 0, "nearest": None})
                info['sightings'] += 1
                geo = self.target_ground(fd, t, pitch)
                if geo is None:
                    info['above_horizon'] += 1
                    continue
                if info['nearest'] is None or geo[0] < info['nearest'][0]:
                    info['nearest'] = (round(geo[0], 2), round(geo[1], 2))
                distance, lateral = geo
                target_range = TARGET_NEAR_RANGE if near else TARGET_CELL_RANGE
                if not (target_range[0] <= distance <= target_range[1]
                        and abs(lateral) <= TARGET_MAX_LATERAL_M):
                    continue
                closest, error = None, TARGET_TRACK_M
                for i, track in enumerate(tracks):
                    if i in used or track['color'] != t.color:
                        continue
                    x, y = np.median(track['geos'], axis=0)
                    delta = math.hypot(distance - x, lateral - y)
                    if delta < error:
                        closest, error = i, delta
                if closest is None:
                    closest = len(tracks)
                    tracks.append({'color': t.color, 'geos': [], 'shapes': Counter()})
                used.add(closest)
                track = tracks[closest]
                track['geos'].append(geo)
                track['shapes'][plate_shape(fd, t, self.color_ranges[t.color], self.blind_zones)] += 1
                votes[t.color] += 1
        self.overlay = []
        results = []
        for track in tracks:
            if len(track['geos']) < TARGET_MIN_HITS:
                continue
            known = Counter({k: v for k, v in track['shapes'].items() if k != '?'})
            distance, lateral = np.median(track['geos'], axis=0)
            results.append((track['color'], known.most_common(1)[0][0] if known else '?',
                            round(float(distance), 3), round(float(lateral), 3)))
        self._last_track_hits = {color: [len(t["geos"]) for t in tracks if t["color"] == color]
                                 for color in seen}
        if results:
            wall_ok = self.confirm_target_wall(d, near=True) if near else self.confirm_target_wall(d)
            if not wall_ok:
                results = []
        self.record_look(d, votes, seen, results or None, last, pitch, near)
        return results

    def record_look(self, d: int, votes: Counter, seen: Dict[str, dict], result: Optional[List[Tuple[str, str, float, float]]], last,
                    pitch: float = TARGET_SCAN_PITCH, near: bool = False):
        """ log ว่ากล้องเห็นอะไร และถ้าไม่นับเป็นเป้า ไม่นับเพราะอะไร + เก็บภาพไว้ดูทีหลัง """
        where = f"{self.cell}->{DIR_NAMES[d]}"
        if seen and result is None:
            reasons = []
            for color, info in seen.items():
                if votes[color]:
                    hits = self._last_track_hits.get(color, [])
                    counts = "/".join(str(h) for h in hits) or "0"
                    reason = "wall not confirmed" if self._last_wall_check and not self._last_wall_check['confirmed'] else f"each plate needs {TARGET_MIN_HITS}/{TARGET_SCAN_FRAMES} frames"
                    reasons.append(f"{color.upper()} plate hits [{counts}]: {reason}")
                elif info["nearest"] is not None:
                    dist, lat = info["nearest"]
                    target_range = TARGET_NEAR_RANGE if near else TARGET_CELL_RANGE
                    why = (f"lateral {lat:+.2f} m" if target_range[0] <= dist <= target_range[1]
                           else f"{dist:.2f} m, need {target_range[0]}-{target_range[1]}")
                    reasons.append(f"{color.upper()} {why}")
                else:
                    reasons.append(f"{color.upper()} above the horizon")
            # ขึ้นจอเฉพาะเคสที่เกือบนับ (เห็นในระยะแต่เฟรมไม่พอ) ที่เหลือ (เป้าไกล) อยู่ใน console + looks.jsonl
            self.log(f"Look {where}: " + "; ".join(reasons), gui=any(votes.values()))
        if self.io.sim:
            return
        try:
            if self.look_dir_path is None:
                self.look_dir_path = LOOK_LOG_ROOT / time.strftime("looks_%Y%m%d_%H%M%S")
                self.look_dir_path.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%H%M%S')}_c{self.cell[0]}{self.cell[1]}_{DIR_NAMES[d]}_{'near' if near else 'far'}_p{pitch:+.0f}"
            if last is not None:
                frame, fd, found = last
                cv2.imwrite(str(self.look_dir_path / f"{name}_raw.jpg"), frame)
                img = frame.copy()
                gs.draw_targets(img, found, img.shape[1] / fd.size[0])
                cv2.imwrite(str(self.look_dir_path / f"{name}.jpg"), img)
            with (self.look_dir_path / "looks.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "cell": list(self.cell), "dir": DIR_NAMES[d],
                                    "gimbal_pitch": pitch, "range_mode": "current" if near else "adjacent",
                                    "result": result, "votes": dict(votes),
                                    "seen": seen, "track_hits": self._last_track_hits, "wall_check": self._last_wall_check, "image": f"{name}.jpg",
                                    "raw_image": f"{name}_raw.jpg"}, allow_nan=False) + "\n")
        except (OSError, ValueError) as e:
            self.log(f"Cannot write look log: {e}")

    @staticmethod
    def target_ground(fd, t: gs.Target, pitch: float = TARGET_SCAN_PITCH) -> Optional[Tuple[float, float]]:
        """ (ระยะจากกลางหุ่น, เยื้องซ้าย-ขวา) ของป้าย คำนวณจากมุมก้ม (กล้องสูงกว่าเป้า)
        None = ป้ายอยู่ระดับกล้องหรือสูงกว่า (ไม่ใช่เป้าเตี้ยบนพื้น) """
        w, h = fd.size
        bearing, dy = gs.pixel_to_degrees(t.cx, t.cy, w, h)
        depression = -(pitch + dy)
        if depression <= 1.0:
            return None
        ground = (CAMERA_HEIGHT_M - TARGET_HEIGHT_M) / math.tan(math.radians(depression))
        return ground + CAMERA_FORWARD_M, ground * math.tan(math.radians(bearing))

    @staticmethod
    def adjacent_plate(fd, found: List[gs.Target]) -> Optional[gs.Target]:
        """ ป้ายที่อยู่ในช่องติดกัน (ระยะ/เยื้องอยู่ในเกณฑ์) ป้ายใหญ่สุดชนะ """
        best = None
        for t in found:
            geo = MazeMission.target_ground(fd, t)
            if geo is None:
                continue
            distance, lateral = geo
            if TARGET_CELL_RANGE[0] <= distance <= TARGET_CELL_RANGE[1] and abs(lateral) <= TARGET_MAX_LATERAL_M:
                if best is None or t.area > best.area:
                    best = t
        return best

    @staticmethod
    def adjacent_target(fd, found: List[gs.Target]) -> Optional[str]:
        """ สีของป้ายในช่องติดกัน (sensor_test.py ใช้ตัดสินแบบเดียวกับโปรแกรมหลัก) """
        plate = MazeMission.adjacent_plate(fd, found)
        return plate.color if plate is not None else None

    # ---------- รอบ 2: ยิง ----------
    def shoot_run(self, fire: bool, start_mode: str, end_mode: str, order_mode: str,
                  picks: List[Cell], goal: Optional[Cell], colors=None, shapes=None):
        if not self.maze.explored:
            self.log("Explore first (round 1)")
            return
        self.armed = self.fire_ctrl.armed = fire
        self.io.set_status_led(fire, False)
        self.results = {}
        self.plate_results = {}
        self.bumper_hits = {}   # เบรกหลอกจากรอบก่อนไม่นับรวม ไม่งั้นเบรกครั้งเดียวก็กลายเป็นกำแพง
        self.log(f"ROUND 2: {'FIRE' if fire else 'DRY RUN'} | {start_mode} | {order_mode} | end: {end_mode}")
        if start_mode == "PLACED AT START":
            if not self.reset_pose():
                return
        elif self.pose_uncertain:
            self.log("Position unknown: put robot at START and use START: PLACED AT START (or press h)")
            return
        elif start_mode == "DRIVE TO START" and self.cell != self.start_cell:
            self.log(f"Back to START {self.start_cell}")
            if not self.go_to(self.start_cell):
                return
        t0 = time.monotonic()
        order = self.plan_targets(order_mode, picks, colors, shapes)
        if not order:
            self.log("No targets on the map: round 1 found none (see Look lines / logs/looks_*)"
                     if not self.maze.targets else
                     "PICK ORDER: click targets on the map first" if order_mode == "PICK ORDER" and not picks else
                     "No reachable target to shoot")
        else:
            self.log("Order: " + " > ".join(f"{self.maze.target_label(t)}{t}" for t in order))
        for target in order:
            for index, plate in enumerate(self.maze.plates_at(target)):
                if not self.matches_target_filter(plate, colors, shapes):
                    continue
                self.check()
                key = (target, index)
                self.results[target] = self.plate_results[key] = "GOING"
                if plate.get("shot", False):
                    self.results[target] = self.plate_results[key] = "ALREADY SHOT"
                    self.log(f"Skip {target} #{index + 1}: already shot")
                    continue
                if not plate['wall_verified']:
                    self.results[target] = self.plate_results[key] = "UNVERIFIED"
                    self.log(f"Skip {target} plate {index + 1}: wall has not been sensor-verified")
                    continue
                spot = self.best_spot(target, index)
                if spot is None or not self.go_to(spot[0]):
                    self.results[target] = self.plate_results[key] = "SKIPPED"
                    continue
                self.face(spot[1])
                result = self.aim_and_fire(target, index)
                self.plate_results[key] = result
                if result == "SHOT":
                    plate["shot"] = True
                    self.save_map(quiet=True)
                states = [self.plate_results.get((target, i)) for i in range(len(self.maze.plates_at(target)))]
                self.results[target] = result if len(states) == 1 else f"{sum(v in ('SHOT', 'AIMED', 'ALREADY SHOT') for v in states)}/{len(states)}"
                self.log(f"{self.maze.target_label(target, index)} {target} #{index + 1}: {result}")
        if end_mode == "BACK TO START":
            self.go_to(self.start_cell)
        elif end_mode == "GO TO GOAL":
            if goal is None:
                self.log("No GOAL set: stay here (click an empty map cell to set GOAL)")
            else:
                self.go_to(goal)
        counts = Counter(self.plate_results.values())
        done = counts["SHOT"] if fire else counts["AIMED"]
        extra = ", ".join(f"{counts[k]} {k.lower()}" for k in ("SKIPPED", "UNREACHABLE", "UNVERIFIED", "ALREADY SHOT") if counts[k])
        self.log(f"ROUND 2 done in {time.monotonic() - t0:.0f} s: {'shot' if fire else 'aimed'} {done}/{len(self.plate_results)}"
                 + (f", {extra}" if extra else ""))
        self.save_map()

    def best_spot(self, target: Cell, plate_index: Optional[int] = None) -> Optional[Tuple[Cell, int]]:
        """ ช่องยิงที่เดินไปถึงใกล้สุดจากตำแหน่งปัจจุบัน """
        dist = self.maze.distances_from(self.cell)
        spots = [s for s in self.maze.shooting_spots(target, plate_index) if s[0] in dist]
        return min(spots, key=lambda s: dist[s[0]]) if spots else None

    def plan_targets(self, order_mode: str, picks: List[Cell], colors=None, shapes=None) -> List[Cell]:
        """ NEAREST   : จากตำแหน่งหุ่น ไปเป้าที่ช่องยิงใกล้สุดก่อน แล้วนับต่อจากช่องยิงนั้นเรื่อยๆ จนครบทุกเป้า
        PICK ORDER: ตามลำดับที่คลิกบนแผนที่ (อาจารย์สั่งสี+รูปทรง หรือชี้ทีละอัน) ไม่ยิงเป้าที่ไม่ได้คลิก
        เป้าที่ไม่มีช่องยิงที่ไปถึงได้ถูกข้ามพร้อมบอกเหตุผล ไม่ทำให้เป้าอื่นยิงไม่ได้ """
        wanted = [t for t in picks if t in self.maze.targets] if order_mode == "PICK ORDER" else sorted(self.maze.targets)
        wanted = [t for t in wanted if any(self.matches_target_filter(p, colors, shapes) for p in self.maze.plates_at(t))]
        cache: Dict[Cell, Dict[Cell, int]] = {}

        def reach(pos: Cell, t: Cell) -> Tuple[float, Cell]:
            if pos not in cache:
                cache[pos] = self.maze.distances_from(pos)
            spots = [spot for index, plate in enumerate(self.maze.plates_at(t))
                     if self.matches_target_filter(plate, colors, shapes)
                     for spot in self.maze.shooting_spots(t, index)]
            return min(((cache[pos].get(source, math.inf), source) for source, _ in spots),
                       default=(math.inf, pos))

        targets = []
        for t in wanted:
            if reach(self.cell, t)[0] < math.inf:
                targets.append(t)
            else:
                self.results[t] = "UNREACHABLE"
                for index, plate in enumerate(self.maze.plates_at(t)):
                    if self.matches_target_filter(plate, colors, shapes):
                        self.plate_results[(t, index)] = "UNREACHABLE"
                self.log(f"Skip {self.maze.target_label(t)} {t}: no reachable shooting cell")
        if order_mode == "PICK ORDER":
            return targets
        order, pos, pool = [], self.cell, list(targets)
        while pool:
            t = min(pool, key=lambda t: reach(pos, t)[0])
            order.append(t)
            pos = reach(pos, t)[1]
            pool.remove(t)
        return order

    def aim_candidates(self, fd, target: Cell, candidates, plate_index: Optional[int] = None):
        plate = self.maze.plates_at(target)[plate_index or 0] if self.maze.plates_at(target) else None
        shape = plate["shape"] if plate is not None else "?"
        if shape == "?":
            return candidates
        color = plate["color"]
        return [cand for cand in candidates
                if plate_shape(fd, cand, self.color_ranges[color], self.blind_zones) == shape]

    def aim_and_fire(self, target: Cell, plate_index: Optional[int] = None, finish_on_request: bool = False) -> str:
        """ เล็งละเอียดด้วยกล้อง + PID ของ gimbalshoot แล้วยิงตามเกณฑ์ล็อกเดียวกัน """
        if target != self.cell:
            self.log(f"Skip {target}: robot must be inside the target cell to fire")
            return "SKIPPED"
        plate = self.maze.plates_at(target)[plate_index or 0]
        if plate.get('shots_fired', 0) >= SHOTS_PER_TARGET:
            plate['shot'] = True
            self.save_map(quiet=True)
            return "SHOT"
        color = plate["color"]
        drop = CAMERA_HEIGHT_M - TARGET_HEIGHT_M
        wall = plate['wall']
        if wall is None or not self.confirm_target_wall(wall, near=True):
            self.log(f"Skip near target {target}: wall range no longer confirmed")
            return "SKIPPED"
        seen = plate.get('near_seen')
        if seen is None:
            # Older maps may only have the distance from the previous cell.
            previous = plate['seen']
            seen = (max(TARGET_NEAR_RANGE[0], previous[0] - CELL_M), previous[1]) if previous else (
                CELL_M / 2 - TARGET_WALL_GAP_M, 0.0)
        far = seen[0]
        ground = max(0.05, far - CAMERA_FORWARD_M)
        yaw0 = math.degrees(math.atan2(seen[1], ground))
        backoff = 0.0
        pose_before_backoff = self.pose_uncertain
        pitch_limit_ground = drop / math.tan(math.radians(-PITCH_MIN_DEG))
        if ground < pitch_limit_ground:
            # Only a plate below the gimbal's physical pitch range needs a
            # short retreat. Keep the robot centre in this cell and require a
            # mapped opening plus a fresh, clear rear ToF reading first.
            desired_ground = drop / math.tan(math.radians(-PITCH_MIN_DEG - AIM_PITCH_MARGIN_DEG))
            backoff = desired_ground - ground
            if backoff > MAX_AIM_BACKOFF_M:
                self.log(f"Skip near target {target}: needs {backoff:.3f} m aim backoff "
                         f"(limit {MAX_AIM_BACKOFF_M:.3f} m)")
                return "SKIPPED"
            rear_dir = (wall + 2) % 4
            rear = MazeMap.step(target, rear_dir)
            if not self.maze.can_drive(target, rear):
                self.log(f"Skip near target {target}: no confirmed open rear for aim backoff")
                return "SKIPPED"
            rear_rel = {0: 0, 1: 90, 2: 180, 3: -90}[(rear_dir - self.heading) % 4]
            self.io.gimbal_to(0, rear_rel)
            try:
                rear_tof = self.io.read_tof_mm()
            finally:
                self.io.gimbal_to(0, 0)
            if rear_tof is None or rear_tof < WALL_TOF_MM:
                self.log(f"Skip near target {target}: rear ToF not clear for "
                         f"{backoff:.3f} m aim backoff ({rear_tof} mm)")
                return "SKIPPED"
            self.move_in_cell(-backoff)
            self.pose_uncertain = True  # Remain uncertain until the return move succeeds.
            self.log(f"Aim backoff {backoff:.3f} m within cell {target}")
        aim_range = far + backoff
        aim_ground = ground + backoff
        yaw0 = math.degrees(math.atan2(seen[1], aim_ground))
        correction_x, correction_y = shot_offset(self.shot_calibration, aim_range)
        aim_x = gs.AIM_OFFSET_X_DEG + correction_x
        aim_y = gs.AIM_OFFSET_Y_DEG + correction_y
        acquire_radius = AIM_ACQUIRE_DEG
        peers = [p for p in self.maze.plates_at(target) if p is not plate
                 and p['color'] == color and p['wall'] == plate['wall'] and p['seen'] is not None]
        for peer in peers:
            peer_seen = (peer.get('near_seen') or
                         (max(TARGET_NEAR_RANGE[0], peer['seen'][0] - CELL_M), peer['seen'][1]))
            angle = math.degrees(math.atan2(peer_seen[1], max(0.05, peer_seen[0] - CAMERA_FORWARD_M) + backoff))
            acquire_radius = min(acquire_radius, max(0.5, abs(angle - yaw0) / 2))
        self.io.gimbal_to(max(PITCH_MIN_DEG, -math.degrees(math.atan2(drop, aim_ground))), yaw0)
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.fire_ctrl.reset_lock()
        result, shots, hold_since, last_deg, misses = "SKIPPED", plate.get('shots_fired', 0), None, None, 0
        last_area = None
        seq, t0 = self.grabber.seq, time.monotonic()
        try:
            while time.monotonic() - t0 < AIM_TIMEOUT_S:
                self.check()
                if finish_on_request and self.finish_requested:
                    break
                frame, seq = self.grabber.wait_new(seq, check=self.check)
                if frame is None:
                    self.io.gimbal_speed(0, 0)  # ไม่มีภาพใหม่ = ห้ามหมุนหัวต่อด้วยคำสั่งเก่า
                    self.fire_ctrl.reset_lock()
                    hold_since = None
                    misses += 1
                    if misses >= 2:
                        raise HardwareFault("camera stream stalled while aiming")
                    continue
                misses = 0
                fd = gs.prepare_frame(frame)
                w, h = fd.size
                cands = gs.find_targets(fd, gs.build_mask(fd.hsv, self.color_ranges[color], self.blind_zones), color)
                self.overlay, self.overlay_w = cands, w
                ref = last_deg or (aim_x, aim_y)
                # ยังไม่ล็อก: รับป้ายที่ห่างจุดเล็งได้กว้างกว่า (หุ่นหยุดเยื้อง/เป้าเยื้อง) ล็อกแล้วค่อยใช้รัศมีแคบ
                best, best_deg, best_dist = None, None, min(gs.MATCH_RADIUS_DEG, acquire_radius) if last_deg else acquire_radius
                for cand in self.aim_candidates(fd, target, cands, plate_index):
                    deg = gs.pixel_to_degrees(cand.cx, cand.cy, w, h)
                    d = math.hypot(deg[0] - ref[0], deg[1] - ref[1])
                    if d < best_dist:
                        best, best_deg, best_dist = cand, deg, d
                if best is None:
                    self.io.gimbal_speed(0, 0)
                    self.pid_yaw.reset()
                    self.pid_pitch.reset()
                    self.fire_ctrl.reset_lock()
                    last_deg, hold_since, last_area = None, None, None
                    self.status = f"Aim {color.upper()}: target not in view"
                    continue
                # A sudden jump or size change may be a different plate:
                # reacquire it with a fresh lock instead of carrying dwell time.
                if last_deg is not None and (
                        math.hypot(best_deg[0] - last_deg[0], best_deg[1] - last_deg[1]) > AIM_TRACK_JUMP_DEG
                        or not AIM_TRACK_AREA_RATIO[0] <= best.area / max(last_area, 1e-6) <= AIM_TRACK_AREA_RATIO[1]):
                    self.fire_ctrl.reset_lock()
                    self.pid_yaw.reset()
                    self.pid_pitch.reset()
                    hold_since = None
                last_deg, last_area = best_deg, best.area
                ex, ey = best_deg[0] - aim_x, best_deg[1] - aim_y
                self.io.gimbal_speed(self.pid_pitch.compute(ey), self.pid_yaw.compute(ex))
                state, should_fire = self.fire_ctrl.evaluate(ex, ey, False)
                self.status = f"Aim {color.upper()}: {state}  err X {ex:+.2f} Y {ey:+.2f}"
                if should_fire:
                    self.check()
                    if finish_on_request and self.finish_requested:
                        break
                    self.io.gimbal_speed(0, 0)
                    remaining = SHOTS_PER_TARGET - shots
                    if self.io.fire(times=remaining):
                        shots += remaining
                        plate['shots_fired'] = shots
                        self.save_map(quiet=True)
                        self.log(f"FIRE {remaining} at {color.upper()} {target} ({shots}/{SHOTS_PER_TARGET})")
                    self.fire_ctrl.mark_fired()
                    if shots >= SHOTS_PER_TARGET:
                        result = "SHOT"
                        break
                elif self.fire_ctrl.settled and not self.armed:
                    hold_since = hold_since or time.monotonic()
                    if time.monotonic() - hold_since >= DRY_HOLD_S:
                        result = "AIMED"
                        break
                else:
                    hold_since = None
        finally:
            self.io.gimbal_speed(0, 0)
            self.overlay = []
        if backoff:
            self.move_in_cell(backoff)
            self.pose_uncertain = pose_before_backoff
        self.io.gimbal_to(0, 0)
        if result == "SKIPPED" and not (finish_on_request and self.finish_requested):
            self.log(f"Skip near target {target}: aim timed out after {AIM_TIMEOUT_S:.0f} s "
                     f"({self.status}; range {aim_range:.3f} m, backoff {backoff:.3f} m)")
        return result

    def recenter(self):
        self.io.gimbal_to(0, 0)

    def restore_state(self, saved: dict):
        """ --load: ของที่รอบก่อนรู้แล้วแต่ไม่ได้อยู่ในแผนที่ (ทิศที่มองแล้ว, ระยะกลางช่องที่เรียนได้) """
        self.looked = {((x, y), d) for x, y, d in saved.get("looked", [])}
        # A target marked complete under the previous two-shot rule still
        # needs its final shot under the current three-shot rule.
        for plates in self.maze.target_plates.values():
            for plate in plates:
                if 0 < plate.get('shots_fired', 0) < SHOTS_PER_TARGET:
                    plate['shot'] = False
        for row in saved.get("plate_results", []):
            cell = tuple(row['cell'])
            index = row['plate_index']
            if 0 <= index < len(self.maze.plates_at(cell)):
                self.plate_results[(cell, index)] = row['result']
        for cell, plates in self.maze.target_plates.items():
            count = sum(p.get('shot', False) for p in plates)
            if count:
                self.results[cell] = 'SHOT' if len(plates) == 1 else f"{count}/{len(plates)}"
        value = saved.get("wall_center_mm")
        if value is not None and WALL_CENTER_RANGE_MM[0] <= value <= WALL_CENTER_RANGE_MM[1]:
            self.io.wall_center_mm = float(value)
            self._wall_center_learned = True
            self.log(f"Loaded wall distance at cell centre: {value:.0f} mm")
        offset = saved.get("slam", {}).get("tof_wall_offset_mm")
        if self.slam is not None and isinstance(offset, (int, float)) and 80 <= offset <= 450:
            self.slam.wall_offset_m = offset / 1000
            self._slam_calibrated = True
            self._tof_offset_known = bool(saved.get("slam", {}).get("tof_offset_calibrated", offset != 250))

    def clear_map(self):
        """ ล้างแผนที่ทั้งหมดเพื่อสำรวจสนามใหม่ (ขนาด/จุดเริ่ม/ทิศเริ่มเดิม)
        ไม่ผูก IMU ที่นี่ (หุ่นอาจยังอยู่กลางสนาม): ต้องวางหุ่นที่ start แล้วกด h ก่อนถึงจะกด e ได้ """
        try:
            MAP_FILE.unlink(missing_ok=True)
        except OSError as e:
            self.log(f"Cannot erase the old map: {e}")
            return
        field = self.maze.field
        self.maze = MazeMap(self.maze.w, self.maze.h)
        self.maze.field = field
        self.results, self.bumper_hits, self.route = {}, {}, []
        self.plate_results = {}
        self.looked = set()
        self._wall_center_learned = self._wall_center_tried = False
        self.cell, self.heading = self.start_cell, self.start_heading
        self.pose_uncertain = True
        if self.slam is not None:
            self.slam = GridSLAM(self.maze, self.cell, self.heading)
            self._slam_calibrated = False
            self._tof_offset_known = False
        self.log(f"Map cleared: put robot at START "
                 f"{self.start_cell} facing {DIR_NAMES[self.start_heading]}, press h, then e")

    def reset_pose(self) -> bool:
        """ รอบ 2 แบบยกหุ่นไปวางที่จุดเริ่มเอง: บอกโปรแกรมว่าตอนนี้อยู่ช่อง start หันทิศเริ่มต้น
        ไม่ขับรถ แค่ผูกมุม IMU ปัจจุบันเข้ากับทิศเริ่มต้นใหม่ (ยกหุ่นแล้ว IMU มักเพี้ยน) """
        self.io.gimbal_to(0, 0)
        time.sleep(0.3)
        if self.io.sim:  # สนามจำลอง: ย้ายหุ่นจำลองตามไปด้วย ไม่งั้นแผนที่กับโลกจำลองหลุดกัน
            self.io.cell, self.io.heading = self.start_cell, self.start_heading
            self.io.yaw, self.io.backoff = YAW_TARGETS[self.start_heading], 0.0
        self.cell, self.heading = self.start_cell, self.start_heading
        self.yaw_offset = self.io.yaw - YAW_TARGETS[self.start_heading]
        self.route = []
        self.pose_uncertain = True
        if not self.verify_start():
            return False
        self.check()
        self.pose_uncertain = False
        if self.slam is not None:
            offset = self.slam.wall_offset_m
            self.slam = GridSLAM(self.maze, self.cell, self.heading, wall_offset_m=offset)
        self.log(f"Pose reset: at START {self.start_cell} facing {DIR_NAMES[self.start_heading]}")
        return True

    def verify_start(self) -> bool:
        """ วัดกำแพงรอบตัว 4 ทิศเทียบกับแผนที่ของช่อง start: วางหุ่นผิดทิศ/ผิดช่อง = ไม่ตรง -> ไม่ยอมเริ่ม
        (แผนที่ยังไม่รู้กำแพงรอบ start เช่นหลังล้างแผนที่ = ข้าม) """
        c = self.start_cell
        known = {d: self.maze.edge(c, MazeMap.step(c, d)) == -1 for d in range(4)
                 if self.maze.edge(c, MazeMap.step(c, d)) != 0}
        if not known:
            return True   # แผนที่ใหม่ยังไม่มีข้อมูลรอบจุดเริ่มให้เทียบ
        rel_of = {0: 0, 1: 90, 2: 180, 3: -90}
        measured = {}
        for d in range(4):
            self.check()
            self.io.gimbal_to(0 if self.slam is not None else TARGET_SCAN_PITCH,
                              rel_of[(d - self.heading) % 4])
            time.sleep(GIMBAL_SETTLE_S if not self.io.sim else 0.02)
            tof = self.io.read_tof_mm()
            if tof is not None:
                measured[d] = tof < WALL_TOF_MM
        self.io.gimbal_to(0, 0)
        missing = sorted(set(known) - set(measured))
        if missing:
            self.pose_uncertain = True
            self.log("START check failed: no ToF reading for " + ", ".join(DIR_NAMES[d] for d in missing))
            return False
        if all(measured[d] == known[d] for d in known):
            return True
        turned = [k * 90 for k in (1, 2, 3) if all(measured[d] == known[(d + k) % 4] for d in measured
                                                   if (d + k) % 4 in known)]
        self.pose_uncertain = True
        self.log("START check failed: walls around the robot do not match the map"
                 + (f" (robot looks turned {turned[0]} deg clockwise)" if turned else " (wrong cell?)")
                 + f". Place it at {c} facing {DIR_NAMES[self.start_heading]}, press h")
        return False

    def save_map(self, quiet: bool = False):
        """เขียนผ่านไฟล์ชั่วคราวแล้วสลับทับแผนที่ไฟล์เดียวแบบ atomic."""
        data = self.maze.to_json()
        if self.slam is not None:
            data["slam"] = {"method": "grid_particle_filter",
                            "pose_confidence": self.slam.confidence,
                            "tof_wall_offset_mm": self.slam.wall_offset_m * 1000,
                            "tof_offset_calibrated": self._tof_offset_known}
        data["pose_uncertain"] = self.pose_uncertain
        data.update({"wall_center_mm": self.io.wall_center_mm if self._wall_center_learned else None,
                     "looked": sorted([c[0], c[1], d] for c, d in self.looked),
                     "start": list(self.start_cell), "start_heading": self.start_heading,
                     "robot": list(self.cell), "heading": self.heading,
                     "results": [{"cell": list(c), "result": r} for c, r in self.results.items()],
                     "plate_results": [{"cell": list(c), "plate_index": i, "result": result}
                                       for (c, i), result in self.plate_results.items()],
                     "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        try:
            tmp = MAP_FILE.with_name(MAP_FILE.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, MAP_FILE)
            if not quiet:
                self.log(f"Map saved: {MAP_FILE.name}")
        except OSError as e:
            self.log(f"Cannot save map: {e}")
