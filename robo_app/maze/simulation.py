"""maze: simulation."""

from .mapping import MazeMap
from .settings import (
    CAMERA_FORWARD_M,
    CAMERA_HEIGHT_M,
    CELL_M,
    Cell,
    DIRS,
    PITCH_MIN_DEG,
    TARGET_HEIGHT_M,
    TARGET_WALL_GAP_M,
    WALL_CENTER_MM,
    YAW_TARGETS,
)
from .vision import normalize_angle
from robo_app import aim as gs
from typing import List
from typing import Optional
from typing import Tuple
import cv2
import math
import numpy as np
import time
import threading
from .errors import Aborted

# ==========================================
# 4. หุ่นจำลอง (--sim)
# ==========================================
class SimRobot:
    """ สนาม 4x4 ตัวอย่าง: กำแพงบาง, บล็อกทึบ 1 ช่อง, ห้องปิด 2 ช่อง, เป้า 4 อัน (สี + รูปทรง)
    เป้าตั้งชิดผนังฝั่งตรงข้ามทางเข้าช่อง หันหน้าออกจากผนัง แบบสนามจริง (มองจากด้านข้างจะเห็นแผ่นบางจนจับไม่ได้)
    ToF (ก้มหัวแล้วลำแสงชนพื้น) และภาพกล้องสร้างจากตำแหน่งหุ่น/มุมหัว แล้วส่งเข้าระบบจับสีตัวจริงของ gimbalshoot
    """
    sim = True
    WALLS = {((0, 1), (1, 1)), ((2, 2), (2, 3)), ((2, 0), (2, 1)), ((3, 2), (3, 3)),
             ((3, 0), (3, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1))}   # ช่อง (2,3),(3,3) = ห้องปิด
    BLOCKS = {(1, 3)}
    # ช่อง -> (สี, รูปทรง, ทิศของผนังที่เป้าตั้งชิด); (1,2) เป็นช่องทางผ่าน ส่วนที่เหลือเป็นทางตัน
    TARGETS = {(0, 3): ("blue", "circle", 0), (3, 0): ("green", "square", 1),
               (2, 1): ("red", "rect", 2), (1, 2): ("yellow", "circle", 0)}
    EXTRA_TARGETS = []  # (cell, color, shape, wall, lateral offset in metres)
    PLATE_M = {"circle": (0.065, 0.065), "square": (0.06, 0.06), "rect": (0.075, 0.045)}   # กว้าง, สูง
    BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
    W, H = 960, 540

    def __init__(self, start: Cell, heading: int, width: int, height: int):
        self._stop_event = threading.Event()
        self.cell, self.heading = start, heading
        self.width, self.height = width, height
        self.pitch = self.yaw_rel = 0.0          # มุมหัวเทียบตัวรถ
        self._speed = (0.0, 0.0)
        self._t = time.monotonic()
        self.yaw = YAW_TARGETS[heading]           # IMU
        self.backoff = 0.0                        # ถอยจากกลางช่อง (ม.)
        self.shots: List[Tuple[Optional[Cell], float]] = []
        self.shot_plates = []  # (cell, lateral offset), for verifying distinct hits
        self.entered_targets: List[Cell] = []     # ช่องเป้าที่หุ่นขับเข้าไป (ตรวจในเทสต์)
        self.tof_mm = 9999
        self.wall_center_mm = float(WALL_CENTER_MM)

    # ---------- โลกจำลอง ----------
    def _blocked(self, a: Cell, b: Cell) -> bool:
        key = (a, b) if a <= b else (b, a)
        inside = 0 <= b[0] < self.width and 0 <= b[1] < self.height
        return not inside or key in self.WALLS or b in self.BLOCKS or a in self.BLOCKS

    def _integrate(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        self.pitch = min(max(self.pitch + self._speed[0] * dt, PITCH_MIN_DEG), gs.PITCH_LIMITS[1])
        self.yaw_rel += self._speed[1] * dt

    def _camera_pose(self) -> Tuple[float, float, float]:
        """ ตำแหน่งกล้องบนพื้น (x, y) และมุมหัวเทียบทิศเหนือ (องศา ตามเข็ม) """
        a = self.heading * 90 + self.yaw_rel
        hx, hy = DIRS[self.heading]
        cx = (self.cell[0] + 0.5) * CELL_M - hx * self.backoff
        cy = (self.cell[1] + 0.5) * CELL_M - hy * self.backoff
        r = math.radians(a)
        return cx + CAMERA_FORWARD_M * math.sin(r), cy + CAMERA_FORWARD_M * math.cos(r), a

    @staticmethod
    def _plate_xy(cell: Cell, wall: int) -> Tuple[float, float]:
        dx, dy = DIRS[wall]
        off = CELL_M / 2 - TARGET_WALL_GAP_M
        return (cell[0] + 0.5) * CELL_M + dx * off, (cell[1] + 0.5) * CELL_M + dy * off

    def _visible_targets(self):
        """ เป้าในช่องเดียวกันหรือช่องตามแนวแถว/คอลัมน์เดียวกัน โดยไม่มีกำแพงกั้นระหว่างช่อง """
        plates = [(cell, color, shape, wall, 0.0) for cell, (color, shape, wall) in self.TARGETS.items()]
        for cell, color, shape, wall, offset in plates + list(self.EXTRA_TARGETS):
            dx, dy = cell[0] - self.cell[0], cell[1] - self.cell[1]
            if dx != 0 and dy != 0:
                continue
            if dx or dy:
                d = DIRS.index((int(math.copysign(1, dx)) if dx else 0, int(math.copysign(1, dy)) if dy else 0))
                c, ok = self.cell, True
                while c != cell:
                    n = MazeMap.step(c, d)
                    if self._blocked(c, n):
                        ok = False
                        break
                    c = n
                if not ok:
                    continue
            yield cell, color, shape, wall, offset

    def _target_angles(self, cell: Cell, wall: int, offset: float = 0.0) -> Tuple[float, float, float, float]:
        """ มุมเป้าเทียบแนวกล้อง: (ซ้ายขวา + = ขวา, ขึ้นลงเทียบแนวราบ, ระยะตามแนวกล้อง, หน้าแผ่นหันหากล้องแค่ไหน 0-1) """
        x, y, a = self._camera_pose()
        tx, ty = self._plate_xy(cell, wall)
        tx += DIRS[wall][1] * offset
        ty -= DIRS[wall][0] * offset
        r = math.radians(a)
        fwd = (tx - x) * math.sin(r) + (ty - y) * math.cos(r)
        lat = (tx - x) * math.cos(r) - (ty - y) * math.sin(r)
        bearing = math.degrees(math.atan2(lat, fwd))
        elev = math.degrees(math.atan2(TARGET_HEIGHT_M - CAMERA_HEIGHT_M, max(fwd, 1e-3)))
        nx, ny = -DIRS[wall][0], -DIRS[wall][1]              # หน้าแผ่นหันออกจากผนัง
        dist = math.hypot(tx - x, ty - y) or 1e-3
        facing = ((x - tx) * nx + (y - ty) * ny) / dist
        return bearing, elev, fwd, facing

    # ---------- คำสั่งเหมือน RealRobot ----------
    def read_tof_mm(self) -> Optional[float]:
        """ ToF บนหัว: หันไปทางไหนวัดกำแพงทางนั้น ก้มหัวแล้วลำแสงไปชนพื้นก่อนถ้ากำแพงไกลกว่า """
        self._pause(0.05)
        self._integrate()
        d = int(round((self.heading * 90 + self.yaw_rel) / 90.0)) % 4
        c, k = self.cell, 0
        while k < 6 and not self._blocked(c, MazeMap.step(c, d)):
            c, k = MazeMap.step(c, d), k + 1
        hit = CELL_M / 2 + k * CELL_M - 0.05
        if self.pitch < -1.0:
            hit = min(hit, CAMERA_HEIGHT_M / math.tan(math.radians(-self.pitch)))
        return hit / math.cos(math.radians(self.pitch)) * 1000

    def read_frame(self) -> np.ndarray:
        self._integrate()
        img = np.full((self.H, self.W, 3), 70, np.uint8)
        horizon = gs.degrees_to_pixel(0, -self.pitch, self.W, self.H)[1]
        cv2.rectangle(img, (0, max(0, horizon)), (self.W, self.H), (40, 60, 90), -1)
        for cell, color, shape, wall, offset in self._visible_targets():
            bearing, elev, fwd, facing = self._target_angles(cell, wall, offset)
            if fwd < 0.15 or facing < 0.3:
                continue  # ข้างหลังกล้อง หรือเห็นแผ่นจากด้านข้าง/ด้านหลัง
            px, py = gs.degrees_to_pixel(bearing, elev - self.pitch, self.W, self.H)
            wm, hm = self.PLATE_M[shape]
            hw = max(1, int(math.degrees(math.atan2(wm * facing / 2, fwd)) * self.W / gs.FOV_X))
            hh = max(1, int(math.degrees(math.atan2(hm / 2, fwd)) * self.H / gs.FOV_Y))
            if shape == "circle":
                cv2.ellipse(img, (px, py), (hw, hh), 0, 0, 360, self.BGR[color], -1)
            else:
                cv2.rectangle(img, (px - hw, py - hh), (px + hw, py + hh), self.BGR[color], -1)
        return img

    def set_stop_event(self, event):
        self._stop_event = event

    def _check_stop(self):
        if self._stop_event.is_set():
            raise Aborted()

    def _pause(self, seconds):
        if self._stop_event.wait(seconds):
            raise Aborted()

    def gimbal_to(self, pitch: float, yaw: float, speed: float = None):
        self._check_stop()
        self._integrate()
        self._speed = (0.0, 0.0)
        self.pitch, self.yaw_rel = min(max(pitch, PITCH_MIN_DEG), gs.PITCH_LIMITS[1]), yaw
        self._pause(0.05)

    def gimbal_speed(self, pitch_speed: float, yaw_speed: float):
        if pitch_speed or yaw_speed:
            self._check_stop()
        self._integrate()
        self._speed = (pitch_speed, yaw_speed)

    def rotate(self, steps: int):
        self._pause(0.15)
        self.heading = (self.heading + steps) % 4
        self.yaw = YAW_TARGETS[self.heading]

    def turn_by(self, deg: float):
        self._check_stop()
        self.yaw = normalize_angle(self.yaw + deg)

    def forward_one_cell(self, target_yaw: float, check, known_end_wall: bool = False) -> float:
        check()
        nxt = MazeMap.step(self.cell, self.heading)
        if self._blocked(self.cell, nxt):
            return 0.1  # ชนกำแพง -> กันชนเบรก
        self._pause(0.2)
        check()
        self.cell = nxt
        if nxt in self.TARGETS:
            self.entered_targets.append(nxt)
        return 1.0

    def move_straight(self, dx_m: float):
        self._check_stop()
        self.backoff -= dx_m

    def move_sideways(self, left_m: float):
        self._check_stop()
        self.last_lateral_move = left_m

    def fire(self, times: int = 1) -> bool:
        """ บันทึกว่ากระสุนไปโดนเป้าไหน (ใกล้แนวเล็งไม่เกิน 3°) """
        self._integrate()
        self._check_stop()
        best, best_err, best_plate = None, 3.0, None
        for cell, _, _, wall, offset in self._visible_targets():
            bearing, elev, fwd, facing = self._target_angles(cell, wall, offset)
            if fwd < 0.15 or facing < 0.3:
                continue
            err = math.hypot(bearing, elev - self.pitch)
            if err < best_err:
                best, best_err = cell, err
                best_plate = (cell, offset)
        for _ in range(times):
            self.shots.append((best, round(best_err, 2)))
            self.shot_plates.append(best_plate)
        return True

    def set_status_led(self, armed: bool, busy: bool):
        pass

    def stop_wheels(self):
        pass

    def stop_all(self):
        self.gimbal_speed(0, 0)

    def close(self):
        pass
