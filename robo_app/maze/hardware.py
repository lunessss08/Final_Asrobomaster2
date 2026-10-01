"""maze: hardware."""

from .errors import Aborted, HardwareFault
from .settings import (
    APPROACH_GAIN,
    APPROACH_ZONE_M,
    APPROACH_MAX_SPEED,
    BASE_SPEED,
    BUMPER_MM,
    BUMPER_SAMPLES,
    CELL_M,
    CELL_ARRIVAL_TOL_M,
    GIMBAL_SPEED,
    GIMBAL_MOVE_TIMEOUT_MIN_S,
    KD_YAW,
    KP_SIDE,
    KP_YAW,
    MAX_SIDE_SHIFT_M,
    MAX_SIDE_SPEED,
    MIN_DRIVE_SPEED,
    MOVE_TIMEOUT_S,
    ODOM_STALE_S,
    SENSOR_MAP_FILE,
    SENSOR_STALE_S,
    TOF_RECOVERY_MARGIN_MM,
    TOF_ARRIVAL_MAX_ODOM_GAP_M,
    TOF_ARRIVAL_WALL_TOL_MM,
    SHARP_ADC_MIN,
    SHARP_RANGE_CM,
    SHARP_STALE_S,
    SIDE_CENTER_CLAMP_CM,
    SIDE_CENTER_CM,
    SIDE_DEADBAND_CM,
    SIDE_KEEP,
    SIDE_WALL_MAX_CM,
    STOP_TOL_M,
    STOP_CHECK_INTERVAL_S,
    STOP_STABLE_SAMPLES,
    STOP_STABLE_TOL_M,
    STOP_VERIFY_TIMEOUT_S,
    TOF_MIN_VALID_MM,
    TOF_SAMPLE_S,
    TURN_SPEED,
    WALL_CENTER_MM,
)
from .vision import normalize_angle
from collections import deque
from robo_app import aim as gs
from robomaster import blaster
from robomaster import dds
from robomaster import robot
from typing import Dict
from typing import Optional
from typing import Tuple
import json
import math
import numpy as np
import threading
import time
from functools import wraps


def chassis_led(method):
    """Show the moving color for the full chassis command, including stop checks."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._check_stop()
        self._chassis_moving = True
        self._refresh_status_led()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._chassis_moving = False
            self._refresh_status_led()
    return wrapped

# ==========================================
# 3. ฮาร์ดแวร์หุ่น (ของจริง)
# ==========================================
class PinboardSubject(dds.Subject):
    """ stream ADC ของ Sensor Adaptor ทุกบอร์ด (6 บอร์ด x 2 พอร์ต)
    robomaster ที่ pip ติดตั้งใน venv (0.1.1.61) ไม่มี sub_adapter แต่ DDS รู้จัก subject "pinboard" อยู่แล้ว
    จึงสร้างเองตามแบบ RoboMaster-SDK/src/robomaster/sensor.py
    """
    name = dds.DDS_PINBOARD
    uid = dds.SUB_UID_MAP[dds.DDS_PINBOARD]
    type = dds.DDS_SUB_TYPE_PERIOD

    def __init__(self, freq: int):
        super().__init__()
        self.freq = freq
        self._io = [0] * 12
        self._ad = [0] * 12

    def data_info(self):
        return self._io, self._ad

    def decode(self, buf):
        for i in range(min(6, len(buf) // 6)):
            self._io[i * 2] = buf[i * 6]
            self._io[i * 2 + 1] = buf[i * 6 + 1]
            self._ad[i * 2] = buf[i * 6 + 2] + buf[i * 6 + 3] * 256
            self._ad[i * 2 + 1] = buf[i * 6 + 4] + buf[i * 6 + 5] * 256


SideChannels = Dict[str, Tuple[Tuple[int, int], Tuple[float, float]]]  # ข้าง -> ((board, port), (A, B))


def load_side_channels() -> Tuple[SideChannels, str]:
    """ อ่านช่อง + ค่าสอบเทียบ Sharp จาก sensor_map.json เอาเฉพาะข้างที่สอบเทียบแล้วและสมการสมเหตุสมผล
    (cm = A * adc^B ของ Sharp จริง B ติดลบราว -1; ค่าที่ฟิตจากสัญญาณตายจะได้ B ใกล้ 0 ต้องทิ้ง) """
    if not SENSOR_MAP_FILE.exists():
        return {}, f"no {SENSOR_MAP_FILE.name} (run sensor_test.py: ports, dist)"
    try:
        data = json.loads(SENSOR_MAP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, f"cannot read {SENSOR_MAP_FILE.name}: {e}"
    calibrated = data.get("sharp_calibrated")
    out: SideChannels = {}
    notes = []
    for side in ("left", "right"):
        try:
            board, port = (int(v) for v in data[f"sharp_{side}"])
            a, b = (float(v) for v in data["sharp_calib"][side])
        except (KeyError, TypeError, ValueError):
            notes.append(f"{side}: not in the file")
            continue
        if not (isinstance(calibrated, dict) and calibrated.get(side)):
            notes.append(f"{side}: not calibrated")
        elif not (a > 0 and -2.0 <= b <= -0.6 and 1 <= board <= 6 and port in (1, 2)):
            notes.append(f"{side}: calibration looks wrong (A={a}, B={b})")
        else:
            out[side] = ((board, port), (a, b))
    return out, "; ".join(notes)


class SharpSides:
    """ อ่าน Sharp IR ข้างซ้าย/ขวา: ใช้ stream pinboard ก่อน ถ้าไม่มีค่าเข้า ถอยไป poll get_adc (แบบแล็บ 5)
    เก็บ 3 ค่าล่าสุดต่อข้างแล้วใช้มัธยฐาน กันค่ากระโดด """

    def __init__(self, ep, channels: SideChannels):
        self.ep = ep
        self.channels = dict(channels)
        self.lock = threading.Lock()
        self.buf = {side: deque(maxlen=3) for side in self.channels}
        self.last_t = {side: 0.0 for side in self.channels}
        self.count = 0
        self.mode = "stream"
        self.subscribed = False
        self._halt = threading.Event()
        self._poller: Optional[threading.Thread] = None

    def _store(self, side: str, value):
        if value is not None and value >= SHARP_ADC_MIN:
            self.buf[side].append(value)
            self.last_t[side] = time.monotonic()

    def _on_adapter(self, info):
        _io, ad = info
        with self.lock:
            for side, ((board, port), _) in self.channels.items():
                self._store(side, ad[(board - 1) * 2 + (port - 1)])
            self.count += 1

    def _poll(self):
        while not self._halt.is_set():
            for side, ((board, port), _) in list(self.channels.items()):
                try:
                    v = self.ep.sensor_adaptor.get_adc(id=board, port=port)
                except Exception:  # SDK จับเองส่วนใหญ่ กันไว้ไม่ให้ thread ตาย
                    v = None
                with self.lock:
                    self._store(side, v)
            with self.lock:
                self.count += 1
            time.sleep(0.02)

    def start(self) -> str:
        """ เริ่มอ่าน แล้วตัดข้างที่ไม่มีสัญญาณทิ้ง คืนข้อความสรุปไว้ log """
        # add_subject_info(subject, callback, args, kw): SDK อ่าน args/kw แบบ positional เสมอ
        ok = gs.safe_call("sub pinboard", self.ep.dds.add_subject_info, PinboardSubject(20), self._on_adapter, (), {})
        self.subscribed = ok is not None and ok is not False
        time.sleep(0.8)
        if self.count == 0:
            if self.subscribed:
                gs.safe_call("unsub pinboard", self.ep.dds.del_subject_info, dds.DDS_PINBOARD)
                self.subscribed = False
            self.mode = "poll"
            self._poller = threading.Thread(target=self._poll, name="sharp-poll", daemon=True)
            self._poller.start()
            time.sleep(0.8)
        with self.lock:
            dead = [side for side in self.channels if not self.buf[side]]
            for side in dead:
                del self.channels[side]
        parts = [f"{side} board {bp[0]} port {bp[1]}" for side, (bp, _) in self.channels.items()]
        parts += [f"{side}: no signal" for side in dead]
        return f"{self.mode}: " + ", ".join(parts)

    def cm(self, side: str) -> Optional[float]:
        with self.lock:
            if side not in self.channels or time.monotonic() - self.last_t[side] > SHARP_STALE_S:
                return None
            values = list(self.buf[side])
            a, b = self.channels[side][1]
        if not values:
            return None
        cm = a * float(np.median(values)) ** b
        return cm if SHARP_RANGE_CM[0] <= cm <= SHARP_RANGE_CM[1] else None

    def stop(self):
        self._halt.set()
        if self.subscribed:
            gs.safe_call("unsub pinboard", self.ep.dds.del_subject_info, dds.DDS_PINBOARD)
            self.subscribed = False
        if self._poller is not None:
            self._poller.join(timeout=2)


class RealRobot:
    """ คำสั่ง SDK ทั้งหมดที่โปรแกรมนี้ใช้: ToF บนหัว, IMU, odometry, Sharp (ถ้ามี), กล้อง, gimbal, ล้อ, blaster, ไฟ """
    sim = False

    def __init__(self):
        self._stop_event = threading.Event()
        self._gimbal_pose = None  # Last completed (pitch, yaw) command.
        self._chassis_moving = False
        self._gimbal_moving = False
        self._led_armed = False
        self._led_busy = False
        self.tof_mm = 0
        self.tof_seq = 0                         # นับค่า ToF ที่เข้ามา (กันชนใช้แยกค่าใหม่กับค่าเดิม)
        self.tof_t = 0.0                         # เวลาที่ได้ค่า ToF ล่าสุด (เช็ค stream ค้าง)
        self._tof_log: deque = deque(maxlen=60)  # (เวลา, มม.)
        self.yaw = 0.0
        self.att_count = 0
        self.attitude_t = 0.0
        self.pos = (0.0, 0.0)                    # odometry (ม.) นับจากตอน subscribe
        self.pos_t = 0.0
        self.wall_center_mm = float(WALL_CENTER_MM)
        self.sides: Optional[SharpSides] = None
        self.side_center_cm = SIDE_CENTER_CM
        self.side_status = ""
        self.ep = robot.Robot()
        print("[INFO] Connecting to RoboMaster (AP)...")
        if not self.ep.initialize(conn_type=gs.CONN_TYPE):
            raise ConnectionError("Cannot connect to the robot - check Wi-Fi")
        self.chassis, self.gimbal = self.ep.chassis, self.ep.gimbal
        self.camera, self.blaster = self.ep.camera, self.ep.blaster
        # FREE = หันหัวแยกจากตัวรถได้ (ต้องใช้ตอนหันหัววัดกำแพงซ้าย/ขวา)
        if (not gs.safe_call("set robot mode", self.ep.set_robot_mode, mode=robot.FREE)
                or gs.safe_call("check robot mode", self.ep.get_robot_mode) != robot.FREE):
            gs.safe_call("robot close", self.ep.close)
            raise ConnectionError("Could not confirm FREE robot mode; Gimbal scan may turn the body")
        self.leds = gs.LedManager(self.ep.led, self.blaster)  # ไฟ Top/Blaster ดับตลอด (จาก gimbalshoot)
        self.leds.silence_gimbal_lights()
        self.leds.show_status("white", False, False)
        tof_ok = self.ep.sensor.sub_distance(freq=20, callback=self._on_tof)
        imu_ok = self.chassis.sub_attitude(freq=20, callback=self._on_attitude)
        gs.safe_call("sub position", self.chassis.sub_position, cs=0, freq=20, callback=self._on_position)
        if not gs.safe_call("start video", self.camera.start_video_stream,
                            display=False, resolution=gs.STREAM_RESOLUTION):
            self.close()
            raise ConnectionError("Camera stream did not start; target detection is unavailable")
        self.gimbal_to(0, 0)
        time.sleep(1.0)  # รอค่า ToF/IMU ชุดแรก
        self._check_sensors(tof_ok, imu_ok)
        if time.monotonic() - self.pos_t > ODOM_STALE_S:
            print("[WARN] No chassis odometry: driving will be blocked until fresh position data arrives")
        self._start_sides()

    def set_stop_event(self, event):
        self._stop_event = event

    def _check_stop(self):
        if self._stop_event.is_set():
            raise Aborted()

    def _pause(self, seconds):
        if self._stop_event.wait(seconds):
            raise Aborted()

    def _wait_action(self, label, action, timeout):
        # SDK wait_for_completed marks an action failed on timeout; do not
        # repeatedly call it with short timeouts to poll a running action.
        try:
            self._check_stop()
            if action is None:
                raise HardwareFault(f"{label}: command was not accepted")
            deadline = time.monotonic() + timeout
            while True:
                self._check_stop()
                if action.is_completed:
                    if action.has_succeeded:
                        return
                    # The SDK sets is_completed at 100% progress even while
                    # its state still says action_running. Wait for the final
                    # success/failure push instead of treating that as failure.
                    if action.state not in ("action_running", "action_started"):
                        raise HardwareFault(f"{label}: action failed ({action.state})")
                if time.monotonic() >= deadline:
                    raise HardwareFault(f"{label}: timed out")
                self._pause(0.02)
        except (Aborted, HardwareFault):
            self.stop_all()
            raise

    def _start_sides(self):
        """ Sharp ข้างซ้าย/ขวาเป็นของเสริม: ไม่มี/ไม่มีสัญญาณ = เดินแบบเดิม ไม่หยุดโปรแกรม """
        if not SIDE_KEEP:
            print("[INFO] Side keeping: off (SIDE_KEEP = False)")
            return
        channels, note = load_side_channels()
        if channels:
            self.sides = SharpSides(self.ep, channels)
            summary = self.sides.start()
            if not self.sides.channels:
                self.sides.stop()
                self.sides = None
            print(f"[INFO] Side keeping with Sharp IR: {summary}" if self.sides else
                  f"[WARN] Side keeping off: Sharp configured but no signal ({summary})")
        if note:
            print(f"[INFO] Sharp: {note}")

    def side_speed(self) -> float:
        """ ความเร็วสไลด์ข้าง (m/s, + = ขวา) ให้อยู่กลางช่อง จาก Sharp ข้างที่มีกำแพงใกล้กว่า SIDE_WALL_MAX_CM
        สองข้างเห็นกำแพง: L + R คงที่ไม่ว่าหุ่นเยื้องแค่ไหน จึงใช้ (L + R) / 2 เรียนระยะกลางช่องจริงไปด้วย """
        if self.sides is None:
            return 0.0
        left, right = self.sides.cm("left"), self.sides.cm("right")
        left = left if left is not None and left <= SIDE_WALL_MAX_CM else None
        right = right if right is not None and right <= SIDE_WALL_MAX_CM else None
        if left is not None and right is not None:
            pair_center = (left + right) / 2.0
            if abs(pair_center - SIDE_CENTER_CM) <= SIDE_CENTER_CLAMP_CM + 1.0:  # ค่าขอบกำแพง/ค่าเพี้ยนไม่เอามาเรียน
                self.side_center_cm += 0.1 * (pair_center - self.side_center_cm)
                self.side_center_cm = min(max(self.side_center_cm, SIDE_CENTER_CM - SIDE_CENTER_CLAMP_CM),
                                          SIDE_CENTER_CM + SIDE_CENTER_CLAMP_CM)
            offset = (left - right) / 2.0
        elif left is not None:
            offset = left - self.side_center_cm
        elif right is not None:
            offset = self.side_center_cm - right
        else:
            self.side_status = "side: open both"
            return 0.0
        fmt = lambda v: "--" if v is None else f"{v:.0f}"
        self.side_status = f"L {fmt(left)} R {fmt(right)} cm, off {offset:+.1f}"  # + = หุ่นเยื้องไปทางขวา
        if abs(offset) < SIDE_DEADBAND_CM:
            return 0.0
        return max(-MAX_SIDE_SPEED, min(MAX_SIDE_SPEED, -KP_SIDE * offset))

    def _check_sensors(self, tof_ok, imu_ok):
        """ ก่อนเริ่มงาน: ToF ต้องมีค่าจริง (ไม่ใช่ 0 ตลอด = เสียบผิดช่อง/สายหลุด) และ IMU ต้องส่งค่ามา
        ไม่งั้นกันชนไม่ทำงาน แผนที่เปิดโล่งหมด หรือรถหมุนวน โดยไม่มีอะไรเตือน """
        problems = []
        if tof_ok is False or not any(v >= TOF_MIN_VALID_MM for _, v in list(self._tof_log)):
            problems.append(f"ToF gives no valid reading (last {self.tof_mm} mm): check the cable and port")
        if imu_ok is False or self.att_count == 0:
            problems.append("chassis IMU attitude sends no data")
        if problems:
            self.close()
            raise ConnectionError("; ".join(problems))

    def _on_tof(self, info):
        now = time.monotonic()
        self.tof_mm = info[0]
        self.tof_t = now
        self.tof_seq += 1
        self._tof_log.append((now, info[0]))

    def _on_attitude(self, info):
        self.yaw = info[0]
        self.att_count += 1
        self.attitude_t = time.monotonic()

    def _on_position(self, info):
        self.pos = (info[0], info[1])
        self.pos_t = time.monotonic()

    def read_tof_mm(self) -> Optional[float]:
        """ ค่ามัธยฐานของ ToF ช่วง TOF_SAMPLE_S ถัดจากนี้ (ตัดค่า 0/อ่านพลาดทิ้ง) """
        t0 = time.monotonic()
        self._check_stop()
        self._pause(TOF_SAMPLE_S)
        values = [v for t, v in list(self._tof_log) if t >= t0 and v >= TOF_MIN_VALID_MM]
        return float(np.median(values)) if len(values) >= 3 else None

    def read_frame(self) -> Optional[np.ndarray]:
        return gs.read_frame(self.camera)

    def gimbal_to(self, pitch: float, yaw: float, speed: float = GIMBAL_SPEED):
        """ หันหัวไปมุมที่กำหนดแล้วรอจนถึง (yaw เทียบตัวรถ + = ขวา, pitch + = เงย) """
        self._check_stop()
        previous = getattr(self, "_gimbal_pose", None)
        angle = (max(abs(pitch - previous[0]), abs(yaw - previous[1]))
                 if previous is not None else 180.0)
        # SDK actions occasionally take longer than their nominal angular travel.
        # Keep a margin for transport/settling, while retaining a finite fault deadline.
        timeout = max(GIMBAL_MOVE_TIMEOUT_MIN_S, angle / max(speed, 1.0) + 3.0)
        self._gimbal_moving = True
        self._refresh_status_led()
        try:
            gs.safe_call("gimbal stop", self.gimbal.drive_speed, pitch_speed=0, yaw_speed=0)
            action = gs.safe_call("gimbal moveto", self.gimbal.moveto, pitch=pitch, yaw=yaw,
                                  pitch_speed=speed, yaw_speed=speed)
            self._wait_action("gimbal move", action, timeout)
        except HardwareFault as exc:
            raise HardwareFault(
                f"{exc} (pitch {pitch:+.0f}°, yaw {yaw:+.0f}°, "
                f"speed {speed:g}°/s, limit {timeout:.1f}s)"
            ) from exc
        finally:
            self._gimbal_moving = False
            self._refresh_status_led()
        self._gimbal_pose = (pitch, yaw)

    def gimbal_speed(self, pitch_speed: float, yaw_speed: float):
        if pitch_speed or yaw_speed:
            self._check_stop()
        gs.safe_call("gimbal drive", self.gimbal.drive_speed, pitch_speed=pitch_speed, yaw_speed=yaw_speed)

    @chassis_led
    def rotate(self, steps: int):
        """ หมุนตัวรถทีละ 90° แบบ assignment2.py (1 = ขวา, 2 = กลับหลัง, 3 = ซ้าย) """
        steps %= 4
        if steps == 0:
            return
        self._check_stop()
        self.stop_wheels()
        self._pause(0.1)
        z = {1: -90, 2: 180, 3: 90}[steps]  # chassis.move: z บวก = หมุนซ้าย
        action = gs.safe_call("chassis turn", self.chassis.move, x=0, y=0, z=z, z_speed=TURN_SPEED)
        self._wait_action("chassis turn", action, MOVE_TIMEOUT_S)
        self._pause(0.3)

    @chassis_led
    def turn_by(self, deg: float):
        """ หมุนตัวรถแก้มุมเล็กๆ (+ = ตามเข็ม/ขวา แบบเดียวกับ IMU) """
        self._check_stop()
        action = gs.safe_call("chassis correct", self.chassis.move, x=0, y=0, z=-deg, z_speed=TURN_SPEED)
        self._wait_action("chassis correction", action, MOVE_TIMEOUT_S)
        self._pause(0.3)

    def _tof_confirms_cell_arrival(self, reading: Optional[float], distance: float,
                                   known_end_wall: bool) -> bool:
        """Accept a short odometry reading only against a mapped, nearby end wall."""
        return (known_end_wall and reading is not None
                and CELL_M - TOF_ARRIVAL_MAX_ODOM_GAP_M <= distance <= CELL_M + CELL_ARRIVAL_TOL_M
                and reading >= BUMPER_MM + TOF_RECOVERY_MARGIN_MM
                and abs(reading - self.wall_center_mm) <= TOF_ARRIVAL_WALL_TOL_MM)

    @chassis_led
    def forward_one_cell(self, target_yaw: float, check, known_end_wall: bool = False) -> float:
        """ เดินหน้า 1 ช่องแล้วหยุดกลางช่องถัดไป คืนสัดส่วนระยะที่เดินได้ (1.0 = ครบช่อง)
        - ระยะที่เหลือจาก odometry (ไม่มีค่าใหม่ = หยุด) และถ้า ToF เห็นกำแพงหน้าช่องถัดไป
          ใช้ ToF ด้วย (หยุดเมื่อ ToF = wall_center_mm) เอาค่าที่น้อยกว่า แล้วชะลอเข้าหาจุดหยุด
          -> ไม่ไถลเกินสะสมทีละช่องจนชนกำแพงหน้า/เฉี่ยวตอนเลี้ยว
        - IMU คุมหัวตรง, Sharp (ถ้ามี) สไลด์ให้อยู่กลางช่อง สะสมไม่เกิน MAX_SIDE_SHIFT_M ต่อช่อง
        - กันชน: ToF ใกล้กว่า BUMPER_MM ติดกัน BUMPER_SAMPLES ค่า = เบรก / ToF ค้าง = หยุดรถทันที """
        use_odom = time.monotonic() - self.pos_t < ODOM_STALE_S
        self.tof_arrival_confirmed = False
        if not use_odom:
            self.stop_wheels()
            raise HardwareFault("no fresh odometry: cannot confirm cell arrival")
        bumper_stopped = False
        tof_rechecked = False
        x0, y0 = self.pos
        t0 = last = time.monotonic()
        timeout = 3.0 * CELL_M / BASE_SPEED + 1.0
        last_err, estimate, side_shift = 0.0, 0.0, 0.0
        seen_seq, close_hits = self.tof_seq, 0
        recent: deque = deque(maxlen=3)   # ToF 3 ค่าล่าสุด ใช้มัธยฐานตัดสินจุดหยุด ค่ากระโดดค่าเดียวไม่ทำให้หยุด
        tof_window_mm = self.wall_center_mm + CELL_M * 1000 + 150  # กำแพงหน้าของช่องถัดไป (หรือใกล้กว่า)
        try:
            while True:
                check()
                now = time.monotonic()
                dt, last = now - last, now
                if now - self.tof_t > SENSOR_STALE_S:
                    raise HardwareFault(f"ToF stream stalled ({now - self.tof_t:.1f} s without data) while driving")
                if use_odom and now - self.pos_t > ODOM_STALE_S:
                    raise HardwareFault("odometry stream stalled while driving")
                if now - getattr(self, "attitude_t", now) > SENSOR_STALE_S:
                    raise HardwareFault("IMU attitude stream stalled while driving")
                if use_odom:
                    estimate = math.hypot(self.pos[0] - x0, self.pos[1] - y0)
                if self.tof_seq != seen_seq:
                    seen_seq = self.tof_seq
                    valid = self.tof_mm >= TOF_MIN_VALID_MM
                    if valid:
                        recent.append(self.tof_mm)
                    close_hits = close_hits + 1 if valid and self.tof_mm <= BUMPER_MM else 0
                    if close_hits >= BUMPER_SAMPLES:
                        bumper_stopped = True
                        break
                remaining = CELL_M - estimate
                front = float(np.median(recent)) if len(recent) == recent.maxlen else None
                if front is not None and front <= tof_window_mm:
                    remaining = min(remaining, (front - self.wall_center_mm) / 1000.0)
                if now - t0 > timeout:
                    raise HardwareFault("drive timed out before confirming cell arrival")
                if remaining <= STOP_TOL_M:
                    # A person crossing the beam can look like the far wall. Stop
                    # first, then require a fresh stationary measurement before
                    # driving the last few centimetres. Never bypass the bumper.
                    if (front is not None and estimate < CELL_M - CELL_ARRIVAL_TOL_M
                            and not tof_rechecked):
                        tof_rechecked = True
                        self.stop_wheels()
                        self.ensure_stopped()
                        check()
                        estimate = math.hypot(self.pos[0] - x0, self.pos[1] - y0)
                        if estimate >= CELL_M - CELL_ARRIVAL_TOL_M:
                            break
                        fresh = self.read_tof_mm()
                        needed = BUMPER_MM + (CELL_M - estimate) * 1000 + TOF_RECOVERY_MARGIN_MM
                        if self._tof_confirms_cell_arrival(fresh, estimate, known_end_wall):
                            # The mapped end wall confirms the centre more closely
                            # than odometry. SLAM will verify the cell on arrival.
                            self.tof_arrival_confirmed = True
                            break
                        if fresh is None or fresh < needed:
                            raise HardwareFault(
                                f"front ToF still blocked at {fresh} mm; moved {estimate:.3f} m, "
                                f"need {needed:.0f} mm clear")
                        recent.clear()
                        seen_seq = self.tof_seq
                        close_hits = 0
                        last = time.monotonic()
                        t0 += 1.0  # Allow only the bounded stationary recheck.
                        continue
                    break
                speed = max(MIN_DRIVE_SPEED, min(BASE_SPEED, APPROACH_GAIN * remaining))
                if remaining <= APPROACH_ZONE_M:
                    speed = min(speed, APPROACH_MAX_SPEED)
                err = normalize_angle(target_yaw - self.yaw)
                if abs(err) >= 18:
                    raise HardwareFault(f"body heading drifted {err:+.1f} deg while driving")
                if abs(err) >= 6:
                    speed = min(speed, 0.08)
                z_speed = max(-25, min(25, KP_YAW * err + KD_YAW * (err - last_err)))
                last_err = err
                side = self.side_speed() if abs(side_shift) < MAX_SIDE_SHIFT_M else 0.0
                if remaining <= APPROACH_ZONE_M:
                    side = max(-speed * 0.5, min(speed * 0.5, side))
                side_shift += side * dt
                self._check_stop()
                self.chassis.drive_speed(x=speed, y=side, z=z_speed)
                time.sleep(0.02)
        finally:
            self.stop_wheels()
            time.sleep(0.25)
        self.ensure_stopped()
        if use_odom:
            estimate = math.hypot(self.pos[0] - x0, self.pos[1] - y0)  # รวมระยะไถลหลังสั่งหยุด
        if time.monotonic() - self.pos_t > ODOM_STALE_S:
            raise HardwareFault("odometry stale at cell arrival")
        if bumper_stopped and estimate >= CELL_M - CELL_ARRIVAL_TOL_M:
            raise HardwareFault("bumper stopped near the cell centre: cannot confirm arrival")
        if not bumper_stopped and not self.tof_arrival_confirmed and abs(estimate - CELL_M) > CELL_ARRIVAL_TOL_M:
            # Odometry can settle a few millimetres short after braking. A
            # fresh, stationary ToF reading against the mapped end wall can
            # confirm arrival without driving closer to that wall.
            fresh = self.read_tof_mm() if known_end_wall and estimate < CELL_M else None
            if self._tof_confirms_cell_arrival(fresh, estimate, known_end_wall):
                self.tof_arrival_confirmed = True
            else:
                raise HardwareFault(f"not at cell centre: moved {estimate:.3f} m, expected {CELL_M:.3f} m; "
                                    f"front ToF {front} mm")
        return estimate / CELL_M

    @chassis_led
    def move_straight(self, dx_m: float):
        """ เดินหน้า (+) / ถอย (-) ระยะสั้นในช่องเดิม """
        self._check_stop()
        action = gs.safe_call("chassis move", self.chassis.move, x=dx_m, y=0, z=0, xy_speed=0.3)
        self._wait_action("straight move", action, 4)
        self.stop_wheels()
        self.ensure_stopped()
        self._check_stop()

    @chassis_led
    def move_sideways(self, left_m: float):
        """Small stationary correction; positive distance is chassis-left."""
        self._check_stop()
        if abs(left_m) > 0.06:
            raise HardwareFault("sideways correction exceeds 60 mm")
        action = gs.safe_call("chassis lateral", self.chassis.move, x=0, y=-left_m, z=0, xy_speed=0.15)
        self._wait_action("lateral correction", action, 5)
        self.stop_wheels()
        self.ensure_stopped()
        self._check_stop()

    def fire(self, times: int = 1) -> bool:
        self._check_stop()
        ok = gs.safe_call("blaster fire", self.blaster.fire, fire_type=blaster.WATER_FIRE, times=times)
        self.leds.silence_gimbal_lights()  # ไม่ใช้ไฟกระพริบตอนยิง กันแสงเข้ากล้อง
        return bool(ok)

    def set_status_led(self, armed: bool, busy: bool):
        self._led_armed, self._led_busy = armed, busy
        self._refresh_status_led()

    def _refresh_status_led(self):
        leds = getattr(self, "leds", None)
        if leds is None:
            return
        if getattr(self, "_gimbal_moving", False):
            leds.show_status("white", getattr(self, "_led_armed", False),
                             getattr(self, "_led_busy", False))
        elif getattr(self, "_chassis_moving", False):
            leds.show_status("red", False, False)
        else:
            leds.show_status("white", getattr(self, "_led_armed", False),
                             getattr(self, "_led_busy", False))

    def stop_wheels(self):
        """ drive_speed เป็น PUSH ไม่มีตอบรับ (แพ็กเก็ตหายได้) จึงส่งศูนย์ซ้ำหลายครั้งก่อน แล้วปิดท้ายด้วย drive_wheels
        ที่รอตอบรับ -> คำสั่งหยุดหายไปครั้งเดียวรถไม่ไหลต่อ """
        for _ in range(3):
            gs.safe_call("wheels stop", self.chassis.drive_speed, x=0, y=0, z=0)
        gs.safe_call("wheels stop", self.chassis.drive_wheels, w1=0, w2=0, w3=0, w4=0)

    def ensure_stopped(self):
        """ยืนยันข้อมูลตำแหน่งใหม่และนิ่งต่อเนื่อง; ห้ามเดิน/เลี้ยวต่อหากยืนยันไม่ได้."""
        deadline = time.monotonic() + STOP_VERIFY_TIMEOUT_S
        anchor, sample_t = self.pos, self.pos_t
        stable = 0
        while time.monotonic() < deadline:
            time.sleep(STOP_CHECK_INTERVAL_S)
            now = time.monotonic()
            position, position_t = self.pos, self.pos_t
            if now - position_t > ODOM_STALE_S:
                self.stop_wheels()
                raise HardwareFault("odometry stale while verifying chassis stop")
            if position_t <= sample_t:
                # A repeated old position cannot prove the robot has stopped.
                stable = 0
                anchor = position
                continue
            sample_t = position_t
            if math.hypot(position[0] - anchor[0], position[1] - anchor[1]) <= STOP_STABLE_TOL_M:
                stable += 1
                if stable >= STOP_STABLE_SAMPLES:
                    return
            else:
                stable = 0
                anchor = position
                print("[WARN] Chassis still moving after stop: resending stop")
                self.stop_wheels()
        self.stop_wheels()
        raise HardwareFault("chassis did not settle: cannot confirm stop")

    def stop_all(self):
        self.stop_wheels()
        self.gimbal_speed(0, 0)

    def close(self):
        self.stop_all()
        if not self._stop_event.is_set():
            gs.safe_call("close recenter", self.gimbal_to, 0, 0)
        if getattr(self, "sides", None) is not None:
            self.sides.stop()
        gs.safe_call("unsub tof", self.ep.sensor.unsub_distance)
        gs.safe_call("unsub imu", self.chassis.unsub_attitude)
        gs.safe_call("unsub position", self.chassis.unsub_position)
        self.leds.close()
        gs.safe_call("stop video", self.camera.stop_video_stream)
        gs.safe_call("robot close", self.ep.close)
