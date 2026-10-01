"""maze: cli."""

from .camera import FrameGrabber
from .hardware import RealRobot
from .mapping import MazeMap
from .mission import MazeMission
from .settings import DIR_NAMES, MAP_FILE
from .simulation import SimRobot
from .ui import App
import argparse
import cv2
import json
import time

# ==========================================
# 7. เริ่มโปรแกรม
# ==========================================
def ask_int(prompt: str, default: int, low: int, high: int) -> int:
    while True:
        text = input(f"{prompt} [Enter = {default}]: ").strip()
        if not text:
            return default
        if text.lstrip("-").isdigit() and low <= int(text) <= high:
            return int(text)
        print(f"  ใส่ตัวเลข {low}-{high}")


def main():
    parser = argparse.ArgumentParser(description="RoboMaster EP: explore a maze (round 1), then shoot targets (round 2)")
    parser.add_argument("--sim", action="store_true", help="จำลองบนคอม ไม่ต่อหุ่น (สนาม 4x4 ตัวอย่าง)")
    parser.add_argument("--no-slam", action="store_true", help="ใช้ระบบสำรวจเดิม ไม่แก้ตำแหน่งด้วย Grid SLAM")
    parser.add_argument("--load", action="store_true",
                        help=f"โหลดแผนที่รอบ 1 จาก {MAP_FILE.name} แล้วเริ่มรอบ 2 ได้เลย (ยกหุ่นมาวางที่จุดเริ่มก่อน)")
    args = parser.parse_args()

    saved = None
    if args.load:
        try:
            saved = json.loads(MAP_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"อ่าน {MAP_FILE.name} ไม่ได้: {e}")
            return
        if saved.get("targets") and (saved.get("schema_version", 1) < 2
                or any(not t.get("wall_verified", False) for t in saved["targets"])):
            print("แผนที่เก่ายังไม่ได้ยืนยันผนังเป้าด้วย ToF: รันใหม่โดยไม่ใช้ --load เพื่อสำรวจใหม่ (จะเขียนทับแผนที่เดิม)")
            return
        if "start_heading" not in saved:
            print(f"{MAP_FILE.name} ไม่มีทิศเริ่มต้น (ไฟล์จากโปรแกรมเวอร์ชันเก่า) ต้องสำรวจรอบ 1 ใหม่")
            return

    print("=" * 56)
    print(" ตั้งค่าสนาม (ช่องละ 60 ซม.)  ทิศ: 0=N(^) 1=E(>) 2=S(v) 3=W(<)")
    print("=" * 56)
    if saved is not None:
        maze = MazeMap.from_json(saved)
        width, height = maze.w, maze.h
        start = (saved["start"][0], saved["start"][1])
        heading = int(saved["start_heading"])
        found = ", ".join(f"{maze.target_label(c)}@{c}" for c in sorted(maze.targets)) or "ไม่มี"
        print(f" โหลด {MAP_FILE.name}: สนาม {width}x{height} เริ่มที่ {start} ทิศ {DIR_NAMES[heading]}")
        print(f" เป้าที่รอบ 1 เจอ: {found}")
        if maze.explored:
            print(" วางหุ่นที่จุดเริ่ม หันทิศเดิม กด h ยืนยัน แล้วกด ENTER ในหน้าต่างโปรแกรมเพื่อยิง")
        else:
            print(f" รอบ 1 ยังไม่จบ (สำรวจแล้ว {len(maze.visited)} ช่อง): วางหุ่นที่จุดเริ่ม หันทิศเดิม กด h ยืนยัน แล้วกด e สำรวจต่อ")
    elif args.sim:
        print(" [SIM] ใช้สนามตัวอย่าง 4x4")
        width = height = 4
    else:
        width = ask_int("ความกว้างสนาม (จำนวนช่องแกน X)", 6, 1, 12)
        height = ask_int("ความยาวสนาม (จำนวนช่องแกน Y)", 6, 1, 12)
    if saved is None:
        known = not args.no_slam or args.sim or input("รู้ไหมว่าหุ่นเริ่มช่องไหนของสนาม? [y = รู้ / Enter = ไม่รู้]: ").strip().lower() == "y"
        if not args.no_slam:
            print(" [SLAM] ระบุช่องเริ่มต้นและทิศจริง วางหุ่นกลางช่องก่อนเริ่ม")
        if known:
            start = (ask_int("จุดเริ่ม X", 0, 0, width - 1), ask_int("จุดเริ่ม Y", 0, 0, height - 1))
            heading = ask_int("หุ่นหันหน้าไปทิศ (0-3)", 0, 0, 3)
        else:
            # ไม่รู้จุดเริ่ม: ตารางเผื่อ (2W-1)x(2H-1) ให้หุ่นอยู่กลาง สนามจริงอยู่มุมไหนก็ครอบได้หมด
            # (กำแพงรอบสนามจริง ToF จะเจอเอง) ทิศที่หุ่นหันตอนเริ่ม = ขึ้น (N) บนแผนที่
            field = (width, height)
            width, height = 2 * width - 1, 2 * height - 1
            start, heading = (width // 2, height // 2), 0
            print(f" ใช้ตารางเผื่อ {width}x{height} หุ่นเริ่มกลาง {start} ทิศที่หุ่นหันอยู่ = N บนแผนที่")
        maze = MazeMap(width, height)
        if not known:
            maze.field = field  # รู้ขนาดสนามจริง: พอสำรวจกว้าง/ยาวครบ ช่องนอกกรอบ = นอกสนาม ไม่ต้องวัด

    if not args.no_slam and maze.field is not None:
        print("แผนที่ตารางเผื่อไม่รองรับ Grid SLAM: ใช้ --no-slam หรือสำรวจใหม่โดยระบุช่องเริ่มจริง")
        return
    try:
        io = SimRobot(start, heading, width, height) if args.sim else RealRobot()
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        return
    grabber = FrameGrabber(io)
    grabber.start()
    mission = MazeMission(io, grabber, maze, start, heading, use_slam=not args.no_slam)
    if saved is not None:
        mission.restore_state(saved)
        # The saved pose can predate a fault or the robot being picked up. A
        # loaded map is only safe to use after the physical START is verified.
        mission.pose_uncertain = True
        mission.status = "Position unconfirmed"
        mission.log("Place robot at START, then CONFIRM START (h) before running")
    try:
        App(mission, grabber).run()
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        mission.stop()
        t0 = time.monotonic()
        while mission.busy and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        grabber.stop()
        io.close()
        cv2.destroyAllWindows()
        print("Closed safely")
