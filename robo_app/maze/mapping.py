"""maze: mapping."""

from .settings import CAMERA_FORWARD_M, Cell, DIRS, SHAPE_SHORT, TARGET_CELL_COST, TARGET_DEPTH_DRIFT_M, TARGET_TRACK_M
from collections import deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import heapq
import math

# ==========================================
# 2. แผนที่เขาวงกต
# ==========================================
class MazeMap:
    """ แผนที่ W x H ช่อง
    edges  : กำแพงระหว่างช่องที่ติดกัน  -1 = กั้น (กำแพงบางหรือบล็อกทึบ), 1 = เปิด, ไม่มีคีย์ = ยังไม่รู้
    targets: ช่องที่มีเป้า -> สี, target_shape: -> รูปทรง, target_wall: -> ทิศของผนังที่เป้าตั้งชิด
             ยิงจากช่องเป้าที่หุ่นอยู่ หันหน้าเข้าหาผนัง / ขับผ่านช่องเป้าได้แต่เส้นทางจะอ้อมก่อน
    ช่องที่สำรวจครบแล้วยังไปไม่ถึง = บล็อกทึบ/ห้องปิด
    """

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.edges: Dict[Tuple[Cell, Cell], int] = {}
        self.visited: Set[Cell] = set()
        self.targets: Dict[Cell, str] = {}
        self.target_shape: Dict[Cell, str] = {}
        self.target_wall: Dict[Cell, int] = {}
        self.target_seen: Dict[Cell, Tuple[float, float]] = {}   # (ระยะ, เยื้อง) ที่เห็นจากช่องทางเข้า ใช้ตั้งหัวรอบ 2
        self.target_plates: Dict[Cell, List[dict]] = {}
        self.field: Optional[Tuple[int, int]] = None               # ขนาดสนามจริง เมื่อใช้ตารางเผื่อ (ไม่รู้จุดเริ่ม)
        self.explored = False
        for x in range(width):  # ขอบนอกสนามเป็นกำแพงเสมอ
            self.set_edge((x, 0), (x, -1), -1)
            self.set_edge((x, height - 1), (x, height), -1)
        for y in range(height):
            self.set_edge((0, y), (-1, y), -1)
            self.set_edge((width - 1, y), (width, y), -1)

    @staticmethod
    def _key(a: Cell, b: Cell) -> Tuple[Cell, Cell]:
        return (a, b) if a <= b else (b, a)

    def inside(self, c: Cell) -> bool:
        return 0 <= c[0] < self.w and 0 <= c[1] < self.h

    def edge(self, a: Cell, b: Cell) -> int:
        return self.edges.get(self._key(a, b), 0)

    def set_edge(self, a: Cell, b: Cell, status: int):
        self.edges[self._key(a, b)] = status

    @staticmethod
    def step(c: Cell, d: int) -> Cell:
        return c[0] + DIRS[d][0], c[1] + DIRS[d][1]

    def can_drive(self, a: Cell, b: Cell) -> bool:
        return self.inside(b) and self.edge(a, b) == 1

    def add_target(self, cell: Cell, color: str, shape: str, wall: Optional[int],
                   seen: Optional[Tuple[float, float]] = None, wall_verified: bool = False,
                   shot: bool = False, near_seen: Optional[Tuple[float, float]] = None,
                   shots_fired: int = 0) -> bool:
        """Record each plate separately. Only sensor measurements create walls."""
        if not self.inside(cell):
            return False
        if wall is not None and self.edge(cell, self.step(cell, wall)) == 1:
            return False
        plates = self.target_plates.setdefault(cell, [])
        matches = []
        for plate in plates:
            if plate['color'] != color or plate['wall'] != wall:
                continue
            old = plate['seen']
            old_view = plate.get('near_seen') or old
            new_view = near_seen or seen
            old_angle = (math.atan2(old_view[1], max(0.05, old_view[0] - CAMERA_FORWARD_M))
                         if old_view is not None else None)
            new_angle = (math.atan2(new_view[1], max(0.05, new_view[0] - CAMERA_FORWARD_M))
                         if new_view is not None else None)
            if (old is None and seen is None) or (old is not None and seen is not None
                    and abs(old[0] - seen[0]) <= TARGET_DEPTH_DRIFT_M
                    and (abs(old[1] - seen[1]) <= TARGET_TRACK_M
                         or abs(old_angle - new_angle) <= math.radians(3))):
                matches.append(plate)
        data = {'color': color, 'shape': shape, 'wall': wall,
                'seen': tuple(seen) if seen is not None else None,
                'near_seen': tuple(near_seen) if near_seen is not None else None,
                'wall_verified': wall_verified, 'shot': bool(shot),
                'shots_fired': max(0, int(shots_fired))}
        if not matches:
            plates.append(data)
        else:
            match = matches[0]
            data["shot"] = data["shot"] or any(p.get("shot", False) for p in matches)
            data["wall_verified"] = data["wall_verified"] or any(
                p.get("wall_verified", False) for p in matches)
            data["near_seen"] = data["near_seen"] or next(
                (p.get("near_seen") for p in matches if p.get("near_seen") is not None), None)
            data["shots_fired"] = max([data["shots_fired"]] + [p.get("shots_fired", 0) for p in matches])
            match.update(data)
            for duplicate in matches[1:]:
                plates.remove(duplicate)
        # Representative dictionaries remain available to older imports/UI.
        first = plates[0]
        self.targets[cell] = first['color']
        self.target_shape[cell] = first['shape']
        if first['wall'] is not None:
            self.target_wall[cell] = first['wall']
        if first['seen'] is not None:
            self.target_seen[cell] = first['seen']
        return True

    def plates_at(self, cell: Cell) -> List[dict]:
        return self.target_plates.get(cell, [])

    @property
    def plate_count(self):
        return sum(len(plates) for plates in self.target_plates.values())

    def outside_field(self, c: Cell) -> bool:
        """ ตารางเผื่อ (ไม่รู้จุดเริ่ม): พอช่องที่รู้แล้วกินความกว้าง/ยาวครบขนาดสนามจริง ช่องนอกกรอบนั้น = นอกสนาม """
        if self.field is None:
            return False
        cells = self.visited | set(self.targets)
        if not cells:
            return False
        xs, ys = [x for x, _ in cells], [y for _, y in cells]
        if max(xs) - min(xs) + 1 >= self.field[0] and not min(xs) <= c[0] <= max(xs):
            return True
        return max(ys) - min(ys) + 1 >= self.field[1] and not min(ys) <= c[1] <= max(ys)

    def target_label(self, cell: Cell, plate_index: Optional[int] = None) -> str:
        plates = self.plates_at(cell)
        chosen = plates if plate_index is None else [plates[plate_index]]
        return " / ".join(f"{p['color'].upper()} {SHAPE_SHORT.get(p['shape'], '?')}" for p in chosen)

    def unknown_dirs(self, c: Cell) -> List[int]:
        """ ทิศที่ยังไม่รู้ว่ามีกำแพงไหม (ขอบสนามรู้อยู่แล้ว) """
        return [d for d in range(4) if self.edge(c, self.step(c, d)) == 0]

    def shooting_spots(self, target: Cell, plate_index: Optional[int] = None) -> List[Tuple[Cell, int]]:
        plates = self.plates_at(target)
        chosen = plates if plate_index is None else [plates[plate_index]]
        spots = []
        for plate in chosen:
            wall = plate['wall']
            if wall is not None:
                spots.append((target, wall))
        return list(dict.fromkeys(spots))

    # ---------- วางเส้นทาง (A* + ค่าปรับการเลี้ยว แบบ assignment2.py) ----------
    def plan(self, start: Cell, heading: int, goal_test, heuristic_to: Optional[Cell] = None
             ) -> Optional[List[Cell]]:
        pq = [(0.0, 0.0, start, heading, [start])]
        seen = set()
        while pq:
            _, cost, cell, head, path = heapq.heappop(pq)
            if cell != start and goal_test(cell):
                return path
            if (cell, head) in seen:
                continue
            seen.add((cell, head))
            for d in range(4):
                nxt = self.step(cell, d)
                if not self.can_drive(cell, nxt):
                    continue
                turn = (d - head) % 4
                new_cost = cost + 1.0 + (0.0 if turn == 0 else 1.5 if turn == 2 else 0.8)
                if nxt in self.targets:
                    new_cost += TARGET_CELL_COST
                h = abs(heuristic_to[0] - nxt[0]) + abs(heuristic_to[1] - nxt[1]) if heuristic_to else 0
                heapq.heappush(pq, (new_cost + h, new_cost, nxt, d, path + [nxt]))
        return None

    def path_to(self, start: Cell, heading: int, goal: Cell) -> Optional[List[Cell]]:
        if start == goal:
            return [start]
        return self.plan(start, heading, lambda c: c == goal, goal)

    def path_to_unvisited(self, start: Cell, heading: int,
                          preferred_targets: Optional[Set[Cell]] = None) -> Optional[List[Cell]]:
        """Round 1 Dijkstra over (cell, heading), including the cost of turns.

        Prefer nearby unfinished targets without taking a long detour. Only
        traverse confirmed open edges; a new scan replans from the next cell.
        Round 2 continues to use A* in path_to().
        """
        origin = (start, heading)
        distances = {origin: 0.0}
        parents = {origin: None}
        queue = [(0.0, 0, origin)]
        sequence = 0
        choice = None
        while queue:
            cost, _, state = heapq.heappop(queue)
            if cost > distances[state]:
                continue
            cell, head = state
            preferred = bool(preferred_targets and cell in preferred_targets)
            if cell != start and (preferred or cell not in self.visited):
                # A one-cell bonus makes a matching target attractive when it
                # is nearby, while the travel cost still controls long trips.
                score = cost - (1.0 if preferred else 0.0)
                # In map-only mode, inspect an empty cell first on near ties.
                is_other_target = cell in self.targets and not preferred
                if is_other_target:
                    score += 0.8
                candidate = (score, 0 if preferred else 2 if is_other_target else 1,
                             cost, cell, head)
                if choice is None or candidate < choice:
                    choice = candidate
            for d in (head, (head + 1) % 4, (head + 3) % 4, (head + 2) % 4):
                nxt = self.step(cell, d)
                if not self.can_drive(cell, nxt):
                    continue
                turn = (d - head) % 4
                new_cost = cost + 1.0 + (0.0 if turn == 0 else 1.5 if turn == 2 else 0.8)
                next_state = (nxt, d)
                if new_cost < distances.get(next_state, float('inf')):
                    distances[next_state] = new_cost
                    parents[next_state] = state
                    sequence += 1
                    heapq.heappush(queue, (new_cost, sequence, next_state))

        if choice is None:
            return None
        path = []
        state = (choice[3], choice[4])
        while state is not None:
            path.append(state[0])
            state = parents[state]
        return path[::-1]

    def distances_from(self, start: Cell) -> Dict[Cell, int]:
        """ จำนวนก้าวจาก start ไปทุกช่องที่ไปถึงได้ (ใช้จัดลำดับเป้า) """
        dist = {start: 0}
        queue = deque([start])
        while queue:
            c = queue.popleft()
            for d in range(4):
                n = self.step(c, d)
                if n not in dist and self.can_drive(c, n):
                    dist[n] = dist[c] + 1
                    queue.append(n)
        return dist

    # ---------- บันทึก ----------
    def to_json(self) -> dict:
        return {
            "schema_version": 2,
            "width": self.w, "height": self.h, "explored": self.explored,
            "visited": sorted(self.visited),
            "walls": [[list(a), list(b)] for (a, b), s in sorted(self.edges.items()) if s == -1],
            "open": [[list(a), list(b)] for (a, b), s in sorted(self.edges.items()) if s == 1],
            "targets": [{"cell": list(c), **plate}
                        for c, plates in sorted(self.target_plates.items()) for plate in plates],
            "field": self.field,
        }

    @classmethod
    def from_json(cls, data: dict) -> "MazeMap":
        """ สร้างแผนที่กลับจากไฟล์ที่รอบ 1 บันทึกไว้ (ใช้ตอนรันรอบ 2 เป็นโปรแกรมใหม่) """
        maze = cls(int(data["width"]), int(data["height"]))
        for status, key in ((-1, "walls"), (1, "open")):
            for a, b in data.get(key, []):
                maze.set_edge((a[0], a[1]), (b[0], b[1]), status)
        maze.visited = {(c[0], c[1]) for c in data.get("visited", [])}
        for t in data.get("targets", []):
            seen = t.get("seen")
            maze.add_target((t["cell"][0], t["cell"][1]), t["color"], t.get("shape", "?"), t.get("wall"),
                            (seen[0], seen[1]) if seen else None,
                            wall_verified=bool(t.get("wall_verified", False)), shot=bool(t.get("shot", False)),
                            near_seen=tuple(t["near_seen"]) if t.get("near_seen") else None,
                            shots_fired=t.get("shots_fired", 0))
        if data.get("field"):
            maze.field = (data["field"][0], data["field"][1])
        maze.explored = bool(data.get("explored", False))
        return maze
