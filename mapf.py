import math
from typing import Tuple, List, Dict


class Node:

    def __init__(self, depth, parent=None, left=None, right=None):
        self.depth = depth
        self.parent = parent
        self.left = left
        self.right = right

        self.constraints = []
        self.solution = []
        self.cost = 0


class MAPF:

    def __init__(self, world_width, world_height, agents):
        self.world_width = world_width
        self.world_height = world_height

        self.agents = agents

    def low_level(self, agent_id: int, constraints: List[Tuple[Tuple[int, int], int]],
                  agent_paths: Dict[int, List[Tuple[int, int]]]):
        start, goal = self.agents[agent_id]

        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        def reconstruct_path(crt):
            old_pos, old_time = crt
            total_path = []
            while crt in predecessor:
                crt = predecessor[crt]
                pos, time = crt
                total_path.extend([old_pos] * (time - old_time))
                old_pos, old_time = pos, time
            total_path.append(old_pos)
            return total_path.reverse()

        def check_constrained(v) -> bool:
            v_pos, v_time = v
            for cons_pos, cons_t in constraints:
                if cons_pos == v_pos and cons_t == v_time:
                    return False
            return True

        def select_lowest():
            min_f_score = math.inf
            for v in open_set:
                if f_score[v] < min_f_score:
                    min_f_score = f_score[v]

            min_v = None
            min_conflicts = math.inf
            # Standley’s tie - breaking conflict avoidance table(CAT)
            for v in open_set:
                _, time = v
                if f_score[v] == min_f_score:
                    num_conflicts = 0
                    for path in agent_paths.values():
                        if path[time] == v:
                            num_conflicts += 1
                    if num_conflicts < min_conflicts:
                        min_conflicts = num_conflicts
                        min_v = v
            return min_v

        open_set = {(start, 0)}
        predecessor = {}
        g_score = {(start, 0): 0}
        f_score = {(start, 0): h(start)}

        while open_set:
            current = select_lowest()

            open_set.remove(current)
            if current == goal:
                return reconstruct_path(current)

            current_pos, current_time = current
            successors = [(current_pos, current_time + 1)]
            for newPos in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neigh = (current_pos + newPos, current_time + 1)
                if 0 <= neigh[0][0] < self.world_height and 0 <= neigh[0][1] <= self.world_width:
                    successors.append(neigh)

            for neigh in successors:
                if check_constrained(neigh):
                    new_g_score = g_score[current] + 1
                    if neigh not in g_score or new_g_score < g_score[neigh]:
                        predecessor[neigh] = current
                        g_score[neigh] = new_g_score
                        f_score[neigh] = g_score[neigh] + h(neigh)
                        if neigh not in open_set:
                            open_set.add(neigh)
        return False

    def high_level(self):
        pass

    def is_valid(self, solution):
        pass
