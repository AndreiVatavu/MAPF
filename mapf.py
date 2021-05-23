from __future__ import annotations

import math
import operator
from copy import deepcopy
from queue import PriorityQueue
from typing import Tuple, List, Dict
from time import sleep


class Node:

    def __init__(self, depth, parent=None, left=None, right=None):
        self.depth = depth
        self.parent = parent
        self.left = left
        self.right = right

        self.constraints = {}
        self.solution = {}
        self.cost = None

    def add_child(self, node: Node):
        if self.left is None:
            self.left = node
        elif self.right is None:
            self.right = node
        else:
            raise RuntimeError("This node already has 2 children")

    def get_cost(self):
        if self.cost is None:
            self.cost = self.sic()

        return self.cost

    def sic(self):
        return 0

    def __lt__(self, other: Node):
        return self.get_cost() < other.get_cost()

class Agent:

    def __init__(self, identifier, start_state, goal_state):
        self.id = identifier
        self.start_state = start_state
        self.goal_state = goal_state

    def __hash__(self):
        return self.id


class CBS:

    def __init__(self, world_width, world_height, agents: [Agent]):
        self.world_width = world_width
        self.world_height = world_height

        self.agents = agents

    def high_level(self):
        queue = PriorityQueue()
        root = Node(depth=0)
        root.solution = self.get_solution(root)
        queue.put(root)

        while not queue.empty():
            print("Step")
            node = queue.get()
            conflict = self.get_first_conflict(node.solution)
            if conflict is None:
                return node.solution

            involved_agents, pos, t = conflict
            for ag in involved_agents:
                node = Node(root.depth + 1)
                node.constraints = deepcopy(root.constraints)
                if ag not in node.constraints:
                    node.constraints[ag] = [(pos, t)]
                else:
                    node.constraints[ag].append((pos, t))
                node.solution = deepcopy(root.solution)
                node.solution.pop(ag, None)
                node.solution[ag] = self.low_level(self.agents[ag], node.constraints[ag], node.solution)
                if node.solution[ag] is not None and node.get_cost() is not None:
                    queue.put(node)

    def get_solution(self, node: Node):
        solution = {}
        for agent in self.agents:
            print("Ag")
            solution[agent.id] = self.low_level(agent, node.constraints.get(agent.id, []), {})

        return solution

    def get_first_conflict(self, solution: Dict[int, List[Tuple[int, int]]]):
        t = 0
        previous_reserved_cells = None
        while True:
            sw = False
            reserved_cells = [[None for _ in range(self.world_width)] for _ in range(self.world_height)]

            # Vertex conflict
            for ag_id in solution:
                path = solution[ag_id]
                if t < len(path):
                    (x, y) = path[t]
                    sw = True
                    if reserved_cells[x][y] is None:
                        reserved_cells[x][y] = ag_id
                    else:
                        return [reserved_cells[x][y], ag_id], (x, y), t
            if not sw:
                break

            # Edge conflict
            if previous_reserved_cells is not None:
                for ag_id in solution:
                    path = solution[ag_id]
                    if t < len(path):
                        (x0, y0) = path[t - 1]
                        (x1, y1) = path[t]

                        ag = reserved_cells[x0][y0]
                        if ag is not None:
                            if previous_reserved_cells[x1][y1] == ag:
                                return [ag, ag_id], (x1, y1), t

            t += 1
            previous_reserved_cells = reserved_cells

        return None

    def is_valid(self, solution):
        pass

    def low_level(self, agent: Agent, constraints: List[Tuple[Tuple[int, int], int]],
                  agent_paths: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        start, goal = agent.start_state, agent.goal_state

        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        def reconstruct_path(crt) -> List[Tuple[int, int]]:
            pos, time = crt
            total_path = [pos]
            while crt in predecessor:
                crt = predecessor[crt]
                pos, time = crt
                total_path.append(pos)
            total_path.reverse()
            return total_path

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
            if current[0] == goal:
                return reconstruct_path(current)

            current_pos, current_time = current
            successors = [(current_pos, current_time + 1)]
            for newPos in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ppos = tuple(map(operator.add, current_pos, newPos))
                neigh = (ppos, current_time + 1)
                if 0 <= ppos[0] < self.world_height and 0 <= ppos[1] <= self.world_width:
                    successors.append(neigh)

            for neigh in successors:
                if check_constrained(neigh):
                    new_g_score = g_score[current] + 1
                    if neigh not in g_score or new_g_score < g_score[neigh]:
                        predecessor[neigh] = current
                        g_score[neigh] = new_g_score
                        f_score[neigh] = g_score[neigh] + h(neigh[0])
                        if neigh not in open_set:
                            open_set.add(neigh)
        return []


def print_solution(solution, cbs):
    t = 0
    while True:
        sw = False
        world = [[' ' for _ in range(cbs.world_width)] for _ in range(cbs.world_height)]

        for ag_id in solution:
            path = solution[ag_id]
            if t < len(path):
                (x, y) = path[t]
                sw = True
                world[x][y] = str(ag_id)
        t += 1
        if not sw:
            break

        for line in world:
            print('+-' * cbs.world_width + '+')
            print('|', end='')
            for x in line:
                print(x + '|', end='')
            print()
        print('+-' * cbs.world_width + '+')
        print()
        sleep(0.5)


if __name__ == '__main__':
    agents = [Agent(0, (0, 0), (3, 3)), Agent(1, (3, 0), (0, 3))]
    cbs = CBS(4, 4, agents)
    solution = cbs.high_level()
    print(solution)
    print_solution(solution, cbs)

