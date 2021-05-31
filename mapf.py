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
        return self.soc()

    def soc(self):
        return sum([len(path) for path in self.solution.values()])

    def makespan(self):
        return max([len(path) for path in self.solution.values()])

    def fuel_cost(self):
        fuel_cost = 0
        for path in self.solution.values():
            old_pos = None
            for pos in path:
                if not old_pos or old_pos != pos:
                    old_pos = pos
                    fuel_cost += 1
        return fuel_cost

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

        self.ll_nodes = 0
        self.hl_nodes = 0

    def high_level(self):
        queue = PriorityQueue()
        root = Node(depth=0)
        root.solution = self.get_solution(root)
        queue.put(root)
        self.hl_nodes += 1

        while not queue.empty():
            root = queue.get()
            conflict = self.get_first_conflict(root.solution)
            if conflict is None:
                return root.solution

            involved_agents, pos, t = conflict
            ag_1, ag_2 = involved_agents
            pos_dict = {ag_1: pos, ag_2: pos}
            if pos[0] == 'E':
                pos_dict = {ag_1: ('E', pos[1], pos[2]), ag_2: ('E', pos[2], pos[1])}

            for ag in involved_agents:
                node = Node(root.depth + 1)
                node.constraints = deepcopy(root.constraints)
                if ag not in node.constraints:
                    node.constraints[ag] = [(pos_dict[ag], t)]
                else:
                    node.constraints[ag].append((pos_dict[ag], t))
                node.solution = deepcopy(root.solution)
                node.solution.pop(ag, None)
                node.solution[ag] = self.low_level(self.agents[ag], node.constraints[ag], node.solution)
                if node.solution[ag] and node.get_cost():
                    self.hl_nodes += 1
                    queue.put(node)
        return None

    def get_solution(self, node: Node):
        solution = {}
        for agent in self.agents:
            solution[agent.id] = self.low_level(agent, node.constraints.get(agent.id, []), solution)

        return solution

    def get_first_conflict(self, solution: Dict[int, List[Tuple[int, int]]]):
        previous_reserved_cells = None
        sol = deepcopy(solution)
        n = max([len(sol[x]) for x in sol])

        for x in sol:
            while len(sol[x]) < n:
                sol[x].append(sol[x][-1])

        for t in range(n):
            reserved_cells = [[None for _ in range(self.world_width)] for _ in range(self.world_height)]

            # Vertex conflict
            for ag_id in sol:
                path = sol[ag_id]
                (x, y) = path[t]
                if reserved_cells[x][y] is None:
                    reserved_cells[x][y] = ag_id
                else:
                    return [reserved_cells[x][y], ag_id], ('V', (x, y)), t

            # Edge conflict
            if previous_reserved_cells is not None:
                for ag_id in solution:
                    path = solution[ag_id]
                    if t < len(path):
                        (x0, y0) = path[t - 1]
                        (x1, y1) = path[t]

                        ag = reserved_cells[x0][y0]
                        if ag is not None and ag != ag_id:
                            if previous_reserved_cells[x1][y1] == ag:
                                return [ag, ag_id], ('E', (x1, y1), (x0, y0)), t
            previous_reserved_cells = reserved_cells
        return None

    def low_level(self, agent: Agent, constraints: List[Tuple[Tuple[int, int], int]],
                  agent_paths: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        start, goal = agent.start_state, agent.goal_state
        count_removed_ll = 0

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

        def check_constrained(v, old_pos) -> bool:
            v_pos, v_time = v
            for cons_pos, cons_t in constraints:
                if cons_pos[0] == 'V':
                    if cons_pos[1] == v_pos and v_pos == goal and cons_t >= v_time:
                        return False
                    if cons_pos[1] == v_pos and cons_t == v_time:
                        return False
                elif cons_pos[0] == 'E':
                    pos_1, pos_2 = cons_pos[1], cons_pos[2]
                    if old_pos == pos_1 and v_pos == pos_2 and cons_t == v_time:
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
                v_pos, time = v
                if f_score[v] == min_f_score:
                    num_conflicts = 0
                    for path in agent_paths.values():
                        if (time >= len(path) and path[-1] == v_pos) or \
                                (time < len(path) and path[time] == v_pos):
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
            count_removed_ll += 1
            if current[0] == goal:
                self.ll_nodes += count_removed_ll + len(open_set)
                return reconstruct_path(current)

            current_pos, current_time = current
            successors = [(current_pos, current_time + 1)]
            for newPos in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ppos = tuple(map(operator.add, current_pos, newPos))
                neigh = (ppos, current_time + 1)
                if 0 <= ppos[0] < self.world_height and 0 <= ppos[1] < self.world_width:
                    successors.append(neigh)

            for neigh in successors:
                if check_constrained(neigh, current_pos):
                    new_g_score = g_score[current] + 1
                    if neigh not in g_score or new_g_score < g_score[neigh]:
                        predecessor[neigh] = current
                        g_score[neigh] = new_g_score
                        f_score[neigh] = g_score[neigh] + h(neigh[0])
                        if neigh not in open_set:
                            open_set.add(neigh)
        self.ll_nodes += count_removed_ll + len(open_set)
        return []


def print_solution(solution, cbs):
    sol = deepcopy(solution)
    n = max([len(sol[x]) for x in sol])

    for x in sol:
        while len(sol[x]) < n:
            sol[x].append(sol[x][-1])

    for t in range(n):
        world = [[' ' for _ in range(cbs.world_width)] for _ in range(cbs.world_height)]
        for ag_id in sol:
            path = sol[ag_id]
            (x, y) = path[t]
            world[x][y] = str(ag_id)

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
    agents = [Agent(0, (0, 1), (3, 2)), Agent(1, (1, 0), (2, 3))]
    cbs = CBS(4, 4, agents)
    solution = cbs.high_level()
    print(solution)
    print_solution(solution, cbs)

