from queue import PriorityQueue


class Node:

    def __init__(self, depth, parent=None, left=None, right=None):
        self.depth = depth
        self.parent = parent
        self.left = left
        self.right = right

        self.constraints = []
        self.solution = []
        self.cost = None

    def get_cost(self):
        if self.cost is None:
            self.cost = self.sic()

        return self.cost

    def sic(self):
        return 0


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
        queue.put((0, root))

        while not queue.empty():
            node = queue.get()


    def get_solution(self, node: Node):
        solution = []
        for agent in self.agents:
            solution.append(self.low_level(agent, node.constraints))

        return solution

    def is_valid(self, solution):
        pass

    def low_level(self, agent, constraints):
        pass
