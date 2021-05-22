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

    def low_level(self, agent_id):
        pass

    def high_level(self):
        pass

    def is_valid(self, solution):
        pass
