import sys
import re
from collections import OrderedDict


class Maze:
    # Attributes to store maze information
    size = []  # [rows, columns]
    verticalWalls = [[]]
    horizantalWalls = [[]]
    traps = [[]]
    start = []  # [start_row, start_column]
    goals = []  # List of goal coordinates [goal1_row, goal1_column, goal2_row, goal2_column, ...]

    def __init__(self):
        # Initialize the maze by reading from the file
        self.read_maze()

    def read_maze(self):
        # Open the maze file for reading
        file = open("maze.txt", "r")

        # Read the first line and remove the newline character at the end
        line = file.readline().rstrip("\n\r")
        counter = 0

        # Loop to read different sections of the maze
        while counter < 2:
            if not line:
                counter += 1
            else:
                counter = 0

            # Check headings to determine the type of information to read
            if line == "size":
                # Read the following two lines for rows and columns
                xSize = file.readline().rstrip("\n\r")
                ySize = file.readline().rstrip("\n\r")
                self.Size(xSize, ySize)
            elif line == "walls":
                walls = []
                line = file.readline().rstrip("\n\r")
                # Read until a blank line to get information
                while line:
                    walls.append(line)
                    line = file.readline().rstrip("\n\r")
                self.Walls(walls)
            elif line == "traps":
                traps = []
                line = file.readline().rstrip("\n\r")
                while line:
                    traps.append(line)
                    line = file.readline().rstrip("\n\r")
                self.Traps(traps)
            elif line == "start":
                start = file.readline().rstrip("\n\r")
                self.Start(start)
            elif line == "goals":
                goals = []
                line = file.readline().rstrip("\n\r")
                while line:
                    goals.append(line)
                    line = file.readline().rstrip("\n\r")
                self.Goals(goals)

            # Read the next line for the next iteration
            line = file.readline().rstrip("\n\r")

        # Close the file after reading
        file.close()

    def Size(self, x, y):
        # Set the row count
        if "rows" in x:
            self.size.append(int(re.sub("[^0-9]", "", x)))
        elif "rows" in y:
            self.size.append(int(re.sub("[^0-9]", "", y)))

        # Set the column count
        if "columns" in x:
            self.size.append(int(re.sub("[^0-9]", "", x)))
        elif "columns" in y:
            self.size.append(int(re.sub("[^0-9]", "", y)))

        # Initialize wall and trap arrays with zeros
        self.verticalWalls = [[0 for i in range(self.size[1] - 1)] for i in range(self.size[0])]
        self.horizantalWalls = [[0 for i in range(self.size[1])] for i in range(self.size[0] - 1)]
        self.traps = [[0 for i in range(self.size[1])] for i in range(self.size[0])]

    def Walls(self, walls):
        # Process the information about walls in the maze
        walls_length = len(walls)

        for i in range(walls_length):
            # Process each line to identify row or column walls and update the maze
            if "row" in walls[i]:
                row_index = int(re.sub("[^0-9]", "", walls[i]))
                column_indexes = walls[i + 1].split()
                for index in column_indexes:
                    self.verticalWalls[row_index - 1][int(index) - 1] = 1
            elif "column" in walls[i]:
                column_index = int(re.sub("[^0-9]", "", walls[i]))
                row_indexes = walls[i + 1].split()
                for index in row_indexes:
                    self.horizantalWalls[int(index) - 1][column_index - 1] = 1

    def Traps(self, traps):
        # Process the information about traps in the maze
        for trap in traps:
            # Convert string coordinates to integers and update the maze
            indexes = list(map(int, trap.split()))
            self.traps[indexes[0] - 1][indexes[1] - 1] = 1

    def Start(self, start):
        # Process the starting point information in the maze
        indexes = list(map(int, start.split()))
        self.start.append(indexes[0] - 1)
        self.start.append(indexes[1] - 1)

    def Goals(self, goals):
        # Process the goal information in the maze
        for goal in goals:
            # Convert string coordinates to integers and update the maze
            indexes = list(map(int, goal.split()))
            indexes = list(map(lambda x: x - 1, indexes))
            self.goals.append(indexes)

    def blocked(self, row, column, direction):
        # Check if the path is blocked in the specified direction
        if direction == "north":
            if row == 0:
                return False
            return self.horizantalWalls[row - 1][column] == 0
        elif direction == "east":
            if column == (self.size[1] - 1):
                return False
            return self.verticalWalls[row][column] == 0
        elif direction == "west":
            if column == 0:
                return False
            return self.verticalWalls[row][column - 1] == 0
        elif direction == "south":
            if row == (self.size[0] - 1):
                return False
            return self.horizantalWalls[row][column] == 0


mazeOutput = Maze()

class Node:
    def __init__(self):
        # Initialize node attributes
        self.x = 0
        self.y = 0
        self.north = None
        self.east = None
        self.west = None
        self.south = None
        self.cost = 0
        self.heuristic = 0
        self.parent = None

    def check_equality(self, x, y):
        return x == self.x and y == self.y

    def __str__(self):
        # String representation of the node
        return "[" + str(self.x) + ", " + str(self.y) + "]"


class Graph:

    nodes = []  # Keep all nodes in a list to prevent duplicate nodes.
    maze = None

    def __init__(self):
        # Initialize the graph
        self.maze = mazeOutput
        self.root = self.createNode(self.maze.start[0], self.maze.start[1])

        # Find the maximum depth
        self.maximum_depth = self.findMaxDepth() - 1

        # Create heuristic
        self.putHeuristic()

        # Set the cost of the root node to 0, as it is the starting point
        self.root.cost = 0

    def createNode(self, x, y):
        # Create a new node and add it to the list
        node = Node()

        # Initialize node's coordinates
        node.x = x
        node.y = y

        # Add the node to the nodes list
        self.nodes.append(node)

        # Set the cost to 7 if it is a trap square, otherwise set it to 1
        if self.maze.traps[node.x][node.y] == 1:
            node.cost = 7
        else:
            node.cost = 1

        # Set all child nodes
        if self.maze.blocked(node.x, node.y, "north"):
            node.north = self.ifNodeExists(node.x - 1, node.y)
            if node.north is None:
                node.north = self.createNode(node.x - 1, node.y)
                node.north.parent = node
        if self.maze.blocked(node.x, node.y, "east"):
            node.east = self.ifNodeExists(node.x, node.y + 1)
            if node.east is None:
                node.east = self.createNode(node.x, node.y + 1)
                node.east.parent = node
        if self.maze.blocked(node.x, node.y, "west"):
            node.west = self.ifNodeExists(node.x, node.y - 1)
            if node.west is None:
                node.west = self.createNode(node.x, node.y - 1)
                node.west.parent = node
        if self.maze.blocked(node.x, node.y, "south"):
            node.south = self.ifNodeExists(node.x + 1, node.y)
            if node.south is None:
                node.south = self.createNode(node.x + 1, node.y)
                node.south.parent = node

        return node

    def ifNodeExists(self, x, y):
        # Check if a node with given coordinates already exists
        for node in self.nodes:
            if node.check_equality(x, y):
                return node
        return None

    def findMaxDepth(self):
        # Find the maximum depth in the graph
        MaxDepth = 0

        for node in self.nodes:
            current_node = node
            local_depth = 0
            while current_node is not None:
                current_node = current_node.parent
                local_depth += 1

            # Set maximum_depth to the greater of the two
            MaxDepth = max(MaxDepth, local_depth)

        return MaxDepth

    def nodeCost(self, x, y):
        # Get the cost of a node with given coordinates
        for node in self.nodes:
            if node.check_equality(x, y):
                return node.cost
        return 0

    def clearParents(self):
        # Clear the parent references for all nodes
        for node in self.nodes:
            node.parent = None

    def putHeuristic(self):
        # Create a heuristic for each node
        for node in self.nodes:
            # Select the minimum distance to the closest goal
            total_cost = sys.maxsize
            for goal in self.maze.goals:
                cost = 0
                vertical_distance = goal[1] - node.y
                horizontal_distance = goal[0] - node.x

                # Add the cost of each node until reaching the goal state
                x = 0
                y = 0
                while vertical_distance > 0:
                    y += 1
                    cost += self.nodeCost(node.x, node.y + y)
                    vertical_distance -= 1
                while horizontal_distance > 0:
                    x += 1
                    cost += self.nodeCost(node.x + x, node.y + y)
                    horizontal_distance -= 1
                while vertical_distance < 0:
                    y -= 1
                    cost += self.nodeCost(node.x + x, node.y + y)
                    vertical_distance += 1
                while horizontal_distance < 0:
                    x -= 1
                    cost += self.nodeCost(node.x + x, node.y + y)
                    horizontal_distance += 1

                # Select the minimum heuristic
                total_cost = min(total_cost, cost)

            # Assign the total cost as the node's heuristic
            node.heuristic = total_cost


graph = Graph()
frontier = []
visited = OrderedDict()  # To prevent duplicates, we use OrderedDict

def dfs():
    algorithm = "DFS:"
    # Initialize variables
    pop_index = 0
    goal_state = None
    solution_cost = 0
    solution = []
    expanded_nodes = []
    iteration = -1

    # Execute
    while goal_state is None and iteration <= graph.maximum_depth:
        # For each iteration, increase it by one, clear frontier and visited, and append the root node.
        iteration += 1
        frontier.clear()
        visited.clear()
        frontier.append(graph.root)

        while len(frontier) > 0:
            # If DFS or IDS, remove the last node from the frontier.
            # If BFS, remove the first node from the frontier.
            pop_index = len(frontier) - 1

            # Remove the correct node from the frontier according to the algorithm and add it to the visited.
            current_node = frontier.pop(pop_index)
            visited[current_node] = None

            # Stop if we are in a goal state...
            if isGoal(current_node):
                goal_state = current_node
                break
            else:
                addToFrontier(current_node, algorithm)

        # Add all visited nodes to expanded nodes before clearing it.
        for node in visited:
            expanded_nodes.append(node)

        # Continue only if this is an IDS search...
        if "IDS" not in algorithm:
            break

    # Check if  successful...
    if goal_state is None:
        print("No goal state found.")
        return

    # Calculate the cost of the solution and get the solution itself...
    current = goal_state
    while current is not None:
        solution_cost += current.cost
        solution.insert(0, current)
        # Get the parent node and continue...
        current = current.parent

    # Print the results...
    results(algorithm, solution_cost, solution, expanded_nodes)


def bfs():
    graph.clearParents()
    algorithm = "BFS:"
    # Initialize variables
    pop_index = 0
    goal_state = None
    solution_cost = 0
    solution = []
    expanded_nodes = []
    iteration = -1

    # Execute
    while goal_state is None and iteration <= graph.maximum_depth:
        # For each iteration, increase it by one, clear frontier and visited, and append the root node.
        iteration += 1
        frontier.clear()
        visited.clear()
        frontier.append(graph.root)

        while len(frontier) > 0:
            # Remove the correct node from the frontier according to the algorithm and add it to the visited.
            current_node = frontier.pop(pop_index)
            visited[current_node] = None

            # Stop if we are in a goal state...
            if isGoal(current_node):
                goal_state = current_node
                break
            else:
                addToFrontier(current_node, algorithm)

        # Add all visited nodes to expanded nodes before clearing it.
        for node in visited:
            expanded_nodes.append(node)

        # Continue only if this is an IDS search...
        if "IDS" not in algorithm:
            break

    # Check if successful...
    if goal_state is None:
        print("No goal state found.")
        return

    # Calculate the cost of the solution and get the solution itself...
    current = goal_state
    while current is not None:
        solution_cost += current.cost
        solution.insert(0, current)
        # Get the parent node and continue...
        current = current.parent

    # Print the results...
    results(algorithm, solution_cost, solution, expanded_nodes)


def ids():
    graph.clearParents()
    algorithm = "IDS:"
    # Initialize variables
    pop_index = 0
    goal_state = None
    solution_cost = 0
    solution = []
    expanded_nodes = []
    iteration = -1

    # Execute
    while goal_state is None and iteration <= graph.maximum_depth:
        # For each iteration, increase it by one, clear frontier and visited, and append the root node.
        iteration += 1
        frontier.clear()
        visited.clear()
        frontier.append(graph.root)

        # If IDS, add iteration number...
        if "IDS" in algorithm:
            expanded_nodes.append("Iteration " + str(iteration) + ":")

        while len(frontier) > 0:
            # Remove the correct node from the frontier according to the algorithm and add it to the visited.
            current_node = frontier.pop(pop_index)
            visited[current_node] = None

            # Stop if we are in a goal state...
            if isGoal(current_node):
                goal_state = current_node
                break

            # Add all child nodes of the current element to the end of the list...
            # If IDS, add child nodes according to the iteration number.
            if "IDS" in algorithm:
                parent = current_node
                for i in range(iteration):
                    # If parent is not none, iterate to upper parent.
                    parent = parent if parent is None else parent.parent

                if parent is None:
                    addToFrontier(current_node, "DFS")
            # Else, add all child nodes.
            else:
                addToFrontier(current_node, algorithm)

        # Add all visited nodes to expanded nodes before clearing it.
        for node in visited:
            expanded_nodes.append(node)

        # Continue only if this is an IDS search...
        if "IDS" not in algorithm:
            break

    # Check if successful...
    if goal_state is None:
        print("No goal state found.")
        return

    # Calculate the cost of the solution and get the solution itself...
    current = goal_state
    while current is not None:
        solution_cost += current.cost
        solution.insert(0, current)
        # Get the parent node and continue...
        current = current.parent

    # Print the results...
    results(algorithm, solution_cost, solution, expanded_nodes)


def ucs():
    graph.clearParents()
    algorithm = "UCS:"
    # Initialize variables
    pop_index = 0
    goal_state = None
    solution_cost = 0
    solution = []
    expanded_nodes = []
    iteration = -1

    # Execute
    while goal_state is None and iteration <= graph.maximum_depth:
        # For each iteration, increase it by one, clear frontier and visited, and append the root node.
        iteration += 1
        frontier.clear()
        visited.clear()
        frontier.append(graph.root)

        while len(frontier) > 0:
            sort_frontier(return_cost)

            # Remove the correct node from the frontier according to the algorithm and add it to the visited.
            current_node = frontier.pop(pop_index)
            visited[current_node] = None

            # Stop if we are in a goal state...
            if isGoal(current_node):
                goal_state = current_node
                break

            # add all child nodes.
            else:
                addToFrontier(current_node, algorithm)

        # Add all visited nodes to expanded nodes before clearing it.
        for node in visited:
            expanded_nodes.append(node)

        # Continue only if this is an IDS search...
        if "IDS" not in algorithm:
            break

    # Check if successful...
    if goal_state is None:
        print("No goal state found.")
        return

    # Calculate the cost of the solution and get the solution itself...
    current = goal_state
    while current is not None:
        solution_cost += current.cost
        solution.insert(0, current)
        # Get the parent node and continue...
        current = current.parent

    # Print the results...
    results(algorithm, solution_cost, solution, expanded_nodes)


def gbfs():
    graph.clearParents()
    algorithm = "GBFS:"
    sort_by = return_heuristic
    # Initialize variables
    goal_state = None
    solution_cost = 0
    solution = []

    # Clear frontier and visited, then add the root element to the frontier.
    frontier.clear()
    visited.clear()
    frontier.append(graph.root)

    while len(frontier) > 0:
        # Sort the frontier according to heuristic...
        sort_frontier(sort_by)

        # Remove the correct node from the frontier and add it to the visited.
        current_node = frontier.pop(0)
        visited[current_node] = None

        # Stop if we are in a goal state...
        if isGoal(current_node):
            goal_state = current_node
            break

        # Add to the frontier as in BFS.
        addToFrontier(current_node, "BFS")

    # Check if successful...
    if goal_state is not None:
        # Calculate the cost of the solution and get the solution itself...
        current = goal_state
        while current is not None:
            solution_cost += current.cost
            solution.insert(0, current)
            # Get the parent node and continue...
            current = current.parent

        # Print the results...
        results(algorithm, solution_cost, solution, visited)
    else:
        print("No goal state found.")


def astar():
    graph.clearParents()
    algorithm = "A*:"
    sort_by = return_cost_and_heuristic
    # Initialize variables
    goal_state = None
    solution_cost = 0
    solution = []

    # Clear frontier and visited, then add the root element to the frontier.
    frontier.clear()
    visited.clear()
    frontier.append(graph.root)

    while len(frontier) > 0:
        # Sort the frontier according to heuristic...
        sort_frontier(sort_by)

        # Remove the correct node from the frontier and add it to the visited.
        current_node = frontier.pop(0)
        visited[current_node] = None

        # Stop if we are in a goal state...
        if isGoal(current_node):
            goal_state = current_node
            break

        # Add to the frontier as in BFS.
        addToFrontier(current_node, "BFS")

    # Check if successful...
    if goal_state is not None:
        # Calculate the cost of the solution and get the solution itself...
        current = goal_state
        while current is not None:
            solution_cost += current.cost
            solution.insert(0, current)
            # Get the parent node and continue...
            current = current.parent

        # Print the results...
        results(algorithm, solution_cost, solution, visited)
    else:
        print("No goal state found.")


def addToFrontier(current_node, algorithm):
    # If the child nodes are not None AND if they are not in visited, add them to the frontier.
    nodes_to_add = []
    if current_node.east is not None and not isVisited(current_node.east):
        nodes_to_add.append(setParent(current_node, current_node.east, algorithm))
    if current_node.south is not None and not isVisited(current_node.south):
        nodes_to_add.append(setParent(current_node, current_node.south, algorithm))
    if current_node.west is not None and not isVisited(current_node.west):
        nodes_to_add.append(setParent(current_node, current_node.west, algorithm))
    if current_node.north is not None and not isVisited(current_node.north):
        nodes_to_add.append(setParent(current_node, current_node.north, algorithm))

    # For DFS, do it in reverse order because we add each node to the end and EAST should be the last node.
    # For BFS, do it in the correct order.
    if "DFS" in algorithm:
        nodes_to_add.reverse()

    # Then add each node to the frontier.
    for node in nodes_to_add:
        frontier.append(node)


def setParent(parent_node, child_node, algorithm):
    # Set the parent node if it is None and if DFS is used.
    if "DFS" in algorithm or child_node.parent is None:
        child_node.parent = parent_node
    return child_node


def isVisited(node):
    return node in visited


def isGoal(node):
    for goal in graph.maze.goals:
        if goal[0] == node.x and goal[1] == node.y:
            return True
    return False


def results(algorithm, solution_cost, solution, expanded_nodes):
    print(algorithm)
    print("Cost of solution:", solution_cost)
    print("Solution path (" + str(len(solution)) + " nodes):", end=" ")
    for node in solution:
        print(node, end=" ")
    print("\nExpanded nodes (" + str(len(expanded_nodes)) + " nodes):", end=" ")
    if "IDS" in algorithm:
        print()
        for i in range(len(expanded_nodes) - 1):
            if type(expanded_nodes[i+1]) == str:
                print(expanded_nodes[i])
            else:
                print(expanded_nodes[i], end=" ")
    else:
        for node in expanded_nodes:
            print(node, end=" ")
    print("\n")


def return_cost(node):
    return node.cost

def return_heuristic(node):
    return node.heuristic

def return_cost_and_heuristic(node):
    return node.heuristic + node.cost

def sort_frontier(sort_by):
    frontier.sort(key=sort_by)



dfs()
bfs()
ids()
ucs()
gbfs()
astar()
