import json
import math
import random
import time
import heapq


class Grid:
    def __init__(self, m, n, start, goal, min_cost, max_cost):
        self.m = m
        self.n = n
        self.start = start
        self.goal = goal
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.costs = {}
        self._assign_costs()

    def _assign_costs(self):
        for r in range(self.m):
            for c in range(self.n):
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.m and 0 <= nc < self.n:
                        self.costs[((r,c),(nr,nc))] = random.randint(
                            self.min_cost, self.max_cost
                        )

    def neighbors(self, node):
        r, c = node
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.m and 0 <= nc < self.n:
                yield nr, nc

    def cost(self, a, b):
        return self.costs[(a, b)]

#A*
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(parent, start, goal):
    if goal not in parent:
        return []
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    return list(reversed(path))

def astar(grid, heuristic="manhattan"):
    start = grid.start
    goal = grid.goal

    if heuristic == "manhattan":
        h = lambda n: manhattan(n, goal)
    else:
        h = lambda n: euclidean(n, goal)

    frontier = []
    heapq.heappush(frontier, (h(start), 0, start))

    parent = {}
    bestCost = {start: 0}
    explored = set()

    generated = 1
    max_frontier = 1

    t0 = time.time()

    while frontier:
        max_frontier = max(max_frontier, len(frontier))

        f, g, node = heapq.heappop(frontier)

        if node in explored:
            continue

        explored.add(node)

        if node == goal:
            t1 = time.time()
            path = reconstruct_path(parent, start, goal)
            total_cost = sum(grid.cost(path[i], path[i+1]) for i in range(len(path)-1))
            return {
                "path": path,
                "steps": len(path)-1,
                "total_cost": total_cost,
                "expanded": len(explored),
                "generated": generated,
                "max_frontier": max_frontier,
                "runtime_ms": (t1 - t0)*1000,
                "status": "success"
            }

        for nbr in grid.neighbors(node):
            new_g = g + grid.cost(node, nbr)

            if nbr not in bestCost or new_g < bestCost[nbr]:
                bestCost[nbr] = new_g
                parent[nbr] = node
                f_nbr = new_g + h(nbr)
                heapq.heappush(frontier, (f_nbr, new_g, nbr))
                generated += 1

    t1 = time.time()
    return {
        "path": [],
        "steps": 0,
        "total_cost": None,
        "expanded": len(explored),
        "generated": generated,
        "max_frontier": max_frontier,
        "runtime_ms": (t1 - t0)*1000,
        "status": "failure"
    }

#TSP
def tour_cost(cities, tour):
    total = 0.0
    for i in range(len(tour)):
        a = cities[tour[i]]
        b = cities[tour[(i+1) % len(tour)]]
        total += math.dist(a, b)
    return total

def two_opt_neighbors(tour):
    n = len(tour)
    for i in range(n):
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue
            new_tour = tour[:]
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
            yield new_tour

def swap_neighbors(tour):
    n = len(tour)
    for i in range(n):
        for j in range(i+1, n):
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            yield new_tour

def insert_neighbors(tour):
    n = len(tour)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            new_tour = tour[:]
            city = new_tour.pop(i)
            new_tour.insert(j, city)
            yield new_tour

def tsp_local_search(n_cities, width, height, restarts, operator):
    global best_initial_cost, best_initial_tour, current_cost
    cities = [(random.uniform(0, width), random.uniform(0, height))
              for _ in range(n_cities)]

    if operator == "twoopt":
        neighbor_fn = two_opt_neighbors
    elif operator == "swap":
        neighbor_fn = swap_neighbors
    else:
        neighbor_fn = insert_neighbors

    t0 = time.time()

    best_overall_tour = None
    best_overall_cost = float("inf")
    iterations_list = []

    for _ in range(restarts):
        tour = list(range(n_cities))
        random.shuffle(tour)
        initial_tour = tour[:]
        initial_cost = tour_cost(cities, tour)

        improved = True
        iterations = 0

        while improved:
            improved = False
            iterations += 1
            current_cost = tour_cost(cities, tour)

            for nbr in neighbor_fn(tour):
                nbr_cost = tour_cost(cities, nbr)
                if nbr_cost < current_cost:
                    tour = nbr
                    current_cost = nbr_cost
                    improved = True
                    break

        iterations_list.append(iterations)

        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_tour = tour[:]
            best_initial_tour = initial_tour[:]
            best_initial_cost = initial_cost

    t1 = time.time()

    return {
        "cities": cities,
        "initial_tour": best_initial_tour,
        "initial_cost": best_initial_cost,
        "best_tour": best_overall_tour,
        "best_cost": best_overall_cost,
        "iterations": iterations_list,
        "runtime_ms": (t1 - t0)*1000,
        "status": "success"
    }


#Tests
def is_legal_move_sequence(grid, path):
    if not path:
        return False
    for i in range(len(path)-1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        if abs(r1-r2) + abs(c1-c2) != 1:
            return False
        if not (0 <= r2 < grid.m and 0 <= c2 < grid.n):
            return False
    return True

def test_astar_start_end():
    random.seed(42)
    grid = Grid(5,5,(0,0),(4,4),1,5)
    result = astar(grid)
    path = result["path"]
    assert path[0] == (0,0)
    assert path[-1] == (4,4)


def test_astar_legal_moves():
    random.seed(123)
    grid = Grid(5,5,(0,0),(4,4),1,5)
    result = astar(grid)
    assert is_legal_move_sequence(grid, result["path"])


def test_astar_cost_matches():
    random.seed(999)
    grid = Grid(5,5,(0,0),(4,4),1,10)
    result = astar(grid)
    path = result["path"]
    reported = result["total_cost"]
    recomputed = sum(grid.cost(path[i], path[i+1]) for i in range(len(path)-1))
    assert reported == recomputed


def test_tsp_valid_tour():
    random.seed(2024)
    result = tsp_local_search(20,100,100,5,"twoopt")
    tour = result["best_tour"]
    assert sorted(tour) == list(range(20))


def test_tsp_closed_cycle():
    random.seed(2024)
    result = tsp_local_search(15,100,100,3,"swap")
    tour = result["best_tour"]
    assert len(tour) == len(set(tour))


def test_tsp_hill_climbing_terminates():
    random.seed(2024)
    result = tsp_local_search(12,100,100,2,"insert")
    assert result["status"] == "success"


def run_tests():
    print("Running A* tests...")
    test_astar_start_end()
    print("✓ A*: start/end correct")
    test_astar_legal_moves()
    print("✓ A*: legal moves")
    test_astar_cost_matches()
    print("✓ A*: cost matches")

    print("\nRunning TSP tests...")
    test_tsp_valid_tour()
    print("✓ TSP: valid tour")
    test_tsp_closed_cycle()
    print("✓ TSP: closed cycle")
    test_tsp_hill_climbing_terminates()
    print("✓ TSP: hill climbing terminates")

    print("\nAll tests passed!")

def inputs():
    print("Select task:")
    print("1 = A* Search")
    print("2 = TSP Local Search")
    print("3 = Run Tests")
    choice = input("Enter choice: ").strip()

    if choice == "3":
        return {"task": "tests"}

    if choice == "1":
        task = "astar"
        m = int(input("Grid rows (m): "))
        n = int(input("Grid cols (n): "))
        rs = int(input("Start row: "))
        cs = int(input("Start col: "))
        rg = int(input("Goal row: "))
        cg = int(input("Goal col: "))
        min_cost = int(input("Min cost: "))
        max_cost = int(input("Max cost: "))
        heuristic = input("Heuristic (manhattan/euclidean): ").strip().lower()
        seed = int(input("Random seed: "))
        return {
            "task": task, "m": m, "n": n,
            "rs": rs, "cs": cs, "rg": rg, "cg": cg,
            "min_cost": min_cost, "max_cost": max_cost,
            "heuristic": heuristic, "seed": seed
        }

    else:
        task = "tsp"
        cities = int(input("Number of cities: "))
        width = float(input("Width of bounding box: "))
        height = float(input("Height of bounding box: "))
        restarts = int(input("Random restarts: "))
        operator = input("Operator (twoopt/swap/insert): ").strip().lower()
        seed = int(input("Random seed: "))
        return {
            "task": task, "cities": cities,
            "width": width, "height": height,
            "restarts": restarts, "operator": operator,
            "seed": seed
        }



#Main

def main():
    user = inputs()

    if user["task"] == "tests":
        run_tests()
        return

    random.seed(user["seed"])

    if user["task"] == "astar":
        grid = Grid(
            user["m"], user["n"],
            (user["rs"], user["cs"]),
            (user["rg"], user["cg"]),
            user["min_cost"], user["max_cost"]
        )

        result = astar(grid, heuristic=user["heuristic"])

        output = {
            "algorithm": "astar",
            "m": user["m"],
            "n": user["n"],
            "start": [user["rs"], user["cs"]],
            "goal": [user["rg"], user["cg"]],
            "min cost": user["min_cost"],
            "max cost": user["max_cost"],
            "seed": user["seed"],
            "heuristic": user["heuristic"],
            "path": result["path"],
            "steps": result["steps"],
            "total cost": result["total_cost"],
            "expanded states": result["expanded"],
            "generated nodes": result["generated"],
            "max frontier size": result["max_frontier"],
            "runtime ms": result["runtime_ms"],
            "status": result["status"]
        }

    else:
        result = tsp_local_search(
            user["cities"], user["width"], user["height"],
            user["restarts"], user["operator"]
        )

        output = {
            "algorithm": "tsp local search",
            "cities": user["cities"],
            "seed": user["seed"],
            "restarts": user["restarts"],
            "operator": user["operator"],
            "initial tour": result["initial_tour"],
            "initial cost": result["initial_cost"],
            "best tour": result["best_tour"],
            "best cost": result["best_cost"],
            "iterations": result["iterations"],
            "runtime ms": result["runtime_ms"],
            "status": result["status"]
        }

    print("\nRESULTS:")
    for k, v in output.items():
        print(f"{k}: {v}")

    with open("results.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()