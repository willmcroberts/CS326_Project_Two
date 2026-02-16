import argparse
import json
import math
import random
import time
import heapq
import sys
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
                yield (nr, nc)

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

#Main
def parse_args():
    parser = argparse.ArgumentParser(description="A* and TSP Local Search")
    parser.add_argument("--task", type=str, choices=["astar", "tsp"], required=True)

    # A*
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--rs", type=int)
    parser.add_argument("--cs", type=int)
    parser.add_argument("--rg", type=int)
    parser.add_argument("--cg", type=int)
    parser.add_argument("--min_cost", type=int)
    parser.add_argument("--max_cost", type=int)
    parser.add_argument("--heuristic", type=str,
                        choices=["manhattan", "euclidean"],
                        default="manhattan")

    # TSP
    parser.add_argument("--cities", type=int)
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--operator", type=str,
                        choices=["twoopt", "swap", "insert"],
                        default="twoopt")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=str, default="results.json")

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.task == "astar":
        grid = Grid(
            args.m, args.n,
            (args.rs, args.cs),
            (args.rg, args.cg),
            args.min_cost, args.max_cost
        )

        result = astar(grid, heuristic=args.heuristic)

        output = {
            "algorithm": "astar",
            "m": args.m,
            "n": args.n,
            "start": [args.rs, args.cs],
            "goal": [args.rg, args.cg],
            "min_cost": args.min_cost,
            "max_cost": args.max_cost,
            "seed": args.seed,
            "heuristic": args.heuristic,
            "path": result["path"],
            "steps": result["steps"],
            "total_cost": result["total_cost"],
            "expanded_states": result["expanded"],
            "generated_nodes": result["generated"],
            "max_frontier_size": result["max_frontier"],
            "runtime_ms": result["runtime_ms"],
            "status": result["status"]
        }

    else:  # TSP
        result = tsp_local_search(
            args.cities, args.width, args.height,
            args.restarts, args.operator
        )

        output = {
            "algorithm": "tsp local search",
            "n_cities": args.cities,
            "seed": args.seed,
            "restarts": args.restarts,
            "operator": args.operator,
            "initial_tour": result["initial_tour"],
            "initial_cost": result["initial_cost"],
            "best_tour": result["best_tour"],
            "best_cost": result["best_cost"],
            "iterations": result["iterations"],
            "runtime_ms": result["runtime_ms"],
            "status": result["status"]
        }

    print(json.dumps(output, indent=4))

    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()


#Tests
def is_legal_move_sequence(grid, path):
    """Check that each move is up/down/left/right and in bounds."""
    if not path:
        return False
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        if dr + dc != 1:
            return False
        if not (0 <= r2 < grid.m and 0 <= c2 < grid.n):
            return False
    return True


