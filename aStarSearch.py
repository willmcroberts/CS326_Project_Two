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
