import numpy as np
import sys

from heuristics.shortest_path_solver import ShortestPathSolver
from routing_problem_solver import Route


# =======================================================
#                GIANT ROUTE HEURISTICS
# =======================================================
def k_nearest_neighbor_giant_tour(depot, customers, k=3):
    last_position = depot
    giant_route = []

    while len(giant_route) != len(customers):
        possible_successors = [n for n in customers if n not in giant_route]
        possible_successors.sort(key=lambda n: n.distance_to(last_position))
        possible_successors = possible_successors[:k]

        successor = min(possible_successors, key=lambda n: n.due_date)
        giant_route.append(successor)

        last_position = successor

    return giant_route


def nearest_neighbor_tolerance_giant_tour(depot, customers, tolerance=1.3):
    last_position = depot
    giant_route = []

    while len(giant_route) != len(customers):
        possible_successors = [n for n in customers if n not in giant_route]
        min_distance = min(possible_successors, key=lambda x: x.distance_to(last_position)).distance_to(
            last_position)
        possible_successors = [n for n in possible_successors if
                               n.distance_to(last_position) <= min_distance * tolerance]

        possible_successors.sort(key=lambda n: n.distance_to(last_position))

        successor = min(possible_successors, key=lambda n: n.due_date)
        giant_route.append(successor)

        last_position = successor

    return giant_route


def generate_basic_route(from_route, target):
    from_route.route += [target]
    return from_route


class BeasleyHeuristic:
    def __init__(self, generate_giant_route, giant_route_args, generate_feasible_route=generate_basic_route):
        self.generate_giant_route = generate_giant_route
        self.generate_feasible_route = generate_feasible_route
        self.giant_route_args = giant_route_args
        self.giant_route = None

    def set_generate_feasible_route_function(self, generate_feasible_route_function):
        self.generate_feasible_route = generate_feasible_route_function

    def __calc_dist(self, i, j, problem_instance):
        new_route = Route(problem_instance.config, problem_instance.depot)
        new_route = self.generate_feasible_route(new_route, self.giant_route[i], problem_instance)

        if new_route is None:
            return sys.maxsize, None

        while i != (j - 1):
            i += 1
            new_route = self.generate_feasible_route(new_route, self.giant_route[i], problem_instance)
            if new_route is None:
                return sys.maxsize, None

        new_route = self.generate_feasible_route(new_route, problem_instance.depot, problem_instance)
        if new_route is None:
            return sys.maxsize, None

        return new_route.calculate_total_distance(), new_route

    def solve(self, problem_instance):
        self.giant_route = self.generate_giant_route(problem_instance.depot, problem_instance.customers,
                                                     *self.giant_route_args)

        solution = []

        cost = np.zeros((len(self.giant_route) + 1, len(self.giant_route) + 1), dtype=float)
        routes = np.zeros((len(self.giant_route) + 1, len(self.giant_route) + 1), dtype=Route)

        cost[:, :] = sys.maxsize
        routes[:, :] = None

        for i in range(0, len(self.giant_route)):
            for j in range(i + 1, len(self.giant_route) + 1):
                d, r = self.__calc_dist(i, j, problem_instance)
                if r is None:
                    break
                else:
                    cost[i, j] = d
                    routes[i, j] = r

        # solve shortest path problem
        sp_solver = ShortestPathSolver(cost)
        result = sp_solver.solve()

        for r in result:
            solution.append(routes[r[0], r[1]])

        return solution
