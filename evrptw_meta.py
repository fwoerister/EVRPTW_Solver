from copy import deepcopy
from queue import LifoQueue
from random import randint, shuffle

from evrptw_solver import RoutingProblemInstance
from targets import Customer

from itertools import combinations, product

K_MAX = 2
NO_IMPROVEMENT_TOLERANCE = 5


class VariableNeighbourhoodSearch:
    """
        N_0 -> 2-exchange within routes
        N_1 -> Moving a customer to another route
        N_2 -> Merge two routes
    """

    def __init__(self, problem_instance: RoutingProblemInstance, solution, distance):
        self.problem_instance = problem_instance
        self.solution = solution
        self.distance = distance

    def improve_solution(self):
        best_dist = self.distance
        best_solution = deepcopy(self.solution)

        result_cache = LifoQueue()  # saves the last 5 results

        route_dist_cache = []

        for r in self.solution:
            route_dist_cache.append(self.calculate_route_distance(r))

        while True:
            k = 0
            while k <= K_MAX:
                # TODO: generate next random feasible solution
                next_rand_sol, next_route_dist_cache = self.get_next_random_feasible_solution(k, best_solution,
                                                                                              route_dist_cache)

                ls_solution, ls_route_dist_cache = self.do_local_search(k, next_rand_sol, next_route_dist_cache)

                if sum(ls_route_dist_cache) < best_dist:
                    best_solution = ls_solution
                    best_dist = sum(ls_route_dist_cache)
                    k = 0
                else:
                    k += 1

            if result_cache.qsize() < NO_IMPROVEMENT_TOLERANCE:
                result_cache.put(best_dist)
            else:
                prev_dist = result_cache.get()

                # In case there was no improvement within the last 5 rounds -> terminate!
                if prev_dist == best_dist:
                    break

        return best_dist, best_solution

    def get_next_random_feasible_solution(self, neighbourhood_index, current_solution, route_dist_cache):
        # N_0 -> 2-exchange within routes
        if neighbourhood_index == 0:
            rand_route_idx = randint(0, len(self.solution) - 1)

            for i in range(0, len(self.solution)):
                route_idx = (rand_route_idx + i) % (len(self.solution))
                route = list(current_solution[route_idx])

                for j in range(0, len(route) * len(route)):
                    from_idx = randint(0, len(route) - 1)
                    to_idx = randint(0, len(route) - 1)

                    if from_idx == to_idx:
                        to_idx = (to_idx + 1) % (len(route))

                    from_v = self.problem_instance.vertices[route[from_idx]]
                    to_v = self.problem_instance.vertices[route[to_idx]]

                    if type(from_v) is Customer and type(to_v) is Customer:
                        route[from_idx], route[to_idx] = route[to_idx], route[from_idx]
                        if self.is_feasible(route):
                            route_dist_cache = route_dist_cache[:route_idx] + [
                                self.calculate_route_distance(route)] + route_dist_cache[route_idx + 1:]
                            new_solution = current_solution[0:route_idx] + [route] + current_solution[
                                                                                     route_idx + 1:]
                            return new_solution, route_dist_cache
                        else:
                            route[from_idx], route[to_idx] = route[to_idx], route[from_idx]

        # N_1 -> Moving a customer to another route
        elif neighbourhood_index == 1:
            route_combinations = list(combinations(range(0, len(current_solution))))
            shuffle(route_combinations)

            for combination in route_combinations:
                from_route = list(current_solution[combination[0]])
                to_route = list(current_solution[combination[1]])

                transfer_points = list(product(range(0, len(from_route)), range(0, len(to_route))))
                shuffle(transfer_points)

                for tp in transfer_points:
                    from_v = self.problem_instance.vertices[from_route[tp[0]]]
                    to_v = self.problem_instance.vertices[to_route[tp[1]]]

                    if type(from_v) is Customer and type(to_v) is Customer:
                        from_route[from_idx], to_route[to_idx] = from_route[to_idx], to_route[from_idx]
                        if self.is_feasible(from_route) and self.is_feasible(to_route):

                            # =============================================
                            # TODO: create new route_dist_cache + solution!
                            # =============================================

                            return new_solution, route_dist_cache
                        else:
                            route[from_idx], route[to_idx] = route[to_idx], route[from_idx]

            pass

        # N_2 -> Merge two routes
        elif neighbourhood_index == 2:
            # TODO: random feasible N_2 generation
            pass

        return current_solution, route_dist_cache

    def do_local_search(self, neighbourhood_index, initial_solution, route_dist_cache):
        # N_0 -> 2-exchange within routes
        if neighbourhood_index == 0:
            best_dist = sum(route_dist_cache)
            best_solution = initial_solution
            best_route_dist_cache = route_dist_cache

            if neighbourhood_index == 0
                next_best_solution, next_best_route_dist_cache = self.get_next_best_n0_neighbour(best_solution,
                                                                                                 best_route_dist_cache)

            while sum(next_best_route_dist_cache) < best_dist:
                best_solution = next_best_solution
                best_dist = sum(next_best_route_dist_cache)
                best_route_dist_cache = next_best_route_dist_cache

                next_best_solution, next_best_route_dist_cache = self.get_next_best_n0_neighbour(best_solution,
                                                                                                 best_route_dist_cache)

        return best_solution, best_route_dist_cache

    def get_next_best_n0_neighbour(self, solution, route_dist_cache):
        best_solution = solution
        for route_idx in range(0, len(solution)):
            route = list(solution[route_idx])
            for from_idx in range(0, len(route) - 1):
                for to_idx in range(from_idx + 1, len(route)):
                    from_v = self.problem_instance.vertices[route[from_idx]]
                    to_v = self.problem_instance.vertices[route[to_idx]]

                    if type(from_v) is Customer and type(to_v) is Customer:
                        route[from_idx], route[to_idx] = route[to_idx], route[from_idx]
                        if self.is_feasible(route):
                            new_route_dist_cache = route_dist_cache[:route_idx] + [
                                self.calculate_route_distance(route)] + route_dist_cache[route_idx + 1:]
                            if sum(new_route_dist_cache) > sum(route_dist_cache):
                                best_solution = solution[0:route_idx] + [route] + best_solution[
                                                                                  route_idx + 1:]
                                return best_solution, new_route_dist_cache
                        else:
                            route[from_idx], route[to_idx] = route[to_idx], route[from_idx]

        return best_solution, route_dist_cache

    def get_next_best_n1_neighbour(self, solution, route_dist_cache):
        best_solution = solution
        return best_solution, route_dist_cache

    def get_next_best_n2_neighbour(self, solution, route_dist_cache):
        best_solution = solution
        return best_solution, route_dist_cache

    def is_feasible(self, route):
        """
        slow implementation for checking the feasibility of a route
        :param route: route to check
        :return: returns True if the route is feasible
        """
        q = self.problem_instance.config.tank_capacity
        energy_capacity = self.problem_instance.config.tank_capacity
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        time = 0

        last_position = self.problem_instance.depot

        for v in route[1:-1]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity
            energy_capacity -= d * fuel_consumption_rate

            if energy_capacity < 0:
                return False

            if time > target.due_date:
                return False

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            else:
                time += (q - energy_capacity) * charging_rate

            if load_capacity < 0:
                return False

            last_position = target
        return True

    def calculate_route_distance(self, route):
        last_pos = self.problem_instance.depot
        dist = 0

        for r in route:
            v = self.problem_instance.vertices[r]
            dist += v.distance_to(last_pos)
            last_pos = v

        return dist
