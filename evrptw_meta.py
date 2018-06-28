from copy import deepcopy
from queue import LifoQueue
from random import randint, shuffle, random, choice
from math import exp

from evrptw_solver import RoutingProblemInstance
from targets import Customer, CharingStation

from itertools import combinations, product

import sys

K_MAX = 4
NO_IMPROVEMENT_TOLERANCE = 1


# neighbourhoods
def two_opt_star(state, distances, feasibility_operator, route_length_operator, min_dist=sys.maxsize):
    route_combinations = list(combinations(range(0, len(state)), 2))
    shuffle(route_combinations)

    for c in route_combinations:
        route_1 = list(state[c[0]])
        route_2 = list(state[c[1]])

        swap_points = list(product(range(1, len(route_1) - 1), range(1, len(route_2) - 1)))
        shuffle(swap_points)

        for sp in swap_points:
            split_index_1 = sp[0]
            split_index_2 = sp[1]

            new_route_1 = route_1[:split_index_1] + route_2[split_index_2:]
            new_route_2 = route_2[:split_index_2] + route_1[split_index_1:]

            d_1 = route_length_operator(new_route_1)
            d_2 = route_length_operator(new_route_2)

            new_distances = distances[:c[0]] + [d_1] + distances[c[0] + 1:c[1]] + [d_2] + distances[c[1] + 1:]
            new_state = state[:c[0]] + [new_route_1] + state[c[0] + 1:c[1]] + [new_route_2] + state[c[1] + 1:]

            if feasibility_operator(new_route_1) and feasibility_operator(new_route_2) \
                    and sum(new_distances) < min_dist:
                return new_state, new_distances
    return state, distances


def two_opt(state, distances, feasibility_operator, route_length_operator, min_dist=sys.maxsize):
    route_indices = list(range(0, len(state)))
    shuffle(route_indices)

    for i in route_indices:
        route = state[i]
        cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
        shuffle(cut_points)

        for cp in cut_points:
            if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                part_1 = route[:cp[0]]
                part_2 = route[cp[0]:cp[1]]
                part_3 = route[cp[1]:]

                part_2.reverse()

                d = route_length_operator(part_1 + part_2 + part_3)

                new_distances = distances[:i] + [d] + distances[i + 1:]

                if feasibility_operator(part_1 + part_2 + part_3) and sum(new_distances) < min_dist:
                    new_state = state[:i] + [part_1 + part_2 + part_3] + state[i + 1:]
                    return new_state, new_distances

    return state, distances


def or_opt(state, distances, feasibility_operator, route_length_operator, min_dist=sys.maxsize):
    route_indices = list(range(0, len(state)))
    shuffle(route_indices)

    for i in route_indices:
        route = state[i]
        cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
        shuffle(cut_points)

        for cp in cut_points:
            if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                part_1 = route[:cp[0]]
                part_2 = route[cp[0]:cp[1]]
                part_3 = route[cp[1]:]

                offset = randint(0, len(part_3) - 1)

                route = part_1 + part_3[:offset] + part_2 + part_3[offset:]

                d = route_length_operator(route)

                new_distances = distances[:i] + [d] + distances[i + 1:]

                if feasibility_operator(route) and sum(new_distances) < min_dist:
                    new_state = state[:i] + [route] + state[i + 1:]
                    return new_state, new_distances
    return state, distances


def cross_exchange(state, distances, feasibility_operator, route_length_operator, min_dist=sys.maxsize):
    route_combinations = list(combinations(range(0, len(state)), 2))
    shuffle(route_combinations)

    for c in route_combinations:
        route_1 = list(state[c[0]])
        route_2 = list(state[c[1]])

        swap_points = list(product(range(1, len(route_1) - 1), range(1, len(route_1) - 1), range(1, len(route_2) - 1),
                                   range(1, len(route_2) - 1)))
        shuffle(swap_points)

        for sp in swap_points:

            if sp[0] < sp[1] - 1 and sp[2] < sp[3] - 1:
                sr_1_s = min(sp[0], sp[1])
                sr_1_e = max(sp[0], sp[1])
                sr_2_s = min(sp[2], sp[3])
                sr_2_e = max(sp[2], sp[3])

                new_route_1 = route_1[:sr_1_s] + route_2[sr_2_s:sr_2_e] + route_1[sr_1_e:]
                new_route_2 = route_2[:sr_2_s] + route_1[sr_1_s:sr_1_e] + route_2[sr_2_e:]

                new_distances = distances[:c[0]] + [route_length_operator(new_route_1)] + distances[c[0] + 1:c[1]] \
                                + [route_length_operator(new_route_2)] + distances[c[1] + 1:]

                if sum(new_distances) < min_dist and feasibility_operator(new_route_1) and feasibility_operator(
                        new_route_2):
                    new_state = state[:c[0]] + [new_route_1] + state[c[0] + 1:c[1]] + [new_route_2] + state[c[1] + 1:]
                    return new_state, new_distances
    return state, distances


def merge_route(state, distances, feasibility_operator, route_length_operator, min_distance=sys.maxsize):
    route_combinations = list(combinations(range(0, len(state)), 2))
    shuffle(route_combinations)

    for c in route_combinations:
        route_1 = list(state[c[0]])
        route_2 = list(state[c[1]])

        new_distances = distances[:c[0]] \
                        + [route_length_operator(route_1[:-1] + route_2[1:])] \
                        + distances[c[0] + 1:c[1]] \
                        + distances[c[1] + 1:]

        if feasibility_operator(route_1[:-1] + route_2[1:]) and sum(new_distances) < min_distance:
            new_state = state[:c[0]] + [route_1[:-1] + route_2[1:]] + state[c[0] + 1:c[1]] + state[c[1] + 1:]
            return new_state, new_distances
    return state, distances


class SimulatedAnnealing:
    def __init__(self, problem_instance: RoutingProblemInstance, state, distances, t_0, cooling_factor, rep=1):
        self.problem_instance = problem_instance
        self.state = state
        self.distances = list()
        for r in self.state:
            self.distances.append(self.calculate_route_distance(r))
        self.t_0 = t_0
        self.temp = t_0
        self.cooling_factor = cooling_factor
        self.rep = rep

        self.neighbour_hoods = [merge_route, two_opt, two_opt_star]  # , or_opt, cross_exchange]

    def improve_solution(self):
        state_approx = self.state
        distance_approx = self.distances
        iteration = 0
        for r in range(0, self.rep):
            self.temp = self.t_0
            current_state = state_approx
            current_distances = distance_approx
            while self.temp > 0.0000000001:
                random_neighbour, random_distances = self.get_random_feasible_neighbour(current_state,
                                                                                        current_distances,
                                                                                        iteration)

                delta = (sum(random_distances) - sum(current_distances)) / sum(current_distances)

                if delta < 0:
                    current_state = random_neighbour
                    current_distances = random_distances

                    if sum(distance_approx) > sum(current_distances):
                        state_approx = current_state
                        distance_approx = current_distances
                else:
                    random_number = random()

                    if random_number < exp(-1 * delta / self.temp):
                        current_state = random_neighbour
                        current_distances = random_distances

                self.temp *= self.cooling_factor
                iteration += 1

        state_approx, distance_approx = self.local_search(state_approx, distance_approx)
        return sum(distance_approx), state_approx

    def get_random_feasible_neighbour(self, state, distances, iteration):
        next_state = state
        next_distances = distances

        for nh_op in self.neighbour_hoods:
            if choice([True, False]):
                next_state, next_distances = nh_op(next_state, next_distances, self.is_feasible,
                                                   self.calculate_route_distance)

        return next_state, next_distances

    def local_search(self, state, distances):
        current_state = state
        current_distances = distances

        while True:
            best_neighbour, best_neighbour_distances = self.get_best_neighbour(current_state, current_distances)

            if sum(best_neighbour_distances) < sum(current_distances):
                current_state = best_neighbour
                current_distances = best_neighbour_distances
                continue

            break

        return current_state, current_distances

    def get_best_neighbour(self, state, distances):
        best_state = state
        best_distances = distances

        for nh_op in self.neighbour_hoods:
            best_state, best_distances = nh_op(best_state, best_distances, self.is_feasible,
                                               self.calculate_route_distance,
                                               sum(best_distances))

        return best_state, best_distances

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

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.ready_time + self.problem_instance.depot.service_time

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity
            energy_capacity -= d * fuel_consumption_rate

            if energy_capacity < 0:
                return False

            if time > target.due_date:
                return False

            if time < target.ready_time:
                time = target.ready_time

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            elif type(target) is CharingStation:
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


class VariableNeighbourhoodSearch:
    """
        N_0 -> 2-exchange within routes (intra route)
        N_1 -> 2-opt (intra route)
        N_2 -> Moving a customer to another route (inter route)
        N_3 -> Merge two routes (inter route)
        N_4 -> 2opt*-operator (inter route)
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
                # generate next random feasible solution
                next_rand_sol, next_route_dist_cache = self.get_next_random_feasible_solution(k, best_solution,
                                                                                              route_dist_cache)

                # find local minimum of the random solution
                ls_solution, ls_route_dist_cache = self.do_local_search(next_rand_sol, next_route_dist_cache)

                if sum(ls_route_dist_cache) < best_dist:
                    # the solution was better than the current -> restart with neighbourhood 0
                    best_solution = ls_solution
                    best_dist = sum(ls_route_dist_cache)
                    k = 0
                else:
                    # solution was not better -> try to improve with next neighbourhood
                    k += 1

            result_cache.put(best_dist)

            if result_cache.qsize() == NO_IMPROVEMENT_TOLERANCE:
                prev_dist = result_cache.get()

                # In case there was no improvement within the last [NO_IMPROVEMENT_TOLERANCE] rounds -> terminate!
                if prev_dist == best_dist:
                    break
                else:
                    print('hallo')

        return best_dist, best_solution

    def get_next_random_feasible_solution(self, neighbourhood_index, current_solution, route_dist_cache):
        # N_0 -> 2-exchange within routes
        if neighbourhood_index == 0:
            rand_route_idx = randint(0, len(self.solution) - 1)

            for i in range(0, len(self.solution)):
                route_idx = (rand_route_idx + i) % (len(current_solution))
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

        # N_1 -> 2opt
        elif neighbourhood_index == 1:
            route_indices = list(range(0, len(current_solution) - 1))
            shuffle(route_indices)

            for route_idx in route_indices:
                route = list(current_solution[route_idx])

                cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
                shuffle(cut_points)

                for cp in cut_points:
                    if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                        part_1 = route[:cp[0]]
                        part_2 = route[cp[0]:cp[1]]
                        part_3 = route[cp[1]:]

                        new_dist = self.calculate_route_distance(part_1 + part_2 + part_3)

                        if self.is_feasible(part_1 + part_2 + part_3):
                            route_dist_cache = route_dist_cache[:route_idx] + [new_dist] + route_dist_cache[
                                                                                           route_idx + 1:]
                            new_solution = current_solution[:route_idx] + [part_1 + part_2 + part_3] + current_solution[
                                                                                                       route_idx + 1:]
                            return new_solution, route_dist_cache

        # N_2 -> Moving a customer to another route
        elif neighbourhood_index == 2:
            route_combinations = list(combinations(range(0, len(current_solution)), 2))
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
                        from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]
                        if self.is_feasible(from_route) and self.is_feasible(to_route):
                            if combination[0] < combination[1]:
                                new_solution = current_solution[:combination[0]]
                                new_solution += [to_route]
                                new_solution += current_solution[combination[0] + 1:combination[1]]
                                new_solution += [from_route]
                                new_solution += current_solution[combination[1] + 1:]

                                new_route_dist_cache = route_dist_cache[:combination[0]]
                                new_route_dist_cache += [self.calculate_route_distance(to_route)]
                                new_route_dist_cache += route_dist_cache[combination[0] + 1:combination[1]]
                                new_route_dist_cache += [self.calculate_route_distance(from_route)]
                                new_route_dist_cache += route_dist_cache[combination[1] + 1:]
                            else:
                                new_solution = current_solution[:combination[1]]
                                new_solution += [from_route]
                                new_solution += current_solution[combination[1] + 1:combination[0]]
                                new_solution += [to_route]
                                new_solution += current_solution[combination[0] + 1:]

                                new_route_dist_cache = route_dist_cache[:combination[1]]
                                new_route_dist_cache += [self.calculate_route_distance(from_route)]
                                new_route_dist_cache += route_dist_cache[combination[1] + 1:combination[0]]
                                new_route_dist_cache += [self.calculate_route_distance(to_route)]
                                new_route_dist_cache += route_dist_cache[combination[0] + 1:]

                            return new_solution, new_route_dist_cache
                        else:
                            from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]

        # N_3 -> Merge two routes
        elif neighbourhood_index == 3:
            route_combinations = list(combinations(range(0, len(current_solution)), 2))
            shuffle(route_combinations)

            for combination in route_combinations:
                from_route = list(current_solution[combination[0]])
                to_route = list(current_solution[combination[1]])

                new_route = None
                if self.is_feasible(from_route[:-1] + to_route[1:]):
                    new_route = from_route[:-1] + to_route[1:]
                elif self.is_feasible(to_route[:-1] + from_route[1:]):
                    new_route = to_route[:-1] + from_route[1:]

                if new_route:
                    new_solution = current_solution[:combination[0]]
                    new_solution += [new_route]
                    new_solution += current_solution[combination[0] + 1:combination[1]]
                    new_solution += current_solution[combination[1] + 1:]

                    new_route_dist_cache = route_dist_cache[:combination[0]]
                    new_route_dist_cache += [self.calculate_route_distance(new_route)]
                    new_route_dist_cache += route_dist_cache[combination[0] + 1:combination[1]]
                    new_route_dist_cache += route_dist_cache[combination[1] + 1:]

                    return new_solution, new_route_dist_cache

        # N_4 -> 2opt*-operator
        elif neighbourhood_index == 4:
            route_combinations = list(combinations(range(0, len(current_solution)), 2))
            shuffle(route_combinations)

            for combination in route_combinations:
                route_1 = list(current_solution[combination[0]])
                route_2 = list(current_solution[combination[1]])

                swap_points = list(product(range(1, len(route_1)), range(1, len(route_2))))
                shuffle(swap_points)

                for sp in swap_points:
                    split_index_1 = sp[0]
                    split_index_2 = sp[1]
                    # sp_1 = self.problem_instance.vertices[route_1[split_index_1]]
                    # sp_2 = self.problem_instance.vertices[route_2[split_index_2]]

                    if split_index_1 >= 2:
                        sp_prev = self.problem_instance.vertices[route_1[split_index_1 - 1]]

                        if type(sp_prev) is CharingStation:
                            split_index_1 -= 1
                            # sp_1 = sp_prev

                    if split_index_2 >= 2:
                        sp_prev = self.problem_instance.vertices[route_2[split_index_2 - 1]]

                        if type(sp_prev) is CharingStation:
                            split_index_2 -= 1
                            # sp_2 = sp_prev

                    new_route_1 = route_1[:split_index_1] + route_2[split_index_2:]
                    new_route_2 = route_2[:split_index_2] + route_1[split_index_1:]

                    if self.is_feasible(new_route_1) and self.is_feasible(new_route_2):
                        current_solution[combination[0]] = new_route_1
                        current_solution[combination[1]] = new_route_2

                        route_dist_cache[combination[0]] = self.calculate_route_distance(new_route_1)
                        route_dist_cache[combination[1]] = self.calculate_route_distance(new_route_2)

                        return current_solution, route_dist_cache

        return current_solution, route_dist_cache

    def do_local_search(self, initial_solution, route_dist_cache):

        best_dist = sum(route_dist_cache)
        best_solution = initial_solution
        best_route_dist_cache = route_dist_cache

        solution_improved = False

        next_best_solution, next_best_route_dist_cache = self.get_next_best_n0_neighbour(best_solution,
                                                                                         best_route_dist_cache)
        if best_dist > sum(next_best_route_dist_cache):
            best_dist = sum(next_best_route_dist_cache)
            best_solution = next_best_solution
            solution_improved = True

        next_best_solution, next_best_route_dist_cache = self.get_next_best_n1_neighbour(best_solution,
                                                                                         best_route_dist_cache)
        if best_dist > sum(next_best_route_dist_cache):
            best_dist = sum(next_best_route_dist_cache)
            best_solution = next_best_solution
            solution_improved = True

        next_best_solution, next_best_route_dist_cache = self.get_next_best_n2_neighbour(best_solution,
                                                                                         best_route_dist_cache)

        if best_dist > sum(next_best_route_dist_cache):
            best_dist = sum(next_best_route_dist_cache)
            best_solution = next_best_solution
            solution_improved = True

        next_best_solution, next_best_route_dist_cache = self.get_next_best_n3_neighbour(best_solution,
                                                                                         best_route_dist_cache)
        if best_dist > sum(next_best_route_dist_cache):
            best_dist = sum(next_best_route_dist_cache)
            best_solution = next_best_solution
            solution_improved = True

        next_best_solution, next_best_route_dist_cache = self.get_next_best_n4_neighbour(best_solution,
                                                                                         best_route_dist_cache)
        while solution_improved:
            solution_improved = False
            best_solution = next_best_solution
            best_dist = sum(next_best_route_dist_cache)
            best_route_dist_cache = next_best_route_dist_cache

            next_best_solution, next_best_route_dist_cache = self.get_next_best_n0_neighbour(best_solution,
                                                                                             best_route_dist_cache)

            if best_dist > sum(next_best_route_dist_cache):
                best_dist = sum(next_best_route_dist_cache)
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_dist_cache = self.get_next_best_n1_neighbour(best_solution,
                                                                                             best_route_dist_cache)

            if best_dist > sum(next_best_route_dist_cache):
                best_dist = sum(next_best_route_dist_cache)
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_dist_cache = self.get_next_best_n2_neighbour(best_solution,
                                                                                             best_route_dist_cache)

            if best_dist > sum(next_best_route_dist_cache):
                best_dist = sum(next_best_route_dist_cache)
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_dist_cache = self.get_next_best_n3_neighbour(best_solution,
                                                                                             best_route_dist_cache)

            if best_dist > sum(next_best_route_dist_cache):
                best_dist = sum(next_best_route_dist_cache)
                best_solution = next_best_solution
                solution_improved = True

            next_best_solution, next_best_route_dist_cache = self.get_next_best_n4_neighbour(best_solution,
                                                                                             best_route_dist_cache)

            if best_dist > sum(next_best_route_dist_cache):
                best_dist = sum(next_best_route_dist_cache)
                best_solution = next_best_solution
                solution_improved = True

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

        route_indices = list(range(0, len(solution) - 1))
        shuffle(route_indices)

        for route_idx in route_indices:
            route = list(solution[route_idx])

            cut_points = list(product(range(1, len(route) - 1), range(0, len(route) - 1)))
            shuffle(cut_points)

            for cp in cut_points:
                if cp[0] < cp[1] and cp[1] - cp[0] > 1:
                    part_1 = route[:cp[0]]
                    part_2 = route[cp[0]:cp[1]]
                    part_3 = route[cp[1]:]

                    new_dist = self.calculate_route_distance(part_1 + part_2 + part_3)
                    new_route_dist_cache = route_dist_cache[:route_idx] + [new_dist] + route_dist_cache[route_idx + 1:]

                    new_solution = solution[:route_idx] + [part_1 + part_2 + part_3] + solution[
                                                                                       route_idx + 1:]

                    if self.is_feasible(part_1 + part_2 + part_3) and sum(route_dist_cache) > sum(new_route_dist_cache):
                        return new_solution, route_dist_cache

        return solution, route_dist_cache

    def get_next_best_n2_neighbour(self, solution, route_dist_cache):
        best_solution = solution
        best_dist = sum(route_dist_cache)

        route_combinations = list(combinations(range(0, len(solution)), 2))
        shuffle(route_combinations)

        for combination in route_combinations:
            from_route = list(solution[combination[0]])
            to_route = list(solution[combination[1]])

            transfer_points = list(product(range(0, len(from_route)), range(0, len(to_route))))
            shuffle(transfer_points)

            for tp in transfer_points:
                from_v = self.problem_instance.vertices[from_route[tp[0]]]
                to_v = self.problem_instance.vertices[to_route[tp[1]]]

                if type(from_v) is Customer and type(to_v) is Customer:
                    from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]
                    if self.is_feasible(from_route) and self.is_feasible(to_route):
                        if combination[0] < combination[1]:
                            new_solution = solution[:combination[0]]
                            new_solution += [to_route]
                            new_solution += solution[combination[0] + 1:combination[1]]
                            new_solution += [from_route]
                            new_solution += solution[combination[1] + 1:]

                            new_route_dist_cache = route_dist_cache[:combination[0]]
                            new_route_dist_cache += [self.calculate_route_distance(to_route)]
                            new_route_dist_cache += route_dist_cache[combination[0] + 1:combination[1]]
                            new_route_dist_cache += [self.calculate_route_distance(from_route)]
                            new_route_dist_cache += route_dist_cache[combination[1] + 1:]
                        else:
                            new_solution = solution[:combination[1]]
                            new_solution += [from_route]
                            new_solution += solution[combination[1] + 1:combination[0]]
                            new_solution += [to_route]
                            new_solution += solution[combination[0] + 1:]

                            new_route_dist_cache = route_dist_cache[:combination[1]]
                            new_route_dist_cache += [self.calculate_route_distance(from_route)]
                            new_route_dist_cache += route_dist_cache[combination[1] + 1:combination[0]]
                            new_route_dist_cache += [self.calculate_route_distance(to_route)]
                            new_route_dist_cache += route_dist_cache[combination[0] + 1:]

                        if sum(new_route_dist_cache) < best_dist:
                            return new_solution, new_route_dist_cache

                    from_route[tp[0]], to_route[tp[1]] = to_route[tp[1]], from_route[tp[0]]

        return best_solution, route_dist_cache

    def get_next_best_n3_neighbour(self, solution, route_dist_cache):
        route_combinations = list(combinations(range(0, len(solution)), 2))
        shuffle(route_combinations)

        for combination in route_combinations:
            from_route = list(solution[combination[0]])
            to_route = list(solution[combination[1]])

            new_route = None
            if self.is_feasible(from_route[:-1] + to_route[1:]):
                new_route = from_route[:-1] + to_route[1:]
            elif self.is_feasible(to_route[:-1] + from_route[1:]):
                new_route = to_route[:-1] + from_route[1:]

            if new_route:
                new_solution = solution[:combination[0]]
                new_solution += [new_route]
                new_solution += solution[combination[0] + 1:combination[1]]
                new_solution += solution[combination[1] + 1:]

                new_route_dist_cache = route_dist_cache[:combination[0]]
                new_route_dist_cache += [self.calculate_route_distance(new_route)]
                new_route_dist_cache += route_dist_cache[combination[0] + 1:combination[1]]
                new_route_dist_cache += route_dist_cache[combination[1] + 1:]

                if sum(new_route_dist_cache) < sum(route_dist_cache):
                    return new_solution, new_route_dist_cache

        return solution, route_dist_cache

    def get_next_best_n4_neighbour(self, solution, route_dist_cache):
        route_combinations = list(combinations(range(0, len(solution)), 2))
        shuffle(route_combinations)

        for combination in route_combinations:
            route_1 = list(solution[combination[0]])
            route_2 = list(solution[combination[1]])

            swap_points = list(product(range(1, len(route_1)), range(1, len(route_2))))
            shuffle(swap_points)

            for sp in swap_points:
                split_index_1 = sp[0]
                split_index_2 = sp[1]

                if split_index_1 >= 2:
                    sp_prev = self.problem_instance.vertices[route_1[split_index_1 - 1]]

                    if type(sp_prev) is CharingStation:
                        split_index_1 -= 1
                        # sp_1 = sp_prev

                if split_index_2 >= 2:
                    sp_prev = self.problem_instance.vertices[route_2[split_index_2 - 1]]

                    if type(sp_prev) is CharingStation:
                        split_index_2 -= 1
                        # sp_2 = sp_prev

                new_route_1 = route_1[:split_index_1] + route_2[split_index_2:]
                new_route_2 = route_2[:split_index_2] + route_1[split_index_1:]

                if self.is_feasible(new_route_1) and self.is_feasible(new_route_2):
                    if (self.calculate_route_distance(new_route_1) + self.calculate_route_distance(new_route_2)) < \
                            route_dist_cache[combination[0]] + route_dist_cache[combination[1]]:
                        solution[combination[0]] = new_route_1
                        solution[combination[1]] = new_route_2

                        route_dist_cache[combination[0]] = self.calculate_route_distance(new_route_1)
                        route_dist_cache[combination[1]] = self.calculate_route_distance(new_route_2)

                        return solution, route_dist_cache
        return solution, route_dist_cache

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

        last_position = self.problem_instance.depot
        time = self.problem_instance.depot.ready_time + self.problem_instance.depot.service_time

        for v in route[1:]:
            target = self.problem_instance.vertices[v]
            d = last_position.distance_to(target)
            time += d / velocity
            energy_capacity -= d * fuel_consumption_rate

            if energy_capacity < 0:
                return False

            if time > target.due_date:
                return False

            if time < target.ready_time:
                time = target.ready_time

            if type(target) is Customer:
                load_capacity -= target.demand
                time += target.service_time
            elif type(target) is CharingStation:
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
