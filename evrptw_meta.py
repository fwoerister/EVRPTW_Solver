from copy import deepcopy
from queue import LifoQueue
from random import randint

from evrptw_solver import RoutingProblemInstance
from targets import Customer

K_MAX = 3
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

        while True:
            k = 0
            while k <= K_MAX:
                # TODO: generate next random feasible solution
                rand_offset = randint(0, len(best_solution))

                ls_dist = 0
                ls_solution = []

                if ls_dist < best_dist:
                    best_solution = ls_solution
                    best_dist = ls_dist
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

    def get_next_random_feasible_solution(self, neighbourhood_index, current_solution):
        next_solution = []

        if neighbourhood_index == 0:
            rand_route_idx = randint(0, len(self.solution) - 1)

            for i in range(0, len(self.solution)):
                route_idx = (rand_route_idx + i) % (len(self.solution) - 1)
                route = list(current_solution[route_idx])

                for from_idx in range(0,len(route)-2):
                    for to_idx in range(from_idx+1, len(route)-1):
                        from_v = self.problem_instance.vertices[route[from_idx]]
                        to_v = self.problem_instance.vertices[route[to_idx]]

                        if type(from_v) is Customer and type(to_v) is Customer:
                            pass

        return next_solution

    def is_feasible(self, route):
        energy_capacity = self.problem_instance.config.tank_capacity
        load_capacity = self.problem_instance.config.payload_capacity
        velocity = self.problem_instance.config.velocity
        fuel_consumption_rate = self.problem_instance.config.fuel_consumption_rate
        charging_rate = self.problem_instance.config.charging_rate

        time = 0

        last_position = self.problem_instance.depot

        # for v in route[1:]:


