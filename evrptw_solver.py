from targets import Customer, CharingStation
import numpy as np


class RoutingProblemConfiguration:
    def __init__(self, tank_capacity, payload_capacity, fuel_consumption_rate, charging_rate, velocity):
        self.tank_capacity = tank_capacity
        self.payload_capacity = payload_capacity
        self.fuel_consumption_rate = fuel_consumption_rate
        self.charging_rate = charging_rate
        self.velocity = velocity


class RoutingProblemInstance:
    def __init__(self, config, depot, customers, charging_stations):
        self.config = config
        self.depot = depot
        self.customers = customers
        self.charging_stations = charging_stations

        # distance matrices
        self.cust_cust_dist = np.zeros((len(self.customers), len(self.customers)))
        self.cust_cs_dist = np.zeros((len(self.customers), len(self.charging_stations)))

        # vertex lookup dict
        self.vertices = dict()

        # initialization of distance matrices
        for i in range(0, len(self.customers)):
            for j in range(0, len(self.customers)):

                if i == 0:
                    from_v = self.depot
                else:
                    from_v = self.customers[i-1]

                if j == 0:
                    to_v = self.depot
                else:
                    to_v = self.customers[j-1]

                self.cust_cust_dist[i, j] = from_v.distance_to(to_v)

        for i in range(1, len(self.customers)):
            for j in range(0, len(self.charging_stations)-1):
                if i == 0:
                    from_v = self.depot
                else:
                    from_v = self.customers[i-1]

                self.cust_cs_dist[i, j] = from_v.distance_to(self.charging_stations[j])

        # initialization of the lookup dict
        self.vertices[self.depot.id] = self.depot
        for c in self.customers:
            self.vertices[c.id] = c
        for cs in self.charging_stations:
            self.vertices[cs.id] = cs


class Route:
    def __init__(self, config, depot):
        self.config = config
        self.route = [depot]
        self.depot = depot

    def is_feasible(self):
        if self.tw_constraint_violated():
            return False
        elif self.tank_capacity_constraint_violated():
            return False
        elif self.payload_capacity_constraint_violated():
            return False
        else:
            return True

    def is_complete(self):
        return self.route[0] == self.depot and self.route[-1] == self.depot and self.depot not in self.route[1:-1]

    # CONSTRAINT VALIDATION METHODS
    def tw_constraint_violated(self):
        elapsed_time = self.route[0].ready_time + self.route[0].service_time

        for i in range(1, len(self.route)):
            elapsed_time = elapsed_time + self.route[i - 1].distance_to(self.route[i]) / self.config.velocity

            if elapsed_time > self.route[i].due_date:
                return True

            if type(self.route[i]) is CharingStation:
                missing_energy = self.config.tank_capacity - self.calculate_remaining_tank_capacity(self.route[i])
                self.route[i].service_time = missing_energy * self.config.charging_rate

            waiting_time = max(self.route[i].ready_time - elapsed_time, 0)
            elapsed_time += waiting_time
            elapsed_time += self.route[i].service_time

        return False

    def tank_capacity_constraint_violated(self):
        last = None
        tank_capacity = self.config.tank_capacity
        for t in self.route:
            if last is not None:
                distance = last.distance_to(t)
                consumption = distance * self.config.fuel_consumption_rate

                tank_capacity -= consumption

                if tank_capacity < 0:
                    return True

                if type(t) is CharingStation:
                    tank_capacity = self.config.tank_capacity
            last = t

        return False

    def payload_capacity_constraint_violated(self):
        total_demand = 0
        for t in self.route:
            if type(t) is Customer:
                total_demand += t.demand

        return total_demand > self.config.payload_capacity

    # STATUS CALCULATION METHODS
    def calculate_total_distance(self):
        last = None
        dist = 0

        for t in self.route:
            if last is not None:
                dist += last.distance_to(t)
            last = t

        return dist

    def calculate_remaining_tank_capacity(self, until=None):
        last = None
        tank_capacity = self.config.tank_capacity
        for t in self.route:
            if last is not None:
                distance = last.distance_to(t)
                consumption = distance * self.config.fuel_consumption_rate
                tank_capacity -= consumption

                if until == t:
                    return tank_capacity

                if type(t) is CharingStation:
                    tank_capacity = self.config.tank_capacity

            last = t
        return tank_capacity

    def calculate_total_duration(self):
        elapsed_time = self.route[0].ready_time + self.route[0].service_time

        for i in range(1, len(self.route)):
            elapsed_time = elapsed_time + self.route[i - 1].distance_to(self.route[i]) / self.config.velocity

            if type(self.route[i]) is CharingStation:
                missing_energy = self.config.tank_capacity - self.calculate_remaining_tank_capacity(self.route[i])
                self.route[i] = missing_energy * self.config.charging_rate

            waiting_time = max(self.route[i].ready_time - elapsed_time, 0)
            elapsed_time += waiting_time
            elapsed_time += self.route[i].service_time

        return elapsed_time

    def calculate_dist_to_first_customer(self, reverse=False):
        dist = 0
        last = None

        if reverse:
            self.route.reverse()

        for t in self.route:
            if last is not None:
                dist += last.distance_to(t)
                if type(t) is Customer:
                    if reverse:
                        self.route.reverse()
                    return dist
            last = t

        return dist

    def get_first_customer(self, reverse=False):
        if reverse:
            self.route.reverse()

        for t in self.route:
            if type(t) is Customer:
                if reverse:
                    self.route.reverse()
                return t

    def append_route(self, new_route):
        if new_route.route[0] == self.depot:
            route_to_append = new_route[1:]

        if self.route[-1] == self.depot:
            self.route = self.route[0:-1]

        self.route = self.route + route_to_append

    def __str__(self):
        route_str = '['

        for t in self.route:
            route_str += t.id + ', '

        route_str += ']'
        return route_str

    def __repr__(self):
        route_str = '['

        for t in self.route:
            route_str += t.id + ', '

        route_str += ']'
        return route_str


class EVRPTWSolver:
    """
    A simple framework for solving the EVRPTW (Electronic Vehicle Routing Problem with Time Windows)
    """

    def __init__(self, construction_heuristic, meta_heuristic=None):
        """
        :param construction_heuristic: heuristic for constructing a initial solution
        :param meta_heuristic: meta heuristic, that improves the construction heuristic solution
        """
        self.construction_heuristic = construction_heuristic
        self.mea_heuristic = meta_heuristic

        self.construction_heuristic.set_generate_feasible_route_function(self.generate_feasible_route_from_to)

    def solve(self, problem_instance):
        solution = self.construction_heuristic.solve(problem_instance)

        if self.mea_heuristic:
            solution = self.mea_heuristic.improve(problem_instance, solution)

        dist = 0

        for route in solution:
            dist += route.calculate_total_distance()

        return dist, solution

    def generate_feasible_route_from_to(self, from_route, to_station, problem_instance) -> Route:
        from_route.route.append(to_station)

        while not from_route.is_feasible():
            if from_route.tw_constraint_violated():
                return None

            from_route.route.pop()
            reachable_stations = self.get_reachable_charging_stations(from_route.route[-1],
                                                                      from_route.calculate_remaining_tank_capacity(),
                                                                      from_route.route,
                                                                      problem_instance)

            if len(reachable_stations) == 0:
                return None

            best_station = min(reachable_stations, key=lambda x: x.distance_to(to_station))

            from_route.route.append(best_station)
            from_route.route.append(to_station)

        if to_station != problem_instance.depot and from_route.calculate_remaining_tank_capacity() < problem_instance.config.tank_capacity / 2:
            from_route.route.pop()

            reachable_stations = self.get_reachable_charging_stations(from_route.route[-1],
                                                                      from_route.calculate_remaining_tank_capacity(),
                                                                      from_route.route,
                                                                      problem_instance)

            if len(reachable_stations) > 0:
                best_station = min(reachable_stations, key=lambda x: x.distance_to(to_station))

                from_route.route.append(best_station)
                from_route.route.append(to_station)
            else:
                return None

            if not from_route.is_feasible():
                from_route.route.pop()
                from_route.route.pop()
                from_route.route.append(to_station)

        return from_route

    @staticmethod
    def get_reachable_charging_stations(cust: Customer, capacity: float, tabu_list: list,
                                        problem_instance) -> list:
        max_dist = capacity / problem_instance.config.fuel_consumption_rate
        reachable_stations = []

        for cs in problem_instance.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id and cs.id not in [x.id for x in tabu_list]:
                reachable_stations.append(cs)

        return reachable_stations
