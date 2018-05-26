import sys

from targets import CharingStation, Customer


class RoutingProblemConfiguration:
    def __init__(self, tank_capacity, payload_capacity, fuel_consumption_rate, charging_rate, velocity):
        self.tank_capacity = tank_capacity
        self.payload_capacity = payload_capacity
        self.fuel_consumption_rate = fuel_consumption_rate
        self.charging_rate = charging_rate
        self.velocity = velocity


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


class RoutingProblemSolver:
    def __init__(self, depot, charging_stations, customers, config):
        self.depot = depot
        self.charging_stations = charging_stations
        self.customers = customers

        self.config = config

        self.giant_route = None
        self.routes = None
        self.n_size=1

    def generate_giant_route(self):
        last_position = self.depot
        self.giant_route = []
        while len(self.giant_route) != len(self.customers):
            possible_successors = [n for n in self.customers if n not in self.giant_route]
            possible_successors.sort(key=lambda n:n.distance_to(last_position))

            if len(possible_successors) >= self.n_size:
                successor = min(possible_successors[:self.n_size], key=lambda n:n.due_date)
                self.giant_route.append(successor)
            else:
                successor = min(possible_successors, key=lambda n: n.due_date)
                self.giant_route.append(successor)

            last_position = successor

    def generate_feasible_route_from_to(self, from_route, to_station):
        from_route.route.append(to_station)

        while not from_route.is_feasible():
            if from_route.tw_constraint_violated():
                return None

            from_route.route.pop()
            reachable_stations = self.get_reachable_charging_stations(from_route.route[-1],
                                                                      from_route.calculate_remaining_tank_capacity(),
                                                                      from_route.route)

            if len(reachable_stations) == 0:
                return None

            best_station = min(reachable_stations, key=lambda x: x.distance_to(to_station))

            from_route.route.append(best_station)
            from_route.route.append(to_station)

        if to_station != self.depot and from_route.calculate_remaining_tank_capacity() < self.config.tank_capacity / 2:
            from_route.route.pop()

            reachable_stations = self.get_reachable_charging_stations(from_route.route[-1],
                                                                      from_route.calculate_remaining_tank_capacity(),
                                                                      from_route.route)

            if len(reachable_stations) > 0:
                best_station = min(reachable_stations, key=lambda x: x.distance_to(to_station))

                from_route.route.append(best_station)
                from_route.route.append(to_station)

            if not from_route.is_feasible():
                from_route.route.pop()
                from_route.route.pop()
                from_route.route.append(to_station)

        return from_route

    def get_reachable_charging_stations(self, cust: Customer, capacity: float, tabu_list: list) -> list:
        max_dist = capacity / self.config.fuel_consumption_rate
        reachable_stations = []

        for cs in self.charging_stations:
            if cs.distance_to(cust) <= max_dist and cust.id != cs.id and cs.id not in [x.id for x in tabu_list]:
                reachable_stations.append(cs)

        return reachable_stations

    def solve(self):
        self.routes = []

        # self.__generate_initial_routes()
        self.generate_giant_route()
        self.giant_route.reverse()

        new_route = Route(self.config, self.depot)
        last_customer = None
        while len(self.giant_route) != 0:
            c = self.giant_route.pop()
            extended_route = self.generate_feasible_route_from_to(new_route, c)

            if extended_route is None:
                self.giant_route.append(c)
                while new_route.route[-1] != last_customer:
                    new_route.route.pop()
                new_route = self.generate_feasible_route_from_to(new_route, self.depot)
                self.routes.append(new_route)
                new_route = Route(self.config, self.depot)
                continue
            else:
                new_route = extended_route

            extended_route = self.generate_feasible_route_from_to(new_route, self.depot)

            if extended_route is None:
                self.giant_route.append(c)
                while new_route.route[-1] != last_customer:
                    new_route.route.pop()
                new_route = self.generate_feasible_route_from_to(new_route, self.depot)
                self.routes.append(new_route)
                new_route = Route(self.config, self.depot)
                continue
            else:
                new_route = extended_route

            if not new_route.is_feasible():
                while new_route.route[-1] != last_customer:
                    new_route.route.pop()
                new_route = self.generate_feasible_route_from_to(new_route, self.depot)
                self.routes.append(new_route)
                new_route = Route(self.config, self.depot)
                self.giant_route.append(c)
            else:
                while new_route.route[-1] != c:
                    new_route.route.pop()

                last_customer = c

        new_route = self.generate_feasible_route_from_to(new_route, self.depot)
        self.routes.append(new_route)

    def calculate_total_distance(self):
        total_dist = 0
        for r in self.routes:
            total_dist += r.calculate_total_distance()

        return total_dist
