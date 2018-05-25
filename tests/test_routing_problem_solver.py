import unittest

from routing_problem_solver import RoutingProblemConfiguration, Route, RoutingProblemSolver
from targets import Target, Customer, CharingStation


class TestRoutingProblemSolver(unittest.TestCase):
    def setUp(self):
        self.depot = Target('DO', 0, 0, 0, 300, 0)
        self.config = RoutingProblemConfiguration(50.0, 200.0, 1.0, 0.5, 1.0)
        self.r = Route(self.config, self.depot)

    def test_generate_feasible_route_from_to(self):
        cust_1 = Customer('C1', 0, 100, 10, 0, 300, 0)
        station_1 = CharingStation('S1', 0, 40, 0, 300, 0)
        station_2 = CharingStation('S2', 0, 80, 0, 300, 0)

        self.rps = RoutingProblemSolver(self.depot, [station_1, station_2], [cust_1], self.config)

        from_route = Route(self.config, self.depot)
        from_route = self.rps.generate_feasible_route_from_to(from_route, cust_1)
        self.assertEqual(from_route.route, [self.depot, station_1, station_2, cust_1])

    def test_generate_infeasible_route_from_to(self):
        cust_1 = Customer('C1', 0, 100, 10, 0, 300, 0)
        station_2 = CharingStation('S2', 0, 80, 0, 300, 0)

        self.rps = RoutingProblemSolver(self.depot, [station_2], [cust_1], self.config)

        from_route = Route(self.config, self.depot)
        from_route = self.rps.generate_feasible_route_from_to(from_route, cust_1)
        self.assertIsNone(from_route)

    def test_generate_initial_routes(self):
        # TODO: define testdata!
        cust_1 = Customer('C1', 0, 25, 10, 0, 300, 0)
        cust_2 = Customer('C2', 0, -30, 10, 0, 300, 0)
        cust_3 = Customer('C3', 20, 20, 10, 0, 300, 0)
        cust_4 = Customer('C4', -15, 20, 10, 0, 300, 0)

        station_1 = CharingStation('S1', 0, 40, 0, 300, 0)
        station_2 = CharingStation('S2', 10, 10, 0, 300, 0)

        self.rps = RoutingProblemSolver(self.depot, [station_1, station_2],
                                        [cust_1, cust_2, cust_3, cust_4], self.config)

        self.rps.solve(only_preprocessing=True)
