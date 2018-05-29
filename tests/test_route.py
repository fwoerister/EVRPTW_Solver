import unittest

from evrptw_solver import RoutingProblemConfiguration, Route
from targets import Target, Customer, CharingStation


class TestRoute(unittest.TestCase):
    def setUp(self):
        d = Target('DO',0,0,0,300,0)
        config = RoutingProblemConfiguration(70.0,200.0,1.0,0.5,1.0)
        self.r = Route(config, d)


    def test_infeasible_timewindow(self):
        cust_1 = Customer('C1',0,30,20,30,60,10)
        cust_2 = Customer('C2',0,15,20, 0,54,10)

        self.r.route.append(cust_1)
        self.r.route.append(cust_2)

        self.assertTrue(self.r.tw_constraint_violated())

    def test_infeasibel_timewindow_with_charing_station(self):
        cust_1 = Customer('C1', 0, 30, 20, 30, 60, 10)
        charging_station = CharingStation('S1',0,15,290,300,0)

        self.r.route.append(cust_1)
        self.r.route.append(charging_station)
        self.r.route.append(self.r.route[0])

        self.assertTrue(self.r.tw_constraint_violated())

    def test_feasible_timewindow(self):
        cust_1 = Customer('C1', 0, 30, 20, 30, 60, 10)
        cust_2 = Customer('C2', 0, 15, 20, 0, 55, 10)

        self.r.route.append(cust_1)
        self.r.route.append(cust_2)

        self.assertFalse(self.r.tw_constraint_violated())

