from os import listdir

from evrptw_route_visualizer import RouteVisualizer
from routing_problem_solver import RoutingProblemSolver, RoutingProblemConfiguration
from targets import Target, CharingStation, Customer

import timeit


def parse_input(file):
    with open(file) as f:
        f.readline()  # ignore header

        target_line = f.readline()

        customers = []
        fuel_stations = []
        depot = None

        while target_line != '\n':
            stl = target_line.split()  # splitted target_line
            if stl[1] == 'd':
                depot = Target(stl[0], int(float(stl[2])), int(float(stl[3])), int(float(stl[5])), int(float(stl[6])),
                               int(float(stl[7])))
            elif stl[1] == 'f':
                new_target = CharingStation(stl[0], int(float(stl[2])), int(float(stl[3])), int(float(stl[5])),
                                            int(float(stl[6])), int(float(stl[7])))
                fuel_stations.append(new_target)
            elif stl[1] == 'c':
                new_target = Customer(stl[0], int(float(stl[2])), int(float(stl[3])), int(float(stl[4])),
                                      int(float(stl[5])), int(float(stl[6])), int(float(stl[7])))
                customers.append(new_target)

            target_line = f.readline()

        configuration_line = f.readline()
        tank_capacity = float(configuration_line.split('/')[1])  # q Vehicle fuel tank capacity

        configuration_line = f.readline()
        load_capacity = float(configuration_line.split('/')[1])  # C Vehicle load capacity

        configuration_line = f.readline()
        fuel_consumption_rate = float(configuration_line.split('/')[1])  # r fuel consumption rate

        configuration_line = f.readline()
        charging_rate = float(configuration_line.split('/')[1])  # g inverse refueling rate

        configuration_line = f.readline()
        velocity = float(configuration_line.split('/')[1])  # v average Velocity

        return RoutingProblemSolver(depot, fuel_stations, customers,
                                    RoutingProblemConfiguration(tank_capacity, load_capacity, fuel_consumption_rate,
                                                                charging_rate, velocity))


def write_solution_to_file(file, distance, routes):
    with open(file, 'w') as f:
        f.write('{0}\n'.format(round(distance, 3)))

        for r in routes:
            for t in r.route:
                f.write('{0}, '.format(t.id))
            f.write('\n')


def write_solution_stats_to_file(file,stat):
    with open(file, 'w') as result_file:
        result_file.writelines('testcase;distance;runtime (in ms)\n')
        for r in stat:
            result_file.write('{0} ; {1} ; {2}\n'.format(r[0], r[1], r[2]))


def main():
    test_case_statistics = []

    for n_size in range(1,11):
        total_dist = []
        for file in listdir('problem_instances/'):
            if file.endswith('.txt'):
                # print('solve {0}'.format(file))
                rps = parse_input('problem_instances/{0}'.format(file))
                rps.n_size=n_size
                runtime = round(timeit.timeit(rps.solve, number=1) * 1000, 3)

                write_solution_to_file('problem_solutions/solution_{0}'.format(file), rps.calculate_total_distance(),
                                       rps.routes)

                test_case_statistics.append((file, round(rps.calculate_total_distance(),3), runtime))
            total_dist.append(rps.calculate_total_distance())
            write_solution_stats_to_file('ex1_result_1126205.csv',test_case_statistics)

        visualizer = RouteVisualizer(rps)
        visualizer.plot()
        print('total dist for {0}:\t sum:{1}\t min:{2}\t max:{3}'.format(n_size, round(sum(total_dist),1), round(min(total_dist),1), round(max(total_dist), 1)))


if __name__ == "__main__":
    main()
