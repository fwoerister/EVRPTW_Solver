from evrptw_solver import RoutingProblemInstance, RoutingProblemConfiguration, Route
from targets import Target, CharingStation, Customer
import numpy as np
import matplotlib.pyplot as plt


def load_problem_instance(file):
    with open(file) as f:
        f.readline()  # ignore header

        target_line = f.readline()

        customers = []
        fuel_stations = []
        depot = None

        while target_line != '\n':
            stl = target_line.split()  # splitted target_line
            idx = int(stl[0][1:])

            if stl[1] == 'd':
                depot = Target(stl[0], idx, int(float(stl[2])), int(float(stl[3])), int(float(stl[5])),
                               int(float(stl[6])),
                               int(float(stl[7])))
            elif stl[1] == 'f':
                new_target = CharingStation(stl[0], idx, int(float(stl[2])), int(float(stl[3])), int(float(stl[5])),
                                            int(float(stl[6])), int(float(stl[7])))
                fuel_stations.append(new_target)
            elif stl[1] == 'c':
                new_target = Customer(stl[0], idx, int(float(stl[2])), int(float(stl[3])), int(float(stl[4])),
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

        return RoutingProblemInstance(RoutingProblemConfiguration(tank_capacity, load_capacity, fuel_consumption_rate,
                                                                  charging_rate, velocity), depot, customers,
                                      fuel_stations)


def load_solution(file):
    with open(file, 'r') as f:
        distance = float(f.readline())
        solution = []

        for line in f:
            # [:-1] is needed, otherwise the last element of the list would be an empty string
            route = line.split(', ')[:-1]

            solution.append(route)

        return distance, solution


def write_solution_to_file(file, distance, routes):
    with open(file, 'w') as f:
        f.write('{0}\n'.format(round(distance, 3)))

        for r in routes:
            if type(r) is Route:
                if len(r.route) > 2:
                    for t in r.route:
                        f.write('{0}, '.format(t.id))
                    f.write('\n')
            elif type(r) is list:
                if len(r) > 2:
                    for v in r:
                        f.write('{0}, '.format(v))
                    f.write('\n')


def write_solution_stats_to_file(file, stat, style='csv'):
    with open(file, 'w') as result_file:
        if style == 'csv':
            result_file.writelines('testcase;distance;runtime (in ms)\n')
            for r in stat:
                result_file.write('{0} ; {1} ; {2}\n'.format(r[0], round(r[1], 3), round(r[2], 3)))
        elif style == 'latex':
            result_file.write("\\begin{table}[t]\n")
            result_file.write("\\label{tab:result}\n")
            result_file.write("\\begin{tabular}{lrr}\n")
            result_file.write("\\toprule\n")
            result_file.write("instance & distance & runtime (in ms) \\\\ \n")
            result_file.write("\\midrule")

            for r in stat:
                result_file.write('{0} & {1} & {2} \\\\ \n'.format(r[0], round(r[1], 3), round(r[2], 3)))

            result_file.write("\\bottomrule \n")
            result_file.write("\\end{tabular} \n")
            result_file.write("\\end{table} \n")


def write_meta_heuristic_result_statistic_to_file(file, dist_stat, time_stat):
    with open(file, 'w') as f:
        f.writelines('file;distance(avg);distance(std);distance(min);time(avg);time(std);time(max)\n')

        instances = list(dist_stat.keys())
        instances.sort()

        for i in instances:
            f.writelines(
                '{0};{1};{2};{3};{4};{5};{6}\n'.format(i, np.average(dist_stat[i]), np.std(dist_stat[i]),
                                                     np.min(dist_stat[i]),
                                                     np.average(time_stat[i]), np.std(time_stat[i]),
                                                     np.max(time_stat[i])))
