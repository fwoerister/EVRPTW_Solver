from evrptw_solver import RoutingProblemInstance, RoutingProblemConfiguration
from targets import Target, CharingStation, Customer


def load_problem_instance(file):
    with open(file) as f:
        f.readline()  # ignore header

        target_line = f.readline()

        customers = []
        fuel_stations = []
        depot = None

        while target_line != '\n':
            stl = target_line.split()  # splitted target_line
            if stl[1] == 'd':
                depot = Target(stl[0], int(float(stl[2])), int(float(stl[3])), int(float(stl[5])),
                               int(float(stl[6])),
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

        return RoutingProblemInstance(RoutingProblemConfiguration(tank_capacity, load_capacity, fuel_consumption_rate,
                                                                  charging_rate, velocity), depot, customers,
                                      fuel_stations)


def write_solution_to_file(file, distance, routes):
    with open(file, 'w') as f:
        f.write('{0}\n'.format(round(distance, 3)))

        for r in routes:
            for t in r.route:
                f.write('{0}, '.format(t.id))
            f.write('\n')


def write_solution_stats_to_file(file, stat):
    with open(file, 'w') as result_file:
        result_file.writelines('testcase;distance;runtime (in ms)\n')
        for r in stat:
            result_file.write('{0} ; {1} ; {2}\n'.format(r[0], r[1], r[2]))

