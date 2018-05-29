import functools
import timeit
import numpy as np
from os import listdir

from evrptw_solver import EVRPTWSolver
from evrptw_utilities import load_problem_instance, write_solution_to_file, write_solution_stats_to_file
from heuristics.construction.beasley_heuristic import BeasleyHeuristic, nearest_neighbor_tolerance_min_due_date, \
    nearest_neighbor_tolerance_min_ready_time


def main():
    test_case_statistics = []

    print('NN heuristic with min ready time:')

    for tolerance in np.arange(1, 3, 0.1):
        distances = []
        construction_heuristic = BeasleyHeuristic(nearest_neighbor_tolerance_min_ready_time, [round(tolerance, 2)])

        solver = EVRPTWSolver(construction_heuristic)

        for file in listdir('_problem_instances/exercise_instances/'):
            if file.endswith('.txt'):
                problem_instance = load_problem_instance('_problem_instances/exercise_instances/' + file)

                duration = timeit.timeit(functools.partial(solver.solve, problem_instance), number=1) * 1000
                distance, solution = solver.solve(problem_instance)
                write_solution_to_file("_problem_solutions/solution_{0}".format(file), distance, solution)
                distances.append(distance)
                test_case_statistics.append((file, distance, duration))

        test_case_statistics.sort(key=lambda x: x[0])
        write_solution_stats_to_file('ex1_result_1126205.csv', test_case_statistics)
        print("{0:.2f}: {1:.2f}".format(tolerance, np.mean(distances)))


if __name__ == "__main__":
    main()
