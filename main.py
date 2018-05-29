import timeit
from os import listdir

from evrptw_solver import EVRPTWSolver
from evrptw_utilities import load_problem_instance, write_solution_to_file, write_solution_stats_to_file
from heuristics.construction.beasley_heuristic import BeasleyHeuristic, nearest_neighbor_tolerance_giant_tour, k_nearest_neighbor_giant_tour

import functools


def main():
    test_case_statistics=[]

    for file in listdir('_problem_instances/'):
        if file.endswith('.txt'):
            problem_instance = load_problem_instance('_problem_instances/'+file)

            beasley_heuristic_1 = BeasleyHeuristic(nearest_neighbor_tolerance_giant_tour, [1.3])
            beasley_heuristic_2 = BeasleyHeuristic(k_nearest_neighbor_giant_tour, [3])
            solver = EVRPTWSolver(beasley_heuristic_2)

            duration = timeit.timeit(functools.partial(solver.solve, problem_instance), number=1)*1000
            distance, solution = solver.solve(problem_instance)
            write_solution_to_file("_problem_solutions/solution_{0}".format(file), distance, solution)

            test_case_statistics.append((file, distance, duration))

    write_solution_stats_to_file('ex1_result_1126205.csv', test_case_statistics)


if __name__ == "__main__":
    main()
