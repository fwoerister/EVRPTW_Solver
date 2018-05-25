#!/bin/sh

for file in ../problem_instances/*.txt; do
  #echo ${file##*/}
  java -jar evrptw-verifier-0.2.0.jar ../problem_instances/${file##*/} ../problem_solutions/solution_${file##*/}
done

java -jar evrptw-verifier-0.2.0.jar ../problem_instances/c103_21.txt ../problem_solutions/solution_c103_21.txt
