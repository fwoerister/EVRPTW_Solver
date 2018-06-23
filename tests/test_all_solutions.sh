#!/bin/sh

for file in ../_problem_instances/exercise_instances/*.txt; do
  #echo ${file##*/}
  java -jar evrptw-verifier-0.2.0.jar ../_problem_instances/exercise_instances/${file##*/} ../_problem_solutions/solution_${file##*/}
done

echo "Test meta solutions"

for file in ../_problem_instances/exercise_instances/*.txt; do
  #echo ${file##*/}
  java -jar evrptw-verifier-0.2.0.jar ../_problem_instances/exercise_instances/${file##*/} ../_meta_solutions/solution_${file##*/}
done
