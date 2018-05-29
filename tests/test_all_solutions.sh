#!/bin/sh

for file in ../_problem_instances/*.txt; do
  #echo ${file##*/}
  java -jar evrptw-verifier-0.2.0.jar ../_problem_instances/${file##*/} ../_problem_solutions/solution_${file##*/}
done
