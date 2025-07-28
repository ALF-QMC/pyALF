#!/bin/bash

for i in minimal_example.ipynb compiling_running.ipynb postprocessing_basic_analysis.ipynb postprocessing_custom_obs.ipynb postprocessing_check_warmup.ipynb postprocessing_symmetrization.ipynb cli.ipynb; do
  time jupyter nbconvert --to notebook --execute $i --inplace
done
