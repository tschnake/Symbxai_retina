#!/bin/bash

logfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/logs/"
resultfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/intermediate_results/"

# Define the variable dry_run
dry_run="false"

# Check if dry_run is true
if [ "$dry_run" = "true" ]; then
  range='0 1'
else
  range='(seq 0 100)'
fi



for ids in $treebank_ids; do
  sbatch --mem=15G perform_perturbation_apptainer_wrapper.sh --sample_range "[${ids}]" --result_dir "${resultfolder}"

done
