#!/bin/bash
#SBATCH --job-name=my_cpu_job
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/logs/job-%j.out

project_rep="/home/thomas_schnake/ResearchProjects/symbXAI_project/"
apptainer run ${project_rep}symbxai_container.sif python ${project_rep}symbolicXAI_code/src/scripts/run_query_auto_search.py "$@"
