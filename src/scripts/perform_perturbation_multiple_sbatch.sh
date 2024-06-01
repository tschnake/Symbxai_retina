#/bin/bash

logfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/logs/"
resultfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/intermediate_results/"

# Define the variable dry_run
dry_run="true"

data_mode="imdb"
# Check if dry_run is true
if [ "$dry_run" = "true" ]; then
  range='0 1'
else
  range=$(seq 0 100)
fi

# Define the two lists
list1=("minimize" "maximize")
list2=("removal" "generation")
for auc_task in "${list1[@]}"; do
    for perturbation_type in "${list2[@]}"; do
        for ids in $range; do
          sbatch --mem=15G perform_perturbation_apptainer_wrapper.sh --sample_range "[${ids}]" --result_dir "${resultfolder}" --data_mode "${data_mode}" --auc_task "$auc_task" --perturbation_type "$perturbation_type"
        done
    done
done
