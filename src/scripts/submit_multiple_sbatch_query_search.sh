#!/bin/bash

logfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/logs/"
resultfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/intermediate_results/"


datamode='sst_treebank'
treebank_ids="259"

max_and_order=3
harsanyi_maxorder=1

for ids in $treebank_ids; do
  sbatch --mail-user=t.schnake@tu-berlin.de run_query_auto_search_apptainer_wrapper.sh --sample_range [259] --comp_mode 'harsanyi' --max_and_order ${max_and_order} --harsanyi_maxorder ${harsanyi_maxorder} --weight_modes "['occlusion','shapley']" --logfolder ${logfolder} --resultfolder ${resultfolder} &
  #sbatch --mail-user=t.schnake@tu-berlin.de run_query_auto_search_apptainer_wrapper.sh --sample_range [259] --comp_mode 'harsanyi' --max_and_order 3 --harsanyi_maxorder 1 --weight_modes "['occlusion','significance-1','significance-2','significance-3']" --logfolder ${logfolder} --resultfolder ${resultfolder} &
done
