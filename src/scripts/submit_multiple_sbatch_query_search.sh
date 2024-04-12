#!/bin/bash

logfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/logs/"
resultfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/intermediate_results/"

# Define the variable dry_run
dry_run="true"

# Check if dry_run is true
if [ "$dry_run" = "true" ]; then
  max_and_order=1
  harsanyi_maxorder=1
  max_setsize=1
  treebank_ids="259 138 1481 4113"
  huggingface_ids="0 1 2 4 5 6 7 8 10 11"
else
  max_and_order=3
  harsanyi_maxorder=5
  max_setsize=-1
  treebank_ids="259 138 1481 4113"
  huggingface_ids="0 1 2 4 5 6 7 8 10 11"
fi


datamode='sst_treebank'
for ids in $treebank_ids; do
  sbatch --mem=256G run_query_auto_search_apptainer_wrapper.sh --sample_range "[${ids}]" --comp_mode 'harsanyi' --max_and_order ${max_and_order} --datamode "${datamode}" --harsanyi_maxorder ${harsanyi_maxorder} --weight_modes "['occlusion']" --max_setsize ${max_setsize} --logfolder "${logfolder}" --resultfolder "${resultfolder}" &
  # sbatch --mail-user=t.schnake@tu-berlin.de run_query_auto_search_apptainer_wrapper.sh --sample_range "[${ids}]" --comp_mode 'directly' --max_and_order ${max_and_order} --datamode "${datamode}" --weight_modes "['occlusion','sigificance-1','significance-2','significance-3']" --logfolder "${logfolder}" --resultfolder "${resultfolder}" &
  #sbatch --mail-user=t.schnake@tu-berlin.de run_query_auto_search_apptainer_wrapper.sh --sample_range [259] --comp_mode 'harsanyi' --max_and_order 3 --harsanyi_maxorder 1 --weight_modes "['occlusion','significance-1','significance-2','significance-3']" --logfolder ${logfolder} --resultfolder ${resultfolder} &
done

datamode='sst_huggingface'
for ids in $huggingface_ids; do
  sbatch --mem=256G run_query_auto_search_apptainer_wrapper.sh --sample_range "[${ids}]" --comp_mode 'harsanyi' --max_and_order ${max_and_order} --datamode "${datamode}" --harsanyi_maxorder ${harsanyi_maxorder} --weight_modes "['occlusion']" --max_setsize ${max_setsize} --logfolder "${logfolder}" --resultfolder "${resultfolder}" &
  # sbatch --mail-user=t.schnake@tu-berlin.de run_query_auto_search_apptainer_wrapper.sh --sample_range "[${ids}]" --comp_mode 'directly' --max_and_order ${max_and_order} --datamode "${datamode}" --weight_modes "['occlusion','sigificance-1','significance-2','significance-3']" --logfolder "${logfolder}" --resultfolder "${resultfolder}" &
  #sbatch --mail-user=t.schnake@tu-berlin.de run_query_auto_search_apptainer_wrapper.sh --sample_range [259] --comp_mode 'harsanyi' --max_and_order 3 --harsanyi_maxorder 1 --weight_modes "['occlusion','significance-1','significance-2','significance-3']" --logfolder ${logfolder} --resultfolder ${resultfolder} &
done
