#!/bin/bash

logfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/logs/"
resultfolder="/home/thomas_schnake/ResearchProjects/symbXAI_project/local_experiments/intermediate_results/"

# Define the variable dry_run
dry_run="false"
attribution_mode="cov(q,f)"

# Check if dry_run is true
if [ "$dry_run" = "true" ]; then
  max_and_order=1
  harsanyi_maxorder=1
  max_setsize=1
  treebank_ids='6072'
  huggingface_ids=""
else
  max_and_order=3
  harsanyi_maxorder=4
  max_setsize=1000
  treebank_ids='138 259 324 385 413 851 1469 1518 1532 1614 1716 2086 2555 3287 3423 3427 3437 3493 3617 4328 4634 4724 5177 5356 5676 5703 5734 6072 6347 6593 7349 7411 7560 8015 8145 8231'
  huggingface_ids=""
fi


datamode='sst_treebank'
for ids in $treebank_ids; do
  sbatch --mem=150G run_query_auto_search_apptainer_wrapper.sh --sample_range "[${ids}]" --max_and_order ${max_and_order} --datamode "${datamode}" --harsanyi_maxorder ${harsanyi_maxorder} --weight_mode "occlusion" --max_setsize ${max_setsize} --logfolder "${logfolder}" --resultfolder "${resultfolder}" --nb_cores 1 --attribution_mode "${attribution_mode}" &
done

datamode='sst_huggingface'
for ids in $huggingface_ids; do
  sbatch --mem=150G run_query_auto_search_apptainer_wrapper.sh --sample_range "[${ids}]"  --max_and_order ${max_and_order} --datamode "${datamode}" --harsanyi_maxorder ${harsanyi_maxorder} --weight_mode "occlusion" --max_setsize ${max_setsize} --logfolder "${logfolder}" --resultfolder "${resultfolder}" --nb_cores 10 &
done
