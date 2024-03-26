#!/bin/bash

# Initialize boolean variable
on_cluster=false

# Check if "--on_cluster" argument is present
if [[ "$1" == "--on_cluster" ]]; then
    on_cluster=true
fi

# Use the boolean variable as needed
if [ "$on_cluster" = true ]; then
	script_path="/home/thomas_schnake/ResearchProjects/symbXAI_project/symbolicXAI_code/src/scripts/"
else
	script_path="/Users/thomasschnake/Research/Projects/symbolic_xai/symbolicXAI_github/src/scripts/"
fi



python ${script_path}run_query_auto_search.py --sample_range [259] --comp_mode 'harsanyi' --max_and_order 3 --harsanyi_maxorder 1 --weight_modes "['occlusion','shapley']" 


