#!/bin/bash

 
echo "Activating virtual environment..."
source /home/vksharma/Documents/python_virtual_env_1/bin/activate


# output_log_folder="output_logs/august_2025_dispersive_j_nn_${j_nn}_omega_${omega_0}_omega_1_${omega_1}_gamma_${gamma_0}"
# mkdir -p "$output_log_folder"
# echo "Output log folder created: $output_log_folder"


# mu_val="${chemical_potential_vals[$SLURM_ARRAY_TASK_ID]}"
# echo "Running with j_nn $j_nn omega $omega_0 omega_1 $omega_1 gamma_0 $gamma_0 and chemical potential = $mu_val"

cd /home/vksharma/Documents/ite_real_space_model_lambda_ln/

python3.12 -u main_nnn_normal.py > "ITE_SC_normal.log" 2>&1

wait
echo "All jobs finished."