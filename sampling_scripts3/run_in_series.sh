#!/bin/bash

model_name="meta-llama/Llama-3.1-8B"
model_short_name=$(echo $model_name | awk -F'/' '{print $2}')
sequence_length=128
k=512
spec_length=1
seed=1
parent_directory="/local/mnt/workspace/sanising/quic-github/efficient-transformers/"
num_cores=16

batch_sizes=(1 2 8 16)
ctx_lengths=(256 512)
tensor_slices=(1 2 4 8)

for num_devices in "${tensor_slices[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for ctx_length in "${ctx_lengths[@]}"; do

            # Without sampler
            script_name="${parent_directory}sampling_scripts3/compile_${model_short_name}_wo_sampler_ts${num_devices}c${num_cores}_bs${batch_size}_ctx${ctx_length}_seqlen${sequence_length}_speclen${spec_length}_k${k}.sh"
            cmd="bash $script_name 2>&1 | tee ${script_name%.sh}.log"
            echo $cmd
            time eval $cmd

            # With sampler
            script_name="${parent_directory}sampling_scripts3/compile_${model_short_name}_w_sampler_ts${num_devices}c${num_cores}_bs${batch_size}_ctx${ctx_length}_seqlen${sequence_length}_speclen${spec_length}_k${k}.sh"
            cmd="bash $script_name 2>&1 | tee ${script_name%.sh}.log"
            echo $cmd
            time eval $cmd       
        done
    done
done
