#!/bin/bash

model_name="meta-llama/Llama-3.1-8B"
model_short_name=$(echo $model_name | awk -F'/' '{print $2}')
vocab_size=128256
sequence_length=128
k=512
spec_length=1
seed=1
num_cores=16

qpc_base_dir="/local/mnt/workspace/sanising/quic-github/efficient-transformers"
qeff_home_dir="/prj/crd/austin/validation/scratch/users/sanising/efficient-transformers/aic_1_20_0_119/"
hf_home_dir="/prj/crd/austin/validation/scratch/users/sanising/models"

batch_sizes=(1 2 8 16)
ctx_lengths=(256 512)
tensor_slices=(1 2 4 8)

for num_devices in "${tensor_slices[@]}"; do

    # ---Without sampler---
    base_cmd_wo="python3 ${qpc_base_dir}/run_sampler_copy2.py --model_name $model_name --no_include_sampler --no_is_tlm --no_return_pdfs --sequence_length $sequence_length --k $k --spec_length $spec_length --seed $seed --parent_directory $qeff_home_dir --num_devices $num_devices --num_cores $num_cores --kv-cache-dtype mxint8 --quantization mxfp6"

    for batch_size in "${batch_sizes[@]}"; do
        for ctx_length in "${ctx_lengths[@]}"; do
            script_name="${qpc_base_dir}/sampling_scripts3/compile_${model_short_name}_wo_sampler_ts${num_devices}c${num_cores}_bs${batch_size}_ctx${ctx_length}_seqlen${sequence_length}_speclen${spec_length}_k${k}.sh"
            dump_ir_graphs_folder="/local/mnt/qt_drive/users/sanising/efficient-transformers/r760/dump_ir_graphs/${model_short_name}_wo_sampler_ts${num_devices}c${num_cores}_bs${batch_size}_ctx${ctx_length}_seqlen${sequence_length}_speclen${spec_length}_k${k}/"
            QAIC_COMPILER_OPTS_UNSUPPORTED="-aic-split-retained-state-io -debug-glow --aic-dump-graphs-dir=${dump_ir_graphs_folder}"

            cat <<EOL > $script_name
#!/bin/bash

export QEFF_HOME=$qeff_home_dir
export HF_HOME=$hf_home_dir
# export AIC_TOOLS_DIR=/local/mnt/workspace/sanising/build_tools/
export TMPDIR=/local/mnt/workspace/sanising/tmp/
# export QAIC_COMPILER_LIB=/local/mnt/workspace/sanising/compiler-glow/build/compiler-qualcomm-release-assert/lib/QAicCompilerManagers/libQAicCompiler.so
# export QAIC_COMPILER_OPTS_UNSUPPORTED="$QAIC_COMPILER_OPTS_UNSUPPORTED"

# mkdir -p $dump_ir_graphs_folder
# chmod 777 $dump_ir_graphs_folder

cmd="$base_cmd_wo --batch_size $batch_size --ctx_length $ctx_length"
echo "Running command: \$cmd"
\$cmd
EOL

            chmod +x $script_name
            echo "Created script: $script_name"
        done
    done

    # ---With sampler---
    base_cmd_w="python3 ${qpc_base_dir}/run_sampler_copy2.py --model_name $model_name --include_sampler --no_is_tlm --no_return_pdfs --sequence_length $sequence_length --k $k --spec_length $spec_length --seed $seed --parent_directory $qeff_home_dir --num_devices $num_devices --num_cores $num_cores --kv-cache-dtype mxint8 --quantization mxfp6"

    for batch_size in "${batch_sizes[@]}"; do
        for ctx_length in "${ctx_lengths[@]}"; do
            script_name="${qpc_base_dir}/sampling_scripts3/compile_${model_short_name}_w_sampler_ts${num_devices}c${num_cores}_bs${batch_size}_ctx${ctx_length}_seqlen${sequence_length}_speclen${spec_length}_k${k}.sh"
            dump_ir_graphs_folder="/local/mnt/qt_drive/users/sanising/efficient-transformers/r760/dump_ir_graphs/${model_short_name}_w_sampler_ts${num_devices}c${num_cores}_bs${batch_size}_ctx${ctx_length}_seqlen${sequence_length}_speclen${spec_length}_k${k}/"
            QAIC_COMPILER_OPTS_UNSUPPORTED="-aic-split-retained-state-io -debug-glow --aic-dump-graphs-dir=${dump_ir_graphs_folder}"

            cat <<EOL > $script_name
#!/bin/bash

export QEFF_HOME=$qeff_home_dir
export HF_HOME=$hf_home_dir
# export AIC_TOOLS_DIR=/local/mnt/workspace/sanising/build_tools/
export TMPDIR=/local/mnt/workspace/sanising/tmp/
# export QAIC_COMPILER_LIB=/local/mnt/workspace/sanising/compiler-glow/build/compiler-qualcomm-release-assert/lib/QAicCompilerManagers/libQAicCompiler.so
# export QAIC_COMPILER_OPTS_UNSUPPORTED="$QAIC_COMPILER_OPTS_UNSUPPORTED"

# mkdir -p $dump_ir_graphs_folder
# chmod 777 $dump_ir_graphs_folder

cmd="$base_cmd_w --batch_size $batch_size --ctx_length $ctx_length"
echo "Running command: \$cmd"
\$cmd
EOL

            chmod +x $script_name
            echo "Created script: $script_name"
        done
    done

done
