import argparse
import numpy as np
import subprocess
import torch

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.transformers.sampler.test_sampler.make_inputs import write_io_files


def initialize_model(model_name, qaic_config):
    qeff_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        num_hidden_layers=2,
        continuous_batching=True,
        qaic_config=qaic_config,
    )
    print(f"{model_name} optimized for AI 100 \n", qeff_model)
    return qeff_model


def export_model(qeff_model, directory_path):
    generated_onnx_path = qeff_model.export(export_dir=directory_path)
    print(generated_onnx_path)
    return generated_onnx_path


def compile_model(
    qeff_model, compile_directory, sequence_length, ctx_length, batch_size, num_devices, num_cores, spec_length, device_id=0, mxint8_kv_cache=False, mxfp6_matmul=False, 
):
    # if onnx_path:
    #     generated_qpc_path = qeff_model.compile(
    #         onnx_path=onnx_path,
    #         prefill_seq_len=sequence_length,
    #         ctx_len=ctx_length,
    #         full_batch_size=batch_size,
    #         num_devices=num_devices,
    #         num_cores=num_cores,
    #         num_speculative_tokens=spec_length - 1,
    #         # device_id=device_id,
    #     )
    # else:
    generated_qpc_path = qeff_model.compile(
        compile_dir=compile_directory,
        prefill_seq_len=sequence_length,
        ctx_len=ctx_length,
        # batch_size=batch_size,
        full_batch_size=batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        # device_id=device_id,
        mxint8_kv_cache=mxint8_kv_cache,
        mxfp6_matmul=mxfp6_matmul,
    )
    print(generated_qpc_path)
    return generated_qpc_path


def generate_inputs(vocab_size, sequence_length, ctx_length, batch_size, seed, parent_directory):
    torch.manual_seed(seed)
    prompt_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    output_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, ctx_length))

    repetition_penalty_retain_state = torch.zeros(batch_size, vocab_size, dtype=torch.int32)
    presence_penalty_retain_state = torch.zeros(batch_size, vocab_size, dtype=torch.int32)

    repetition_penalty_retain_state.scatter_(1, prompt_token_ids, 1)
    repetition_penalty_retain_state.scatter_(1, output_token_ids[:, :-1], 1)
    presence_penalty_retain_state.scatter_(1, output_token_ids[:, :-1], 1)

    repetition_penalties = torch.randint(1, 21, (batch_size,)) / 10.0
    presence_penalties = torch.randint(-10, 10, (batch_size,)) / 10.0

    temperatures = torch.randint(1, 11, (batch_size,)) / 10.0

    top_ks = torch.randint(1, 512, (batch_size,), dtype=torch.int32)
    top_ps = torch.randint(50, 100, (batch_size,)) / 100.0
    min_ps = torch.randint(50, 100, (batch_size,)) / 100.0

    pseudo_random_generator = torch.Generator()
    random_numbers = torch.rand(batch_size, generator=pseudo_random_generator)

    inputs = {
        "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "position_ids": np.tile(np.arange(ctx_length - 1, ctx_length), (batch_size, 1)),
        "repetition_penalty_retain_state": repetition_penalty_retain_state.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "presence_penalty_retain_state": presence_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
        "min_ps": min_ps.detach().cpu().numpy(),
        "random_numbers": random_numbers.detach().cpu().numpy(),
    }
    print("Inputs", inputs)
    write_io_files(
        inputs,
        dict(),
        parent_directory,
        "data",
        "model_end_to_end",
        True,
        False,
    )
    return inputs


def run_qpc(generated_qpc_path, parent_directory):
    cmd = [
        "/opt/qti-aic/exec/qaic-api-test",
        "-t",
        f"{generated_qpc_path}",
        "-n",
        "1",
        "--aic-profiling-type",
        "stats",
        "--aic-profiling-start-iter",
        "0",
        "--aic-profiling-num-samples",
        "1",
        "--aic-batch-json-input",
        f"{parent_directory}model_end_to_end.json",
        "--write-output-dir",
        f"{parent_directory}outputs_from_qpcs/",
    ]

    qeff_output_offline = subprocess.run(cmd, capture_output=True, text=True)
    print(qeff_output_offline.stdout)
    if qeff_output_offline.returncode != 0:
        print(qeff_output_offline.stderr)
    return qeff_output_offline


def main():
    parser = argparse.ArgumentParser(description="Run QEfficient model and sampler")
    parser.add_argument(
        "--model_name", type=str, required=True, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    # parser.add_argument("--num_hidden_layers", type=int, required=False, default=2)    
    parser.add_argument("--include_sampler", action="store_true", help="Include the sampler")
    parser.add_argument("--no_include_sampler", action="store_false", dest="include_sampler", help="Do not include the sampler")
    parser.set_defaults(include_sampler=True)
    
    parser.add_argument("--is_tlm", action="store_true")
    parser.add_argument("--no_is_tlm", action="store_false", dest="is_tlm")
    parser.set_defaults(is_tlm=False)

    parser.add_argument("--return_pdfs", action="store_true")
    parser.add_argument("--no_return_pdfs", action="store_false", dest="return_pdfs", help="Do not return PDFs")
    parser.set_defaults(return_pdfs=False)
    
    parser.add_argument("--sequence_length", type=int, required=True, default=128)
    parser.add_argument("--batch_size", type=int, required=True, default=1)
    parser.add_argument("--ctx_length", type=int, required=True, default=256)
    parser.add_argument("--k", type=int, required=True, default=512)
    parser.add_argument("--spec_length", type=int, required=True, default=1)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument(
        "--parent_directory",
        type=str,
        required=True,
        default="/local/mnt/qt_drive/users/sanising/efficient-transformers",
    )
    parser.add_argument("--num_devices", type=int, required=False, default=1)
    parser.add_argument("--num_cores", type=int, required=False, default=16)
    parser.add_argument("--device_id", type=int, required=False, default=0)
    parser.add_argument("--kv-cache-dtype", type=str, required=False, default="mxint8")
    parser.add_argument("--quantization", type=str, required=False, default="mxfp6")
    args = parser.parse_args()
    # print(args.__dict__)
    # print("\n")
    
    qaic_config = {
        "is_tlm": args.is_tlm,
        "include_sampler": args.include_sampler,
        "return_pdfs": args.return_pdfs,
    }
    qeff_model = initialize_model(
        model_name=args.model_name,
        # args.num_hidden_layers,
        qaic_config=qaic_config,
    )
    
    directory_path = f"{args.parent_directory}/{args.model_name.split('/')[-1]}_{'w' if args.include_sampler==True else 'wo'}_sampler_ts{args.num_devices}c{args.num_cores}_bs{args.batch_size}_ctx{args.ctx_length}_seqlen{args.sequence_length}_speclen{args.spec_length}_k{args.k}"
    print(directory_path)
    
    # result = subprocess.run(["rm", "-rf", f"{directory_path}-10ce98ae3ff843b9/"], capture_output=True, text=True)
    # print(result.stdout)
    # if (result.returncode != 0):
    #     print(result.stderr)
    
    # generated_onnx_path = export_model(
    #     qeff_model=qeff_model,
    #     directory_path=directory_path,
    # )
    
    generated_qpc_path = compile_model(
        qeff_model=qeff_model,
        # onnx_path=generated_onnx_path,
        compile_directory=directory_path,
        sequence_length=args.sequence_length,
        ctx_length=args.ctx_length,
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        num_cores=args.num_cores,
        spec_length=args.spec_length,
        device_id=args.device_id,
        mxint8_kv_cache=args.kv_cache_dtype=="mxint8",
        mxfp6_matmul=args.quantization=="mxfp6",
    )
    
    # inputs = generate_inputs(
    #     qeff_model.model.config.vocab_size,
    #     args.sequence_length,
    #     args.ctx_length,
    #     args.batch_size,
    #     args.seed if args.seed else np.random.randint(1, 101),
    #     args.parent_directory,
    # )
    
    # qeff_output_offline = run_qpc(
    #     generated_qpc_path,
    #     args.parent_directory,
    # )


if __name__ == "__main__":
    main()

"""
python3 /local/mnt/workspace/sanising/quic-github/efficient-transformers/run_sampler_copy2.py --model_name meta-llama/Llama-3.1-8B --include_sampler --no_is_tlm --return_pdfs --sequence_length 128 --k 512 --spec_length 1 --seed 1 --parent_directory /local/mnt/workspace/sanising/quic-github/efficient-transformers --num_devices 1 --num_cores 16 --kv-cache-dtype mxint8 --quantization mxfp6 --batch_size 2 --ctx_length 256
"""
