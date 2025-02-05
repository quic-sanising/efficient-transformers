from copy import deepcopy
import numpy as np
import subprocess
import torch
import torch.nn as nn

from QEfficient.transformers.sampler.test_sampler.make_inputs import write_io_files
from QEfficient.transformers.sampler.test_sampler.sampler_top_ps import sampler_forward
from QEfficient.transformers.sampler.test_sampler.vllm_sampler_topkp import (
    Sampler,
    SamplingMetadata,
)


def test_cpu_vs_vllm_cpu(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"]
    vllm_logits = deepcopy(setup_data_top_ps["logits"]).squeeze(1)

    top_ks = setup_data_top_ps["top_ks"]
    top_ps = setup_data_top_ps["top_ps"]

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=top_ps,
        top_k=top_ks,
        no_top_p=False,
        no_top_k=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
        output_token_ids=None,
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        logits,
        top_ks,
        top_ps,
    )
    vllm_output_logits = vllm_sampler(vllm_logits, sampling_metadata)
    print(f"QEff Output: {qeff_output.logits.squeeze(1)}")
    print(f"VLLM Output: {vllm_output_logits}")

    assert torch.allclose(
        qeff_output.logits.squeeze(1), vllm_output_logits, atol=1e-6
    ), "Output logits do not match"


def test_cpu_vs_qaic(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"]
    print("Logits", logits)
    qaic_logits = deepcopy(setup_data_top_ps["logits"])
    print("QAIC logits", qaic_logits)

    top_ks = setup_data_top_ps["top_ks"]
    qaic_top_ks = deepcopy(setup_data_top_ps["top_ks"])

    top_ps = setup_data_top_ps["top_ps"]
    qaic_top_ps = deepcopy(setup_data_top_ps["top_ps"])

    batch_size = (setup_data_top_ps["batch_size"],)
    vocab_size = (setup_data_top_ps["vocab_size"],)

    qeff_output = sampler_forward(
        None,
        logits,
        top_ks,
        top_ps
    )
    print(qeff_output)

    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "top_ks": qaic_top_ks.detach().cpu().numpy(),
        "top_ps": qaic_top_ps.detach().cpu().numpy(),
    }
    outputs = {
        "logits": qaic_logits.detach().cpu().numpy(),
    }
    print("Inputs", inputs)
    print("Outputs", outputs)
    write_io_files(
        inputs,
        outputs,
        "./io_data",
        "data",
        "top_ps",
        True,
        False,
    )

    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_top_ps_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            qaic_logits,
            qaic_top_ks,
            qaic_top_ps,
        ),
        onnx_path,
        input_names=[
            "input_logits",
            "top_ks",
            "top_ps"
        ],
        output_names=[
            "logits",
        ],
        dynamo=False,
        verbose=True,
    )

    qpc_dir_path = "./on_device_sampling_qpcs/"
    compile_cmd = [
        "/opt/qti-aic/exec/qaic-exec",
        "-v",
        "-aic-hw",
        "-convert-to-fp16",
        "-aic-num-cores=14",
        "-aic-num-of-instances=1",
        "-num-iter=100",
        f"-m={onnx_path}",
        "-stats-level=70",
        "-aic-pmu-recipe=KernelUtil",
        "-ddr-stats",
        "-time-passes",
        "-mxfp6-matmul",
        "-aic-enable-depth-first",
        "-retained-state=true",
        "-aic-perf-metrics",
        f"-aic-binary-dir={qpc_dir_path}",
        "-aic-hw-version=2.0",
        "-compile-only",
    ]
    subprocess.run(["rm", "-rf", f"{qpc_dir_path}"])
    print("Compile command", " ".join(compile_cmd))
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    print(result)

    cmd = [
        "/opt/qti-aic/exec/qaic-api-test",
        "-t",
        f"{qpc_dir_path}",
        "-n",
        "1",
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        "1",
        "--aic-profiling-num-samples",
        "1",
        "--aic-batch-json-input",
        "./io_data/top_ps.json",
        "--write-output-dir",
        "./outputs_from_qpcs/",
    ]

    qeff_output_offline = subprocess.run(cmd, capture_output=True, text=True)
    print(qeff_output_offline)
    # for k, v in qeff_output_offline.__dict__.items():
    #     print(k, v)

    hw_output_logits = torch.from_numpy(
        np.fromfile("./outputs_from_qpcs/logits-activation-0-inf-0.bin", dtype=np.float32)
    ).reshape(batch_size[0], 1, vocab_size[0])

    print(qeff_output.logits)
    print(hw_output_logits)

    assert torch.allclose(
        qeff_output.logits, hw_output_logits, atol=1e-3
    ), "Output logits do not match"


def test_gpu_vs_qaic(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"].cuda()
    print("Logits", logits)
    qaic_logits = deepcopy(setup_data_top_ps["logits"])
    print("QAIC logits", qaic_logits)

    top_ks = setup_data_top_ps["top_ks"].cuda()
    qaic_top_ks = deepcopy(setup_data_top_ps["top_ks"])

    top_ps = setup_data_top_ps["top_ps"].cuda()
    qaic_top_ps = deepcopy(setup_data_top_ps["top_ps"])

    batch_size = (setup_data_top_ps["batch_size"],)
    vocab_size = (setup_data_top_ps["vocab_size"],)

    qeff_output = sampler_forward(
        None,
        logits,
        top_ks,
        top_ps
    )
    print(qeff_output)

    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "top_ks": qaic_top_ks.detach().cpu().numpy(),
        "top_ps": qaic_top_ps.detach().cpu().numpy(),
    }
    outputs = {
        "logits": qaic_logits.detach().cpu().numpy(),
    }
    print("Inputs", inputs)
    print("Outputs", outputs)
    write_io_files(
        inputs,
        outputs,
        "./io_data",
        "data",
        "top_ps",
        True,
        False,
    )

    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_top_ps_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            qaic_logits,
            qaic_top_ks,
            qaic_top_ps,
        ),
        onnx_path,
        input_names=[
            "input_logits",
            "top_ks",
            "top_ps"
        ],
        output_names=[
            "logits",
        ],
        dynamo=False,
        verbose=True,
    )

    qpc_dir_path = "./on_device_sampling_qpcs/"
    compile_cmd = [
        "/opt/qti-aic/exec/qaic-exec",
        "-v",
        "-aic-hw",
        "-convert-to-fp16",
        "-aic-num-cores=14",
        "-aic-num-of-instances=1",
        "-num-iter=100",
        f"-m={onnx_path}",
        "-stats-level=70",
        "-aic-pmu-recipe=KernelUtil",
        "-ddr-stats",
        "-time-passes",
        "-mxfp6-matmul",
        "-aic-enable-depth-first",
        "-retained-state=true",
        "-aic-perf-metrics",
        f"-aic-binary-dir={qpc_dir_path}",
        "-aic-hw-version=2.0",
        "-compile-only",
    ]
    subprocess.run(["rm", "-rf", f"{qpc_dir_path}"])
    print("Compile command", " ".join(compile_cmd))
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    print(result)

    cmd = [
        "/opt/qti-aic/exec/qaic-api-test",
        "-t",
        f"{qpc_dir_path}",
        "-n",
        "1",
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        "1",
        "--aic-profiling-num-samples",
        "1",
        "--aic-batch-json-input",
        "./io_data/top_ps.json",
        "--write-output-dir",
        "./outputs_from_qpcs/",
    ]

    qeff_output_offline = subprocess.run(cmd, capture_output=True, text=True)
    print(qeff_output_offline)
    # for k, v in qeff_output_offline.__dict__.items():
    #     print(k, v)

    hw_output_logits = torch.from_numpy(
        np.fromfile("./outputs_from_qpcs/logits-activation-0-inf-0.bin", dtype=np.float32)
    ).reshape(batch_size[0], 1, vocab_size[0])

    print(qeff_output.logits)
    print(hw_output_logits)

    assert torch.allclose(
        qeff_output.logits.cpu(), hw_output_logits, atol=1e-3
    ), "Output logits do not match"


def test_gpu_vs_vllm_gpu(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"].cuda()
    vllm_logits = deepcopy(setup_data_top_ps["logits"]).squeeze(1).cuda()

    top_ks = setup_data_top_ps["top_ks"].cuda()
    top_ps = setup_data_top_ps["top_ps"].cuda()

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=top_ps,
        top_k=top_ks,
        no_top_p=False,
        no_top_k=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
        output_token_ids=None,
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        logits,
        top_ks,
        top_ps,
    )
    vllm_output_logits = vllm_sampler(vllm_logits, sampling_metadata)
    print(f"QEff Output: {qeff_output.logits.squeeze(1)}")
    print(f"VLLM Output: {vllm_output_logits}")

    assert torch.allclose(
        qeff_output.logits.squeeze(1).cpu(), vllm_output_logits.cpu(), atol=1e-6
    ), "Output logits do not match"
