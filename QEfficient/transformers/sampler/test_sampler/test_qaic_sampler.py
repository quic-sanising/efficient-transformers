from copy import deepcopy
import numpy as np
import subprocess
import torch
import torch.nn as nn

from QEfficient.transformers.sampler.test_sampler.make_inputs import write_io_files
from QEfficient.transformers.sampler.test_sampler.sampler import sampler_forward
from QEfficient.transformers.sampler.test_sampler.vllm_sampler import (
    Sampler,
    SamplingMetadata,
)


def test_cpu_vs_vllm_cpu(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"]
    output_token_ids = setup_data["output_token_ids"]

    logits = setup_data["logits"]
    vllm_logits = deepcopy(setup_data["logits"]).squeeze(1)

    repetition_penalty_retain_state = setup_data["repetition_penalty_retain_state"]
    presence_penalty_retain_state = setup_data["presence_penalty_retain_state"]

    repetition_penalties = setup_data["repetition_penalties"]
    presence_penalties = setup_data["presence_penalties"]

    temperatures = setup_data["temperatures"]

    top_ks = setup_data["top_ks"]
    top_ps = setup_data["top_ps"]

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=temperatures,
        all_greedy=False,
        all_random=False,
        top_p=top_ps,
        top_k=top_ks,
        no_top_p=False,
        no_top_k=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=None,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=output_token_ids.tolist(),
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        logits,
        output_token_ids[:, -1:],
        repetition_penalty_retain_state,
        repetition_penalties,
        presence_penalty_retain_state,
        presence_penalties,
        temperatures,
        top_ks,
        top_ps,
    )
    vllm_output_logits, vllm_prompt_mask, vllm_output_mask = vllm_sampler(
        vllm_logits, sampling_metadata
    )

    # print(qeff_output.logits.squeeze(1))
    # print(vllm_output_logits)

    assert torch.allclose(
        qeff_output.logits.squeeze(1), vllm_output_logits, atol=1e-3
    ), "Output logits do not match"
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state.bool(),
        vllm_prompt_mask | vllm_output_mask,
        atol=1e-3,
    ), "Incorrect Repetition Penalty Retain State"
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state.bool(), vllm_output_mask, atol=1e-3,
    ), "Incorrect Presence Penalty Retain State"


def test_cpu_vs_qaic(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"]
    output_token_ids = setup_data["output_token_ids"]

    logits = setup_data["logits"]
    print("Logits", logits)
    qaic_logits = deepcopy(setup_data["logits"])
    print("QAIC logits", qaic_logits)

    repetition_penalty_retain_state = setup_data["repetition_penalty_retain_state"]
    qaic_repetition_penalty_retain_state = deepcopy(setup_data["repetition_penalty_retain_state"])
    presence_penalty_retain_state = setup_data["presence_penalty_retain_state"]
    qaic_presence_penalty_retain_state = deepcopy(setup_data["presence_penalty_retain_state"])

    repetition_penalties = setup_data["repetition_penalties"]
    presence_penalties = setup_data["presence_penalties"]

    temperatures = setup_data["temperatures"]

    top_ks = setup_data["top_ks"]
    top_ps = setup_data["top_ps"]

    sequence_length = (setup_data["sequence_length"],)
    batch_size = (setup_data["batch_size"],)
    vocab_size = (setup_data["vocab_size"],)
    ctx_length = (setup_data["ctx_length"],)

    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        logits,
        output_token_ids[:, -1:],
        repetition_penalty_retain_state,
        repetition_penalties,
        presence_penalty_retain_state,
        presence_penalties,
        temperatures,
        top_ks,
        top_ps,
    )
    print(qeff_output)
    
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "last_accepted_output_tokens": output_token_ids[:, -1:].detach().cpu().numpy(),
        "repetition_penalty_retain_state": qaic_repetition_penalty_retain_state.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "presence_penalty_retain_state": qaic_presence_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
    }
    outputs = {
        "logits": qaic_logits.detach().cpu().numpy(),
        "repetition_penalty_retain_state_RetainedState": qaic_repetition_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalty_retain_state_RetainedState": qaic_presence_penalty_retain_state.detach().cpu().numpy(),
    }
    print("Inputs", inputs)
    print("Outputs", outputs)
    write_io_files(
        inputs,
        outputs,
        "./io_data",
        "data",
        "end_to_end",
        True,
        False,
    )

    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            output_token_ids[:, -1:],
            qaic_logits,
            output_token_ids[:, -1:],
            qaic_repetition_penalty_retain_state,
            repetition_penalties,
            qaic_presence_penalty_retain_state,
            presence_penalties,
            temperatures,
            top_ks,
            top_ps,
        ),
        onnx_path,
        input_names=[
            "input_ids",
            "input_logits",
            "last_accepted_output_tokens",
            "repetition_penalty_retain_state",
            "repetition_penalties",
            "presence_penalty_retain_state",
            "presence_penalties",
            "temperatures",
            "top_ks",
            "top_ps",
        ],
        output_names=[
            "logits",
            "repetition_penalty_retain_state_RetainedState",
            "presence_penalty_retain_state_RetainedState",
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
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    print(result)

    cmd = [
        "/opt/qti-aic/exec/qaic-api-test",
        "-t",
        f"{qpc_dir_path}",
        "-n",
        "10",
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        "9",
        "--aic-profiling-num-samples",
        "1",
        "--aic-batch-json-input",
        "./io_data/end_to_end.json",
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
    hw_repetition_penalty_retain_state = torch.from_numpy(
        np.fromfile(
            "./outputs_from_qpcs/repetition_penalty_retain_state_RetainedState-activation-0-inf-0.bin",
            dtype=np.int32,
        )
    ).reshape(batch_size[0], vocab_size[0])
    hw_presence_penalty_retain_state = torch.from_numpy(
        np.fromfile(
            "./outputs_from_qpcs/presence_penalty_retain_state_RetainedState-activation-0-inf-0.bin",
            dtype=np.int32,
        )
    ).reshape(batch_size[0], vocab_size[0])

    print(qeff_output.logits)
    print(hw_output_logits)
    print(hw_repetition_penalty_retain_state)
    print(hw_presence_penalty_retain_state)

    assert torch.allclose(
        qeff_output.logits, hw_output_logits, atol=1e-2
    ), "Output logits do not match"
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state, hw_repetition_penalty_retain_state, atol=1e-6
    ), "Incorrect Repetition Penalty Retain State"
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state, hw_presence_penalty_retain_state, atol=1e-6
    ), "Incorrect Presence Penalty Retain State"


def test_gpu_vs_qaic(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"].cuda()
    output_token_ids = setup_data["output_token_ids"].cuda()

    logits = setup_data["logits"].cuda()
    print("Logits", logits)
    qaic_logits = deepcopy(setup_data["logits"])
    print("QAIC logits", qaic_logits)

    repetition_penalty_retain_state = setup_data["repetition_penalty_retain_state"].cuda()
    qaic_repetition_penalty_retain_state = deepcopy(setup_data["repetition_penalty_retain_state"])
    presence_penalty_retain_state = setup_data["presence_penalty_retain_state"].cuda()
    qaic_presence_penalty_retain_state = deepcopy(setup_data["presence_penalty_retain_state"])

    repetition_penalties = setup_data["repetition_penalties"].cuda()
    presence_penalties = setup_data["presence_penalties"].cuda()

    temperatures = setup_data["temperatures"].cuda()

    top_ks = setup_data["top_ks"].cuda()
    top_ps = setup_data["top_ps"].cuda()

    sequence_length = (setup_data["sequence_length"],)
    batch_size = (setup_data["batch_size"],)
    vocab_size = (setup_data["vocab_size"],)
    ctx_length = (setup_data["ctx_length"],)

    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        logits,
        output_token_ids[:, -1:],
        repetition_penalty_retain_state,
        repetition_penalties,
        presence_penalty_retain_state,
        presence_penalties,
        temperatures,
        top_ks,
        top_ps,
    )
    print(qeff_output)
    
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "last_accepted_output_tokens": output_token_ids[:, -1:].detach().cpu().numpy(),
        "repetition_penalty_retain_state": qaic_repetition_penalty_retain_state.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "presence_penalty_retain_state": qaic_presence_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
    }
    outputs = {
        "logits": qaic_logits.detach().cpu().numpy(),
        "repetition_penalty_retain_state_RetainedState": qaic_repetition_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalty_retain_state_RetainedState": qaic_presence_penalty_retain_state.detach().cpu().numpy(),
    }
    print("Inputs", inputs)
    print("Outputs", outputs)
    write_io_files(
        inputs,
        outputs,
        "./io_data",
        "data",
        "end_to_end",
        True,
        False,
    )

    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            output_token_ids[:, -1:],
            qaic_logits,
            output_token_ids[:, -1:],
            qaic_repetition_penalty_retain_state,
            repetition_penalties,
            qaic_presence_penalty_retain_state,
            presence_penalties,
            temperatures,
            top_ks,
            top_ps,
        ),
        onnx_path,
        input_names=[
            "input_ids",
            "input_logits",
            "last_accepted_output_tokens",
            "repetition_penalty_retain_state",
            "repetition_penalties",
            "presence_penalty_retain_state",
            "presence_penalties",
            "temperatures",
            "top_ks",
            "top_ps",
        ],
        output_names=[
            "logits",
            "repetition_penalty_retain_state_RetainedState",
            "presence_penalty_retain_state_RetainedState",
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
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    print(result)

    cmd = [
        "/opt/qti-aic/exec/qaic-api-test",
        "-t",
        f"{qpc_dir_path}",
        "-n",
        "10",
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        "9",
        "--aic-profiling-num-samples",
        "1",
        "--aic-batch-json-input",
        "./io_data/end_to_end.json",
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
    hw_repetition_penalty_retain_state = torch.from_numpy(
        np.fromfile(
            "./outputs_from_qpcs/repetition_penalty_retain_state_RetainedState-activation-0-inf-0.bin",
            dtype=np.int32,
        )
    ).reshape(batch_size[0], vocab_size[0])
    hw_presence_penalty_retain_state = torch.from_numpy(
        np.fromfile(
            "./outputs_from_qpcs/presence_penalty_retain_state_RetainedState-activation-0-inf-0.bin",
            dtype=np.int32,
        )
    ).reshape(batch_size[0], vocab_size[0])

    print(qeff_output.logits)
    print(hw_output_logits)
    print(hw_repetition_penalty_retain_state)
    print(hw_presence_penalty_retain_state)

    assert torch.allclose(
        qeff_output.logits.cpu(), hw_output_logits, atol=1e-2
    ), "Output logits do not match"
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state.cpu(), hw_repetition_penalty_retain_state, atol=1e-6
    ), "Incorrect Repetition Penalty Retain State"
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state.cpu(), hw_presence_penalty_retain_state, atol=1e-6
    ), "Incorrect Presence Penalty Retain State"


def test_gpu_vs_vllm_gpu(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"].cuda()
    output_token_ids = setup_data["output_token_ids"].cuda()

    logits = setup_data["logits"].cuda()
    vllm_logits = deepcopy(setup_data["logits"]).squeeze(1).cuda()

    repetition_penalty_retain_state = setup_data["repetition_penalty_retain_state"].cuda()
    presence_penalty_retain_state = setup_data["presence_penalty_retain_state"].cuda()

    repetition_penalties = setup_data["repetition_penalties"].cuda()
    presence_penalties = setup_data["presence_penalties"].cuda()

    temperatures = setup_data["temperatures"].cuda()

    top_ks = setup_data["top_ks"].cuda()
    top_ps = setup_data["top_ps"].cuda()

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=temperatures,
        all_greedy=False,
        all_random=False,
        top_p=top_ps,
        top_k=top_ks,
        no_top_p=False,
        no_top_k=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=None,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=output_token_ids.tolist(),
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        logits,
        output_token_ids[:, -1:],
        repetition_penalty_retain_state,
        repetition_penalties,
        presence_penalty_retain_state,
        presence_penalties,
        temperatures,
        top_ks,
        top_ps,
    )
    vllm_output_logits, vllm_prompt_mask, vllm_output_mask = vllm_sampler(
        vllm_logits, sampling_metadata
    )

    # print(qeff_output.logits.squeeze(1))
    # print(vllm_output_logits)

    assert torch.allclose(
        qeff_output.logits.squeeze(1).cpu(), vllm_output_logits.cpu(), atol=1e-3
    ), "Output logits do not match"
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state.bool().cpu(),
        vllm_prompt_mask.cpu() | vllm_output_mask.cpu(),
        atol=1e-3,
    ), "Incorrect Repetition Penalty Retain State"
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state.bool().cpu(), vllm_output_mask.cpu(), atol=1e-3,
    ), "Incorrect Presence Penalty Retain State"
