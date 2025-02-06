from copy import deepcopy
import numpy as np
import subprocess
import torch
import torch.nn as nn

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.sampler.test_sampler.sampler_penalties import sampler_forward
from QEfficient.transformers.sampler.test_sampler.vllm_sampler_penalties import (
    Sampler,
    SamplingMetadata,
)
from QEfficient.transformers.sampler.test_sampler.utils import print_difference_in_tensors


def test_cpu_vs_vllm_cpu(setup_data_penalties):
    print(setup_data_penalties["seed"])

    prompt_token_ids = setup_data_penalties["prompt_token_ids"]
    output_token_ids = setup_data_penalties["output_token_ids"]

    logits = setup_data_penalties["logits"]

    repetition_penalty_retain_state = setup_data_penalties["repetition_penalty_retain_state"]
    presence_penalty_retain_state = setup_data_penalties["presence_penalty_retain_state"]

    repetition_penalties = setup_data_penalties["repetition_penalties"]
    presence_penalties = setup_data_penalties["presence_penalties"]

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
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
    )
    vllm_output_logits, vllm_prompt_mask, vllm_output_mask = vllm_sampler(
        logits.squeeze(1), sampling_metadata
    )

    assert torch.allclose(
        qeff_output.logits.squeeze(1), vllm_output_logits, atol=1e-6
    ), "Output logits do not match"
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state.bool(),
        vllm_prompt_mask | vllm_output_mask,
        atol=1e-6,
    ), "Incorrect Repetition Penalty Retain State"
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state.bool(), vllm_output_mask, atol=1e-6
    ), "Incorrect Presence Penalty Retain State"


def test_cpu_vs_qaic(setup_data_penalties):
    print(setup_data_penalties["seed"])

    prompt_token_ids = setup_data_penalties["prompt_token_ids"]
    output_token_ids = setup_data_penalties["output_token_ids"]

    logits = setup_data_penalties["logits"]
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data_penalties["logits"])
    print("QAIC Input Logits", qaic_logits)

    repetition_penalty_retain_state = setup_data_penalties["repetition_penalty_retain_state"]
    qaic_repetition_penalty_retain_state = deepcopy(setup_data_penalties["repetition_penalty_retain_state"])
    presence_penalty_retain_state = setup_data_penalties["presence_penalty_retain_state"]
    qaic_presence_penalty_retain_state = deepcopy(setup_data_penalties["presence_penalty_retain_state"])

    repetition_penalties = setup_data_penalties["repetition_penalties"]
    presence_penalties = setup_data_penalties["presence_penalties"]

    # ---Run on CPU---
    qeff_output = sampler_forward(
        self=None,
        input_ids=output_token_ids[:, -1:],
        input_logits=logits.to(torch.float16),
        last_accepted_output_tokens=output_token_ids[:, -1:],
        repetition_penalty_retain_state=repetition_penalty_retain_state,
        repetition_penalties=repetition_penalties.to(torch.float16),
        presence_penalty_retain_state=presence_penalty_retain_state,
        presence_penalties=presence_penalties.to(torch.float16),
    )
    print("\nOutput\n", qeff_output)

    # ---Run on QAIC---
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    # Export ONNX file
    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_penalties_hardware.onnx"
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
        ],
        output_names=[
            "logits",
            "repetition_penalty_retain_state_RetainedState",
            "presence_penalty_retain_state_RetainedState",
        ],
        dynamo=False,
        verbose=True,
    )

    # Compile QPC file
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
    print(result.stdout)
    if (result.returncode != 0):
        print(result.stderr)

    # Run QPC file
    session = QAICInferenceSession(qpc_path=qpc_dir_path, device_ids=[0], enable_debug_logs=False)
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "last_accepted_output_tokens": output_token_ids[:, -1:].detach().cpu().numpy(),
        "repetition_penalty_retain_state": qaic_repetition_penalty_retain_state.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "presence_penalty_retain_state": qaic_presence_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    outputs = session.run(inputs)
    print("\nQAIC Output\n", outputs)

    hw_output_logits = torch.from_numpy(outputs["logits"])
    hw_repetition_penalty_retain_state = torch.from_numpy(outputs["repetition_penalty_retain_state_RetainedState"])
    hw_presence_penalty_retain_state = torch.from_numpy(outputs["presence_penalty_retain_state_RetainedState"])

    print("\nLogits\n", qeff_output.logits)
    print("\nQAIC Logits\n", hw_output_logits)

    print("\nRepetition Penalty Retain State\n", qeff_output.repetition_penalty_retain_state)
    print("\nQAIC Repetition Penalty Retain State\n", hw_repetition_penalty_retain_state)

    print("\nPresence Penalty Retain State\n", qeff_output.presence_penalty_retain_state)
    print("\nQAIC Presence Penalty Retain State\n", hw_presence_penalty_retain_state)

    # Compare outputs
    assert torch.allclose(
        qeff_output.logits.to(torch.float32), hw_output_logits, atol=1e-6
    ), print_difference_in_tensors(
        qeff_output.logits.to(torch.float32),
        "Logits",
        hw_output_logits,
        "QAIC Logits",
        1e-6,
    )
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state, hw_repetition_penalty_retain_state
    ), print_difference_in_tensors(
        qeff_output.repetition_penalty_retain_state, 
        "Repetition Penalty Retain State",
        hw_repetition_penalty_retain_state,
        "QAIC Repetition Penalty Retain State",
    )
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state, hw_presence_penalty_retain_state
    ), print_difference_in_tensors(
        qeff_output.presence_penalty_retain_state,
        "Presence Penalty Retain State",
        hw_presence_penalty_retain_state,
        "QAIC Presence Penalty Retain State",
    )


def test_gpu_vs_qaic(setup_data_penalties):
    print(setup_data_penalties["seed"])

    # TODO: Check how to send tensors to cuda in float16 

    prompt_token_ids = setup_data_penalties["prompt_token_ids"].cuda()
    output_token_ids = setup_data_penalties["output_token_ids"].cuda()

    logits = setup_data_penalties["logits"].cuda()
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data_penalties["logits"]).cuda()
    print("QAIC Input Logits", qaic_logits)

    repetition_penalty_retain_state = setup_data_penalties["repetition_penalty_retain_state"].cuda()
    qaic_repetition_penalty_retain_state = deepcopy(setup_data_penalties["repetition_penalty_retain_state"])
    presence_penalty_retain_state = setup_data_penalties["presence_penalty_retain_state"].cuda()
    qaic_presence_penalty_retain_state = deepcopy(setup_data_penalties["presence_penalty_retain_state"])

    repetition_penalties = setup_data_penalties["repetition_penalties"].cuda()
    presence_penalties = setup_data_penalties["presence_penalties"].cuda()

    # ---Run on CPU---
    qeff_output = sampler_forward(
        self=None,
        input_ids=output_token_ids[:, -1:],
        input_logits=logits.to(torch.float16),
        last_accepted_output_tokens=output_token_ids[:, -1:],
        repetition_penalty_retain_state=repetition_penalty_retain_state,
        repetition_penalties=repetition_penalties.to(torch.float16),
        presence_penalty_retain_state=presence_penalty_retain_state,
        presence_penalties=presence_penalties.to(torch.float16),
    )
    print("\nOutput\n", qeff_output)

    # ---Run on QAIC---
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    # Export ONNX file
    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_penalties_hardware.onnx"
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
        ],
        output_names=[
            "logits",
            "repetition_penalty_retain_state_RetainedState",
            "presence_penalty_retain_state_RetainedState",
        ],
        dynamo=False,
        verbose=True,
    )

    # Compile QPC file
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
    print(result.stdout)
    if (result.returncode != 0):
        print(result.stderr)

    # Run QPC file
    session = QAICInferenceSession(qpc_path=qpc_dir_path, device_ids=[0], enable_debug_logs=False)
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "last_accepted_output_tokens": output_token_ids[:, -1:].detach().cpu().numpy(),
        "repetition_penalty_retain_state": qaic_repetition_penalty_retain_state.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "presence_penalty_retain_state": qaic_presence_penalty_retain_state.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    outputs = session.run(inputs)
    print("\nQAIC Output\n", outputs)

    hw_output_logits = torch.from_numpy(outputs["logits"])
    hw_repetition_penalty_retain_state = torch.from_numpy(outputs["repetition_penalty_retain_state_RetainedState"])
    hw_presence_penalty_retain_state = torch.from_numpy(outputs["presence_penalty_retain_state_RetainedState"])

    print("\nLogits\n", qeff_output.logits)
    print("\nQAIC Logits\n", hw_output_logits)

    print("\nRepetition Penalty Retain State\n", qeff_output.repetition_penalty_retain_state)
    print("\nQAIC Repetition Penalty Retain State\n", hw_repetition_penalty_retain_state)

    print("\nPresence Penalty Retain State\n", qeff_output.presence_penalty_retain_state)
    print("\nQAIC Presence Penalty Retain State\n", hw_presence_penalty_retain_state)

    # Compare outputs
    assert torch.allclose(
        qeff_output.logits.cpu().to(torch.float32), hw_output_logits, atol=1e-6
    ), print_difference_in_tensors(
        qeff_output.logits.cpu().to(torch.float32),
        "Logits",
        hw_output_logits,
        "QAIC Logits",
        1e-6,
    )
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state.cpu(), hw_repetition_penalty_retain_state
    ), print_difference_in_tensors(
        qeff_output.repetition_penalty_retain_state.cpu(), 
        "Repetition Penalty Retain State",
        hw_repetition_penalty_retain_state,
        "QAIC Repetition Penalty Retain State",
    )
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state.cpu(), hw_presence_penalty_retain_state
    ), print_difference_in_tensors(
        qeff_output.presence_penalty_retain_state.cpu(),
        "Presence Penalty Retain State",
        hw_presence_penalty_retain_state,
        "QAIC Presence Penalty Retain State",
    )


def test_gpu_vs_vllm_gpu(setup_data_penalties):
    print(setup_data_penalties["seed"])

    prompt_token_ids = setup_data_penalties["prompt_token_ids"].cuda()
    output_token_ids = setup_data_penalties["output_token_ids"].cuda()

    logits = setup_data_penalties["logits"].cuda()

    repetition_penalty_retain_state = setup_data_penalties["repetition_penalty_retain_state"].cuda()
    presence_penalty_retain_state = setup_data_penalties["presence_penalty_retain_state"].cuda()

    repetition_penalties = setup_data_penalties["repetition_penalties"].cuda()
    presence_penalties = setup_data_penalties["presence_penalties"].cuda()

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
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
    )
    vllm_output_logits, vllm_prompt_mask, vllm_output_mask = vllm_sampler(
        logits.squeeze(1), sampling_metadata
    )

    assert torch.allclose(
        qeff_output.logits.squeeze(1).cpu(), vllm_output_logits.cpu(), atol=1e-6
    ), "Output logits do not match"
    assert torch.allclose(
        qeff_output.repetition_penalty_retain_state.bool().cpu(),
        vllm_prompt_mask.cpu() | vllm_output_mask.cpu(),
        atol=1e-6,
    ), "Incorrect Repetition Penalty Retain State"
    assert torch.allclose(
        qeff_output.presence_penalty_retain_state.bool().cpu(), vllm_output_mask.cpu(), atol=1e-6
    ), "Incorrect Presence Penalty Retain State"
