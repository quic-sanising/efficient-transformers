from copy import deepcopy
import subprocess
import torch
import torch.nn as nn

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.sampler.test_sampler.sampler import sampler_forward
from QEfficient.transformers.sampler.test_sampler.utils import print_difference_in_tensors
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
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data["logits"])
    print("QAIC Input Logits", qaic_logits)

    repetition_penalty_retain_state = setup_data["repetition_penalty_retain_state"]
    qaic_repetition_penalty_retain_state = deepcopy(setup_data["repetition_penalty_retain_state"])
    presence_penalty_retain_state = setup_data["presence_penalty_retain_state"]
    qaic_presence_penalty_retain_state = deepcopy(setup_data["presence_penalty_retain_state"])

    repetition_penalties = setup_data["repetition_penalties"]
    presence_penalties = setup_data["presence_penalties"]

    temperatures = setup_data["temperatures"]

    top_ks = setup_data["top_ks"]
    top_ps = setup_data["top_ps"]

    # ---Run on CPU---
    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        logits.to(torch.float16),
        output_token_ids[:, -1:],
        repetition_penalty_retain_state,
        repetition_penalties.to(torch.float16),
        presence_penalty_retain_state,
        presence_penalties.to(torch.float16),
        temperatures.to(torch.float16),
        top_ks,
        top_ps,
    )
    print("\nOutput\n", qeff_output)

    # ---Run on QAIC---
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    # Export ONNX file
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
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
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
    has_failed = False
    if not torch.allclose(
        qeff_output.logits.to(torch.float32), hw_output_logits, atol=1e-4
    ): 
        print_difference_in_tensors(
            qeff_output.logits.to(torch.float32),
            "Logits",
            hw_output_logits,
            "QAIC Logits",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(
        qeff_output.repetition_penalty_retain_state, hw_repetition_penalty_retain_state
    ):
        print_difference_in_tensors(
            qeff_output.repetition_penalty_retain_state, 
            "Repetition Penalty Retain State",
            hw_repetition_penalty_retain_state,
            "QAIC Repetition Penalty Retain State",
        )
        has_failed = True
        
    if not torch.allclose(
        qeff_output.presence_penalty_retain_state, hw_presence_penalty_retain_state
    ): 
        print_difference_in_tensors(
            qeff_output.presence_penalty_retain_state,
            "Presence Penalty Retain State",
            hw_presence_penalty_retain_state,
            "QAIC Presence Penalty Retain State",
        )
        has_failed = True
    
    assert not has_failed, "Test failed"


def test_gpu_vs_qaic(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"].cuda()
    output_token_ids = setup_data["output_token_ids"].cuda()

    logits = setup_data["logits"].cuda()
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data["logits"])
    print("QAIC Input Logits", qaic_logits)

    repetition_penalty_retain_state = setup_data["repetition_penalty_retain_state"].cuda()
    qaic_repetition_penalty_retain_state = deepcopy(setup_data["repetition_penalty_retain_state"])
    presence_penalty_retain_state = setup_data["presence_penalty_retain_state"].cuda()
    qaic_presence_penalty_retain_state = deepcopy(setup_data["presence_penalty_retain_state"])

    repetition_penalties = setup_data["repetition_penalties"].cuda()
    presence_penalties = setup_data["presence_penalties"].cuda()

    temperatures = setup_data["temperatures"].cuda()

    top_ks = setup_data["top_ks"].cuda()
    top_ps = setup_data["top_ps"].cuda()

    # ---Run on GPU---
    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        logits.to(torch.float16),
        output_token_ids[:, -1:],
        repetition_penalty_retain_state,
        repetition_penalties.to(torch.float16),
        presence_penalty_retain_state,
        presence_penalties.to(torch.float16),
        temperatures.to(torch.float16),
        top_ks,
        top_ps,
    )
    print("\nOutput\n", qeff_output)

    # ---Run on QAIC---
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    # Export ONNX file
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
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
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
    has_failed = False
    if not torch.allclose(
        qeff_output.logits.cpu().to(torch.float32), hw_output_logits, atol=1e-4
    ): 
        print_difference_in_tensors(
            qeff_output.logits.cpu().to(torch.float32),
            "Logits",
            hw_output_logits,
            "QAIC Logits",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(
        qeff_output.repetition_penalty_retain_state.cpu(), hw_repetition_penalty_retain_state
    ):
        print_difference_in_tensors(
            qeff_output.repetition_penalty_retain_state.cpu(), 
            "Repetition Penalty Retain State",
            hw_repetition_penalty_retain_state,
            "QAIC Repetition Penalty Retain State",
        )
        has_failed = True
        
    if not torch.allclose(
        qeff_output.presence_penalty_retain_state.cpu(), hw_presence_penalty_retain_state
    ): 
        print_difference_in_tensors(
            qeff_output.presence_penalty_retain_state.cpu(),
            "Presence Penalty Retain State",
            hw_presence_penalty_retain_state,
            "QAIC Presence Penalty Retain State",
        )
        has_failed = True
    
    assert not has_failed, "Test failed"


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
