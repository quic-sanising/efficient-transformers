import torch

from copy import deepcopy
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
