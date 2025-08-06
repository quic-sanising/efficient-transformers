from dataclasses import dataclass
import torch
import torch.nn.functional as F

from QEfficient.customop import CtxScatterFuncCB3D
from QEfficient.utils.constants import Constants
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from typing import List, Optional, Tuple, Union


@dataclass
class SamplerOutput(ModelOutput):
    probs: torch.FloatTensor = None
    next_tokens: torch.IntTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_repetition_penalty_buffer: Optional[torch.Tensor] = None
    past_presence_penalty_buffer: Optional[torch.Tensor] = None


def prefill_path(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor,
    batch_index: torch.LongTensor,
    batch_index_reshaped: torch.LongTensor,
    past_repetition_penalty_buffer: torch.Tensor,
    past_presence_penalty_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize or update RetainedState buffers for prefill stage based on `input_ids`.
    """
    # Initialize retain states for first input chunk
    mul_value = torch.ones(past_repetition_penalty_buffer.shape[0], 1, dtype=torch.bool)
    zero_tensor = torch.zeros(batch_index.shape, dtype=torch.long)
    positions_mask = (position_ids[:, :1] != zero_tensor).view(-1, 1)
    mul_value = CtxScatterFuncCB3D.apply(
        mul_value, batch_index, zero_tensor, positions_mask
    )
    past_repetition_penalty_buffer *= mul_value
    past_presence_penalty_buffer *= mul_value

    # Mask out-of-bounds or invalid position_ids or input_ids
    input_ids = torch.where(position_ids == -1, torch.iinfo(torch.int32).max, input_ids)

    # Update retain states for chunked input
    past_repetition_penalty_buffer = CtxScatterFuncCB3D.apply(
        past_repetition_penalty_buffer,
        batch_index,
        input_ids,
        torch.ones(input_ids.shape, dtype=torch.bool),
    )
    return past_repetition_penalty_buffer, past_presence_penalty_buffer


def decode_path(
    last_accepted_output_tokens: torch.LongTensor,
    position_ids: torch.LongTensor,
    batch_index: torch.LongTensor,
    batch_index_reshaped: torch.LongTensor,
    past_repetition_penalty_buffer: torch.Tensor,
    past_presence_penalty_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update RetainedState buffers for decode stage based on `last_accepted_output_tokens`.
    """
    # Mask out-of-bounds or invalid position_ids or last_accepted_output_tokens
    last_accepted_output_tokens = torch.where(
        position_ids == -1, torch.iinfo(torch.int32).max, last_accepted_output_tokens
    )

    # Update retained states
    scatter_values = torch.ones(last_accepted_output_tokens.shape, dtype=torch.bool)
    past_repetition_penalty_buffer = CtxScatterFuncCB3D.apply(
        past_repetition_penalty_buffer,
        batch_index,
        last_accepted_output_tokens,
        scatter_values,
    )
    past_presence_penalty_buffer = CtxScatterFuncCB3D.apply(
        past_presence_penalty_buffer,
        batch_index,
        last_accepted_output_tokens,
        scatter_values,
    )
    # TODO: For frequency retain state, first gather and then scatter
    return past_repetition_penalty_buffer, past_presence_penalty_buffer


def sampler_forward(
    self,
    input_ids: torch.LongTensor,
    position_ids: Optional[torch.LongTensor],
    batch_index: Optional[torch.LongTensor],
    input_logits: torch.Tensor,
    last_accepted_output_tokens: Optional[torch.Tensor] = None,  # (batch_size, spec_length or less)
    past_repetition_penalty_buffer: Optional[torch.Tensor] = None,
    repetition_penalties: Optional[torch.Tensor] = None,
    past_presence_penalty_buffer: Optional[torch.Tensor] = None,
    presence_penalties: Optional[torch.Tensor] = None,
    temperatures: Optional[torch.Tensor] = None,
    top_ks: Optional[torch.Tensor] = None,
    top_ps: Optional[torch.Tensor] = None,
    min_ps: Optional[torch.Tensor] = None,
) -> Union[Tuple, SamplerOutput]:
    # Move to device
    # device = input_logits.device
    # last_accepted_output_tokens = last_accepted_output_tokens.to(device)
    # past_repetition_penalty_buffer = past_repetition_penalty_buffer.to(device)
    # past_presence_penalty_buffer = past_presence_penalty_buffer.to(device)
    # repetition_penalties = repetition_penalties.to(device)
    # presence_penalties = presence_penalties.to(device)
    # temperatures = temperatures.to(device)
    # top_ks = top_ks.to(device)
    # top_ps = top_ps.to(device)
    # min_ps = min_ps.to(device)

    # Perform Sampling
    batch_size, spec_length, vocab_size = input_logits.shape
    logits = input_logits.reshape(-1, vocab_size)  # Reshape tensor to 2D

    batch_index_reshaped = batch_index.view(-1)
    # Prefill
    past_repetition_penalty_buffer_prefill, past_presence_penalty_buffer_prefill = prefill_path(
        input_ids=input_ids,
        position_ids=position_ids,
        batch_index=batch_index,
        batch_index_reshaped=batch_index_reshaped,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer.clone(),
        past_presence_penalty_buffer=past_presence_penalty_buffer.clone(),
    )
    # Decode
    past_repetition_penalty_buffer_decode, past_presence_penalty_buffer_decode = decode_path(
        last_accepted_output_tokens=last_accepted_output_tokens,
        position_ids=position_ids,
        batch_index=batch_index,
        batch_index_reshaped=batch_index_reshaped,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer.clone(),
        past_presence_penalty_buffer=past_presence_penalty_buffer.clone(),
    )
    # Select the correct repetition and presence penalty buffers
    is_prefill = torch.ones(past_repetition_penalty_buffer.shape, dtype=torch.bool) * (input_ids.shape[1] > spec_length)
    past_repetition_penalty_buffer = torch.where(
        is_prefill, past_repetition_penalty_buffer_prefill, past_repetition_penalty_buffer_decode
    )
    past_presence_penalty_buffer = torch.where(
        is_prefill, past_presence_penalty_buffer_prefill, past_presence_penalty_buffer_decode
    )

    # Repetition Penalty
    if (repetition_penalties != 1.0).any():
        past_repetition_penalty_buffer_selected = \
            past_repetition_penalty_buffer[batch_index_reshaped].repeat(spec_length, 1)  # (batch_size * spec_length, vocab_size)
        repetition_penalties_mask = torch.where(past_repetition_penalty_buffer_selected, repetition_penalties, 1.0)
        logits *= repetition_penalties_mask ** (-torch.sign(logits))

    # Presence Penalty
    if (presence_penalties != 0.0).any():
        presence_penalties = presence_penalties.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
        past_presence_penalty_buffer_selected = \
            past_presence_penalty_buffer[batch_index_reshaped].repeat(spec_length, 1)  # (batch_size * spec_length, vocab_size)
        logits -= presence_penalties * past_presence_penalty_buffer_selected

    # TODO: Frequency Penalty

    # Temperature Scaling
    temperatures = temperatures.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
    logits = torch.where(temperatures != 0, logits / temperatures, logits)

    # Top K
    # TODO (Optimization): if (top_ks != -1 or top_ks != max_top_k_ids).any() is False: skip but will need topk_values_asc and topk_indices_asc
    max_top_k_ids = Constants.MAX_TOP_K_IDS
    topk_values, topk_indices = torch.topk(logits, k=max_top_k_ids, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_values_asc = topk_values.flip(dims=[1])
    topk_indices_asc = topk_indices.flip(dims=[1])
    top_ks[top_ks > max_top_k_ids] = max_top_k_ids  # Clip k to max value
    # True values in this mask indicate the positions of the non-top K values
    topk_mask = torch.arange(topk_values_asc.shape[1]).unsqueeze(0) < (topk_values_asc.size(1) - top_ks.to(torch.long)).repeat(spec_length, 1)  # (batch_size * spec_length, max_top_k_ids)
    topk_values_asc[topk_mask] = torch.finfo(torch.float16).min

    # Top P
    # TODO (Optimization): if (top_ps != 1.).any() is False: skip but will need top_probs for Min P
    top_probs = torch.softmax(topk_values_asc, dim=1)  # (batch_size * spec_length, max_top_k_ids)
    topk_probs_sum = torch.cumsum(top_probs, dim=1)
    top_p_mask = topk_probs_sum <= 1 - top_ps.repeat(spec_length, 1)  # (batch_size * spec_length, max_top_k_ids)
    top_p_mask[:, max_top_k_ids - 1] = False
    topk_values_asc[top_p_mask] = torch.finfo(torch.float16).min

    # Min P
    if (min_ps != 0.0).any():
        scaled_min_p = torch.mul(
            min_ps.repeat(spec_length, 1),
            top_probs[:, max_top_k_ids - 1 :],
        )  # (batch_size * spec_length, 1)
        min_p_mask = top_probs < scaled_min_p  # (batch_size * spec_length, max_top_k_ids)
        topk_values_asc[min_p_mask] = torch.finfo(torch.float16).min

    # Update the logits
    logits.fill_(torch.finfo(torch.float16).min)
    logits = logits.scatter(1, topk_indices_asc, topk_values_asc)  # (batch_size * spec_length, vocab_size)
    # Softmax
    probs = torch.softmax(logits, dim=1).reshape(-1, spec_length, vocab_size)  # (batch_size, spec_length, vocab_size)

    return SamplerOutput(
        probs=probs,
        next_tokens=None,  # Return sampled next tokens instead of logits
        past_key_values=None,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer,
        past_presence_penalty_buffer=past_presence_penalty_buffer,
    )
