from dataclasses import dataclass
import torch
import torch.nn.functional as F

from QEfficient.customop import CtxScatterFuncCB3D, CtxScatterFunc3D
from QEfficient.utils.constants import Constants
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from typing import List, Optional, Tuple, Union


@dataclass
class QEffCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    probs: torch.FloatTensor = None
    next_tokens: torch.IntTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_repetition_penalty_buffer: Optional[torch.Tensor] = None
    past_presence_penalty_buffer: Optional[torch.Tensor] = None


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
) -> Union[Tuple, CausalLMOutputWithPast]:
    # Move to device
    device = input_logits.device
    last_accepted_output_tokens = last_accepted_output_tokens.to(device)
    past_repetition_penalty_buffer = past_repetition_penalty_buffer.to(device)
    past_presence_penalty_buffer = past_presence_penalty_buffer.to(device)
    repetition_penalties = repetition_penalties.to(device)
    presence_penalties = presence_penalties.to(device)
    temperatures = temperatures.to(device)
    top_ks = top_ks.to(device)
    top_ps = top_ps.to(device)
    min_ps = min_ps.to(device)

    # Perform Sampling
    batch_size, spec_length, vocab_size = logits.shape
    logits = input_logits.reshape(-1, vocab_size)  # Reshape tensor to 2D
    # FIXME: Create 3D retain states

    # --- Prefill ---
    if input_ids.shape[1] > spec_length:
        prefill_indices = torch.nonzero(position_ids[:, 0] == 0)
        if prefill_indices.shape[0] > 0:
            # First input chunk, so initialize retain states
            mul_value = torch.ones(past_repetition_penalty_buffer.shape[0], 1, dtype=torch.bool)
            mul_value = CtxScatterFunc3D.apply(
                mul_value, prefill_indices, torch.zeros(prefill_indices.shape, dtype=torch.bool))
            # mul_value[prefill_indices] = 0
            past_repetition_penalty_buffer *= mul_value
            past_presence_penalty_buffer *= mul_value

        # Mask out-of-bounds or invalid position_ids or input_ids
        input_ids = torch.where(position_ids == -1, -1, input_ids)
        input_ids = torch.where(
            (input_ids < 0) | (input_ids >= vocab_size), torch.iinfo(torch.int32).max, input_ids)

        # Chunked input, so update retain states
        past_repetition_penalty_buffer = CtxScatterFuncCB3D.apply(
            past_repetition_penalty_buffer, batch_index, input_ids, torch.ones(input_ids.shape, dtype=torch.bool))

    # --- Decode ---
    else:
        # Mask out-of-bounds or invalid position_ids or last_accepted_output_tokens
        last_accepted_output_tokens = torch.where(position_ids == -1, -1, last_accepted_output_tokens)
        last_accepted_output_tokens = torch.where(
            (last_accepted_output_tokens < 0) | (last_accepted_output_tokens >= vocab_size), torch.iinfo(torch.int32).max, last_accepted_output_tokens)
        
        # Update retained states
        scatter_values = torch.ones(
            last_accepted_output_tokens.shape, dtype=torch.bool)
        past_repetition_penalty_buffer = CtxScatterFuncCB3D.apply(
            past_repetition_penalty_buffer, batch_index, last_accepted_output_tokens, scatter_values)
        past_presence_penalty_buffer = CtxScatterFuncCB3D.apply(
            past_presence_penalty_buffer, batch_index, last_accepted_output_tokens, scatter_values)
        # TODO: For frequency retain state, first gather and then scatter

    batch_index_reshaped = batch_index.view(-1)
    # Repetition Penalty
    if (repetition_penalties != 1.).any():
        past_repetition_penalty_buffer_selected = past_repetition_penalty_buffer[batch_index_reshaped].repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        repetition_penalties_mask = torch.where(past_repetition_penalty_buffer_selected, repetition_penalties, 1.)
        logits *= (repetition_penalties_mask ** (-torch.sign(logits)))

    # Presence Penalty
    if (presence_penalties != 0.).any():
        presence_penalties = presence_penalties.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
        past_presence_penalty_buffer_selected = past_presence_penalty_buffer[batch_index_reshaped].repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        logits -= presence_penalties * past_presence_penalty_buffer_selected

    # TODO: Frequency Penalty

    # Temperature Scaling
    temperatures = temperatures.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
    logits = torch.where(temperatures != 0, logits / temperatures, logits)

    # Top K
    # TODO (Optimization): if (top_ks != -1 or top_ks != Constants.MAX_TOP_K_IDS).any() is False: skip but will need topk_values_asc and topk_indices_asc
    topk_values, topk_indices = torch.topk(logits, k=Constants.MAX_TOP_K_IDS, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_values_asc = topk_values.flip(dims=[1])
    topk_indices_asc = topk_indices.flip(dims=[1])
    top_ks[top_ks > Constants.MAX_TOP_K_IDS] = Constants.MAX_TOP_K_IDS  # Clip k to max value
    # True values in this mask indicate the positions of the non-top K values
    topk_mask = torch.arange(topk_values_asc.shape[1]).unsqueeze(0) < (topk_values_asc.size(1) - top_ks.to(torch.long)).repeat(spec_length, 1)  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    topk_values_asc[topk_mask] = torch.finfo(torch.float16).min

    # Top P
    # TODO (Optimization): if (top_ps != 1.).any() is False: skip but will need top_probs for Min P
    top_probs = torch.softmax(topk_values_asc, dim=1)  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    topk_probs_sum = torch.cumsum(top_probs, dim=1)
    top_p_mask = topk_probs_sum <= 1 - top_ps.repeat(spec_length, 1)  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    top_p_mask[:, Constants.MAX_TOP_K_IDS - 1] = False
    topk_values_asc[top_p_mask] = torch.finfo(torch.float16).min

    # Min P
    if (min_ps != 0.).any():
        scaled_min_p = torch.mul(min_ps.repeat(spec_length, 1), top_probs[:, Constants.MAX_TOP_K_IDS - 1:])  # (batch_size * spec_length, 1)
        min_p_mask = top_probs < scaled_min_p  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
        topk_values_asc[min_p_mask] = torch.finfo(torch.float16).min

    # Update the logits
    logits.fill_(torch.finfo(torch.float16).min)
    logits = logits.scatter(1, topk_indices_asc, topk_values_asc)  # (batch_size * spec_length, vocab_size)
    # Softmax
    probs = torch.softmax(logits, dim=1).reshape(-1, spec_length, vocab_size)  # (batch_size, spec_length, vocab_size)

    return QEffCausalLMOutputWithPast(
        loss=None,
        probs=probs,  # Return probabilities instead of logits
        next_tokens=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer,
        past_presence_penalty_buffer=past_presence_penalty_buffer,
    )
