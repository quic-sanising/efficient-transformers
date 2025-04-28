from dataclasses import dataclass
import torch
import torch.nn.functional as F

# from QEfficient.customop import CtxScatterFunc
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
    batch_size, spec_length, vocab_size = input_logits.shape
    
    # Select relevant rows
    batch_index_reshaped = batch_index.view(-1)
    past_repetition_penalty_buffer_selected = past_repetition_penalty_buffer[batch_index_reshaped]
    past_presence_penalty_buffer_selected = past_presence_penalty_buffer[batch_index_reshaped]

    logits = input_logits.reshape(-1, vocab_size)  # Reshape tensor to 2D

    if input_ids.shape[1] > spec_length:  # Prefill phase, initialize retained states
        # TODO: Replace scatter_ with CtxScatterFunc; Replace -1 with int_max while exporting on onnx
        # past_repetition_penalty_buffer_selected = CtxScatterFunc.apply(past_repetition_penalty_buffer_selected.unsqueeze(1), input_ids, 1).squeeze(1)
        if position_ids[0, 0] == 0:
            past_repetition_penalty_buffer_selected = torch.zeros(past_repetition_penalty_buffer_selected.shape, dtype=torch.bool)
            past_presence_penalty_buffer_selected = torch.zeros(past_presence_penalty_buffer_selected.shape, dtype=torch.bool)
        past_repetition_penalty_buffer_selected.scatter_(1, input_ids, 1)

    else:  # Decode phase, update retained states
        past_repetition_penalty_buffer_selected.scatter_(1, last_accepted_output_tokens, 1)
        past_presence_penalty_buffer_selected.scatter_(1, last_accepted_output_tokens, 1)
        # TODO: For frequency retain state, first gather and then scatter

    # Update relevant rows in original tensors
    past_repetition_penalty_buffer[batch_index_reshaped] = past_repetition_penalty_buffer_selected
    past_presence_penalty_buffer[batch_index_reshaped] = past_presence_penalty_buffer_selected
                                         
    # Repetition Penalty
    if (repetition_penalties != 1.).any():
        repetition_penalties = repetition_penalties.repeat(spec_length, vocab_size)  # (batch_size, 1) -> (batch_size * spec_length, vocab_size)
        past_repetition_penalty_buffer_selected = past_repetition_penalty_buffer_selected.repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        repetition_penalties[past_repetition_penalty_buffer_selected == 0] = 1.0
        logits = torch.where(logits > 0, logits / repetition_penalties, logits * repetition_penalties)

    # Presence Penalty
    if (presence_penalties != 0.).any():
        presence_penalties = presence_penalties.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
        past_presence_penalty_buffer_selected = past_presence_penalty_buffer_selected.repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
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
