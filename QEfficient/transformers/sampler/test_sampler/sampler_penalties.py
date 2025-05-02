from dataclasses import dataclass
import torch
import copy

from QEfficient.customop import CtxScatterFuncCB3D, CtxScatterFunc3D
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from typing import Optional, Tuple, Union


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
) -> Union[Tuple, CausalLMOutputWithPast]:
    # Move to device
    device = input_logits.device
    last_accepted_output_tokens = last_accepted_output_tokens.to(device)
    past_repetition_penalty_buffer = past_repetition_penalty_buffer.to(device)
    past_presence_penalty_buffer = past_presence_penalty_buffer.to(device)
    repetition_penalties = repetition_penalties.to(device)
    presence_penalties = presence_penalties.to(device)

    # Perform Sampling
    batch_size, spec_length, vocab_size = input_logits.shape
    
    logits = input_logits.reshape(-1, vocab_size)  # Reshape tensor to 2D

    batch_index_reshaped = batch_index.view(-1)
    # --- Prefill ---
    if input_ids.shape[1] > spec_length:
        if (position_ids[:, 0] == 0).any():
            # First input chunk, so initialize retain states
            mul_value = torch.ones(past_repetition_penalty_buffer.shape[0], 1, dtype=torch.bool)
            mul_value[batch_index_reshaped] = position_ids[:, :1] != 0
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
    
    return QEffCausalLMOutputWithPast(
        loss=None,
        probs=logits,
        next_tokens=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer,
        past_presence_penalty_buffer=past_presence_penalty_buffer,
    )
