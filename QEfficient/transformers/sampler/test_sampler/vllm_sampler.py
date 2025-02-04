import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union


_SAMPLING_EPS = 1e-5

@dataclass
class SamplingMetadata:

    temperature: torch.Tensor
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor
    top_k: torch.Tensor
    no_top_p: bool
    no_top_k: bool

    generators: Dict[int, torch.Generator]

    max_num_logprobs: int

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

    output_token_ids: List[List[int]]
    min_tokens: List[int]
    stop_token_ids: List[Set[int]]


@dataclass
class SamplerOutput:
    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: Optional[torch.Tensor]

    # TODO: Support prompt logprobs.
    prompt_logprob_token_ids: Optional[torch.Tensor]
    prompt_logprobs: Optional[torch.Tensor]


class TopKTopPSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
        no_top_k: bool,
        k: torch.Tensor,
        no_top_p: bool,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation of top-k and top-p sampling."""
        
        # print("-"*100)
        # params = locals()
        # for param, value in params.items():
        #     print(f"{param}: {value}")
        # print("-"*100)

        logits = apply_top_k_top_p(logits, no_top_k, k, no_top_p, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return probs


def apply_top_k_top_p(
    logits: torch.Tensor,
    no_top_k: bool,
    k: torch.Tensor,
    no_top_p: bool,
    p: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    This function sorts the logits tensor, which can be slow for large batches.
    """
    if no_top_k and no_top_p:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if not no_top_k:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, torch.finfo(torch.float16).tiny)

    if not no_top_p:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, torch.finfo(torch.float16).tiny)

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


# def random_sample(
#     probs: torch.Tensor,
#     generators: Dict[int, torch.Generator],
# ) -> torch.Tensor:
#     """Randomly sample from the probabilities.

#     We use this function instead of torch.multinomial because torch.multinomial
#     causes CPU-GPU synchronization.
#     """
#     q = torch.empty_like(probs)
#     # NOTE(woosuk): To batch-process the requests without their own seeds,
#     # which is the common case, we first assume that every request does
#     # not have its own seed. Then, we overwrite the values for the requests
#     # that have their own seeds.
#     if len(generators) != probs.shape[0]:
#         q.exponential_()
#     if generators:
#         # TODO(woosuk): This can be slow because we handle each request
#         # one by one. Optimize this.
#         for i, generator in generators.items():
#             q[i].exponential_(generator=generator)
#     return probs.div_(q).argmax(dim=-1).view(-1)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        
        # print("-"*100)
        # params = locals()
        # for param, value in params.items():
        #     print(f"{param}: {value}")
        # print("-"*100)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits, prompt_mask, output_mask = self.apply_penalties(logits, sampling_metadata)
        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Sample the next token.
        probs = self.sample(logits, sampling_metadata)
        return probs, prompt_mask, output_mask

    # def forward(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> SamplerOutput:
    #     needs_logprobs = sampling_metadata.max_num_logprobs > 0
    #     if needs_logprobs:
    #         # NOTE(woosuk): Use the original logits (before any penalties or
    #         # temperature scaling) for the top-k logprobs.
    #         # This is different from the V0 sampler, which uses the logits that
    #         # is used for sampling (after penalties and temperature scaling).
    #         # NOTE: We compute logprobs first because the below ops may
    #         # modify the logits tensor in-place (and we don't want to clone
    #         # the logits tensor for memory efficiency).
    #         topk_logprobs, topk_indices = self.get_topk_logprobs(
    #             logits, sampling_metadata)
    #     else:
    #         topk_logprobs = None
    #         topk_indices = None

    #     # Use float32 for the logits.
    #     logits = logits.to(torch.float32)
    #     # Apply penalties (e.g., min_tokens, freq_penalties).
    #     logits = self.apply_penalties(logits, sampling_metadata)
    #     # Apply temperature.
    #     logits = self.apply_temperature(logits, sampling_metadata.temperature)
    #     # Sample the next token.
    #     sampled = self.sample(logits, sampling_metadata)
    #     # Use int32 to reduce the tensor size.
    #     sampled = sampled.to(torch.int32)

    #     # NOTE: CPU-GPU synchronization happens here.
    #     sampler_output = SamplerOutput(
    #         sampled_token_ids=sampled,
    #         logprob_token_ids=topk_indices,
    #         logprobs=topk_logprobs,
    #         prompt_logprob_token_ids=None,
    #         prompt_logprobs=None,
    #     )
    #     return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Avoid division by zero.
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        # Use in-place division to avoid creating a new tensor.
        logits.div_(temp.unsqueeze(dim=1))
        return logits

    # def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
    #     return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.no_top_k,
            sampling_metadata.top_k,
            sampling_metadata.no_top_p,
            sampling_metadata.top_p,
        )
    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> torch.Tensor:
    #     return self.topk_topp_sampler(
    #         logits,
    #         sampling_metadata.generators,
    #         sampling_metadata.no_top_k,
    #         sampling_metadata.top_k,
    #         sampling_metadata.no_top_p,
    #         sampling_metadata.top_p,
    #     )
    #     # assert not (sampling_metadata.all_greedy
    #     #             and sampling_metadata.all_random)
    #     # if sampling_metadata.all_greedy:
    #     #     return self.greedy_sample(logits)

    #     # random_sampled = self.topk_topp_sampler(
    #     #     logits,
    #     #     sampling_metadata.generators,
    #     #     sampling_metadata.no_top_k,
    #     #     sampling_metadata.top_k,
    #     #     sampling_metadata.no_top_p,
    #     #     sampling_metadata.top_p,
    #     # )
    #     # if sampling_metadata.all_random:
    #     #     return random_sampled

    #     # greedy_sampled = self.greedy_sample(logits)
    #     # sampled = torch.where(
    #     #     sampling_metadata.temperature < _SAMPLING_EPS,
    #     #     greedy_sampled,
    #     #     random_sampled,
    #     # )
    #     # return sampled

    # def get_topk_logprobs(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
    #     # FIXME: Mask the sampled token_id, get topk logprobs,
    #     # and concatenate the topk with the sampled token_id.
    #     topk_logprobs, topk_indices = torch.topk(
    #         logprobs, sampling_metadata.max_num_logprobs, dim=-1)
    #     # Use int32 to reduce the tensor size.
    #     topk_indices = topk_indices.to(torch.int32)
    #     return topk_logprobs, topk_indices

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # apply_min_token_penalties(logits, sampling_metadata.output_token_ids,
        #                           sampling_metadata.stop_token_ids,
        #                           sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits, prompt_mask, output_mask = apply_all_penalties(
                logits, sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids)
        return logits, prompt_mask, output_mask


# def apply_min_token_penalties(logits: torch.Tensor,
#                               output_token_ids: List[List[int]],
#                               stop_token_ids: List[Set[int]],
#                               min_tokens: List[int]) -> None:
#     """
#     Applies minimum token penalty by setting the logits of the stop tokens
#     to -inf.
#     """
#     min_tokens_logits_to_penalize: List[Tuple[int, int]] = []
#     if min_tokens:
#         for index, min_token in enumerate(min_tokens):
#             if len(output_token_ids[index]) < min_token:
#                 for stop_token_id in stop_token_ids[index]:
#                     min_tokens_logits_to_penalize.append((index, stop_token_id))
#     if min_tokens_logits_to_penalize:
#         logits[tuple(zip(*min_tokens_logits_to_penalize))] = -float("inf")


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: List[List[int]],
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size,
                                          logits.device)
    return apply_penalties(logits, prompt_token_ids, output_tokens_t,
                           presence_penalties, frequency_penalties,
                           repetition_penalties)

T = TypeVar("T")
TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


def make_ndarray_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: npt.DTypeLike,
    *,
    max_len: Optional[int] = None,
) -> npt.NDArray:
    """
    Make a padded array from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    if max_len is None:
        # Unlike for most functions, map is faster than a genexpr over `len`
        max_len = max(map(len, x), default=0)

    padded_x = np.full((len(x), max_len), pad, dtype=dtype)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, :len(blocktb)] = blocktb

    return padded_x


def make_tensor_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


def _convert_to_tensors(output_token_ids: List[List[int]], vocab_size: int,
                        device: torch.device) -> torch.Tensor:
    """
    Convert the different list data structures to tensors.
    """

    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=False,
    )
    return output_tokens_tensor.to(device, non_blocking=True)


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)
    repetition_penalties = repetition_penalties.unsqueeze_(dim=1).repeat(
        1, vocab_size)
    logits[logits > 0] /= torch.where(prompt_mask | output_mask,
                                      repetition_penalties, 1.0)[logits > 0]
    logits[logits <= 0] *= torch.where(prompt_mask | output_mask,
                                       repetition_penalties, 1.0)[logits <= 0]
    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    # logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits, prompt_mask, output_mask
