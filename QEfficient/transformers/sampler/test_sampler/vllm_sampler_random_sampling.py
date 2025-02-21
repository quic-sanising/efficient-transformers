import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


# _SAMPLING_EPS = 1e-5

@dataclass
class SamplingMetadata:

    temperature: torch.Tensor
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor
    top_k: torch.Tensor
    min_p: torch.Tensor
    no_top_p: bool
    no_top_k: bool
    no_min_p: bool

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


class TopKTopPSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
    ):

        # print("-"*100)
        # params = locals()
        # for param, value in params.items():
        #     print(f"{param}: {value}")
        # print("-"*100)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        sampled = random_sample(probs, generators)
        return probs, sampled


def random_sample(
    probs: torch.Tensor,
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div(q).argmax(dim=-1).view(-1)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        
        # print("-"*100)
        # params = locals()
        # for param, value in params.items():
        #     print(f"{param}: {value}")
        # print("-"*100)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Sample the next token.
        probs, sampled = self.sample(logits, sampling_metadata)
        return logits, probs, sampled
    
    
    # def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
    #     return logits.argmax(dim=-1).view(-1)


    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        # assert not (sampling_metadata.all_greedy
        #             and sampling_metadata.all_random)
        # if sampling_metadata.all_greedy:
        #     return self.greedy_sample(logits)

        probs, random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
        )
        # if sampling_metadata.all_random:
        #     return random_sampled

        # greedy_sampled = self.greedy_sample(logits)
        # sampled = torch.where(
        #     sampling_metadata.temperature < _SAMPLING_EPS,
        #     greedy_sampled,
        #     random_sampled,
        # )
        return probs, random_sampled
