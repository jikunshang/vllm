# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn.utils.rnn import pad_sequence

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.sample.ops.utils import compiled_softmax
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32
INVALID_TOKEN_ID = -1


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        if current_platform.is_xpu():
            target_probs = compute_probs_native(
                target_logits,
                sampling_metadata,
                metadata.num_draft_tokens,
            )

            return self.forward_xpu(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                draft_probs,
                bonus_token_ids,
                target_probs,
                sampling_metadata,
            )
        # [num_tokens, vocab_size]
        target_probs = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
        )
        return output_token_ids

    def forward_xpu(
        self,
        draft_token_ids_tensor: torch.Tensor,
        num_draft_tokens: list[int],
        draft_probs: Optional[torch.Tensor],
        bonus_token_ids_tensor: torch.Tensor,  # [batch_size, 1]
        target_probs: torch.Tensor,  # [num_total_tokens, vocab_size]
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            draft_token_ids (List[List[int]]):
                A 2D list of token IDs for each request in the batch.
                Each request might have different number of draft tokens.
                It may also contain empty lists for requests that have
                no draft tokens.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [batch_size, max_spec_len, vocab_size]. Can be None if
                probabilities are not provided, which is the case for
                ngram spec decode.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            target_probs (torch.Tensor):
                Target model probability distribution.
                Shape is [num_total_tokens, vocab_size]. num_total_tokens
                is the total number of tokens from all requests. Here,
                probabilities from different requests are flattened into
                a single tensor because this is the shape of the output
                logits.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        # NOTE: The following input preparationg can be moved
        # to the model runner with a persistent manner for better
        # performance.
        # Convert draft token IDs to a tensor, split by sample_lens, then pad.
        #draft_token_ids = [
        #    torch.tensor(x, dtype=int, device='cpu') for x in draft_token_ids
        #]
        draft_token_ids_split = torch.split(draft_token_ids_tensor,
                                            num_draft_tokens,
                                            dim=0)
        draft_token_ids_tensor = pad_sequence(draft_token_ids_split,
                                              batch_first=True,
                                              padding_value=INVALID_TOKEN_ID)
        # NOTE: CPU <-> GPU synchronization happens here.
        draft_token_ids_tensor = draft_token_ids_tensor.to(target_probs.device)

        # Create one-hot tensor for draft token ids.
        # This is used for ngram where we don't have draft_probs.
        if draft_probs is None and not sampling_metadata.all_greedy:
            vocab_size = target_probs.size(-1)
            draft_probs = _create_greedy_token_probs(draft_token_ids_tensor,
                                                     vocab_size,
                                                     target_probs.device)
        sample_lens = num_draft_tokens
        #[len(x) + 1 for x in draft_token_ids]
        target_probs = _convert_2d_probs(target_probs, sample_lens)

        return self.forward_native(draft_token_ids_tensor, draft_probs,
                                   bonus_token_ids_tensor, target_probs,
                                   num_draft_tokens, sampling_metadata)

    # TODO: The following method can be optimized for better performance.
    def forward_native(
        self,
        draft_token_ids_tensor: torch.Tensor,
        # [batch_size, max_spec_len, vocab_size]
        draft_probs: Optional[torch.Tensor],
        bonus_token_ids_tensor: torch.Tensor,
        # [batch_size, max_spec_len + 1, vocab_size]
        target_probs: torch.Tensor,
        num_draft_tokens: list[int],
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # Add 1 to include the 'bonus' token.
        if sampling_metadata.all_greedy:
            # Produce a mask that remains 1 (True) until the first
            # mismatch (cumprod turns 0 after a mismatch).
            target_token_ids_tensor = target_probs.argmax(dim=-1)
            accept_mask = (target_token_ids_tensor[:, :] ==
                           draft_token_ids_tensor).cumprod(dim=1)

            # Identify valid positions (non-padding).
            valid_mask = target_token_ids_tensor != INVALID_TOKEN_ID
            # Generate mask with bonus token.
            '''
            valid_mask_extended = torch.cat([
                valid_mask,
                torch.ones_like(valid_mask[:, 0:1])
                ], dim=1)
            '''
            generate_mask = accept_mask.to(torch.bool) & valid_mask
            zeros_mask = (generate_mask == 0)
            first_zero_idx = zeros_mask.float().argmax(dim=1)
            # Figure out which rows actually contain at least one zero.
            rows_with_zero = zeros_mask.any(dim=1)
            # Use indexing to set the first zero in each of those rows to 1.
            generate_mask[rows_with_zero, first_zero_idx[rows_with_zero]] = 1

            output_token_ids = target_token_ids_tensor
            output_token_ids[~generate_mask] = INVALID_TOKEN_ID
        else:
            # Reference: https://arxiv.org/pdf/2211.17192
            # 1. Extract the probabilities of the draft tokens.
            # [batch_size, max_spec_len]
            batch_size = draft_token_ids_tensor.size(0)
            max_spec_len = draft_token_ids_tensor.size(1)
            invalid_idx = draft_token_ids_tensor == INVALID_TOKEN_ID
            draft_token_ids_tensor[invalid_idx] = 0
            assert draft_probs is not None
            draft_token_probs = draft_probs.gather(
                dim=-1, index=draft_token_ids_tensor.unsqueeze(-1)).squeeze(-1)
            target_token_probs = target_probs.gather(
                dim=-1, index=draft_token_ids_tensor.unsqueeze(-1)).squeeze(-1)
            # Force the probabilities of invalid tokens to inf
            # so that they are not accepted.
            draft_token_probs[invalid_idx] = float('inf')

            # 2. Generate uniform samples.
            # [batch_size, max_spec_len + 1]
            uniform_samples = _create_uniform_samples(
                sampling_metadata.generators, batch_size, max_spec_len,
                target_probs.device)

            # 3. Accept or reject the samples.
            # [batch_size, max_spec_len]
            # If the draft token probabilities are 0, set them to the smallest
            # positive normal value representable by float32.
            safe_draft_probs = torch.where(draft_token_probs > 0,
                                           draft_token_probs,
                                           torch.finfo(torch.float32).tiny)
            accepted = uniform_samples <= target_token_probs / safe_draft_probs
            accept_mask = accepted.cumprod(dim=1)
            # Set the token ids to the draft token ids if accepted, otherwise
            # set them to INVALID_TOKEN_ID.
            accepted_token_ids = (draft_token_ids_tensor * accept_mask +
                                  INVALID_TOKEN_ID * (1 - accept_mask))

            # 4. Adjust the distribution for the recovered tokens.
            # Clamp the bonus probabilities to the smallest positive normal
            # value representable by float32.
            bonus_prob = torch.clamp(target_probs[:, :-1, :] - draft_probs,
                                     min=torch.finfo(torch.float32).tiny)
            normalized_bonus_prob = bonus_prob / bonus_prob.sum(dim=-1,
                                                                keepdim=True)

            # 5. Sample recovered token ids.
            recovered_token_ids = random_sample(
                normalized_bonus_prob,
                sampling_metadata.generators).reshape(batch_size, max_spec_len)

            # 6. Get the final output token ids.
            # output_token_ids = accepted_token_ids +
            #                    recovered_token_ids +
            #                    bonus_token_id
            recovered_bonus_token_ids = torch.cat(
                [recovered_token_ids, bonus_token_ids_tensor], dim=1)
            # Generate mask with bonus tokens.
            generate_mask = torch.cat([
                accept_mask,
                torch.zeros(batch_size, 1, device=accept_mask.device)
            ],
                                      dim=1).to(torch.bool)
            zeros_mask = (generate_mask == 0)
            first_zero_idx = zeros_mask.float().argmax(dim=1)
            output_token_ids = torch.cat([
                accepted_token_ids,
                torch.full((batch_size, 1),
                           fill_value=INVALID_TOKEN_ID,
                           device=accept_mask.device)
            ],
                                         dim=1)
            output_token_ids[torch.arange(batch_size),
                             first_zero_idx] = recovered_bonus_token_ids[
                                 torch.arange(batch_size), first_zero_idx]

        return output_token_ids

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
    ) -> list[list[int]]:
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = ((output_token_ids_np != PLACEHOLDER_TOKEN_ID) &
                      (output_token_ids_np < vocab_size))
        outputs = [
            row[valid_mask[i]].tolist()
            for i, row in enumerate(output_token_ids_np)
        ]
        return outputs


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size, )](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            num_warps=1,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel[(batch_size, )](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        IS_NGRAM=draft_probs is None,
        num_warps=1,
    )
    return output_token_ids


def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    batch_size = cu_num_draft_tokens.shape[0]
    expanded_temperature = torch.empty(
        (num_tokens, 1),
        dtype=torch.float32,
        device=logits.device,
    )
    expand_kernel[(batch_size, )](
        expanded_temperature,
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        GREEDY_TEMPERATURE,  # replace_from
        1,  # replace_to
        MAX_NUM_TOKENS=MAX_SPEC_LEN,
        num_warps=1,
    )
    output_prob = compiled_softmax(logits, expanded_temperature)
    return output_prob


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens : int
            Total number of tokens.
        num_draft_tokens : List[List[int]]
            Number of draft tokens per request.
        generators : Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device : torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand : torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    uniform_probs = torch.rand(
        (num_tokens, ),
        dtype=torch.float32,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        triton.next_power_of_2(vocab_size),
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    if is_greedy_ptr is None:
        is_greedy = True
    else:
        is_greedy = tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                     target_argmax_id)
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            num_draft_tokens, bonus_token_id)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if IS_NGRAM:
                draft_prob = 1
            else:
                draft_prob = tl.load(draft_probs_ptr +
                                     (start_idx + pos) * vocab_size +
                                     draft_token_id)
            target_prob = tl.load(target_probs_ptr +
                                  (start_idx + pos) * vocab_size +
                                  draft_token_id)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                     token_id)

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            num_draft_tokens, bonus_token_id)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset,
             src_val,
             mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    IS_NGRAM: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    if IS_NGRAM:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                            draft_token_id)
        # Temporarily zero out the probability of the draft token.
        # This is essentially the same as target_prob - draft_prob, except that
        # n-gram does not have draft_prob. We regard it as 1.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            0)
        prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                       vocab_offset,
                       mask=vocab_offset < vocab_size,
                       other=0)
    else:
        draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size +
                             vocab_offset,
                             mask=vocab_offset < vocab_size,
                             other=0)
        target_prob = tl.load(target_probs_ptr +
                              (start_idx + pos) * vocab_size + vocab_offset,
                              mask=vocab_offset < vocab_size,
                              other=0)
        prob = tl.maximum(target_prob - draft_prob, 0)
        # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
        # `tl.argmax` will select the maximum value.

    q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset,
                mask=vocab_offset < vocab_size,
                other=float("-inf"))
    recovered_id = tl.argmax(prob / q, axis=-1)
    tl.store(output_token_ids_ptr + start_idx + pos, recovered_id)

    if IS_NGRAM:
        # Restore the original probability.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            orig_prob)


def _create_greedy_token_probs(
    token_ids: torch.Tensor,
    vocab_size: int,
    out_device: torch.device,
) -> torch.Tensor:
    batch_size, num_tokens = token_ids.shape

    token_probs = torch.zeros(batch_size,
                              num_tokens,
                              vocab_size,
                              dtype=torch.float,
                              device=out_device)

    # Ignore INVALID_TOKEN_ID.
    valid_mask = (token_ids != INVALID_TOKEN_ID)
    valid_indices = token_ids.clone()
    valid_indices[~valid_mask] = 0
    token_probs.scatter_(dim=2,
                         index=valid_indices.unsqueeze(-1),
                         src=valid_mask.unsqueeze(-1).float())

    return token_probs


def _convert_2d_probs(
        probs: torch.Tensor,  # [num_total_tokens, vocab_size]
        sample_lens: list[int]) -> torch.Tensor:
    """
        Converts a 2D tensor of probabilities to a 3D tensor with padding.
        [num_total_tokens, vocab_size] ->
            [batch_size, max_spec_len + 1, vocab_size]
    """
    cumulative_lens = torch.cumsum(torch.tensor(sample_lens,
                                                device=probs.device),
                                   dim=0)
    split_indices = cumulative_lens[:-1].tolist()  # Exclude last index

    # Split into chunks without loops
    chunks = torch.tensor_split(probs, split_indices, dim=0)

    # Pad all sequences to maximum length
    padded_probs = pad_sequence(chunks, batch_first=True, padding_value=0.0)
    return padded_probs


def _create_uniform_samples(seeded_seqs: dict[int, torch.Generator],
                            batch_size: int, k: int,
                            device: torch.device) -> torch.Tensor:
    """
        Generates a batch of uniform random samples, with optional seeding
        for specific sequences.

        This method creates a tensor of shape `(batch_size, k)` filled
        with uniform random values in the range [0, 1). If `seeded_seqs`
        is provided, the sequences corresponding to specific indices
        will be generated using the provided `torch.Generator` for
        reproducibility. The other sequences will be generated without
        a seed.

        Args:
            seeded_seqs : Optional[Dict[int, torch.Generator]]
                A dictionary mapping indices in the batch to
                `torch.Generator` objects.
            batch_size : int
                The number of sequences to generate.
            k : int
                The number of random samples per sequence.
            device : torch.device
                The device on which to allocate the tensor.

        Returns:
            uniform_rand : torch.Tensor
                A tensor of shape `(batch_size, k)` containing uniform
                random values in the range [0, 1).
        """

    uniform_rand = torch.rand(batch_size,
                              k,
                              dtype=torch.float32,
                              device=device)
    # Apply seeded generators only where needed
    if seeded_seqs:
        for idx, generator in seeded_seqs.items():
            uniform_rand[idx].uniform_(0, 1, generator=generator)
    return uniform_rand


def compute_probs_native(logits: torch.Tensor,
                         sampling_metadata: SamplingMetadata,
                         sample_lens: list[int]) -> torch.Tensor:
    """
        Compute probability distribution from logits based on sampling metadata.

        This function applies temperature scaling to the logits and converts
        them to probabilities using softmax. Note that division by
        temperature is not performed inplace to preserve the original logits
        tensor, which will be used by the original sampler to get bonus tokens.

        Args:
            logits: Input logits tensor to be converted to probabilities
            sampling_metadata: Metadata containing sampling parameters such
                    as temperature and whether greedy sampling is used
            sample_lens: List of sample lengths used for repeating
                    temperature values

        Returns:
            torch.Tensor: Probability distribution (softmax of scaled logits)
                    if non-greedy sampling is used, otherwise returns the
                    original logits
        """
    if sampling_metadata.all_greedy:
        return logits
    assert sampling_metadata.temperature is not None
    # We should optimize the following code as
    # it will cause CPU -> GPU synchronization.
    temperature = torch.repeat_interleave(
        sampling_metadata.temperature,
        torch.tensor(sample_lens, device=sampling_metadata.temperature.device))
    temperature = temperature.unsqueeze(dim=1)
    logits = logits / temperature
    return logits.softmax(dim=-1, dtype=torch.float32)
