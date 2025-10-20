"""boostedprob

Small utilities to compute "dominant tokens" and boosted probabilities.

Public API:
- find_dominant(log_probs, ...)
- calculate_boostedprob(log_probs, target, ...)
"""
from typing import Optional
import torch
import itertools

__all__ = ["find_dominant", "calculate_boostedprob"]

def find_dominant(
    log_probs: torch.Tensor, 
    find_dominant_method: str = "difference_jump", 
    epsilon: Optional[float] = 0.005,
    k: Optional[int] = 5,
    p_jump: Optional[float] = 0.3, 
    minp: Optional[float] = 0.9,
    topp: Optional[float] = 0.9,
) -> torch.Tensor:
    """
    Find dominant tokens based on various methods.
    Args:
        log_probs (torch.Tensor): Model log probabilities (output of final softmax) 
            of shape [batch_size, nr_tokens, vocab_size].
        find_dominant_method (str): Method to find dominant tokens. Options include:
            - "epsilon-cut": Tokens with probability above a certain threshold.
            - "eta-cut": Tokens based on entropy and a threshold epsilon.
            - "top-k": Top k tokens by probability.
            - "top-p": Tokens whose cumulative probability is below p.
            - "min-p": Tokens with probability above a fraction of the highest probability token.
            - "difference_jump": Tokens where the difference in sorted probabilities exceeds a jump threshold.
        p_threshold (float, optional): Probability threshold for "prob_threshold" method.
        epsilon (float, optional): Minimum threshold for 
        k (int, optional): Number of top tokens for "top-k" method.
        p_jump (float, optional): Jump threshold for "difference_jump" method.
        minp (float, optional): Minimum probability fraction for "min-p" method.
        topp (float, optional): Cumulative probability threshold for "top-p" method.
    Returns:
        torch.Tensor: Indices of dominant tokens, shape [batch_size, nr_tokens, vocab_size]. -1 values are used to mask out non-dominant tokens.
    """

    prob_dist = log_probs.exp_()
    # sorted_prob_dist, indices = torch.sort(prob_dist, descending=True, dim=-1)
    
    # Choose a smaller k internally for memory efficiency
    internal_k = min(1024, prob_dist.size(-1))  # adjust based on memory
    sorted_prob_dist, indices = torch.topk(prob_dist, k=internal_k, dim=-1, largest=True, sorted=True)

    if find_dominant_method == "epsilon-cut":
        assert epsilon is not None
        mask = sorted_prob_dist > epsilon
    elif find_dominant_method == "eta-cut":
        assert epsilon is not None
        epsilon = torch.tensor(epsilon)
        entropy = -torch.mul(torch.log(prob_dist), prob_dist).sum(dim=-1)
        entropy = entropy.unsqueeze(-1).expand_as(prob_dist)
        mask = (sorted_prob_dist > epsilon) | (sorted_prob_dist > torch.sqrt(epsilon) * torch.exp(-entropy))
    elif find_dominant_method == "top-k":
        assert k is not None
        mask = torch.zeros_like(sorted_prob_dist, dtype=torch.bool)
        mask[...,:k] = True
    elif find_dominant_method == "top-p":
        assert topp is not None
        cumulative_sum = torch.cumsum(sorted_prob_dist, dim=-1)
        mask = cumulative_sum < topp
    elif find_dominant_method == "min-p":
        assert minp is not None
        mask = sorted_prob_dist > minp * sorted_prob_dist[...,0].unsqueeze(-1).expand_as(sorted_prob_dist)
    elif find_dominant_method == "difference_jump":
        assert p_jump is not None
        diff = sorted_prob_dist[..., :-1] - sorted_prob_dist[..., 1:]
        # Identify the cutoff condition along the last dimension
        mask = (diff > p_jump * sorted_prob_dist[..., :-1]) & (diff > epsilon)
    else:
        raise RuntimeError(f"Unknown find_dominant_method {find_dominant_method}")

    # Get the last occurrence of True along the last axis
    cut_points = mask.shape[-1] - 1 - torch.argmax(torch.flip(mask, dims=[-1]).int(),
                                                   dim=-1)  # Shape: [batch_size, nr_tokens]

    # Handle cases where no cutoff is found (all False)
    no_cutoff = ~mask.any(axis=-1)
    cut_points[no_cutoff] = -1  # Use -1 to indicate no valid cutoff found


    # Assuming `indices` is of shape [batch_size, nr_tokens, vocab_size]
    batch_indices = torch.arange(indices.shape[-1], device=indices.device).expand_as(indices)

    # Ensure cut_point has the same shape as batch_indices (for broadcasting)
    cut_point_expanded = cut_points.unsqueeze(-1)  # Shape: [batch_size, nr_tokens, 1]

    # Create mask: Select elements up to cut_point, but disable selection when cut_point == -1
    mask = (batch_indices <= cut_point_expanded) & (cut_point_expanded != -1)

    # Mask out indices beyond the cut-off point with value -1
    indices = torch.where(mask, indices, -1)

    # Return prob to log prob inplace to avoid downstream error
    log_probs = prob_dist.log_()

    # The indices so far only contains the `internal_k` positions. Now pad to vocab_size
    # full_indices = torch.full_like(prob_dist, -1, dtype=torch.long)
    # full_indices[..., :internal_k] = indices

    return indices


def calculate_boostedprob(
        log_probs: torch.Tensor, 
        target: torch.Tensor,
        ue_method: str = "sum_dominant_mass",
        find_dominant_method: str = "difference_jump", 
        epsilon: Optional[float] = 0.005,
        k: Optional[int] = 5,
        p_jump: Optional[float] = 0.3, 
        minp: Optional[float] = 0.9,
        topp: Optional[float] = 0.9,
    ):
    """
    Calculate boosted probabilities based on dominant tokens.
    Args:
        log_probs (torch.Tensor): Model log probabilities (output of final softmax) 
            of shape [batch_size, nr_tokens, vocab_size].
        target (torch.Tensor): Indices finally output tokens of shape [batch_size, nr_tokens].
        ue_method (str): Uncertainty estimation method. Options include:
            - "is_dominant": Returns 1 if the predicted token is dominant, else 0.
            - "sum_dominant_mass": Returns the sum of probabilities of dominant tokens, or the predicted token's probability if it's not dominant.
        find_dominant_method (str): Method to find dominant tokens. Options include:
            - "epsilon-cut": Tokens with probability above a certain threshold.
            - "eta-cut": Tokens based on entropy and a threshold epsilon.
            - "top-k": Top k tokens by probability.
            - "top-p": Tokens whose cumulative probability is below p.
            - "min-p": Tokens with probability above a fraction of the highest probability token.
            - "difference_jump": Tokens where the difference in sorted probabilities exceeds a jump threshold.
        epsilon (float, optional): Minimum threshold for 
        k (int, optional): Number of top tokens for "top-k" method.
        p_jump (float, optional): Jump threshold for "difference_jump" method.
        minp (float, optional): Minimum probability fraction for "min-p" method.
        topp (float, optional): Cumulative probability threshold for "top-p" method.
    Returns:
        torch.Tensor: Boosted probabilities or binary indicators, shape [batch_size, nr_tokens].
    """
    dominant_indices = find_dominant(
        log_probs=log_probs,
        find_dominant_method=find_dominant_method,
        epsilon=epsilon,
        k=k,
        p_jump=p_jump,
        minp=minp, 
        topp=topp, 
    )  # Shape: [batch_size, nr_tokens, vocab_size]

    # Check if each predicted_id is in the corresponding dominant indices
    is_dominant = (target.unsqueeze(-1) == dominant_indices).any(dim=-1)  # Shape: [batch_size, nr_tokens]

    if ue_method == "is_dominant":
        return is_dominant.int()  # Convert to 0/1 format
    elif ue_method == "sum_dominant_mass":
        trans_probs = torch.exp(log_probs)
        dominant_binary = torch.zeros_like(trans_probs, dtype=torch.uint8)  # Initialize binary tensor

        lists = [list(range(x)) for x in trans_probs.shape[:-1]]
        if len(lists) > 0:
            combs = tuple(itertools.product(*lists))
            for comb in combs:
                dominant_binary[comb][dominant_indices[comb][dominant_indices[comb] != -1]] = 1
        else:
            dominant_binary[dominant_indices[dominant_indices != -1]] = 1

        dominant_mass = (dominant_binary * trans_probs).sum(dim=-1)
        selected_prob = trans_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        sum_dominant_mass = torch.where(is_dominant, dominant_mass, selected_prob)
        return sum_dominant_mass
    else:
        raise RuntimeError(f"Unknown ue_method {ue_method}")
