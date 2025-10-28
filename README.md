# BoostedProb

Implementation of Boosted Model Probability (BoostedProb) for Quality Estimation introduced in the paper: [Are Generative Models Underconfident? Better Quality Estimation with Boosted Model Probability](https://arxiv.org/abs/2502.11115) (EMNLP 2025 Main).

We provide 2 functions:
- `find_dominant(log_probs)` <br>
    Identifies the indices of tokens that have dominant probability values within each distribution.
- `calculate_boostedprob(log_probs, target)` <br> Computes the BoostedProb for each output token ID specified in `target`:
    - If the token is dominant: returns the sum of probabilities of all dominant tokens
    - If the token is not dominant: returns the probability of that token itself.

Toy example:
```python
import torch
import boostedprob

log_probs = torch.log(torch.tensor([
    [0.5, 0.4, 0.05, 0.05],
    [0.5, 0.4, 0.05, 0.05],
]))  # shape [nr_tokens, vocab_size]

target = torch.tensor([2, 1])  # shape [nr_tokens, 1]

# Find dominant tokens
print(boostedprob.find_dominant(log_probs))
# Output
# tensor([[ 0,  1, -1, -1],  tokens at position 0 and 1 are dominant
#         [ 0,  1, -1, -1]])   tokens at position 0 and 1 are dominant
# -1 are dummy values to be ignored.

# Calculate boosted prob (find_dominant() runs internally)
result = boostedprob.calculate_boostedprob(log_probs, target)   # shape [nr_tokens, 1]
# Output
# tensor([0.0500, 0.9000])
```

## Install

From PyPI (recommended):
```bash
pip install boostedprob
```

From GitHub (latest development version):
```bash
pip install "git+https://github.com/TuAnh23/boostedprob.git"
```

Or install locally in editable mode:

```bash
git clone https://github.com/TuAnh23/boostedprob.git
cd boostedprob
pip install -e .
```

## Examples

See the `examples/` folder for integration with Hugging Face models.