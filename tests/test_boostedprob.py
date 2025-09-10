import torch
import boostedprob


def test_calculate_boostedprob_example():
    log_probs = torch.log(torch.tensor([
        [0.5, 0.4, 0.05, 0.05],
        [0.5, 0.4, 0.05, 0.05],
    ]))

    target = torch.tensor([2, 1])

    result = boostedprob.calculate_boostedprob(log_probs, target)

    expected = torch.tensor([0.0500, 0.9000])
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
