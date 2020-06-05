from experiments.misc import negative_sampling
import torch


def test_negative_sampling():
    """ Unit test for negative_sampling function """

    i = {i: i for i in range(5000)}
    x = [[a, a, a] for a in range(5000)]

    # Check if head and tail are different
    counter = 0
    for head, _, tail in torch.tensor(negative_sampling(x, i, 1)):
        if head == tail:
            counter += 1
    print(counter)
    assert counter/50000 < 0.01, "Number of false negatives exceeds 1%!"

    # Check sampling rate
    assert torch.tensor(negative_sampling(x, i, 1)).shape[0] == len(x) * 1, \
        "Negative sampling rate of 1 did not return tensor with expected shape"
    assert torch.tensor(negative_sampling(x, i, 2)).shape[0] == len(x) * 2, \
        "Negative sampling rate of 2 did not return tensor with expected shape"
    assert torch.tensor(negative_sampling(x, i, 3)).shape[0] == len(x) * 3, \
        "Negative sampling rate of 3 did not return tensor with expected shape"
