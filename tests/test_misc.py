from experiments.misc import negative_sampling, perturb, filter_triples, compute_mrr, compute_hits, rank_triple
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

    assert torch.tensor(negative_sampling(x, i, 1)).shape[0] == len(x) * 1, \
        "Negative sampling rate of 1 did not return tensor with expected shape"
    assert torch.tensor(negative_sampling(x, i, 2)).shape[0] == len(x) * 2, \
        "Negative sampling rate of 2 did not return tensor with expected shape"
    assert torch.tensor(negative_sampling(x, i, 3)).shape[0] == len(x) * 3, \
        "Negative sampling rate of 3 did not return tensor with expected shape"


def test_perturb():
    """ Unit test for perturb function """
    entity_dictionary = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    num_entities = len(entity_dictionary)
    test_triple = [1, 2, 3]
    candidates = perturb(test_triple, entity_dictionary)
    assert candidates.shape == (num_entities, 3), \
        "perturb function did not return a tensor with the expected shape"

    tail_entities = torch.tensor([1, 2, 3, 4, 5])
    assert torch.all(torch.eq(candidates[:, 2], tail_entities)), \
        "perturb function did not produce all the expected tail entities"


def test_filter_triples():
    """ Unit test for filter_triples function """
    candidate_triples = torch.tensor([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]
    ])
    all_triples = [
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ]
    test_triple = [3, 3, 3]
    filtered = filter_triples(candidate_triples, all_triples, test_triple)

    expected_result = torch.tensor([[1, 1, 1], [3, 3, 3], [5, 5, 5]], dtype=torch.long)
    assert torch.all(torch.eq(filtered, expected_result)), \
        "filter_triples function did not produce the expected output"


def test_compute_mrr():
    """ Unit test for compute_mrr function """
    assert compute_mrr(2) == 0.5


def test_compute_hits():
    """ Unit test for compute_hits function """
    assert compute_hits(2, 1) == 0
    assert compute_hits(2, 3) == 1
    assert compute_hits(2, 10) == 1


def test_rank_triple():
    """ Unit test for rank_triple function """
    scores = torch.tensor([
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    ])
    candidate_triples = torch.tensor([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]
    ])
    correct_triple = [3, 3, 3]
    assert rank_triple(scores, candidate_triples, correct_triple) == 3
