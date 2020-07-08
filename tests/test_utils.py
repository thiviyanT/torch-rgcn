from torch_rgcn.utils import add_inverse_and_self, sum_sparse, stack_matrices, drop_edges
import torch


def test_add_inverse_and_self():
    """ Unit test for add_inverse_and_self """

    triples = torch.tensor([
        [0, 0, -1],  # Edge from 0 to 1 with the relation 0
        [1, 1, -2],  # Edge from 1 to -2 with the relation 1
        [2, 2, -3]   # Edge from 2 to -3 with the relation 2
    ])
    expected_response = torch.tensor([
        [ 0,  0, -1],
        [ 1,  1, -2],
        [ 2,  2, -3],
        [-1,  3,  0],
        [-2,  4,  1],
        [-3,  5,  2],
        [ 0,  6,  0],
        [ 1,  6,  1],
        [ 2,  6,  2]
    ])
    triples_inverse_self = add_inverse_and_self(triples, 3, 3)
    assert torch.all(torch.eq(triples_inverse_self, expected_response))


def test_stack_matrices():
    """ Unit test for stack_matrices """

    triples = torch.tensor([
        [0, 0, -1],
        [1, 1, -2],
        [2, 2, -3],
        [-1, 3, 0],
        [-2, 4, 1],
        [-3, 5, 2],
        [0, 6, 0],
        [1, 6, 1],
        [2, 6, 2]
    ])
    num_nodes = 6
    num_relations = 3

    # Test vertical stacking
    ver_ind, ver_size = stack_matrices(triples, num_nodes, num_relations * 2 + 1, vertical_stacking=True)
    expected_response = torch.tensor([
        [0, -1],
        [7, -2],
        [14, -3],
        [17, 0],
        [22, 1],
        [27, 2],
        [36, 0],
        [37, 1],
        [38, 2]
    ])
    assert torch.all(torch.eq(ver_ind, expected_response))
    assert ver_size == (42, 6)

    # Test horizontal stacking
    hor_ind, hor_size = stack_matrices(triples, num_nodes, num_relations * 2 + 1, vertical_stacking=False)
    expected_response = torch.tensor([
        [0, -1],
        [1, 4],
        [2, 9],
        [-1, 18],
        [-2, 25],
        [-3, 32],
        [0, 36],
        [1, 37],
        [2, 38]
    ])
    assert torch.all(torch.eq(hor_ind, expected_response))
    assert hor_size == (6, 42)


def test_sum_sparse():
    """ Unit test for sum_sparse (used for row-wise normalisation) """

    # Test with vertical stacking
    ver_ind = torch.tensor([
        [0, 0],
        [0, 1],
        [0, 2],
        [4, 1],
        [8, 2],
        [7, 2]
    ])
    ver_size = (9, 3)
    num_edges = ver_ind.size(0)
    vals = torch.ones(num_edges, dtype=torch.float)
    vals = vals / sum_sparse(ver_ind, vals, ver_size, row_normalisation=True)
    # adj = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size).to_dense()
    expected_response = torch.tensor([1/3, 1/3, 1/3, 1, 1, 1])
    assert torch.all(torch.eq(vals, expected_response))

    # Test with horizontal stacking
    hor_ind = torch.tensor([
        [0, 0],
        [1, 0],
        [2, 0],
        [1, 4],
        [2, 8],
        [2, 7]
    ])
    hor_size = (3, 9)
    num_edges = hor_ind.size(0)
    vals = torch.ones(num_edges, dtype=torch.float)
    vals = vals / sum_sparse(hor_ind, vals, hor_size, row_normalisation=False)
    # adj = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size).to_dense()
    expected_response = torch.tensor([1/3, 1/3, 1/3, 1, 1, 1])
    assert torch.all(torch.eq(vals, expected_response))


def test_drop_edges():
    """ Unit test for drop_edges util function """

    triples = torch.tensor([
        [0, 0, 1],
        [1, 1, 2],
        [2, 2, 3],
        [1, 3, 0],
        [2, 4, 1],
        [3, 5, 2],
        [0, 6, 0],
        [1, 6, 1],
        [2, 6, 2],
        [3, 6, 3]
    ])
    num_nodes = 4
    general_edo = 0.5  # General edge dropout rate (applied to all edges except self-loops)
    self_loop_edo = 0.25  # Self-loop edge dropout rate

    triples = drop_edges(triples, num_nodes, general_edo, self_loop_edo)
    self_loops = [
        [0, 6, 0],
        [1, 6, 1],
        [2, 6, 2],
        [3, 6, 3]
    ]
    other_edges = [
        [0, 0, 1],
        [1, 1, 2],
        [2, 2, 3],
        [1, 3, 0],
        [2, 4, 1],
        [3, 5, 2]
    ]
    count_self_loops = 0
    count_other_edges = 0
    for i in triples.tolist():
        if i in self_loops:
            count_self_loops += 1
        if i in other_edges:
            count_other_edges += 1
    assert count_self_loops == 3 and count_other_edges == 3
