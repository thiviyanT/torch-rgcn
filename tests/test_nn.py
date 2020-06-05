from rgcn.utils import generate_adj, add_inverse_and_self
from rgcn.layers import RelationalGraphConvolution
import torch


def test_generate_adj():
    """ Unit test for generate_adj (node classification) """

    edges = {0: ([0], [0]), 1: ([1], [1]), 2: ([2], [2])}
    adj, size = generate_adj(edges, 3, 'cpu')
    print(adj, size)
    # TODO: Check adj

    assert True is True


def test_inverse_self_loops():
    """ Unit test for add_inverse_and_self (link prediction) """

    triples = torch.tensor([[0, 0, -1], [1, 1, -2], [2, 2, -3]])
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


def test_normalisation():
    """ Unit test for ... """
    assert True is True


def test_edge_dropout():
    """ Unit test for ... """
    assert True is True


def test_basis_decomposition():
    """ Unit test for ... """

    """
    Basis function decomposition tackles overfitting by significantly reducing the number of learnable parameters!

    For example:
        320 parameters to fit (without basis decomposition)
        320 + 25 = 345 parameters to fit (with 5 bases)
        256 + 20 = 276 parameters to fit (with 4 bases)
        192 + 15 = 207 parameters to fit (with 3 bases)
        128 + 10 = 138 parameters to fit (with 2 bases)
         64 +  5 =  69 parameters to fit (with 1 base)
    """
    nnodes = 5
    nrel = 5
    nhid = 16
    nbases = 5
    reg = 'basis'
    layer = RelationalGraphConvolution(nnodes, nrel, in_features=5, out_features=nhid, num_bases=nbases, weight_reg=reg)
    print(type(layer.weight))

    assert True is True


def test_block_diagonal_decomposition():
    """ Unit test for ... """

    """
    Block Diagonal Decomposition tackles overfitting by dropping the number of learnable parameters!

    For example:
        320 parameters to fit (without basis decomposition)
        320 + 25 = 345 parameters to fit (with 5 bases)
        256 + 20 = 276 parameters to fit (with 4 bases)
        192 + 15 = 207 parameters to fit (with 3 bases)
        128 + 10 = 138 parameters to fit (with 2 bases)
         64 +  5 =  69 parameters to fit (with 1 base)
    """
    nnodes = 5
    nrel = 5
    nhid = 16
    nbases = 5
    reg = 'basis'
    layer = RelationalGraphConvolution(nnodes, nrel, in_features=5, out_features=nhid, num_bases=nbases, weight_reg=reg)
    print(type(layer.weight))

    assert True is True
