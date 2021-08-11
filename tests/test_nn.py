from torch_rgcn.layers import RelationalGraphConvolutionNC
import torch


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
nnodes = 4
nrel = 3
nhid = 16


def test_no_decomposition():
    """ Unit test for when no decomposition is required """

    layer1 = RelationalGraphConvolutionNC(
        triples=triples,
        num_nodes=nnodes,
        num_relations=nrel * 2 + 1,
        in_features=None,
        out_features=nhid,
        edge_dropout=None,
        decomposition=None
    )
    layer2 = RelationalGraphConvolutionNC(
        triples=triples,
        num_nodes=nnodes,
        num_relations=nrel * 2 + 1,
        in_features=nhid,
        out_features=nhid,
        edge_dropout=None,
        decomposition=None
    )

    z = layer1.forward()
    z2 = layer2.forward(z)
    assert layer1.weights.size() == torch.Size([7, 4, 16])
    assert layer2.weights.size() == torch.Size([7, 16, 16])
    assert z.size() == z2.size() == torch.Size([4, 16])


def test_basis_decomposition():
    """ Unit test for basis function decomposition """

    """
    Basis function decomposition tackles over-fitting by significantly reducing the number of learnable parameters!

    For example:
        320 parameters to fit (without basis decomposition)
        320 + 25 = 345 parameters to fit (with 5 bases)
        256 + 20 = 276 parameters to fit (with 4 bases)
        192 + 15 = 207 parameters to fit (with 3 bases)
        128 + 10 = 138 parameters to fit (with 2 bases)
         64 +  5 =  69 parameters to fit (with 1 base)
    """
    decomposition = {'type': 'basis', 'num_bases': 2}

    layer1 = RelationalGraphConvolutionNC(
        triples=triples,
        num_nodes=nnodes,
        num_relations=nrel * 2 + 1,
        in_features=None,
        out_features=nhid,
        edge_dropout=None,
        decomposition=decomposition
    )
    layer2 = RelationalGraphConvolutionNC(
        triples=triples,
        num_nodes=nnodes,
        num_relations=nrel * 2 + 1,
        in_features=nhid,
        out_features=nhid,
        edge_dropout=None,
        decomposition=decomposition
    )

    z = layer1.forward()
    z2 = layer2.forward(z)
    assert layer1.bases.size() == torch.Size([2, 4, 16])
    assert layer2.bases.size() == torch.Size([2, 16, 16])
    assert layer1.comps.size() == layer2.comps.size() == torch.Size([7, 2])
    assert z.size() == z2.size() == torch.Size([4, 16])


def test_block_diagonal_decomposition():
    """ Unit test for block diagonal decomposition """

    decomposition = {'type': 'block', 'num_blocks': 2}

    layer1 = RelationalGraphConvolutionNC(
        triples=triples,
        num_nodes=nnodes,
        num_relations=nrel * 2 + 1,
        in_features=None,
        out_features=nhid,
        edge_dropout=None,
        decomposition=decomposition
    )
    layer2 = RelationalGraphConvolutionNC(
        triples=triples,
        num_nodes=nnodes,
        num_relations=nrel * 2 + 1,
        in_features=nhid,
        out_features=nhid,
        edge_dropout=None,
        decomposition=decomposition
    )
    z = layer1.forward()
    z2 = layer2.forward(z)
    assert layer1.blocks.size() == torch.Size([7, 2, 2, 8])
    assert layer2.blocks.size() == torch.Size([7, 2, 8, 8])
    assert z.size() == z2.size() == torch.Size([4, 16])
