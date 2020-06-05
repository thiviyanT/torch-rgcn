from rgcn.layers import RelationalGraphConvolution
from rgcn.utils import add_inverse_and_self, generate_adj, stack_matrices, sum_sparse
import torch.nn.functional as F
from torch import nn
import torch


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

        # TODO: Instantiate the weights and biases

    def forward(self, g):
        return self.embedding(g)


class LinkPredictor(nn.Module):
    """ Link prediction via 2 layer R-GCN encoder and DistMult decoder """
    def __init__(self,
                 triples,
                 nnodes,
                 nrel,
                 nfeat=None,
                 nhid=16,
                 out=None,
                 dropout=.0,
                 edge_dropout=None,
                 edge_dropout_self_loop=None,
                 nblocks=-1,
                 decomp='block',
                 device='cpu'):
        super(LinkPredictor, self).__init__()

        self.num_nodes = nnodes
        self.num_rels = nrel
        self.device = device
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.edge_dropout_self_loop = edge_dropout_self_loop

        assert out is not None, "Size of the latent dimension must be specified!"

        self.register_buffer('triples', triples)

        with torch.no_grad():
            self.register_buffer('triples_inverse_self', add_inverse_and_self(triples, nnodes, nrel, device))

        # Message Passing encoder
        self.rgc1 = RelationalGraphConvolution(
            nnodes,
            nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            num_blocks=nblocks,
            weight_reg=decomp
        )
        self.rgc2 = RelationalGraphConvolution(
            nnodes,
            nrel * 2 + 1,
            in_features=nhid,
            out_features=out,
            num_blocks=nblocks,
            weight_reg=decomp
        )

        # DistMult decoder
        self.entities = EmbeddingLayer(nnodes, nhid)
        self.relations = EmbeddingLayer(nrel, nhid)

        # horizontally and vertically stacked versions of the adjacency graph
        ver_ind, ver_size = stack_matrices(self.triples_inverse_self, nnodes, nrel * 2 + 1, vertical_stacking=True)
        hor_ind, hor_size = stack_matrices(self.triples_inverse_self, nnodes, nrel * 2 + 1, vertical_stacking=False)

        if self.edge_dropout is not None and self.training:
            # Separate self-loops from other edges
            self_loop = self.triples_inverse_self[-nnodes:, :]
            other_edges = self.triples_inverse_self[:-nnodes, :]

            # TODO: Apply general dropout if one for self loop is not given. Should I be more transparent about this?
            edge_dropout_self_loop = edge_dropout_self_loop if edge_dropout_self_loop is not None else edge_dropout

            # Other edges
            # self.edge_dropout

            # TODO: Finish Edge dropout implementation
            raise NotImplementedError('Edge dropout not available!')
        else:
            num_edges = ver_ind.size(0)
            vals = torch.ones(num_edges, dtype=torch.float, device=device)

        vals = vals / sum_sparse(ver_ind, vals, ver_size, self.device)

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size, device=device)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size, device=device)
        self.register_buffer('ver_graph', ver_graph)

    def distmult(self, triples, nodes, relations):
        """ DistMult scoring function (as described in https://arxiv.org/pdf/1412.6575.pdf) """

        s = triples[:, 0]
        p = triples[:, 1]
        o = triples[:, 2]
        s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

        # TODO Add biases to subjects, predicates and objects separately

        return (s * p * o).sum(dim=1)

    def forward(self, adj):
        """ Embed relational graph and then compute DistMult score """
        # TODO: 2 layers of  RGCN and then DistMult
        return


class NodeClassifier(nn.Module):
    """ Node classification with a two layer R-GCN message passing """
    def __init__(self, edges, nnodes, nrel, nfeat=None, nhid=16, nclass=None, dropout=.0, nbases=-1, decomp='basis', device='cpu'):
        super(NodeClassifier, self).__init__()

        self.num_nodes = nnodes
        self.nrel = nrel
        self.nfeat = nfeat
        self.device = device
        self.dropout = dropout

        assert nclass is not None, "Number of classes must be specified!"

        self.rgc1 = RelationalGraphConvolution(
            nnodes,
            nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            num_bases=nbases,
            weight_reg=decomp)
        self.rgc2 = RelationalGraphConvolution(
            nnodes,
            nrel * 2 + 1,
            in_features=nhid,
            out_features=nclass,
            num_bases=nbases,
            weight_reg=decomp)

        ver_ind, ver_size = generate_adj(edges, self.num_nodes, self.device, vertical_stacking=True)
        hor_ind, hor_size = generate_adj(edges, self.num_nodes, self.device, vertical_stacking=False)

        num_edges = ver_ind.size(0)
        vals = torch.ones(num_edges, dtype=torch.float)
        # Row-wise normalisation is the same regardless of how the adjacency matrices are stacked!
        vals = vals / sum_sparse(ver_ind, vals, ver_size, self.device)

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        exit()  # DEBUG
        # TODO: Create sparse matrix multiplication!
        x = F.relu(self.rgc1(self.hor_graph))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.rgc2(self.ver_graph, features=x)
        return F.log_softmax(x, dim=1)
