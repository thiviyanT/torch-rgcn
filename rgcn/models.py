from rgcn.layers import RelationalGraphConvolution, EmbeddingLayer
from rgcn.utils import add_inverse_and_self
import torch.nn.functional as F
from torch import nn
import torch


######################################################################################
# Models for Experiment Reproduction
######################################################################################

class LinkPredictor(nn.Module):
    """ Link prediction via R-GCN encoder and DistMult decoder """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nemb=128,
                 out=None,
                 dropout=.0,
                 edge_dropout=None,
                 decomposition=None,
                 device='cpu'):
        super(LinkPredictor, self).__init__()

        self.num_nodes = nnodes
        self.num_rels = nrel
        self.device = device
        self.dropout = dropout

        assert (triples is not None or nnodes is not None or nrel is not None or out is not None), \
            "The following must be specified: triples, number of nodes, number of relations and output dimension!"

        triples = torch.tensor(triples, dtype=torch.long, device=device)
        with torch.no_grad():
            self.register_buffer('triples', triples)
            # Add inverse relations and self-loops to triples
            self.register_buffer('triples_plus', add_inverse_and_self(triples, nnodes, nrel, device))

        self.rgc1 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition
        )
        self.rgc2 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nhid,
            out_features=out,
            edge_dropout=edge_dropout,
            decomposition=decomposition
        )

        self.entities = EmbeddingLayer(nnodes, nemb)
        self.relations = EmbeddingLayer(nrel, nemb)

    def score(self, triples):
        """ Simple DistMult scoring function (from https://arxiv.org/pdf/1412.6575.pdf) """

        s = triples[:, 0]
        p = triples[:, 1]
        o = triples[:, 2]
        s, p, o = self.entities[s, :], self.relations[p, :], self.entities[o, :]

        return (s * p * o).sum(dim=1)

    def forward(self):
        """ Embed relational graph and then compute DistMult score """

        x = self.rgc1()
        x = F.relu(x)
        x = self.rgc2(features=x)
        scores = self.decoder(x)

        return scores.view()


class NodeClassifier(nn.Module):
    """ Node classification with R-GCN message passing """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nclass=None,
                 edge_dropout=None,
                 decomposition=None,
                 device='cpu'):
        super(NodeClassifier, self).__init__()

        self.num_nodes = nnodes
        self.nrel = nrel
        self.nfeat = nfeat
        self.device = device
        self.edge_dropout = edge_dropout

        assert (triples is not None or nnodes is not None or nrel is not None or nclass is not None), \
            "The following must be specified: triples, number of nodes, number of relations and number of classes!"

        triples = torch.tensor(triples, dtype=torch.long, device=device)
        with torch.no_grad():
            self.register_buffer('triples', triples)
            # Add inverse relations and self-loops to triples
            self.register_buffer('triples_plus', add_inverse_and_self(triples, nnodes, nrel, device))

        self.rgc1 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition
        )
        self.rgc2 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nhid,
            out_features=nclass,
            edge_dropout=edge_dropout,
            decomposition=decomposition
        )

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        x = self.rgc1()
        x = F.relu(x)
        x = self.rgc2(features=x)
        x = F.log_softmax(x, dim=1)
        return x


######################################################################################
# RGCN Extensions
######################################################################################


class CompressionLinkPredictor(LinkPredictor):
    """ A standard link prediction model where the input to the RGCN is compressed using MLP """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 out=None,
                 dropout=.0,
                 edge_dropout=None,
                 decomposition=None,
                 device='cpu'):
        # super(CompressionLinkPredictor, self).__init__()
        raise NotImplementedError('This model has not been implemented yet!')

    def forward(self):
        pass


class EmbeddingLinkPredictor(LinkPredictor):
    """ A standard link prediction model with node embeddings as the feature matrix """

    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 out=None,
                 dropout=.0,
                 edge_dropout=None,
                 decomposition=None,
                 device='cpu'):
        # super(EmbeddingLinkPredictor, self).__init__()
        raise NotImplementedError('This model has not been implemented yet!')

    def forward(self):
        pass


class GlobalLinkPredictor(LinkPredictor):
    """ A standard link prediction model with global readouts """

    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 out=None,
                 dropout=.0,
                 edge_dropout=None,
                 decomposition=None,
                 device='cpu'):
        # super(GlobalLinkPredictor, self).__init__()
        raise NotImplementedError('This model has not been implemented yet!')

    def forward(self):
        pass
